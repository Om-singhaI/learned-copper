"""Freerouting baseline on the PCBWorld test boards under our own KiCad 10 pipeline (issue 6).

Problem definition: the mirror's unrouted.dsn for each test board. The mirror built it from the human
board with every track, via and copper zone removed: the zones themselves are gone, not only their fills,
the DSN carries no planes and unrouted.kicad_pcb has no zones. Freerouting therefore routes every
connection as tracks, including the nets the human covered with a pour, and the zone refill before the
design rule check changes nothing on the routed boards. Keep that in mind when reading the human columns.

Freerouting runs headless under a wall clock cap that kills Java on expiry. The -mp pass limit is passed
through but Freerouting 2.1.0 does not apply it with the GUI off: its stop pass number is fixed at 999 and
only the GUI windows copy max_passes into it (checked in the jar's bytecode), so a board that stays
incomplete runs 999 passes. fr_passes_run records what actually ran. Freerouting also seeds its random
generators from the clock, so two runs of one board differ; --repeats runs each board several times, every
attempt is kept, and the summary reports attempt 1 (single shot) beside the best attempt (PCBWorld used
best of five).

Each session file is imported into the mirror's unrouted.kicad_pcb with scripts/kicad_dsn.py (one KiCad
Python subprocess with a timeout). scripts/metrics.py then scores every routed attempt and the human
raw.kicad_pcb the same way: zone refill, KiCad 10 design rule check with all severities, copper rule
whitelist from scripts/drc_batch.py.

    .venv/bin/python scripts/freerouting_batch.py --clean-only --limit 10 --repeats 3   # pilot
    .venv/bin/python scripts/freerouting_batch.py --repeats 3                           # all 119 test boards
    .venv/bin/python scripts/freerouting_batch.py BOARD_ID [BOARD_ID ...] --force
    .venv/bin/python scripts/freerouting_batch.py --summary-only       # rebuild csv rows and summary
    .venv/bin/python scripts/freerouting_batch.py --compare-mirror     # side by side with the mirror's runs

One worker. Resumable: attempts already in data/results/freerouting_test_attempts.csv are kept and only the
missing ones up to --repeats run, unless --force. Outputs per attempt under data/freerouting/:
<id>.r<k>.ses, <id>.r<k>.freerouting.log (Freerouting's stdout and stderr), <id>.r<k>.freerouting.kicad_pcb,
<id>.r<k>.import.log, and the DRC reports under data/freerouting/drc/. Results: data/results/
freerouting_test.csv (one row per board), freerouting_test_attempts.csv (one row per attempt) and
freerouting_test_summary.json.
"""
import argparse
import collections
import csv
import json
import re
import statistics
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kicad_dsn  # noqa: E402
from drc_batch import COPPER_TYPES, ELECTRICAL_TYPES, FIX  # noqa: E402
from metrics import board_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
JAVA = "/opt/homebrew/opt/openjdk@21/bin/java"
JAR = ROOT / "tools/freerouting.jar"
OUT_DIR = ROOT / "data/freerouting"
DRC_DIR = OUT_DIR / "drc"
RESULTS_CSV = ROOT / "data/results/freerouting_test.csv"
ATTEMPTS_CSV = ROOT / "data/results/freerouting_test_attempts.csv"
SUMMARY_JSON = ROOT / "data/results/freerouting_test_summary.json"
COMPARE_CSV = ROOT / "data/results/freerouting_test_mirror_compare.csv"
MIRROR_RESULTS = ROOT / "data/mirror/freerouting/scripts/benchmark/results/benchmarks.json"
POOL = ROOT / "splits/mirror_pool.json"
D3 = ROOT / "splits/test_d3.json"
FR_INTERNAL_PASS_CAP = 999

METRIC_COLS = ["unconnected", "copper_violations", "electrical_violations", "track_length_mm", "segments", "vias"]
RUN_COLS = ["fr_rc", "fr_runtime_s", "fr_timed_out", "fr_ses_written", "fr_passes_run", "fr_incomplete", "fr_clearance",
            "import_ok", "import_seconds", "complete", "clean_pass"] + METRIC_COLS
ATTEMPT_FIELDS = ["board_id", "attempt", "fr_version", "fr_passes_limit", "fr_timeout_s"] + RUN_COLS + ["metrics_seconds", "error"]
AGG_COLS = ["complete_attempts", "clean_pass_attempts", "complete_any", "clean_pass_any", "complete_all", "clean_pass_all",
            "best_attempt", "best_unconnected", "best_copper_violations", "best_track_length_mm", "best_vias",
            "fr_passes_run_max", "fr_runtime_total_s"]
HUMAN_COLS = ["human_" + c for c in METRIC_COLS] + ["human_copper_zones"]
FIELDS = (["board_id", "pcbworld_tier", "clean", "layers", "fr_version", "fr_passes_limit", "fr_timeout_s", "attempts"]
          + RUN_COLS + AGG_COLS + HUMAN_COLS + ["metrics_seconds", "error"])
INT_COLS = {"layers", "attempt", "attempts", "fr_passes_limit", "fr_timeout_s", "fr_rc", "fr_passes_run", "fr_incomplete",
            "fr_clearance", "unconnected", "copper_violations", "electrical_violations", "segments", "vias",
            "complete_attempts", "clean_pass_attempts", "best_attempt", "best_unconnected", "best_copper_violations",
            "best_vias", "fr_passes_run_max", "human_unconnected", "human_copper_violations",
            "human_electrical_violations", "human_segments", "human_vias", "human_copper_zones"}
FLOAT_COLS = {"fr_runtime_s", "import_seconds", "track_length_mm", "best_track_length_mm", "fr_runtime_total_s",
              "human_track_length_mm", "metrics_seconds"}
BOOL_COLS = {"clean", "fr_timed_out", "fr_ses_written", "import_ok", "complete", "clean_pass", "complete_any",
             "clean_pass_any", "complete_all", "clean_pass_all"}


def load_test_boards():
    pool = json.load(open(POOL))["boards"]
    tiers = {b["board_id"]: b["pcbworld_tier"] for b in json.load(open(D3))["boards"]}
    boards = []
    for b in pool:
        if b["in_test_set"]:
            boards.append({"board_id": b["board_id"], "clean": b["tier"] == "clean", "layers": b["layers"],
                           "pcbworld_tier": tiers.get(b["board_id"], "")})
    return sorted(boards, key=lambda b: b["board_id"])


def coerce(row, fields):
    out = {}
    for k in fields:
        v = row.get(k, "")
        if v == "" or v is None:
            out[k] = ""
        elif k in INT_COLS:
            out[k] = int(float(v))
        elif k in FLOAT_COLS:
            out[k] = float(v)
        elif k in BOOL_COLS:
            out[k] = v if isinstance(v, bool) else v == "True"
        else:
            out[k] = v
    return out


def load_csv(path, fields, key):
    if not Path(path).exists():
        return {}
    with open(path, newline="") as f:
        return {key(r): coerce(r, fields) for r in csv.DictReader(f)}


def save_csv(path, rows, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k in sorted(rows):
            w.writerow({c: rows[k].get(c, "") for c in fields})


def load_rows():
    return load_csv(RESULTS_CSV, FIELDS, lambda r: r["board_id"])


def load_attempts():
    return load_csv(ATTEMPTS_CSV, ATTEMPT_FIELDS, lambda r: (r["board_id"], int(r["attempt"])))


def jar_build_date():
    try:
        with zipfile.ZipFile(JAR) as z:
            m = z.read("META-INF/MANIFEST.MF").decode("utf-8", "replace")
        d = re.search(r"Build-Date:\s*(\S+)", m)
        return d.group(1) if d else ""
    except Exception:
        return ""


def parse_freerouting_log(text):
    """Version, passes run, and the incomplete and clearance counts from the JSON block Freerouting prints last."""
    info = {"fr_version": "", "fr_passes_run": 0, "fr_incomplete": "", "fr_clearance": ""}
    m = re.search(r"Freerouting v([\w.\-]+)", text)
    if m:
        info["fr_version"] = m.group(1)
    info["fr_passes_run"] = len(re.findall(r"Auto-router pass #\d+", text))
    i = text.rfind("\n{\n")
    if i >= 0:
        try:
            d = json.loads(text[i + 1:])
            info["fr_incomplete"] = d.get("connections", {}).get("incomplete_count", "")
            info["fr_clearance"] = d.get("clearance_violations", {}).get("total_count", "")
        except json.JSONDecodeError:
            pass
    return info


def run_freerouting(dsn, ses, log_path, passes, timeout):
    """Known headless invocation. subprocess.run kills Java when the timeout expires."""
    if ses.exists():
        ses.unlink()
    cmd = [JAVA, "-Xmx2g", "-jar", str(JAR), "-de", str(dsn), "-do", str(ses), "-mp", str(passes), "--gui.enabled=false"]
    t = time.time()
    timed_out, rc, text = False, "", ""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=OUT_DIR)
        rc, text = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        timed_out = True
        text = ((e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""))
        text += ((e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or ""))
        text += f"\n[killed after {timeout} s]\n"
    runtime = round(time.time() - t, 1)
    log_path.write_text("$ " + " ".join(cmd) + "\n" + text)
    r = {"fr_rc": rc, "fr_runtime_s": runtime, "fr_timed_out": timed_out,
         "fr_ses_written": ses.exists() and ses.stat().st_size > 0}
    r.update(parse_freerouting_log(text))
    return r


def run_attempt(b, k, passes, timeout, import_timeout=300):
    """One Freerouting run, import and metrics for attempt k of board b. Never raises."""
    bid = b["board_id"]
    fix = FIX / bid
    dsn, unrouted = fix / "unrouted.dsn", fix / "unrouted.kicad_pcb"
    tag = f"{bid}.r{k}"
    ses, routed = OUT_DIR / f"{tag}.ses", OUT_DIR / f"{tag}.freerouting.kicad_pcb"
    row = {c: "" for c in ATTEMPT_FIELDS}
    row.update(board_id=bid, attempt=k, fr_passes_limit=passes, fr_timeout_s=timeout, import_ok=False,
               complete=False, clean_pass=False, error="")
    errors = [f"missing {p.name}" for p in (dsn, unrouted) if not p.exists()]
    if errors:
        row["error"] = "; ".join(errors)
        return row

    row.update(run_freerouting(dsn, ses, OUT_DIR / f"{tag}.freerouting.log", passes, timeout))
    if row["fr_timed_out"]:
        errors.append(f"freerouting timed out at {timeout} s")
    elif row["fr_rc"] != 0:
        errors.append(f"freerouting exit code {row['fr_rc']}")
    if not row["fr_ses_written"]:
        errors.append("no session file")

    if row["fr_ses_written"]:
        if routed.exists():
            routed.unlink()
        t = time.time()
        rc, out = kicad_dsn.run_with_timeout(["import", str(unrouted), str(ses), str(routed)], timeout=import_timeout)
        row["import_seconds"] = round(time.time() - t, 1)
        (OUT_DIR / f"{tag}.import.log").write_text(out)
        row["import_ok"] = rc == 0 and routed.exists()
        if not row["import_ok"]:
            errors.append("session import failed" + (" (timeout)" if rc == 124 else f" (rc {rc})"))

    if row["import_ok"]:
        t = time.time()
        m = board_metrics(routed, DRC_DIR, board_id=bid)
        row["metrics_seconds"] = round(time.time() - t, 1)
        for c in METRIC_COLS:
            row[c] = m[c]
        if m["complete"] != "":
            row["complete"], row["clean_pass"] = m["complete"], m["clean_pass"]
        if m["error"]:
            errors.append("routed metrics: " + m["error"])
    row["error"] = "; ".join(errors)
    return row


def human_metrics(b):
    """The human reference (raw.kicad_pcb) scored once per board. Returns the human_* columns and seconds."""
    raw = FIX / b["board_id"] / "raw.kicad_pcb"
    out = {c: "" for c in HUMAN_COLS}
    if not raw.exists():
        return out, 0.0, "missing raw.kicad_pcb"
    t = time.time()
    h = board_metrics(raw, DRC_DIR, board_id=b["board_id"])
    for c in METRIC_COLS:
        out["human_" + c] = h[c]
    out["human_copper_zones"] = h["copper_zones"]
    return out, round(time.time() - t, 1), ("human metrics: " + h["error"]) if h["error"] else ""


def best_key(a):
    """PCBWorld style best attempt: fewest unconnected, then fewest copper violations, then shortest."""
    return (a["unconnected"], a["copper_violations"], a["track_length_mm"] or 0.0, a["attempt"])


def board_row(b, attempts, human, human_seconds, human_error):
    """One row per board from its attempts (sorted by attempt number) and the human reference."""
    attempts = sorted(attempts, key=lambda a: a["attempt"])
    first = attempts[0]
    row = {c: "" for c in FIELDS}
    row.update(board_id=b["board_id"], pcbworld_tier=b["pcbworld_tier"], clean=b["clean"], layers=b["layers"],
               fr_version=first["fr_version"], fr_passes_limit=first["fr_passes_limit"], fr_timeout_s=first["fr_timeout_s"],
               attempts=len(attempts))
    for c in RUN_COLS:
        row[c] = first[c]
    n_complete = sum(1 for a in attempts if a["complete"] is True)
    n_clean = sum(1 for a in attempts if a["clean_pass"] is True)
    scored = [a for a in attempts if a["import_ok"] is True and a["unconnected"] != ""]
    best = min(scored, key=best_key) if scored else None
    row.update(complete_attempts=n_complete, clean_pass_attempts=n_clean,
               complete_any=n_complete > 0, clean_pass_any=n_clean > 0,
               complete_all=n_complete == len(attempts), clean_pass_all=n_clean == len(attempts),
               best_attempt=best["attempt"] if best else "", best_unconnected=best["unconnected"] if best else "",
               best_copper_violations=best["copper_violations"] if best else "",
               best_track_length_mm=best["track_length_mm"] if best else "", best_vias=best["vias"] if best else "",
               fr_passes_run_max=max((a["fr_passes_run"] for a in attempts if a["fr_passes_run"] != ""), default=""),
               fr_runtime_total_s=round(sum(a["fr_runtime_s"] for a in attempts if a["fr_runtime_s"] != ""), 1))
    row.update(human)
    row["metrics_seconds"] = round(sum(a["metrics_seconds"] for a in attempts if a["metrics_seconds"] != "") + human_seconds, 1)
    errs = [f"r{a['attempt']}: {a['error']}" for a in attempts if a["error"]]
    if human_error:
        errs.append(human_error)
    row["error"] = "; ".join(errs)
    return row


def subset_stats(rows, attempts):
    n = len(rows)
    out = {"n_run": n}
    if n == 0:
        return out
    ids = {r["board_id"] for r in rows}
    att = [a for (bid, _), a in attempts.items() if bid in ids]
    runtimes = [a["fr_runtime_s"] for a in att if a["fr_runtime_s"] != ""]
    out.update({
        "attempts_per_board": {"min": min(r["attempts"] for r in rows), "max": max(r["attempts"] for r in rows)},
        "attempt_1": {
            "complete": sum(1 for r in rows if r["complete"] is True),
            "clean_pass": sum(1 for r in rows if r["clean_pass"] is True)},
        "best_attempt": {
            "complete": sum(1 for r in rows if r["complete_any"] is True),
            "clean_pass": sum(1 for r in rows if r["clean_pass_any"] is True)},
        "every_attempt": {
            "complete": sum(1 for r in rows if r["complete_all"] is True),
            "clean_pass": sum(1 for r in rows if r["clean_pass_all"] is True)},
        "boards_whose_attempts_disagree": sum(
            1 for r in rows if r["attempts"] > 1 and (r["complete_attempts"] not in (0, r["attempts"])
                                                      or r["clean_pass_attempts"] not in (0, r["attempts"]))),
        "attempts_total": len(att),
        "attempts_complete": sum(1 for a in att if a["complete"] is True),
        "attempts_clean_pass": sum(1 for a in att if a["clean_pass"] is True),
        "timed_out_attempts": sum(1 for a in att if a["fr_timed_out"] is True),
        "no_session_attempts": sum(1 for a in att if a["fr_ses_written"] is False),
        "import_failed_attempts": sum(1 for a in att if a["fr_ses_written"] is True and a["import_ok"] is False),
        "freerouting_nonzero_exit_attempts": sum(1 for a in att if a["fr_rc"] not in ("", 0)),
        "attempts_over_passes_limit": sum(1 for a in att if a["fr_passes_run"] != "" and a["fr_passes_limit"] != ""
                                          and a["fr_passes_run"] > a["fr_passes_limit"]),
        "passes_run_max": max((a["fr_passes_run"] for a in att if a["fr_passes_run"] != ""), default=0),
        "human_complete": sum(1 for r in rows if r["human_unconnected"] == 0),
        "human_clean_pass": sum(1 for r in rows if r["human_unconnected"] == 0 and r["human_copper_violations"] == 0),
        "human_with_copper_zones": sum(1 for r in rows if r["human_copper_zones"] not in ("", 0)),
        "freerouting_runtime_s": {"mean": round(statistics.mean(runtimes), 1), "median": round(statistics.median(runtimes), 1),
                                  "max": round(max(runtimes), 1), "total": round(sum(runtimes), 1)} if runtimes else {},
    })
    for name, cond in (("attempt_1_vs_human", lambda r: True),
                       ("attempt_1_vs_human_no_zones", lambda r: r["human_copper_zones"] == 0)):
        scored = [r for r in rows if cond(r) and r["import_ok"] is True and r["track_length_mm"] != ""
                  and r["human_track_length_mm"] not in ("", 0)]
        out[name] = {"n": len(scored)}
        if scored:
            out[name].update({
                "complete": sum(1 for r in scored if r["complete"] is True),
                "length_ratio_median": round(statistics.median(r["track_length_mm"] / r["human_track_length_mm"] for r in scored), 3),
                "vias_routed": sum(r["vias"] for r in scored), "vias_human": sum(r["human_vias"] for r in scored)})
    for block in ("attempt_1", "best_attempt", "every_attempt"):
        out[block]["completion_rate"] = round(out[block]["complete"] / n, 4)
        out[block]["clean_pass_rate"] = round(out[block]["clean_pass"] / n, 4)
    return out


def build_summary(rows, attempts, boards, mirror=None):
    by_id = {b["board_id"]: b for b in boards}
    done = [r for r in rows.values() if r["board_id"] in by_id]
    sets = {"clean": [r for r in done if r["clean"] is True], "all": done}
    kicad = ""
    for p in sorted(DRC_DIR.glob("*.json")):
        try:
            kicad = json.load(open(p)).get("kicad_version", "")
            break
        except Exception:
            continue
    att = [a for (bid, _), a in attempts.items() if bid in by_id]
    summary = {
        "protocol": {
            "problem": "mirror unrouted.dsn: the human board with every track, via and copper zone removed (the zones "
                       "themselves, not only their fills; no planes in the DSN, no zones in unrouted.kicad_pcb); the "
                       "session is imported into that unrouted.kicad_pcb",
            "freerouting_version": sorted({a["fr_version"] for a in att if a["fr_version"]}),
            "jar_build_date": jar_build_date(),
            "passes_flag": "-mp passed on the command line but not applied by Freerouting 2.1.0 with the GUI off: its "
                           f"stop pass number is fixed at {FR_INTERNAL_PASS_CAP} and only the GUI windows copy "
                           "max_passes into it, so routing ends when every connection is routed, after "
                           f"{FR_INTERNAL_PASS_CAP} passes, or at the wall clock cap",
            "passes_limit_requested": sorted({a["fr_passes_limit"] for a in att if a["fr_passes_limit"] != ""}),
            "timeout_s": sorted({a["fr_timeout_s"] for a in att if a["fr_timeout_s"] != ""}),
            "repeats": "every board is routed --repeats times; attempt_1 is the single shot result, best_attempt takes "
                       "per board the attempt with the fewest unconnected, then fewest copper violations, then shortest "
                       "length (PCBWorld reported best of five), every_attempt counts boards where all attempts pass",
            "random_seed": "none available: Freerouting 2.1.0 constructs java.util.Random without a seed in "
                           "BatchAutorouter and MazeSearchAlgo, so attempts differ run to run",
            "kicad_version": kicad, "drc": "kicad-cli pcb drc --refill-zones --severity-all --units mm",
            "complete": "unconnected items 0 after zone refill",
            "clean_pass": "complete and zero error severity violations of a type in copper_types",
            "electrical_violations": "error severity violations of a type in electrical_types (a subset of copper_violations)",
            "copper_types": sorted(COPPER_TYPES), "electrical_types": sorted(ELECTRICAL_TYPES),
        },
        "boards_in_test_set": len(boards), "boards_clean": sum(1 for b in boards if b["clean"]),
        "boards_run": len(done), "attempts_run": len(att),
    }
    for name, sub in sets.items():
        s = subset_stats(sub, attempts)
        s["n_in_set"] = len(boards) if name == "all" else sum(1 for b in boards if b["clean"])
        s["by_pcbworld_tier"] = {}
        for tier in ("easy", "medium", "hard"):
            ts = subset_stats([r for r in sub if r["pcbworld_tier"] == tier], attempts)
            ts["n_in_set"] = sum(1 for b in boards if b["pcbworld_tier"] == tier and (name == "all" or b["clean"]))
            s["by_pcbworld_tier"][tier] = ts
        summary[name] = s
    if mirror is None and SUMMARY_JSON.exists():
        try:
            mirror = json.load(open(SUMMARY_JSON)).get("mirror_reference")
        except (OSError, json.JSONDecodeError):
            mirror = None
    if mirror:
        summary["mirror_reference"] = mirror
    return summary


def print_table(rows):
    print(f"{'board':40} {'tier':6} {'cl':2} {'att':>3} {'fr s':>6} {'inc':>4} {'unc':>4} {'cu':>4} {'el':>3} {'len mm':>8} "
          f"{'vias':>4} {'C':>4} {'CP':>4} | {'h unc':>5} {'h cu':>4} {'h len':>8} {'h via':>5} {'zn':>3}")
    for r in (rows[k] for k in sorted(rows)):
        yn = lambda v: ("y" if v else "n") if isinstance(v, bool) else str(v)
        print(f"{r['board_id'][:40]:40} {r['pcbworld_tier']:6} {yn(r['clean']):2} {r['attempts']!s:>3} {r['fr_runtime_s']!s:>6} "
              f"{r['fr_incomplete']!s:>4} {r['unconnected']!s:>4} {r['copper_violations']!s:>4} {r['electrical_violations']!s:>3} "
              f"{r['track_length_mm']!s:>8} {r['vias']!s:>4} {str(r['complete_attempts']) + '/' + str(r['attempts']):>4} "
              f"{str(r['clean_pass_attempts']) + '/' + str(r['attempts']):>4} | {r['human_unconnected']!s:>5} "
              f"{r['human_copper_violations']!s:>4} {r['human_track_length_mm']!s:>8} {r['human_vias']!s:>5} "
              f"{r['human_copper_zones']!s:>3}" + (f"  {r['error']}" if r["error"] else ""))
    print("columns: attempt 1 (fr s, inc, unc, cu, el, len, vias), then complete and clean pass attempts out of all, "
          "then the human reference (unc, cu, len, vias, copper zones)")


def compare_mirror(rows, boards):
    """Side by side with the mirror's published per board runs (Freerouting's own counts, not KiCad DRC), and an
    aggregate over every test board per mirror version. Returns the aggregate, which build_summary stores under
    mirror_reference and keeps across later rebuilds. Loads the 69 MB benchmarks.json; run it after a batch, not beside one."""
    test_ids = {b["board_id"] for b in boards}
    clean_ids = {b["board_id"] for b in boards if b["clean"]}
    found = {}
    data = json.load(open(MIRROR_RESULTS))
    for run in data["runs"]:
        rel = run["fixture"]["relative_path"]
        bid = rel.split("/")[1] if rel.count("/") >= 2 else ""
        if bid in test_ids and rel.endswith("/unrouted.dsn"):
            q, e = run["quality"], run["exit"]
            found.setdefault(bid, {})[run["binary"]["version_label"]] = {
                "unrouted": q.get("unrouted_connections"), "clearance": q.get("clearance_violations"),
                "seconds": q.get("wall_clock_seconds"), "state": e.get("state"), "timed_out": e.get("timed_out")}
    del data
    versions = sorted({v for d in found.values() for v in d})
    agg = {}
    for v in versions:
        runs = {bid: d[v] for bid, d in found.items() if v in d}
        routed = {bid for bid, m in runs.items() if m["unrouted"] == 0}
        secs = [m["seconds"] for m in runs.values() if m["seconds"] is not None]
        agg[v] = {"boards": len(runs), "fully_routed": len(routed), "fully_routed_clean_set": len(routed & clean_ids),
                  "fully_routed_zero_clearance": sum(1 for bid in routed if runs[bid]["clearance"] == 0),
                  "state_counts": dict(sorted(collections.Counter(m["state"] for m in runs.values()).items())),
                  "seconds_total": round(sum(secs), 1), "seconds_max": round(max(secs), 1) if secs else None,
                  "pilot_rows_incomplete": sum(1 for bid in rows if bid in runs and runs[bid]["unrouted"] not in (0, None))}
    fields = ["board_id", "ours_attempts", "ours_fr_incomplete_r1", "ours_fr_clearance_r1", "ours_unconnected_kicad_r1",
              "ours_copper_kicad_r1", "ours_best_unconnected", "ours_complete_attempts", "ours_runtime_s_r1"]
    for v in versions:
        fields += [f"{v}_unrouted", f"{v}_clearance", f"{v}_seconds", f"{v}_state"]
    COMPARE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(COMPARE_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for bid in sorted(rows):
            r = rows[bid]
            out = {"board_id": bid, "ours_attempts": r["attempts"], "ours_fr_incomplete_r1": r["fr_incomplete"],
                   "ours_fr_clearance_r1": r["fr_clearance"], "ours_unconnected_kicad_r1": r["unconnected"],
                   "ours_copper_kicad_r1": r["copper_violations"], "ours_best_unconnected": r["best_unconnected"],
                   "ours_complete_attempts": f"{r['complete_attempts']}/{r['attempts']}", "ours_runtime_s_r1": r["fr_runtime_s"]}
            for v in versions:
                m = found.get(bid, {}).get(v, {})
                out.update({f"{v}_unrouted": m.get("unrouted", ""), f"{v}_clearance": m.get("clearance", ""),
                            f"{v}_seconds": m.get("seconds", ""), f"{v}_state": m.get("state", "")})
            w.writerow(out)
    print(f"{'board':40} {'ours r1 inc/unc':>15} {'ours C':>6} " + " ".join(f"{v:>10}" for v in versions)
          + "   (mirror: unrouted connections by Freerouting's own count; T marks a TIMED_OUT state)")
    for bid in sorted(rows):
        r = rows[bid]
        cells = []
        for v in versions:
            m = found.get(bid, {}).get(v, {})
            cells.append(f"{str(m.get('unrouted', '-')) + ('T' if m.get('state') == 'TIMED_OUT' else ''):>10}")
        print(f"{bid[:40]:40} {str(r['fr_incomplete']) + '/' + str(r['unconnected']):>15} "
              f"{str(r['complete_attempts']) + '/' + str(r['attempts']):>6} " + " ".join(cells))
    print("mirror aggregate over the test boards:")
    for v, a in agg.items():
        print(f"  {v:10} fully routed {a['fully_routed']}/{a['boards']} (clean set {a['fully_routed_clean_set']}, "
              f"zero clearance too {a['fully_routed_zero_clearance']}), states {a['state_counts']}, "
              f"total {a['seconds_total']} s, max {a['seconds_max']} s")
    print(f"wrote {COMPARE_CSV}")
    return agg


def write_summary(rows, attempts, boards, mirror=None):
    summary = build_summary(rows, attempts, boards, mirror)
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=1))
    return summary


def print_rates(summary):
    for name in ("clean", "all"):
        s = summary[name]
        if s["n_run"] == 0:
            continue
        print(f"{name}: {s['n_run']} of {s['n_in_set']} boards run, attempts per board {s['attempts_per_board']}; "
              f"complete attempt 1 {s['attempt_1']['complete']}, best {s['best_attempt']['complete']}, every "
              f"{s['every_attempt']['complete']}; clean pass attempt 1 {s['attempt_1']['clean_pass']}, best "
              f"{s['best_attempt']['clean_pass']}, every {s['every_attempt']['clean_pass']}; attempts disagree on "
              f"{s['boards_whose_attempts_disagree']} boards; timeouts {s['timed_out_attempts']}; "
              f"attempts over the pass limit {s['attempts_over_passes_limit']} (max passes {s['passes_run_max']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("board_ids", nargs="*", help="explicit board ids; default is every test board")
    ap.add_argument("--limit", type=int, default=0, help="first N boards of the selection by sorted id")
    ap.add_argument("--clean-only", action="store_true", help="only boards whose human reference is tier clean")
    ap.add_argument("--passes", type=int, default=20, help="Freerouting -mp value (recorded; not applied by 2.1.0 headless)")
    ap.add_argument("--timeout", type=int, default=600, help="wall clock seconds per attempt before Java is killed")
    ap.add_argument("--repeats", type=int, default=1, help="attempts per board; existing attempts are kept and topped up")
    ap.add_argument("--force", action="store_true", help="discard existing attempts of the selected boards and rerun")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--compare-mirror", action="store_true")
    a = ap.parse_args()

    boards = load_test_boards()
    rows, attempts = load_rows(), load_attempts()
    if a.summary_only or a.compare_mirror:
        mirror = compare_mirror(rows, boards) if a.compare_mirror else None
        summary = write_summary(rows, attempts, boards, mirror)
        print_table(rows)
        print_rates(summary)
        return

    if a.board_ids:
        known = {b["board_id"]: b for b in boards}
        missing = [b for b in a.board_ids if b not in known]
        if missing:
            sys.exit(f"not test boards: {missing}")
        selection = [known[b] for b in sorted(set(a.board_ids))]
    else:
        selection = boards
    if a.clean_only:
        selection = [b for b in selection if b["clean"]]
    if a.limit:
        selection = selection[: a.limit]
    todo = []
    for b in selection:
        have = {k for (bid, k) in attempts if bid == b["board_id"]}
        ks = [k for k in range(1, a.repeats + 1) if a.force or k not in have]
        if ks:
            todo.append((b, ks))
    n_att = sum(len(ks) for _, ks in todo)
    print(f"{len(selection)} boards selected, {len(selection) - len(todo)} already done, {len(todo)} to run "
          f"({n_att} attempts); repeats {a.repeats}, -mp {a.passes}, timeout {a.timeout} s", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DRC_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        for i, (b, ks) in enumerate(todo, 1):
            bid = b["board_id"]
            if a.force:
                for key in [key for key in attempts if key[0] == bid]:
                    del attempts[key]
            for k in ks:
                r = run_attempt(b, k, a.passes, a.timeout)
                attempts[(bid, k)] = coerce(r, ATTEMPT_FIELDS)
                save_csv(ATTEMPTS_CSV, attempts, ATTEMPT_FIELDS)
                print(f"  {i}/{len(todo)} {bid[:36]:36} r{k} fr {r['fr_runtime_s']}s rc {r['fr_rc']} passes {r['fr_passes_run']} "
                      f"inc {r['fr_incomplete']} | import {r['import_ok']} | unc {r['unconnected']} cu {r['copper_violations']} "
                      f"len {r['track_length_mm']} vias {r['vias']} | {time.time() - t0:.0f}s"
                      + (f" | {r['error']}" if r["error"] else ""), flush=True)
            old = rows.get(bid)
            if old and not a.force and all(old[c] != "" for c in HUMAN_COLS):
                human, hs, herr = {c: old[c] for c in HUMAN_COLS}, 0.0, ""
            else:
                human, hs, herr = human_metrics(b)
            mine = [v for (x, _), v in attempts.items() if x == bid]
            rows[bid] = coerce(board_row(b, mine, human, hs, herr), FIELDS)
            save_csv(RESULTS_CSV, rows, FIELDS)
            r = rows[bid]
            print(f"  {i}/{len(todo)} {bid[:36]:36} board: complete {r['complete_attempts']}/{r['attempts']} clean "
                  f"{r['clean_pass_attempts']}/{r['attempts']} | human unc {r['human_unconnected']} cu {r['human_copper_violations']} "
                  f"len {r['human_track_length_mm']} vias {r['human_vias']} zones {r['human_copper_zones']}", flush=True)
    finally:
        save_csv(ATTEMPTS_CSV, attempts, ATTEMPT_FIELDS)
        save_csv(RESULTS_CSV, rows, FIELDS)
        summary = write_summary(rows, attempts, boards)
    print()
    print_table({b["board_id"]: rows[b["board_id"]] for b in selection if b["board_id"] in rows})
    print()
    print_rates(summary)
    print(f"total {time.time() - t0:.0f}s, rows in {RESULTS_CSV}: {len(rows)}, attempts in {ATTEMPTS_CSV}: {len(attempts)}")


if __name__ == "__main__":
    main()
