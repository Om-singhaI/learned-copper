"""Routing metrics for any .kicad_pcb, human routed or machine routed (issue 7).

Per board: unconnected count after zone refill (KiCad 10 design rule check with --refill-zones and
--severity-all), copper rule violations (errors whose type is in drc_batch.COPPER_TYPES) and the
electrical subset of them (errors whose type is in drc_batch.ELECTRICAL_TYPES, so always a subset of
the copper count; drc_batch.summarize counts the electrical types at any severity, this script does
not), total track length in mm, track segment count, via count and the number of copper zones. The
counts come from KiCad's bundled Python through scripts/inventory_worker.py, one subprocess per
board with a timeout. Arcs count as segments; segments on non copper layers are not tracks in KiCad
and are not counted, which is why C-BISCUIT_crowbar reads 40 segments and 98.73 mm against the
mirror's ground_truth.json 41 and 116.53 (its extra segment sits on Dwgs.User with no net).

    .venv/bin/python scripts/metrics.py a.kicad_pcb b.kicad_pcb --out data/metrics.csv
    .venv/bin/python scripts/metrics.py data/freerouting/*.freerouting.kicad_pcb --summary

A board is complete when unconnected is 0 and passes cleanly when it is complete with no copper rule
violation. The refill matters: fills stored in old files are stale and produce clearance errors that
vanish once the zones are refilled, so every check here refills first. The DRC JSON reports are kept
under --drc-dir (default data/drc/metrics/) and are overwritten on every run, never read stale.
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drc_batch import ELECTRICAL_TYPES, FIX, KICAD_CLI, summarize  # noqa: E402
from inventory import run_chunk  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRC_DIR = ROOT / "data/drc/metrics"
FIELDS = ["board_id", "path", "unconnected", "copper_violations", "electrical_violations", "drc_errors",
          "drc_warnings", "track_length_mm", "segments", "vias", "copper_zones", "copper_layers", "complete",
          "clean_pass", "drc_seconds", "counts_seconds", "by_type", "error"]
OUTPUT_SUFFIXES = (".freerouting", ".routed", ".guided", ".unguided")
ATTEMPT_TAG = re.compile(r"\.r\d+$")


def board_id_for(path):
    """Mirror boards are named by their folder; routed outputs by the file name minus the router tag
    and the attempt tag (board.r2.freerouting.kicad_pcb is attempt 2 of board)."""
    p = Path(path).resolve()
    if p.parent.parent == FIX.resolve():
        return p.parent.name
    stem = p.stem
    for s in OUTPUT_SUFFIXES:
        if stem.endswith(s):
            stem = stem[: -len(s)]
            break
    return ATTEMPT_TAG.sub("", stem)


def electrical_errors(report_path):
    """Error severity violations whose type is electrical. Returns None if the report cannot be read."""
    try:
        d = json.load(open(report_path))
    except (OSError, json.JSONDecodeError):
        return None
    return sum(1 for v in d.get("violations", [])
               if v.get("severity") == "error" and v.get("type") in ELECTRICAL_TYPES)


def report_name(path):
    """Unique name for the DRC report: folder.file for mirror boards, file stem otherwise."""
    p = Path(path).resolve()
    if p.parent.parent == FIX.resolve():
        return f"{p.parent.name}.{p.stem}"
    return p.stem


def run_drc_refill(board, out_json, timeout):
    """Same kicad-cli call as drc_batch.run_drc with refill on, but the report path is the caller's.
    Returns (ok, error_text, seconds)."""
    out_json = Path(out_json)
    if out_json.exists():
        out_json.unlink()
    cmd = [KICAD_CLI, "pcb", "drc", "--format", "json", "--severity-all", "--units", "mm",
           "--refill-zones", "-o", str(out_json), str(board)]
    t = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "drc timeout", round(time.time() - t, 1)
    if not out_json.exists():
        tail = (p.stderr or p.stdout).strip().replace("\n", " ")[-200:]
        return False, "drc produced no report: " + tail, round(time.time() - t, 1)
    return True, "", round(time.time() - t, 1)


def board_metrics(board, drc_dir=DEFAULT_DRC_DIR, board_id=None, drc_timeout=300, counts_timeout=120):
    """One metrics row for one board. Never raises; failures land in row['error']."""
    board = str(Path(board).resolve())
    drc_dir = Path(drc_dir)
    drc_dir.mkdir(parents=True, exist_ok=True)
    row = {k: "" for k in FIELDS}
    row.update(board_id=board_id or board_id_for(board), path=board, error="")
    errors = []
    if not Path(board).exists():
        row["error"] = "missing file"
        return row

    out_json = drc_dir / (report_name(board) + ".json")
    ok, err, secs = run_drc_refill(board, out_json, drc_timeout)
    row["drc_seconds"] = secs
    if ok:
        s = summarize(out_json, row["board_id"], secs)
        el = electrical_errors(out_json)
        row.update(unconnected=s["unconnected"], copper_violations=s["copper_errors"],
                   electrical_violations=el if el is not None else "", drc_errors=s["errors"],
                   drc_warnings=s["warnings"], by_type=s["by_type"])
        row["complete"] = s["unconnected"] == 0
        row["clean_pass"] = row["complete"] and s["copper_errors"] == 0
    else:
        errors.append(err)

    t = time.time()
    got = run_chunk([board], counts_timeout).get(board)
    row["counts_seconds"] = round(time.time() - t, 1)
    if not got:
        errors.append("counts timeout or crash")
    elif got.get("error"):
        errors.append("counts: " + got["error"][:160])
    else:
        row.update(track_length_mm=got["track_length_mm"], segments=got["segments"], vias=got["vias"],
                   copper_zones=got.get("copper_zones", ""), copper_layers=got["copper_layers"])
    row["error"] = "; ".join(errors)
    return row


def write_csv(rows, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(r[k]) if k == "by_type" and r[k] != "" else r.get(k, "")) for k in FIELDS})


def print_summary(rows):
    cols = [("board_id", 44), ("unconnected", 11), ("copper_violations", 6), ("electrical_violations", 4),
            ("track_length_mm", 10), ("segments", 8), ("vias", 5), ("complete", 8), ("clean_pass", 5)]
    heads = ["board", "unconnected", "copper", "elec", "length mm", "segments", "vias", "complete", "clean"]
    print("  ".join(h.ljust(w) if i == 0 else h.rjust(w) for i, ((_, w), h) in enumerate(zip(cols, heads))))
    for r in rows:
        cells = []
        for i, (k, w) in enumerate(cols):
            v = r.get(k, "")
            if isinstance(v, bool):
                v = "yes" if v else "no"
            cells.append(str(v)[:w].ljust(w) if i == 0 else str(v).rjust(w))
        print("  ".join(cells) + (f"  {r['error']}" if r.get("error") else ""))
    scored = [r for r in rows if r.get("complete") != ""]
    n = len(rows)
    print(f"\n{n} boards, {len(scored)} checked, complete (unconnected 0): {sum(1 for r in scored if r['complete'])}, "
          f"clean pass (complete and no copper violation): {sum(1 for r in scored if r['clean_pass'])}, "
          f"failed: {n - len(scored)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("boards", nargs="+", help=".kicad_pcb files")
    ap.add_argument("--out", default=str(ROOT / "data/metrics.csv"))
    ap.add_argument("--drc-dir", default=str(DEFAULT_DRC_DIR))
    ap.add_argument("--summary", action="store_true", help="print a table after the run")
    ap.add_argument("--timeout", type=int, default=300, help="seconds per board for the design rule check")
    a = ap.parse_args()
    t0 = time.time()
    rows = []
    for i, b in enumerate(a.boards, 1):
        r = board_metrics(b, a.drc_dir, drc_timeout=a.timeout)
        rows.append(r)
        print(f"  {i}/{len(a.boards)} {r['board_id']}: unconnected {r['unconnected']}, copper {r['copper_violations']}, "
              f"length {r['track_length_mm']} mm, vias {r['vias']} {r['error']}", flush=True)
    write_csv(rows, a.out)
    print(f"wrote {len(rows)} rows to {a.out} in {time.time() - t0:.0f}s")
    if a.summary:
        print()
        print_summary(rows)


if __name__ == "__main__":
    main()
