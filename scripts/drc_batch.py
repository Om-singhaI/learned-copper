"""Run KiCad's design rule check on many boards in parallel and summarize the results.

    .venv/bin/python scripts/drc_batch.py                 # all mirror boards -> data/drc/plain/
    .venv/bin/python scripts/drc_batch.py --limit 30      # pilot
    .venv/bin/python scripts/drc_batch.py --refill        # refill copper zones before checking

Writes one JSON report per board under data/drc/<tag>/ and data/drc_summary_<tag>.csv with, per board:
unconnected count, error and warning counts, counts per violation type, and how many clearance errors
involve a zone fill.
"""
import argparse
import collections
import csv
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KICAD_CLI = "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli"
FIX = ROOT / "data/mirror/freerouting/scripts/benchmark/fixtures/PCBench"

# Rule types that concern copper and routing. Everything else (silkscreen, courtyards, library
# comparisons, solder mask, text) is reported but never used to reject a board.
COPPER_TYPES = {
    "clearance", "shorting_items", "tracks_crossing", "copper_edge_clearance", "zones_intersect",
    "via_dangling", "track_dangling", "hole_clearance", "hole_near_hole", "holes_co_located",
    "connection_width", "track_width", "annular_width", "drill_out_of_range", "via_diameter",
    "microvia_drill_out_of_range", "copper_sliver", "items_not_allowed", "isolated_copper",
    "starved_thermal", "padstack", "padstack_invalid", "invalid_outline",
}
ELECTRICAL_TYPES = {"shorting_items", "tracks_crossing", "zones_intersect"}


def run_drc(board, out_dir, refill, timeout):
    board_id = Path(board).parent.name
    out = out_dir / (board_id + ".json")
    cmd = [KICAD_CLI, "pcb", "drc", "--format", "json", "--severity-all", "--units", "mm", "-o", str(out)]
    if refill:
        cmd.append("--refill-zones")
    cmd.append(board)
    t = time.time()
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"board_id": board_id, "error": "timeout", "seconds": round(time.time() - t, 1)}
    if not out.exists():
        return {"board_id": board_id, "error": "no report", "seconds": round(time.time() - t, 1)}
    return summarize(out, board_id, round(time.time() - t, 1))


def summarize(report_path, board_id, seconds):
    d = json.load(open(report_path))
    viol = d.get("violations", [])
    by_type = collections.Counter()
    errors = warnings = 0
    clearance_zone = clearance_total = 0
    for v in viol:
        t = v.get("type", "?")
        by_type[t] += 1
        if v.get("severity") == "error":
            errors += 1
        else:
            warnings += 1
        if t == "clearance":
            clearance_total += 1
            desc = " ".join(str(i.get("description", "")) for i in v.get("items", []))
            if "zone" in desc.lower():
                clearance_zone += 1
    copper_errors = sum(1 for v in viol if v.get("severity") == "error" and v.get("type") in COPPER_TYPES)
    electrical = sum(by_type[t] for t in ELECTRICAL_TYPES)
    return {
        "board_id": board_id, "seconds": seconds, "unconnected": len(d.get("unconnected_items", [])),
        "errors": errors, "warnings": warnings, "copper_errors": copper_errors, "electrical_errors": electrical,
        "clearance_total": clearance_total, "clearance_with_zone": clearance_zone,
        "invalid_outline": by_type["invalid_outline"], "by_type": dict(by_type), "error": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("boards", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--refill", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=180)
    a = ap.parse_args()
    boards = a.boards or sorted(str(p) for p in FIX.glob("*/raw.kicad_pcb"))
    if a.limit:
        boards = boards[: a.limit]
    tag = "refill" if a.refill else "plain"
    out_dir = ROOT / "data/drc" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(lambda b: run_drc(b, out_dir, a.refill, a.timeout), boards), 1):
            rows.append(r)
            if i % 100 == 0 or i == len(boards):
                print(f"  {i}/{len(boards)} boards, {time.time() - t0:.0f}s", flush=True)
    fields = ["board_id", "seconds", "unconnected", "errors", "warnings", "copper_errors", "electrical_errors",
              "clearance_total", "clearance_with_zone", "invalid_outline", "by_type", "error"]
    with open(ROOT / f"data/drc_summary_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(r.get(k)) if k == "by_type" else r.get(k, "")) for k in fields})
    ok = [r for r in rows if not r.get("error")]
    print(f"\n{len(rows)} boards, {len(ok)} checked, {len(rows) - len(ok)} failed, {time.time() - t0:.0f}s, mode={tag}")
    if ok:
        q = lambda xs, p: sorted(xs)[int(len(xs) * p)]
        ce = [r["copper_errors"] for r in ok]
        print(f"unconnected==0: {sum(1 for r in ok if r['unconnected'] == 0)} | invalid outline: {sum(1 for r in ok if r['invalid_outline'])} | electrical errors>0: {sum(1 for r in ok if r['electrical_errors'])}")
        print(f"copper errors per board: zero on {sum(1 for c in ce if c == 0)}, median {q(ce, .5)}, p75 {q(ce, .75)}, p90 {q(ce, .9)}")
        print(f"clearance: boards with any {sum(1 for r in ok if r['clearance_total'])}, involving a zone {sum(r['clearance_with_zone'] for r in ok)} of {sum(r['clearance_total'] for r in ok)}")
        agg = collections.Counter()
        for r in ok:
            agg.update(r["by_type"])
        print("top types:", agg.most_common(12))
    for r in rows:
        if r.get("error"):
            print("  failed:", r["board_id"], r["error"])


if __name__ == "__main__":
    main()
