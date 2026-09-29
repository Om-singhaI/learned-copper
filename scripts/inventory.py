"""Inventory every board in the PCBench mirror (or any list of .kicad_pcb files).

Runs scripts/inventory_worker.py under KiCad's Python in chunks with timeouts, then writes data/inventory.csv
and cross checks segments, vias, layers and unconnected counts against the mirror's ground_truth.json.

    .venv/bin/python scripts/inventory.py            # all mirror boards
    .venv/bin/python scripts/inventory.py a.kicad_pcb b.kicad_pcb
"""
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KICAD_PY = "/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3"
WORKER = ROOT / "scripts/inventory_worker.py"
FIX = ROOT / "data/mirror/freerouting/scripts/benchmark/fixtures/PCBench"
OUT = ROOT / "data/inventory.csv"
CHUNK, WORKERS, CHUNK_TIMEOUT, SINGLE_TIMEOUT = 40, 4, 600, 90


def run_chunk(paths, timeout):
    try:
        p = subprocess.run([KICAD_PY, str(WORKER), *paths], capture_output=True, text=True, timeout=timeout)
        rows = {}
        for line in p.stdout.splitlines():
            if line.startswith("{"):
                r = json.loads(line)
                rows[r["path"]] = r
        return rows
    except subprocess.TimeoutExpired:
        return {}


def main():
    if len(sys.argv) > 1:
        boards = [str(Path(p).resolve()) for p in sys.argv[1:]]
    else:
        boards = sorted(str(p) for p in FIX.glob("*/raw.kicad_pcb"))
    t0 = time.time()
    chunks = [boards[i:i + CHUNK] for i in range(0, len(boards), CHUNK)]
    rows = {}
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, got in enumerate(ex.map(lambda c: run_chunk(c, CHUNK_TIMEOUT), chunks)):
            rows.update(got)
            print(f"  chunk {i + 1}/{len(chunks)} done, {len(rows)} boards, {time.time() - t0:.0f}s", flush=True)
    missing = [b for b in boards if b not in rows]
    if missing:
        print(f"retrying {len(missing)} boards one at a time", flush=True)
        for b in missing:
            rows.update(run_chunk([b], SINGLE_TIMEOUT) or {b: {"path": b, "error": "timeout or crash"}})

    # cross check against ground truth where it exists
    fields = ["board_id", "kicad_file_version", "copper_layers", "width_mm", "height_mm", "area_cm2", "footprints", "pads", "nets",
              "segments", "vias", "track_length_mm", "copper_zones", "unconnected", "gt_segments", "gt_vias", "gt_layers",
              "gt_unconnected", "gt_track_length_mm", "error"]
    out_rows, mismatches = [], {"segments": 0, "vias": 0, "layers": 0, "unconnected": 0}
    for b in boards:
        r = dict(rows.get(b, {"path": b, "error": "no result"}))
        folder = Path(b).parent
        r["board_id"] = folder.name if folder.parent == FIX else Path(b).stem
        gt_path = folder / "ground_truth.json"
        if gt_path.exists():
            gt = json.load(open(gt_path))
            r["gt_segments"], r["gt_vias"], r["gt_layers"] = gt.get("segment_count"), gt.get("via_count"), gt.get("layers")
            r["gt_unconnected"] = (gt.get("kicad_drc_breakdown") or {}).get("unconnected_count")
            r["gt_track_length_mm"] = gt.get("approximate_track_length_mm")
            if "error" not in r:
                for k, g in (("segments", "gt_segments"), ("vias", "gt_vias"), ("copper_layers", "gt_layers"), ("unconnected", "gt_unconnected")):
                    if r.get(g) is not None and r.get(k) != r.get(g):
                        mismatches[k.replace("copper_", "")] += 1
        out_rows.append({k: r.get(k, "") for k in fields})
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    ok = [r for r in out_rows if not r["error"]]
    print(f"\n{len(out_rows)} boards, {len(ok)} parsed, {len(out_rows) - len(ok)} failed, {time.time() - t0:.0f}s -> {OUT}")
    if ok:
        two = sum(1 for r in ok if r["copper_layers"] == 2)
        tracks = sum(1 for r in ok if r["segments"] > 0)
        routed = sum(1 for r in ok if r["segments"] > 0 and r["unconnected"] == 0)
        fps = sorted(r["footprints"] for r in ok)
        q = lambda p: fps[int(len(fps) * p)]
        print(f"two layer: {two} | with any tracks: {tracks} | tracks and zero unconnected: {routed}")
        print(f"footprints per board: min {fps[0]}, p25 {q(0.25)}, median {q(0.5)}, p75 {q(0.75)}, p95 {q(0.95)}, max {fps[-1]}")
        print(f"mismatches vs ground_truth.json: {mismatches} (of {sum(1 for r in ok if r['gt_segments'] != '')} boards with ground truth)")
    for r in out_rows:
        if r["error"]:
            print("  failed:", r["board_id"], r["error"][:120])


if __name__ == "__main__":
    main()
