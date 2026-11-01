"""Dump geometry JSON for every usable mirror board (tier clean or flagged, held out or not).

    .venv/bin/python scripts/extract_geometry.py            # all usable boards
    .venv/bin/python scripts/extract_geometry.py --limit 5
"""
import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KICAD_PY = "/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3"
FIX = ROOT / "data/mirror/freerouting/scripts/benchmark/fixtures/PCBench"
OUT = ROOT / "data/geom"


def run_chunk(paths, timeout=600):
    try:
        p = subprocess.run([KICAD_PY, str(ROOT / "scripts/geom_worker.py"), str(OUT), *paths], capture_output=True, text=True, timeout=timeout)
        return [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    except subprocess.TimeoutExpired:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    pool = json.load(open(ROOT / "splits/mirror_pool.json"))["boards"]
    boards = [str(FIX / b["board_id"] / "raw.kicad_pcb") for b in pool if b["tier"] != "rejected"]
    if a.limit:
        boards = boards[: a.limit]
    t0 = time.time()
    chunks = [boards[i:i + 40] for i in range(0, len(boards), 40)]
    results = []
    with ThreadPoolExecutor(4) as ex:
        for got in ex.map(run_chunk, chunks):
            results.extend(got)
    done = {r["board_id"] for r in results if r.get("ok")}
    missing = [b for b in boards if Path(b).parent.name not in done]
    for b in missing:
        results.extend(run_chunk([b], 120))
    ok = [r for r in results if r.get("ok")]
    print(f"{len(boards)} boards, {len(ok)} extracted, {time.time() - t0:.0f}s -> {OUT}")
    for r in results:
        if not r.get("ok"):
            print("  failed:", r.get("board_id"), r.get("error", "")[:120])


if __name__ == "__main__":
    main()
