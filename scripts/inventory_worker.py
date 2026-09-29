"""Runs inside KiCad's bundled Python. Prints one JSON line per board with structural counts.

Usage: $KICAD_PY scripts/inventory_worker.py BOARD.kicad_pcb [BOARD2.kicad_pcb ...]
"""
import json
import re
import sys
import traceback

import pcbnew

MM = 1e-6  # nanometres to millimetres


def describe(path):
    head = open(path, "rb").read(200).decode("utf-8", "replace")
    m = re.search(r"\(version\s+(\d+)", head)
    version = m.group(1) if m else ""
    board = pcbnew.LoadBoard(path)
    bbox = board.GetBoardEdgesBoundingBox()
    w, h = bbox.GetWidth() * MM, bbox.GetHeight() * MM
    footprints = list(board.GetFootprints())
    pads = sum(len(list(fp.Pads())) for fp in footprints)
    segs = vias = 0
    track_len = 0.0
    for t in board.GetTracks():
        cls = t.GetClass()
        if cls == "PCB_VIA":
            vias += 1
        elif cls in ("PCB_TRACK", "PCB_ARC"):
            segs += 1
            track_len += t.GetLength() * MM
    zones = sum(1 for z in board.Zones() if z.IsOnCopperLayer())
    conn = board.GetConnectivity()
    try:
        conn.Build(board)
    except TypeError:
        board.BuildConnectivity()
    try:
        unconnected = conn.GetUnconnectedCount(True)
    except TypeError:
        unconnected = conn.GetUnconnectedCount()
    return {
        "path": path,
        "kicad_file_version": version,
        "copper_layers": board.GetCopperLayerCount(),
        "width_mm": round(w, 2),
        "height_mm": round(h, 2),
        "area_cm2": round(w * h / 100, 2),
        "footprints": len(footprints),
        "pads": pads,
        "nets": max(board.GetNetCount() - 1, 0),
        "segments": segs,
        "vias": vias,
        "track_length_mm": round(track_len, 2),
        "copper_zones": zones,
        "unconnected": unconnected,
    }


if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            print(json.dumps(describe(p)), flush=True)
        except Exception as e:  # keep going, the driver records the failure
            print(json.dumps({"path": p, "error": f"{type(e).__name__}: {e}"}), flush=True)
