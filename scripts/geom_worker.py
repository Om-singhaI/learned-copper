"""Runs inside KiCad's bundled Python. Dumps board geometry as JSON for rasterization.

Usage: $KICAD_PY scripts/geom_worker.py OUT_DIR BOARD.kicad_pcb [BOARD2 ...]
Units: millimetres. Copper layers are named by KiCad (F.Cu, B.Cu, In1.Cu ...).
"""
import json
import sys
import traceback
from pathlib import Path

import pcbnew

MM = 1e-6


def box(b):
    return [round(b.GetX() * MM, 4), round(b.GetY() * MM, 4), round(b.GetWidth() * MM, 4), round(b.GetHeight() * MM, 4)]


def poly_points(poly_set):
    outs = []
    for i in range(poly_set.OutlineCount()):
        o = poly_set.Outline(i)
        outs.append([[round(o.CPoint(j).x * MM, 4), round(o.CPoint(j).y * MM, 4)] for j in range(o.PointCount())])
    return outs


def describe(path):
    board = pcbnew.LoadBoard(path)
    copper = [board.GetLayerName(l) for l in board.GetEnabledLayers().CuStack()]
    bbox = board.GetBoardEdgesBoundingBox()
    outline = []
    try:
        ps = pcbnew.SHAPE_POLY_SET()
        if board.GetBoardPolygonOutlines(ps):
            outline = poly_points(ps)
    except Exception:
        outline = []
    nets = {}
    for code, net in board.GetNetInfo().NetsByNetcode().items():
        if code == 0:
            continue
        entry = {"name": net.GetNetname()}
        try:
            nc = net.GetNetClass()
            entry["netclass"] = nc.GetName()
            entry["track_width_mm"] = round(nc.GetTrackWidth() * MM, 4)
            entry["clearance_mm"] = round(nc.GetClearance() * MM, 4)
        except Exception:
            pass
        nets[int(code)] = entry
    pads = []
    for fp in board.GetFootprints():
        ref = fp.GetReference()
        for pad in fp.Pads():
            layers = [board.GetLayerName(l) for l in pad.GetLayerSet().CuStack()]
            if not layers:
                continue
            pads.append({"ref": ref, "net": int(pad.GetNetCode()), "box": box(pad.GetBoundingBox()),
                         "layers": layers, "th": bool(pad.HasHole())})
    tracks, vias = [], []
    for t in board.GetTracks():
        cls = t.GetClass()
        if cls == "PCB_VIA":
            vias.append({"net": int(t.GetNetCode()), "box": box(t.GetBoundingBox()),
                         "layers": [board.GetLayerName(l) for l in t.GetLayerSet().CuStack()]})
        elif cls in ("PCB_TRACK", "PCB_ARC"):
            s, e = t.GetStart(), t.GetEnd()
            tracks.append({"net": int(t.GetNetCode()), "layer": board.GetLayerName(t.GetLayer()),
                           "w": round(t.GetWidth() * MM, 4),
                           "a": [round(s.x * MM, 4), round(s.y * MM, 4)], "b": [round(e.x * MM, 4), round(e.y * MM, 4)]})
    zones = []
    for z in board.Zones():
        if not z.IsOnCopperLayer():
            continue
        try:
            zones.append({"net": int(z.GetNetCode()), "layers": [board.GetLayerName(l) for l in z.GetLayerSet().CuStack()],
                          "outline": poly_points(z.Outline())})
        except Exception:
            pass
    return {"path": path, "copper_layers": copper, "bbox": box(bbox), "outline": outline, "nets": nets,
            "pads": pads, "tracks": tracks, "vias": vias, "zones": zones}


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in sys.argv[2:]:
        bid = Path(p).parent.name
        try:
            d = describe(p)
            json.dump(d, open(out_dir / f"{bid}.json", "w"))
            print(json.dumps({"board_id": bid, "ok": True, "pads": len(d["pads"]), "tracks": len(d["tracks"]), "nets": len(d["nets"])}), flush=True)
        except Exception as e:
            print(json.dumps({"board_id": bid, "ok": False, "error": f"{type(e).__name__}: {e}"}), flush=True)
