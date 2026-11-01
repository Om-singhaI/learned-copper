"""Rasterize board geometry into per net training samples.

    .venv/bin/python scripts/rasterize.py                # all boards in data/geom -> data/raster, splits/raster_manifest.json
    .venv/bin/python scripts/rasterize.py --limit 3 --size 256

Each board becomes <board_id>.npz with
  x: uint8 [N, 6, S, S]: outline, this net pads front, this net pads back, other pads front, other pads back,
     other nets' copper on either layer
  y: uint8 [N, 2, S, S]: this net's copper (tracks and vias) front, back
  net_ids: int32 [N], mm_per_px: float32
One sample per net with at least two pads, at most MAX_NETS nets per board (fixed seed). Two layer boards only.
"""
import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
GEOM, OUT = ROOT / "data/geom", ROOT / "data/raster"
MAX_NETS = 60
MARGIN_PX = 6


class Raster:
    def __init__(self, bbox, size):
        x, y, w, h = bbox
        self.size = size
        self.mm_per_px = max(w, h) / (size - 2 * MARGIN_PX)
        self.ox = x - ((size * self.mm_per_px) - w) / 2
        self.oy = y - ((size * self.mm_per_px) - h) / 2

    def px(self, mx, my):
        return (mx - self.ox) / self.mm_per_px, (my - self.oy) / self.mm_per_px

    def canvas(self):
        return Image.new("L", (self.size, self.size), 0)

    def rect(self, img, box):
        x, y, w, h = box
        x0, y0 = self.px(x, y)
        x1, y1 = self.px(x + w, y + h)
        ImageDraw.Draw(img).rectangle([x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)], fill=1)

    def disc(self, img, box):
        x, y, w, h = box
        x0, y0 = self.px(x, y)
        x1, y1 = self.px(x + w, y + h)
        ImageDraw.Draw(img).ellipse([x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)], fill=1)

    def line(self, img, a, b, w_mm):
        w = max(1, int(round(w_mm / self.mm_per_px)))
        (x0, y0), (x1, y1) = self.px(*a), self.px(*b)
        d = ImageDraw.Draw(img)
        d.line([x0, y0, x1, y1], fill=1, width=w)
        r = w / 2
        for cx, cy in ((x0, y0), (x1, y1)):
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=1)

    def polygon(self, img, pts):
        ImageDraw.Draw(img).polygon([self.px(*p) for p in pts], fill=1)


def rasterize_board(args):
    path, size = args
    d = json.load(open(path))
    bid = Path(path).stem
    if len(d["copper_layers"]) != 2:
        return {"board_id": bid, "skipped": f"{len(d['copper_layers'])} copper layers"}
    # designers rename layers; the first copper layer is the front, the last is the back
    rename = {d["copper_layers"][0]: "F.Cu", d["copper_layers"][-1]: "B.Cu"}
    for p in d["pads"]:
        p["layers"] = [rename.get(l, l) for l in p["layers"]]
    for t in d["tracks"]:
        t["layer"] = rename.get(t["layer"], t["layer"])
    R = Raster(d["bbox"], size)
    outline = R.canvas()
    if d["outline"]:
        for poly in d["outline"]:
            R.polygon(outline, poly)
    else:
        R.rect(outline, d["bbox"])
    outline = np.array(outline, dtype=np.uint8)

    pads_by_net = {}
    for p in d["pads"]:
        pads_by_net.setdefault(p["net"], []).append(p)
    nets = sorted(n for n, ps in pads_by_net.items() if n != 0 and len(ps) >= 2)
    rng = random.Random(0)
    if len(nets) > MAX_NETS:
        nets = sorted(rng.sample(nets, MAX_NETS))
    if not nets:
        return {"board_id": bid, "skipped": "no nets with two pads"}

    # pad layers per net, copper per net, drawn once and reused
    def pad_layers(p):
        layers = set(p["layers"]) & {"F.Cu", "B.Cu"}
        if p["th"]:
            layers = {"F.Cu", "B.Cu"}
        return layers

    all_pads = {"F.Cu": R.canvas(), "B.Cu": R.canvas()}
    net_pads = {n: {"F.Cu": R.canvas(), "B.Cu": R.canvas()} for n in nets}
    for p in d["pads"]:
        for layer in pad_layers(p):
            R.rect(all_pads[layer], p["box"])
            if p["net"] in net_pads:
                R.rect(net_pads[p["net"]][layer], p["box"])
    all_copper = {"F.Cu": R.canvas(), "B.Cu": R.canvas()}
    net_copper = {n: {"F.Cu": R.canvas(), "B.Cu": R.canvas()} for n in nets}
    for t in d["tracks"]:
        if t["layer"] not in all_copper:
            continue
        R.line(all_copper[t["layer"]], t["a"], t["b"], t["w"])
        if t["net"] in net_copper:
            R.line(net_copper[t["net"]][t["layer"]], t["a"], t["b"], t["w"])
    for v in d["vias"]:
        for layer in ("F.Cu", "B.Cu"):
            R.disc(all_copper[layer], v["box"])
            if v["net"] in net_copper:
                R.disc(net_copper[v["net"]][layer], v["box"])
    A = {k: np.array(v, dtype=np.uint8) for k, v in all_pads.items()}
    C = {k: np.array(v, dtype=np.uint8) for k, v in all_copper.items()}
    X, Y, ids = [], [], []
    for n in nets:
        pf, pb = (np.array(net_pads[n][l], dtype=np.uint8) for l in ("F.Cu", "B.Cu"))
        cf, cb = (np.array(net_copper[n][l], dtype=np.uint8) for l in ("F.Cu", "B.Cu"))
        if cf.sum() + cb.sum() == 0:
            continue  # net with pads but no copper of its own (zone only or unrouted)
        other_f, other_b = np.clip(A["F.Cu"] - pf, 0, 1), np.clip(A["B.Cu"] - pb, 0, 1)
        other_copper = np.clip(np.maximum(C["F.Cu"], C["B.Cu"]) - np.maximum(cf, cb), 0, 1)
        X.append(np.stack([outline, pf, pb, other_f, other_b, other_copper]))
        Y.append(np.stack([cf, cb]))
        ids.append(n)
    if not X:
        return {"board_id": bid, "skipped": "no net with copper"}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{bid}.npz", x=np.stack(X).astype(np.uint8), y=np.stack(Y).astype(np.uint8),
                        net_ids=np.array(ids, dtype=np.int32), mm_per_px=np.float32(R.mm_per_px))
    return {"board_id": bid, "n_samples": len(X), "mm_per_px": round(R.mm_per_px, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    files = sorted(GEOM.glob("*.json"))
    if a.limit:
        files = files[: a.limit]
    pool = {b["board_id"]: b for b in json.load(open(ROOT / "splits/mirror_pool.json"))["boards"]}
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(rasterize_board, [(str(f), a.size) for f in files]))
    boards = []
    for r in results:
        if "n_samples" in r:
            b = pool[r["board_id"]]
            boards.append({"board_id": r["board_id"], "source_repo": b["source_repo"], "held_out": b["held_out"],
                           "in_test_set": b["in_test_set"], "tier": b["tier"], "n_samples": r["n_samples"], "mm_per_px": r["mm_per_px"]})
    json.dump({"size": a.size, "boards": boards}, open(ROOT / "splits/raster_manifest.json", "w"), indent=1)
    skipped = [r for r in results if "skipped" in r]
    n = sum(b["n_samples"] for b in boards)
    print(f"{len(files)} boards, {len(boards)} rasterized ({n} samples, {sum(1 for b in boards if not b['held_out'])} train boards, {sum(1 for b in boards if b['held_out'])} held out), {len(skipped)} skipped, {time.time() - t0:.0f}s")
    import collections
    print("skip reasons:", collections.Counter(r["skipped"] for r in skipped).most_common(5))
    mm = sorted(b["mm_per_px"] for b in boards)
    if mm:
        print(f"mm per pixel: median {mm[len(mm)//2]:.3f}, min {mm[0]:.3f}, max {mm[-1]:.3f}")


if __name__ == "__main__":
    main()
