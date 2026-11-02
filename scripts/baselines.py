"""Two baselines for copper prediction: a distance heuristic and LightGBM on per pixel features.

    .venv/bin/python scripts/baselines.py --split owner
    .venv/bin/python scripts/baselines.py --fixture        # smoke test on synthetic boards

Reads the same data/raster files and splits/raster_manifest.json as scripts/train_unet.py and
reuses its split logic, so both baselines see exactly the samples the UNet sees.
Writes <out>/metrics.json and, with --eval-held-out, <out>/held_out_metrics.json.

Baseline A, heuristic: p = exp(-d / tau), d the distance in pixels to this net's pads on the same
layer, zero outside the outline; tau is picked on the training set from [4, 8, 16, 32].
Baseline B, LightGBM: one model per copper layer on nine per pixel features (distances to pads
and other copper, outline flag, position, distance to the pad bounding box), trained on at most
200k sampled pixels with at most 30 percent positives; thresholds 0.5 and tuned on train.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_unet import (
    DEFAULT_DATA,
    DEFAULT_MANIFEST,
    FIXTURE_DIR,
    LAYER_NAMES,
    OUT_CHANNELS,
    ROOT,
    RasterStore,
    board_samples,
    count_pixels,
    fmt,
    group_by_board,
    load_manifest,
    make_fixture,
    metrics_from_counts,
    split_samples,
)

TAU_GRID = (4.0, 8.0, 16.0, 32.0)
DEFAULT_TAU = 8.0
LN2 = math.log(2.0)
FEATURES = ("d_pads_front", "d_pads_back", "d_other_pads_front", "d_other_pads_back",
            "d_other_copper", "outline", "pos_x", "pos_y", "d_pad_bbox")
# Real copper covers well under one percent of pixels while the sampled training set is far
# richer in positives, so the useful thresholds sit close to one; the grid is finer up there.
THRESHOLD_GRID = np.round(np.concatenate([np.arange(0.05, 0.96, 0.05), [0.97, 0.98, 0.99]]), 2)
LGB_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1}
LGB_ROUNDS = 300


# ----------------------------------------------------------------------------- baseline A

def edt_or_far(mask, far):
    """Distance from every pixel to the nearest set pixel of mask, or far if the mask is empty."""
    if mask.any():
        return distance_transform_edt(mask == 0).astype(np.float32)
    return np.full(mask.shape, far, np.float32)


def pad_distance(x):
    """Distance to this net's pads per layer: float32 [N, 2, S, S].

    A layer without any pad of this net falls back to the pads on the other layer, since copper
    can still reach that layer through a via. A net with no pads at all gets a large constant.
    """
    n, _, s, _ = x.shape
    d = np.empty((n, 2, s, s), np.float32)
    for i in range(n):
        for l in range(2):
            mask = x[i, 1 + l]
            if not mask.any():
                mask = x[i, 1] | x[i, 2]
            d[i, l] = edt_or_far(mask, 2.0 * s)
    return d


def heuristic_predict(x, tau=DEFAULT_TAU):
    """Copper probability exp(-d / tau) inside the outline, float32 [N, 2, S, S]."""
    p = np.exp(-pad_distance(x) / np.float32(tau)).astype(np.float32)
    return p * x[:, :1].astype(np.float32)


def choose_tau(store, items):
    """Grid search tau on the training samples; p >= 0.5 is the same as d <= tau ln 2."""
    counts = {tau: [np.zeros(OUT_CHANNELS, np.int64) for _ in range(3)] for tau in TAU_GRID}
    for bid, idxs in group_by_board(items).items():
        x, y = store.load(bid)
        x, y = x[idxs], y[idxs]
        d, outline, target = pad_distance(x), x[:, :1] > 0, y > 0
        for tau in TAU_GRID:
            pred = (d <= tau * LN2) & outline
            for acc, c in zip(counts[tau], count_pixels(pred, target)):
                acc += c
    table = {str(tau): metrics_from_counts(*counts[tau])["overall"]["iou"] for tau in TAU_GRID}
    best = max(TAU_GRID, key=lambda tau: table[str(tau)])
    return best, table


def heuristic_eval(store, items, tau):
    tp = fp = fn = np.zeros(OUT_CHANNELS, np.int64)
    n_pixels = 0
    for bid, idxs in group_by_board(items).items():
        x, y = store.load(bid)
        x, y = x[idxs], y[idxs]
        a, b, c = count_pixels(heuristic_predict(x, tau) >= 0.5, y > 0)
        tp, fp, fn = tp + a, fp + b, fn + c
        n_pixels += x.shape[0] * x.shape[2] * x.shape[3]
    out = metrics_from_counts(tp, fp, fn, n_pixels)
    out["n_samples"] = len(items)
    return out


# ----------------------------------------------------------------------------- baseline B

def pixel_features(x1):
    """Feature matrix for one sample x1 uint8 [6, S, S]: float32 [S*S, 9] in FEATURES order."""
    s = x1.shape[-1]
    far = 2.0 * s
    feats = [edt_or_far(x1[c], far) for c in (1, 2, 3, 4, 5)]
    feats.append(x1[0].astype(np.float32))
    rows, cols = np.mgrid[0:s, 0:s]
    feats.append((cols / max(s - 1, 1)).astype(np.float32))
    feats.append((rows / max(s - 1, 1)).astype(np.float32))
    pads = (x1[1] | x1[2]) > 0
    if pads.any():
        r, c = np.nonzero(pads)
        dr = np.maximum(r.min() - rows, 0) + np.maximum(rows - r.max(), 0)
        dc = np.maximum(c.min() - cols, 0) + np.maximum(cols - c.max(), 0)
        feats.append((dr + dc).astype(np.float32))  # zero inside the bounding box
    else:
        feats.append(np.full((s, s), far, np.float32))
    return np.stack(feats, axis=-1).reshape(s * s, len(FEATURES))


def sample_pixels(store, items, max_pixels, pos_frac, rng, max_samples=0):
    """Draw training pixels per layer: at most max_pixels, at most pos_frac of them positive.

    Each sample contributes a small quota so the draw spans many boards; the pool is then
    trimmed to the global limits. Returns {layer: (X, y)}.
    """
    items = list(items)
    rng.shuffle(items)
    if max_samples:
        items = items[:max_samples]
    quota = max(8, math.ceil(2 * max_pixels / len(items)))
    pos_quota = max(1, math.ceil(quota * pos_frac))
    pool = {l: ([], []) for l in range(OUT_CHANNELS)}
    for bid, idxs in group_by_board(items).items():
        x, y = store.load(bid)
        for idx in idxs:
            feats = pixel_features(x[idx])
            for l in range(OUT_CHANNELS):
                label = y[idx, l].reshape(-1) > 0
                pos, neg = np.flatnonzero(label), np.flatnonzero(~label)
                take_pos = rng.choice(pos, min(len(pos), pos_quota), replace=False)
                take_neg = rng.choice(neg, min(len(neg), quota - len(take_pos)), replace=False)
                take = np.concatenate([take_pos, take_neg])
                pool[l][0].append(feats[take])
                pool[l][1].append(label[take])
    out = {}
    for l in range(OUT_CHANNELS):
        X, yv = np.concatenate(pool[l][0]), np.concatenate(pool[l][1])
        pos, neg = np.flatnonzero(yv), np.flatnonzero(~yv)
        keep_pos = rng.choice(pos, min(len(pos), int(pos_frac * max_pixels)), replace=False)
        keep_neg = rng.choice(neg, min(len(neg), max_pixels - len(keep_pos)), replace=False)
        keep = rng.permutation(np.concatenate([keep_pos, keep_neg]))
        out[l] = (X[keep], yv[keep].astype(np.uint8))
    return out


def train_lightgbm(sampled, seed):
    boosters = {}
    for l, (X, yv) in sampled.items():
        params = dict(LGB_PARAMS, seed=seed)
        boosters[l] = lgb.train(params, lgb.Dataset(X, yv, feature_name=list(FEATURES)), LGB_ROUNDS)
    return boosters


def lightgbm_probabilities(boosters, store, items):
    """Yield (probabilities float32 [2, S, S], target bool [2, S, S]) for every sample."""
    for bid, idxs in group_by_board(items).items():
        x, y = store.load(bid)
        s = x.shape[-1]
        for idx in idxs:
            feats = pixel_features(x[idx])
            p = np.stack([boosters[l].predict(feats).reshape(s, s) for l in range(OUT_CHANNELS)])
            yield p.astype(np.float32), y[idx] > 0


def tune_thresholds(boosters, store, items, rng, max_samples=64):
    """Pick the threshold per layer that maximises IoU on whole training samples.

    The sampled training pixels are balanced, so 0.5 is not calibrated to the real copper
    rate; the tuning set is full samples, every pixel included, so it reflects that rate.
    """
    items = list(items)
    rng.shuffle(items)
    items = items[:max_samples]
    tp = np.zeros((len(THRESHOLD_GRID), OUT_CHANNELS), np.int64)
    fp, fn = tp.copy(), tp.copy()
    for p, target in lightgbm_probabilities(boosters, store, items):
        for i, t in enumerate(THRESHOLD_GRID):
            a, b, c = count_pixels((p >= t)[None], target[None])
            tp[i] += a
            fp[i] += b
            fn[i] += c
    iou = tp / np.maximum(tp + fp + fn, 1)
    best = iou.argmax(axis=0)
    return {LAYER_NAMES[l]: float(THRESHOLD_GRID[best[l]]) for l in range(OUT_CHANNELS)}


def lightgbm_eval(boosters, store, items, thresholds):
    """Metrics at threshold 0.5 and at the tuned per layer thresholds, in one pass."""
    fixed = np.full(OUT_CHANNELS, 0.5, np.float32)
    tuned = np.array([thresholds[n] for n in LAYER_NAMES], np.float32)
    acc = {"at_0.5": [np.zeros(OUT_CHANNELS, np.int64) for _ in range(3)],
           "tuned": [np.zeros(OUT_CHANNELS, np.int64) for _ in range(3)]}
    n_pixels = 0
    for p, target in lightgbm_probabilities(boosters, store, items):
        n_pixels += p.shape[-1] * p.shape[-2]
        for key, th in (("at_0.5", fixed), ("tuned", tuned)):
            pred = p >= th[:, None, None]
            for a, c in zip(acc[key], count_pixels(pred[None], target[None])):
                a += c
    out = {key: metrics_from_counts(*acc[key], n_pixels) for key in acc}
    out["thresholds"] = thresholds
    out["n_samples"] = len(items)
    return out


# ------------------------------------------------------------------------------------ main

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=("owner", "random"), default="owner")
    p.add_argument("--limit-boards", type=int, default=0, help="0 keeps all boards")
    p.add_argument("--size", type=int, default=None, help="working raster size, default 256, 64 with --fixture")
    p.add_argument("--out", default=None, help="default data/models/baselines_<split>")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--max-pixels", type=int, default=200_000, help="LightGBM training pixels per layer")
    p.add_argument("--pos-frac", type=float, default=0.3, help="largest share of positives in the sample")
    p.add_argument("--max-train-samples", type=int, default=0,
                   help="cap the training samples LightGBM draws pixels from (0 uses all)")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA))
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--eval-held-out", action="store_true", help="also score the held out boards")
    p.add_argument("--fixture", action="store_true", help="run on a small synthetic fixture")
    a = p.parse_args()
    if a.fixture:
        a.data_dir, a.manifest = (str(q) for q in make_fixture(FIXTURE_DIR))
    a.size = a.size if a.size is not None else (64 if a.fixture else 256)
    if a.out is None:
        a.out = str((FIXTURE_DIR / "models" if a.fixture else ROOT / "data/models") / f"baselines_{a.split}")
    return a


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    boards = load_manifest(args.manifest)
    train_items, val_items, held_out, desc = split_samples(
        boards, args.split, args.val_frac, args.seed, args.limit_boards)
    used = {bid for bid, _ in train_items + val_items}
    store = RasterStore(args.data_dir, [b for b in boards if b["board_id"] in used] + held_out, args.size)
    print(f"split {args.split}: {desc['n_train_samples']} train samples from {desc['n_train_boards']} "
          f"boards ({desc['n_train_groups']} repos), {desc['n_val_samples']} val samples from "
          f"{desc['n_val_boards']} boards ({desc['n_val_groups']} repos), "
          f"{desc['n_held_out_boards']} held out boards; size {args.size}")

    t0 = time.time()
    tau, tau_table = choose_tau(store, train_items)
    heuristic_val = heuristic_eval(store, val_items, tau)
    print(f"heuristic: train IoU by tau {tau_table}, chose tau {tau}")
    print(f"heuristic val: {fmt(heuristic_val)}  ({time.time() - t0:.0f}s)")

    t0 = time.time()
    sampled = sample_pixels(store, train_items, args.max_pixels, args.pos_frac, rng, args.max_train_samples)
    n_train_pixels = {LAYER_NAMES[l]: {"n": len(yv), "positives": int(yv.sum())}
                      for l, (_, yv) in sampled.items()}
    print(f"lightgbm: sampled pixels {n_train_pixels}  ({time.time() - t0:.0f}s)")
    t0 = time.time()
    boosters = train_lightgbm(sampled, args.seed)
    thresholds = tune_thresholds(boosters, store, train_items, rng)
    lgb_val = lightgbm_eval(boosters, store, val_items, thresholds)
    print(f"lightgbm trained ({time.time() - t0:.0f}s); thresholds tuned on train {thresholds}")
    print(f"lightgbm val at 0.5:   {fmt(lgb_val['at_0.5'])}")
    print(f"lightgbm val tuned:    {fmt(lgb_val['tuned'])}")
    importance = {LAYER_NAMES[l]: dict(zip(FEATURES, map(int, b.feature_importance("gain"))))
                  for l, b in boosters.items()}

    metrics = {
        "args": vars(args), "split": desc,
        "heuristic": {"tau_grid_train_iou": tau_table, "tau": tau, "val": heuristic_val},
        "lightgbm": {"params": dict(LGB_PARAMS, seed=args.seed, rounds=LGB_ROUNDS),
                     "features": list(FEATURES), "n_train_pixels": n_train_pixels,
                     "feature_importance_gain": importance, "val": lgb_val},
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)
    for l, b in boosters.items():
        b.save_model(str(out / f"lightgbm_{LAYER_NAMES[l]}.txt"))
    print(f"wrote {out / 'metrics.json'}")

    if args.eval_held_out:
        if not held_out:
            print("no held out boards in the manifest")
            return
        held_items = board_samples(held_out)
        held = {"n_boards": len(held_out), "n_samples": len(held_items),
                "boards": [b["board_id"] for b in held_out],
                "heuristic": heuristic_eval(store, held_items, tau),
                "lightgbm": lightgbm_eval(boosters, store, held_items, thresholds)}
        print(f"held out heuristic:     {fmt(held['heuristic'])}")
        print(f"held out lightgbm 0.5:  {fmt(held['lightgbm']['at_0.5'])}")
        print(f"held out lightgbm tuned:{fmt(held['lightgbm']['tuned'])}")
        with open(out / "held_out_metrics.json", "w") as f:
            json.dump(held, f, indent=1)
        print(f"wrote {out / 'held_out_metrics.json'}")


if __name__ == "__main__":
    main()
