"""Train a small UNet that predicts one net's copper from its pads, the other nets and the outline.

    .venv/bin/python scripts/train_unet.py --split owner --epochs 10
    .venv/bin/python scripts/train_unet.py --fixture                   # smoke test on synthetic boards
    .venv/bin/python scripts/train_unet.py --epochs 0 --eval-held-out  # score a saved model
    python scripts/train_unet.py --size 256 --amp --workers 2 --time-limit-minutes 660 --resume
                                                       # a cloud session, see docs/cloud-training.md

Reads  data/raster/<board_id>.npz with x uint8 [N, 6, S, S], y uint8 [N, 2, S, S], net_ids int32 [N]
       and mm_per_px float32, plus splits/raster_manifest.json.
Writes <out>/model.pt (best validation IoU), <out>/last.pt (resume state after every epoch),
       <out>/metrics.json (rewritten after every epoch), <out>/examples.png and, with
       --eval-held-out, <out>/held_out_metrics.json. --resume continues from last.pt, --amp uses
       float16 autocast on CUDA, --save-every-minutes adds checkpoints inside an epoch and
       --time-limit-minutes stops cleanly before a session limit (docs/cloud-training.md).

Input channels: 0 outline, 1 this net's pads on front, 2 this net's pads on back, 3 other nets'
pads on front, 4 other nets' pads on back, 5 other nets' copper on either layer.
Targets: this net's tracks and vias on front (channel 0) and on back (channel 1).
"""
import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/raster"
DEFAULT_MANIFEST = ROOT / "splits/raster_manifest.json"
FIXTURE_DIR = ROOT / "data/fixture"
CACHE_LIMIT_BYTES = 2 * 1024 ** 3
IN_CHANNELS, OUT_CHANNELS = 6, 2
WIDTHS = (32, 64, 128, 256)
LAYER_NAMES = ("front", "back")


# ----------------------------------------------------------------------------- manifest and splits

def load_manifest(path):
    """Return the manifest's board list sorted by board id, so every run sees the same order."""
    with open(path) as f:
        boards = json.load(f)["boards"]
    return sorted(boards, key=lambda b: b["board_id"])


def board_samples(boards):
    """Expand boards into (board_id, index) samples."""
    return [(b["board_id"], i) for b in boards for i in range(int(b["n_samples"]))]


def group_by_board(items):
    """Map board_id to the sorted sample indices requested from it."""
    out = {}
    for bid, idx in items:
        out.setdefault(bid, []).append(idx)
    return {bid: sorted(v) for bid, v in out.items()}


def split_samples(boards, split, val_frac=0.15, seed=42, limit_boards=0):
    """Return (train, val, held_out_boards, description) for the requested split.

    train and val are lists of (board_id, index). Held out boards never enter either list.
    owner:  boards are grouped by source_repo and whole groups go to train or to validation,
            so validation boards come from repositories the model never saw.
    random: samples from all non held out boards are pooled and split at random. This split is
            leaky on purpose. Forks and revisions of one board, and the other nets of the very
            same board, land on both sides, so its validation scores overstate what the model
            knows about boards it has not seen. The gap between the two splits is the point.
    limit_boards caps the pool and the held out list separately (0 keeps everything).
    """
    held_out = [b for b in boards if b["held_out"]]
    pool = [b for b in boards if not b["held_out"]]
    if limit_boards:
        pool, held_out = pool[:limit_boards], held_out[:limit_boards]
    if not pool:
        raise SystemExit("no boards left after removing the held out boards")

    if split == "owner":
        from sklearn.model_selection import GroupShuffleSplit

        groups = [b["source_repo"] for b in pool]
        if len(set(groups)) < 2:
            raise SystemExit("owner split needs boards from at least two source repositories")
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        tr_idx, va_idx = next(splitter.split(pool, groups=groups))
        train_boards = [pool[i] for i in tr_idx]
        val_boards = [pool[i] for i in va_idx]
        train, val = board_samples(train_boards), board_samples(val_boards)
    elif split == "random":
        pooled = board_samples(pool)
        random.Random(seed).shuffle(pooled)
        n_val = min(len(pooled) - 1, max(1, round(val_frac * len(pooled))))
        val, train = pooled[:n_val], pooled[n_val:]
        train_ids, val_ids = {s[0] for s in train}, {s[0] for s in val}
        train_boards = [b for b in pool if b["board_id"] in train_ids]
        val_boards = [b for b in pool if b["board_id"] in val_ids]
    else:
        raise SystemExit(f"unknown split {split!r}, expected owner or random")
    if not train or not val:
        raise SystemExit(f"split {split} left train={len(train)} val={len(val)} samples")

    repos = lambda bs: sorted({b["source_repo"] for b in bs})
    description = {
        "split": split,
        "val_frac": val_frac,
        "seed": seed,
        "n_train_boards": len(train_boards),
        "n_val_boards": len(val_boards),
        "n_boards_in_both": len({b["board_id"] for b in train_boards} & {b["board_id"] for b in val_boards}),
        "n_train_groups": len(repos(train_boards)),
        "n_val_groups": len(repos(val_boards)),
        "n_train_samples": len(train),
        "n_val_samples": len(val),
        "n_held_out_boards": len(held_out),
        "n_held_out_samples": sum(int(b["n_samples"]) for b in held_out),
        "val_groups": repos(val_boards),
        "val_boards": sorted(b["board_id"] for b in val_boards),
    }
    return train, val, held_out, description


# ------------------------------------------------------------------------------------------ data

def downsample(a, factor):
    """Max pool the last two axes of a uint8 array by an integer factor."""
    if factor == 1:
        return a
    *lead, h, w = a.shape
    return a.reshape(*lead, h // factor, factor, w // factor, factor).max(axis=(-3, -1))


class RasterStore:
    """Loads board npz files on demand, max pools them to the working size and caches them.

    Everything is kept in memory when the estimated total is under 2 GB; otherwise only the
    most recently used boards stay cached.
    """

    def __init__(self, data_dir, boards, size):
        self.data_dir = Path(data_dir)
        self.size = size
        self.cache = OrderedDict()
        with np.load(self.path(boards[0]["board_id"])) as f:
            native = int(f["x"].shape[-1])
        if size > native or native % size:
            raise SystemExit(f"size {size} must divide the native raster size {native}")
        self.factor = native // size
        estimate = sum(int(b["n_samples"]) for b in boards) * (IN_CHANNELS + OUT_CHANNELS) * size * size
        self.cache_all = estimate < CACHE_LIMIT_BYTES
        self.max_cached = len(boards) + 1 if self.cache_all else 4

    def path(self, board_id):
        return self.data_dir / f"{board_id}.npz"

    def load(self, board_id):
        """Return (x, y) for a whole board as uint8 arrays at the working size."""
        if board_id in self.cache:
            self.cache.move_to_end(board_id)
            return self.cache[board_id]
        with np.load(self.path(board_id)) as f:
            x, y = f["x"], f["y"]
        if x.ndim != 4 or x.shape[1] != IN_CHANNELS or y.shape[1] != OUT_CHANNELS:
            raise SystemExit(f"{board_id}: unexpected shapes x{x.shape} y{y.shape}")
        pair = (downsample(x, self.factor), downsample(y, self.factor))
        self.cache[board_id] = pair
        while len(self.cache) > self.max_cached:
            self.cache.popitem(last=False)
        return pair

    def get(self, board_id, idx):
        x, y = self.load(board_id)
        return x[idx], y[idx]


class RasterDataset(Dataset):
    """One sample per (board, net). Train time augmentation is rotations by 90 degrees and flips."""

    def __init__(self, store, items, augment):
        self.store, self.items, self.augment = store, list(items), augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        bid, idx = self.items[i]
        x, y = self.store.get(bid, idx)
        if self.augment:
            # The same transform goes to x and y. A left right flip mirrors the board as seen
            # from the front; both copper layers mirror together and front stays front, so the
            # channels are never swapped. Rotations plus one flip cover all eight symmetries.
            k = random.randrange(4)
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k, axes=(1, 2))
            if random.random() < 0.5:
                x, y = np.flip(x, axis=2), np.flip(y, axis=2)
        x = torch.from_numpy(np.ascontiguousarray(x)).float()
        y = torch.from_numpy(np.ascontiguousarray(y)).float()
        return x, y


# ----------------------------------------------------------------------------------------- model

class ConvBlock(nn.Module):
    """Two 3x3 convolutions, each followed by BatchNorm and ReLU."""

    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Bilinear upsample, reduce channels with a 3x3 convolution, concat the skip, convolve."""

    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.block = ConvBlock(cout + cskip, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.block(torch.cat([self.reduce(x), skip], dim=1))


class UNet(nn.Module):
    """Four level UNet, widths 32 64 128 256, returns logits with one channel per copper layer."""

    def __init__(self, cin=IN_CHANNELS, cout=OUT_CHANNELS, widths=WIDTHS):
        super().__init__()
        w = widths
        self.enc1, self.enc2 = ConvBlock(cin, w[0]), ConvBlock(w[0], w[1])
        self.enc3, self.enc4 = ConvBlock(w[1], w[2]), ConvBlock(w[2], w[3])
        self.pool = nn.MaxPool2d(2)
        self.up3, self.up2, self.up1 = Up(w[3], w[2], w[2]), Up(w[2], w[1], w[1]), Up(w[1], w[0], w[0])
        self.head = nn.Conv2d(w[0], cout, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d = self.up1(self.up2(self.up3(e4, e3), e2), e1)
        return self.head(d)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def dice_loss(logits, target, smooth=1.0):
    """Soft Dice per channel over the batch, averaged over channels."""
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(0, 2, 3))
    denom = p.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    return (1 - (2 * inter + smooth) / (denom + smooth)).mean()


def loss_fn(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


# --------------------------------------------------------------------------------------- metrics

def count_pixels(pred, target):
    """Return (tp, fp, fn) per channel for boolean numpy arrays shaped [N, C, H, W]."""
    tp = (pred & target).sum(axis=(0, 2, 3))
    fp = (pred & ~target).sum(axis=(0, 2, 3))
    fn = (~pred & target).sum(axis=(0, 2, 3))
    return tp.astype(np.int64), fp.astype(np.int64), fn.astype(np.int64)


def metrics_from_counts(tp, fp, fn, n_pixels=None):
    """IoU, F1, precision and recall on copper pixels, per channel and overall, from counts."""

    def one(tp, fp, fn, n):
        out = {
            "iou": float(tp / max(tp + fp + fn, 1)),
            "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1)),
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
        }
        if n:
            out["copper_fraction"] = float((tp + fn) / n)
        return out

    tp, fp, fn = np.asarray(tp), np.asarray(fp), np.asarray(fn)
    out = {name: one(tp[i], fp[i], fn[i], n_pixels) for i, name in enumerate(LAYER_NAMES)}
    out["overall"] = one(tp.sum(), fp.sum(), fn.sum(), n_pixels * len(LAYER_NAMES) if n_pixels else None)
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    """Loss and pixel metrics at threshold 0.5 over every sample the loader yields."""
    model.eval()
    tp = fp = fn = np.zeros(OUT_CHANNELS, np.int64)
    loss_sum, n, n_pixels = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += loss_fn(logits, y).item() * x.shape[0]
        n += x.shape[0]
        n_pixels += x.shape[0] * x.shape[2] * x.shape[3]
        pred, t = logits >= 0, y > 0.5
        tp = tp + (pred & t).sum(dim=(0, 2, 3)).cpu().numpy()
        fp = fp + (pred & ~t).sum(dim=(0, 2, 3)).cpu().numpy()
        fn = fn + (~pred & t).sum(dim=(0, 2, 3)).cpu().numpy()
    out = metrics_from_counts(tp, fp, fn, n_pixels)
    out["loss"] = loss_sum / max(n, 1)
    out["n_samples"] = n
    return out


def heuristic_metrics(store, items):
    """Score baselines.heuristic_predict on the same samples, or return None if it is absent.

    This uses the module's default tau; baselines.py tunes tau on the training set itself.
    """
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import baselines
        heuristic_predict = baselines.heuristic_predict
    except (ImportError, AttributeError):
        return None
    tp = fp = fn = np.zeros(OUT_CHANNELS, np.int64)
    n_pixels = 0
    for bid, idxs in group_by_board(items).items():
        x, y = store.load(bid)
        x, y = x[idxs], y[idxs]
        pred = heuristic_predict(x) >= 0.5
        a, b, c = count_pixels(pred, y > 0)
        tp, fp, fn = tp + a, fp + b, fn + c
        n_pixels += x.shape[0] * x.shape[2] * x.shape[3]
    out = metrics_from_counts(tp, fp, fn, n_pixels)
    out["n_samples"] = len(items)
    out["tau"] = getattr(baselines, "DEFAULT_TAU", None)
    return out


def fmt(m):
    """One line summary of a metrics dict."""
    o = m["overall"]
    return (f"IoU {o['iou']:.3f} F1 {o['f1']:.3f} P {o['precision']:.3f} R {o['recall']:.3f}"
            f" (front IoU {m['front']['iou']:.3f}, back IoU {m['back']['iou']:.3f})")


# -------------------------------------------------------------------------------------- training

def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(store, items, batch, shuffle, augment, workers):
    ds = RasterDataset(store, items, augment)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      drop_last=shuffle and len(ds) > batch)


def save_checkpoint(model, path):
    torch.save({"state_dict": model.state_dict(), "in_channels": IN_CHANNELS,
                "out_channels": OUT_CHANNELS, "widths": list(WIDTHS)}, path)


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device)
    model = UNet(ck["in_channels"], ck["out_channels"], tuple(ck["widths"])).to(device)
    model.load_state_dict(ck["state_dict"])
    return model


# ------------------------------------------------------------------------- resume and bookkeeping

def atomic_save(obj, path):
    """torch.save through a temporary file, so a kill during the write never leaves a torn file."""
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(obj, path):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def rng_state():
    """Every random generator the training loop draws from, for an exact resume."""
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state):
    """Inverse of rng_state. The tensors go back to the CPU first: torch.load with a map_location
    moves them onto the device, and the setters accept CPU ByteTensors only."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def run_signature(args, n_train, n_val, steps_per_epoch):
    """The settings a resumed run must share with the run that wrote last.pt (epochs may differ)."""
    return {"split": args.split, "size": args.size, "seed": args.seed, "batch": args.batch,
            "lr": args.lr, "val_frac": args.val_frac, "limit_boards": args.limit_boards,
            "n_train_samples": n_train, "n_val_samples": n_val, "steps_per_epoch": steps_per_epoch}


def save_resume_state(path, model, opt, sched, scaler, epoch, batches_done, loss_sum, n,
                      best_iou, best_epoch, history, run, rng):
    """Write last.pt: everything needed to continue training as if it had not stopped.

    epoch counts finished epochs. batches_done > 0 means epoch + 1 was interrupted after that many
    loader batches; rng is then the generator state from the start of that epoch, so the resumed run
    can replay its data order and skip the batches already trained.
    """
    atomic_save({
        "model": {"state_dict": model.state_dict(), "in_channels": IN_CHANNELS,
                  "out_channels": OUT_CHANNELS, "widths": list(WIDTHS)},
        "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch, "batches_done": batches_done, "loss_sum": loss_sum, "n": n,
        "best_iou": best_iou, "best_epoch": best_epoch, "history": history, "run": run, "rng": rng,
        "saved_at": time.time(),
    }, path)


def load_resume_state(path, device, run):
    """Read last.pt or return None if it is missing; refuse one written with other settings."""
    path = Path(path)
    if not path.exists():
        return None
    ck = torch.load(path, map_location=device, weights_only=False)
    differences = {k: (ck["run"].get(k), v) for k, v in run.items() if ck["run"].get(k) != v}
    if differences:
        raise SystemExit(f"{path} was written by a run with different settings (saved, requested): "
                         f"{differences}; change --out or remove last.pt to start fresh")
    return ck


def place_on_schedule(opt, sched, total_steps):
    """Keep the scheduler's step count but move the learning rate onto a cosine of total_steps.

    Used when a run is resumed with a different --epochs: the optimizer continues from the same
    point in the new schedule, and later steps follow that curve exactly.
    """
    if sched.T_max == total_steps:
        return
    sched.T_max = total_steps
    for g, base in zip(opt.param_groups, sched.base_lrs):
        g["lr"] = sched.eta_min + (base - sched.eta_min) * (
            1 + math.cos(math.pi * sched.last_epoch / total_steps)) / 2


def write_metrics(out, meta, history, best_epoch, status, final_val=None, heuristic=None):
    """metrics.json: the header in meta, then the epoch history and the final scores when known."""
    metrics = {k: v for k, v in meta.items() if k != "status"}
    metrics.update({"epochs": history, "best_epoch": best_epoch, "final_val": final_val,
                    "heuristic_baseline_val": heuristic, "status": status})
    write_json(metrics, Path(out) / "metrics.json")


def train(model, store, train_items, val_items, args, device, out, meta=None, started=None):
    """Train with AdamW and a cosine schedule; keep the weights with the best validation IoU.

    Every epoch ends with <out>/last.pt (model, optimizer, scheduler, scaler, epoch, best metric,
    RNG state, history) and a rewritten metrics.json. Optional args: --resume continues from
    last.pt, --amp uses float16 autocast with a GradScaler on CUDA, --save-every-minutes writes
    last.pt inside long epochs, --time-limit-minutes stops after the last epoch that fits.
    meta is the metrics.json header; its "status" key is set to complete or stopped_at_time_limit.
    started is the run's start time for the time limit. Returns (history, best_epoch).
    """
    out = Path(out)
    meta = {} if meta is None else meta
    started = time.time() if started is None else started
    use_amp = bool(getattr(args, "amp", False)) and device.type == "cuda"
    save_every = float(getattr(args, "save_every_minutes", 0) or 0) * 60
    time_limit = float(getattr(args, "time_limit_minutes", 0) or 0) * 60

    train_loader = make_loader(store, train_items, args.batch, True, True, args.workers)
    val_loader = make_loader(store, val_items, args.batch, False, False, args.workers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = max(1, args.epochs * len(train_loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    run = run_signature(args, len(train_items), len(val_items), len(train_loader))

    def autocast():
        return torch.autocast("cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext()

    history, best_iou, best_epoch = [], -1.0, 0
    start_epoch, skip, loss_sum, n = 1, 0, 0.0, 0
    ck = load_resume_state(out / "last.pt", device, run) if getattr(args, "resume", False) else None
    if ck is None and getattr(args, "resume", False):
        print(f"no {out / 'last.pt'} to resume from; starting fresh")
    if ck is not None:
        model.load_state_dict(ck["model"]["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        place_on_schedule(opt, sched, total_steps)
        if use_amp and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        history, best_iou, best_epoch = ck["history"], ck["best_iou"], ck["best_epoch"]
        start_epoch, skip = ck["epoch"] + 1, ck["batches_done"]
        loss_sum, n = ck["loss_sum"], ck["n"]
        set_rng_state(ck["rng"])
        print(f"resumed from {out / 'last.pt'}: {ck['epoch']} epochs done"
              + (f", {skip} batches into epoch {start_epoch}" if skip else "")
              + f", best val IoU {best_iou:.3f} at epoch {best_epoch}, "
              f"lr {opt.param_groups[0]['lr']:.3e}, scheduler step {sched.last_epoch} "
              f"of {total_steps}")
        if start_epoch > args.epochs:
            print(f"all {args.epochs} epochs are done already")

    def save_last(epoch_done, batches_done, rng):
        save_resume_state(out / "last.pt", model, opt, sched, scaler, epoch_done, batches_done,
                          loss_sum, n, best_iou, best_epoch, history, run, rng)

    meta["status"] = "complete"
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0, last_save = time.time(), time.time()
        if not skip:
            loss_sum, n = 0.0, 0
        epoch_rng = rng_state()  # state at the start of this epoch, for mid epoch checkpoints
        batches_done = 0
        for x, y in train_loader:
            batches_done += 1
            if skip:
                skip -= 1  # replay the data order of the interrupted epoch without training
                continue
            if x.shape[0] < 2:
                continue  # BatchNorm cannot normalise a batch of one
            x, y = x.to(device), y.to(device)
            with autocast():
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * x.shape[0]
            n += x.shape[0]
            now = time.time()
            if time_limit and now - started > time_limit:
                save_last(epoch - 1, batches_done, epoch_rng)
                print(f"time limit of {time_limit / 60:g} minutes reached {batches_done} batches "
                      f"into epoch {epoch}; saved {out / 'last.pt'}, continue with --resume",
                      flush=True)
                meta["status"] = "stopped_at_time_limit"
                return history, best_epoch
            if save_every and now - last_save > save_every:
                save_last(epoch - 1, batches_done, epoch_rng)
                last_save = now
                print(f"  checkpoint after {batches_done} batches of epoch {epoch} "
                      f"({(now - started) / 60:.1f} minutes into the run)", flush=True)
        val = evaluate(model, val_loader, device)
        rec = {"epoch": epoch, "train_loss": loss_sum / max(n, 1), "val": val,
               "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"epoch {epoch:3d}  train loss {rec['train_loss']:.4f}  val loss {val['loss']:.4f}  "
              f"val {fmt(val)}  {rec['seconds']:.0f}s", flush=True)
        if val["overall"]["iou"] > best_iou:
            best_iou, best_epoch = val["overall"]["iou"], epoch
            save_checkpoint(model, out / "model.pt")
        save_last(epoch, 0, rng_state())
        write_metrics(out, meta, history, best_epoch, "running")
        if time_limit and epoch < args.epochs:
            elapsed, longest = time.time() - started, max(r["seconds"] for r in history)
            if elapsed + longest > time_limit:
                print(f"time limit: {elapsed / 60:.1f} of {time_limit / 60:g} minutes used and an "
                      f"epoch takes up to {longest / 60:.1f}; stopping after epoch {epoch}, "
                      f"continue with --resume", flush=True)
                meta["status"] = "stopped_at_time_limit"
                break
    return history, best_epoch


# -------------------------------------------------------------------------------------- pictures

def render_input(x):
    """RGB view of the six input channels: this net's pads red (front) and blue (back)."""
    s = x.shape[-1]
    img = np.full((s, s, 3), 0.35, np.float32)
    img[x[0] > 0] = 0.95
    img[x[5] > 0] = 0.72
    img[(x[3] | x[4]) > 0] = 0.5
    img[x[1] > 0] = (0.9, 0.1, 0.1)
    img[x[2] > 0] = (0.1, 0.25, 0.9)
    img[(x[1] & x[2]) > 0] = (0.6, 0.1, 0.6)
    return img


def render_layers(p):
    """RGB view of two copper probability maps: front red, back blue, both dark."""
    f, b = p[0].astype(np.float32), p[1].astype(np.float32)
    return np.stack([1 - b, 1 - np.maximum(f, b), 1 - f], axis=-1)


@torch.no_grad()
def save_examples(model, store, items, path, device, n=8):
    """Grid of validation examples: input, target and prediction per row."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = items[:: max(1, len(items) // n)][:n]
    model.eval()
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)), squeeze=False)
    for r, (bid, idx) in enumerate(rows):
        x, y = store.get(bid, idx)
        logits = model(torch.from_numpy(np.ascontiguousarray(x)[None]).float().to(device))
        p = torch.sigmoid(logits)[0].cpu().numpy()
        for c, img in enumerate((render_input(x), render_layers(y), render_layers(p))):
            axes[r, c].imshow(img, interpolation="nearest")
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(f"{bid[:20]} net {idx}", fontsize=8)
    for c, title in enumerate(("input (pads of this net in colour)", "target copper", "predicted copper")):
        axes[0, c].set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# --------------------------------------------------------------------------------------- fixture

def synthetic_board(rng, s, n_nets):
    """One synthetic board: pads joined by L shaped tracks, with vias where the layer changes."""
    m = rng.integers(4, 12, size=4)
    outline = np.zeros((s, s), np.uint8)
    outline[m[0]:s - m[1], m[2]:s - m[3]] = 1
    lo, hi = int(m.max()) + 4, s - int(m.max()) - 4
    pads = np.zeros((n_nets, 2, s, s), np.uint8)
    copper = np.zeros((n_nets, 2, s, s), np.uint8)
    for k in range(n_nets):
        n_pads = int(rng.integers(2, 5))
        pts = rng.integers(lo, hi, size=(n_pads, 2))
        kinds = rng.choice(3, size=n_pads, p=[0.45, 0.3, 0.25])  # 0 front, 1 back, 2 through hole
        layers = [{0: (0,), 1: (1,), 2: (0, 1)}[int(t)] for t in kinds]
        for (r, c), ls in zip(pts, layers):
            h = int(rng.integers(1, 3))
            for l in ls:
                pads[k, l, r - h:r + h + 1, c - h:c + h + 1] = 1
        for a in range(n_pads - 1):
            (ra, ca), (rb, cb) = pts[a], pts[a + 1]
            shared = set(layers[a]) & set(layers[a + 1])
            la = lb = min(shared) if shared else None
            if la is None:
                la, lb = layers[a][0], layers[a + 1][0]
            copper[k, la, ra:ra + 2, min(ca, cb):max(ca, cb) + 1] = 1  # horizontal leg
            copper[k, lb, min(ra, rb):max(ra, rb) + 1, cb:cb + 2] = 1  # vertical leg
            if la != lb:
                copper[k, :, ra - 1:ra + 2, cb - 1:cb + 2] = 1  # via at the corner
    pad_total = pads.sum(axis=0, dtype=np.int16)
    copper_total = copper.sum(axis=0, dtype=np.int16)
    x = np.zeros((n_nets, IN_CHANNELS, s, s), np.uint8)
    x[:, 0] = outline
    x[:, 1:3] = pads
    for k in range(n_nets):
        x[k, 3:5] = (pad_total - pads[k]) > 0
        x[k, 5] = ((copper_total - copper[k]) > 0).any(axis=0)
    return x * outline, copper * outline


def make_fixture(dir, n_boards=6, size=128, nets_per_board=12, seed=0):
    """Write synthetic boards and a manifest under dir. Returns (raster_dir, manifest_path).

    Boards pair up into pretend repositories; the last board sits alone and is held out.
    """
    dir = Path(dir)
    raster = dir / "raster"
    raster.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    boards = []
    for i in range(n_boards):
        x, y = synthetic_board(rng, size, nets_per_board)
        bid = f"fixture_{i:02d}"
        np.savez_compressed(raster / f"{bid}.npz", x=x, y=y,
                            net_ids=np.arange(1, len(x) + 1, dtype=np.int32), mm_per_px=np.float32(0.25))
        last = i == n_boards - 1
        boards.append({"board_id": bid, "source_repo": "repo_held" if last else f"repo_{i // 2}",
                       "held_out": last, "n_samples": len(x)})
    manifest = dir / "raster_manifest.json"
    with open(manifest, "w") as f:
        json.dump({"boards": boards}, f, indent=1)
    return raster, manifest


# ------------------------------------------------------------------------------------------ main

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=("owner", "random"), default="owner")
    p.add_argument("--epochs", type=int, default=None, help="default 10, or 2 with --fixture")
    p.add_argument("--size", type=int, default=None, help="working raster size, default 256, 64 with --fixture")
    p.add_argument("--limit-boards", type=int, default=0, help="0 keeps all boards")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="auto", help="auto picks cuda, then mps, then cpu")
    p.add_argument("--out", default=None, help="default data/models/unet_<split>")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=0,
                   help="DataLoader workers; 0 keeps the cache in process. Linux workers fork "
                        "and each keeps its own board cache (see docs/cloud-training.md)")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA))
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--eval-held-out", action="store_true", help="score the best model on held out boards")
    p.add_argument("--fixture", action="store_true", help="train on a small synthetic fixture")
    p.add_argument("--amp", action="store_true",
                   help="float16 autocast with a GradScaler on CUDA; ignored on other devices")
    p.add_argument("--resume", action="store_true",
                   help="continue from <out>/last.pt when it exists")
    p.add_argument("--save-every-minutes", type=float, default=0,
                   help="also write last.pt this often inside an epoch "
                        "(0 writes it at epoch ends only)")
    p.add_argument("--time-limit-minutes", type=float, default=0,
                   help="stop cleanly, with last.pt and metrics.json written, before this much "
                        "wall time has passed since the start (0 means no limit)")
    a = p.parse_args()
    if a.fixture:
        a.data_dir, a.manifest = (str(q) for q in make_fixture(FIXTURE_DIR))
    a.epochs = a.epochs if a.epochs is not None else (2 if a.fixture else 10)
    a.size = a.size if a.size is not None else (64 if a.fixture else 256)
    if a.out is None:
        a.out = str((FIXTURE_DIR / "models" if a.fixture else ROOT / "data/models") / f"unet_{a.split}")
    if a.size % 8:
        raise SystemExit("size must be a multiple of 8 (four resolution levels)")
    return a


def main():
    started = time.time()
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed input sizes, let cuDNN tune its kernels
    if args.amp and device.type != "cuda":
        print(f"--amp needs a CUDA device; training in full precision on {device}")
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
          f"{desc['n_held_out_boards']} held out boards; size {args.size} "
          f"(pool factor {store.factor}); cache all {store.cache_all}; device {device}")

    model = UNet().to(device)
    n_params = count_parameters(model)
    print(f"UNet parameters: {n_params:,}")

    meta = {"args": vars(args), "device": str(device), "model_parameters": n_params, "split": desc}
    history, best_epoch = [], 0
    if args.epochs > 0:
        history, best_epoch = train(model, store, train_items, val_items, args, device, out,
                                    meta, started)
    elif (out / "last.pt").exists():
        # Scoring only: keep the epoch history of the run that trained this model in metrics.json.
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        history, best_epoch = ck["history"], ck["best_epoch"]
    if not (out / "model.pt").exists():
        if meta.get("status") == "stopped_at_time_limit":
            raise SystemExit(f"stopped before the first epoch finished, so there is no "
                             f"{out / 'model.pt'} yet; continue with --resume")
        raise SystemExit(f"no model at {out / 'model.pt'}; train first")
    model = load_checkpoint(out / "model.pt", device)

    val_loader = make_loader(store, val_items, args.batch, False, False, args.workers)
    final = evaluate(model, val_loader, device)
    print(f"best epoch {best_epoch}: val {fmt(final)}")
    heuristic = heuristic_metrics(store, val_items)
    if heuristic:
        print(f"heuristic baseline on val (reference): {fmt(heuristic)}")
    save_examples(model, store, val_items, out / "examples.png", device)
    write_metrics(out, meta, history, best_epoch, meta.get("status", "complete"), final, heuristic)
    print(f"wrote {out / 'metrics.json'} and {out / 'examples.png'}")

    if args.eval_held_out:
        if not held_out:
            print("no held out boards in the manifest")
            return
        held_items = board_samples(held_out)
        held_loader = make_loader(store, held_items, args.batch, False, False, args.workers)
        held = evaluate(model, held_loader, device)
        print(f"held out ({len(held_out)} boards, {len(held_items)} samples): {fmt(held)}")
        held_heuristic = heuristic_metrics(store, held_items)
        if held_heuristic:
            print(f"heuristic baseline on held out (reference): {fmt(held_heuristic)}")
        with open(out / "held_out_metrics.json", "w") as f:
            json.dump({"split": args.split, "n_boards": len(held_out), "n_samples": len(held_items),
                       "boards": [b["board_id"] for b in held_out], "unet": held,
                       "heuristic_baseline": held_heuristic}, f, indent=1)
        print(f"wrote {out / 'held_out_metrics.json'}")


if __name__ == "__main__":
    main()
