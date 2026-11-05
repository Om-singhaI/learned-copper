"""Guided grid routing: does the UNet copper prediction help a classical maze router?

    .venv/bin/python scripts/route_grid.py --phase sweep                 # lambda sweep, 30 owner split validation boards
    .venv/bin/python scripts/route_grid.py --phase held-out              # both settings on the held out boards, chosen lambda
    .venv/bin/python scripts/route_grid.py --phase all --limit-boards 2  # quick end to end run
    .venv/bin/python scripts/route_grid.py --summarize-only              # rebuild the summary from the CSV files

Grid. The 256 px rasters are max pooled to 128 (train_unet.downsample), the resolution of the trained
model data/models/unet_owner_128/model.pt. One cell is mm_per_px * 2 mm wide.

Router. Two layer A star. Within a layer the grid is 8 connected; a diagonal move is refused when either
of the two orthogonal cells it passes between is blocked, so a track cannot squeeze between two touching
obstacles. A via moves between the layers at the same cell and occupies that cell on both layers.
Obstacles are cells outside the outline, pads of other nets on that layer, copper of the context (channel 5)
on both layers, and copper placed by earlier nets of the same run. This net's own pads are free cells and
its terminals. A terminal is one 4 connected component of the net's pad cells, with front and back joined
at the same cell (a through hole pad is one terminal). Multi terminal nets are built one pad at a time:
every cell of the connected tree is an A star source and the search stops at the first cell of any not yet
connected pad, so each new pad is joined to the nearest point of the tree. When the search exhausts the
grid the connection fails, the partial copper stays, and the routing continues from the next pad.

Costs. Unguided: 1 per step (sqrt 2 on diagonals), V per via (default 10). Guided with the prediction
p = sigmoid(logits) per layer: entering a cell costs 1 + lambda * (1 - p_layer(cell)) (times sqrt 2 on
diagonals) and a via costs V * (1 + lambda * (1 - max(p_front, p_back))). Lambda 0 is the unguided router.

Settings. (a) single net: the human copper of all other nets (channel 5 as stored) is fixed and each net is
routed alone. (b) full board: channel 5 starts empty and all sampled nets of the board are routed in a
fixed order; the router's own copper accumulates into the obstacles and into channel 5, and the prediction
for each net is recomputed from that state. Orders: the stored sample order, and shortest first by the
half perimeter of the net's pad bounding box.

Outputs under data/results: route_grid_sweep.json, route_grid_single_net.csv (one row per net and run),
route_grid_full_board.csv (one row per board, order and guidance), route_grid_summary.json (also copied to
docs/results).
"""
import argparse
import csv
import heapq
import json
import math
import shutil
import sys
import time
import zlib
from itertools import pairwise
import pathlib
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _import_train_unet(retries=5, wait=60):
    """Import train_unet, waiting a little if another process is mid way through rewriting it."""
    for attempt in range(retries):
        try:
            import train_unet
            return train_unet
        except (ImportError, SyntaxError) as e:
            if attempt == retries - 1:
                raise
            print(f"import train_unet failed ({e}); retrying in {wait}s", flush=True)
            time.sleep(wait)


tu = _import_train_unet()

GRID = 128
SQ2 = math.sqrt(2.0)
INF = float("inf")
LAMBDAS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)
DEFAULT_MODEL = ROOT / "data/models/unet_owner_128/model.pt"


def rel_path(path):
    """Path as stored in result files: relative to the repository when inside it."""
    path = pathlib.Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)
RESULTS = ROOT / "data/results"
DOC_RESULTS = ROOT / "docs/results"
FORWARD_CHUNK = 16

# Label structure on a [layer, row, col] stack: 8 connected within a layer, the two layers joined only
# at the same cell. Used for terminals (pads) and for the human copper connectivity checks.
STRUCT_8 = np.zeros((3, 3, 3), bool)
STRUCT_8[1] = True
STRUCT_8[0, 1, 1] = STRUCT_8[2, 1, 1] = True
STRUCT_4 = np.zeros((3, 3, 3), bool)
STRUCT_4[1, 1, :] = STRUCT_4[1, :, 1] = True
STRUCT_4[0, 1, 1] = STRUCT_4[2, 1, 1] = True

SINGLE_NET_FIELDS = [
    "phase", "board_id", "idx", "net_id", "split", "in_test_set", "tier", "run", "lam", "guided",
    "n_terminals", "n_pad_cells", "hpwl_cells", "completed", "failed_connections", "length_mm",
    "length_cells", "vias", "expansions", "seconds", "copper_crc", "human_length_mm", "human_length_cells",
    "human_via_cells", "human_via_blobs", "human_components", "human_free_components",
    "human_free_components_4", "human_connected", "mm_per_cell",
]
FULL_BOARD_FIELDS = [
    "phase", "board_id", "split", "in_test_set", "tier", "order", "run", "lam", "guided", "n_nets",
    "attempted", "completed", "connections", "failed_connections", "length_mm", "vias", "expansions",
    "seconds", "human_length_mm", "human_via_cells", "human_via_blobs", "human_connected_nets", "mm_per_cell",
]


# ------------------------------------------------------------------------------------------ grid

_NEIGH_CACHE = {}


def neighbour_table(H, W):
    """Per cell, the in layer moves as (target cell, length multiplier, orth1, orth2).

    orth1 and orth2 are the two orthogonal cells a diagonal move passes between, or -1 for a straight
    move. Cells are flat row major indices within one layer. Built once per grid size.
    """
    key = (H, W)
    if key not in _NEIGH_CACHE:
        table = []
        for r in range(H):
            for c in range(W):
                moves = []
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if (dr == 0 and dc == 0) or not (0 <= rr < H and 0 <= cc < W):
                            continue
                        if dr and dc:
                            moves.append((rr * W + cc, SQ2, r * W + cc, rr * W + c))
                        else:
                            moves.append((rr * W + cc, 1.0, -1, -1))
                table.append(tuple(moves))
        _NEIGH_CACHE[key] = table
    return _NEIGH_CACHE[key]


def astar(blocked, cost, via_cost, sources, goal, h, N, neigh):
    """Heap based A star from all sources to the first popped goal cell.

    blocked, goal: bytes or bytearray of length 2N (layer major), cost: list of 2N entry costs,
    via_cost: list of N via costs, h: list of N heuristic values (layer independent), neigh: the
    neighbour table. Returns (goal node or -1, parent list, number of expanded nodes).
    """
    g = [INF] * (2 * N)
    parent = [-1] * (2 * N)
    closed = bytearray(2 * N)
    heap = []
    for s in sources:
        g[s] = 0.0
        heap.append((h[s % N], 0.0, s))
    heapq.heapify(heap)
    push, pop = heapq.heappush, heapq.heappop
    expansions = 0
    while heap:
        _, ng_u, u = pop(heap)
        if closed[u]:
            continue
        closed[u] = 1
        expansions += 1
        if goal[u]:
            return u, parent, expansions
        gu = -ng_u
        if u >= N:
            base, rc = N, u - N
        else:
            base, rc = 0, u
        for v, mult, o1, o2 in neigh[rc]:
            vv = base + v
            if blocked[vv] or closed[vv]:
                continue
            if o1 >= 0 and (blocked[base + o1] or blocked[base + o2]):
                continue
            ng = gu + cost[vv] * mult
            if ng < g[vv]:
                g[vv] = ng
                parent[vv] = u
                push(heap, (ng + h[v], -ng, vv))
        vv = rc + (N - base)  # the same cell on the other layer
        if not blocked[vv] and not closed[vv]:
            ng = gu + via_cost[rc]
            if ng < g[vv]:
                g[vv] = ng
                parent[vv] = u
                push(heap, (ng + h[rc], -ng, vv))
    return -1, parent, expansions


def heuristic_grid(bboxes, H, W):
    """Octile distance from every cell to the nearest of the given (r0, r1, c0, c1) boxes, as a flat list."""
    rows = np.arange(H, dtype=np.float64)[:, None]
    cols = np.arange(W, dtype=np.float64)[None, :]
    h = np.full((H, W), INF)
    for r0, r1, c0, c1 in bboxes:
        dy = np.maximum(0.0, np.maximum(r0 - rows, rows - r1))
        dx = np.maximum(0.0, np.maximum(c0 - cols, cols - c1))
        h = np.minimum(h, np.maximum(dx, dy) + (SQ2 - 1.0) * np.minimum(dx, dy))
    return h.ravel().tolist()


def label_terminals(pads):
    """Connected components of the [2, H, W] pad mask; returns (labels, n, cells per label, bbox per label).

    Components are 4 connected within a layer, which matches the router's rule that two cells touching
    only at a corner between two blocked cells are not joined; pads are axis aligned rectangles, so a
    real pad is never split by this.
    """
    labels, n = ndimage.label(pads, structure=STRUCT_4)
    flat = labels.ravel()
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=n + 1)
    starts = np.cumsum(counts) - counts
    cells = {k: order[starts[k]:starts[k] + counts[k]].tolist() for k in range(1, n + 1)}
    bboxes = {}
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is not None:
            bboxes[k] = (sl[1].start, sl[1].stop - 1, sl[2].start, sl[2].stop - 1)
    return labels, n, cells, bboxes


def base_obstacles(x6, context, own=None):
    """Blocked cells per layer before this net's pads are freed.

    Outside the outline and the context copper block both layers; other nets' pads block their layer;
    own (the router's copper from earlier nets, [2, H, W] bool) blocks per layer.
    """
    outside = x6[0] == 0
    ctx = context > 0
    obs = np.stack([outside | (x6[3] > 0) | ctx, outside | (x6[4] > 0) | ctx])
    if own is not None:
        obs |= own
    return obs


def hpwl(x6):
    """Half perimeter of the bounding box of this net's pads, in cells."""
    pads = (x6[1] > 0) | (x6[2] > 0)
    rows, cols = np.flatnonzero(pads.any(axis=1)), np.flatnonzero(pads.any(axis=0))
    if len(rows) == 0:
        return 0
    return int(rows[-1] - rows[0] + cols[-1] - cols[0])


def route_net(x6, obstacles, p, lam, via_cost_base, neigh=None):
    """Route one net. Returns a dict with the copper placed and the counts.

    x6: [6, H, W] uint8 input (channels 1 and 2 are this net's pads). obstacles: [2, H, W] bool.
    p: [2, H, W] copper probabilities, or None for the unguided router. lam: lambda.
    """
    t0 = time.time()
    _, H, W = x6.shape
    N = H * W
    neigh = neigh or neighbour_table(H, W)
    pads = x6[1:3] > 0
    blocked_arr = obstacles & ~pads
    blocked = bytes(blocked_arr.astype(np.uint8).ravel())
    if p is None:
        cost = [1.0] * (2 * N)
        via_cost = [float(via_cost_base)] * N
    else:
        p = np.asarray(p, dtype=np.float64)
        cost = (1.0 + lam * (1.0 - p)).ravel().tolist()
        via_cost = (via_cost_base * (1.0 + lam * (1.0 - p.max(axis=0)))).ravel().tolist()

    labels, n_term, cells, bboxes = label_terminals(pads)
    copper = np.zeros((2, H, W), bool)
    out = {"n_terminals": int(n_term), "n_pad_cells": int(pads.sum()), "hpwl_cells": hpwl(x6),
           "length_cells": 0.0, "vias": 0, "failed_connections": 0, "expansions": 0}
    if n_term >= 2:
        label_flat = labels.ravel().tolist()
        goal = bytearray(2 * N)
        for k in range(2, n_term + 1):
            for u in cells[k]:
                goal[u] = 1
        remaining = list(range(2, n_term + 1))
        tree = list(cells[1])
        copper_flat = copper.ravel()
        while remaining:
            h = heuristic_grid([bboxes[k] for k in remaining], H, W)
            u, parent, exp = astar(blocked, cost, via_cost, tree, goal, h, N, neigh)
            out["expansions"] += exp
            if u < 0:
                out["failed_connections"] += 1
                k = remaining.pop(0)  # start a fresh tree from the next pad
                for c in cells[k]:
                    goal[c] = 0
                tree = list(cells[k])
                continue
            k = label_flat[u]
            remaining.remove(k)
            for c in cells[k]:
                goal[c] = 0
            path = [u]
            while parent[path[-1]] >= 0:
                path.append(parent[path[-1]])
            for a, b in pairwise(path):
                if abs(a - b) == N:
                    out["vias"] += 1
                else:
                    ra, ca = divmod(a % N, W)
                    rb, cb = divmod(b % N, W)
                    out["length_cells"] += SQ2 if (ra != rb and ca != cb) else 1.0
            copper_flat[path] = True
            tree.extend(path[:-1])
            tree.extend(cells[k])
    copper &= ~pads
    out["completed"] = n_term >= 1 and out["failed_connections"] == 0
    out["copper"] = copper
    out["copper_crc"] = zlib.crc32(np.packbits(copper).tobytes())
    out["seconds"] = time.time() - t0
    return out


# ------------------------------------------------------------------------------------- reference

def human_reference(y2, x6, obstacles):
    """Lengths and vias of the human net from y, and whether its copper connects the terminals at 128."""
    hy = y2 > 0
    pads = x6[1:3] > 0
    both = hy[0] & hy[1]
    free = obstacles & ~pads
    out = {
        "human_length_cells": int(hy.sum()),
        "human_via_cells": int(both.sum()),
        "human_via_blobs": int(ndimage.label(both, structure=np.ones((3, 3), bool))[1]),
        "human_components": int(ndimage.label(pads | hy, structure=STRUCT_8)[1]),
        "human_free_components": int(ndimage.label(pads | (hy & ~free), structure=STRUCT_8)[1]),
        "human_free_components_4": int(ndimage.label(pads | (hy & ~free), structure=STRUCT_4)[1]),
    }
    out["human_connected"] = out["human_components"] <= 1
    return out


# ------------------------------------------------------------------------------------------ model

def predict(model, device, x_batch):
    """Sigmoid probabilities [n, 2, H, W] float32 for uint8 inputs [n, 6, H, W]."""
    out = []
    with torch.no_grad():
        for i in range(0, len(x_batch), FORWARD_CHUNK):
            xb = torch.from_numpy(np.ascontiguousarray(x_batch[i:i + FORWARD_CHUNK])).float().to(device)
            out.append(torch.sigmoid(model(xb)).cpu().numpy())
    if not out:
        return np.zeros((0, 2) + tuple(x_batch.shape[-2:]), np.float32)
    return np.concatenate(out)


def load_board(data_dir, board_id, grid=GRID):
    """Return (x, y, net_ids, mm_per_cell) at the requested grid (max pooled from the stored size)."""
    with np.load(Path(data_dir) / f"{board_id}.npz") as f:
        x, y, net_ids, mm = f["x"], f["y"], f["net_ids"], float(f["mm_per_px"])
    factor = x.shape[-1] // grid
    if factor < 1 or x.shape[-1] % grid:
        raise SystemExit(f"{board_id}: raster size {x.shape[-1]} is not a multiple of {grid}")
    return tu.downsample(x, factor), tu.downsample(y, factor), net_ids, mm * factor


# --------------------------------------------------------------------------------------- settings

def run_single_net(board, x, y, net_ids, mm, model, device, lams, via_cost, phase, neigh):
    """Setting (a) on one board: each net alone against the human copper of the other nets."""
    rows = []
    p_all = predict(model, device, x) if lams else None
    for idx in range(len(x)):
        x6 = x[idx]
        obs = base_obstacles(x6, x6[5])
        human = human_reference(y[idx], x6, obs)
        human["human_length_mm"] = human["human_length_cells"] * mm
        runs = [("unguided", None, 0.0)] + [("guided", p_all[idx], lam) for lam in lams]
        for name, p, lam in runs:
            res = route_net(x6, obs, p, lam, via_cost, neigh)
            res.pop("copper")
            rows.append({"phase": phase, "board_id": board["board_id"], "idx": idx, "net_id": int(net_ids[idx]),
                         "split": board["_split"], "in_test_set": board.get("in_test_set", False),
                         "tier": board.get("tier", ""), "run": name, "lam": lam, "guided": name == "guided",
                         "length_mm": res["length_cells"] * mm, "mm_per_cell": mm, **res, **human})
    return rows


def run_full_board(board, x, y, net_ids, mm, model, device, lam, via_cost, order_name, guided, phase, neigh):
    """Setting (b) on one board: sequential routing from an empty context in the given order."""
    t0 = time.time()
    n, _, H, W = x.shape
    if order_name == "stored":
        order = list(range(n))
    elif order_name == "shortest":
        order = sorted(range(n), key=lambda i: (hpwl(x[i]), i))
    else:
        raise ValueError(order_name)
    own = np.zeros((2, H, W), bool)
    context = np.zeros((H, W), np.uint8)
    tot = {"attempted": 0, "completed": 0, "connections": 0, "failed_connections": 0, "length_mm": 0.0,
           "vias": 0, "expansions": 0, "human_length_mm": 0.0, "human_via_cells": 0, "human_via_blobs": 0,
           "human_connected_nets": 0}
    for idx in order:
        x6 = x[idx]
        obs = base_obstacles(x6, np.zeros((H, W), np.uint8), own)
        p = None
        if guided:
            x_in = x6.copy()
            x_in[5] = context
            p = predict(model, device, x_in[None])[0]
        res = route_net(x6, obs, p, lam, via_cost, neigh)
        own |= res["copper"]
        context |= res["copper"].any(axis=0).astype(np.uint8)
        human = human_reference(y[idx], x6, obs)
        if res["n_terminals"] >= 2:
            tot["attempted"] += 1
            tot["completed"] += int(res["completed"])
            tot["connections"] += res["n_terminals"] - 1
            tot["failed_connections"] += res["failed_connections"]
        tot["length_mm"] += res["length_cells"] * mm
        tot["vias"] += res["vias"]
        tot["expansions"] += res["expansions"]
        tot["human_length_mm"] += human["human_length_cells"] * mm
        tot["human_via_cells"] += human["human_via_cells"]
        tot["human_via_blobs"] += human["human_via_blobs"]
        tot["human_connected_nets"] += int(human["human_connected"])
    return {"phase": phase, "board_id": board["board_id"], "split": board["_split"],
            "in_test_set": board.get("in_test_set", False), "tier": board.get("tier", ""), "order": order_name,
            "run": "guided" if guided else "unguided", "lam": lam if guided else 0.0, "guided": guided,
            "n_nets": n, "seconds": time.time() - t0, "mm_per_cell": mm, **tot}


# ---------------------------------------------------------------------------------------- summary

def mean(v):
    v = list(v)
    return float(np.mean(v)) if v else None


def median(v):
    v = list(v)
    return float(np.median(v)) if v else None


def summarize_single(rows, reference_rows=None):
    """Completion, length ratio and vias for one run's rows (nets with at least two terminals).

    The length ratio is averaged over nets completed by this run and by the reference run (the unguided
    router), with a positive human length; without a reference, over the nets this run completed.
    """
    rows = [r for r in rows if r["n_terminals"] >= 2]
    ref_done = None
    if reference_rows is not None:
        ref_done = {(r["board_id"], r["idx"]) for r in reference_rows if r["completed"] and r["n_terminals"] >= 2}
    done = [r for r in rows if r["completed"]]
    both = [r for r in done if ref_done is None or (r["board_id"], r["idx"]) in ref_done]
    ratio_rows = [r for r in both if r["human_length_cells"] > 0]
    ratios = [r["length_cells"] / r["human_length_cells"] for r in ratio_rows]
    connected = [r for r in rows if r["human_connected"]]
    out = {
        "n_nets": len(rows),
        "n_boards": len({r["board_id"] for r in rows}),
        "completed": len(done),
        "completion_rate": len(done) / len(rows) if rows else None,
        "n_human_connected": len(connected),
        "completion_rate_human_connected": (sum(r["completed"] for r in connected) / len(connected)
                                            if connected else None),
        "n_both_completed": len(both),
        "length_ratio_mean": mean(ratios),
        "length_ratio_median": median(ratios),
        "length_ratio_of_sums": (sum(r["length_cells"] for r in ratio_rows)
                                 / sum(r["human_length_cells"] for r in ratio_rows)) if ratio_rows else None,
        "vias_mean_both_completed": mean(r["vias"] for r in both),
        "vias_mean_all": mean(r["vias"] for r in rows),
        "human_via_cells_mean_both_completed": mean(r["human_via_cells"] for r in both),
        "human_via_blobs_mean_both_completed": mean(r["human_via_blobs"] for r in both),
        "expansions_mean": mean(r["expansions"] for r in rows),
        "seconds_total": float(sum(r["seconds"] for r in rows)),
        "seconds_per_board": float(sum(r["seconds"] for r in rows)) / max(1, len({r["board_id"] for r in rows})),
    }
    return out


def diagnose_failures(rows, limit=12):
    """Why the unguided single net router failed: classes and concrete examples."""
    failed = [r for r in rows if r["n_terminals"] >= 2 and not r["completed"]]
    classes = {"human_not_connected_by_tracks": [], "human_path_cut_by_obstacles_at_128": [],
               "human_path_only_diagonal_squeeze": [], "unexplained": []}
    for r in failed:
        if r["human_components"] > 1:
            key = "human_not_connected_by_tracks"
        elif r["human_free_components"] > 1:
            key = "human_path_cut_by_obstacles_at_128"
        elif r["human_free_components_4"] > 1:
            key = "human_path_only_diagonal_squeeze"
        else:
            key = "unexplained"
        classes[key].append(r)
    single = [r for r in rows if r["n_terminals"] < 2]
    out = {"n_failed": len(failed), "n_single_terminal_excluded": len(single),
           "single_terminal_examples": [f"{r['board_id']} idx {r['idx']} net {r['net_id']}" for r in single[:6]],
           "classes": {}}
    for key, rs in classes.items():
        out["classes"][key] = {
            "count": len(rs),
            "examples": [f"{r['board_id']} idx {r['idx']} net {r['net_id']} terminals {r['n_terminals']} "
                         f"failed {r['failed_connections']} hpwl {r['hpwl_cells']}" for r in rs[:limit]],
        }
    return out


def summarize_full(rows):
    """Totals for one (order, run) group of full board rows."""
    if not rows:
        return None
    att, comp = sum(r["attempted"] for r in rows), sum(r["completed"] for r in rows)
    hl = sum(r["human_length_mm"] for r in rows)
    return {
        "n_boards": len(rows),
        "nets_attempted": att,
        "nets_completed": comp,
        "completion_rate": comp / att if att else None,
        "connections": sum(r["connections"] for r in rows),
        "failed_connections": sum(r["failed_connections"] for r in rows),
        "length_mm": float(sum(r["length_mm"] for r in rows)),
        "human_length_mm": float(hl),
        "length_ratio_of_sums": float(sum(r["length_mm"] for r in rows) / hl) if hl else None,
        "vias": int(sum(r["vias"] for r in rows)),
        "human_via_cells": int(sum(r["human_via_cells"] for r in rows)),
        "human_via_blobs": int(sum(r["human_via_blobs"] for r in rows)),
        "seconds_per_board": float(sum(r["seconds"] for r in rows)) / len(rows),
        "boards_all_nets_completed": int(sum(r["completed"] == r["attempted"] for r in rows)),
    }


def lambda0_check(rows):
    """Compare guided lambda 0 rows with unguided rows net by net."""
    unguided = {(r["board_id"], r["idx"]): r for r in rows if r["run"] == "unguided"}
    zero = [r for r in rows if r["run"] == "guided" and float(r["lam"]) == 0.0]
    keys = ("completed", "length_cells", "vias", "copper_crc", "expansions", "failed_connections")
    mismatches = []
    for r in zero:
        u = unguided.get((r["board_id"], r["idx"]))
        if u is None:
            continue
        bad = [k for k in keys if u[k] != r[k] and not (isinstance(u[k], float) and abs(u[k] - r[k]) < 1e-9)]
        if bad:
            mismatches.append({"board_id": r["board_id"], "idx": r["idx"], "fields": bad})
    return {"n_nets_compared": len(zero), "n_boards": len({r["board_id"] for r in zero}),
            "n_mismatches": len(mismatches), "mismatch_examples": mismatches[:10],
            "identical": bool(zero) and not mismatches}


# ------------------------------------------------------------------------------------------- io

def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    if not Path(path).exists():
        return []
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out = {}
            for k, v in r.items():
                if v in ("True", "False"):
                    out[k] = v == "True"
                else:
                    try:
                        out[k] = int(v)
                    except ValueError:
                        try:
                            out[k] = float(v)
                        except ValueError:
                            out[k] = v
            rows.append(out)
    return rows


class CsvAppender:
    """Writes rows as they arrive so a killed run still leaves its finished boards on disk."""

    def __init__(self, path, fields):
        self.f = open(path, "w", newline="")  # noqa: SIM115, closed by close() at the end of the phase
        self.w = csv.DictWriter(self.f, fieldnames=fields, extrasaction="ignore")
        self.w.writeheader()

    def add(self, rows):
        self.w.writerows(rows)
        self.f.flush()

    def close(self):
        self.f.close()


def pick_boards(manifest_path, limit_boards, sweep_boards, max_held_out):
    """Sweep boards: the first sweep_boards owner split validation boards by id. Held out: all with held_out."""
    boards = tu.load_manifest(manifest_path)
    _, val, held_out, _ = tu.split_samples(boards, "owner")
    by_id = {b["board_id"]: b for b in boards}
    val_ids = sorted({bid for bid, _ in val})[:sweep_boards]
    sweep = [dict(by_id[b], _split="validation") for b in val_ids]
    held = [dict(b, _split="held_out") for b in sorted(held_out, key=lambda b: b["board_id"])]
    if max_held_out:
        held = held[:max_held_out]
    if limit_boards:
        sweep, held = sweep[:limit_boards], held[:limit_boards]
    return sweep, held


def choose_lambda(per_lambda, ratio_key="length_ratio_mean"):
    """Highest completion rate, ties (within 1e-9) broken by the lowest length ratio.

    per_lambda maps the lambda (as a string) to a summary with completion_rate and ratio_key.
    """
    best, best_key = None, None
    for lam_str, s in per_lambda.items():
        if not s:
            continue
        ratio = s.get(ratio_key)
        key = (-round(s["completion_rate"] or 0.0, 9), ratio if ratio is not None else INF)
        if best_key is None or key < best_key:
            best, best_key = float(lam_str), key
    return best


def resolution_check(boards, data_dir, via_cost, grid=256):
    """Unguided setting (a) at the stored 256 px grid, to measure how much the 128 grid alone blocks.

    Returns the completion rate at the finer grid over the same boards, with the human path checks.
    """
    neigh = neighbour_table(grid, grid)
    n = done = connected_free = single = 0
    t0 = time.time()
    for b in boards:
        x, y, _ids, _mm = load_board(data_dir, b["board_id"], grid)
        for idx in range(len(x)):
            x6 = x[idx]
            obs = base_obstacles(x6, x6[5])
            res = route_net(x6, obs, None, 0.0, via_cost, neigh)
            if res["n_terminals"] < 2:
                single += 1
                continue
            human = human_reference(y[idx], x6, obs)
            n += 1
            done += int(res["completed"])
            connected_free += int(human["human_free_components"] <= 1)
    return {"grid": grid, "n_boards": len(boards), "n_nets": n, "completed": done,
            "completion_rate": done / n if n else None,
            "human_path_unblocked_rate": connected_free / n if n else None,
            "n_single_terminal_excluded": single, "seconds": time.time() - t0}


def print_table(title, groups):
    print(f"\n{title}")
    print(f"{'run':<34}{'nets':>7}{'done':>7}{'rate':>8}{'len ratio':>11}{'vias':>8}{'s/board':>9}")
    for name, s in groups:
        if not s:
            continue
        nets = s.get("n_nets", s.get("nets_attempted"))
        done = s.get("completed", s.get("nets_completed"))
        ratio = s.get("length_ratio_mean", s.get("length_ratio_of_sums"))
        vias = s.get("vias_mean_both_completed", s.get("vias"))
        rate = s.get("completion_rate")
        print(f"{name:<34}{nets:>7}{done:>7}{(rate if rate is not None else 0):>8.3f}"
              f"{(ratio if ratio is not None else 0):>11.3f}{(vias if vias is not None else 0):>8.2f}"
              f"{s.get('seconds_per_board', 0):>9.1f}")


# ------------------------------------------------------------------------------------------ main

def phase_sweep(args, model, device, sweep_boards, neigh):
    lams = [float(v) for v in args.lambdas.split(",")]
    app = CsvAppender(RESULTS / "route_grid_sweep_single_net.csv", SINGLE_NET_FIELDS)
    rows, t0 = [], time.time()
    for i, b in enumerate(sweep_boards, start=1):
        x, y, net_ids, mm = load_board(args.data_dir, b["board_id"])
        tb = time.time()
        new = run_single_net(b, x, y, net_ids, mm, model, device, lams, args.via_cost, "sweep", neigh)
        app.add(new)
        rows.extend(new)
        done = sum(r["completed"] for r in new if r["run"] == "unguided" and r["n_terminals"] >= 2)
        att = sum(1 for r in new if r["run"] == "unguided" and r["n_terminals"] >= 2)
        print(f"sweep {i}/{len(sweep_boards)} {b['board_id'][:40]:<40} nets {len(x):3d} unguided {done}/{att} "
              f"{time.time() - tb:6.1f}s", flush=True)
    app.close()
    ung = [r for r in rows if r["run"] == "unguided"]
    sweep = {
        "boards": [b["board_id"] for b in sweep_boards],
        "lambdas": lams, "via_cost": args.via_cost, "grid": GRID, "model": rel_path(args.model),
        "unguided": summarize_single(ung),
        "per_lambda": {str(lam): summarize_single([r for r in rows if r["run"] == "guided" and r["lam"] == lam], ung)
                       for lam in lams},
        "lambda0_check": lambda0_check(rows),
        "unguided_failure_diagnosis": diagnose_failures(ung),
        "seconds_total": time.time() - t0,
    }
    # a common set: nets completed by every run, so the length comparison is over the same nets
    common = None
    for name in ["unguided"] + [f"g{lam}" for lam in lams]:
        sel = ([r for r in rows if r["run"] == "unguided"] if name == "unguided"
               else [r for r in rows if r["run"] == "guided" and r["lam"] == float(name[1:])])
        done = {(r["board_id"], r["idx"]) for r in sel if r["completed"] and r["n_terminals"] >= 2
                and r["human_length_cells"] > 0}
        common = done if common is None else common & done
    sweep["common_completed_nets"] = len(common)
    sweep["length_ratio_mean_common"] = {
        "unguided": mean(r["length_cells"] / r["human_length_cells"] for r in ung if (r["board_id"], r["idx"]) in common),
        **{str(lam): mean(r["length_cells"] / r["human_length_cells"] for r in rows
                          if r["run"] == "guided" and r["lam"] == lam and (r["board_id"], r["idx"]) in common)
           for lam in lams}}
    sweep["vias_mean_common"] = {
        "unguided": mean(r["vias"] for r in ung if (r["board_id"], r["idx"]) in common),
        **{str(lam): mean(r["vias"] for r in rows if r["run"] == "guided" and r["lam"] == lam
                          and (r["board_id"], r["idx"]) in common) for lam in lams}}
    sweep["chosen_lambda_single_net"] = choose_lambda(sweep["per_lambda"])
    print_table("lambda sweep, setting (a), validation boards",
                [("unguided", sweep["unguided"])] + [(f"guided lambda {lam:g}", sweep["per_lambda"][str(lam)])
                                                    for lam in lams])
    print(f"lambda 0 check: {sweep['lambda0_check']}")
    print(f"setting (a) rule picks lambda {sweep['chosen_lambda_single_net']}", flush=True)

    # Setting (a) cannot separate lambdas on completion: the search is complete, so every lambda finds a
    # path whenever one exists, and the unguided path is the shortest by construction. The same rule
    # is therefore also applied to setting (b), where earlier nets change what later nets can do.
    fapp = CsvAppender(RESULTS / "route_grid_sweep_full_board.csv", FULL_BOARD_FIELDS)
    frows, tf = [], time.time()
    for i, b in enumerate(sweep_boards, start=1):
        x, y, net_ids, mm = load_board(args.data_dir, b["board_id"])
        tb = time.time()
        new = []
        for order_name in ("stored", "shortest"):
            new.append(run_full_board(b, x, y, net_ids, mm, model, device, 0.0, args.via_cost, order_name,
                                      False, "sweep", neigh))
            for lam in lams:
                new.append(run_full_board(b, x, y, net_ids, mm, model, device, lam, args.via_cost, order_name,
                                          True, "sweep", neigh))
        fapp.add(new)
        frows.extend(new)
        print(f"sweep (b) {i}/{len(sweep_boards)} {b['board_id'][:40]:<40} nets {len(x):3d} {time.time() - tb:6.1f}s",
              flush=True)
    fapp.close()
    def full_group(sel):
        group = {"unguided": summarize_full([r for r in sel if r["run"] == "unguided"])}
        for lam in lams:
            group[str(lam)] = summarize_full([r for r in sel if r["run"] == "guided" and r["lam"] == lam])
        return group

    fb = {"seconds_total": time.time() - tf,
          "per_order": {o: full_group([r for r in frows if r["order"] == o]) for o in ("stored", "shortest")},
          "pooled": full_group(frows)}
    per_lambda_b = {k: v for k, v in fb["pooled"].items() if k != "unguided"}
    fb["chosen_lambda"] = choose_lambda(per_lambda_b, "length_ratio_of_sums")
    nonzero = {k: v for k, v in per_lambda_b.items() if float(k) > 0}
    fb["best_nonzero_lambda"] = choose_lambda(nonzero, "length_ratio_of_sums") if nonzero else None
    sweep["full_board"] = fb
    sweep["chosen_lambda_full_board"] = fb["chosen_lambda"]
    # The lambda used on the held out boards: the rule applied to setting (b). When that rule also lands
    # on 0 (no guidance), the best nonzero lambda is used instead so the held out numbers measure the
    # size of the effect rather than repeating the unguided run; the flag records which case applied.
    sweep["rule_chose_zero"] = not fb["chosen_lambda"]
    sweep["chosen_lambda"] = fb["chosen_lambda"] if fb["chosen_lambda"] else fb["best_nonzero_lambda"]
    print_table("lambda sweep, setting (b) pooled over both orders (ratio of sums, total vias)",
                [(k, v) for k, v in fb["pooled"].items()])
    for order_name in ("stored", "shortest"):
        print_table(f"lambda sweep, setting (b), {order_name} order",
                    [(k, v) for k, v in fb["per_order"][order_name].items()])
    print(f"setting (b) rule picks lambda {fb['chosen_lambda']}; held out runs use {sweep['chosen_lambda']}",
          flush=True)

    if not args.no_resolution_check:
        sweep["resolution_check_256"] = resolution_check(sweep_boards, args.data_dir, args.via_cost)
        print(f"unguided setting (a) at 256: {sweep['resolution_check_256']}")
    with open(RESULTS / "route_grid_sweep.json", "w") as f:
        json.dump(sweep, f, indent=1)
    return sweep


def phase_held_out(args, model, device, held, lam, neigh):
    single = CsvAppender(RESULTS / "route_grid_single_net.csv", SINGLE_NET_FIELDS)
    full = CsvAppender(RESULTS / "route_grid_full_board.csv", FULL_BOARD_FIELDS)
    srows, frows, t0 = [], [], time.time()
    for i, b in enumerate(held, start=1):
        x, y, net_ids, mm = load_board(args.data_dir, b["board_id"])
        tb = time.time()
        new = run_single_net(b, x, y, net_ids, mm, model, device, [lam], args.via_cost, "held_out", neigh)
        single.add(new)
        srows.extend(new)
        ta = time.time() - tb
        newf = []
        for order_name in ("stored", "shortest"):
            for guided in (False, True):
                newf.append(run_full_board(b, x, y, net_ids, mm, model, device, lam, args.via_cost, order_name,
                                           guided, "held_out", neigh))
        full.add(newf)
        frows.extend(newf)
        print(f"held out {i}/{len(held)} {b['board_id'][:36]:<36} nets {len(x):3d} test {b['in_test_set']!s:<5} "
              f"(a) {ta:5.1f}s (b) {time.time() - tb - ta:5.1f}s  elapsed {(time.time() - t0) / 60:5.1f} min",
              flush=True)
    single.close()
    full.close()
    return srows, frows


def build_summary(args, sweep, srows, frows, lam, held_requested, held_total, elapsed):
    def single_groups(rows):
        ung = [r for r in rows if r["run"] == "unguided"]
        gd = [r for r in rows if r["run"] == "guided"]
        return {"unguided": summarize_single(ung), "guided": summarize_single(gd, ung),
                "lambda0_check": None, "unguided_failure_diagnosis": diagnose_failures(ung)}

    def full_groups(rows):
        out = {}
        for order_name in ("stored", "shortest"):
            for run in ("unguided", "guided"):
                out[f"{order_name}_{run}"] = summarize_full([r for r in rows if r["order"] == order_name and r["run"] == run])
        return out

    subsets = {"all": lambda r: True, "in_test_set": lambda r: bool(r["in_test_set"]),
               "not_in_test_set": lambda r: not bool(r["in_test_set"])}
    summary = {
        "grid": GRID, "via_cost": args.via_cost, "model": rel_path(args.model), "device": str(args.device_used),
        "chosen_lambda": lam,
        "sweep": {k: sweep.get(k) for k in ("boards", "lambdas", "unguided", "per_lambda", "lambda0_check",
                                            "unguided_failure_diagnosis", "common_completed_nets",
                                            "length_ratio_mean_common", "vias_mean_common",
                                            "chosen_lambda_single_net", "full_board", "chosen_lambda_full_board",
                                            "rule_chose_zero", "chosen_lambda", "resolution_check_256",
                                            "seconds_total")} if sweep else None,
        "held_out": {
            "boards_run": len({r["board_id"] for r in srows}),
            "boards_requested": held_requested,
            "boards_available": held_total,
            "truncated": held_requested < held_total,
            "seconds_total": elapsed,
            "single_net": {name: single_groups([r for r in srows if f(r)]) for name, f in subsets.items()},
            "full_board": {name: full_groups([r for r in frows if f(r)]) for name, f in subsets.items()},
        } if srows else None,
    }
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("sweep", "held-out", "all"), default="all")
    ap.add_argument("--lambdas", default=",".join(f"{v:g}" for v in LAMBDAS))
    ap.add_argument("--lam", type=float, default=None, help="lambda for the held out run; default from the sweep")
    ap.add_argument("--via-cost", type=float, default=10.0)
    ap.add_argument("--sweep-boards", type=int, default=30)
    ap.add_argument("--max-held-out-boards", type=int, default=0, help="0 runs every held out board")
    ap.add_argument("--limit-boards", type=int, default=0, help="cap both board lists, for quick runs")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data-dir", default=str(tu.DEFAULT_DATA))
    ap.add_argument("--manifest", default=str(tu.DEFAULT_MANIFEST))
    ap.add_argument("--summarize-only", action="store_true", help="rebuild the summary from the CSV files")
    ap.add_argument("--no-resolution-check", action="store_true",
                    help="skip the unguided 256 px rerun of setting (a) on the sweep boards")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    DOC_RESULTS.mkdir(parents=True, exist_ok=True)
    device = tu.pick_device(args.device)
    args.device_used = device
    sweep_boards, held = pick_boards(args.manifest, args.limit_boards, args.sweep_boards, args.max_held_out_boards)
    held_total = len(pick_boards(args.manifest, 0, args.sweep_boards, 0)[1])

    sweep_path = RESULTS / "route_grid_sweep.json"
    sweep = None
    if sweep_path.exists():
        with open(sweep_path) as f:
            sweep = json.load(f)
    if args.summarize_only:
        srows = read_csv(RESULTS / "route_grid_single_net.csv")
        frows = read_csv(RESULTS / "route_grid_full_board.csv")
        lam = args.lam if args.lam is not None else (sweep["chosen_lambda"] if sweep else None)
        summary = build_summary(args, sweep, srows, frows, lam, len({r["board_id"] for r in srows}), held_total, None)
    else:
        model = tu.load_checkpoint(args.model, device)
        model.eval()
        neigh = neighbour_table(GRID, GRID)
        t0 = time.time()
        if args.phase in ("sweep", "all"):
            sweep = phase_sweep(args, model, device, sweep_boards, neigh)
        srows, frows, elapsed = [], [], None
        lam = args.lam if args.lam is not None else (sweep["chosen_lambda"] if sweep else None)
        if args.phase in ("held-out", "all"):
            if lam is None:
                raise SystemExit("no lambda: run the sweep first or pass --lam")
            th = time.time()
            srows, frows = phase_held_out(args, model, device, held, lam, neigh)
            elapsed = time.time() - th
        summary = build_summary(args, sweep, srows, frows, lam, len(held), held_total, elapsed)
        summary["seconds_total_this_run"] = time.time() - t0

    with open(RESULTS / "route_grid_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    shutil.copy(RESULTS / "route_grid_summary.json", DOC_RESULTS / "route_grid_summary.json")

    if summary.get("held_out"):
        for name in ("all", "in_test_set", "not_in_test_set"):
            s = summary["held_out"]["single_net"][name]
            print_table(f"held out {name}: setting (a) single net, lambda {lam}",
                        [("unguided", s["unguided"]), ("guided", s["guided"])])
            fb = summary["held_out"]["full_board"][name]
            print_table(f"held out {name}: setting (b) full board (ratio is of sums, vias are totals)",
                        [(k, v) for k, v in fb.items()])
            print(f"  (a) unguided failure diagnosis: "
                  f"{ {k: v['count'] for k, v in s['unguided_failure_diagnosis']['classes'].items()} }")
    print(f"\nwrote {RESULTS / 'route_grid_summary.json'} and {DOC_RESULTS / 'route_grid_summary.json'}")


if __name__ == "__main__":
    main()
