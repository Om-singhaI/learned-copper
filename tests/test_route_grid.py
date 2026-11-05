"""Synthetic 16 by 16 checks for scripts/route_grid.py.

    .venv/bin/python -m pytest tests/test_route_grid.py -q
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import route_grid as rg

S = 16
SQ2 = math.sqrt(2.0)


def board(pads_f=(), pads_b=(), other_f=(), other_b=(), context=()):
    """A 16 by 16 input with a full outline and single cell pads at the given (r, c) positions."""
    x = np.zeros((6, S, S), np.uint8)
    x[0] = 1
    for ch, cells in ((1, pads_f), (2, pads_b), (3, other_f), (4, other_b), (5, context)):
        for r, c in cells:
            x[ch, r, c] = 1
    return x


def route(x, p=None, lam=0.0, via=10.0, own=None):
    obs = rg.base_obstacles(x, x[5], own)
    return rg.route_net(x, obs, p, lam, via)


def test_straight_path():
    res = route(board(pads_f=[(8, 2), (8, 13)]))
    assert res["completed"] and res["n_terminals"] == 2
    assert abs(res["length_cells"] - 11) < 1e-9
    assert res["vias"] == 0
    assert res["copper"][0, 8, 3:13].all() and res["copper"].sum() == 10  # pads are not copper


def test_diagonal_path():
    res = route(board(pads_f=[(2, 2), (10, 10)]))
    assert res["completed"]
    assert abs(res["length_cells"] - 8 * SQ2) < 1e-9


def test_wall_with_gap_known_optimum():
    # a wall on both layers down column 8 with one gap at row 14; the gap must be entered and left
    # straight because the diagonal squeeze rule forbids cutting past the wall cells
    wall = [(r, 8) for r in range(S) if r != 14]
    x = board(pads_f=[(2, 2), (2, 13)], other_f=wall, other_b=[(r, c) for r in range(S) for c in range(S)])
    res = route(x)
    assert res["completed"] and res["vias"] == 0
    expected = (5 * SQ2 + 7) + 2 + (4 * SQ2 + 8)
    assert abs(res["length_cells"] - expected) < 1e-9
    assert res["copper"][0, 14, 8]


def test_needs_a_via():
    # pads on different layers: one via, eleven steps
    res = route(board(pads_f=[(2, 2)], pads_b=[(2, 13)]))
    assert res["completed"]
    assert res["vias"] == 1
    assert abs(res["length_cells"] - 11) < 1e-9
    # eleven moves and one via touch twelve layer cells, one of which is the back pad
    assert res["copper"].sum() == 11


def test_front_wall_needs_two_vias():
    # both pads on the front with a full front wall between them: under the wall and back up
    wall = [(r, 8) for r in range(S)]
    x = board(pads_f=[(2, 2), (2, 13)], other_f=wall)
    res = route(x)
    assert res["completed"]
    assert res["vias"] == 2
    assert abs(res["length_cells"] - 11) < 1e-9
    assert res["copper"][1, 2, 8]


def test_diagonal_squeeze_is_refused():
    # pads at (7,7) and (8,8); the two cells between them are blocked and the back layer is full
    x = board(pads_f=[(7, 7), (8, 8)], other_f=[(7, 8), (8, 7)],
              other_b=[(r, c) for r in range(S) for c in range(S)])
    res = route(x)
    assert res["completed"]
    assert abs(res["length_cells"] - 6) < 1e-9


def test_multi_terminal_tree():
    res = route(board(pads_f=[(8, 2), (8, 8), (8, 14)]))
    assert res["completed"] and res["n_terminals"] == 3
    assert abs(res["length_cells"] - 12) < 1e-9


def test_enclosed_pad_fails_without_crash():
    ring = [(r, c) for r in range(11, 16) for c in range(11, 16) if (r, c) != (13, 13)]
    x = board(pads_f=[(2, 2), (13, 13)], other_f=ring, other_b=ring)
    res = route(x)
    assert not res["completed"]
    assert res["failed_connections"] == 1
    assert res["length_cells"] == 0


def test_lambda_zero_matches_unguided():
    rng = np.random.default_rng(3)
    for trial in range(5):
        x = board(pads_f=[(1, 1), (14, 14)], pads_b=[(14, 1)])
        obs_f = [(int(r), int(c)) for r, c in rng.integers(2, 13, size=(30, 2))]
        obs_b = [(int(r), int(c)) for r, c in rng.integers(2, 13, size=(30, 2))]
        for r, c in obs_f:
            x[3, r, c] = 1
        for r, c in obs_b:
            x[4, r, c] = 1
        p = rng.random((2, S, S))
        a, b = route(x), route(x, p=p, lam=0.0)
        assert a["completed"] == b["completed"]
        assert abs(a["length_cells"] - b["length_cells"]) < 1e-9
        assert a["vias"] == b["vias"] and a["expansions"] == b["expansions"]
        assert np.array_equal(a["copper"], b["copper"])


def test_guidance_steers_the_path():
    x = board(pads_f=[(8, 1), (8, 14)])
    p = np.full((2, S, S), 0.01)
    p[0, 3, :] = 0.99
    unguided, guided = route(x), route(x, p=p, lam=16.0)
    assert abs(unguided["length_cells"] - 13) < 1e-9
    assert guided["completed"]
    assert guided["copper"][0, 3].sum() >= 4
    assert guided["length_cells"] > unguided["length_cells"]


def test_guided_via_cost_uses_prediction():
    x = board(pads_f=[(2, 2)], pads_b=[(2, 13)])
    p = np.full((2, S, S), 0.0)
    p[:, 2, 5] = 1.0  # a cheap via only at (2, 5); everywhere else a via costs 90 and a step 9
    res = route(x, p=p, lam=8.0)
    assert res["completed"] and res["vias"] == 1
    assert res["copper"][0, 2, 5] and res["copper"][1, 2, 5]
    assert (res["copper"][0] & res["copper"][1]).sum() == 1  # the via occupies its cell on both layers


def test_own_copper_blocks_per_layer():
    own = np.zeros((2, S, S), bool)
    own[0, :, 8] = True  # an earlier net runs down column 8 on the front only
    x = board(pads_f=[(2, 2), (2, 13)])
    res = route(x, own=own)
    assert res["completed"] and res["vias"] == 2
    own[1, :, 8] = True  # on both layers the net cannot cross
    res = route(x, own=own)
    assert not res["completed"] and res["failed_connections"] == 1


def test_human_reference_counts():
    x = board(pads_f=[(8, 2), (8, 13)])
    y = np.zeros((2, S, S), np.uint8)
    y[0, 8, 2:14] = 1
    y[:, 8, 6] = 1  # one via cell
    obs = rg.base_obstacles(x, x[5])
    h = rg.human_reference(y, x, obs)
    assert h["human_length_cells"] == 13 and h["human_via_cells"] == 1 and h["human_via_blobs"] == 1
    assert h["human_connected"] and h["human_free_components"] == 1
    x[5, 8, 9] = 1  # other copper on the human path cuts it
    obs = rg.base_obstacles(x, x[5])
    h = rg.human_reference(y, x, obs)
    assert h["human_components"] == 1 and h["human_free_components"] == 2


def test_hpwl():
    assert rg.hpwl(board(pads_f=[(2, 3)], pads_b=[(7, 10)])) == 12


def test_heuristic_is_admissible_and_consistent():
    # h never exceeds the octile distance to the nearest goal cell, and neighbours differ by at most
    # the move length, so A star with a closed set stays exact
    H = W = 12
    bboxes = [(2, 3, 8, 9), (9, 9, 1, 1)]
    h = np.array(rg.heuristic_grid(bboxes, H, W)).reshape(H, W)
    goals = [(r, c) for r0, r1, c0, c1 in bboxes for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    for r in range(H):
        for c in range(W):
            true = min(max(abs(r - gr), abs(c - gc)) + (SQ2 - 1) * min(abs(r - gr), abs(c - gc))
                       for gr, gc in goals)
            assert h[r, c] <= true + 1e-9
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    assert abs(h[r, c] - h[rr, cc]) <= (SQ2 if dr and dc else 1.0) + 1e-9


def test_choose_lambda_rule():
    per = {"0.0": {"completion_rate": 0.5, "length_ratio_mean": 0.7},
           "2.0": {"completion_rate": 0.6, "length_ratio_mean": 0.9},
           "4.0": {"completion_rate": 0.6, "length_ratio_mean": 0.8},
           "8.0": {"completion_rate": 0.6, "length_ratio_mean": None}}
    assert rg.choose_lambda(per) == 4.0
    assert rg.choose_lambda({k: v for k, v in per.items() if k != "4.0"}) == 2.0


def test_lambda0_check_detects_a_difference():
    base = {"board_id": "b", "idx": 0, "completed": True, "length_cells": 3.0, "vias": 0,
            "copper_crc": 1, "expansions": 5, "failed_connections": 0, "lam": 0.0}
    rows = [dict(base, run="unguided"), dict(base, run="guided")]
    assert rg.lambda0_check(rows)["identical"]
    rows[1]["copper_crc"] = 2
    out = rg.lambda0_check(rows)
    assert out["n_mismatches"] == 1 and out["mismatch_examples"][0]["fields"] == ["copper_crc"]


def test_terminal_labels_join_layers_at_the_same_cell():
    pads = np.zeros((2, S, S), bool)
    pads[0, 4, 4] = pads[1, 4, 4] = True  # a through hole pad
    pads[0, 4, 5] = True  # touching it on the front
    pads[1, 4, 6] = True  # diagonal to nothing it touches on the back
    _, n, cells, _bboxes = rg.label_terminals(pads)
    assert n == 2
    assert sorted(len(v) for v in cells.values()) == [1, 3]
