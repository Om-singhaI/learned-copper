# Frozen splits

Never train on anything listed here. Regenerate with `.venv/bin/python scripts/freeze_test_split.py` after `data/mirror/freerouting` is cloned (see docs/data-plan.md).

## test_d3.json

The held out test set: PCBWorld's D3 test lists (99 easy, 10 medium, 10 hard, 119 boards) from configs/datasets/d3.json at commit ed411746224f, with its one test override applied (0373_badge2016 replaced by 0376_cat-trainer), mapped onto the Freerouting PCBench mirror (freerouting/freerouting, scripts/benchmark/fixtures/PCBench, commit recorded in the file). Each entry carries the board id, source repository, license, layer, net and component counts, the human reference's segment count, via count, track length and unconnected count, and two hashes of raw.kicad_pcb: the mirror file as checked out (LF line endings) and the upstream PCBench hash from board-manifest.json, which was generated on Windows with CRLF line endings. The script verifies the two agree after line ending conversion.

## test_exclude_repos.txt

The 94 source repositories behind the 119 test boards. Every board from these repositories, in any dataset, stays out of training and validation.

## mirror_inventory.csv

Structural counts for all 1,157 mirror boards as loaded by KiCad 10 (`scripts/inventory.py`): file version, copper layers, outline size, footprints, pads, nets, segments, vias, track length, copper zones, unconnected connections, with the mirror's ground truth beside them for cross checking.

## mirror_drc_refill.csv and mirror_pool.json

`scripts/drc_batch.py --refill` runs KiCad 10's design rule check on every mirror board with copper zones refilled first (stale fills in old files otherwise produce thousands of false clearance errors). `scripts/filter_boards.py` turns that summary plus the inventory into mirror_pool.json: every board with a tier (clean, flagged, rejected), its rejection reasons, source repository, license, and whether it is held out because it is a test board or shares a repository with one. Training uses clean and flagged boards with held_out false.
