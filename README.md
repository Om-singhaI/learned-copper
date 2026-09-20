# Learned Copper

Guiding PCB autorouting with a model trained on human routed boards.

Term project for CS57600 Machine Learning, Purdue University, Fall 2026. Solo project by Om Singhal.

## The idea

Routing is the slowest manual step in printed circuit board design, and the automatic routers people bolt onto tools such as KiCad are known for layouts that get thrown away. Thousands of boards routed by real engineers sit in public open hardware repositories. This project uses them as training data.

Given a board with placed footprints, a netlist and design rules but no traces, a model predicts where copper belongs on each layer. That prediction is then used as a cost map to guide a router, and the guided router is compared with the unguided one on held out boards: completion rate, wire length, via count and design rule violations, all measured by KiCad.

The evaluation is the point as much as the model. Open hardware collections are full of forks and revisions of the same board, so every experiment is run twice: once with a random split, and once with splits by repository owner, to measure how much near duplicate boards inflate the numbers.

The full proposal is in [docs/proposal.md](docs/proposal.md).

## Data

Primary source: [bshada/open-schematics](https://huggingface.co/datasets/bshada/open-schematics) on Hugging Face (CC BY 4.0), a collection of KiCad projects gathered from public repositories, each with its PCB layout files. Backup and evaluation sets: the boards packaged with [PCBWorld](https://github.com/LGAI-Research/PCBWorld) and the [PCBench](https://github.com/PCBench/PCBench) pool.

Raw boards are never committed to this repository. Derived data records the license of every source board.

## Plan

| Sprint | Weeks | Goal |
| --- | --- | --- |
| 1 | 1 to 2 | Parse and rasterize about 200 boards, run Freerouting on them, record the baseline completion rate. Go or no go. |
| 2 | 3 to 4 | Process the full collection, filter, deduplicate, explore, freeze owner based splits. |
| 3 | 5 to 6 | Copper prediction model and gradient boosting baseline, error study, midway report. |
| 4 | 7 to 8 | Router guidance experiments with design rule checked evaluation. |
| 5 | 9 | Random split comparison and error analysis. |
| 6 | 10 to 11 | Presentation and final report. |

Progress is tracked on the [project board](https://github.com/users/Om-singhaI/projects/2). Sprint 1 runs September 19 to September 25, 2026.

## Repository layout

```
docs/       proposal and sprint notes
scripts/    one off data and evaluation scripts
src/        the learned_copper package
data/       local only, ignored by git
```

## Setup

Planned toolchain, to be confirmed in sprint 1:

- Python 3.11 or newer
- KiCad 9 for the pcbnew Python module and the design rule checker
- Java 21 for Freerouting
- Free Kaggle or Colab GPUs for training

```
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## License

Code is released under the MIT License. Derived datasets carry the licenses of their source boards.
