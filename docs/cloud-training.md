# Training at 256 px on a free cloud GPU

Written November 5, 2026. At 128 px the laptop needs about 23 minutes per epoch, so the 256 px runs go to Kaggle first and Colab second.

## The bundle

`.venv/bin/python scripts/make_raster_bundle.py --tag 2026-11-05` writes `data/bundles/learned-copper-rasters-2026-11-05.tar.gz` with the 645 rasters, the two split files and a README with the terms, and prints size and sha256; the November 5 bundle is 50.2 MB. The rasters derive from PCBench boards through the Freerouting mirror, so the bundle is never committed or published, and the Kaggle dataset and notebook stay private.

## Kaggle

1. Datasets, New Dataset: upload the archive and keep it private. Kaggle may unpack it on upload; the notebook accepts both forms.
2. Code, New Notebook, File, Import Notebook: upload `notebooks/kaggle_train_unet.ipynb`. The notebook clones the public repository at main, so any change to `train_unet.py` must be committed and pushed before the run.
3. Side panel: Add Input, pick the dataset; Accelerator GPU P100 (or T4 x2; the script uses one GPU); Internet on.
4. Check the settings cell (`EPOCHS` 30, `RUN_OWNER` true, `RUN_RANDOM` false), then Save Version, Save & Run All. A saved version runs unattended for up to 12 hours.
5. The finished version's Output tab holds `models/unet_owner_256/` and `unet_256_outputs.zip`; download the zip and keep the version so its output can feed the next one.
6. The random split is a second version with `RUN_OWNER` false and `RUN_RANDOM` true.

## Colab

Put the archive in `MyDrive/learned-copper/`, upload the notebook, choose a T4 runtime and run all cells. The settings cell detects Colab, mounts Drive and writes the models to `MyDrive/learned-copper/models/`, so checkpoints survive a disconnect. Free sessions are shorter and end without warning; the settings cell gives them a lower time limit.

## Resuming after a disconnect

`train_unet.py` writes `last.pt` after every epoch (model, optimizer, scheduler, epoch, best metric, RNG state) and rewrites `metrics.json` at the same time, so a dead session loses at most one epoch. The notebook always passes `--resume`: an existing `last.pt` continues the run with the cosine schedule at the same step, otherwise the run starts fresh. A new Kaggle version starts empty, so attach the previous version's output (Add Input, Your Work) and the runner cell copies `last.pt` from it. `--time-limit-minutes` stops the script after the last epoch that fits, so the final evaluation and the zip cell still run inside the session. On the laptop, a 40 epoch fixture run stopped after epoch 9 and resumed matched the uninterrupted run exactly.

## Time and budget

Epochs 2 to 10 of the 128 px owner run on the laptop took 1,345 to 1,431 seconds each over 12,961 training samples. At 256 px each sample has four times the pixels, so roughly 90 minutes per epoch on the laptop. A T4 runs these convolutions several times faster than the M3 GPU in float32 and faster again with float16 autocast (`--amp`). If the T4 is four times faster, an epoch is about 23 minutes in float32 and perhaps 10 to 15 with autocast. None of this is measured yet; the script prints every epoch's seconds. Kaggle allows 30 GPU hours per week. One 12 hour session per split uses 24 of them; at 15 minutes per epoch, 30 epochs fit in one session with time left for the held out evaluation.

## Workers and memory

`--workers 2` forks two loader processes on Linux, each with its own board cache. At 256 px over all boards the cache holds four boards per process, so memory stays small and a miss costs an npz load of about 6 ms. On macOS workers are spawned instead, so `--workers 0` stays the laptop default.
