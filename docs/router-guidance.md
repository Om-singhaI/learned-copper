# Guided grid routing

The UNet copper prediction becomes a per cell cost for a maze router, compared with the same router unguided. Numbers: `docs/results/route_grid_summary.json`.

## Router and settings

The 256 px raster is max pooled to 128, the model's resolution. The router is a two layer A star, 8 connected, with no diagonal move between two blocked cells and vias occupying their cell on both layers. Unguided, a step costs 1 and a via 10. Guided, a step costs 1 + lambda (1 minus p) on its layer and a via 10 (1 + lambda (1 minus the larger layer p)). Lambda 0 reproduced the unguided router on all 894 sweep nets.

Setting (a) routes each net alone against the human copper of all other nets. Setting (b) routes all sampled nets of a board in sequence from an empty context, in stored order or shortest first by pad bounding box half perimeter; the router's copper becomes obstacle and context. Pooled human tracks are wider than one cell, so length ratios sit below one.

## Sweep on 30 validation boards

Setting (a), 863 nets: the unguided router completes 58.9 percent, unchanged by lambda, because the search is complete and the unguided path is already the shortest. The rule picks lambda 0 there; guidance only adds length (ratio 0.752 to 0.764) and vias (0.15 to 0.26 per net) at lambda 16. In setting (b), both orders pooled, completion is 89.8 percent unguided and 92.0 at lambda 4, the best of 1, 2, 4, 8 and 16, with the length ratio of sums up from 0.544 to 0.583 and vias from 1334 to 1692. Lambda 4 went to the held out run.

Of the 355 failures in setting (a), 333 have the human path cut by obstacles after pooling, 16 need a diagonal squeeze, 6 a zone, none are unexplained. On the stored 256 grid the unguided router completes 89.5 percent of 891 nets; at 128, nets on boards at 0.6 mm per cell or coarser complete 43.3 percent (219 of 506) against 88.9 under 0.3 mm. The grid is too coarse for setting (a).

## Held out boards

All 168 boards ran in 3.8 minutes: 96 test boards (808 nets) and 72 from their repositories (1768 nets). Setting (a) completes 66.3 percent guided or not, 81.3 on test boards and 59.4 on the rest. Setting (b), lambda 4, test boards: stored order 98.8 percent unguided against 99.8 guided, shortest first 99.6 against 100.0, length ratio 0.341 against 0.345, vias 405 against 515 and 295 against 391. Other boards: stored 93.7 against 94.2, shortest 94.4 against 94.7, vias 1261 against 1432 and 859 against 1098.

Guidance adds a few completed nets in setting (b) for 14 to 33 percent more vias and 0.2 to 1.2 percent more wire; shortest first ordering alone gains about as much. Suspect: the grid hides a third of setting (a), human vias come from both layer cells, and in setting (b) the model sees an empty context it never trained on.
