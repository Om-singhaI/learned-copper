# Data plan (revised September 24, 2026)

The proposal named open-schematics as the primary source and PCBWorld as the evaluation set. A full search on September 24 (Hub, GitHub, papers, benchmark suites, data portals, 44 candidates checked) found no ready made train, validation and test split with baselines anywhere. It did find a much better starting point than sampling open-schematics by hand.

## Test set, frozen, never trained on

PCBWorld's fixed test lists (configs/datasets/d3.json): 99 easy, 10 medium, 10 hard, 119 boards. Their "train" lists are just tier membership and contain the test boards, so every D3 board in a test list, and every source repository behind it, is excluded from training. Optional ablation later: hold out all 678 D3 boards and compare zero shot, which is how PCBWorld's own agents were evaluated.

Where the boards come from: the Freerouting project mirrors PCBench at scripts/benchmark/fixtures/PCBench, 1,157 boards, each folder holding raw.kicad_pcb (human copper, zones intact), unrouted.kicad_pcb (tracks, vias and zones stripped), both DSNs, and ground_truth.json with layer count, segment and via counts, track length and a KiCad 10 DRC breakdown including the unconnected count. catalog.json lists tiers for all 1,157. The same repo publishes per board Freerouting results for nine Freerouting versions, so a baseline number exists for every test board before we run anything.

Published reference numbers (PCBWorld Table 3, clean pass at 5 attempts): Freerouting 0.80 on D3 A and 0.78 on D3 B, PPO agent 0.86 and 0.45, GPT based agents 0.65 and 0.00. Those were produced with a KiCad 9.0.8 engine and patched rule severities, so we quote them as the reference and re run Freerouting ourselves under KiCad 10 DRC for the apples to apples comparison.

Secondary held out set: the DAC 2020 and ASP DAC 2021 PCB benchmarks, 12 human routed boards with unrouted twins and published Freerouting results, reported separately.

## Training pool

Stage 1, local, about 1,400 boards. The emshotton/PCBench superset (MIT, 1,594 boards: the 1,182 PCBench originals plus 412 new KiCad boards from 171 repositories), minus the 119 test boards, minus their source repositories, minus a validation slice. Train on raw.kicad_pcb, never processed.kicad_pcb, whose zone connected nets were re routed by Freerouting in the original PCBench cleaning. Drop boards with an unconnected count above zero in ground_truth.json.

Stage 2, local, about 2,000 more boards, all disjoint from PCBench. Adafruit and SparkFun Eagle boards (roughly 1,000 distinct two layer designs, CC BY SA; pcbnew imports Eagle, with a layer remap and the copper layer count forced to 2). OLIMEX, Tinkerforge and Great Scott Gadgets KiCad boards (about 500 routed, 300 to 350 distinct designs, mixed open hardware licenses; six OLIMEX boards are in PCBench and must be dropped). Antmicro open hardware boards (about 80 distinct 4 to 14 layer designs, Apache 2.0), which are the only real multilayer coverage. Small clean extras: EDA bench gold boards, KiCad demos, OpenHardware.io.

Stage 3, Kaggle only. open-schematics, streamed shard by shard with column projection. Measured on September 24: 87,931 records, 60,280 with a KiCad board, 21,039 projects with boards from 14,184 owners, about 2.8 board files per project, so roughly 55,000 board files before deduplication and filtering, and an estimated 9,000 to 20,000 clean routed boards after. Every repository already in PCBench, in the emshotton superset or in stage 2 is excluded by name. Needs a Hugging Face token; anonymous reads were rate limited.

## Validation

A frozen slice of about 120 boards from the non test part of the PCBench pool, grouped by source repository and stratified by pin count like PCBWorld's tiers, stored as a list of file hashes. The open-schematics run gets its own slice of about 500 boards from its own pool.

## Rules that apply to every source

- Group by source repository or product, never by file. PCBench has 846 repositories for 1,194 designs, OLIMEX has up to 16 revisions of one product, Adafruit keeps revisions side by side, open-schematics repeats the same boards across schematic sheets.
- One stripping rule for all inputs. The mirror's unrouted twin strips zones; PCBWorld keeps them. Decision pending in issue 5, then applied uniformly, or completion and wire length numbers will not match either baseline.
- Splits are frozen as hash lists with pinned commits and shipped in the repository.
- Scoring is scripted end to end: unconnected count zero for completion, copper rule DRC count, wire length and via count from pcbnew, compared with ground_truth.json. A handful of outputs per model still get eyeballed for degenerate routings that pass DRC.

## Licensing, which decides what can be published

The PCBench originals carry no usable license and are research use only; they are never redistributed. The Freerouting mirror is used the same way. What can go into a public dataset release: the Eagle vendor boards (CC BY SA, share alike), the Apache and CERN OHL boards, Antmicro, and the open-schematics derived set with per board upstream licenses recorded. The model weights can be released regardless; the dataset card states exactly which boards were seen in training.

## Known limits

The pool is about 89 percent two layer hobby boards. A model trained on it will not transfer to 6 to 14 layer industrial boards, and the report says so. Human reference boards are not DRC clean under KiCad 10 defaults (median about 208 violations, mostly silkscreen and footprint library types), so DRC comparisons use copper rule types only or the delta against the human reference. Boards over about 10 MB stay on Kaggle; the laptop cannot open them comfortably.
