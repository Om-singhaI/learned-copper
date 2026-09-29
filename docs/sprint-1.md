# Sprint 1: go or no go

Dates: September 19 to September 25, 2026.

## Goal

Decide whether the whole idea is worth pursuing. By the end of the sprint there is a working pipeline on about 200 boards from open-schematics, Freerouting has been run on them, and the baseline completion rate is written down.

## Exit criteria

- 200 boards sampled, parsed, and classified by layer count, footprint count and net count
- A script that strips tracks and vias from a board and keeps the original as the answer
- Freerouting run headless on the stripped boards, with completion rate, wire length, via count and DRC violations recorded per board
- One board rasterized to a multichannel array and plotted
- A short write up with the numbers and the decision

## Fallback

If the parsing yield is too low or batch routing does not work, the project narrows to copper prediction on the PCBWorld boards and the router guidance experiments are dropped.

## Toolchain notes (issue 1, September 24)

Everything the pipeline needs already ran on the laptop by the end of the day. `scripts/check_toolchain.sh` verifies it in one go.

| Tool | What is installed | Notes |
| --- | --- | --- |
| KiCad | 10.0.4 (Homebrew cask), 4.8 GB | Newer than the KiCad 9 in the proposal. Fine for parsing, DRC and DSN export. PCBWorld pins its own KiCad 9 engine, which is separate. |
| pcbnew Python | Python 3.9.13 bundled inside KiCad.app | `import pcbnew` only works with that interpreter, so KiCad scripts run through it and the ML code runs in the venv. `scripts/kicad_dsn.py` is the bridge. |
| kicad-cli | 10.0.4 | `kicad-cli pcb drc --format json --severity-all` gives violations and unconnected items as JSON. There is no DSN export in kicad-cli, so DSN goes through pcbnew. |
| Java | OpenJDK 21.0.12 (Homebrew) at /opt/homebrew/opt/openjdk@21/bin/java | The `java` on PATH is an old Intel JDK 18. Always use the Homebrew path. |
| Freerouting | jar built 2025 04 12, 64 MB, at tools/freerouting.jar (ignored by git) | Download from github.com/freerouting/freerouting/releases. Headless call: `java -jar tools/freerouting.jar -de in.dsn -do out.ses -mp 20 --gui.enabled=false`. Never pass `--help`, it opens the GUI and never returns. |
| Python venv | Python 3.13.15, torch 2.14.0 with MPS, 1.2G | Create with `/opt/homebrew/bin/python3.13 -m venv .venv` and always call `.venv/bin/python` by path. The shell aliases `python` and `python3` to a system 3.10, which broke the first install. |

Disk: 20 GB free before, 18 GB after (the venv is the only new install; KiCad, Java and the jar were already present).

### What the smoke test showed

Three real boards were pulled from the open-schematics rows API into data/raw/sample.

- DRC runs on all three. The counts are dominated by things that have nothing to do with routing: library footprint mismatches, solder mask bridges, silkscreen overlaps. The routing filter in issue 4 has to look at a short list of copper rule types (clearance, shorting items, track width, unconnected items) and ignore the rest.
- One of the three boards (B7) has 5 unconnected items, so "has tracks" is not the same as "fully routed". The filter must use the unconnected count.
- DSN export needs a wx application object and, on one of the three boards, fails outright. In a single long process the failure hung on a dialog; in its own subprocess it returns in 0.3 seconds with a clean failure. Every board gets its own subprocess with a timeout from now on.
- Freerouting on the B7 board: two passes in well under a second, incomplete count 0. Importing the session back gave 19 tracks and vias where there were none, DRC unconnected went from 5 to 0, and the other 5 violations were unchanged. The whole export, route, import, check loop works.

Open question for issue 6: why the Uart programmer board refuses to export. Probably a footprint or pad the exporter rejects.

## Test set frozen (issue 2, September 28)

The Freerouting PCBench mirror is cloned under data/mirror/freerouting (partial clone, fixtures and benchmark results only, 1.3 GB). All 119 PCBWorld D3 test boards map onto mirror folders, with PCBWorld's one override applied. splits/test_d3.json and splits/test_exclude_repos.txt are committed.

Numbers from the mirror catalog (1,157 boards): tiers A 453, B 560, C 122, D 22; 1,037 two layer, 118 four layer, one six, one eight; 179 boards whose human reference still has unconnected items; 752 boards with no license recorded, then MIT 106, GPL 3.0 104. Of the 119 test boards, 118 are two layer and 14 have unconnected items in the human reference, which the completion metric will have to account for.

Trap: board-manifest.json hashes were computed on Windows from CRLF files; the mirror stores LF. The freeze script checks hashes after converting line endings.

## Inventory (issue 3, September 29)

`scripts/inventory.py` drives `scripts/inventory_worker.py` under KiCad's Python, four workers, chunks of 40 with timeouts. All 1,157 mirror boards parsed in 20 seconds, none failed. Vias, layers and unconnected counts agree with the mirror's ground truth on every board; segment counts differ on 12 boards, eleven of them by one to five segments and one (kitspace_Unifying) by about a thousand, which looks like a difference in how arcs or overlapping segments were counted upstream. Not chased.

What the pool looks like:

- Layers: 1,037 two layer, 118 four layer, one six, one eight.
- File formats: almost all KiCad 4 and 5 era (640 version 4, 366 from 2017, 87 version 3). KiCad 10 loads them all. This matters for open-schematics later, which spans the same range.
- Size: median board 25 cm2, 90th percentile 98 cm2, largest 666 cm2. Median 33 footprints, 36 nets, 279 track segments; the biggest board has 1,441 footprints and 9,875 segments.
- Copper zones on 991 boards, so the stripping rule in issue 5 has to handle zones on almost every board.
- 14 boards have no board outline at all (zero area). They cannot be routed sensibly and fail DRC with an invalid outline, so the filter drops them.
- 5 boards have no tracks, 180 have unconnected items in the human reference. 976 boards have tracks and zero unconnected items, which is the pool before the DRC filter in issue 4.
