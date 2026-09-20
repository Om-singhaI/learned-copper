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
