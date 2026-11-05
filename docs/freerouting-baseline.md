# Freerouting baseline on the test boards

Pilot of 10 boards, three attempts each, 5 November 2026.

## Protocol

The problem is the mirror's unrouted.dsn: the human board with every track, via and copper zone removed, zones included and not only their fills, with no planes in the DSN, so the refill changes nothing on a routed board. 69 of the 97 clean test boards have zones in the human reference, so the human columns are a reference, not a like for like comparison.

Freerouting 2.1.0 (jar built 12 April 2025) runs headless and is killed after 600 seconds. The pass limit option is passed as 20 but this version ignores it without the GUI (the stop pass number is fixed at 999 in the bytecode); 4 of 30 pilot attempts ran past 20 passes. Freerouting has no seed, so attempts differ. Each board is routed three times; attempt 1 is the single shot number and best of three mirrors PCBWorld's best of five.

Sessions are imported into unrouted.kicad_pcb, and scripts/metrics.py scores routed attempts and the human raw.kicad_pcb alike: KiCad 10.0.4's design rule check, zones refilled, all severities. Complete means zero unconnected items; a clean pass adds zero errors on the copper whitelist of scripts/drc_batch.py.

## Pilot

The first 10 clean test boards (8 easy, 1 medium, 1 hard) took 361 seconds for 30 attempts, Freerouting median 4.1 seconds, maximum 24.5, no timeouts or failures. Complete: 8 of 10 on attempt 1, 9 best of three, 7 on every attempt. Clean pass: 3, 4 and 3. BB PWR 8009 never completed; Brushless ESC completed once, and twice Freerouting's own count said complete while KiCad found open connections.

Attempt 1 has 34 copper violations: 26 copper to edge clearance, 4 shorts, 3 clearance, 1 annular width; the DSN carries no edge clearance rule. Without edge clearance, 7 of 10 are complete and otherwise clean. The human boards score 10 complete and 6 clean. Routed length median 1.008 of human, 37 vias against 80. Human rows match ground_truth.json except C BISCUIT crowbar (one drawing layer segment).

The mirror's 1.9.0, 2.4.1, 2.5.0 RC1 and 2.5.0 RC8 runs left 3, 2, 2 and 3 pilot boards incomplete by Freerouting's count; our attempt 1 left 1 (2 by KiCad). Over all 119 they fully routed 79, 89, 89 and 94; four runs hit the mirror's cap near 300 seconds, so its slowest times are censored. At 36 seconds per board the full run takes about 72 minutes for three attempts, 28 for one.

## Caveats

PCBWorld's numbers (0.80 clean on easy, 0.78 on medium) came from a KiCad 9 engine with patched severities, best of five attempts and five boards failing to load. 22 of the 119 test boards fail the human reference filter after refill, 14 on connectivity; the 97 clean boards are the primary set.

Issue 6: the Uart programmer board would not export a DSN because three logo footprints share the reference G***; renamed, it exports in one second.
