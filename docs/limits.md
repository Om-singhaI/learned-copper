# What this model knows and does not know

Written October 10, 2026, after a discussion of whether a copper predictor trained on hobby boards has any real use.

## The honest statement

A pixel model trained on human routed boards learns geometric habit: where traces tend to go given pads, nets and obstacles. It does not learn why. Trace width chosen for current, impedance control, differential pairs that must travel together, length matching, return paths under fast signals, creepage for high voltage, thermal relief and EMI are not visible in a copper raster. On its own the model is a geometry prior for two layer boards, and the report describes it as exactly that.

## Where electronics knowledge lives in every router

No router has electronics knowledge of its own. Freerouting, KiCad's interactive router and PCBWorld's learned agent are geometry engines; the knowledge enters through the design rules the engineer wrote (net class widths and clearances, differential pair rules, length tuning) and through the placement the engineer chose. Quilter's physics is a separate scoring stage over candidate layouts. OmniRouting showed that language models given the full schematic still route 12.6 percent of nets correctly. The question for any router is how much of the constraint knowledge gets in, not whether it reasons about circuits.

## What is added to close part of the gap

1. Net attribute channels (Sprint 3). The board file carries net class width and clearance for every net, and net names carry meaning: GND, VCC, USB_D+, USB_D-, CLK. The raster gains per net channels for width class, power versus signal and differential pair membership, so the model can learn that wide nets take short wide paths and pairs stay together, because the human boards show it.
2. Net ordering and priority (Sprint 4 stretch, replacing Freerouting integration). Routers are sequential and the order decides which nets get clean paths. A ranking model over the netlist graph predicts route order and layer preference from net attributes and connectivity. This is learning to rank, sits above the pixels, and does not pretend to simulate physics.
3. Physics feedback (signal integrity, thermal, EMI as a scoring loop) is out of scope. It is a company, not a term project.

## The regime

The model is trained on thousands of boards and tested on 119 from repositories it never saw, so it is not tied to a board. It is tied to a regime: two layer hobby boards, the same regime PCBWorld evaluates in. Claims stay inside it. The claim is that a learned prior from human copper makes a classical router complete more boards with less wire and fewer vias under KiCad's design rule check. That is a real result if it holds, and it has not been published.
