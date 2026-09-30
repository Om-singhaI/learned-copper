# Related work: how learned PCB routing and layout models are trained

Survey completed September 30, 2026. Every paper below was opened at the URL given and every count is quoted from the paper, its repository or its dataset card. Entries that could not be fully opened say so.

## The short answer

No published system trains a copper predicting model on thousands of real, human routed boards. Academic routers train on synthetic grids or on instances labelled by a classical router, with small models on a single GPU. Industrial routers (DeepPCB, Quilter, Flux) train by self play in private simulators and publish neither papers nor data; Quilter states outright that it never learns from human boards. The one open real board corpus that papers use is PCBench, whose author proposed generative routing research on it in his 2024 dissertation, and PCBWorld uses 679 of its boards for evaluation only while training on synthetic data. The largest real board corpus in any paper, OmniRouting's 1,681 Eagle designs, is used only to benchmark frozen language models and has not been released. Learned Copper therefore sits in an open gap: a supervised model of human copper on real boards, scored with the same completion, wire length, via and DRC metrics that PCBWorld established, against the same Freerouting baseline.

## Comparison of systems

| System | Year | Task | Method | Training data | Evaluation data | Data released | Code |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PCBWorld (LG AI Research) | 2026 | routing agent inside KiCad | decoder only Transformer, PPO and GRPO, DRC as reward | 10,000 synthetic 2 layer boards with 4 to 6 nets (D2), plus 10,000 grid boards (D1); no human copper | 128 synthetic plus 679 real PCBench boards zero shot | generators and split file, real boards rebuilt from PCBench | yes, BSD 3 |
| DeepPCB (InstaDeep) | 2019 to 2026 | RL autorouter | RL in a proprietary simulator | "millions of routing attempts across thousands of board configurations", synthetic plus a private corpus | vendor reported: 95.6 percent of 27,721 customer boards at 95 percent completion | no | plugin only |
| Quilter | 2023 to 2026 | RL place and route | self play scored by physics checks | "millions of synthetic boards"; never human designed boards | none public | no | no |
| Flux AI Auto Layout | 2024 | RL router | RL, no details | "thousands of professionally routed boards", no counts | none public | no | no |
| FanoutNet (AAAI 2023) | 2023 | BGA fanout, A* draws copper | CNN plus attention policy, PPO, router success as reward | 11 open boards (ASP DAC set) plus 5 Huawei 6 layer boards | same boards | 11 public, 5 private | no |
| He and Bao (Iowa State) | 2020 | grid routing | MCTS with a small CNN rollout policy | 2,000 random 30 by 30 circuits, 9,459 samples routed by a maze router | 30 generated circuits | no | no |
| Ranking Cost | 2021 | grid routing | per instance cost maps plus net order, evolution strategies | 300 random grid maps, 16 by 16 to 64 by 64 | same generator | no | yes |
| Liao et al. DQN | 2019 | global routing | DQN on 8 by 8 by 2 grids | synthetic problem generator | synthetic | generator promised | unverified |
| Attention Routing (Cadence) | 2020 | analog track assignment | attention encoder decoder, REINFORCE | real sub 16 nm analog placements, private | held out 20 percent | no | no |
| Jain and Okabe | 2017 | copper per pixel per layer | 15 stage fully convolutional network | 50,000 synthetic samples | 10,000 synthetic | generator | yes |
| PRNet (NeurIPS 2022) | 2022 | chip routing as image generation | conditional GAN | 30K to 400K instances derived from ISPD 2007 benchmarks, router labelled | ISPD 98 | via public benchmarks | yes |
| HubRouter (NeurIPS 2023) | 2023 | chip global routing | GAN, VAE or diffusion hub generator plus actor critic connector | about 240K router labelled ISPD 2007 instances | ISPD 2007 and ISPD 98 | via public benchmarks | yes |
| DSBRouter (ICML 2025) | 2025 | chip global routing | diffusion Schrodinger bridge from pin maps to route maps | about 300K samples routed by nthurouter on ISPD 2007 | ISPD 98 and ISPD 2007 | via public benchmarks | yes |
| Unet Astar (IEEE Access 2023) | 2023 | PCB routing regions | deeper UNet proposes regions for a unified A* router | synthetic problem generator, counts in full text | generated cases, about 70 percent faster | code repo is a placeholder | no |
| TRouter (TCAD 2023) | 2023 | thermal aware PCB routing | crisscross attention CNN predicts thermal maps from rasterised layouts | 11 ASP DAC boards plus 4 new boards, 0.1 mm grid | same boards, KiCad DRC | benchmarks yes | no |
| GPCB (TCAD 2025) | 2025 | PCB routing as token prediction | GPT style model over network flow encodings | learns from human expert routing, source and count not disclosed | own test set; on OmniRouting's real boards NRR 0.69 percent | no | no |
| RouteNet (ICCAD 2018) | 2018 | chip routability, DRC hotspots | convolutional network on placement features | placements generated from ISPD 2015 inputs | held out placements | inputs public | no |
| Painting on Placement (DAC 2019) | 2019 | chip congestion maps | conditional GAN, congestion as image translation | placement features and router congestion labels | held out designs | benchmarks public | no |
| PROS (ICCAD 2020) | 2020 | chip congestion | fully convolutional predictor inside a commercial flow | placement results labelled by global routing | held out; DRC violations down 11.65 percent | no | no |
| CLDRoute | 2026 | chip DRC and congestion maps | conditional latent diffusion with a UNet denoiser | CircuitNet 2.0: N28 7,872 train, N14 10,368 train | CircuitNet test splits | public dataset | yes |
| AlphaChip (Nature 2021) | 2021 | chip macro placement | GNN policy, PPO, supervised reward pretraining | 20 proprietary TPU blocks plus 10,000 placements | 5 unseen TPU blocks | no | partial |
| RL_PCB (DATE 2024) | 2024 | PCB placement | per component TD3 and SAC agents | 6 KiCad circuits, 3 to 12 components | 3 unseen circuits | yes | yes |
| Yonekura and Echigo | 2026 | PCB placement completion | GNN retrieval plus WGAN | 200 expert layouts | same | no | no |
| SchGen (Microsoft) | 2026 | schematic generation | LoRA on GPT oss 20B, about 21 GPU hours | 2,105 human corrected SparkFun schematics, quadrupled to 8,420 | 500 held out plus 988 GitHub samples | yes, MIT | yes |

## Learned autorouters

Academic learned routers almost all train on synthetic or router labelled instances with modest compute. Liao et al. train one DQN per generated 8 by 8 by 2 problem on a GTX 1080 Ti. He and Bao fit a one convolution layer CNN on 9,459 samples from 2,000 random 30 by 30 circuits, routed by a maze router or by hand, and use it as the rollout policy inside Monte Carlo tree search. Ranking Cost optimises per instance cost maps plus a net order with evolution strategies on 300 random grid maps. Unet Astar trains a UNet on a parameterised problem generator to propose routing regions for an A* router and reports about 70 percent faster routing. None of these touches a real board file.

PCBWorld is the first academic system evaluated on real boards at scale, and it still trains only on synthetic data. Its policy is a four layer, 128 wide decoder only Transformer trained from scratch with PPO on 10,000 synthetic two layer boards of 4 to 6 nets, with KiCad's design rule checker as the reward, one NVIDIA L40 per run, 12 to 22 hours per seed. Zero shot on real PCBench boards it passes 86 percent of the 99 small boards cleanly against Freerouting's 80 percent, but only 45 percent of the 10 medium boards against Freerouting's 78 percent, and the authors call large boards an open challenge. The lesson for Learned Copper is that the synthetic to real gap is where their approach breaks, and human copper is the signal they never used.

The industrial routers are the opposite: real boards, no disclosure. DeepPCB describes "millions of routing attempts across thousands of board configurations", some synthetic and some from a private training corpus, in a simulator that Google Cloud's engineering post says was accelerated 15 to 235 times on TPUs; its public numbers are self reported completion rates on 27,721 customer uploads. Quilter says it "learned layout by playing millions of synthetic boards against the laws of physics" and is "never trained on human designed boards". Flux says its router is "trained on thousands of professionally routed boards" and nothing more. GPCB, from a group with Huawei co authors, learns token encodings of human routing but does not disclose its boards, and when OmniRouting ran it on 1,681 real boards it completed 0.69 percent of nets with shorts on 99 percent of boards.

The only learned router trained on real boards with numbers we can check is FanoutNet, which trains PPO on 16 boards (11 open, 5 Huawei HDI boards) but learns only the fanout decision while a 3 D A* draws every trace.

## Supervised copper, routability and congestion prediction

This is the family Learned Copper belongs to, and it grew up in chip design rather than PCB. Jain and Okabe (2017) trained a 15 stage fully convolutional network to emit copper per pixel per layer from pin locations, on 50,000 synthetic samples, reaching F1 of 92 percent on validation, with no connectivity check. That is the direct ancestor of a UNet copper predictor. PRNet (NeurIPS 2022) made the same idea generative with a conditional GAN that draws each net's route as an image, trained on 30K to 400K instances derived from the ISPD 2007 benchmarks and labelled by a classical router. HubRouter (NeurIPS 2023) diagnosed the weakness of image generated routes, broken connectivity, and fixed it with a second stage that connects pins through generated hubs, training on about 240K router labelled instances. DSBRouter (ICML 2025) replaced the GAN with a diffusion bridge from pin maps to route maps on about 300K samples. The pattern across all three is rasterise, generate, then repair; Learned Copper's router guidance step plays the repair role.

Routability prediction is the same input with a coarser output. RouteNet (ICCAD 2018) predicts DRC hotspot locations from placement features with a convolutional network, Painting on Placement (DAC 2019) forecasts congestion maps with a conditional GAN as an image translation problem, and PROS (ICCAD 2020) puts a fully convolutional congestion predictor inside a commercial flow and cuts DRC violations by 11.65 percent. CircuitNet (2022, 2.0 at ICLR 2024) turned this into an open benchmark: 10,242 layouts from six RISC V designs at 28 nm and more than 10,000 at 14 nm, each a stack of per tile feature maps paired with congestion, DRC and IR drop label maps, so that prediction "is formulated as an image to image translation task". Its baselines are fully convolutional networks and a UNet; the 2024 Inception boosted UNet and the 2026 CLDRoute latent diffusion model are the current state of that benchmark. On the PCB side the only comparable work is TRouter (TCAD 2023), which rasterises components, pads, vias and segments per layer at 0.1 mm and trains a crisscross attention CNN to predict thermal maps on 15 open boards, and a 2025 Chinese journal paper that trains a conditional GAN to generate PCB routing as images on a self constructed, undisclosed dataset. No PCB paper trains a copper or routability predictor on more than a few dozen real boards.

## Placement

Placement work matters here because it shares the data pipelines and the labels. Chip placement trains either per circuit online (MaskPlace on 24 public circuits, DeepPlace on 8 ISPD 2005 circuits, EfficientPlace, all on one consumer GPU), or by pretraining on a pool then fine tuning (AlphaChip on 20 proprietary TPU blocks with 48 hours of pretraining across 16 GPUs, plus a supervised set of 10,000 placements), or supervised from solver outputs and synthetic generators (ChiPFormer, TransPlace, the Berkeley diffusion placer on 40,000 plus 5,000 fully synthetic circuits, MacroDiff+ on 8,000 augmented netlists labelled by DREAMPlace). On the PCB side RL_PCB trains TD3 and SAC agents on six KiCad circuits and tests on three, DRLPlace and PCBAgent train on undisclosed industrial boards, and the only PCB placement paper that learns from human layouts, Yonekura and Echigo (2026), uses 200 expert layouts. NS Place, which is not learned, established the evaluation habit we reuse: place, route with Freerouting, check with KiCad DRC, report routed wire length, vias and violations against the manual design.

## Generative and language model layout

Since 2025 the volume of work is in benchmarks over frozen language models, and none of it trains a model that emits copper. OmniRouting and OmniLayout (2026) collected 1,681 schematic coupled Eagle designs from SparkFun, Adafruit, Arduino, Seeed, ProtoCentral and GitHub, with 77,242 placements, 168,815 pads and 494,349 human wire segments, and used them to score multimodal models: the best reaches 12.6 percent net routability against 93.6 percent for the human reference, with 70 to 431 DRC violations per board against 0.8. PCB Bench (ICLR 2026) and EDA Bench (ICML 2026 workshop) reach similar conclusions on 174 and 35 projects. The trained models in this family generate schematics or netlists rather than boards: SchGen fine tunes GPT oss 20B on 2,105 human corrected SparkFun schematics, CircuitFormer trains a 511M parameter transformer on 18,011 SPICE netlists, and several KiCad LoRA models on Hugging Face train on about 100,000 netlist examples, roughly 45,000 of them derived from open schematics. Reports of a September 2026 OpenAI demo placing and routing a small amplifier board in KiCad exist only as a clip and one third party reproduction, with no training details.

## Dataset index

| Dataset | Contents | Size | Domain | Real or synthetic | Public and license | Used by |
| --- | --- | --- | --- | --- | --- | --- |
| PCBench | KiCad boards with human copper, per board metadata and source URL | 1,182 board folders, 1,194 designs from 846 repositories | PCB | real | yes, repository MIT, per board licenses recorded for 563, none for the rest | PCBWorld (evaluation), Learned Copper |
| Freerouting PCBench mirror | raw and unrouted board pairs, DSNs, ground truth, Freerouting results for 9 versions | 1,157 boards | PCB | real | yes, research use, board licenses as upstream | Learned Copper |
| PCBWorld D3 | tiered lists over PCBench boards that survive KiCad 9 conversion and DRC | 679 boards, tests of 99, 10 and 10 | PCB | real | split file BSD 3, boards rebuilt locally | PCBWorld, Learned Copper test set |
| PCBWorld D1 and D2 | synthetic grid and gridless boards | 10,000 train, 128 validation, 128 test each | PCB | synthetic | yes, generators BSD 3 | PCBWorld |
| OmniRouting and OmniLayout | 1,681 Eagle designs with schematics, placements and human routing | 494,349 wire segments, 84,603 vias | PCB | real | not released; license stated inconsistently | OmniRouting, OmniLayout |
| ASP DAC PCB benchmarks | KiCad boards bm1 to bm11 | 11 boards, 14 in the NS Place paper | PCB | real | repository without a license | PcbRouter, NS Place, TRouter, FanoutNet |
| TRouterBM | unrouted KiCad boards d1 to d4 | 4 boards | PCB | real | repository without a license | TRouter |
| RL_PCB | small KiCad circuits for placement | 9 circuits, 6 train and 3 test | PCB | real topologies | yes, MIT | RL_PCB |
| tscircuit sets | Simple Route JSON problems from Arduino and Antmicro boards, BGA breakouts, one million synthetic problems | 16, 200 and 1,000,000 | PCB | mixed | yes, mixed MIT and unlicensed | tscircuit benchmarks |
| open schematics | schematics with the PCB files of the same project | 87,931 records, 60,280 with a KiCad board, 21,039 projects, 317 GB | PCB | real | yes, CC BY 4.0 on the compilation | KiCad LoRA models, Learned Copper stage 3 |
| Warren PCB Layout | photograph and layout image pairs | 668 pairs | PCB images | real | yes, no license stated | none |
| CircuitNet 1.0 and 2.0 | per tile feature maps with congestion, DRC and IR drop labels | 10,242 layouts at 28 nm, more than 10,000 at 14 nm, 568 GB | ASIC | real flows on open designs | yes, BSD 3 | Inception UNet, CLDRoute, many |
| CircuitNet 3.0 | RTL, netlists, timing and power | 8,659 designs, 15,863 instances | ASIC | real | yes, Apache 2.0 | timing and power prediction |
| ISPD 98, 2005, 2007, 2015, 2018, 2019, 2024 and ICCAD contests | chip placement and routing benchmarks | varies | ASIC | real netlists | yes | PRNet, HubRouter, DSBRouter, MaskPlace, RouteNet and others |
| DAC 2012 | routability driven placement suite | 8 superblue circuits | ASIC | real | historically yes | TransPlace and others |
| SchGen dataset | schematic generation pairs | 8,420 rows | schematics | human corrected | yes, MIT | SchGen |

## Systems that trained on private data and said so

DeepPCB (private corpus plus synthetic), Quilter (private synthetic boards), Flux (thousands of professionally routed boards, owner not stated), Cadence's Attention Routing (sub 16 nm analog placements), FanoutNet (5 Huawei HDI boards alongside 11 public ones), GPCB (human routing of undisclosed origin), AlphaChip (20 TPU blocks and a 10,000 placement reward set), PCBAgent (17 industrial tasks), DRLPlace (undisclosed boards), the Trieste and Infineon analog floorplanner (6 industrial circuits), MediaTek's multi objective placement work, and Yonekura and Echigo (200 expert layouts).

## Implications for Learned Copper

The closest training recipes are the chip side image generators. Rasterisation at a fixed grid with one channel per layer for pads, one for obstacles and one target channel per layer is exactly how Jain and Okabe, PRNet and CircuitNet frame the problem; TRouter shows the PCB version at 0.1 mm per pixel with 2L plus 2 input channels for L layers. Our 0.2 mm raster of outline, pads and keepouts per layer with copper per layer as the target is in that family, and intersection over union on trace pixels is the metric Jain and Okabe reported as F1. The loss that worked for them is per pixel cross entropy; PRNet and HubRouter add adversarial or diffusion objectives, which we do not need for a first result.

Every image generation router in this lineage learned the same lesson: generated copper is not connected copper. HubRouter and DSBRouter spend their second stage repairing connectivity, and OmniRouting's benchmark shows that models which draw plausible traces can still short 99 percent of boards. Learned Copper's design, prediction as a cost map for a real router with KiCad DRC as the judge, is the same repair idea applied with a classical router instead of a learned connector, which is cheaper and easier to trust.

On data, the field has been training on synthetic grids because nobody assembled the real corpus. PCBench's author wrote that the dataset was built to enable generative PCB routing research; PCBWorld used it for evaluation only; OmniRouting collected 1,681 human routed boards and used them only to test frozen models. A model trained on the PCBench pool and scored on PCBWorld's split, against Freerouting's 80 and 78 percent and PPO's 86 and 45 percent, is the experiment those papers set up and did not run.

On compute, the academic precedents used one consumer or datacenter GPU: a GTX 1080 Ti, RTX 3090s, one L40 for 12 to 22 hours, one A5000 for a 233K parameter diffusion placer. A free T4 with a capped raster and subset is within that range for a UNet of a few million parameters.

Two cautions carry over. The benchmark boards are small: PCBWorld's main table covers 99 boards of at most 31 pads and 10 of at most 100, and the pool is 89 percent two layer hobby boards, so claims stay within that regime. And comparability with PCBWorld's numbers is conditional on rule handling: their DRC ran through a KiCad 9 engine with patched severities and a best of five protocol, so we quote their table as the reference and rerun Freerouting under our own KiCad 10 pipeline for the direct comparison.

Papers to cite in the report: PCBWorld, PCBench and the He dissertation, OmniRouting, Jain and Okabe, PRNet, HubRouter, DSBRouter, CircuitNet and CLDRoute, RouteNet, Painting on Placement, PROS, TRouter, FanoutNet, NS Place, and the industrial statements from DeepPCB, Quilter and Flux.

## References

1. PCBWorld: A Benchmark Environment for Engine-Grounded PCB Design Automation. arXiv 2607.05915, 2026. https://arxiv.org/abs/2607.05915
2. Automation of PCB autorouting via world-model reinforcement learning and freerouting integration (DreamerV3+FR). Expert Systems with Applications, vol. 311, article 131424, 2026. https://doi.org/10.1016/j.eswa.2026.131424
3. Attention Routing: track-assignment detailed routing using attention-based reinforcement learning. arXiv 2004.09473, 2020. https://arxiv.org/abs/2004.09473
4. Ranking Cost: Building An Efficient and Scalable Circuit Routing Planner with Evolution-Based Optimization. arXiv 2110.03939, 2021. https://arxiv.org/abs/2110.03939
5. Circuit Routing Using Monte Carlo Tree Search and Deep Neural Networks (arXiv); published as 'Circuit Routing Using Monte Carlo Tree Search . arXiv 2006.13607, 2020. https://arxiv.org/abs/2006.13607
6. Two-stage PCB Routing Using Polygon-based Dynamic Partitioning and MCTS. DATE 2023, 2023. https://past.date-conference.com/proceedings-archive/2023/DATA/603.pdf
7. Towards automated PCB routing: Leveraging machine learning and heuristic techniques (PhD dissertation). PhD dissertation, Iowa State University, 2024. https://doi.org/10.31274/td-20240617-74
8. PCBench: A Dataset for Printed Circuit Board Routing. GitHub repository, 2024. https://github.com/PCBench/PCBench
9. FanoutNet: A Neuralized PCB Fanout Automation Method Using Deep Reinforcement Learning. AAAI-23, Proceedings of the AAAI Conference on Artificial Intelligence, 2023. https://ojs.aaai.org/index.php/AAAI/article/view/26030
10. A Deep Reinforcement Learning Approach for Global Routing. arXiv 1906.08809, 2019. https://arxiv.org/abs/1906.08809
11. DeepPCB (InstaDeep) - RL cloud autorouter: public sources = Google Cloud TPU/Sebulba post (2023), deeppcb.ai technical posts (2025-2026), Ra. company blog / Google Cloud blog, 2019. https://deeppcb.ai/reinforcement-learning-pcb-routing-explained/
12. Quilter - physics-driven reinforcement-learning PCB layout (placement + routing candidates). company documentation / technology page / press release, 2024. https://quilter.ai/technology
13. Flux.ai AI Auto-Layout (one-click routing in the Flux browser EDA). company blog 'Introducing AI Auto-Layout', 2024. https://flux.ai/p/blog/introducing-ai-auto-layout
14. JITX - code-defined PCB design automation. IEEE Spectrum, 2018. https://spectrum.ieee.org/startup-jitx-uses-ai-to-automate-complex-circuit-board-design
15. Celus Engineering Platform (schematic generation, component selection, PCB floorplan). TechCrunch, 2022. https://techcrunch.com/2022/07/06/celus-which-uses-ai-to-automate-circuit-board-design-raises-25-6m/
16. Cadence Allegro X AI (Placement AI, Copper AI, Route AI). vendor page, 2023. https://www.ema-eda.com/allegro-x-ai/
17. Zuken AIPR (Autonomous Intelligent Place and Route) with Smart Autorouter and 'Basic Brain' for CR-8000. Zuken press release, 2023. https://www.zuken.com/en/resource/zuken-introduces-ai-powered-place-and-route-technology/
18. Sable: a Performant, Efficient and Scalable Sequence Model for MARL (with the Jumanji Connector environment) - InstaDeep baselines used in P. arXiv 2410.01706, 2024. https://arxiv.org/abs/2410.01706
19. OmniRouting: A Semantic-Coupled Multimodal Benchmark for Constraint-Aware Spatial Reasoning in PCB Routing. arXiv 2608.04434, 2026. https://arxiv.org/abs/2608.04434
20. PCB-Bench: Benchmarking LLMs for Printed Circuit Board Placement and Routing. ICLR 2026 poster, 2026. https://iclr.cc/virtual/2026/poster/10009621
21. . NeurIPS 2022, 2022. https://proceedings.neurips.cc/paper_files/paper/2022/hash/a8b8c1ad51df1b93d9e3d1fca75debbf-Abstract.html
22. (foundational supervised copper prediction). arXiv 1706.08948, 2017. https://arxiv.org/abs/1706.08948
23. (supervised PCB routability prediction). MIT M.Eng. thesis, September 2020, 2020. https://dspace.mit.edu/handle/1721.1/129238
24. (PCB-specific RL with open code and data). DATE 2024, pp. 1-6, DOI 10.23919/DATE58400.2024.10546526, 2024. https://github.com/LukeVassallo/RL_PCB
25. (2026 survey). arXiv 2606.17074, 2026. https://arxiv.org/abs/2606.17074
26. A graph placement methodology for fast chip design (AlphaChip; arXiv preprint: Chip Placement with Deep Reinforcement Learning). Nature 594, 207-212, 2021. https://arxiv.org/abs/2004.10746
27. Placement Optimization with Deep Reinforcement Learning. ISPD 2020, 2020. https://arxiv.org/abs/2003.08445
28. Assessment of Reinforcement Learning for Macro Placement (+ 'An Updated Assessment of Reinforcement Learning for Macro Placement', IEEE TCAD. ISPD 2023, DOI 10.1145/3569052.3578926, 2023. https://github.com/TILOS-AI-Institute/MacroPlacement
29. GoodFloorplan: Graph Convolutional Network and Reinforcement Learning-Based Floorplanning. IEEE TCAD vol. 41 no. 10, pp. 3492-3502, DOI 10.1109/TCAD.2021.3131550, 2022. https://doi.org/10.1109/TCAD.2021.3131550
30. Floorplanning with Edge-aware Graph Attention Network and Hindsight Experience Replay. ACM TODAES vol. 29 no. 3, Article 56, pp. 1-17, DOI 10.1145/3653453, 2024. https://www.cse.cuhk.edu.hk/~byu/papers/J105-TODAES2024-RLFloorplan.pdf
31. CORE: Collaborative Optimization with Reinforcement Learning and Evolutionary Algorithm for Floorplanning. NeurIPS 2025, 2025. https://neurips.cc/virtual/2025/poster/119653
32. Effective Analog ICs Floorplanning with Relational Graph Neural Networks and Reinforcement Learning. DATE 2025, 2024. https://arxiv.org/abs/2411.15212
33. FloorSet - a VLSI Floorplanning Dataset with Design Constraints of Real-World SoCs. arXiv:2405.05480, 2024. https://arxiv.org/abs/2405.05480
34. DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration for Modern VLSI Placement. DAC 2019, 2019. https://github.com/limbo018/DREAMPlace
35. On Joint Learning for Solving Placement and Routing in Chip Design (DeepPlace / DeepPR). NeurIPS 2021, 2021. https://arxiv.org/abs/2111.00234
36. The Policy-gradient Placement and Generative Routing Neural Networks for Chip Design (PRNet). NeurIPS 2022, 2022. https://proceedings.neurips.cc/paper_files/paper/2022/hash/a8b8c1ad51df1b93d9e3d1fca75debbf-Abstract.html
37. MaskPlace: Fast Chip Placement via Reinforced Visual Representation Learning. NeurIPS 2022, 2022. https://arxiv.org/abs/2211.13382
38. ChiPFormer: Transferable Chip Placement via Offline Decision Transformer. ICML 2023, PMLR 202:18346-18364, 2023. https://proceedings.mlr.press/v202/lai23c.html
39. Macro Placement by Wire-Mask-Guided Black-Box Optimization (WireMask-BBO). NeurIPS 2023, 2023. https://arxiv.org/abs/2306.16844
40. Reinforcement Learning within Tree Search for Fast Macro Placement (EfficientPlace). ICML 2024, PMLR 235:15402-15417, 2024. https://proceedings.mlr.press/v235/geng24b.html
41. Flexible Multiple-Objective Reinforcement Learning for Chip Placement. DAC 2022 Late Breaking Results, 2022. https://arxiv.org/abs/2204.06407
42. Chip Placement with Diffusion Models. arXiv:2407.12282, 2024. https://arxiv.org/abs/2407.12282
43. TransPlace: Transferable Circuit Global Placement via Graph Neural Network. KDD 2025, 2025. https://arxiv.org/abs/2501.05667
44. See it to Place it: Evolving Macro Placements with Vision-Language Models (VeoPlace). arXiv:2603.28733, 2026. https://arxiv.org/abs/2603.28733
45. Learning Circuit Placement Techniques Through Reinforcement Learning with Adaptive Rewards (RL_PCB). DATE 2024, DOI 10.23919/DATE58400.2024.10546526, 2024. https://www.um.edu.mt/library/oar/handle/123456789/131782
46. Physically Constrained PCB Placement Using Deep Reinforcement Learning (M.Eng. thesis). MIT M.Eng. thesis, DSpace handle 1721.1/139247, 2021. https://dspace.mit.edu/handle/1721.1/139247
47. Component Centric Placement Using Deep Reinforcement Learning. arXiv:2602.23540, 2026. https://arxiv.org/abs/2602.23540
48. DRLPlace: A Deep Reinforcement Learning-based Irregular and High-Density Printed Circuit Board Placement Method. ASP-DAC 2026, DOI 10.1109/ASP-DAC66049.2026.11420655, 2026. https://doi.org/10.1109/ASP-DAC66049.2026.11420655
49. Optimized PCB Module Automatic Layout Using Deep Reinforcement Learning with Enhanced State Encoding and Neural Network Design (and 2024 pre. International Journal of High Speed Electronics and Systems, DOI 10.11, 2025. https://doi.org/10.1142/S0129156425405789
50. Cypress: VLSI-Inspired PCB Placement with GPU Acceleration (+ open PCB placement benchmark suite). ISPD 2025, DOI 10.1145/3698364.3705346, 2025. https://www.csl.cornell.edu/~zhiruz/pdfs/cypress-ispd2025.pdf
51. Net Separation-Oriented Printed Circuit Board Placement via Margin Maximization (NS-Place). ASP-DAC 2022, DOI 10.1109/ASP-DAC52403.2022.9712480, 2022. https://arxiv.org/abs/2210.14259
52. Clustering Algorithms for Component Placement in Printed Circuit Boards (M.Eng. thesis). MIT M.Eng. thesis, DSpace handle 1721.1/164641, 2025. https://dspace.mit.edu/handle/1721.1/164641
53. CircuitNet (2022) and CircuitNet 2.0 (ICLR 2024): open datasets for routability / congestion / DRC / IR-drop prediction. SCIENCE CHINA Information Sciences 2022, 2022. https://arxiv.org/abs/2208.01040
54. Graph-Based Reinforcement Learning Approach for Multi-Power-Domain PCB PDN Shape and Stackup Synthesis. 2025 IEEE International Symposium on Electromagnetic Compatibility, Si, 2025. https://scholarsmine.mst.edu/ele_comeng_facwork/7329
55. DeepPCB (InstaDeep) - commercial learned routing system (no paper). product site, 2020. https://deeppcb.ai
56. Quilter - commercial RL + physics PCB place-and-route (no paper). company blog, 2023. https://www.quilter.ai/blog/pcb-autorouting-in-2026-a-review-of-traditional-tools-vs-quilters-ai-approach
57. LaMPlace: Learning to Optimize Cross-Stage Metrics in Macro Placement . ICLR 2025, 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/04c0399a47ee4107cd03b08f1f8c3eeb-Abstract-Conference.html
58. RoutePlacer: An End-to-End Routability-Aware Placer with Graph Neural Network . KDD 2024, 2024. https://arxiv.org/abs/2406.02651
59. Delving into Macro Placement with Reinforcement Learning . MLCAD 2021, 2021. https://arxiv.org/abs/2109.02587
60. Automatic PCB Component Placement via GNN-Guided Similarity Retrieval Coupled with GAN-Based Completion . 2026 International Conference on Electronics Packaging and Hybrid Bond, 2026. https://doi.org/10.23919/ICEP-HBS69241.2026.11550476
61. OmniLayout: A Schematic-Coupled Multimodal Benchmark for Constraint-Aware Geometric Reasoning in PCB Layout. arXiv 2607.03261, 2026. https://arxiv.org/abs/2607.03261
62. OmniSch: A Multimodal PCB Schematic Benchmark For Structured Diagram Visual Reasoning. arXiv 2604.00270, 2026. https://arxiv.org/abs/2604.00270
63. SchGen: PCB Schematic Generation with Semantic-Grounded Code Representations. arXiv preprint 2605.30345, 2026. https://arxiv.org/abs/2605.30345
64. pcbGPT: Automatic PCB Schematic Synthesis from Natural Language Requirements. arXiv preprint 2606.01188, 2026. https://arxiv.org/abs/2606.01188
65. PCBSchemaGen: Reward-Guided LLM Code Synthesis for Printed Circuit Boards (PCB) Schematic Design with Structured Verification. arXiv 2602.00510, 2026. https://arxiv.org/abs/2602.00510
66. EDA Bench: Functional Failure Modes in PCB-Generating Agents. ICML 2026 Workshop 'Failure Modes in Agentic AI: Reproducible Triggers, 2026. https://icml.cc/virtual/2026/77908
67. ChipLingo: A Systematic Training Framework for Large Language Models in EDA (defines a chip-level 'EDA-Bench'). arXiv preprint 2604.27415, 2026. https://arxiv.org/abs/2604.27415
68. HWE-Bench: Can Language Models Perform Board-level Schematic Designs?. arXiv preprint 2603.18102, 2026. https://arxiv.org/abs/2603.18102
69. PCB-QA: Evaluating LLMs over the First Printed Circuit Board Design Question-Answer Dataset. arXiv preprint 2606.23704, 2026. https://arxiv.org/abs/2606.23704
70. PCBench: A Dataset for Printed Circuit Board Routing (GitHub dataset; cited by PCBWorld as a DAC 2024 work-in-progress poster). GitHub, 2023. https://github.com/PCBench/PCBench
71. CircuitNet 3.0: A Multi-Modal Dataset with Task-Oriented Augmentation for AI-Driven Circuit Design. ICLR 2026, 2026. https://github.com/sklp-eda-lab/iclr-circuitnet_3.0/
72. Teaching a Small LLM to Design Electronic Circuits: Fine-Tuning Qwen3-4B on 100K KiCad Netlists (blog + HF model/dataset). Blog post, 2026. https://blog.abijah.me/teaching-a-small-llm-to-design-electronic-circuits-fine-tuning-qwen3-4b-on-100k-kicad-netlists
73. open-schematics (Hugging Face dataset). Hugging Face dataset, 2025. https://huggingface.co/datasets/bshada/open-schematics
74. STEM-AI-mtl/phi-2-electrical-engineering (Hugging Face model). Hugging Face, 2024. https://huggingface.co/STEM-AI-mtl/phi-2-electrical-engineering
75. empulse/KiCAD-MCP-Qwen3.5-4B-GGUF (Hugging Face model). Hugging Face, 2026. https://huggingface.co/empulse/KiCAD-MCP-Qwen3.5-4B-GGUF
76. Ailiance-fr/devstral-kicad-pcb-lora and the mascarade-kicad / kicad9plus datasets (Hugging Face). Hugging Face, 2026. https://huggingface.co/Ailiance-fr/devstral-kicad-pcb-lora
77. bretbouchard/kicad-agent-pcb-adapter (Hugging Face model). Hugging Face, 2026. https://huggingface.co/bretbouchard/kicad-agent-pcb-adapter
78. niko3x/kicadrouterai (Hugging Face). Hugging Face, 2026. https://huggingface.co/niko3x/kicadrouterai
79. CircuitLM: A Multi-Agent LLM-Aided Design Framework for Generating Circuit Schematics from Natural Language Prompts. arXiv 2601.04505, 2026. https://arxiv.org/abs/2601.04505
80. CircuitFormer: A Circuit Language Model for Analog Topology Design from Natural Language Prompt. arXiv preprint 2605.05773, 2026. https://arxiv.org/abs/2605.05773
81. Multiterminal Pathfinding in Practical VLSI Systems with Deep Neural Networks. ACM Transactions on Design Automation of Electronic Systems, 2023. https://doi.org/10.1145/3564930
82. TRouter: Thermal-Driven PCB Routing via Nonlocal Crisscross Attention Networks. IEEE TCAD vol. 42 no. 10, 2023. https://doi.org/10.1109/TCAD.2023.3243544
83. Unet-Astar: A Deep Learning-Based Fast Routing Algorithm for Unified PCB Routing. IEEE Access vol. 11, pp. 113712-113725, 2023. https://doi.org/10.1109/ACCESS.2023.3323589
84. A Generation Method for PCB Routing Based on Generative Adversarial Networks (基于生成对抗网络的PCB布线生成方法). Application of Electronic Technique, 2025. https://chinaaet.com/article/3000174673
85. Hierarchical Automatic Power Plane Generation with Genetic Optimization and Multilayer Perceptron. arXiv preprint 2210.16314, 2022. https://arxiv.org/abs/2210.16314
86. Leveraging Large Language Models for Intelligent Power Electronics PCB Routing Optimization. ECCE Europe 2025, Birmingham, 2025. https://doi.org/10.1109/ECCE-Europe62795.2025.11238369
87. Evaluating LLM-based Workflows for Switched-Mode Power Supply Design. arXiv 2507.10639, 2025. https://arxiv.org/abs/2507.10639
88. Symbol and Footprint Database for Electronic Components by Agentic Recognition and Generation (SFgen / SFnet). arXiv 2607.19767, 2026. https://arxiv.org/abs/2607.19767
89. New Interaction Paradigm for Complex EDA Software Leveraging GPT (SmartonAI). arXiv 2307.14740, 2023. https://arxiv.org/abs/2307.14740
90. From Prompt to Prototype: Towards a Frontier LLM Driven RF Engineering Workflow. arXiv preprint 2608.31006, 2026. https://arxiv.org/abs/2608.31006
91. Surveying GenAI-based Automation in Printed Circuit Board Design and Test. arXiv preprint 2606.17074, 2026. https://arxiv.org/abs/2606.17074
92. Graph Neural Networks for Automatic Addition of Optimizing Components in Printed Circuit Board Schematics. ECML PKDD 2025: LNCS 'Machine Learning and Knowledge Discovery in Data, 2025. https://arxiv.org/abs/2506.10577
93. Quilter (industry system) - blog 'LLM PCB layout, GPT-6 Astra'. Company blog, 2026. https://www.quilter.ai/blog/llm-pcb-layout-gpt-6-astra
94. DeepPCB (InstaDeep industry system) - 'KiCad Autorouter Alternative: 27,721 Boards'. Company blog, 2026. https://deeppcb.ai/kicad-autorouter-alternative/
95. GPT-6 Astra PCB layout demo in KiCad (OpenAI), as reported by third parties. Product launch demo, 2026. https://hackaday.com/2026/09/05/can-ai-now-design-pcbs-that-just-work/
96. CLDRoute: Conditional Latent Diffusion for Routability Map Generation in Physical Design. arXiv preprint 2607.16674, 2026. https://arxiv.org/abs/2607.16674
97. Physics-Guided Geometric Diffusion for Macro Placement Generation (MacroDiff+). arXiv 2605.16451, 2026. https://arxiv.org/abs/2605.16451
98. CircuitNet: An Open-Source Dataset for Machine Learning Applications in Electronic Design Automation (N28) plus CircuitNet 2.0 (N14, ICLR 20. SCIENCE CHINA Information Sciences 2022, 2022. https://arxiv.org/abs/2208.01040
99. A Lightweight Inception Boosted U-Net Neural Network for Routability Prediction. arXiv 2402.10937, 2024. https://arxiv.org/abs/2402.10937
100. RouteNet: Routability Prediction for Mixed-Size Designs Using Convolutional Neural Network. ICCAD 2018, 2018. https://research.nvidia.com/sites/default/files/pubs/2018-11_RouteNet:-routability-prediction/a80-xie.pdf
101. ISPD 2018 Initial Detailed Routing Contest and ISPD 2019 Initial Detailed Routing Contest (Cadence). ISPD 2018 and ISPD 2019 contests, 2018. https://www.ispd.cc/contests/18/index.html
102. GPU/ML-Enhanced Large Scale Global Routing Contest (ISPD 2024). ISPD 2024, 2024. https://research.nvidia.com/publication/2024-03_gpuml-enhanced-large-scale-global-routing-contest
103. ICCAD 2019 CAD Contest Problem C: LEF/DEF Based Open-Source Global Router; ICCAD 2023 CAD Contest. ICCAD 2019 / ICCAD 2023, 2019. https://www.iccad-contest.org/2019/problems.html
104. The DAC 2012 routability-driven placement contest and benchmark suite (superblue suite). DAC 2012, DOI 10.1145/2228360.2228500, 2012. https://doi.org/10.1145/2228360.2228500
105. Foundational ASIC benchmark lineage: ISPD98 circuit benchmark suite; ISPD 2005 placement contest; ISPD 2007/2008 global routing contests; MC. ISPD 1998, 1998. https://vlsicad.ucsd.edu/GSRC/bookshelf/Slots/Placement/
106. PCBAgent: An Agent-based Framework for High-Density Printed Circuit Board Placement. ASP-DAC 2025, DOI 10.1145/3658617.3697736, 2025. https://doi.org/10.1145/3658617.3697736
107. Training a Fully Convolutional Neural Network to Route Integrated Circuits (deep-route). arXiv 1706.08948, 2017. https://arxiv.org/abs/1706.08948
108. A Deep Reinforcement Learning Approach for Global Routing; Attention Routing: track-assignment detailed routing using attention-based reinfo. arXiv 1906.08809, 2019. https://arxiv.org/abs/1906.08809
109. HubRouter: Learning Global Routing via Hub Generation and Pin-hub Connection. NeurIPS 2023 main track, 2023. https://papers.neurips.cc/paper_files/paper/2023/hash/f7f98663c516fceb582354ee2d9d274d-Abstract-Conference.html
110. DSBRouter: End-to-end Global Routing via Diffusion Schrodinger Bridge. ICML 2025, PMLR vol. 267 pp. 55065-55081, 2025. https://proceedings.mlr.press/v267/shi25k.html
111. Chip Placement with Deep Reinforcement Learning (Google; Nature 2021 'A graph placement methodology for fast chip design') and An Updated As. arXiv 2004.10746, 2020. https://arxiv.org/abs/2302.11014
112. MaskPlace (NeurIPS 2022), ChiPFormer (ICML 2023), Chip Placement with Diffusion Models (ICML 2025), ChiPBench (2024). arXiv 2211.13382, 2022. https://arxiv.org/abs/2407.12282
113. DeepPCB (InstaDeep) - RL autorouter: 2019 press release, 2026 'What Reinforcement Learning Actually Learns When Routing a PCB', 'PCB Autorou. Press release 28 Nov 2019, 2019. https://deeppcb.ai/reinforcement-learning-pcb-routing-explained/
114. Flux AI Auto-Layout (Flux.ai). Company product page, 2024. https://www.flux.ai/p/pcb-design-software/routing
115. Quilter (physics-driven RL PCB layout). Company product page https://quilter.ai/product, 2024. https://quilter.ai/product
116. open-schematics (bshada) - HuggingFace dataset of KiCad/Altium schematics with paired PCB files. HuggingFace dataset, 2025. https://huggingface.co/datasets/bshada/open-schematics
117. Warren-wzw/PCB-Layout: Aligned PCB-Layout Pairs (HuggingFace). HuggingFace dataset, 2026. https://huggingface.co/datasets/Warren-wzw/PCB-Layout
118. tscircuit autorouting datasets: autorouting-dataset-01, dataset-srj18, dataset-srj19, tscircuit-autorouter. GitHub repos, 2025. https://github.com/tscircuit/dataset-srj18
119. PCBRouteNet: Simulation Environment for PCB Routing Optimization Dataset (placeholder for a forthcoming industrial BGA routing dataset). GitHub, 2025. https://github.com/CharonRen/PCBRouteNet-Simulation-Environment-for-PCB-Routing-Dataset
120. Adjacent PCB datasets that are NOT layout/routing data: PCB-QA, PCBnet, SchGen, PCBSchemaGen, GraphPCB, PRISM, Kaggle defect sets. arXiv 2606.23704, 2026. https://arxiv.org/abs/2606.23704
121. GPCB Routing: Generative Pretrained Transformers-Based Printed Circuit Board Routing Method. IEEE TCAD vol. 44 no. 4, pp. 1420-1433, DOI 10.1109/TCAD.2024.3486241, 2025. https://doi.org/10.1109/TCAD.2024.3486241
122. Painting on Placement: Forecasting Routing Congestion using Conditional Generative Adversarial Nets. DAC 2019, 2019. https://arxiv.org/abs/1904.07077
123. PROS: A Plug-in for Routability Optimization applied in the State-of-the-art commercial EDA tool using deep learning. ICCAD 2020, 2020. https://dl.acm.org/doi/10.1145/3400302.3415662
