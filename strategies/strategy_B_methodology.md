# Research Methodology: Multi-Instance KV-Sharing Swarm for GSM8K Reasoning

## 0. VOID — Assumptions & Inversions
7 key assumptions identified, including: KV replacement transfers genuine signal vs stochastic artifact, L10 heads 2&3 are optimal communication channel, 3B instances are capable enough, bidirectional KV sharing adds value, more instances monotonically improve accuracy.

## 1. DECOMPOSITION — 7 Orthogonal Research Facets
F1: Communication Topology (star, ring, small-world, fully-connected, dynamic)
F2: Communication Frequency & Timing (pre-gen, early, periodic, triggered, continuous)
F3: Source Selection (best-in-swarm, weighted average, majority consensus, diversity-preserving)
F4: KV Integration Method (full replacement, interpolation, attention-weighted, LB-compressed)
F5: Diversity Maintenance (temperature annealing, forced exploration, repulsive KV penalty)
F6: Scaling Laws (logarithmic, power law, inverse-U, step function)
F7: VRAM & Throughput Budget (must use sequential execution with KV snapshotting)

## 2. Experimental Program — 6 Sequential Phases
P0: Infrastructure Validation (2-3 days) — bidirectional KV swap, LB compression fidelity, KV serialization
P1: Baseline Characterization (1-2 days) — 3B independent ensemble, 7B single-instance, ensemble scaling
P2: Communication Topology Sweep (3-5 days) — 6 topologies × 5 N sizes
P3: Communication Protocol Sweep (3-5 days) — timing × integration grid, dynamic triggering
P4: Collective Intelligence Experiments (3-5 days) — CIG measurement, 7B comparison
P5: Ablation Studies (2-3 days)

## 3. Key Metrics
Primary: Collective Accuracy, Single-Instance Accuracy, Accuracy Delta, 7B Parity Gap
Secondary: Pairwise Agreement, Swarm Diversity, Convergence Time, Confidence Calibration
Efficiency: Total FLOPs, Communication Bandwidth, Wall-clock Latency

## 4. Collective Intelligence Gain Measurement
CIG = CIG_early_consensus + CIG_error_correction + CIG_emergent_reasoning
Control experiment comparing to "oracle" K/V to distinguish genuine emergence from averaging.

## 5. Statistical Framework
Bootstrap CI, McNemar's test, Bonferroni-Holm correction, power analysis (N=200 for sweeps, N=1319 for final).
