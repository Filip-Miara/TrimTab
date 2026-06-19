# Phase 1: Atomic Decomposition & Pyramid Construction

## Subject: TrajectoryTransformer Training Pipeline

---

## Atomic Concepts (Level 1 — indecomposable)

### Data Atoms

| ID | Atom | Evidence Grounding | Confidence |
|----|------|-------------------|------------|
| D1 | Trajectory = hidden states [28, 3584] + velocity targets [28, 3584] | Pipeline spec | 1.0 |
| D2 | 90K trajectories from Qwen2.5-7B-Instruct on GSM8K | Pipeline spec | 1.0 |
| D3 | Float16 storage (17GB per tensor) | Pipeline spec | 1.0 |
| D4 | Memory-mapped file format | Pipeline spec | 1.0 |
| D5 | 500 validation trajectories | Pipeline spec | 1.0 |
| D6 | Global normalization: mean(dim=(0,1)) | Pipeline spec | 1.0 |

### Architecture Atoms

| ID | Atom | Evidence Grounding | Confidence |
|----|------|-------------------|------------|
| A1 | 6-layer TransformerEncoder | Pipeline spec | 1.0 |
| A2 | d_model=768 | Pipeline spec | 1.0 |
| A3 | d_ff=3072 (4× expansion) | Pipeline spec | 1.0 |
| A4 | 8 attention heads | Pipeline spec | 1.0 |
| A5 | Input projection: 3584→768 + LayerNorm | Pipeline spec | 1.0 |
| A6 | Learned positional embeddings | Pipeline spec | 1.0 |
| A7 | Bidirectional self-attention (not causal) | Pipeline spec | 1.0 |
| A8 | Output projection: 768→3584 + LayerNorm | Pipeline spec | 1.0 |
| A9 | 48M total parameters | Pipeline spec | 1.0 |

### Training Atoms

| ID | Atom | Evidence Grounding | Confidence |
|----|------|-------------------|------------|
| T1 | Batch size 64 | Pipeline spec | 1.0 |
| T2 | Learning rate 3e-4 | Pipeline spec | 1.0 |
| T3 | AdamW optimizer | Pipeline spec | 1.0 |
| T4 | Gradient clipping at 1.0 | Pipeline spec | 1.0 |
| T5 | VRAM double-buffer: two float16 GPU buffers (4200×28×3584) | Pipeline spec | 1.0 |
| T6 | Async prefetch via CUDA stream | Pipeline spec | 1.0 |
| T7 | TF32 matmul enabled | Pipeline spec | 1.0 |
| T8 | Loss: MSE (per-layer, uniform weight) | Pipeline spec | 1.0 |
| T9 | Cosine loss variant (λ=0.5) available | Pipeline spec | 1.0 |

### Performance Atoms

| ID | Atom | Evidence Grounding | Confidence |
|----|------|-------------------|------------|
| P1 | R²=0.848 best result | Empirical result | 1.0 |
| P2 | Cos=0.770 (MSE-trained) | Empirical result | 1.0 |
| P3 | Cos=0.709 (cosine loss trained) | Empirical result | 1.0 |
| P4 | R² drops 0.85→0.45 when transferring AWQ | Known issue | 1.0 |
| P5 | Catastrophic forgetting on AWQ fine-tuning | Known issue | 1.0 |
| P6 | CUDA crashes under sustained load | Known issue | 0.8 |
| P7 | Causal attention worse than bidirectional | Known result | 1.0 |
| P8 | Data I/O bottleneck on slow drives | Known issue | 1.0 |

### Concept Atoms

| ID | Atom | Evidence Grounding | Confidence |
|----|------|-------------------|------------|
| C1 | Velocity = hidden state delta between consecutive layers [28, 3584] | Pipeline spec | 1.0 |
| C2 | Velocity predicts steering direction for KV-cache | Pipeline objective | 0.9 |
| C3 | KV-cache steering improves reasoning accuracy | Downstream objective | 0.8 |

### Frozen System Atoms

| ID | Atom | Evidence Grounding | Confidence |
|----|------|-------------------|------------|
| F1 | Qwen2.5-7B-Instruct — 7B parameter LLM | Pipeline spec | 1.0 |
| F2 | 28 transformer layers with hidden dim 3584 | Pipeline spec | 1.0 |
| F3 | Frozen weights (no gradient to Qwen) | Pipeline spec | 1.0 |
| F4 | AWQ quantized variant: different hidden state distribution | Known issue | 1.0 |

---

## Composites (Level 2+)

### Level 2

| ID | Composite | Atoms | Junctions |
|----|-----------|-------|-----------|
| L2-1 | Preprocessing Pipeline | D1-D6 | D6 processes D1; D4 stores D3 |
| L2-2 | Transformer Block | A1-A4, A7 | A1 contains A2-A4; A7 modifies A1 |
| L2-3 | Input/Output Projection | A5, A8 | A5 feeds into A8 via Transformer Block |
| L2-4 | Optimization Setup | T1-T4, T8, T9 | T3 modulates all; T8/T9 alternatives |
| L2-5 | Hardware Acceleration | T5-T7 | T6 feeds T5; T7 modifies compute |
| L2-6 | Data Source | D2, F1-F3 | D2 generated from F1/F2 |
| L2-7 | Known Failure Modes | P4-P8 | Independent observations |
| L2-8 | Performance Metrics | P1-P3 | Joint evaluation |
| L2-9 | Steering Target | C1-C3 | C1→C2→C3 causal chain |

### Level 3

| ID | Composite | Sub-Composites | Junctions |
|----|-----------|----------------|-----------|
| L3-1 | Data System | L2-6 → L2-1 | Sequential: generate→normalize→store |
| L3-2 | Training Core | L2-2 + L2-3 + L2-4 | Transformer processes projected input; Optimization drives weight updates |
| L3-3 | Inference Stack | L2-2 + L2-3 + L2-9 + F1-F4 | TT predicts velocity→steers KV-cache→improves reasoning |
| L3-4 | Performance Layer | L2-8 + L2-7 + L2-5 | Hardware constrains; Failures cap ceiling |

### Level 4 (Peak)

| ID | Composite | Constituents | Role |
|----|-----------|-------------|------|
| L4-1 | TrajectoryTransformer Pipeline | L3-1→L3-2→L3-3 + L3-4 | Complete: data→train→infer→evaluate |

---

## Junction Types

| Junc | Source→Target | Type | Description |
|------|---------------|------|-------------|
| J01 | L2-6→L2-1 | Sequential (causal) | Generate before preprocess |
| J02 | L2-1→L3-2 | Sequential (causal) | Preprocessed data feeds training |
| J03 | L3-2→L4-1 | Compositional | Training is core component |
| J04 | T5→T1 | Constraint | Buffer size limits batch size |
| J05 | A5→A1 | Sequential | Input through projection first |
| J06 | A1→A8 | Sequential | Output through projection |
| J07 | F3→T3 | Constraint | Only TT weights updated |
| J08 | D6→A1 | Dependency | Norm determines input distribution |
| J09 | T8→A1 | Causal (feedback) | Loss drives weight updates |
| J10 | P4-P8→P1 | Causal (limiting) | Failures cap R² ceiling |
| J11 | A7→P7 | Causal (determinant) | Attention type affects results |
| J12 | C1→C2→C3 | Sequential | Predict→steer→improve reasoning |
| J13 | D2→A1 | Dependency | Data limits model capacity |
| J14 | T5→P6 | Antagonistic | Buffering may cause CUDA crashes |
| J15 | T7→P1 | Modulatory | TF32 precision affects R² |
| J16 | C2→C3 | Causal | Steering determines reasoning improvement |
| J17 | F4→A1 | Environmental | AWQ dist shift creates mismatch |

---

## Hallucinatory Pre-Seeds (for Phase 4)

| Atom | Ideal Form (without constraints) |
|------|----------------------------------|
| D6 | Per-layer, per-sample adaptive normalization preserving velocity structure |
| A1 | Depth-adaptive: layers allocated per-sample based on velocity complexity |
| A7 | Hybrid: causal for inference, bidirectional for training supervision |
| T8 | Multi-objective: MSE (magnitude) + cosine (direction) + contrastive (structure) + adversarial (realism) |
| T5 | Online trajectory generation from Qwen → zero storage |
| P4 | Quantization-robust representations via explicit distribution alignment |
| F3 | LoRA adapters on Qwen hidden states: learnable perturbations instead of external steering |
