# Rethinking Audio Authenticity: When Authenticity Becomes Compositional

> **Anonymous submission** — code and data provided for review purposes only.

---

## Overview

Most audio deepfake detectors treat authenticity as a binary label over the full recording. This work challenges that assumption. Real-world audio is *compositional*: a recording can contain genuine speech over a synthesised acoustic scene, or vice versa. We introduce **HARPER** (**H**ierarchical **A**udio **P**rovenance **R**easoning **E**ncoder), a model that performs component-level authenticity analysis by independently diagnosing speech and acoustic-scene authenticity and grounding its decision in structured natural-language reasoning.

HARPER is evaluated on **SAS-CF**, a new benchmark covering 8 compositional authenticity states (CAS), spanning the full cross-product of \{Absent, Real, Fake\} for both the speech and acoustic-scene components.

---

## Architecture

```
16-kHz audio
    │
    ▼
MultiResSTFT  ──  25 / 50 / 100 ms windows, 10 ms hop
    │               (3 × 201 × T)
    ▼
AcousticCNN   ──  3 layers [64 → 128 → 384], GELU, strides (2,2)/(2,2)/(2,4)
    │
    ▼
AcousticTransformer  ──  4 layers, 6 heads, d=384, FFN=1536
    │
    ▼
ComponentQueryExtractor  ──  q_speech / q_scene / q_shared  (2× cross-attention)
    │
    ├──► HierarchicalPromptLearner  ──  8 learnable 384-dim hierarchy nodes
    │
    ▼
LorentzCASModule  ──  384 → 256 → 128 (Lorentz H^128, κ=1.0)
    │                  presence logits + authenticity logits + entailment cones
    ▼
EvidenceAdapters  ──  5 × MLP(386 → 896)  →  d_s_P, d_s_A, d_e_P, d_e_A, d_g
    │
    ▼
Qwen2-0.5B-Instruct  ──  frozen + LoRA (r=16, α=32) on q/k/v/o
    │                     structured 6-token generation:
    │                     <SP=?> <SA=?> <EP=?> <EA=?> <SPH=?> <SCN=?>
    ▼
Component CAS prediction  +  natural-language forensic rationale
```

---

## Directory Structure

```
Speech_EnvFake/
│
├── harper/                      # HARPER model package
│   ├── __init__.py
│   ├── config.py                # HARPERConfig dataclass — all hyperparameters
│   ├── model.py                 # Full model: MultiResSTFT → Qwen2 + LoRA
│   ├── losses.py                # HARPERLoss — 6 loss terms (L_inv/eff/pre/auth/ent/LM)
│   ├── dataset.py               # SASCFDataset, QuadrupleDataset, build_splits
│   ├── train.py                 # Progressive 4-stage training loop
│   ├── evaluate.py              # EER / ACC / Joint_ACC / confusion matrix
│   ├── train_harper_v2.sh       # SLURM submission script
│   │
│   └── plot_*/                  # Result visualisation scripts
│       ├── plot_confusion.py
│       ├── plot_curves.py
│       ├── plot_idempotence.py
│       ├── plot_per_cond.py
│       └── plot_provenance.py
│
├── spec_plots/                  # Generated figures (spectrograms, t-SNE)
│
├── garuda_model.py              # Baseline: GARUDA
├── satyam_model.py              # Baseline: SATYAM
│
└── plot_hyperbolic_tsne.py      # Hyperbolic t-SNE visualisation (4 CAS splits)
```

---

## SAS-CF Dataset

SAS-CF covers **8 Compositional Authenticity States (CAS)**:

| Condition | Speech | Scene | Split |
|-----------|--------|-------|-------|
| `M_rr` | Real | Real | Train / Val / Test |
| `M_rf` | Real | Fake | Train / Val / Test |
| `M_fr` | Fake | Real | Train / Val / Test |
| `M_ff` | Fake | Fake | Train / Val / Test |
| `s_r`  | Real | Absent | Train / Val / Test |
| `s_f`  | Fake | Absent | Train / Val / Test |
| `e_r`  | Absent | Real | Train / Val / Test |
| `e_f`  | Absent | Fake | Train / Val / Test |

Fake speech: 20+ neural codec models (EnCodec, DAC, FunCodec, SpeechTokenizer, …).  
Fake scenes: 5 codec families (A / B1 / B2 / B3 / C) applied to TAU Urban Acoustic Scenes.

---

## Training

Four progressive stages:

| Stage | Modules trained | Loss terms | Epochs |
|-------|----------------|------------|--------|
| 1 — Acoustic | CNN + Transformer + queries | $\mathcal{L}_\text{inv}$, $\mathcal{L}_\text{eff}$ | 2 |
| 2 — Hyperbolic | + Lorentz + prompts | + $\mathcal{L}_\text{pre}$, $\mathcal{L}_\text{auth}$, $\mathcal{L}_\text{ent}$ | 1 |
| 3 — LLM | + evidence adapters + LoRA | + $\mathcal{L}_\text{LM}$ | 1 |
| 4 — Joint | all modules | all 6 terms | 1 |

```bash
sbatch harper/train_harper_v2.sh
```

Key hyperparameters (`harper/config.py`):

```python
lr            = 1e-4      # acoustic stages
lr_lm         = 2e-5      # LLM stage
lr_joint      = 5e-6      # joint fine-tune
batch_size    = 8
lora_r        = 16
lora_alpha    = 32
lorentz_dim   = 128
n_prompt_ctx  = 16
```

---

## Model Scale

| | Parameters |
|---|---|
| Total | ~517M |
| Trainable | ~24M (4.6%) |
| Frozen (Qwen2-0.5B) | ~494M |
| Acoustic modules | ~10M |
| Evidence adapters | ~11.5M |
| LoRA (q/k/v/o) | ~2.2M |

---

## Evaluation Metrics

- **EER** — joint binary (any fake component → spoof)
- **EER\_s / EER\_e** — speech-only / scene-only binary EER
- **ACC\_s / ACC\_e** — 3-class (A / R / F) component accuracy
- **Joint\_ACC** — both components correct simultaneously
- **Per-CAS breakdown** — metrics per each of the 8 authenticity states

```bash
python -m harper.evaluate --ckpt <checkpoint.pt>
```

---

## Requirements

```
torch >= 2.1
transformers >= 4.40
peft >= 0.10
librosa >= 0.10
scikit-learn >= 1.3
```

---

*Anonymous submission. Author details withheld for review.*
