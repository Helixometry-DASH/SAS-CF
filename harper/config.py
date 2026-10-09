"""HARPER v2 hyperparameters — new paper architecture."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class HARPERConfig:
    # ── Backbone (Qwen2-0.5B as LLM) ────────────────────────────────
    backbone_id: str   = "Qwen/Qwen2-0.5B-Instruct"
    lora_r:      int   = 16
    lora_alpha:  int   = 32           # scale = alpha/r = 2
    lora_target: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    d_llm:       int   = 896          # Qwen2-0.5B hidden dim
    fp16:        bool  = True

    # ── Audio ────────────────────────────────────────────────────────
    sr:          int   = 16000
    max_audio_s: float = 4.0

    # ── Multi-resolution STFT ────────────────────────────────────────
    stft_win_ms: List[int] = field(default_factory=lambda: [25, 50, 100])
    hop_ms:      int       = 10       # 10 ms hop → 160 samples @ 16kHz

    # ── Acoustic CNN (from scratch) ──────────────────────────────────
    cnn_dims:    List[int] = field(default_factory=lambda: [64, 128, 384])
    # strides: conv1=(2,2), conv2=(2,2), conv3=(2,4)

    # ── Acoustic Transformer ─────────────────────────────────────────
    acoustic_dim:        int = 384
    transformer_layers:  int = 4
    transformer_heads:   int = 6
    transformer_ffn_dim: int = 1536
    n_ca_layers:         int = 2      # cross-attention blocks for queries

    # ── Hierarchical prompt learning ─────────────────────────────────
    n_prompt_ctx: int = 16            # M learnable context vectors

    # ── Lorentz hyperbolic geometry ──────────────────────────────────
    lorentz_kappa: float = 1.0        # curvature κ
    lorentz_mid:   int   = 256        # intermediate dim 384→256→128
    lorentz_dim:   int   = 128        # output dim on manifold
    lorentz_K:     float = 0.1        # entailment cone constant
    lorentz_eta:   float = 0.9        # cone aperture margin
    hyp_gamma:     float = 10.0       # distance → logit scaling

    # ── CAS ──────────────────────────────────────────────────────────
    N_CLS: int = 3                    # {A=0, R=1, F=2}

    # ── Training / optimisation ──────────────────────────────────────
    train_subset: int   = 8000        # files per condition
    val_frac:     float = 0.1
    test_frac:    float = 0.1

    # Progressive stages
    epochs_stage1: int   = 2          # CNN+Transformer+queries
    epochs_stage2: int   = 1          # + hyperbolic prompts
    epochs_stage3: int   = 1          # + LLM adapters
    epochs_joint:  int   = 1          # joint fine-tune (lower lr)

    batch_size:    int   = 8
    lr:            float = 1e-4       # stages 1-2
    lr_lm:         float = 2e-5       # stage 3
    lr_joint:      float = 5e-6       # joint stage
    weight_decay:  float = 1e-2
    warmup_steps:  int   = 200
    grad_clip:     float = 1.0

    # ── Loss weights ─────────────────────────────────────────────────
    lam_inv:  float = 1.0   # L_inv  — component invariance
    lam_eff:  float = 0.5   # L_eff  — intervention effect
    lam_pre:  float = 1.0   # L_pre  — presence CE
    lam_auth: float = 1.0   # L_auth — authenticity CE
    lam_ent:  float = 0.1   # L_ent  — entailment cones
    lam_lm:   float = 2.0   # L_LM   — LLM autoregressive
    w_final_token: float = 5.0  # weight for <SPEECH=> and <SCENE=> tokens

    # ── Paths ─────────────────────────────────────────────────────────
    data_root:  str = "/mnt/scratch2/users/gmadaan/SAS_CF_mix"
    output_dir: str = "/mnt/scratch2/users/gmadaan/harper_checkpoints_v2"

    # ── CAS label map ─────────────────────────────────────────────────
    condition_labels: dict = field(default_factory=lambda: {
        "M_rr": (1, 1),
        "M_ff": (2, 2),
        "M_rf": (1, 2),
        "M_fr": (2, 1),
        "s_r":  (1, 0),
        "s_f":  (2, 0),
        "e_r":  (0, 1),
        "e_f":  (0, 2),
    })

    condition_roots: dict = field(default_factory=lambda: {
        "M_rr": "/mnt/scratch2/users/gmadaan/SAS_CF_mix/M_rr",
        "M_ff": "/mnt/scratch2/users/gmadaan/SAS_CF_mix/M_ff",
        "M_rf": "/mnt/scratch2/users/gmadaan/SAS_CF_mix/M_rf",
        "M_fr": "/mnt/scratch2/users/gmadaan/SAS_CF_mix/M_fr",
        "s_r":  "/mnt/scratch2/users/gmadaan/L2-ARCTIC_Real",
        "s_f":  "/mnt/scratch2/users/gmadaan/CodecFake_speech",
        "e_r":  "/mnt/scratch2/users/gmadaan/TAU-urban-acoustic-scenes-2019-openset",
        "e_f":  [
            "/mnt/scratch2/users/gmadaan/TAU_CodecFake/A",
            "/mnt/scratch2/users/gmadaan/TAU_CodecFake/B1",
            "/mnt/scratch2/users/gmadaan/TAU_CodecFake/B2",
            "/mnt/scratch2/users/gmadaan/TAU_CodecFake/B3",
            "/mnt/scratch2/users/gmadaan/TAU_CodecFake/C",
        ],
    })

    unseen_codec_subdirs: list = field(
        default_factory=lambda: ["B1", "B2", "B3", "D1", "D2", "D3"]
    )

    unseen_test_roots: dict = field(default_factory=lambda: {
        "m_f":       ("/mnt/scratch2/users/gmadaan/SAS_CF_mix/m_f",        (1, 2)),
        "M_rr_flat": ("/mnt/scratch2/users/gmadaan/SAS_CF_mix/M_rr_flat",  (1, 1)),
    })

    unseen_speakers_sr: list = field(default_factory=lambda: [
        "ERMS", "MBMPS", "SKA", "TXHC", "ZHAA",
    ])
    unseen_speakers_sf: list = field(default_factory=lambda: [
        "p329", "p330", "p333", "p334", "p335", "p336", "p339", "p340",
        "p341", "p343", "p345", "p347", "p351", "p360", "p361", "p362",
        "p363", "p364", "p374", "p376",
    ])
    unseen_scenes_env: list = field(default_factory=lambda: [
        "street_traffic", "tram",
    ])
