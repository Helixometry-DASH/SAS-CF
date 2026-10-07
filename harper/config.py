"""HARPER hyperparameters and paths."""
from dataclasses import dataclass, field
from typing import List

@dataclass
class HARPERConfig:
    # ── Backbone ────────────────────────────────────────────────────
    backbone_id: str = "Qwen/Qwen2-0.5B-Instruct"
    lora_r:      int = 8
    lora_alpha:  int = 256  # scale = alpha/r = 32 in PEFT (paper: r=8, scale=32 → alpha=256)
    lora_target: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    hidden:      int = 896  # Qwen2-0.5B hidden dim

    # ── NAC bank ────────────────────────────────────────────────────
    # Use EnCodec-24k and DAC-16k for fast inference
    codec_names: List[str] = field(default_factory=lambda: ["encodec_24k", "dac_16k"])
    sr:          int = 16000   # model sample rate

    # ── Spectrogram ─────────────────────────────────────────────────
    n_fft:   int = 512
    hop:     int = 160   # 10 ms at 16kHz
    n_mel:   int = 80
    f_min:   float = 0.0
    f_max:   float = 8000.0

    # ── Patch projection ────────────────────────────────────────────
    patch_frames: int = 16     # time frames per patch
    # full 80-dim mel per patch → patch_size = patch_frames × n_mel = 1280

    # ── Hyperbolic geometry ─────────────────────────────────────────
    hyp_dim: int = 128
    hyp_c:   float = 1.0       # Poincaré ball curvature

    # ── CAS ─────────────────────────────────────────────────────────
    # y_s, y_e ∈ {A=0, R=1, F=2}
    N_CLS:   int = 3
    # 8 valid joint states (exclude (A,A)): enumerate for indexing
    # (A,R)=0, (A,F)=1, (R,A)=2, (R,R)=3, (R,F)=4, (F,A)=5, (F,R)=6, (F,F)=7

    # ── Training ────────────────────────────────────────────────────
    max_audio_s:  float = 4.0
    train_subset: int   = 15000   # files per condition for training
    val_frac:     float = 0.1
    test_frac:    float = 0.1
    epochs:       int   = 3
    batch_size:   int   = 16
    lr:           float = 1e-5
    weight_decay: float = 1e-2
    warmup_steps: int   = 200
    grad_clip:    float = 1.0
    fp16:         bool  = True

    # ── Loss weights ────────────────────────────────────────────────
    lam_lm:    float = 1.0
    lam_route: float = 1.0
    lam_tree:  float = 0.5
    lam_dec:   float = 1.0
    lam_geo:   float = 0.5

    # ── Paths ───────────────────────────────────────────────────────
    data_root:  str = "/mnt/scratch2/users/gmadaan/SAS_CF_mix"
    output_dir: str = "/users/gmadaan/Girish/Speech_EnvFake/harper/checkpoints"

    # ── CAS label map — all 8 valid CAS states ───────────────────────
    # condition → (y_s, y_e);  A=0, R=1, F=2
    # m_f is UNSEEN TEST ONLY — not in this dict
    condition_labels: dict = field(default_factory=lambda: {
        "M_rr": (1, 1),   # R, R  — mixed real+real
        "M_ff": (2, 2),   # F, F  — mixed fake+fake
        "M_rf": (1, 2),   # R, F  — mixed real speech + fake env
        "M_fr": (2, 1),   # F, R  — mixed fake speech + real env
        "s_r":  (1, 0),   # R, A  — speech-only real
        "s_f":  (2, 0),   # F, A  — speech-only fake
        "e_r":  (0, 1),   # A, R  — env-only real
        "e_f":  (0, 2),   # A, F  — env-only fake
    })

    # ── Per-condition root directories ───────────────────────────────
    # str  → single root, all *.wav found recursively
    # list → multiple roots merged (used for e_f: 5 codec systems)
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

    # ── DAC codec subdirs — held-out from training ───────────────────
    # B1/B2/B3: DAC used for speech synthesis (descript-audio-codec-*)
    # D1/D2/D3: DAC used for env synthesis (dac_16/24/44khz)
    # These appear under M_ff/, M_rf/, M_fr/ and are excluded from
    # build_splits() → form Unseen 1 evaluation set
    unseen_codec_subdirs: list = field(
        default_factory=lambda: ["B1", "B2", "B3", "D1", "D2", "D3"]
    )

    # ── Unseen test sets ─────────────────────────────────────────────
    # Unseen 1: DAC codec files from M_ff/M_rf/M_fr (same labels as parent cond)
    #   → built dynamically in build_unseen1() using unseen_codec_subdirs
    # Unseen 2: m_f (genuine speech + codec-faked env) paired with M_rr
    #   m_f label = (R=1, F=2) — speech is GENUINE, env is codec-resynthesized
    unseen_test_roots: dict = field(default_factory=lambda: {
        "m_f":       ("/mnt/scratch2/users/gmadaan/SAS_CF_mix/m_f",        (1, 2)),
        "M_rr_flat": ("/mnt/scratch2/users/gmadaan/SAS_CF_mix/M_rr_flat",  (1, 1)),
    })

    # ── Speaker / scene hold-out — Unseen (Spk/Scene) eval ──────────
    # s_r: L2-ARCTIC speakers withheld from training (5 / 24 = 20.8%)
    unseen_speakers_sr: list = field(default_factory=lambda: [
        "ERMS", "MBMPS", "SKA", "TXHC", "ZHAA",
    ])
    # s_f: VCTK speakers withheld from training (20 / 107 = 18.7%)
    unseen_speakers_sf: list = field(default_factory=lambda: [
        "p329", "p330", "p333", "p334", "p335", "p336", "p339", "p340",
        "p341", "p343", "p345", "p347", "p351", "p360", "p361", "p362",
        "p363", "p364", "p374", "p376",
    ])
    # e_r / e_f: TAU acoustic scenes withheld from training (2 / 11 = 18.2%)
    unseen_scenes_env: list = field(default_factory=lambda: [
        "street_traffic", "tram",
    ])
