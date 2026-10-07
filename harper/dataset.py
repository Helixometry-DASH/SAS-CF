"""
SAS-CF Dataset loader for HARPER training.

8 CAS conditions (train/val/test):
  M_rr  (R,R)  M_ff  (F,F)  M_rf  (R,F)  M_fr  (F,R)
  s_r   (R,A)  s_f   (F,A)  e_r   (A,R)  e_f   (A,F)

Unseen test only (never in splits):
  Unseen 1: DAC codec files from M_ff/M_rf/M_fr (B1/B2/B3/D1/D2/D3 subdirs)
  Unseen 2: m_f (R,F) vs M_rr_flat (R,R) — binary
  Unseen Spk/Scene: held-out speakers (s_r/s_f) + held-out scenes (e_r/e_f)
"""

import os
import glob
import random
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset


class SASCFDataset(Dataset):
    """
    Each item: (waveform, y_s, y_e) where y_s, y_e ∈ {0=A, 1=R, 2=F}.
    Audio is clipped/padded to max_audio_s seconds.
    """

    def __init__(self, file_list, sr=16000, max_audio_s=4.0, augment=False):
        self.files       = file_list          # [(path, y_s, y_e), ...]
        self.sr          = sr
        self.max_samples = int(max_audio_s * sr)
        self.augment     = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, y_s, y_e = self.files[idx]
        try:
            wav, sr = torchaudio.load(path)
        except Exception:
            return (torch.zeros(self.max_samples),
                    torch.tensor(y_s, dtype=torch.long),
                    torch.tensor(y_e, dtype=torch.long))

        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        wav = wav.squeeze(0)

        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)

        T = wav.shape[-1]
        if T >= self.max_samples:
            if self.augment:
                start = random.randint(0, T - self.max_samples)
                wav   = wav[start: start + self.max_samples]
            else:
                wav = wav[:self.max_samples]
        else:
            wav = F.pad(wav, (0, self.max_samples - T))

        return (wav,
                torch.tensor(y_s, dtype=torch.long),
                torch.tensor(y_e, dtype=torch.long))


# ─────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────

def _collect_wavs(roots):
    """roots is a str or list of str; returns sorted list of all .wav paths."""
    if isinstance(roots, str):
        roots = [roots]
    wavs = []
    for r in roots:
        if os.path.isdir(r):
            wavs.extend(glob.glob(os.path.join(r, '**', '*.wav'), recursive=True))
    return wavs


def _is_unseen_codec(path, unseen_subdirs):
    """Return True if path passes through any unseen codec subdir."""
    parts = path.replace("\\", "/").split("/")
    return any(p in unseen_subdirs for p in parts)


def _extract_speaker_sr(path):
    """Speaker ID from L2-ARCTIC path: .../L2-ARCTIC_Real/{SPEAKER}/..."""
    parts = path.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if "L2-ARCTIC_Real" in p and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _extract_speaker_sf(path):
    """Speaker ID from CodecFake filename: {codec}+{SPEAKER}_{utt}.wav"""
    fname = os.path.basename(path)
    if "+" in fname:
        return fname.split("+", 1)[1].split("_")[0]
    return None


def _extract_scene_env(path):
    """Scene name from TAU env filename: {SCENE}-{location}-..."""
    fname = os.path.basename(path)
    return fname.split("-")[0] if "-" in fname else None


# ─────────────────────────────────────────────────────────────────────
# Split builders
# ─────────────────────────────────────────────────────────────────────

def build_splits(cfg, seed=42):
    """
    Scan cfg.condition_roots for all WAV files, apply three exclusion layers:
      1. DAC codec subdirs  (B1/B2/B3/D1/D2/D3) → Unseen 1
      2. Held-out speakers  (s_r: L2-ARCTIC, s_f: VCTK) → Unseen Spk/Scene
      3. Held-out scenes    (e_r/e_f: TAU)               → Unseen Spk/Scene
    Subsample to cfg.train_subset per condition, split 80/10/10 train/val/test.
    Returns (train_files, val_files, test_files) — each a list of (path, y_s, y_e).
    """
    rng = random.Random(seed)
    unseen_codec  = set(getattr(cfg, "unseen_codec_subdirs", []))
    unseen_spk_sr = set(getattr(cfg, "unseen_speakers_sr", []))
    unseen_spk_sf = set(getattr(cfg, "unseen_speakers_sf", []))
    unseen_scenes = set(getattr(cfg, "unseen_scenes_env", []))

    train_all, val_all, test_all = [], [], []

    for cond, (y_s, y_e) in cfg.condition_labels.items():
        roots = cfg.condition_roots.get(cond)
        if roots is None:
            roots = os.path.join(cfg.data_root, cond)

        all_wavs = _collect_wavs(roots)

        # 1. Exclude DAC codec files (held-out for Unseen 1)
        wavs = [p for p in all_wavs if not _is_unseen_codec(p, unseen_codec)] \
               if unseen_codec else all_wavs
        n_excl_dac = len(all_wavs) - len(wavs)

        # 2. Exclude held-out speakers / scenes (Unseen Spk/Scene)
        n_before_id = len(wavs)
        if cond == "s_r" and unseen_spk_sr:
            wavs = [p for p in wavs if _extract_speaker_sr(p) not in unseen_spk_sr]
        elif cond == "s_f" and unseen_spk_sf:
            wavs = [p for p in wavs if _extract_speaker_sf(p) not in unseen_spk_sf]
        elif cond in ("e_r", "e_f") and unseen_scenes:
            wavs = [p for p in wavs if _extract_scene_env(p) not in unseen_scenes]
        n_excl_id = n_before_id - len(wavs)

        if not wavs:
            print(f"  [WARN] no .wav files for {cond} after filtering — skipping")
            continue

        rng.shuffle(wavs)
        n      = min(len(wavs), cfg.train_subset)
        subset = [(p, y_s, y_e) for p in wavs[:n]]

        n_val   = max(1, int(n * cfg.val_frac))
        n_test  = max(1, int(n * cfg.test_frac))
        n_train = n - n_val - n_test

        train_all.extend(subset[:n_train])
        val_all  .extend(subset[n_train: n_train + n_val])
        test_all .extend(subset[n_train + n_val:])

        excl_parts = []
        if n_excl_dac: excl_parts.append(f"dac={n_excl_dac}")
        if n_excl_id:  excl_parts.append(f"id={n_excl_id}")
        excl_str = f"  excl({','.join(excl_parts)})" if excl_parts else ""
        print(f"  {cond:<6}: total={len(all_wavs):>7d}{excl_str}  "
              f"use={n:>6d}  train={n_train:>6d}  "
              f"val={n_val:>5d}  test={n_test:>5d}  "
              f"→ ({y_s},{y_e})")

    rng.shuffle(train_all)
    rng.shuffle(val_all)

    print(f"\n  Total — train={len(train_all)}, val={len(val_all)}, test={len(test_all)}")
    return train_all, val_all, test_all


def build_unseen1(cfg, seed=42):
    """
    Unseen 1: DAC-codec files (B1/B2/B3/D1/D2/D3) from M_ff/M_rf/M_fr.
    Same CAS labels as the parent condition (M_ff→(2,2), M_rf→(1,2), M_fr→(2,1)).
    """
    rng = random.Random(seed)
    unseen = set(getattr(cfg, "unseen_codec_subdirs", []))
    result = {}

    mix_conds = {k: v for k, v in cfg.condition_labels.items()
                 if k in ("M_ff", "M_rf", "M_fr")}

    for cond, (y_s, y_e) in mix_conds.items():
        roots = cfg.condition_roots.get(cond, os.path.join(cfg.data_root, cond))
        all_wavs = _collect_wavs(roots)
        wavs = [p for p in all_wavs if _is_unseen_codec(p, unseen)]
        if not wavs:
            continue
        rng.shuffle(wavs)
        result[cond] = [(p, y_s, y_e) for p in wavs]
        print(f"  unseen1/{cond:<6}: {len(wavs):>7d} DAC files → ({y_s},{y_e})")

    return result


def build_unseen2(cfg, seed=42):
    """
    Unseen 2: m_f (genuine speech + codec-faked env, label R,F=1,2) vs M_rr (R,R=1,1).
    Binary classification: bonafide=M_rr_flat, fake=m_f.
    """
    rng = random.Random(seed)
    result = {}

    for cond, (root, (y_s, y_e)) in cfg.unseen_test_roots.items():
        wavs = _collect_wavs(root)
        if not wavs:
            print(f"  [WARN] unseen2: no .wav for {cond}")
            continue
        rng.shuffle(wavs)
        result[cond] = [(p, y_s, y_e) for p in wavs]
        print(f"  unseen2/{cond:<12}: {len(wavs):>7d} files → ({y_s},{y_e})")

    return result


def build_unseen_spk_scene(cfg, seed=42):
    """
    Unseen (Spk/Scene): held-out speakers from s_r/s_f and held-out scenes
    from e_r/e_f (non-DAC roots only). Tests identity/environment generalization.
    """
    rng = random.Random(seed)
    result = {}

    unseen_spk_sr = set(getattr(cfg, "unseen_speakers_sr", []))
    unseen_spk_sf = set(getattr(cfg, "unseen_speakers_sf", []))
    unseen_scenes = set(getattr(cfg, "unseen_scenes_env", []))
    unseen_codec  = set(getattr(cfg, "unseen_codec_subdirs", []))

    # s_r — held-out L2-ARCTIC speakers
    if unseen_spk_sr:
        y_s, y_e = cfg.condition_labels["s_r"]
        roots    = cfg.condition_roots.get("s_r", os.path.join(cfg.data_root, "s_r"))
        wavs = [p for p in _collect_wavs(roots)
                if _extract_speaker_sr(p) in unseen_spk_sr]
        if wavs:
            rng.shuffle(wavs)
            result["s_r_unseen"] = [(p, y_s, y_e) for p in wavs]
            print(f"  unseen_spk/s_r : {len(wavs):>7d} files  "
                  f"({len(unseen_spk_sr)} held-out speakers) → ({y_s},{y_e})")

    # s_f — held-out VCTK speakers
    if unseen_spk_sf:
        y_s, y_e = cfg.condition_labels["s_f"]
        roots    = cfg.condition_roots.get("s_f", os.path.join(cfg.data_root, "s_f"))
        wavs = [p for p in _collect_wavs(roots)
                if _extract_speaker_sf(p) in unseen_spk_sf]
        if wavs:
            rng.shuffle(wavs)
            result["s_f_unseen"] = [(p, y_s, y_e) for p in wavs]
            print(f"  unseen_spk/s_f : {len(wavs):>7d} files  "
                  f"({len(unseen_spk_sf)} held-out speakers) → ({y_s},{y_e})")

    # e_r — held-out TAU scenes (real env)
    if unseen_scenes:
        y_s, y_e = cfg.condition_labels["e_r"]
        roots    = cfg.condition_roots.get("e_r", os.path.join(cfg.data_root, "e_r"))
        wavs = [p for p in _collect_wavs(roots)
                if _extract_scene_env(p) in unseen_scenes]
        if wavs:
            rng.shuffle(wavs)
            result["e_r_unseen"] = [(p, y_s, y_e) for p in wavs]
            print(f"  unseen_scn/e_r : {len(wavs):>7d} files  "
                  f"scenes={sorted(unseen_scenes)} → ({y_s},{y_e})")

    # e_f — held-out TAU scenes from non-DAC codec roots (A, C only)
    if unseen_scenes:
        y_s, y_e = cfg.condition_labels["e_f"]
        roots    = cfg.condition_roots.get("e_f")
        all_wavs = _collect_wavs(roots)
        no_dac   = [p for p in all_wavs if not _is_unseen_codec(p, unseen_codec)]
        wavs     = [p for p in no_dac if _extract_scene_env(p) in unseen_scenes]
        if wavs:
            rng.shuffle(wavs)
            result["e_f_unseen"] = [(p, y_s, y_e) for p in wavs]
            print(f"  unseen_scn/e_f : {len(wavs):>7d} files  "
                  f"scenes={sorted(unseen_scenes)} → ({y_s},{y_e})")

    return result
