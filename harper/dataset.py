"""
SAS-CF Dataset — HARPER v2

Regular dataset: (waveform, y_s, y_e)
Quadruple dataset: 4-way (x_RR, x_RF, x_FR, x_FF) for intervention invariance loss.

Filename format in SAS_CF_mix/:
  M_rr:  genuine+{KEY}.wav
  M_rf:  genuine+{KEY}_{env_codec}.wav        (in subdirs A/B1/.../C)
  M_fr:  {spk_codec}+{KEY}.wav               (in subdirs A/B1/.../C)
  M_ff:  {spk_codec}+{KEY}_{env_codec}.wav   (in subdirs A/B1/.../C)
The shared KEY = {speaker}_{utt}__{scene}-{loc}-{id}-a
"""

import os
import re
import glob
import random
import torch
import torchaudio
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import Dataset


# ── Regular single-clip dataset ───────────────────────────────────────────

class SASCFDataset(Dataset):
    def __init__(self, file_list, sr=16000, max_audio_s=4.0, augment=False):
        self.files       = file_list          # [(path, y_s, y_e), ...]
        self.sr          = sr
        self.max_samples = int(max_audio_s * sr)
        self.augment     = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, y_s, y_e = self.files[idx]
        return (self._load(path),
                torch.tensor(y_s, dtype=torch.long),
                torch.tensor(y_e, dtype=torch.long))

    def _load(self, path):
        try:
            wav, sr = torchaudio.load(path)
        except Exception:
            return torch.zeros(self.max_samples)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        wav = wav.squeeze(0)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        T = wav.shape[-1]
        if T >= self.max_samples:
            start = random.randint(0, T - self.max_samples) if self.augment else 0
            wav = wav[start: start + self.max_samples]
        else:
            wav = F.pad(wav, (0, self.max_samples - T))
        return wav


# ── Quadruple dataset for L_inv / L_eff ──────────────────────────────────

class QuadrupleDataset(Dataset):
    """
    Each item returns 4 waveforms sharing the same (speech_key, scene_key):
      (x_RR, x_RF, x_FR, x_FF)
    Labels are fixed: RR=(1,1), RF=(1,2), FR=(2,1), FF=(2,2).
    """
    def __init__(self, quads, sr=16000, max_audio_s=4.0):
        self.quads       = quads   # list of (path_RR, path_RF, path_FR, path_FF)
        self.sr          = sr
        self.max_samples = int(max_audio_s * sr)

    def __len__(self):
        return len(self.quads)

    def __getitem__(self, idx):
        paths = self.quads[idx]
        wavs  = []
        for p in paths:
            try:
                wav, sr = torchaudio.load(p)
                if wav.shape[0] > 1:
                    wav = wav.mean(0, keepdim=True)
                wav = wav.squeeze(0)
                if sr != self.sr:
                    wav = torchaudio.functional.resample(wav, sr, self.sr)
                T = wav.shape[-1]
                if T >= self.max_samples:
                    wav = wav[:self.max_samples]
                else:
                    wav = F.pad(wav, (0, self.max_samples - T))
            except Exception:
                wav = torch.zeros(self.max_samples)
            wavs.append(wav)
        return tuple(wavs)   # (x_RR, x_RF, x_FR, x_FF)


# ── Path helpers ─────────────────────────────────────────────────────────

def _collect_wavs(roots):
    if isinstance(roots, str):
        roots = [roots]
    wavs = []
    for r in roots:
        if os.path.isdir(r):
            wavs.extend(glob.glob(os.path.join(r, '**', '*.wav'), recursive=True))
    return wavs


def _is_unseen_codec(path, unseen_subdirs):
    parts = path.replace("\\", "/").split("/")
    return any(p in unseen_subdirs for p in parts)


def _extract_speaker_sr(path):
    parts = path.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if "L2-ARCTIC_Real" in p and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _extract_speaker_sf(path):
    fname = os.path.basename(path)
    if "+" in fname:
        return fname.split("+", 1)[1].split("_")[0]
    return None


def _extract_scene_env(path):
    fname = os.path.basename(path)
    return fname.split("-")[0] if "-" in fname else None


def _extract_mix_key(path: str) -> str:
    """
    Extract shared KEY from mixed-condition filenames.
    Format after '+': KEY[_codec_suffix].wav
    KEY = {speaker}_{utt}__{scene}-{loc}-{id}-a
    We strip any trailing _<lower>[a-zA-Z0-9_]{4,} suffix that follows the KEY.
    """
    fname  = os.path.splitext(os.path.basename(path))[0]  # no .wav
    if '+' not in fname:
        return fname
    after_plus = fname.split('+', 1)[1]
    # Strip trailing codec suffix (starts with underscore + lower-alpha word ≥5 chars)
    key = re.sub(r'_[a-z][a-zA-Z0-9_]{4,}$', '', after_plus)
    return key


# ── Split builders ────────────────────────────────────────────────────────

def build_splits(cfg, seed=42):
    rng           = random.Random(seed)
    unseen_codec  = set(getattr(cfg, "unseen_codec_subdirs", []))
    unseen_spk_sr = set(getattr(cfg, "unseen_speakers_sr",   []))
    unseen_spk_sf = set(getattr(cfg, "unseen_speakers_sf",   []))
    unseen_scenes = set(getattr(cfg, "unseen_scenes_env",    []))

    train_all, val_all, test_all = [], [], []

    for cond, (y_s, y_e) in cfg.condition_labels.items():
        roots    = cfg.condition_roots.get(cond) or os.path.join(cfg.data_root, cond)
        all_wavs = _collect_wavs(roots)

        wavs = ([p for p in all_wavs if not _is_unseen_codec(p, unseen_codec)]
                if unseen_codec else all_wavs)
        n_excl_dac = len(all_wavs) - len(wavs)

        n_before_id = len(wavs)
        if   cond == "s_r" and unseen_spk_sr:
            wavs = [p for p in wavs if _extract_speaker_sr(p) not in unseen_spk_sr]
        elif cond == "s_f" and unseen_spk_sf:
            wavs = [p for p in wavs if _extract_speaker_sf(p) not in unseen_spk_sf]
        elif cond in ("e_r", "e_f") and unseen_scenes:
            wavs = [p for p in wavs if _extract_scene_env(p) not in unseen_scenes]
        n_excl_id = n_before_id - len(wavs)

        if not wavs:
            print(f"  [WARN] {cond}: no .wav after filtering")
            continue

        rng.shuffle(wavs)
        n       = min(len(wavs), cfg.train_subset)
        subset  = [(p, y_s, y_e) for p in wavs[:n]]
        n_val   = max(1, int(n * cfg.val_frac))
        n_test  = max(1, int(n * cfg.test_frac))
        n_train = n - n_val - n_test

        train_all.extend(subset[:n_train])
        val_all  .extend(subset[n_train: n_train + n_val])
        test_all .extend(subset[n_train + n_val:])

        excl = (f"dac={n_excl_dac} " if n_excl_dac else "") + \
               (f"id={n_excl_id}"    if n_excl_id  else "")
        print(f"  {cond:<6}: total={len(all_wavs):>7d}  use={n:>6d}  "
              f"train={n_train:>5d}  val={n_val:>4d}  test={n_test:>4d}  "
              f"→({y_s},{y_e})  {excl}")

    rng.shuffle(train_all)
    rng.shuffle(val_all)
    print(f"  Total — train={len(train_all)}, val={len(val_all)}, test={len(test_all)}")
    return train_all, val_all, test_all


def build_quadruples(cfg, seed=42, max_quads=5000):
    """
    Build (x_RR, x_RF, x_FR, x_FF) quadruples from M_rr/M_rf/M_fr/M_ff.
    Matching is done by the shared KEY extracted from filenames.
    Excludes unseen codec subdirs.
    """
    rng          = random.Random(seed)
    unseen_codec = set(getattr(cfg, "unseen_codec_subdirs", []))

    def collect_by_key(cond):
        roots = cfg.condition_roots.get(cond) or os.path.join(cfg.data_root, cond)
        wavs  = _collect_wavs(roots)
        if unseen_codec:
            wavs = [p for p in wavs if not _is_unseen_codec(p, unseen_codec)]
        by_key = defaultdict(list)
        for p in wavs:
            by_key[_extract_mix_key(p)].append(p)
        return by_key

    rr = collect_by_key("M_rr")
    rf = collect_by_key("M_rf")
    fr = collect_by_key("M_fr")
    ff = collect_by_key("M_ff")

    # Find common keys
    keys = sorted(set(rr) & set(rf) & set(fr) & set(ff))
    rng.shuffle(keys)
    if not keys:
        print("  [WARN] No matching quadruple keys found; quadruple loss disabled.")
        return []

    quads = []
    for k in keys[:max_quads]:
        p_rr = rng.choice(rr[k])
        p_rf = rng.choice(rf[k])
        p_fr = rng.choice(fr[k])
        p_ff = rng.choice(ff[k])
        quads.append((p_rr, p_rf, p_fr, p_ff))

    print(f"  Quadruples: {len(quads)} (from {len(keys)} matching keys)")
    return quads


def build_unseen1(cfg, seed=42):
    rng    = random.Random(seed)
    unseen = set(getattr(cfg, "unseen_codec_subdirs", []))
    result = {}
    for cond, (y_s, y_e) in cfg.condition_labels.items():
        if cond not in ("M_ff", "M_rf", "M_fr"):
            continue
        roots    = cfg.condition_roots.get(cond) or os.path.join(cfg.data_root, cond)
        all_wavs = _collect_wavs(roots)
        wavs = [p for p in all_wavs if _is_unseen_codec(p, unseen)]
        if not wavs:
            continue
        rng.shuffle(wavs)
        result[cond] = [(p, y_s, y_e) for p in wavs]
        print(f"  unseen1/{cond:<6}: {len(wavs):>7d} DAC files → ({y_s},{y_e})")
    return result


def build_unseen2(cfg, seed=42):
    rng    = random.Random(seed)
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
    rng           = random.Random(seed)
    result        = {}
    unseen_spk_sr = set(getattr(cfg, "unseen_speakers_sr", []))
    unseen_spk_sf = set(getattr(cfg, "unseen_speakers_sf", []))
    unseen_scenes = set(getattr(cfg, "unseen_scenes_env",  []))
    unseen_codec  = set(getattr(cfg, "unseen_codec_subdirs", []))

    if unseen_spk_sr:
        y_s, y_e = cfg.condition_labels["s_r"]
        roots    = cfg.condition_roots.get("s_r") or os.path.join(cfg.data_root, "s_r")
        wavs = [p for p in _collect_wavs(roots)
                if _extract_speaker_sr(p) in unseen_spk_sr]
        if wavs:
            rng.shuffle(wavs)
            result["s_r_unseen"] = [(p, y_s, y_e) for p in wavs]
            print(f"  unseen_spk/s_r : {len(wavs):>7d} files → ({y_s},{y_e})")

    if unseen_spk_sf:
        y_s, y_e = cfg.condition_labels["s_f"]
        roots    = cfg.condition_roots.get("s_f") or os.path.join(cfg.data_root, "s_f")
        wavs = [p for p in _collect_wavs(roots)
                if _extract_speaker_sf(p) in unseen_spk_sf]
        if wavs:
            rng.shuffle(wavs)
            result["s_f_unseen"] = [(p, y_s, y_e) for p in wavs]
            print(f"  unseen_spk/s_f : {len(wavs):>7d} files → ({y_s},{y_e})")

    if unseen_scenes:
        y_s, y_e = cfg.condition_labels["e_r"]
        roots    = cfg.condition_roots.get("e_r") or os.path.join(cfg.data_root, "e_r")
        wavs = [p for p in _collect_wavs(roots)
                if _extract_scene_env(p) in unseen_scenes]
        if wavs:
            rng.shuffle(wavs)
            result["e_r_unseen"] = [(p, y_s, y_e) for p in wavs]

        y_s, y_e = cfg.condition_labels["e_f"]
        roots    = cfg.condition_roots.get("e_f")
        no_dac   = [p for p in _collect_wavs(roots)
                    if not _is_unseen_codec(p, unseen_codec)]
        wavs     = [p for p in no_dac if _extract_scene_env(p) in unseen_scenes]
        if wavs:
            rng.shuffle(wavs)
            result["e_f_unseen"] = [(p, y_s, y_e) for p in wavs]

    return result
