"""
NAC Bank — frozen codec probes for HARPER.

For each codec C_k:
  x̃_k = C_k(x)  (encode → decode, same rate, amplitude-matched)
  Δ_{k,m}(t,f) = Φ_m(x)_{t,f} − Φ_m(x̃_k)_{t,f}
  R_{k,m}(t,f) = (Δ_{k,m}(t,f) − μ^B_{k,m}(f)) / (σ^B_{k,m}(f) + ε)

Output:  S_x patches (original mel) + K residual patch sets
  → (B, (1+K)*n_patches, patch_size)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ─────────────────────────────────────────────────────────────────────
# Mel spectrogram extractor
# ─────────────────────────────────────────────────────────────────────

class MelExtractor(nn.Module):
    def __init__(self, sr=16000, n_fft=512, hop=160, n_mel=80,
                 f_min=0., f_max=8000.):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mel,
            f_min=f_min, f_max=f_max, power=2.0, norm='slaney',
        )

    def forward(self, x):
        # x: (B, T_samples)
        mel = self.mel_transform(x)            # (B, n_mel, T_frames)
        return torch.log(mel.clamp(min=1e-6)).transpose(1, 2)  # (B, T_frames, n_mel)


# ─────────────────────────────────────────────────────────────────────
# Codec probes (lazy-loaded, frozen)
# ─────────────────────────────────────────────────────────────────────

class EnCodecProbe(nn.Module):
    """EnCodec 24 kHz encode → decode, output at 16 kHz."""

    def __init__(self, sr_in=16000):
        super().__init__()
        self.sr_in    = sr_in
        self.codec_sr = 24000
        self._codec   = None

    def _load(self, device):
        if self._codec is not None:
            return
        from encodec import EncodecModel
        m = EncodecModel.encodec_model_24khz()
        m.set_target_bandwidth(6.0)
        for p in m.parameters():
            p.requires_grad_(False)
        self._codec = m.to(device).eval()

    @torch.no_grad()
    def forward(self, x):
        # x: (B, T) @ sr_in — cast to float32: frozen codec weights are fp32
        self._load(x.device)
        x = x.float()
        x24 = torchaudio.functional.resample(x, self.sr_in, self.codec_sr)
        x24 = x24.unsqueeze(1)                  # (B, 1, T_24k)
        frames = self._codec.encode(x24)
        recon  = self._codec.decode(frames)     # (B, 1, T_24k')
        recon  = recon.squeeze(1)
        recon  = torchaudio.functional.resample(recon, self.codec_sr, self.sr_in)
        return recon[..., :x.shape[-1]]


class DACProbe(nn.Module):
    """DAC 16 kHz encode → decode."""

    def __init__(self, sr_in=16000):
        super().__init__()
        self.sr_in  = sr_in
        self._codec = None

    def _load(self, device):
        if self._codec is not None:
            return
        import dac
        path = dac.utils.download(model_type='16khz')
        m    = dac.DAC.load(path)
        for p in m.parameters():
            p.requires_grad_(False)
        self._codec = m.to(device).eval()

    @torch.no_grad()
    def forward(self, x):
        # x: (B, T) @ sr_in — cast to float32: frozen codec weights are fp32
        self._load(x.device)
        x = x.float()
        inp = x.unsqueeze(1)                    # (B, 1, T)
        z, codes, latents, _, _ = self._codec.encode(inp)
        recon = self._codec.decode(z)           # (B, 1, T')
        recon = recon.squeeze(1)
        return recon[..., :x.shape[-1]]


def _amplitude_match(recon, orig, eps=1e-8):
    """Scale recon to match RMS of orig."""
    rms_orig  = orig .pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=eps)
    rms_recon = recon.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=eps)
    return recon * (rms_orig / rms_recon)


# ─────────────────────────────────────────────────────────────────────
# NAC Bank
# ─────────────────────────────────────────────────────────────────────

class NACBank(nn.Module):
    """
    Returns (B, (1+K)*n_patches, patch_size).
    Dimension 1 layout: [S_x patches | R_1 patches | ... | R_K patches]
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg        = cfg
        self.mel        = MelExtractor(cfg.sr, cfg.n_fft, cfg.hop, cfg.n_mel,
                                       cfg.f_min, cfg.f_max)
        # Per-codec per-mel-bin running BatchNorm
        K = len(cfg.codec_names)
        self.bn = nn.ModuleList([nn.BatchNorm1d(cfg.n_mel) for _ in range(K)])
        self._probes    = None

    def _build_probes(self, device):
        probes = []
        for name in self.cfg.codec_names:
            if 'encodec' in name:
                probes.append(EnCodecProbe(self.cfg.sr).to(device))
            elif 'dac' in name:
                probes.append(DACProbe(self.cfg.sr).to(device))
            else:
                raise ValueError(f"Unknown codec: {name}")
        self._probes = probes

    def _patch(self, mel):
        """(B, T, n_mel) → (B, n_patches, patch_size)."""
        B, T, C  = mel.shape
        pf       = self.cfg.patch_frames
        n_p      = T // pf
        T_trim   = n_p * pf
        return mel[:, :T_trim, :].reshape(B, n_p, pf * C)

    def forward(self, x):
        # x: (B, T_audio) — keep codec/mel in float32; trainable BN can handle fp16→fp32
        if self._probes is None:
            self._build_probes(x.device)

        x_f32 = x.float()
        mel_orig = self.mel(x_f32)              # (B, T, n_mel)
        B, T, C  = mel_orig.shape

        # Codec-id 0 = original S_x
        sx_patches = self._patch(mel_orig)      # (B, n_p, patch_size)
        all_patches = [sx_patches]

        for k, probe in enumerate(self._probes):
            x_recon = probe(x_f32)              # (B, T_audio), float32
            x_recon = _amplitude_match(x_recon, x_f32)

            mel_recon = self.mel(x_recon.float())  # (B, T', n_mel)
            T2 = mel_recon.shape[1]
            if T2 > T:
                mel_recon = mel_recon[:, :T, :]
            elif T2 < T:
                mel_recon = F.pad(mel_recon, (0, 0, 0, T - T2))

            residual = mel_orig - mel_recon     # (B, T, n_mel)

            # Batch-normalize over mel bins
            res_flat = residual.reshape(B * T, C)
            res_norm = self.bn[k](res_flat).reshape(B, T, C)

            all_patches.append(self._patch(res_norm))

        # (B, (1+K)*n_patches, patch_size)
        return torch.cat(all_patches, dim=1)

    @property
    def n_streams(self):
        return 1 + len(self.cfg.codec_names)
