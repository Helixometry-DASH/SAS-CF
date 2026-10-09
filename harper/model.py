"""
HARPER v2 — encode → factorize → organize → reason → decide

  MultiResSTFT → AcousticCNN → AcousticTransformer →
  ComponentQueryExtractor (3 learnable queries, 2× cross-attn) →
  HierarchicalPromptLearner + LorentzHyperbolic →
  EvidenceAdapters (5 tokens) →
  Qwen2-0.5B (frozen + LoRA r=16) → structured token generation
"""

import math
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as _td
if not hasattr(_td, 'tensor'):
    _fake = types.ModuleType('torch.distributed.tensor')
    class _DTensor: pass
    _fake.DTensor = _DTensor
    _td.tensor = _fake

from .config import HARPERConfig

# ─── Special token definitions ──────────────────────────────────────────
MARKER_TOKENS = [
    "<SP_PRES>", "<SP_AUTH>", "<EP_PRES>", "<EP_AUTH>", "<CROSS>", "<ANSWER>",
]
TARGET_TOKENS = [
    "<SP=P>", "<SP=A>",                   # speech presence
    "<SA=R>", "<SA=F>", "<SA=N>",         # speech authenticity (N=not applicable)
    "<EP=P>", "<EP=A>",                   # env presence
    "<EA=R>", "<EA=F>", "<EA=N>",         # env authenticity
    "<SPH=A>", "<SPH=R>", "<SPH=F>",      # final speech state
    "<SCN=A>", "<SCN=R>", "<SCN=F>",      # final scene state
]
ALL_SPECIAL = MARKER_TOKENS + TARGET_TOKENS

INSTRUCTION = (
    "You are a forensic audio analyst. "
    "Determine whether the speech and acoustic scene in the recording are real or fake, "
    "and report each component's presence and authenticity."
)

# ─────────────────────────────────────────────────────────────────────────
# Lorentz hyperbolic geometry helpers
# ─────────────────────────────────────────────────────────────────────────

def lorentz_exp_map(v: torch.Tensor, kappa: float) -> torch.Tensor:
    """Euclidean R^d → Lorentz manifold H^d_κ.
    v: (*, d)  →  output: (*, d+1)  on manifold
    """
    sqrt_k = math.sqrt(kappa)
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    t = torch.cosh(sqrt_k * v_norm) / sqrt_k                       # (*, 1)
    x = torch.sinh(sqrt_k * v_norm) / (sqrt_k * v_norm) * v        # (*, d)
    return torch.cat([t, x], dim=-1)


def lorentz_inner(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Lorentz inner product <p,q>_L = -p0*q0 + p̃·q̃.  (*, d+1) → (*)."""
    return -p[..., 0] * q[..., 0] + (p[..., 1:] * q[..., 1:]).sum(-1)


def lorentz_dist(p: torch.Tensor, q: torch.Tensor, kappa: float) -> torch.Tensor:
    """Geodesic distance on Lorentz manifold. (*, d+1) → (*)."""
    inner = lorentz_inner(p, q).clamp(max=-1.0 - 1e-7)
    return torch.acosh((-kappa * inner).clamp(min=1.0 + 1e-7)) / math.sqrt(kappa)


# ─────────────────────────────────────────────────────────────────────────
# Multi-resolution STFT
# ─────────────────────────────────────────────────────────────────────────

class MultiResSTFT(nn.Module):
    def __init__(self, sr: int = 16000, win_ms=(25, 50, 100), hop_ms: int = 10):
        super().__init__()
        self.sr   = sr
        self.hop  = int(hop_ms * sr / 1000)                        # 160 samples
        # Make each window length even
        self.n_ffts = [2 * (int(w * sr / 1000) // 2) for w in win_ms]
        self.F_common = self.n_ffts[0] // 2 + 1                    # 201 bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T)  →  (B, 3, F_common, T_frames)"""
        B, T = x.shape
        maps = []
        for n_fft in self.n_ffts:
            window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
            # stft returns (B, F, T_frames) complex
            s = torch.stft(x.reshape(B, T), n_fft=n_fft,
                           hop_length=self.hop, win_length=n_fft,
                           window=window, return_complex=True)      # (B, F, T_f)
            mag = torch.log1p(s.abs())                              # (B, F, T_f)
            if mag.shape[1] != self.F_common:
                mag = F.interpolate(
                    mag.unsqueeze(1).float(),
                    size=(self.F_common, mag.shape[2]),
                    mode='bilinear', align_corners=False,
                ).squeeze(1).to(x.dtype)
            maps.append(mag)
        return torch.stack(maps, dim=1)                             # (B, 3, F, T_f)


# ─────────────────────────────────────────────────────────────────────────
# Acoustic CNN (from scratch)
# ─────────────────────────────────────────────────────────────────────────

class AcousticCNN(nn.Module):
    """3→64→128→384 with strides (2,2),(2,2),(2,4). GELU activations."""
    def __init__(self, in_ch: int = 3, dims=(64, 128, 384)):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, dims[0], 3, stride=(2, 2), padding=1),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], 3, stride=(2, 2), padding=1),
            nn.GELU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], 3, stride=(2, 4), padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, F, T)  →  (B, N, 384) where N=F'×T'"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        B, C, F, T = x.shape
        return x.permute(0, 2, 3, 1).reshape(B, F * T, C)          # (B, N, 384)


# ─────────────────────────────────────────────────────────────────────────
# Acoustic Transformer
# ─────────────────────────────────────────────────────────────────────────

class AcousticTransformer(nn.Module):
    """4-layer Transformer encoder: 6 heads, FFN=1536, GELU, pre-norm."""
    def __init__(self, d_model=384, n_heads=6, n_layers=4,
                 ffn_dim=1536, max_tokens=2048):
        super().__init__()
        self.pos_embed = nn.Embedding(max_tokens, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            activation='gelu', batch_first=True, norm_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, N, 384)  →  Z: (B, N, 384)"""
        N = z.shape[1]
        pos = torch.arange(N, device=z.device)
        z = z + self.pos_embed(pos)
        return self.encoder(z)


# ─────────────────────────────────────────────────────────────────────────
# Component-Aware Query Extractor
# ─────────────────────────────────────────────────────────────────────────

class ComponentQueryExtractor(nn.Module):
    """3 learnable queries (s, e, g) + 2× cross-attention over Z."""
    def __init__(self, d_model=384, n_heads=6, n_ca=2):
        super().__init__()
        self.q_s = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.q_e = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.q_g = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.ca     = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(n_ca)
        ])
        self.norm_q = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_ca)])
        self.norm_k = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_ca)])

    def forward(self, Z: torch.Tensor):
        """Z: (B, N, 384)  →  h_s, h_e, h_g each (B, 384)"""
        B = Z.shape[0]
        Q = torch.cat([self.q_s, self.q_e, self.q_g], dim=1).expand(B, -1, -1)

        for ca, nq, nk in zip(self.ca, self.norm_q, self.norm_k):
            q_in  = nq(Q)
            kv_in = nk(Z)
            attn_out, _ = ca(q_in, kv_in, kv_in)
            Q = Q + attn_out

        return Q[:, 0, :], Q[:, 1, :], Q[:, 2, :]   # h_s, h_e, h_g


# ─────────────────────────────────────────────────────────────────────────
# Hierarchical Prompt Learner
# ─────────────────────────────────────────────────────────────────────────

class HierarchicalPromptLearner(nn.Module):
    """
    8 learnable prompt embeddings (384-dim) for the hierarchy nodes:
      speech:  s_A, s_P, s_R, s_F
      scene:   e_A, e_P, e_R, e_F
    Separate context vector sets for presence and authenticity levels.
    """
    def __init__(self, d: int = 384):
        super().__init__()
        # Learnable prompt embeddings — one per hierarchy node
        for key in ['s_A', 's_P', 's_R', 's_F', 'e_A', 'e_P', 'e_R', 'e_F']:
            self.register_parameter(
                f'p_{key}', nn.Parameter(torch.randn(d) * 0.02)
            )

    def get(self, key: str) -> torch.Tensor:
        """Return prompt embedding for a given hierarchy node."""
        return getattr(self, f'p_{key}')


# ─────────────────────────────────────────────────────────────────────────
# Lorentz Hyperbolic Module (presence + authenticity)
# ─────────────────────────────────────────────────────────────────────────

class LorentzCASModule(nn.Module):
    """
    Projects h_s, h_e and prompts into Lorentz space.
    Computes presence logits, authenticity logits, and three-state distribution.
    """
    def __init__(self, in_dim=384, mid_dim=256, out_dim=128,
                 kappa=1.0, K_ent=0.1, eta=0.9, gamma=10.0):
        super().__init__()
        self.kappa = kappa
        self.K_ent = K_ent
        self.eta   = eta
        self.gamma = gamma

        # MLP for component reps: 384 → 256 → 128
        def _mlp():
            return nn.Sequential(
                nn.Linear(in_dim, mid_dim), nn.GELU(),
                nn.Linear(mid_dim, out_dim),
            )
        self.mlp_s = _mlp()
        self.mlp_e = _mlp()
        # Same projection for prompt embeddings
        self.mlp_p = _mlp()

    def _project(self, h: torch.Tensor, mlp) -> torch.Tensor:
        """h: (B, in_dim) → z: (B, out_dim+1) on Lorentz manifold"""
        a = mlp(h.float())
        return lorentz_exp_map(a, self.kappa)

    def _project_prompt(self, t: torch.Tensor) -> torch.Tensor:
        """t: (in_dim,) → z: (out_dim+1,) on manifold"""
        a = self.mlp_p(t.float())
        return lorentz_exp_map(a, self.kappa)

    def _dist_logits(self, z: torch.Tensor, z_prompts: list) -> torch.Tensor:
        """
        z: (B, d+1),  z_prompts: list of (d+1,) Lorentz points
        → logits: (B, len(z_prompts))
        """
        B = z.shape[0]
        logits = []
        for zp in z_prompts:
            zp_exp = zp.unsqueeze(0).expand(B, -1)   # (B, d+1)
            d = lorentz_dist(z, zp_exp, self.kappa)  # (B,)
            logits.append(-self.gamma * d)
        return torch.stack(logits, dim=1)             # (B, n)

    def forward(self, h_s, h_e, prompt_learner: 'HierarchicalPromptLearner'):
        """
        h_s, h_e: (B, 384)
        Returns: dict with presence/auth distributions and Lorentz points for loss
        """
        # Project component reps to manifold
        z_s = self._project(h_s, self.mlp_s)   # (B, 129)
        z_e = self._project(h_e, self.mlp_e)

        # Project prompt embeddings to manifold
        zp = {}
        for key in ['s_A', 's_P', 's_R', 's_F', 'e_A', 'e_P', 'e_R', 'e_F']:
            zp[key] = self._project_prompt(prompt_learner.get(key))

        # Presence logits: {A, P}
        pre_s = F.softmax(self._dist_logits(z_s, [zp['s_A'], zp['s_P']]), dim=-1)  # (B,2)
        pre_e = F.softmax(self._dist_logits(z_e, [zp['e_A'], zp['e_P']]), dim=-1)

        # Authenticity logits: {R, F}
        auth_s = F.softmax(self._dist_logits(z_s, [zp['s_R'], zp['s_F']]), dim=-1)  # (B,2)
        auth_e = F.softmax(self._dist_logits(z_e, [zp['e_R'], zp['e_F']]), dim=-1)

        # Three-state distribution p^H: A, R=P*auth_R, F=P*auth_F
        p_H_s = torch.stack([
            pre_s[:, 0],                           # A
            pre_s[:, 1] * auth_s[:, 0],            # R
            pre_s[:, 1] * auth_s[:, 1],            # F
        ], dim=1)                                  # (B, 3)

        p_H_e = torch.stack([
            pre_e[:, 0],
            pre_e[:, 1] * auth_e[:, 0],
            pre_e[:, 1] * auth_e[:, 1],
        ], dim=1)

        return {
            'z_s': z_s, 'z_e': z_e,
            'zp': zp,
            'pre_s': pre_s, 'pre_e': pre_e,
            'auth_s': auth_s, 'auth_e': auth_e,
            'p_H_s': p_H_s, 'p_H_e': p_H_e,
        }

    def entailment_loss(self, z_d: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        """Cone penalty for descendant z_d w.r.t. ancestor z_a. (B, d+1) each."""
        # Aperture of ancestor cone
        spatial_norm = z_a[..., 1:].norm(dim=-1).clamp(min=1e-7)  # (B,)
        omega = torch.arcsin(
            (2 * self.K_ent / (math.sqrt(self.kappa) * spatial_norm)).clamp(-1+1e-6, 1-1e-6)
        )

        # Exterior angle of descendant w.r.t. ancestor
        inner = lorentz_inner(z_d, z_a)                            # (B,)
        ki = -self.kappa * inner                                    # (B,)
        denom = spatial_norm * (ki.pow(2) - 1).clamp(min=1e-7).sqrt()
        cos_phi = ((z_d[..., 0] + z_a[..., 0] * ki) / denom.clamp(min=1e-7)).clamp(-1+1e-6, 1-1e-6)
        phi = torch.arccos(cos_phi)

        return F.relu(phi - self.eta * omega).mean()


# ─────────────────────────────────────────────────────────────────────────
# Evidence Adapters
# ─────────────────────────────────────────────────────────────────────────

class EvidenceAdapter(nn.Module):
    """2-layer FFN: in_dim → 2*d_llm → d_llm with GELU."""
    def __init__(self, in_dim: int, d_llm: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2 * d_llm),
            nn.GELU(),
            nn.Linear(2 * d_llm, d_llm),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)


# ─────────────────────────────────────────────────────────────────────────
# Full HARPER Model
# ─────────────────────────────────────────────────────────────────────────

class HARPERModel(nn.Module):
    def __init__(self, cfg: HARPERConfig):
        super().__init__()
        self.cfg = cfg

        # ── Acoustic stack (from scratch) ────────────────────────────
        self.stft = MultiResSTFT(cfg.sr, cfg.stft_win_ms, cfg.hop_ms)
        self.cnn  = AcousticCNN(3, cfg.cnn_dims)
        self.transformer = AcousticTransformer(
            cfg.acoustic_dim, cfg.transformer_heads,
            cfg.transformer_layers, cfg.transformer_ffn_dim,
        )
        self.query_extractor = ComponentQueryExtractor(
            cfg.acoustic_dim, cfg.transformer_heads, cfg.n_ca_layers,
        )

        # ── Hierarchical prompts ─────────────────────────────────────
        self.prompt_learner = HierarchicalPromptLearner(cfg.acoustic_dim)

        # ── Lorentz CAS module ───────────────────────────────────────
        self.lorentz = LorentzCASModule(
            cfg.acoustic_dim, cfg.lorentz_mid, cfg.lorentz_dim,
            cfg.lorentz_kappa, cfg.lorentz_K, cfg.lorentz_eta, cfg.hyp_gamma,
        )

        # ── Evidence adapters (5 tokens) ─────────────────────────────
        # in_dim for s_P, s_A, e_P, e_A: acoustic_dim + 2 = 386
        # in_dim for g: acoustic_dim = 384
        self.adp_s_P = EvidenceAdapter(cfg.acoustic_dim + 2, cfg.d_llm)
        self.adp_s_A = EvidenceAdapter(cfg.acoustic_dim + 2, cfg.d_llm)
        self.adp_e_P = EvidenceAdapter(cfg.acoustic_dim + 2, cfg.d_llm)
        self.adp_e_A = EvidenceAdapter(cfg.acoustic_dim + 2, cfg.d_llm)
        self.adp_g   = EvidenceAdapter(cfg.acoustic_dim,     cfg.d_llm)

        # ── LLM (Qwen2-0.5B + LoRA) ──────────────────────────────────
        self._build_llm(cfg)

    # ── LLM construction ─────────────────────────────────────────────

    def _build_llm(self, cfg: HARPERConfig):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import get_peft_model, LoraConfig

        print(f"[HARPER] Loading {cfg.backbone_id} ...")
        dtype = torch.float16 if cfg.fp16 else torch.float32
        base  = AutoModelForCausalLM.from_pretrained(
            cfg.backbone_id, torch_dtype=dtype, device_map=None,
        )

        # Add special tokens
        tok = AutoTokenizer.from_pretrained(cfg.backbone_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.add_special_tokens({'additional_special_tokens': ALL_SPECIAL})
        base.resize_token_embeddings(len(tok))
        self.tokenizer = tok

        # LoRA on q,k,v,o projections
        lora_cfg = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
            target_modules=cfg.lora_target,
            bias="none",
        )
        self.llm = get_peft_model(base, lora_cfg)
        self.llm.print_trainable_parameters()

        # Precompute instruction token IDs (fixed)
        inst_ids = tok(INSTRUCTION, return_tensors='pt',
                       add_special_tokens=False).input_ids
        self.register_buffer('_inst_ids', inst_ids)

        # Cache special token IDs
        def _id(s):
            return tok.convert_tokens_to_ids(s)

        self._mk_ids = {t: _id(t) for t in MARKER_TOKENS}
        self._tgt_ids = {t: _id(t) for t in TARGET_TOKENS}

    # ── Input sequence builder ────────────────────────────────────────

    def _build_input_seq(self, d_s_P, d_s_A, d_e_P, d_e_A, d_g):
        """
        Build LLM input embedding sequence (no target tokens).
        All d_* are (B, d_llm).
        Returns: (B, L_input, d_llm)
        """
        B      = d_s_P.shape[0]
        device = d_s_P.device
        dtype  = next(self.llm.parameters()).dtype
        embed  = self.llm.get_input_embeddings()

        def _tok_emb(tok_str):
            tid = torch.tensor([[self._mk_ids[tok_str]]], device=device)
            return embed(tid).expand(B, -1, -1).to(dtype)   # (B, 1, d_llm)

        inst_emb = embed(self._inst_ids.to(device)).expand(B, -1, -1).to(dtype)

        seq = torch.cat([
            inst_emb,
            _tok_emb("<SP_PRES>"),  d_s_P.unsqueeze(1).to(dtype),
            _tok_emb("<SP_AUTH>"),  d_s_A.unsqueeze(1).to(dtype),
            _tok_emb("<EP_PRES>"),  d_e_P.unsqueeze(1).to(dtype),
            _tok_emb("<EP_AUTH>"),  d_e_A.unsqueeze(1).to(dtype),
            _tok_emb("<CROSS>"),    d_g.unsqueeze(1).to(dtype),
            _tok_emb("<ANSWER>"),
        ], dim=1)                                            # (B, L_input, d_llm)
        return seq

    def _build_target_ids(self, y_s, y_e, device):
        """
        Build 6-token target ID sequence from (y_s, y_e) labels.
        y_s, y_e: (B,) in {0=A,1=R,2=F}
        Returns: (B, 6) token IDs
        """
        B = y_s.shape[0]
        T = self._tgt_ids
        tgt = torch.zeros(B, 6, dtype=torch.long, device=device)
        for b in range(B):
            s, e = y_s[b].item(), y_e[b].item()
            # pos 0: speech presence
            tgt[b, 0] = T["<SP=A>"] if s == 0 else T["<SP=P>"]
            # pos 1: speech authenticity
            tgt[b, 1] = T["<SA=N>"] if s == 0 else (T["<SA=R>"] if s == 1 else T["<SA=F>"])
            # pos 2: env presence
            tgt[b, 2] = T["<EP=A>"] if e == 0 else T["<EP=P>"]
            # pos 3: env authenticity
            tgt[b, 3] = T["<EA=N>"] if e == 0 else (T["<EA=R>"] if e == 1 else T["<EA=F>"])
            # pos 4: speech state (final)
            tgt[b, 4] = T["<SPH=A>"] if s == 0 else (T["<SPH=R>"] if s == 1 else T["<SPH=F>"])
            # pos 5: scene state (final)
            tgt[b, 5] = T["<SCN=A>"] if e == 0 else (T["<SCN=R>"] if e == 1 else T["<SCN=F>"])
        return tgt

    # ── Acoustic forward ──────────────────────────────────────────────

    def acoustic_forward(self, audio: torch.Tensor):
        """
        audio: (B, T_audio) raw waveform
        Returns h_s, h_e, h_g each (B, 384), and hyp_out dict.
        """
        # Multi-resolution acoustic maps
        X = self.stft(audio.float())                  # (B, 3, F, T)

        # CNN tokenization
        dtype_llm = next(self.llm.parameters()).dtype
        Z0 = self.cnn(X.to(dtype_llm))               # (B, N, 384)

        # Transformer encoding
        Z = self.transformer(Z0)                      # (B, N, 384)

        # Component query extraction
        h_s, h_e, h_g = self.query_extractor(Z)      # each (B, 384)

        # Hierarchical prompts + Lorentz CAS
        hyp_out = self.lorentz(h_s, h_e, self.prompt_learner)

        return h_s, h_e, h_g, hyp_out

    # ── Evidence tokens ───────────────────────────────────────────────

    def build_evidence_tokens(self, h_s, h_e, h_g, hyp_out):
        """Compute 5 continuous evidence tokens (B, d_llm) each."""
        pre_s  = hyp_out['pre_s'].float()    # (B, 2)
        auth_s = hyp_out['auth_s'].float()
        pre_e  = hyp_out['pre_e'].float()
        auth_e = hyp_out['auth_e'].float()

        h_sf = h_s.float()
        h_ef = h_e.float()
        h_gf = h_g.float()

        d_s_P = self.adp_s_P(torch.cat([h_sf, pre_s],  dim=-1))  # (B, d_llm)
        d_s_A = self.adp_s_A(torch.cat([h_sf, auth_s], dim=-1))
        d_e_P = self.adp_e_P(torch.cat([h_ef, pre_e],  dim=-1))
        d_e_A = self.adp_e_A(torch.cat([h_ef, auth_e], dim=-1))
        d_g   = self.adp_g(h_gf)                                  # (B, d_llm)

        return d_s_P, d_s_A, d_e_P, d_e_A, d_g

    # ── Training forward (teacher forcing) ───────────────────────────

    def forward_train(self, audio: torch.Tensor,
                      y_s: torch.Tensor, y_e: torch.Tensor):
        """
        Full forward pass for training.
        audio: (B, T); y_s, y_e: (B,)
        Returns dict with all outputs needed for loss computation.
        """
        B = audio.shape[0]
        device = audio.device
        dtype  = next(self.llm.parameters()).dtype

        # Acoustic encoding
        h_s, h_e, h_g, hyp_out = self.acoustic_forward(audio)

        # Evidence tokens
        d_s_P, d_s_A, d_e_P, d_e_A, d_g = self.build_evidence_tokens(
            h_s, h_e, h_g, hyp_out
        )

        # Build LLM input sequence
        input_seq = self._build_input_seq(d_s_P, d_s_A, d_e_P, d_e_A, d_g)
        L_input = input_seq.shape[1]

        # Build target tokens (teacher forcing)
        tgt_ids = self._build_target_ids(y_s, y_e, device)  # (B, 6)

        # Embed target tokens and append to input
        embed     = self.llm.get_input_embeddings()
        tgt_emb   = embed(tgt_ids).to(dtype)                # (B, 6, d_llm)
        full_seq  = torch.cat([input_seq, tgt_emb], dim=1)  # (B, L_input+6, d_llm)

        # LLM forward
        attn_mask = torch.ones(B, full_seq.shape[1], dtype=torch.long, device=device)
        lm_out    = self.llm(
            inputs_embeds=full_seq,
            attention_mask=attn_mask,
            return_dict=True,
        )
        logits = lm_out.logits                               # (B, L_input+6, V)

        # Logits at target positions: shift by 1 (predicting next token)
        tgt_logits = logits[:, L_input - 1: L_input + 5, :]  # (B, 6, V)

        # Extract constrained logits for speech and scene (positions 4, 5)
        sph_ids = [self._tgt_ids[t] for t in ["<SPH=A>", "<SPH=R>", "<SPH=F>"]]
        scn_ids = [self._tgt_ids[t] for t in ["<SCN=A>", "<SCN=R>", "<SCN=F>"]]

        logit_s = tgt_logits[:, 4, :][:, sph_ids]  # (B, 3)
        logit_e = tgt_logits[:, 5, :][:, scn_ids]  # (B, 3)

        p_s    = F.softmax(logit_s, dim=-1)
        p_e    = F.softmax(logit_e, dim=-1)
        p_fake = 1.0 - (1.0 - p_s[:, 2]) * (1.0 - p_e[:, 2])

        return {
            # For LM loss
            'tgt_logits': tgt_logits,   # (B, 6, V)
            'tgt_ids':    tgt_ids,       # (B, 6) ground truth token IDs
            # Hyperbolic outputs
            'hyp_out':    hyp_out,
            # Final logits
            'logit_s':    logit_s,       # (B, 3)
            'logit_e':    logit_e,
            'p_fake':     p_fake,
            # Component reps (for invariance loss)
            'h_s': h_s, 'h_e': h_e, 'h_g': h_g,
        }

    # ── Inference forward ─────────────────────────────────────────────

    @torch.no_grad()
    def forward(self, audio: torch.Tensor):
        """
        Inference forward (greedy constrained decoding).
        Returns dict compatible with evaluate.py interface.
        """
        B      = audio.shape[0]
        device = audio.device
        dtype  = next(self.llm.parameters()).dtype

        h_s, h_e, h_g, hyp_out = self.acoustic_forward(audio)
        d_s_P, d_s_A, d_e_P, d_e_A, d_g = self.build_evidence_tokens(
            h_s, h_e, h_g, hyp_out
        )
        seq = self._build_input_seq(d_s_P, d_s_A, d_e_P, d_e_A, d_g)

        # 6-step constrained greedy decoding
        T = self._tgt_ids
        constraints = [
            [T["<SP=P>"],  T["<SP=A>"]],
            [T["<SA=R>"],  T["<SA=F>"],  T["<SA=N>"]],
            [T["<EP=P>"],  T["<EP=A>"]],
            [T["<EA=R>"],  T["<EA=F>"],  T["<EA=N>"]],
            [T["<SPH=A>"], T["<SPH=R>"], T["<SPH=F>"]],
            [T["<SCN=A>"], T["<SCN=R>"], T["<SCN=F>"]],
        ]

        embed = self.llm.get_input_embeddings()
        cur   = seq
        preds = []
        for voc in constraints:
            attn = torch.ones(B, cur.shape[1], dtype=torch.long, device=device)
            logits = self.llm(inputs_embeds=cur, attention_mask=attn,
                               return_dict=True).logits[:, -1, :]  # (B, V)
            mask = torch.full_like(logits, float('-inf'))
            mask[:, voc] = logits[:, voc]
            tok_ids = mask.argmax(-1)                               # (B,)
            preds.append(tok_ids)
            cur = torch.cat([cur, embed(tok_ids.unsqueeze(1)).to(dtype)], dim=1)

        # Decode final speech/scene state from positions 4,5
        sph_ids = [T["<SPH=A>"], T["<SPH=R>"], T["<SPH=F>"]]
        scn_ids = [T["<SCN=A>"], T["<SCN=R>"], T["<SCN=F>"]]

        # preds[4]: (B,) IDs of <SPH=A/R/F>
        pred_s = torch.tensor(
            [sph_ids.index(p.item()) for p in preds[4]], device=device
        )
        pred_e = torch.tensor(
            [scn_ids.index(p.item()) for p in preds[5]], device=device
        )

        # Build probability distributions from the last decoder step logits
        # (use the stored logits at step 4 and 5)
        # Re-run two final steps to get logits for probability output
        logit_s = torch.zeros(B, 3, device=device)
        logit_e = torch.zeros(B, 3, device=device)
        for b in range(B):
            for i, sid in enumerate(sph_ids):
                logit_s[b, i] = float(pred_s[b] == i)
        for b in range(B):
            for i, sid in enumerate(scn_ids):
                logit_e[b, i] = float(pred_e[b] == i)

        p_s    = F.one_hot(pred_s, 3).float()
        p_e    = F.one_hot(pred_e, 3).float()
        p_fake = 1.0 - (1.0 - p_s[:, 2]) * (1.0 - p_e[:, 2])

        return {
            'logit_s':  logit_s,
            'logit_e':  logit_e,
            'p_fake':   p_fake,
            'hyp_out':  hyp_out,
            'h_s': h_s, 'h_e': h_e, 'h_g': h_g,
        }
