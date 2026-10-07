"""
HARPER — Hierarchical Audio Reasoning with Probe-Enhanced Representations
Paper: "Rethinking Audio Spoofing: When Authenticity Becomes Compositional"

Architecture (two-pass ALM):
  Pass I  : [T_inst | V | g_p | q_s | q_e | q_g] → h_s, h_e, h_g
  Routing : z_i = exp_0(W_E v_i); token update with global context h_g
  Hyp-CAS : h̃_s=[h_s;h_g] → z^D_s → p^H_s; same for scene; joint q^H
  Pass II : [T_inst | V̂ | g_p | d_s | d_e | d_se | <ANSWER>] → p^L_s, p^L_e
"""

import math
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

# PEFT 0.21.2 checks torch.distributed.tensor.DTensor at import time;
# some cluster PyTorch builds expose the hasattr but not the actual module.
import torch.distributed as _td
if not hasattr(_td, 'tensor'):
    _fake = types.ModuleType('torch.distributed.tensor')
    class _DTensor: pass
    _fake.DTensor = _DTensor
    _td.tensor = _fake

from .codec_bank import NACBank
from .config    import HARPERConfig

# ─────────────────────────────────────────────────────────────────────
# Poincaré ball helpers
# ─────────────────────────────────────────────────────────────────────

class PoincareOps(nn.Module):
    def __init__(self, c=1.0):
        super().__init__()
        self.c = c

    def exp_map(self, v, eps=1e-7):
        """Tangent vector at origin → point on ball."""
        sqrt_c  = math.sqrt(self.c)
        v_norm  = v.norm(dim=-1, keepdim=True).clamp(min=eps)
        tanh_in = (sqrt_c * v_norm).clamp(max=15.0)
        return torch.tanh(tanh_in) / (sqrt_c * v_norm) * v

    def mobius_add(self, x, y, eps=1e-7):
        c  = self.c
        xy = (x * y).sum(-1, keepdim=True)
        x2 = (x.pow(2)).sum(-1, keepdim=True)
        y2 = (y.pow(2)).sum(-1, keepdim=True)
        num = (1 + 2*c*xy + c*y2) * x + (1 - c*x2) * y
        den = (1 + 2*c*xy + c**2 * x2 * y2).clamp(min=eps)
        return num / den

    def dist(self, x, y, eps=1e-7):
        sqrt_c = math.sqrt(self.c)
        diff   = self.mobius_add(-x, y)
        d_norm = diff.norm(dim=-1).clamp(min=0., max=1. - eps)
        return (2 / sqrt_c) * torch.atanh(sqrt_c * d_norm)


# ─────────────────────────────────────────────────────────────────────
# Patch projector
# ─────────────────────────────────────────────────────────────────────

class PatchProjector(nn.Module):
    """
    Projects (1+K)*n_patches local time-frequency patches to LM hidden dim.
    Adds position, evidence-type, and probe-identity embeddings.
    """
    def __init__(self, patch_size, hidden, n_streams, max_patches=200):
        super().__init__()
        self.proj      = nn.Linear(patch_size, hidden)
        self.pos_embed = nn.Embedding(max_patches, hidden)
        # evidence-type: 0 = original, 1 = codec residual
        self.type_embed  = nn.Embedding(2, hidden)
        # probe-identity: 0 = original, 1..K = codec k
        self.codec_embed = nn.Embedding(n_streams, hidden)

    def forward(self, patches, n_streams, n_patches_per_stream):
        """
        patches: (B, N_total, patch_size) where N_total = n_streams * n_patches_per_stream
        """
        B, N, _ = patches.shape
        x = self.proj(patches)                          # (B, N, hidden)

        # positional index: 0..n_patches_per_stream-1 repeated n_streams times
        pos = torch.arange(n_patches_per_stream, device=x.device).repeat(n_streams)
        x   = x + self.pos_embed(pos).unsqueeze(0)

        # type index: 0 for original stream, 1 for all codec residual streams
        # streams: [S_x(0), R_1(1), R_2(1), ..., R_K(1)]
        types = torch.zeros(N, dtype=torch.long, device=x.device)
        types[n_patches_per_stream:] = 1
        x = x + self.type_embed(types).unsqueeze(0)

        # codec-identity index: stream k repeats n_patches_per_stream times
        codec_ids = torch.arange(n_streams, device=x.device).repeat_interleave(n_patches_per_stream)
        x = x + self.codec_embed(codec_ids).unsqueeze(0)

        return x  # (B, N, hidden)


# ─────────────────────────────────────────────────────────────────────
# Hyperbolic Router
# ─────────────────────────────────────────────────────────────────────

class HyperbolicRouter(nn.Module):
    """
    Maps each patch token to Poincaré ball; routes to A/R/F prototypes.
    Produces hierarchy-aware token update using global context.
    """
    def __init__(self, hidden, hyp_dim, n_cls=3, c=1.0):
        super().__init__()
        self.hyp   = PoincareOps(c)
        self.W_E   = nn.Linear(hidden, hyp_dim, bias=False)    # Euclidean → Poincaré
        self.W_G   = nn.Linear(hyp_dim, hidden, bias=False)    # Poincaré → Euclidean (for token update)
        self.w_g   = nn.Parameter(torch.randn(hidden) * 0.02)  # gate vector

    def route(self, v_embeds, prototypes):
        """
        v_embeds : (B, N, hidden)
        prototypes: (n_cls, hyp_dim) — raw (on Poincaré ball)
        Returns routing weights (B, N, n_cls) and hyperbolic tokens (B, N, hyp_dim).
        """
        z = self.hyp.exp_map(self.W_E(v_embeds))               # (B, N, hyp_dim)
        p = prototypes.unsqueeze(0).unsqueeze(0)                # (1, 1, n_cls, hyp_dim)
        z_exp = z.unsqueeze(2)                                  # (B, N, 1, hyp_dim)
        # Pairwise distances
        dists = self.hyp.dist(z_exp, p.expand(z.shape[0], z.shape[1], -1, -1))  # (B, N, n_cls)
        weights = F.softmax(-dists, dim=-1)                     # (B, N, n_cls)
        return weights, z

    def update_tokens(self, v, g_E):
        """
        v  : (B, N, hidden) — patch tokens
        g_E: (B, hyp_dim)   — global probe descriptor on Poincaré ball
        Returns V̂ (B, N, hidden)
        """
        # gate: σ(w_g · v_i) scalar per token
        gate   = torch.sigmoid((v * self.w_g).sum(-1, keepdim=True))   # (B, N, 1)
        g_lm   = self.W_G(g_E).unsqueeze(1)                            # (B, 1, hidden)
        return v + gate * g_lm


# ─────────────────────────────────────────────────────────────────────
# Product-Hyperbolic CAS
# ─────────────────────────────────────────────────────────────────────

# 8 valid CAS states: (A,R),(A,F),(R,A),(R,R),(R,F),(F,A),(F,R),(F,F)
CAS_STATES = [(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]

def make_cas_index(y_s, y_e):
    """Batch (y_s, y_e) → CAS state index for L_dec."""
    pair = list(zip(y_s.tolist(), y_e.tolist()))
    return torch.tensor([CAS_STATES.index(p) for p in pair],
                        dtype=torch.long, device=y_s.device)


class ProductHyperbolicCAS(nn.Module):
    """
    Two Poincaré balls (speech, scene) with learned {A,R,F} prototypes.
    Produces component distributions p^H_s, p^H_e and joint distribution q^H.
    """
    def __init__(self, hidden, hyp_dim, n_cls=3, c=1.0):
        super().__init__()
        self.hyp     = PoincareOps(c)
        self.n_cls   = n_cls
        # Input: [h_s; h_g] or [h_e; h_g] — 2*hidden → hyp_dim
        self.proj_s  = nn.Linear(2 * hidden, hyp_dim)
        self.proj_e  = nn.Linear(2 * hidden, hyp_dim)
        # Learnable prototypes on Poincaré ball (raw, mapped via exp_map)
        self.proto_s = nn.Parameter(torch.randn(n_cls, hyp_dim) * 0.01)
        self.proto_e = nn.Parameter(torch.randn(n_cls, hyp_dim) * 0.01)
        # Project decision vectors to LM hidden dim
        self.lm_s    = nn.Linear(hyp_dim, hidden)
        self.lm_e    = nn.Linear(hyp_dim, hidden)
        self.lm_se   = nn.Linear(2 * hyp_dim, hidden)

    def forward(self, h_s, h_e, h_g):
        """
        h_s, h_e, h_g: (B, hidden)
        Returns: z^D_s, z^D_e, p^H_s, p^H_e, q^H, d_s, d_e, d_se
        """
        # Augment with global context
        h_ts = torch.cat([h_s, h_g], dim=-1)   # (B, 2*hidden)
        h_te = torch.cat([h_e, h_g], dim=-1)

        z_s = self.hyp.exp_map(self.proj_s(h_ts))  # (B, hyp_dim) on Poincaré ball
        z_e = self.hyp.exp_map(self.proj_e(h_te))

        # Component distributions: softmax over -dist² to prototypes
        P_s = self.hyp.exp_map(self.proto_s)        # (n_cls, hyp_dim)
        P_e = self.hyp.exp_map(self.proto_e)

        d_s_vecs = z_s.unsqueeze(1) - P_s.unsqueeze(0)  # rough proxy
        # Proper hyperbolic distances
        p_H_s = self._component_dist(z_s, P_s)     # (B, n_cls)
        p_H_e = self._component_dist(z_e, P_e)     # (B, n_cls)

        # Joint distribution q^H over 8 valid CAS states
        q_H = self._joint_dist(z_s, z_e, P_s, P_e)  # (B, 8)

        # Decision tokens for ALM-II
        d_s_tok  = self.lm_s(z_s).unsqueeze(1)              # (B, 1, hidden)
        d_e_tok  = self.lm_e(z_e).unsqueeze(1)
        z_se     = torch.cat([z_s, z_e], dim=-1)             # (B, 2*hyp_dim)
        d_se_tok = self.lm_se(z_se).unsqueeze(1)             # (B, 1, hidden)

        return (z_s, z_e, p_H_s, p_H_e, q_H,
                d_s_tok, d_e_tok, d_se_tok,
                P_s, P_e)

    def _component_dist(self, z, P):
        """(B, hyp_dim), (n_cls, hyp_dim) → (B, n_cls) softmax distribution."""
        B = z.shape[0]
        z_exp = z.unsqueeze(1).expand(B, P.shape[0], -1)
        P_exp = P.unsqueeze(0).expand(B, -1, -1)
        dists = self.hyp.dist(z_exp, P_exp)          # (B, n_cls)
        return F.softmax(-dists, dim=-1)

    def _joint_dist(self, z_s, z_e, P_s, P_e, lam_s=1.0, lam_e=1.0):
        """Compute product-hyperbolic joint distribution over 8 valid CAS states."""
        B = z_s.shape[0]
        log_p = []
        for (a, b) in CAS_STATES:
            d_s = self.hyp.dist(z_s, P_s[a].unsqueeze(0).expand(B, -1))   # (B,)
            d_e = self.hyp.dist(z_e, P_e[b].unsqueeze(0).expand(B, -1))   # (B,)
            log_p.append(-(lam_s * d_s**2 + lam_e * d_e**2))
        log_p = torch.stack(log_p, dim=1)   # (B, 8)
        return F.softmax(log_p, dim=-1)


# ─────────────────────────────────────────────────────────────────────
# Full HARPER model
# ─────────────────────────────────────────────────────────────────────

class HARPERModel(nn.Module):
    def __init__(self, cfg: HARPERConfig):
        super().__init__()
        self.cfg = cfg

        # ── NAC bank ────────────────────────────────────────────────
        self.nac = NACBank(cfg)

        # ── Patch projector ─────────────────────────────────────────
        patch_size = cfg.patch_frames * cfg.n_mel          # 1280
        n_streams  = 1 + len(cfg.codec_names)              # original + K codecs
        # max_patches: upper bound on n_patches_per_stream
        max_frames = int(cfg.max_audio_s * cfg.sr / cfg.hop) + 8
        max_pps    = max_frames // cfg.patch_frames + 4
        self.patch_proj = PatchProjector(patch_size, cfg.hidden, n_streams,
                                          max_patches=max_pps)

        # ── Global pooling projection (R(x) only → R^hidden) ────────
        # Projects global mean of residual patches to hidden
        self.global_proj = nn.Linear(patch_size, cfg.hidden)

        # ── Hyperbolic router ────────────────────────────────────────
        self.router  = HyperbolicRouter(cfg.hidden, cfg.hyp_dim, cfg.N_CLS, cfg.hyp_c)
        # Global routing prototypes (shared; used for routing the global token)
        self.proto_g = nn.Parameter(torch.randn(cfg.N_CLS, cfg.hyp_dim) * 0.01)

        # ── Product-hyperbolic CAS ───────────────────────────────────
        self.cas = ProductHyperbolicCAS(cfg.hidden, cfg.hyp_dim, cfg.N_CLS, cfg.hyp_c)

        # ── ALM query tokens (ALM-I) ─────────────────────────────────
        self.q_speech = nn.Parameter(torch.randn(1, 1, cfg.hidden) * 0.02)
        self.q_scene  = nn.Parameter(torch.randn(1, 1, cfg.hidden) * 0.02)
        self.q_global = nn.Parameter(torch.randn(1, 1, cfg.hidden) * 0.02)

        # ── ALM answer token (ALM-II) ────────────────────────────────
        self.ans_token = nn.Parameter(torch.randn(1, 1, cfg.hidden) * 0.02)

        # ── Classification heads ─────────────────────────────────────
        self.cls_s = nn.Linear(cfg.hidden, cfg.N_CLS)
        self.cls_e = nn.Linear(cfg.hidden, cfg.N_CLS)

        # ── Backbone + LoRA ──────────────────────────────────────────
        self._build_backbone(cfg)

    # ── backbone construction ──────────────────────────────────────

    def _build_backbone(self, cfg):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import get_peft_model, LoraConfig

        print(f"[HARPER] Loading {cfg.backbone_id} ...")
        dtype = torch.float16 if cfg.fp16 else torch.float32
        base  = AutoModelForCausalLM.from_pretrained(
            cfg.backbone_id,
            torch_dtype=dtype,
            device_map=None,
        )
        lora_cfg = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
            target_modules=cfg.lora_target,
            bias="none",
        )
        self.backbone = get_peft_model(base, lora_cfg)
        self.backbone.print_trainable_parameters()

        tok = AutoTokenizer.from_pretrained(cfg.backbone_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.tokenizer = tok

        # Pre-compute instruction embeddings once (they don't change)
        self._inst_ids = None

    # ── forward utilities ─────────────────────────────────────────

    def _inst_embeds(self, B, device, dtype):
        """Instruction token embeddings, expanded to batch."""
        if self._inst_ids is None:
            text = "Analyze and classify speech and acoustic scene authenticity:"
            ids  = self.tokenizer(text, return_tensors='pt',
                                   add_special_tokens=False).input_ids
            self._inst_ids = ids
        ids   = self._inst_ids.to(device)
        emb   = self.backbone.get_input_embeddings()(ids)   # (1, L, H)
        return emb.to(dtype).expand(B, -1, -1)              # (B, L, H)

    def _alm_forward(self, embeds):
        """Run LM backbone on sequence of embeddings, return last hidden states."""
        B, L, _ = embeds.shape
        device  = embeds.device
        mask    = torch.ones(B, L, dtype=torch.long, device=device)
        outputs = self.backbone(
            inputs_embeds=embeds,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[-1]    # (B, L, H)

    # ── main forward pass ─────────────────────────────────────────

    def forward(self, audio):
        """
        audio: (B, T_audio) raw waveform @ cfg.sr
        Returns dict of all intermediate and final outputs.
        """
        B      = audio.shape[0]
        device = audio.device
        dtype  = next(self.backbone.parameters()).dtype

        # cast audio to backbone dtype for downstream ops
        audio  = audio.to(dtype)

        # ── 1. NAC bank ───────────────────────────────────────────────
        # patches: (B, (1+K)*n_p, patch_size)
        patches = self.nac(audio)
        n_streams   = self.nac.n_streams
        n_p_total   = patches.shape[1]
        n_p         = n_p_total // n_streams   # patches per stream

        # ── 2. Patch projection ───────────────────────────────────────
        # V: (B, N, hidden)
        V = self.patch_proj(patches.to(dtype), n_streams, n_p)

        # ── 3. Global descriptor g_p = Pool(residual patches) ────────
        # Use only codec-residual streams (skip stream 0 = original)
        res_patches = patches[:, n_p:, :]          # (B, K*n_p, patch_size)
        g_p = self.global_proj(res_patches.to(dtype).mean(dim=1))  # (B, hidden)
        g_p_tok = g_p.unsqueeze(1)                 # (B, 1, hidden)

        # ── 4. ALM-I pass ─────────────────────────────────────────────
        inst = self._inst_embeds(B, device, dtype)
        qs   = self.q_speech.to(dtype).expand(B, -1, -1)
        qe   = self.q_scene.to(dtype).expand(B, -1, -1)
        qg   = self.q_global.to(dtype).expand(B, -1, -1)

        alm1_in  = torch.cat([inst, V, g_p_tok, qs, qe, qg], dim=1)
        alm1_out = self._alm_forward(alm1_in)      # (B, L1, hidden)

        # Extract query hidden states (last 3 positions)
        h_s = alm1_out[:, -3, :].float()           # speech
        h_e = alm1_out[:, -2, :].float()           # scene
        h_g = alm1_out[:, -1, :].float()           # global

        # ── 5. Hyperbolic routing + token update ─────────────────────
        # Global embedding on Poincaré ball
        g_E = self.router.hyp.exp_map(
            self.router.W_E(g_p.float()))           # (B, hyp_dim)

        # Routing of all patch tokens (for routing loss)
        P_g    = self.router.hyp.exp_map(self.proto_g)  # (N_CLS, hyp_dim)
        route_w, z_tokens = self.router.route(V.float(), P_g)  # (B, N, 3), (B, N, hyp_dim)

        # Hierarchy-aware token update: V̂
        V_hat = self.router.update_tokens(V.float(), g_E)  # (B, N, hidden)
        V_hat = V_hat.to(dtype)

        # ── 6. Product-hyperbolic CAS ─────────────────────────────────
        (z_s, z_e, p_H_s, p_H_e, q_H,
         d_s_tok, d_e_tok, d_se_tok,
         P_s, P_e) = self.cas(h_s, h_e, h_g)

        d_s_tok  = d_s_tok.to(dtype)
        d_e_tok  = d_e_tok.to(dtype)
        d_se_tok = d_se_tok.to(dtype)

        # ── 7. ALM-II pass ────────────────────────────────────────────
        ans = self.ans_token.to(dtype).expand(B, -1, -1)

        alm2_in  = torch.cat([inst, V_hat, g_p_tok,
                               d_s_tok, d_e_tok, d_se_tok, ans], dim=1)
        alm2_out = self._alm_forward(alm2_in)      # (B, L2, hidden)

        h_ans    = alm2_out[:, -1, :].float()      # (B, hidden) — ANSWER position

        logit_s  = self.cls_s(h_ans)               # (B, N_CLS)
        logit_e  = self.cls_e(h_ans)               # (B, N_CLS)

        # ── 8. P_fake ─────────────────────────────────────────────────
        p_s = F.softmax(logit_s, dim=-1)
        p_e = F.softmax(logit_e, dim=-1)
        p_fake = 1.0 - (1.0 - p_s[:, 2]) * (1.0 - p_e[:, 2])  # 1-(1-P_F_s)(1-P_F_e)

        return {
            # Final predictions
            'logit_s':  logit_s,
            'logit_e':  logit_e,
            'p_fake':   p_fake,
            # Hyperbolic component distributions
            'p_H_s':    p_H_s,
            'p_H_e':    p_H_e,
            'q_H':      q_H,
            # LM component distribution (factored)
            'p_L_s':    p_s,
            'p_L_e':    p_e,
            # Routing outputs
            'route_w':  route_w,
            # Prototypes (for geometry loss)
            'proto_s':  P_s,
            'proto_e':  P_e,
        }
