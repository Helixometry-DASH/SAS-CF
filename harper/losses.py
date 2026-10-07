"""
HARPER Training Objective:
  L = L_LM + λ_R·L_route + λ_E·L_tree + λ_D·L_dec + λ_G·L_geo

L_LM   : CE on final ALM predictions (p^L_s, p^L_e)
L_route: CE on routing assignments  (r_s = 1[y_s ≠ A], r_e = 1[y_e ≠ A])
L_tree : Hierarchical regularization of routing space
L_dec  : CE on hyperbolic distributions (p^H_s, p^H_e) + joint CAS (q^H)
L_geo  : JS divergence between product-hyperbolic (q^H) and ALM (q^L) distributions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import CAS_STATES, make_cas_index


class HARPERLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, out, y_s, y_e):
        """
        out : dict from HARPERModel.forward
        y_s : (B,) ground truth speech label   {0=A,1=R,2=F}
        y_e : (B,) ground truth scene label    {0=A,1=R,2=F}
        Returns (total_loss, loss_dict).
        """
        # ── L_LM ──────────────────────────────────────────────────────
        L_lm = (F.cross_entropy(out['logit_s'], y_s)
              + F.cross_entropy(out['logit_e'], y_e))

        # ── L_route ───────────────────────────────────────────────────
        # Routing supervision: r_s = 1[y_s ≠ A], r_e = 1[y_e ≠ A]
        # route_w: (B, N, n_cls) — routing weights for each patch token
        # Pooled routing label: mean weight over tokens should peak at y_s / y_e
        route_mean_s = out['route_w'].mean(dim=1)   # (B, n_cls)
        route_mean_e = out['route_w'].mean(dim=1)   # same routing; TODO: dual-branch
        # For tokens that "belong" to active components, push routing to correct class
        r_mask_s = (y_s != 0).float()               # 1 if speech present
        r_mask_e = (y_e != 0).float()
        L_route_s = (F.cross_entropy(route_mean_s, y_s, reduction='none') * r_mask_s).mean()
        L_route_e = (F.cross_entropy(route_mean_e, y_e, reduction='none') * r_mask_e).mean()
        L_route   = L_route_s + L_route_e

        # ── L_tree ────────────────────────────────────────────────────
        # Preserve CAS hierarchy in routing space:
        # Absent prototype should be distinguishable from R and F.
        # Push A prototype away from R and F in hyperbolic space.
        L_tree = self._tree_loss(out['proto_s']) + self._tree_loss(out['proto_e'])

        # ── L_dec ─────────────────────────────────────────────────────
        # Supervise hyperbolic distributions p^H_s, p^H_e and joint q^H
        L_dec_s = F.cross_entropy(out['p_H_s'], y_s)
        L_dec_e = F.cross_entropy(out['p_H_e'], y_e)
        # Joint CAS cross-entropy
        cas_idx = make_cas_index(y_s, y_e)           # (B,) index into 8 CAS states
        L_dec_joint = F.cross_entropy(out['q_H'], cas_idx)
        L_dec = L_dec_s + L_dec_e + L_dec_joint

        # ── L_geo ─────────────────────────────────────────────────────
        # JS divergence between q^H (hyperbolic) and q^L (ALM factored)
        # q^L(a,b) = p^L_s(a) * p^L_e(b) for 8 valid states
        q_L = self._factored_alm_dist(out['p_L_s'], out['p_L_e'])   # (B, 8)
        q_H = out['q_H'].detach()                     # treat q^H as reference
        L_geo = self._js_divergence(q_H, q_L)

        # ── Total ─────────────────────────────────────────────────────
        c = self.cfg
        total = (c.lam_lm    * L_lm
               + c.lam_route * L_route
               + c.lam_tree  * L_tree
               + c.lam_dec   * L_dec
               + c.lam_geo   * L_geo)

        return total, {
            'L_lm':    L_lm.item(),
            'L_route': L_route.item(),
            'L_tree':  L_tree.item(),
            'L_dec':   L_dec.item(),
            'L_geo':   L_geo.item(),
        }

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _tree_loss(proto_raw, margin=1.0):
        """
        Push A prototype (index 0) away from R (1) and F (2).
        Euclidean distance margin loss on raw (pre-exp_map) prototype params.
        """
        p_A = proto_raw[0]
        p_R = proto_raw[1]
        p_F = proto_raw[2]
        d_AR = (p_A - p_R).norm()
        d_AF = (p_A - p_F).norm()
        return F.relu(margin - d_AR) + F.relu(margin - d_AF)

    @staticmethod
    def _factored_alm_dist(p_L_s, p_L_e):
        """
        Factored joint distribution over 8 valid CAS states from ALM outputs.
        p_L_s: (B, n_cls), p_L_e: (B, n_cls)
        Returns q_L: (B, 8)
        """
        q = []
        for (a, b) in CAS_STATES:
            q.append(p_L_s[:, a] * p_L_e[:, b])   # (B,)
        q = torch.stack(q, dim=1)                   # (B, 8)
        # Re-normalize (in case joint doesn't sum to 1 exactly)
        return q / q.sum(dim=1, keepdim=True).clamp(min=1e-8)

    @staticmethod
    def _js_divergence(p, q, eps=1e-8):
        """Jensen-Shannon divergence JS(p||q) = 0.5*KL(p||M) + 0.5*KL(q||M)."""
        m = 0.5 * (p + q)
        kl_pm = (p * (p / m.clamp(min=eps)).clamp(min=eps).log()).sum(dim=-1)
        kl_qm = (q * (q / m.clamp(min=eps)).clamp(min=eps).log()).sum(dim=-1)
        return (0.5 * kl_pm + 0.5 * kl_qm).mean()
