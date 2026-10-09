"""
HARPER v2 Joint Loss:
  L = λ_I·L_inv + λ_D·L_eff + λ_P·L_pre + λ_A·L_auth + λ_E·L_ent + λ_L·L_LM

L_inv  : cosine distance between same-component pairs (intervention invariance)
L_eff  : cosine distance between same-intervention effect vectors
L_pre  : CE on component presence (binary: absent vs present)
L_auth : CE on authenticity, masked to present components only
L_ent  : hyperbolic entailment cone violations
L_LM   : autoregressive LM cross-entropy on structured 6-token target sequence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """D(a,b) = 1 - cosine_similarity(a,b).  (B, d) → (B,)"""
    return 1.0 - F.cosine_similarity(a, b, dim=-1)


class HARPERLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, out, y_s: torch.Tensor, y_e: torch.Tensor,
                quad=None, lorentz_module=None, prompt_learner=None,
                stage: str = 'joint'):
        """
        out   : dict from HARPERModel.forward_train
        y_s, y_e : (B,) labels in {0=A,1=R,2=F}
        quad  : optional dict with quadruple outputs for L_inv / L_eff
                keys: 'h_s_RR','h_s_RF','h_s_FR','h_s_FF',
                      'h_e_RR','h_e_RF','h_e_FR','h_e_FF'
        lorentz_module: for entailment loss
        stage : 'acoustic' | 'hyperbolic' | 'lm' | 'joint'
        Returns (total_loss, loss_dict).
        """
        cfg = self.cfg
        device = y_s.device
        losses = {}

        # ── L_pre ─────────────────────────────────────────────────────
        hyp = out['hyp_out']
        r_s = (y_s != 0).long()   # 1 if speech present
        r_e = (y_e != 0).long()
        L_pre = (F.cross_entropy(hyp['pre_s'], r_s)
               + F.cross_entropy(hyp['pre_e'], r_e))
        losses['L_pre'] = L_pre.item()

        # ── L_auth ────────────────────────────────────────────────────
        # auth labels: R→0, F→1; only meaningful for present components
        # We convert y_s ∈ {1,2} → {0,1}  (ignore y_s==0 with mask)
        mask_s = (y_s != 0).float()
        mask_e = (y_e != 0).float()
        # auth target: 0=R,1=F  (y_s-1 for present components, clamped)
        auth_tgt_s = (y_s - 1).clamp(min=0).long()  # {0,1}
        auth_tgt_e = (y_e - 1).clamp(min=0).long()
        L_auth = (
            (F.cross_entropy(hyp['auth_s'], auth_tgt_s, reduction='none') * mask_s).mean()
          + (F.cross_entropy(hyp['auth_e'], auth_tgt_e, reduction='none') * mask_e).mean()
        )
        losses['L_auth'] = L_auth.item()

        # ── L_inv + L_eff ─────────────────────────────────────────────
        if quad is not None:
            h_s_RR = quad['h_s_RR'];  h_s_RF = quad['h_s_RF']
            h_s_FR = quad['h_s_FR'];  h_s_FF = quad['h_s_FF']
            h_e_RR = quad['h_e_RR'];  h_e_RF = quad['h_e_RF']
            h_e_FR = quad['h_e_FR'];  h_e_FF = quad['h_e_FF']

            L_inv = (
                cosine_dist(h_s_RR, h_s_RF).mean()
              + cosine_dist(h_s_FR, h_s_FF).mean()
              + cosine_dist(h_e_RR, h_e_FR).mean()
              + cosine_dist(h_e_RF, h_e_FF).mean()
            )

            delta_s_R = h_s_FR - h_s_RR
            delta_s_F = h_s_FF - h_s_RF
            delta_e_R = h_e_RF - h_e_RR
            delta_e_F = h_e_FF - h_e_FR

            L_eff = (
                cosine_dist(delta_s_R, delta_s_F).mean()
              + cosine_dist(delta_e_R, delta_e_F).mean()
            )
        else:
            L_inv = torch.tensor(0., device=device)
            L_eff = torch.tensor(0., device=device)
        losses['L_inv'] = L_inv.item()
        losses['L_eff'] = L_eff.item()

        # ── L_ent ─────────────────────────────────────────────────────
        if lorentz_module is not None and stage in ('hyperbolic', 'lm', 'joint'):
            L_ent = self._entailment_loss(
                hyp, y_s, y_e, lorentz_module, prompt_learner
            )
        else:
            L_ent = torch.tensor(0., device=device)
        losses['L_ent'] = L_ent.item()

        # ── L_LM ──────────────────────────────────────────────────────
        if stage in ('lm', 'joint') and 'tgt_logits' in out:
            L_lm = self._lm_loss(out['tgt_logits'], out['tgt_ids'])
        else:
            L_lm = torch.tensor(0., device=device)
        losses['L_lm'] = L_lm.item()

        # ── Total ─────────────────────────────────────────────────────
        total = (
            cfg.lam_inv  * L_inv
          + cfg.lam_eff  * L_eff
          + cfg.lam_pre  * L_pre
          + cfg.lam_auth * L_auth
          + cfg.lam_ent  * L_ent
          + cfg.lam_lm   * L_lm
        )
        losses['total'] = total.item()
        return total, losses

    def _lm_loss(self, tgt_logits, tgt_ids):
        """
        tgt_logits: (B, 6, V) — logits at 6 target positions
        tgt_ids:    (B, 6)    — ground truth token IDs
        Positions 4,5 (final speech/scene) get higher weight.
        """
        B, T, V = tgt_logits.shape
        cfg = self.cfg

        # Per-position weights: [1,1,1,1, w_final, w_final]
        weights = torch.ones(T, device=tgt_ids.device)
        weights[4] = cfg.w_final_token
        weights[5] = cfg.w_final_token

        loss = 0.0
        for t in range(T):
            ce = F.cross_entropy(tgt_logits[:, t, :], tgt_ids[:, t])
            loss = loss + weights[t] * ce
        return loss / weights.sum()

    def _entailment_loss(self, hyp, y_s, y_e, lorentz, prompt_learner):
        """Enforce ancestor→descendant entailment for each sample's path."""
        B = y_s.shape[0]
        zp = hyp['zp']
        z_s = hyp['z_s']  # (B, d+1) on manifold
        z_e = hyp['z_e']

        total = torch.tensor(0., device=y_s.device)
        n = 0

        for b in range(B):
            s, e = y_s[b].item(), y_e[b].item()

            # Speech hierarchy path
            if s == 0:   # Absent: Component → Absent → z_s
                total = total + lorentz.entailment_loss(
                    z_s[b:b+1], zp['s_A'].unsqueeze(0))
            elif s == 1:  # Real: Present → Real → z_s
                total = total + lorentz.entailment_loss(
                    z_s[b:b+1], zp['s_P'].unsqueeze(0))
                total = total + lorentz.entailment_loss(
                    z_s[b:b+1], zp['s_R'].unsqueeze(0))
            else:         # Fake: Present → Fake → z_s
                total = total + lorentz.entailment_loss(
                    z_s[b:b+1], zp['s_P'].unsqueeze(0))
                total = total + lorentz.entailment_loss(
                    z_s[b:b+1], zp['s_F'].unsqueeze(0))

            # Scene hierarchy path
            if e == 0:
                total = total + lorentz.entailment_loss(
                    z_e[b:b+1], zp['e_A'].unsqueeze(0))
            elif e == 1:
                total = total + lorentz.entailment_loss(
                    z_e[b:b+1], zp['e_P'].unsqueeze(0))
                total = total + lorentz.entailment_loss(
                    z_e[b:b+1], zp['e_R'].unsqueeze(0))
            else:
                total = total + lorentz.entailment_loss(
                    z_e[b:b+1], zp['e_P'].unsqueeze(0))
                total = total + lorentz.entailment_loss(
                    z_e[b:b+1], zp['e_F'].unsqueeze(0))
            n += 1

        return total / max(n, 1)
