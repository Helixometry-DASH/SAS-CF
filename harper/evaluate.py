"""
HARPER Evaluation — comprehensive metrics on SAS-CF test splits.

Metrics:
  EER       : joint binary — any F component is fake
  EER_s     : speech-only binary — y_s=F vs y_s∈{A,R}
  EER_e     : env-only binary — y_e=F vs y_e∈{A,R}
  ACC       : mean(ACC_s, ACC_e) — component-level 3-class
  ACC_s     : speech 3-class (A/R/F)
  ACC_e     : env 3-class (A/R/F)
  Joint_ACC : both y_s AND y_e correct simultaneously
  Per-cond  : all above broken down per (y_s, y_e) group
  Confusion : 8×8 joint CAS state confusion matrix
"""

import torch
from torch.amp import autocast
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict


# ── Label helpers ────────────────────────────────────────────────────

CAS_NAMES = {
    (0, 1): 'e_r  (A,R)',
    (0, 2): 'e_f  (A,F)',
    (1, 0): 's_r  (R,A)',
    (1, 1): 'M_rr (R,R)',
    (1, 2): 'M_rf (R,F)',
    (2, 0): 's_f  (F,A)',
    (2, 1): 'M_fr (F,R)',
    (2, 2): 'M_ff (F,F)',
}

LABEL_STR = {0: 'A', 1: 'R', 2: 'F'}


def binary_label(y_s, y_e):
    """Any fake component → 1 (fake); no F → 0 (bonafide)."""
    return int(y_s == 2 or y_e == 2)


def speech_fake_label(y_s):
    """Speech fake binary: y_s=F → 1, else → 0."""
    return int(y_s == 2)


def env_fake_label(y_e):
    """Env fake binary: y_e=F → 1, else → 0."""
    return int(y_e == 2)


def compute_eer(scores, labels, pos_label=1):
    """EER from continuous scores and binary labels (higher = more likely pos)."""
    from sklearn.metrics import roc_curve
    if len(set(labels)) < 2:
        return float('nan')
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=pos_label)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(0.5 * (fpr[idx] + fnr[idx]))


# ── Core inference pass ───────────────────────────────────────────────

@torch.no_grad()
def _run_inference(model, file_list, cfg, batch_size, device):
    """
    Run model on file_list; return arrays of per-sample outputs.
    Returns dict of numpy arrays, all length N.
    """
    from .dataset import SASCFDataset

    model.eval()
    ds     = SASCFDataset(file_list, cfg.sr, cfg.max_audio_s, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4,
                        pin_memory=True, drop_last=False)

    pfake_all, ps_all, pe_all = [], [], []
    pred_s_all, pred_e_all    = [], []
    true_s_all, true_e_all    = [], []

    for wav, y_s, y_e in loader:
        wav = wav.to(device).float()
        with autocast('cuda', enabled=cfg.fp16):
            out = model(wav)

        pfake_all.extend(out['p_fake'].cpu().float().tolist())
        # logit_s / logit_e are (B,3) — argmax gives predicted class
        ps_all.extend(out['logit_s'].softmax(-1)[:, 2].cpu().float().tolist())
        pe_all.extend(out['logit_e'].softmax(-1)[:, 2].cpu().float().tolist())

        pred_s_all.extend(out['logit_s'].argmax(-1).cpu().tolist())
        pred_e_all.extend(out['logit_e'].argmax(-1).cpu().tolist())
        true_s_all.extend(y_s.tolist())
        true_e_all.extend(y_e.tolist())

    return {
        'pfake':  np.array(pfake_all),
        'ps':     np.array(ps_all),
        'pe':     np.array(pe_all),
        'pred_s': np.array(pred_s_all),
        'pred_e': np.array(pred_e_all),
        'true_s': np.array(true_s_all),
        'true_e': np.array(true_e_all),
    }


def _compute_metrics(r):
    """Compute all metrics from inference result dict."""
    N = len(r['true_s'])
    labels_joint = np.array([binary_label(s, e)
                              for s, e in zip(r['true_s'], r['true_e'])])
    labels_s = np.array([speech_fake_label(s) for s in r['true_s']])
    labels_e = np.array([env_fake_label(e)    for e in r['true_e']])

    correct_s    = (r['pred_s'] == r['true_s']).sum()
    correct_e    = (r['pred_e'] == r['true_e']).sum()
    correct_both = ((r['pred_s'] == r['true_s']) & (r['pred_e'] == r['true_e'])).sum()

    return {
        'n':         N,
        'eer':       compute_eer(r['pfake'], labels_joint) * 100,
        'eer_s':     compute_eer(r['ps'],    labels_s)     * 100,
        'eer_e':     compute_eer(r['pe'],    labels_e)     * 100,
        'acc_s':     correct_s    / N * 100,
        'acc_e':     correct_e    / N * 100,
        'acc':       (correct_s + correct_e) / (2 * N) * 100,
        'joint_acc': correct_both / N * 100,
    }


# ── Public API ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, file_list, cfg, batch_size=32, device='cuda'):
    """
    Full evaluation: joint + component EERs, ACC, joint_ACC.
    Returns metrics dict.
    """
    r = _run_inference(model, file_list, cfg, batch_size, device)
    return _compute_metrics(r)


@torch.no_grad()
def evaluate_detailed(model, file_list, cfg, batch_size=32, device='cuda'):
    """
    Full evaluation with per-condition breakdown and confusion matrix.
    Returns (overall_metrics, per_cond_metrics, confusion_matrix).
    """
    r       = _run_inference(model, file_list, cfg, batch_size, device)
    overall = _compute_metrics(r)

    # ── Per-condition breakdown ──────────────────────────────────────
    per_cond = {}
    unique_states = set(zip(r['true_s'].tolist(), r['true_e'].tolist()))
    for (ys, ye) in sorted(unique_states):
        mask = (r['true_s'] == ys) & (r['true_e'] == ye)
        if mask.sum() == 0:
            continue
        sub = {k: v[mask] for k, v in r.items()}
        m   = _compute_metrics(sub)
        name = CAS_NAMES.get((ys, ye), f'({ys},{ye})')
        per_cond[name] = m

    # ── Confusion matrix over 8 joint CAS states ─────────────────────
    states     = sorted(CAS_NAMES.keys())
    state_idx  = {s: i for i, s in enumerate(states)}
    n_states   = len(states)
    conf       = np.zeros((n_states, n_states), dtype=int)
    for ps, pe, ts, te in zip(r['pred_s'], r['pred_e'], r['true_s'], r['true_e']):
        true_key = (int(ts), int(te))
        pred_key = (int(ps), int(pe))
        if true_key in state_idx and pred_key in state_idx:
            conf[state_idx[true_key], state_idx[pred_key]] += 1

    return overall, per_cond, conf, states


def print_detailed_results(overall, per_cond, conf, states, title=''):
    """Pretty-print full evaluation results."""
    w = 62
    if title:
        print(f'\n{"═"*w}')
        print(f'  {title}')
    print(f'{"─"*w}')

    # Overall
    print(f'  OVERALL  (N={overall["n"]:,})')
    print(f'  {"Metric":<14} {"Value":>8}')
    print(f'  {"─"*24}')
    print(f'  {"EER (joint)":<14} {overall["eer"]:>7.2f}%')
    print(f'  {"EER_s (speech)":<14} {overall["eer_s"]:>7.2f}%')
    print(f'  {"EER_e (env)":<14} {overall["eer_e"]:>7.2f}%')
    print(f'  {"ACC":<14} {overall["acc"]:>7.2f}%')
    print(f'  {"ACC_s":<14} {overall["acc_s"]:>7.2f}%')
    print(f'  {"ACC_e":<14} {overall["acc_e"]:>7.2f}%')
    print(f'  {"Joint_ACC":<14} {overall["joint_acc"]:>7.2f}%')

    # Per-condition
    print(f'\n{"─"*w}')
    print(f'  {"Condition":<14} {"N":>6} {"EER":>7} {"EER_s":>7} '
          f'{"EER_e":>7} {"ACC_s":>7} {"ACC_e":>7} {"Jnt_ACC":>8}')
    print(f'  {"─"*60}')
    for name, m in per_cond.items():
        eer_s = f'{m["eer_s"]:>6.1f}%' if not np.isnan(m['eer_s']) else '    n/a'
        eer_e = f'{m["eer_e"]:>6.1f}%' if not np.isnan(m['eer_e']) else '    n/a'
        eer   = f'{m["eer"]:>6.1f}%'   if not np.isnan(m['eer'])   else '    n/a'
        print(f'  {name:<14} {m["n"]:>6,} {eer:>7} {eer_s:>7} '
              f'{eer_e:>7} {m["acc_s"]:>6.1f}% {m["acc_e"]:>6.1f}% '
              f'{m["joint_acc"]:>7.1f}%')

    # Confusion matrix
    print(f'\n{"─"*w}')
    print('  CONFUSION MATRIX  (row=true, col=pred)')
    labels = [CAS_NAMES.get(s, str(s)).split()[0] for s in states]
    col_w  = 7
    print('  ' + ' ' * 14 + ''.join(f'{l:>{col_w}}' for l in labels))
    for i, s in enumerate(states):
        row_name = CAS_NAMES.get(s, str(s))
        row = conf[i]
        total = row.sum()
        print('  ' + f'{row_name:<14}' +
              ''.join(f'{v:>{col_w}}' for v in row) +
              f'  | {total:>5}')
    print(f'{"─"*w}')
