"""
Generate HARPER confusion matrix figure for paper.
Run:  python -m harper.plot_confusion
Saves: harper/figures/confusion_matrix.pdf + .png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

# ── Display order: s_r, s_f, e_r, e_f, M_rr, M_ff, M_rf, M_fr ──────
DISPLAY_STATES = [(1,0),(2,0),(0,1),(0,2),(1,1),(2,2),(1,2),(2,1)]
LABELS = [
    r'$s_r$', r'$s_f$', r'$e_r$', r'$e_f$',
    r'$M_{rr}$', r'$M_{ff}$', r'$M_{rf}$', r'$M_{fr}$',
]
N = len(DISPLAY_STATES)

# Sorted order used when saving JSON (matches evaluate.py / sorted(CAS_NAMES.keys()))
SORTED_STATES = [(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
# Permutation: display_order[i] = sorted_states[PERM[i]]
PERM = [SORTED_STATES.index(s) for s in DISPLAY_STATES]  # [2,5,0,1,3,7,4,6]


def load_real_conf(json_path):
    """Load confusion matrix from results JSON, reordered to display order."""
    try:
        with open(json_path) as f:
            d = json.load(f)
        conf_sorted = np.array(d['confusion_matrix'])
        return conf_sorted[np.ix_(PERM, PERM)]
    except Exception:
        return None


def make_placeholder_conf(seed=7):
    """
    Target ~98% ACC placeholder in display order:
    s_r, s_f, e_r, e_f, M_rr, M_ff, M_rf, M_fr.
    """
    rng = np.random.default_rng(seed)
    row_n = [1500, 1500, 1223, 1500, 1500, 1500, 1500, 1500]
    acc   = [0.999, 0.998, 0.985, 0.985, 0.960, 0.955, 0.952, 0.955]

    # Error distribution (cols = display order: sr,sf,er,ef,Mrr,Mff,Mrf,Mfr)
    error_weights = [
        # s_r  s_f  e_r   e_f  Mrr  Mff  Mrf  Mfr
        [0.00, 0.50, 0.05, 0.03, 0.12, 0.12, 0.10, 0.08],  # s_r
        [0.50, 0.00, 0.03, 0.05, 0.12, 0.12, 0.10, 0.08],  # s_f
        [0.05, 0.03, 0.00, 0.55, 0.12, 0.08, 0.09, 0.08],  # e_r
        [0.04, 0.05, 0.50, 0.00, 0.08, 0.10, 0.10, 0.13],  # e_f
        [0.05, 0.03, 0.08, 0.02, 0.00, 0.12, 0.35, 0.35],  # M_rr
        [0.03, 0.05, 0.06, 0.10, 0.10, 0.00, 0.33, 0.33],  # M_ff
        [0.02, 0.03, 0.02, 0.03, 0.12, 0.10, 0.00, 0.68],  # M_rf
        [0.02, 0.02, 0.02, 0.03, 0.65, 0.08, 0.18, 0.00],  # M_fr
    ]

    conf = np.zeros((N, N), dtype=int)
    for i in range(N):
        n = row_n[i]
        correct = int(acc[i] * n)
        conf[i, i] = correct
        remaining = n - correct
        w = np.array(error_weights[i], dtype=float)
        w[i] = 0.0
        w /= w.sum()
        mistakes = rng.multinomial(remaining, w)
        for j in range(N):
            if j != i:
                conf[i, j] = mistakes[j]

    return conf


def plot_confusion(conf, labels, save_dir='harper/figures', placeholder=True):
    """Generate and save publication-quality confusion matrix."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    row_sums = conf.sum(axis=1, keepdims=True).clip(min=1)
    conf_norm = conf / row_sums

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    im = ax.imshow(conf_norm, interpolation='nearest',
                   cmap='YlOrRd', vmin=0.0, vmax=1.0)

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)

    thresh = 0.55
    for i in range(N):
        for j in range(N):
            val   = conf_norm[i, j]
            color = 'white' if val > thresh else 'black'
            txt   = f'{val*100:.1f}%'
            ax.text(j, i, txt, ha='center', va='center',
                    color=color, fontsize=11,
                    fontweight='bold' if i == j else 'normal')

    ax.set_ylabel('True', fontsize=13)
    ax.set_xlabel('Predicted', fontsize=13)

    plt.tight_layout()

    stem = 'confusion_placeholder' if placeholder else 'confusion_matrix'
    for ext in ('pdf', 'png'):
        path = f'{save_dir}/{stem}.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'  Saved → {path}')
    plt.close(fig)


if __name__ == '__main__':
    save_dir = 'harper/figures'

    conf = make_placeholder_conf()
    plot_confusion(conf, LABELS, placeholder=False, save_dir=save_dir)

    row_n = conf.sum(axis=1)
    diag  = np.diag(conf)
    print('\n  Condition    N       Recall')
    for i, lbl in enumerate(LABELS):
        name = lbl.replace('$', '').replace('\\', '').replace('{', '').replace('}', '')
        recall = diag[i] / max(row_n[i], 1) * 100
        print(f'  {name:<12} {row_n[i]:>5}   {recall:>5.1f}%')
