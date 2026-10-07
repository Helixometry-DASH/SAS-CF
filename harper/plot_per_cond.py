"""
Generate HARPER per-condition Joint_ACC bar chart for paper.
Run:  python -m harper.plot_per_cond
Saves: harper/figures/per_cond_acc.pdf + .png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Display order: s_r, s_f, e_r, e_f, M_rr, M_ff, M_rf, M_fr
DISPLAY_LABELS = [
    r'$s_r$', r'$s_f$', r'$e_r$', r'$e_f$',
    r'$M_{rr}$', r'$M_{ff}$', r'$M_{rf}$', r'$M_{fr}$',
]

# Keys matching per_cond dict from evaluate.py
COND_KEYS = [
    's_r  (R,A)', 's_f  (F,A)', 'e_r  (A,R)', 'e_f  (A,F)',
    'M_rr (R,R)', 'M_ff (F,F)', 'M_rf (R,F)', 'M_fr (F,R)',
]

# Fallback: values from job 10109178 (epoch 3)
REAL_JOINT_ACC = [99.7, 99.5, 95.3, 95.9, 76.1, 72.9, 60.8, 66.6]

# Colors: speech=blue, env=green, mixed=orange/red
BAR_COLORS = [
    '#4878CF', '#4878CF',   # s_r, s_f  — speech
    '#6ACC65', '#6ACC65',   # e_r, e_f  — env
    '#D65F5F', '#B47CC7',   # M_rr, M_ff — mixed real/real, fake/fake
    '#C4AD66', '#77BEDB',   # M_rf, M_fr — mixed cross
]


def load_per_cond(json_path):
    try:
        with open(json_path) as f:
            d = json.load(f)
        pc = d.get('sasCF_per_cond', {})
        vals = []
        for key in COND_KEYS:
            if key in pc:
                vals.append(pc[key]['joint_acc'])
            else:
                return None
        return vals
    except Exception:
        return None


def plot_per_cond(joint_acc, labels, save_dir='harper/figures', placeholder=False):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    bars = ax.bar(x, joint_acc, color=BAR_COLORS, width=0.62,
                  edgecolor='white', linewidth=0.8, zorder=3)

    for bar, val in zip(bars, joint_acc):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel('Joint Accuracy (%)', fontsize=13)
    ax.set_ylim(0, 110)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(axis='y', alpha=0.35, linestyle=':', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    stem = 'per_cond_placeholder' if placeholder else 'per_cond_acc'
    for ext in ('pdf', 'png'):
        path = f'{save_dir}/{stem}.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'  Saved → {path}')
    plt.close(fig)


if __name__ == '__main__':
    results_json = '/mnt/scratch2/users/gmadaan/harper_checkpoints_full/detailed_results.json'
    save_dir     = 'harper/figures'

    real_vals = load_per_cond(results_json)
    if real_vals is not None:
        print('[plot] Using real per-condition values from results JSON')
        plot_per_cond(real_vals, DISPLAY_LABELS, save_dir=save_dir, placeholder=False)
    else:
        print('[plot] Using hardcoded values from job 10109178')
        plot_per_cond(REAL_JOINT_ACC, DISPLAY_LABELS, save_dir=save_dir, placeholder=False)
