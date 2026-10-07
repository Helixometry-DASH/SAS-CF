"""
Generate HARPER training curves (EER line chart) for paper.
Run:  python -m harper.plot_curves
Saves: harper/figures/eer_curve.pdf + .png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import json


# ── Known values from training log ───────────────────────────────────
EPOCHS = [1, 2, 3]

# Validation EER per epoch (from training log — confirmed)
VAL_EER = [12.48, 9.49, 9.42]
VAL_ACC = [85.22, 90.52, 90.91]

# Placeholder EER for unseen sets at final epoch (update when job 10109178 finishes)
# These are representative values — will be replaced with real measurements
UNSEEN1_EER_PLACEHOLDER = [None, None, 18.7]   # DAC codec
UNSEEN2_EER_PLACEHOLDER = [None, None, 22.4]   # m_f binary
UNSEEN_SS_EER_PLACEHOLDER = [None, None, 15.1]  # Spk/Scene


def load_results(json_path):
    """Load real EER values from detailed_results.json if available."""
    try:
        with open(json_path) as f:
            d = json.load(f)
        return d
    except Exception:
        return None


def plot_eer_curve(save_dir='harper/figures', placeholder=True):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    colors = {
        'sasCF':    '#1f77b4',
        'unseen1':  '#ff7f0e',
        'unseen2':  '#2ca02c',
        'unseen_ss':'#9467bd',
    }

    # ── SAS-CF val EER (confirmed) ───────────────────────────────────
    ax.plot(EPOCHS, VAL_EER, '-o', color=colors['sasCF'],
            linewidth=2, markersize=7, label='SAS-CF (in-domain)', zorder=5)
    for x, y in zip(EPOCHS, VAL_EER):
        ax.annotate(f'{y:.2f}%', (x, y), textcoords='offset points',
                    xytext=(6, 4), fontsize=8.5, color=colors['sasCF'])

    # ── Unseen sets — placeholder final-epoch points ─────────────────
    if placeholder:
        unseen_pts = [
            (UNSEEN1_EER_PLACEHOLDER[-1],  colors['unseen1'],  'Unseen 1 (DAC codec)'),
            (UNSEEN2_EER_PLACEHOLDER[-1],  colors['unseen2'],  'Unseen 2 ($m_f$ binary)'),
            (UNSEEN_SS_EER_PLACEHOLDER[-1],colors['unseen_ss'],'Unseen (Spk/Scene)'),
        ]
        for val, col, lbl in unseen_pts:
            ax.plot(3, val, 's', color=col, markersize=9,
                    label=f'{lbl} [placeholder]', zorder=5)
            ax.annotate(f'{val:.1f}%', (3, val), textcoords='offset points',
                        xytext=(6, 4), fontsize=8.5, color=col)

    # ── Formatting ───────────────────────────────────────────────────
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('EER (%)', fontsize=13)
    ax.set_xticks(EPOCHS)
    ax.set_xticklabels([f'Epoch {e}' for e in EPOCHS], fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    ax.tick_params(axis='y', labelsize=11)

    # Highlight best epoch
    best_ep  = EPOCHS[np.argmin(VAL_EER)]
    best_eer = min(VAL_EER)
    ax.axvline(best_ep, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    ax.text(best_ep + 0.05, ax.get_ylim()[1] * 0.98,
            f'Best\n{best_eer:.2f}%', fontsize=8, color='gray', va='top')

    ax.set_xlim(0.7, 3.6)
    ax.set_ylim(bottom=max(0, min(VAL_EER) - 3))
    ax.grid(axis='y', alpha=0.35, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(fontsize=10, loc='upper right', framealpha=0.85)

    plt.tight_layout()
    stem = 'eer_curve_placeholder' if placeholder else 'eer_curve'
    for ext in ('pdf', 'png'):
        path = f'{save_dir}/{stem}.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'  Saved → {path}')
    plt.close(fig)


def plot_acc_curve(save_dir='harper/figures'):
    """Secondary: ACC over epochs alongside EER on twin axes."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(6.0, 3.8))
    ax2 = ax1.twinx()

    color_eer = '#1f77b4'
    color_acc = '#d62728'

    l1, = ax1.plot(EPOCHS, VAL_EER, '-o', color=color_eer,
                   linewidth=2, markersize=7, label='val EER (%)')
    l2, = ax2.plot(EPOCHS, VAL_ACC, '-s', color=color_acc,
                   linewidth=2, markersize=7, label='val ACC (%)')

    for x, y in zip(EPOCHS, VAL_EER):
        ax1.annotate(f'{y:.2f}%', (x, y), textcoords='offset points',
                     xytext=(-22, 6), fontsize=8.5, color=color_eer)
    for x, y in zip(EPOCHS, VAL_ACC):
        ax2.annotate(f'{y:.2f}%', (x, y), textcoords='offset points',
                     xytext=(5, -14), fontsize=8.5, color=color_acc)

    ax1.set_xlabel('Epoch', fontsize=13)
    ax1.set_ylabel('EER (%) ↓', fontsize=13, color=color_eer)
    ax2.set_ylabel('ACC (%) ↑', fontsize=13, color=color_acc)
    ax1.tick_params(axis='y', labelcolor=color_eer, labelsize=11)
    ax2.tick_params(axis='y', labelcolor=color_acc, labelsize=11)
    ax1.set_xticks(EPOCHS)
    ax1.set_xticklabels([f'Epoch {e}' for e in EPOCHS], fontsize=11)
    ax1.set_xlim(0.7, 3.3)
    ax1.grid(axis='y', alpha=0.3, linestyle=':')
    ax1.spines['top'].set_visible(False)

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=10, loc='center right')

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        path = f'{save_dir}/eer_acc_curve.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'  Saved → {path}')
    plt.close(fig)


if __name__ == '__main__':
    results_json = '/mnt/scratch2/users/gmadaan/harper_checkpoints_full/detailed_results.json'
    results = load_results(results_json)
    placeholder = results is None

    print('[EER curve]')
    plot_eer_curve(placeholder=placeholder)

    print('[EER+ACC curve]')
    plot_acc_curve()

    print('\nDone. Re-run after job 10109178 finishes to update unseen EER values.')
