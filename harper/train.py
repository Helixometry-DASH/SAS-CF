"""
HARPER Training Script.
Usage:
  python -m harper.train [--config_overrides key=val ...]

Training details (from paper):
  - 3 epochs, AdamW, batch=32, lr=1e-5
  - LoRA r=8, scale=32 on q,v projections of Qwen2-0.5B-Instruct
  - NAC bank frozen throughout
  - Mixed precision (fp16)
"""

import os
import sys
import math
import time
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent.parent))

from harper.config  import HARPERConfig
from harper.model   import HARPERModel
from harper.dataset import SASCFDataset, build_splits
from harper.losses  import HARPERLoss
from harper.evaluate import evaluate


# ─────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────

def cosine_lr(optimizer, step, warmup_steps, total_steps, min_lr=1e-7):
    if step < warmup_steps:
        scale = step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        scale    = 0.5 * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = max(min_lr, pg['lr_base'] * scale)


def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6


def save_ckpt(model, optimizer, epoch, step, val_eer, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'epoch{epoch}_step{step}_eer{val_eer:.2f}.pt')
    torch.save({
        'epoch':   epoch,
        'step':    step,
        'val_eer': val_eer,
        'model':   model.state_dict(),
        'opt':     optimizer.state_dict(),
    }, path)
    print(f"  [ckpt] saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────

def train(cfg: HARPERConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[train] device={device}  fp16={cfg.fp16}")

    # ── Data ──────────────────────────────────────────────────────────
    print("\n[data] scanning SAS-CF ...")
    train_files, val_files, test_files = build_splits(cfg)

    train_ds = SASCFDataset(train_files, cfg.sr, cfg.max_audio_s, augment=True)
    val_ds   = SASCFDataset(val_files,   cfg.sr, cfg.max_audio_s, augment=False)
    test_ds  = SASCFDataset(test_files,  cfg.sr, cfg.max_audio_s, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, num_workers=4,
        pin_memory=True, drop_last=False,
    )

    # ── Model ─────────────────────────────────────────────────────────
    print("\n[model] building HARPER ...")
    model = HARPERModel(cfg).to(device)
    total_m, train_m = count_params(model)
    print(f"  Total: {total_m:.1f}M   Trainable: {train_m:.3f}M")

    # Freeze NAC bank (codec probes + BN in NACBank — BN trained only)
    for name, p in model.nac.named_parameters():
        if 'bn' not in name:
            p.requires_grad_(False)

    # Enable gradient checkpointing on backbone to save memory
    if hasattr(model.backbone, 'gradient_checkpointing_enable'):
        model.backbone.gradient_checkpointing_enable()

    # ── Loss ──────────────────────────────────────────────────────────
    criterion = HARPERLoss(cfg).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.lr, weight_decay=cfg.weight_decay, eps=1e-8,
    )
    for pg in optimizer.param_groups:
        pg['lr_base'] = cfg.lr

    scaler = GradScaler(enabled=cfg.fp16)

    total_steps = cfg.epochs * len(train_loader)
    print(f"\n[train] {cfg.epochs} epochs, {len(train_loader)} steps/epoch, "
          f"{total_steps} total steps, warmup={cfg.warmup_steps}")

    # ── Training ──────────────────────────────────────────────────────
    best_eer  = 100.0
    best_path = None
    step      = 0
    log       = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (wav, y_s, y_e) in enumerate(train_loader):
            wav  = wav.to(device)
            y_s  = y_s.to(device)
            y_e  = y_e.to(device)
            step += 1

            cosine_lr(optimizer, step, cfg.warmup_steps, total_steps)

            with autocast(enabled=cfg.fp16):
                out  = model(wav)
                loss, loss_dict = criterion(out, y_s, y_e)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if step % 50 == 0:
                lr_now = optimizer.param_groups[0]['lr']
                print(f"  ep={epoch} step={step}/{total_steps} "
                      f"loss={loss.item():.4f}  "
                      f"LM={loss_dict['L_lm']:.3f} "
                      f"route={loss_dict['L_route']:.3f} "
                      f"dec={loss_dict['L_dec']:.3f} "
                      f"geo={loss_dict['L_geo']:.3f}  "
                      f"lr={lr_now:.2e}")

        epoch_loss /= len(train_loader)
        elapsed     = time.time() - t0

        # ── Validation ───────────────────────────────────────────────
        print(f"\n[val] epoch {epoch} ...")
        val_metrics = evaluate(model, val_files, cfg, batch_size=cfg.batch_size,
                                device=device)
        val_eer = val_metrics['eer']
        val_acc = val_metrics['acc']

        print(f"  epoch={epoch}  train_loss={epoch_loss:.4f}  "
              f"val_EER={val_eer:.2f}%  val_ACC={val_acc:.2f}%  "
              f"elapsed={elapsed:.0f}s")

        log.append({'epoch': epoch, 'step': step,
                    'train_loss': epoch_loss, **val_metrics})

        # Save checkpoint
        ckpt_path = save_ckpt(model, optimizer, epoch, step, val_eer, cfg.output_dir)

        if val_eer < best_eer:
            best_eer  = val_eer
            best_path = ckpt_path
            print(f"  *** new best EER: {best_eer:.2f}% ***")

        model.train()

    # ── Final evaluation on best checkpoint ──────────────────────────
    print(f"\n[test] Loading best checkpoint: {best_path}")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])

    # SAS-CF in-domain: val + test combined (full 20% seen data)
    seen_files = val_files + test_files
    seen_metrics = evaluate(model, seen_files, cfg,
                            batch_size=cfg.batch_size, device=device)
    print(f"\n── SAS-CF (in-domain, val+test = 20%) ──")
    print(f"  EER={seen_metrics['eer']:.2f}%  ACC={seen_metrics['acc']:.2f}%  "
          f"ACC_s={seen_metrics['acc_s']:.2f}%  ACC_e={seen_metrics['acc_e']:.2f}%")

    # Unseen 1: DAC codec held-out conditions
    from harper.dataset import build_unseen1, build_unseen2, build_unseen_spk_scene
    print(f"\n── Unseen 1 (DAC codec, held-out) ──")
    u1 = build_unseen1(cfg)
    u1_files = [item for files in u1.values() for item in files]
    if u1_files:
        u1_metrics = evaluate(model, u1_files, cfg,
                              batch_size=cfg.batch_size, device=device)
        print(f"  EER={u1_metrics['eer']:.2f}%  ACC={u1_metrics['acc']:.2f}%  "
              f"ACC_s={u1_metrics['acc_s']:.2f}%  ACC_e={u1_metrics['acc_e']:.2f}%")
    else:
        u1_metrics = {}
        print("  [SKIP] no Unseen 1 files found")

    # Unseen 2: m_f vs M_rr_flat (binary, genuine speech + unseen codec env)
    print(f"\n── Unseen 2 (m_f binary: genuine+codec_env vs M_rr_flat) ──")
    u2 = build_unseen2(cfg)
    u2_files = [item for files in u2.values() for item in files]
    if u2_files:
        u2_metrics = evaluate(model, u2_files, cfg,
                              batch_size=cfg.batch_size, device=device)
        print(f"  EER={u2_metrics['eer']:.2f}%  ACC={u2_metrics['acc']:.2f}%  "
              f"ACC_s={u2_metrics['acc_s']:.2f}%  ACC_e={u2_metrics['acc_e']:.2f}%")
    else:
        u2_metrics = {}
        print("  [SKIP] no Unseen 2 files found")

    # Unseen Spk/Scene: held-out speakers (s_r/s_f) + held-out scenes (e_r/e_f)
    print(f"\n── Unseen Spk/Scene (held-out speakers & scenes) ──")
    uss = build_unseen_spk_scene(cfg)
    uss_files = [item for files in uss.values() for item in files]
    if uss_files:
        uss_metrics = evaluate(model, uss_files, cfg,
                               batch_size=cfg.batch_size, device=device)
        print(f"  EER={uss_metrics['eer']:.2f}%  ACC={uss_metrics['acc']:.2f}%  "
              f"ACC_s={uss_metrics['acc_s']:.2f}%  ACC_e={uss_metrics['acc_e']:.2f}%")
    else:
        uss_metrics = {}
        print("  [SKIP] no Unseen Spk/Scene files found")

    # Summary table
    print(f"\n{'─'*62}")
    print(f"{'Setting':<24} {'EER':>8} {'ACC':>8} {'ACC_s':>8} {'ACC_e':>8}")
    print(f"{'─'*62}")
    for name, m in [("SAS-CF (seen)",       seen_metrics),
                    ("Unseen 1 (DAC)",       u1_metrics),
                    ("Unseen 2 (m_f)",       u2_metrics),
                    ("Unseen (Spk/Scene)",   uss_metrics)]:
        if m:
            print(f"  {name:<22} {m['eer']:>7.2f}% {m['acc']:>7.2f}% "
                  f"{m['acc_s']:>7.2f}% {m['acc_e']:>7.2f}%")
    print(f"{'─'*62}")

    # Save final results
    results = {
        'best_val_eer': best_eer,
        'seen':         seen_metrics,
        'unseen1':      u1_metrics,
        'unseen2':      u2_metrics,
        'unseen_spk_scene': uss_metrics,
        'log':          log,
    }
    res_path = os.path.join(cfg.output_dir, 'results.json')
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {res_path}")

    # ── RTF measurement (runs on same GPU, no extra queue wait) ───────
    _measure_rtf(model, cfg, device)

    return results


def _measure_rtf(model, cfg, device, audio_s=4.0, n_runs=50, warmup=5):
    import time
    from torch.cuda.amp import autocast
    print(f"\n[RTF] measuring on {device} ({audio_s}s clip, {n_runs} runs) ...")
    model.eval()
    n_samples = int(audio_s * cfg.sr)
    dummy = torch.randn(1, n_samples, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            with autocast(enabled=cfg.fp16):
                model(dummy)
        if device == 'cuda':
            torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            with autocast(enabled=cfg.fp16):
                model(dummy)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    mean_t = float(torch.tensor(times).mean())
    std_t  = float(torch.tensor(times).std())
    rtf    = mean_t / audio_s
    print(f"  HARPER RTF = {rtf:.4f}  "
          f"({mean_t*1000:.1f} ± {std_t*1000:.1f} ms  per {audio_s}s clip)")
    print(f"  [RTF] Reference — GARUDA: 0.3025  SATYAM: 2.045")

    # Append RTF to results.json
    res_path = os.path.join(cfg.output_dir, 'results.json')
    try:
        with open(res_path) as f:
            res = json.load(f)
        res['rtf'] = {'mean': mean_t, 'std': std_t, 'rtf': rtf, 'audio_s': audio_s}
        with open(res_path, 'w') as f:
            json.dump(res, f, indent=2)
        print(f"  RTF saved → {res_path}")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Train HARPER on SAS-CF')
    ap.add_argument('--data_root',    default=None)
    ap.add_argument('--output_dir',   default=None)
    ap.add_argument('--batch_size',   type=int, default=None)
    ap.add_argument('--epochs',       type=int, default=None)
    ap.add_argument('--lr',           type=float, default=None)
    ap.add_argument('--train_subset', type=int, default=None)
    ap.add_argument('--no_fp16',      action='store_true')
    args = ap.parse_args()

    cfg = HARPERConfig()
    if args.data_root:    cfg.data_root    = args.data_root
    if args.output_dir:   cfg.output_dir   = args.output_dir
    if args.batch_size:   cfg.batch_size   = args.batch_size
    if args.epochs:       cfg.epochs       = args.epochs
    if args.lr:           cfg.lr           = args.lr
    if args.train_subset: cfg.train_subset = args.train_subset
    if args.no_fp16:      cfg.fp16         = False

    train(cfg)


if __name__ == '__main__':
    main()
