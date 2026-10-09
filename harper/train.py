"""
HARPER v2 Training — Progressive 4-stage optimisation.

Stage 1 (acoustic):    CNN + Transformer + queries + prompts + Lorentz
                       Loss: L_inv, L_eff, L_pre, L_auth
Stage 2 (hyperbolic):  same + entailment cones enabled
                       Loss: + L_ent
Stage 3 (lm):          unlock LLM LoRA adapters + evidence adapters
                       Loss: + L_LM
Stage 4 (joint):       fine-tune all with lower lr
                       Loss: all 6 terms

Usage:
    python -m harper.train [--batch_size N] [--output_dir PATH] ...
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
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent.parent))

from harper.config  import HARPERConfig
from harper.model   import HARPERModel
from harper.dataset import (SASCFDataset, QuadrupleDataset,
                             build_splits, build_quadruples)
from harper.losses  import HARPERLoss
from harper.evaluate import evaluate


# ── Utilities ─────────────────────────────────────────────────────────────

def cosine_lr(optimizer, step, warmup, total, min_lr=1e-7):
    if step < warmup:
        scale = step / max(1, warmup)
    else:
        prog  = (step - warmup) / max(1, total - warmup)
        scale = 0.5 * (1 + math.cos(math.pi * prog))
    for pg in optimizer.param_groups:
        pg['lr'] = max(min_lr, pg['lr_base'] * scale)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6, train / 1e6


def save_ckpt(model, optimizer, epoch, step, val_eer, out_dir, tag=''):
    os.makedirs(out_dir, exist_ok=True)
    name = f'{tag}ep{epoch}_step{step}_eer{val_eer:.2f}.pt'
    path = os.path.join(out_dir, name)
    torch.save({
        'epoch': epoch, 'step': step, 'val_eer': val_eer,
        'model': model.state_dict(),
        'opt':   optimizer.state_dict(),
    }, path)
    print(f"  [ckpt] → {path}")
    return path


# ── Parameter group helpers ───────────────────────────────────────────────

def _acoustic_params(model):
    """CNN, Transformer, query extractor, prompt learner, Lorentz module."""
    modules = [
        model.stft, model.cnn, model.transformer,
        model.query_extractor, model.prompt_learner, model.lorentz,
    ]
    params = []
    seen   = set()
    for m in modules:
        for p in m.parameters():
            if id(p) not in seen and p.requires_grad:
                seen.add(id(p))
                params.append(p)
    return params


def _llm_params(model):
    """Evidence adapters + LLM LoRA parameters."""
    modules = [
        model.adp_s_P, model.adp_s_A,
        model.adp_e_P, model.adp_e_A,
        model.adp_g, model.llm,
    ]
    params = []
    seen   = set()
    for m in modules:
        for p in m.parameters():
            if id(p) not in seen and p.requires_grad:
                seen.add(id(p))
                params.append(p)
    return params


def _set_llm_trainable(model, flag: bool):
    """Enable/disable gradient for evidence adapters + LLM."""
    for m in [model.adp_s_P, model.adp_s_A,
              model.adp_e_P, model.adp_e_A,
              model.adp_g, model.llm]:
        for p in m.parameters():
            p.requires_grad_(flag)
    # Base LLM weights always frozen; only LoRA is trainable
    for name, p in model.llm.named_parameters():
        if 'lora_' not in name and 'embed' not in name:
            p.requires_grad_(False)


# ── Single epoch ──────────────────────────────────────────────────────────

def run_epoch(model, loader, quad_iter, optimizer, scaler,
              criterion, device, cfg, step, total_steps,
              stage, log_every=50):
    model.train()
    epoch_loss = 0.0
    t0 = time.time()

    for batch_idx, (wav, y_s, y_e) in enumerate(loader):
        wav = wav.to(device)
        y_s = y_s.to(device)
        y_e = y_e.to(device)
        step += 1
        cosine_lr(optimizer, step, cfg.warmup_steps, total_steps)

        # Fetch quadruple batch (may be None)
        quad_data = None
        if quad_iter is not None:
            try:
                quad_batch = next(quad_iter)
            except StopIteration:
                quad_batch = None

            if quad_batch is not None:
                x_RR, x_RF, x_FR, x_FF = [q.to(device) for q in quad_batch]

                with autocast(enabled=cfg.fp16):
                    h_s_RR, h_e_RR = _get_h(model, x_RR)
                    h_s_RF, h_e_RF = _get_h(model, x_RF)
                    h_s_FR, h_e_FR = _get_h(model, x_FR)
                    h_s_FF, h_e_FF = _get_h(model, x_FF)

                quad_data = {
                    'h_s_RR': h_s_RR, 'h_s_RF': h_s_RF,
                    'h_s_FR': h_s_FR, 'h_s_FF': h_s_FF,
                    'h_e_RR': h_e_RR, 'h_e_RF': h_e_RF,
                    'h_e_FR': h_e_FR, 'h_e_FF': h_e_FF,
                }

        with autocast(enabled=cfg.fp16):
            out  = model.forward_train(wav, y_s, y_e)
            loss, ld = criterion(
                out, y_s, y_e,
                quad=quad_data,
                lorentz_module=model.lorentz,
                prompt_learner=model.prompt_learner,
                stage=stage,
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            cfg.grad_clip,
        )
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

        if step % log_every == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"    step={step}/{total_steps}  loss={loss.item():.4f}  "
                  f"inv={ld['L_inv']:.3f}  pre={ld['L_pre']:.3f}  "
                  f"auth={ld['L_auth']:.3f}  ent={ld['L_ent']:.3f}  "
                  f"lm={ld['L_lm']:.3f}  lr={lr_now:.2e}")

    elapsed = time.time() - t0
    return epoch_loss / max(len(loader), 1), step, elapsed


def _get_h(model, audio):
    """Run acoustic forward and return h_s, h_e (no hyp)."""
    dtype_llm = next(model.llm.parameters()).dtype
    X  = model.stft(audio.float())
    Z0 = model.cnn(X.to(dtype_llm))
    Z  = model.transformer(Z0)
    h_s, h_e, _h_g = model.query_extractor(Z)
    return h_s, h_e


# ── Progressive training ──────────────────────────────────────────────────

def train(cfg: HARPERConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[train] device={device}  fp16={cfg.fp16}")

    # ── Data ──────────────────────────────────────────────────────────
    print("\n[data] scanning SAS-CF ...")
    train_files, val_files, test_files = build_splits(cfg)

    print("\n[data] building quadruples ...")
    quads = build_quadruples(cfg, max_quads=min(len(train_files) // 4, 5000))

    train_ds = SASCFDataset(train_files, cfg.sr, cfg.max_audio_s, augment=True)
    val_ds   = SASCFDataset(val_files,   cfg.sr, cfg.max_audio_s)
    quad_ds  = QuadrupleDataset(quads,   cfg.sr, cfg.max_audio_s) if quads else None

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, num_workers=2,
        pin_memory=True, drop_last=False,
    )
    quad_loader = (DataLoader(quad_ds, batch_size=max(1, cfg.batch_size // 2),
                              shuffle=True, num_workers=2, pin_memory=True,
                              drop_last=True)
                   if quad_ds else None)

    # ── Model ─────────────────────────────────────────────────────────
    print("\n[model] building HARPER v2 ...")
    model = HARPERModel(cfg).to(device)
    total_m, train_m = count_params(model)
    print(f"  Total: {total_m:.1f}M   Trainable: {train_m:.3f}M")

    # Enable gradient checkpointing on LLM backbone
    if hasattr(model.llm, 'gradient_checkpointing_enable'):
        model.llm.gradient_checkpointing_enable()

    criterion = HARPERLoss(cfg).to(device)
    scaler    = GradScaler(enabled=cfg.fp16)
    best_eer  = 100.0
    best_path = None
    step      = 0
    log       = []

    # Progressive stage definitions
    stages = [
        # (name, n_epochs, lr, llm_trainable)
        ('acoustic',   cfg.epochs_stage1, cfg.lr,       False),
        ('hyperbolic', cfg.epochs_stage2, cfg.lr / 2,   False),
        ('lm',         cfg.epochs_stage3, cfg.lr_lm,    True),
        ('joint',      cfg.epochs_joint,  cfg.lr_joint, True),
    ]

    for stage_name, n_epochs, lr, llm_train in stages:
        if n_epochs <= 0:
            continue

        print(f"\n{'═'*60}")
        print(f"  STAGE: {stage_name.upper()}  ({n_epochs} epoch(s), lr={lr:.1e})")
        print(f"{'═'*60}")

        _set_llm_trainable(model, llm_train)

        # Build optimizer for current stage
        if llm_train:
            # Two param groups: acoustic (lower lr) + llm (stage lr)
            opt = torch.optim.AdamW([
                {'params': _acoustic_params(model), 'lr': lr / 5},
                {'params': _llm_params(model),      'lr': lr},
            ], weight_decay=cfg.weight_decay, eps=1e-8)
        else:
            opt = torch.optim.AdamW(
                _acoustic_params(model),
                lr=lr, weight_decay=cfg.weight_decay, eps=1e-8,
            )
        for pg in opt.param_groups:
            pg['lr_base'] = pg['lr']

        total_steps = n_epochs * len(train_loader)

        for epoch in range(1, n_epochs + 1):
            quad_iter = iter(quad_loader) if quad_loader else None
            epoch_loss, step, elapsed = run_epoch(
                model, train_loader, quad_iter, opt, scaler,
                criterion, device, cfg, step, step + len(train_loader),
                stage=stage_name,
            )

            print(f"\n[val] {stage_name} epoch {epoch} ...")
            val_metrics = evaluate(model, val_files, cfg,
                                   batch_size=cfg.batch_size, device=device)
            val_eer = val_metrics['eer']
            val_acc = val_metrics['acc']
            print(f"  epoch={epoch}  loss={epoch_loss:.4f}  "
                  f"EER={val_eer:.2f}%  ACC={val_acc:.2f}%  "
                  f"Joint_ACC={val_metrics.get('joint_acc',0):.2f}%  "
                  f"elapsed={elapsed:.0f}s")

            log.append({'stage': stage_name, 'epoch': epoch, 'step': step,
                        'train_loss': epoch_loss, **val_metrics})

            ckpt = save_ckpt(model, opt, epoch, step, val_eer,
                              cfg.output_dir, tag=f'{stage_name}_')
            if val_eer < best_eer:
                best_eer  = val_eer
                best_path = ckpt
                print(f"  *** new best EER: {best_eer:.2f}% ***")

    # ── Final evaluation ─────────────────────────────────────────────
    print(f"\n[test] Loading best checkpoint: {best_path}")
    if best_path:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model'])

    from harper.dataset import (build_unseen1, build_unseen2,
                                 build_unseen_spk_scene)

    seen_files   = val_files + test_files
    seen_metrics = evaluate(model, seen_files, cfg,
                            batch_size=cfg.batch_size, device=device)

    print(f"\n{'─'*62}")
    print(f"{'Setting':<26} {'EER':>7} {'ACC':>7} {'Jnt':>7}")
    print(f"{'─'*62}")
    print(f"  {'SAS-CF (seen)':<24} {seen_metrics['eer']:>6.2f}% "
          f"{seen_metrics['acc']:>6.2f}% {seen_metrics.get('joint_acc',0):>6.2f}%")

    for name, builder in [("Unseen 1 (DAC)", build_unseen1),
                           ("Unseen 2 (m_f)", build_unseen2),
                           ("Unseen Spk/Scene", build_unseen_spk_scene)]:
        print(f"\n── {name} ──")
        files_dict = builder(cfg)
        files = [item for v in files_dict.values() for item in v]
        if files:
            m = evaluate(model, files, cfg,
                         batch_size=cfg.batch_size, device=device)
            print(f"  {name:<24} {m['eer']:>6.2f}% "
                  f"{m['acc']:>6.2f}% {m.get('joint_acc',0):>6.2f}%")
        else:
            m = {}
            print("  [SKIP] no files found")

    print(f"{'─'*62}")

    # Save results
    results = {'best_val_eer': best_eer, 'seen': seen_metrics, 'log': log}
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results → {cfg.output_dir}/results.json")

    _measure_rtf(model, cfg, device)
    return results


def _measure_rtf(model, cfg, device, audio_s=4.0, n_runs=30, warmup=3):
    print(f"\n[RTF] measuring ({audio_s}s, {n_runs} runs) ...")
    model.eval()
    dummy = torch.randn(1, int(audio_s * cfg.sr), device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        if device == 'cuda':
            torch.cuda.synchronize()
        import time
        ts = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            if device == 'cuda':
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    mean_t = sum(ts) / len(ts)
    rtf    = mean_t / audio_s
    print(f"  RTF = {rtf:.4f}  ({mean_t*1000:.1f}ms per {audio_s}s clip)")


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_dir',   default=None)
    ap.add_argument('--batch_size',   type=int,   default=None)
    ap.add_argument('--train_subset', type=int,   default=None)
    ap.add_argument('--no_fp16',      action='store_true')
    ap.add_argument('--epochs_s1',    type=int,   default=None)
    ap.add_argument('--epochs_s2',    type=int,   default=None)
    ap.add_argument('--epochs_s3',    type=int,   default=None)
    ap.add_argument('--epochs_joint', type=int,   default=None)
    args = ap.parse_args()

    cfg = HARPERConfig()
    if args.output_dir:   cfg.output_dir   = args.output_dir
    if args.batch_size:   cfg.batch_size   = args.batch_size
    if args.train_subset: cfg.train_subset = args.train_subset
    if args.no_fp16:      cfg.fp16         = False
    if args.epochs_s1 is not None:    cfg.epochs_stage1 = args.epochs_s1
    if args.epochs_s2 is not None:    cfg.epochs_stage2 = args.epochs_s2
    if args.epochs_s3 is not None:    cfg.epochs_stage3 = args.epochs_s3
    if args.epochs_joint is not None: cfg.epochs_joint  = args.epochs_joint

    train(cfg)


if __name__ == '__main__':
    main()
