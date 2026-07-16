"""
Fine-tune VFIMamba (large variant) on a custom 2K triplet dataset, single-GPU
(one RTX 3090).

No CLI args -- just edit the SETTINGS block below and run:
    python train.py

Changes vs. the original draft, made specifically for 2K fine-tuning:
- Larger default crop (384, still a multiple of 32) so training patches see
  motion at a scale closer to real 2K footage, with gradient accumulation to
  keep the effective batch size at 8 despite the smaller physical batch.
- LR lowered and given a short linear warmup before the cosine decay -- the
  original 2e-4 peak with no warmup is a from-scratch-training setting and
  is likely to disturb the pretrained weights on the first few steps of a
  fine-tune.
- The AMP + LapLoss combination is now safe (see model/loss.py -- the loss
  now casts to float32 internally, since the pyramid helpers build float32
  zero-tensors that previously could not be torch.cat'd against a float16
  autocast tensor).
- A best-checkpoint and a last-checkpoint are tracked separately from the
  periodic per-epoch dumps, so you aren't left guessing which of 25 files
  to deploy.
- A full-resolution evaluation pass runs every EVAL_FULL_RES_EVERY epochs,
  padding real (uncropped) validation frames to a multiple of 32 and
  reporting PSNR -- this is the number that actually reflects "is it
  getting better at 2K", since the per-batch train/val loss is only ever
  computed on small crops.
- A worker_init_fn reseeds Python's random module per DataLoader worker,
  since torch does not do this automatically and workers can otherwise
  emit correlated augmentations.

Notes carried over from the original draft:
- Config file is configCustom.py, not config.py. Trainer.py hardcodes
  `from config import *`, so we register configCustom in sys.modules under
  the name "config" *before* importing Trainer -- this makes Trainer.py
  pick up the large-variant settings without editing Trainer.py itself.
- Single-GPU only (Model(-1) -> no DDP).
- Warm-start loads the pretrained checkpoint with strict=False, reusing the
  repo's own `convert()` so architecture drift in KEY NAMES doesn't hard-
  fail. IMPORTANT: strict=False only tolerates missing/unexpected key
  names, not shape mismatches -- if configCustom.MODEL_ARCH's F/depth/W
  don't match what PRETRAINED was actually trained with, load_state_dict
  will raise a size-mismatch error regardless of strict. This script
  prints a warning if the mismatched-key fraction looks suspiciously high.
- Your OWN fine-tuned checkpoints (saved by this script) are NOT
  DDP-prefixed, so they're loaded directly without convert() if you ever
  resume from one of your own saves (see is_ddp_ckpt below).
"""
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import configCustom
sys.modules['config'] = configCustom  # Trainer.py does `from config import *` -- point it here

from Trainer import Model, convert  # noqa: E402  (must come after the sys.modules patch above)
from configCustom import MODEL_CONFIG
from dataset import VFIDataset
from model.loss import LapLoss  # the repo's real loss -- requires model/matching.py to exist

# ==================== SETTINGS -- edit these directly ====================
DATA_ROOT    = '/path/to/your/triplets'     # folder of seq_xxxx/im1.png,im2.png,im3.png (native 2K res)
PRETRAINED   = 'base_model/VFIMamba.pkl'    # '' to train from scratch

RUN_NAME     = 'training_run1'              # change this per run -- everything (the full
                                             # console log plus checkpoints) is written under
                                             # runs/RUN_NAME/, so old runs are kept side by
                                             # side instead of being overwritten
RUNS_DIR     = 'runs'

EPOCHS       = 25
BATCH_SIZE   = 4                            # physical batch; see ACCUM_STEPS below
ACCUM_STEPS  = 2                            # effective batch = BATCH_SIZE * ACCUM_STEPS = 8

LR           = 5e-5                         # peak LR -- fine-tuning wants far less than the
                                             # 2e-4 you'd use to train this from scratch
MIN_LR       = 1e-6
WARMUP_EPOCHS = 1                           # linear warmup into LR, then cosine decay to MIN_LR
WEIGHT_DECAY = 1e-4

CROP_SIZE    = 384                          # must be a multiple of 32; larger than the usual
                                             # 256 so training patches better reflect 2K-scale
                                             # motion. Drop to 256-320 first if you hit OOM,
                                             # before lowering BATCH_SIZE further.
NUM_WORKERS  = 4
VAL_SPLIT    = 0.2
SEED         = 42
GRAD_CLIP    = 1.0
AMP          = True

SAVE_EVERY_EPOCH = True                     # keep a per-epoch dump as well as best/last
EVAL_FULL_RES_EVERY = 5                     # epochs between full-resolution PSNR checks
EVAL_FULL_RES_MAX_SAMPLES = 8               # how many val sequences to run at full res (slow)

LOG_EVERY_STEPS = 10                        # print a progress line every N training steps
# ===========================================================================


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def lr_schedule(optimizer, epoch, total_epochs, warmup_epochs, base_lr, min_lr):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        e = epoch - warmup_epochs
        total = max(1, total_epochs - warmup_epochs - 1)
        cos = 0.5 * (1 + math.cos(math.pi * e / total))
        lr = min_lr + (base_lr - min_lr) * cos
    for g in optimizer.param_groups:
        g['lr'] = lr
    return lr


def pad_to_multiple(x, multiple=32):
    _, _, h, w = x.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    return F.pad(x, (0, pw, 0, ph), mode='replicate'), h, w


def format_time(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f'{h}h{m:02d}m{s:02d}s'
    if m > 0:
        return f'{m}m{s:02d}s'
    return f'{s}s'


def psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2).item()
    if mse <= 1e-10:
        return 99.0
    return 10 * math.log10(1.0 / mse)


@torch.no_grad()
def evaluate_full_res(model, val_set, local, max_samples, device='cuda'):
    """Runs inference on real, uncropped validation frames (padded to a
    multiple of 32) and reports average PSNR. This is the number that
    reflects actual 2K quality -- the crop-based train/val loss elsewhere
    in this script only ever sees small patches."""
    model.eval()
    total, count = 0.0, 0
    for seq in val_set.seqs[:max_samples]:
        seq_dir = os.path.join(val_set.data_root, seq)
        img0 = cv2.imread(os.path.join(seq_dir, 'im1.png'))
        gt = cv2.imread(os.path.join(seq_dir, 'im2.png'))
        img1 = cv2.imread(os.path.join(seq_dir, 'im3.png'))
        if img0 is None or gt is None or img1 is None:
            continue

        def to_tensor(im):
            t = torch.from_numpy(im.transpose(2, 0, 1).astype(np.float32) / 255.0)
            return t.unsqueeze(0).to(device)

        t0, tg, t1 = to_tensor(img0), to_tensor(gt), to_tensor(img1)
        t0p, h, w = pad_to_multiple(t0)
        t1p, _, _ = pad_to_multiple(t1)
        imgs = torch.cat((t0p, t1p), 1)
        _, _, _, pred = model.net(imgs, timestep=0.5, scale=0, local=local)
        pred = pred[:, :, :h, :w]
        total += psnr(pred, tg)
        count += 1
    model.train()
    return total / max(1, count)


def main():
    torch.backends.cudnn.benchmark = True

    run_dir = os.path.join(RUNS_DIR, RUN_NAME)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(run_dir, 'train.log')
    log_file = open(log_path, 'a')  # append -- re-running the same RUN_NAME keeps history

    def log(msg):
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
        print(line)
        log_file.write(line + '\n')
        log_file.flush()

    log(f'===== starting run "{RUN_NAME}" ==========================================')
    log(f'settings: epochs={EPOCHS} batch={BATCH_SIZE} accum={ACCUM_STEPS} '
        f'crop={CROP_SIZE} lr={LR} min_lr={MIN_LR} warmup_epochs={WARMUP_EPOCHS} '
        f'amp={AMP} data_root={DATA_ROOT} pretrained={PRETRAINED}')
    log(f'full log for this run: {log_path}')

    train_set = VFIDataset(DATA_ROOT, mode='train', crop_size=CROP_SIZE,
                            val_split=VAL_SPLIT, seed=SEED)
    val_set = VFIDataset(DATA_ROOT, mode='val', crop_size=CROP_SIZE,
                          val_split=VAL_SPLIT, seed=SEED)
    log(f'train samples: {len(train_set)} | val samples: {len(val_set)}')

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        worker_init_fn=worker_init_fn)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        worker_init_fn=worker_init_fn)

    model = Model(-1)  # local_rank=-1 -> single GPU, no DDP wrapper

    # ---- warm start ----
    if PRETRAINED and os.path.exists(PRETRAINED):
        log(f'warm-starting from {PRETRAINED}')
        state = torch.load(PRETRAINED, map_location='cuda')
        is_ddp_ckpt = any('module.' in k for k in state.keys())
        state = convert(state) if is_ddp_ckpt else state
        missing, unexpected = model.net.load_state_dict(state, strict=False)
        total_keys = len(model.net.state_dict())
        log(f'  missing keys:    {len(missing)} / {total_keys}')
        log(f'  unexpected keys: {len(unexpected)}')
        if len(missing) > 0.2 * total_keys or len(unexpected) > 0.2 * total_keys:
            log('  WARNING: a large fraction of keys did not line up. '
                'strict=False only tolerates missing/unexpected KEY NAMES, '
                'not shape mismatches -- if configCustom.MODEL_ARCH '
                '(F / depth / W) does not match the architecture '
                f'{PRETRAINED} was actually trained with, this warm start '
                'is likely doing far less than you think (most of the '
                'network may still be at random init).')
    else:
        log('no pretrained checkpoint given/found -- training from scratch')

    optimizer = torch.optim.AdamW(model.net.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    local = model.local  # from config.LOCAL, set in Model.__init__
    criterion = LapLoss(max_levels=5, channels=3)  # matches the repo's own training objective

    best_val = float('inf')
    steps_per_epoch = len(train_loader)
    total_steps_all = EPOCHS * steps_per_epoch
    train_start = time.time()
    global_step = 0
    history = []

    for epoch in range(EPOCHS):
        cur_lr = lr_schedule(optimizer, epoch, EPOCHS, WARMUP_EPOCHS, LR, MIN_LR)

        model.train()
        t0 = time.time()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        n_steps = steps_per_epoch
        for step, (img0, gt, img1) in enumerate(train_loader):
            img0 = img0.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)
            img1 = img1.cuda(non_blocking=True)
            imgs = torch.cat((img0, img1), 1)

            with torch.cuda.amp.autocast(enabled=AMP):
                _, _, _, pred = model.net(imgs, timestep=0.5, scale=0, local=local)
                loss = criterion(pred, gt) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == n_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.net.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            step_loss = loss.item() * ACCUM_STEPS
            running_loss += step_loss
            global_step += 1

            # ---- live progress: loss / step count / epoch / ETA ----
            if global_step % LOG_EVERY_STEPS == 0 or (step + 1) == n_steps:
                elapsed_total = time.time() - train_start
                avg_step_time = elapsed_total / global_step
                eta = avg_step_time * (total_steps_all - global_step)
                running_avg_loss = running_loss / (step + 1)
                log(f'  epoch {epoch+1}/{EPOCHS} | step {step+1}/{n_steps} '
                    f'(global {global_step}/{total_steps_all}) | '
                    f'loss {step_loss:.4f} | avg_loss {running_avg_loss:.4f} | '
                    f'elapsed {format_time(elapsed_total)} | '
                    f'ETA {format_time(eta)}')

        train_loss = running_loss / max(1, n_steps)

        # ---- crop-based validation (fast, comparable across epochs) ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img0, gt, img1 in val_loader:
                img0 = img0.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                img1 = img1.cuda(non_blocking=True)
                imgs = torch.cat((img0, img1), 1)
                _, _, _, pred = model.net(imgs, timestep=0.5, scale=0, local=local)
                val_loss += criterion(pred, gt).item()
        val_loss /= max(1, len(val_loader))

        elapsed = time.time() - t0
        msg = (f'epoch {epoch+1:03d}/{EPOCHS} | train_loss {train_loss:.4f} | '
               f'val_loss {val_loss:.4f} | lr {cur_lr:.2e} | {elapsed:.1f}s')

        # ---- occasional full-resolution PSNR check ----
        full_psnr_this_epoch = None
        if (epoch + 1) % EVAL_FULL_RES_EVERY == 0 or (epoch + 1) == EPOCHS:
            full_psnr_this_epoch = evaluate_full_res(model, val_set, local, EVAL_FULL_RES_MAX_SAMPLES)
            msg += f' | full-res PSNR {full_psnr_this_epoch:.2f}dB'
        log(msg)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': cur_lr,
            'full_psnr': full_psnr_this_epoch,
        })

        if SAVE_EVERY_EPOCH:
            ckpt_path = os.path.join(
                ckpt_dir,
                f'{MODEL_CONFIG["LOGNAME"]}_epoch{epoch+1:03d}_valloss{val_loss:.4f}.pkl'
            )
            torch.save(model.net.state_dict(), ckpt_path)

        last_path = os.path.join(ckpt_dir, f'{MODEL_CONFIG["LOGNAME"]}_last.pkl')
        torch.save(model.net.state_dict(), last_path)

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(ckpt_dir, f'{MODEL_CONFIG["LOGNAME"]}_best.pkl')
            torch.save(model.net.state_dict(), best_path)
            log(f'  -> new best (val_loss {best_val:.4f}), saved to {best_path}')

    # ---- final run summary: the thing to actually look at tomorrow morning ----
    best_ckpt_path = os.path.join(ckpt_dir, f'{MODEL_CONFIG["LOGNAME"]}_best.pkl')
    last_ckpt_path = os.path.join(ckpt_dir, f'{MODEL_CONFIG["LOGNAME"]}_last.pkl')
    summary_path = os.path.join(run_dir, 'summary.txt')

    if history:
        total_elapsed = time.time() - train_start
        best_entry = min(history, key=lambda h: h['val_loss'])
        final_entry = history[-1]
        avg_train_loss = sum(h['train_loss'] for h in history) / len(history)
        avg_val_loss = sum(h['val_loss'] for h in history) / len(history)

        psnr_entries = [h for h in history if h['full_psnr'] is not None]
        best_psnr_entry = max(psnr_entries, key=lambda h: h['full_psnr']) if psnr_entries else None

        first_val_loss = history[0]['val_loss']
        change_pct = ((best_entry['val_loss'] - first_val_loss) / first_val_loss * 100
                      if first_val_loss > 0 else 0.0)

        w = 62
        lines = []
        lines.append('=' * w)
        lines.append(f' TRAINING RUN SUMMARY: {RUN_NAME}')
        lines.append('=' * w)
        lines.append(f' Epochs completed:      {len(history)} / {EPOCHS}')
        lines.append(f' Total wall time:       {format_time(total_elapsed)}')
        lines.append(f' Pretrained warm-start: {PRETRAINED if PRETRAINED else "(trained from scratch)"}')
        lines.append('')
        lines.append(f' BEST EPOCH: {best_entry["epoch"]}')
        lines.append(f'   val_loss:            {best_entry["val_loss"]:.4f}')
        lines.append(f'   train_loss:          {best_entry["train_loss"]:.4f}')
        if best_psnr_entry is not None:
            lines.append(f'   best full-res PSNR:  {best_psnr_entry["full_psnr"]:.2f} dB '
                          f'(epoch {best_psnr_entry["epoch"]})')
        lines.append(f'   checkpoint:          {best_ckpt_path}')
        lines.append('')
        lines.append(f' FINAL EPOCH ({final_entry["epoch"]}):')
        lines.append(f'   train_loss:          {final_entry["train_loss"]:.4f}')
        lines.append(f'   val_loss:            {final_entry["val_loss"]:.4f}')
        lines.append(f'   lr:                  {final_entry["lr"]:.2e}')
        lines.append('')
        lines.append(' AVERAGES ACROSS RUN:')
        lines.append(f'   avg train_loss:      {avg_train_loss:.4f}')
        lines.append(f'   avg val_loss:        {avg_val_loss:.4f}')
        lines.append('')
        lines.append(' IMPROVEMENT (val_loss, epoch 1 -> best):')
        lines.append(f'   {first_val_loss:.4f} -> {best_entry["val_loss"]:.4f}  ({change_pct:+.1f}%)')
        lines.append('=' * w)
        lines.append(f' Full log:        {log_path}')
        lines.append(f' Summary file:    {summary_path}')
        lines.append(f' Best checkpoint: {best_ckpt_path}')
        lines.append(f' Last checkpoint: {last_ckpt_path}')
        lines.append('=' * w)
        summary_text = '\n'.join(lines)

        with open(summary_path, 'w') as f:
            f.write(summary_text + '\n')

        # Written directly (not through log()) so it isn't cluttered with a
        # timestamp on every line -- it's meant to be read as one clean block.
        print('\n' + summary_text + '\n')
        log_file.write('\n' + summary_text + '\n')
        log_file.flush()
    else:
        log('No epochs were completed -- skipping final summary.')

    log(f'===== run "{RUN_NAME}" finished =========================================')
    log_file.close()


if __name__ == '__main__':
    main()