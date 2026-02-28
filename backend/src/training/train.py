"""
Multi-task training loop for MindSync.
Implements Section 3.6 (AdamW + linear warm-up, multi-task loss).
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.models.mindsync import MindSync
from src.training.evaluate import evaluate_epoch
from src.utils.seed import set_seed


logger = logging.getLogger(__name__)


def get_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int):
    """
    Linear warm-up followed by cosine annealing (Section 3.6).
    Warm-up: 10% of total training steps.
    """
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=num_warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - num_warmup_steps,
        eta_min=1e-7,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[num_warmup_steps],
    )


def train_epoch(
    model: MindSync,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    grad_clip: float = 1.0,
    log_interval: int = 50,
    epoch: int = 0,
) -> dict:
    """
    Single training epoch.

    Returns:
        dict: avg_loss, avg_loss_final, avg_loss_text, avg_loss_audio,
              samples_per_sec, incongruence_rate
    """
    model.train()
    total_loss = total_lf = total_lt = total_la = 0.0
    total_incongruent = total_samples = 0
    start = time.time()

    pbar = tqdm(loader, desc=f"[Train Epoch {epoch}]", dynamic_ncols=True)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        input_values = batch["input_values"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_values=input_values,
            labels=labels,
        )

        loss = outputs["loss"]
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()
        scheduler.step()

        b = labels.size(0)
        total_loss += loss.item() * b
        total_lf += outputs["loss_final"].item() * b
        total_lt += outputs["loss_text"].item() * b
        total_la += outputs["loss_audio"].item() * b
        total_incongruent += outputs["incongruence_flags"].sum().item()
        total_samples += b

        if (step + 1) % log_interval == 0:
            avg = total_loss / total_samples
            lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else 0
            pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr:.2e}")

    n = total_samples
    elapsed = time.time() - start
    return {
        "avg_loss": total_loss / n,
        "avg_loss_final": total_lf / n,
        "avg_loss_text": total_lt / n,
        "avg_loss_audio": total_la / n,
        "samples_per_sec": n / elapsed,
        "incongruence_rate": total_incongruent / n,
    }


def train(
    model: MindSync,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    text_lr: float = 2e-5,
    audio_lr: float = 1e-4,
    cmaf_lr: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.10,
    grad_clip: float = 1.0,
    output_dir: str = "checkpoints/mindsync",
    device: Optional[torch.device] = None,
    use_wandb: bool = False,
    wandb_project: str = "mindsync",
    seed: int = 42,
    log_interval: int = 50,
) -> dict:
    """
    Full training loop for MindSync.

    Differential learning rates (Section 3.6):
        - RoBERTa-Large encoder: lr = 2e-5
        - wav2vec 2.0 encoder: lr = 1e-4
        - CMAF + heads: lr = 1e-4

    Args:
        model: MindSync model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        num_epochs: Training epochs.
        text_lr: LR for text encoder (2e-5 per paper).
        audio_lr: LR for audio encoder (1e-4 per paper).
        cmaf_lr: LR for CMAF + classification heads.
        weight_decay: AdamW weight decay.
        warmup_ratio: Fraction of training steps for linear warm-up.
        grad_clip: Gradient norm clipping.
        output_dir: Directory for saving checkpoints.
        device: Compute device.
        use_wandb: Enable Weights & Biases logging.
        wandb_project: W&B project name.
        seed: Random seed (paper: 42).
        log_interval: Log every N steps.

    Returns:
        dict: Training history and best checkpoint path.
    """
    set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Training on {device} | output_dir={output_dir}")

    # ── Parameter groups with differential LR ───────────────────────────────────
    text_params = list(model.text_model.parameters())
    audio_params = list(model.audio_model.parameters())
    cmaf_params = list(model.cmaf.parameters())

    optimizer = AdamW(
        [
            {"params": text_params, "lr": text_lr, "weight_decay": weight_decay},
            {"params": audio_params, "lr": audio_lr, "weight_decay": weight_decay},
            {"params": cmaf_params, "lr": cmaf_lr, "weight_decay": weight_decay},
        ]
    )

    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_scheduler(optimizer, warmup_steps, total_steps)

    # ── W&B ─────────────────────────────────────────────────────────────────────
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            config={
                "epochs": num_epochs,
                "text_lr": text_lr,
                "audio_lr": audio_lr,
                "cmaf_lr": cmaf_lr,
                "seed": seed,
                "lambda_text": model.lambda_text,
                "lambda_audio": model.lambda_audio,
            },
        )

    history = {"train": [], "val": []}
    best_val_f1 = 0.0
    best_ckpt_path = ""

    for epoch in range(1, num_epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{num_epochs}")

        # ── Train ────────────────────────────────────────────────────────────
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            grad_clip=grad_clip, log_interval=log_interval, epoch=epoch,
        )
        history["train"].append(train_metrics)

        # ── Validate ─────────────────────────────────────────────────────────
        val_metrics = evaluate_epoch(model, val_loader, device)
        history["val"].append(val_metrics)

        logger.info(
            f"  Train Loss: {train_metrics['avg_loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['macro_f1']:.4f}"
        )

        # ── Checkpoint ───────────────────────────────────────────────────────
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_ckpt_path = os.path.join(output_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "best_val_f1": best_val_f1,
                },
                best_ckpt_path,
            )
            logger.info(f"  ✓ Saved best checkpoint (F1={best_val_f1:.4f})")

        # Save latest checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
            },
            os.path.join(output_dir, "latest_model.pt"),
        )

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch,
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
            })

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    logger.info(f"\nTraining complete. Best Val F1: {best_val_f1:.4f}")
    logger.info(f"Best checkpoint: {best_ckpt_path}")

    return {
        "history": history,
        "best_val_f1": best_val_f1,
        "best_checkpoint": best_ckpt_path,
    }
