"""
Script: Train audio-only wav2vec 2.0 baseline on RAVDESS + IEMOCAP.
Usage:
    python scripts/train_audio.py --config configs/audio_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from tqdm import tqdm

from src.models.audio_model import MindSyncAudioModel
from src.data.dataset import RAVDESSDataset, build_dataloader
from src.data.audio_preprocessing import AudioPreprocessor
from src.training.train import get_scheduler
from src.training.evaluate import evaluate_audio_only
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train wav2vec 2.0 audio baseline")
    parser.add_argument("--config", type=str, default="configs/audio_config.yaml")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    audio_proc = AudioPreprocessor(
        model_name=cfg["model"]["name"],
        max_length_sec=cfg["data"].get("max_length_sec", 10.0),
    )

    ravdess_root = cfg["data"]["ravdess"]["root_dir"]
    train_ds = RAVDESSDataset(ravdess_root, "train", audio_proc)
    val_ds = RAVDESSDataset(ravdess_root, "validation", audio_proc)
    test_ds = RAVDESSDataset(ravdess_root, "test", audio_proc)

    train_loader = build_dataloader(train_ds, cfg["training"]["batch_size"])
    val_loader = build_dataloader(val_ds, cfg["evaluation"]["batch_size"], shuffle=False)
    test_loader = build_dataloader(test_ds, cfg["evaluation"]["batch_size"], shuffle=False)

    model = MindSyncAudioModel(
        model_name=cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        hidden_size=cfg["model"]["hidden_size"],
        dropout=cfg["model"]["dropout"],
        freeze_feature_encoder=cfg["model"]["freeze_feature_encoder"],
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    total_steps = len(train_loader) * cfg["training"]["num_epochs"]
    warmup_steps = int(total_steps * cfg["training"].get("warmup_ratio", 0.1))
    scheduler = get_scheduler(optimizer, warmup_steps, total_steps)

    out_dir = Path(cfg["output"]["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0

    for epoch in range(1, cfg["training"]["num_epochs"] + 1):
        model.train()
        total_loss, n = 0.0, 0

        for batch in tqdm(train_loader, desc=f"[Train Epoch {epoch}]"):
            input_values = batch["input_values"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_values)
            loss = F.cross_entropy(outputs["logits"], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * labels.size(0)
            n += labels.size(0)

        val_metrics = evaluate_audio_only(model, val_loader, device)
        logger.info(
            f"Epoch {epoch} | Loss: {total_loss/n:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), out_dir / "best_audio_model.pt")
            logger.info(f"  ✓ Saved (F1={best_f1:.4f})")

    logger.info("\nFinal test evaluation:")
    model.load_state_dict(torch.load(out_dir / "best_audio_model.pt"))
    test_metrics = evaluate_audio_only(model, test_loader, device)
    logger.info(f"  Test Acc: {test_metrics['accuracy']:.4f} | Test F1: {test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
