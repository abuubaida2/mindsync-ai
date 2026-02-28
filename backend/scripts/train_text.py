"""
Script: Train text-only RoBERTa-Large baseline on GoEmotions.
Usage:
    python scripts/train_text.py --config configs/text_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from src.models.text_model import MindSyncTextModel
from src.data.dataset import GoEmotionsDataset, build_dataloader
from src.data.text_preprocessing import TextPreprocessor
from src.training.train import get_scheduler
from src.training.evaluate import evaluate_text_only
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train RoBERTa-Large text baseline")
    parser.add_argument("--config", type=str, default="configs/text_config.yaml")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    text_proc = TextPreprocessor(
        model_name=cfg["model"]["name"],
        max_length=cfg["data"].get("max_length", 128),
    )
    max_samples = 100 if args.debug else None

    train_ds = GoEmotionsDataset("train", text_proc, max_samples=max_samples)
    val_ds = GoEmotionsDataset("validation", text_proc, max_samples=max_samples)
    test_ds = GoEmotionsDataset("test", text_proc, max_samples=max_samples)

    train_loader = build_dataloader(
        train_ds,
        cfg["training"]["batch_size"],
        use_weighted_sampler=cfg["training"].get("use_weighted_sampler", True),
    )
    val_loader = build_dataloader(val_ds, cfg["evaluation"]["batch_size"], shuffle=False)
    test_loader = build_dataloader(test_ds, cfg["evaluation"]["batch_size"], shuffle=False)

    model = MindSyncTextModel(
        model_name=cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
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
        total_loss = 0.0
        n = 0
        import torch.nn.functional as F

        for batch in tqdm(train_loader, desc=f"[Train Epoch {epoch}]"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = F.cross_entropy(outputs["logits"], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * labels.size(0)
            n += labels.size(0)

        val_metrics = evaluate_text_only(model, val_loader, device)
        logger.info(
            f"Epoch {epoch} | Loss: {total_loss/n:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), out_dir / "best_text_model.pt")
            logger.info(f"  ✓ Saved (F1={best_f1:.4f})")

    logger.info("\nFinal test evaluation:")
    model.load_state_dict(torch.load(out_dir / "best_text_model.pt"))
    test_metrics = evaluate_text_only(model, test_loader, device)
    logger.info(f"  Test Acc: {test_metrics['accuracy']:.4f} | Test F1: {test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
