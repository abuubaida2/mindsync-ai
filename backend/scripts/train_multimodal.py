"""
Script: Train the full MindSync multimodal model.
Usage:
    python scripts/train_multimodal.py --config configs/multimodal_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from src.models.mindsync import MindSync
from src.data.dataset import (
    GoEmotionsDataset,
    RAVDESSDataset,
    SyntheticMultimodalDataset,
    build_dataloader,
)
from src.data.text_preprocessing import TextPreprocessor
from src.data.audio_preprocessing import AudioPreprocessor
from src.training.train import train
from src.training.evaluate import evaluate_epoch
from src.utils.seed import set_seed
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_tsne,
    plot_training_curves,
    plot_cluster_f1_comparison,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MindSync multimodal model")
    parser.add_argument(
        "--config", type=str, default="configs/multimodal_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (small dataset)")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate, skip training")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to load for eval")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Preprocessors ────────────────────────────────────────────────────────
    text_proc = TextPreprocessor(
        model_name=cfg["model"]["text_model_name"],
        max_length=cfg["data"].get("max_text_length", 128),
    )
    audio_proc = AudioPreprocessor(
        model_name=cfg["model"]["audio_model_name"],
        max_length_sec=cfg["data"].get("audio_max_length_sec", 10.0),
    )

    max_samples = 100 if args.debug else None
    logger.info("Loading datasets...")

    # ── Datasets ────────────────────────────────────────────────────────────
    train_text = GoEmotionsDataset("train", text_proc, max_samples=max_samples)
    val_text = GoEmotionsDataset("validation", text_proc, max_samples=max_samples)
    test_text = GoEmotionsDataset("test", text_proc, max_samples=max_samples)

    train_audio = RAVDESSDataset(cfg["data"]["ravdess_root"], "train", audio_proc)
    val_audio = RAVDESSDataset(cfg["data"]["ravdess_root"], "validation", audio_proc)
    test_audio = RAVDESSDataset(cfg["data"]["ravdess_root"], "test", audio_proc)

    train_dataset = SyntheticMultimodalDataset(train_text, train_audio, seed=cfg.get("seed", 42))
    val_dataset = SyntheticMultimodalDataset(val_text, val_audio, seed=0)
    test_dataset = SyntheticMultimodalDataset(test_text, test_audio, seed=1)

    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]

    train_loader = build_dataloader(
        train_dataset, train_cfg["batch_size"], shuffle=True,
        num_workers=cfg.get("hardware", {}).get("num_workers", 4),
        use_weighted_sampler=cfg["data"].get("use_weighted_sampler", True),
    )
    val_loader = build_dataloader(
        val_dataset, eval_cfg["batch_size"], shuffle=False,
    )
    test_loader = build_dataloader(
        test_dataset, eval_cfg["batch_size"], shuffle=False,
    )

    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # ── Model ────────────────────────────────────────────────────────────────
    model_cfg = cfg["model"]
    incong_cfg = cfg.get("incongruence", {})
    model = MindSync(
        text_model_name=model_cfg["text_model_name"],
        audio_model_name=model_cfg["audio_model_name"],
        d_model=model_cfg.get("d_model", 512),
        num_heads=model_cfg.get("num_heads", 8),
        num_cmaf_layers=model_cfg.get("num_cmaf_layers", 2),
        dropout=model_cfg.get("dropout", 0.1),
        lambda_text=model_cfg.get("lambda_text", 0.3),
        lambda_audio=model_cfg.get("lambda_audio", 0.3),
        incongruence_threshold=incong_cfg.get("threshold", 0.5),
    )

    params = model.count_parameters()
    logger.info(f"Model parameters: {params['total']['total']:,} total, {params['total']['trainable']:,} trainable")

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        logger.info(f"Loaded checkpoint: {args.checkpoint}")

    if not args.eval_only:
        output_cfg = cfg["output"]
        wandb_cfg = cfg.get("wandb", {})

        result = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=train_cfg["num_epochs"],
            text_lr=train_cfg["text_lr"],
            audio_lr=train_cfg["audio_lr"],
            cmaf_lr=train_cfg["cmaf_lr"],
            weight_decay=train_cfg["weight_decay"],
            warmup_ratio=train_cfg["warmup_ratio"],
            grad_clip=train_cfg["grad_clip"],
            output_dir=output_cfg["checkpoint_dir"],
            device=device,
            use_wandb=wandb_cfg.get("use_wandb", False),
            wandb_project=wandb_cfg.get("project", "mindsync"),
            seed=cfg.get("seed", 42),
            log_interval=train_cfg.get("log_interval", 50),
        )

        # ── Plot training curves ─────────────────────────────────────────────
        figs_dir = output_cfg.get("figures_dir", "docs/figures")
        plot_training_curves(
            result["history"],
            save_path=f"{figs_dir}/training_curves.png",
        )

        # Load best checkpoint for final evaluation
        if result["best_checkpoint"]:
            ckpt = torch.load(result["best_checkpoint"], map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])

    # ── Final Test Evaluation ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST SET EVALUATION")
    test_metrics = evaluate_epoch(model, test_loader, device, return_embeddings=True)

    logger.info(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
    logger.info(f"  Macro F1:    {test_metrics['macro_f1']:.4f}")
    logger.info(f"  Precision:   {test_metrics['macro_precision']:.4f}")
    logger.info(f"  Recall:      {test_metrics['macro_recall']:.4f}")
    logger.info("\n" + test_metrics["classification_report"])

    # ── Generate paper figures ────────────────────────────────────────────────
    output_cfg = cfg.get("output", {})
    figs_dir = output_cfg.get("figures_dir", "docs/figures")
    Path(figs_dir).mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        save_path=f"{figs_dir}/Figure2_ConfusionMatrix.png",
    )

    if "embeddings" in test_metrics:
        plot_tsne(
            test_metrics["embeddings"],
            test_metrics["true_labels"],
            save_path=f"{figs_dir}/Figure3_tSNE.png",
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
