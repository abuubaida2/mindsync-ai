"""
Evaluation utilities for MindSync.
Computes accuracy, macro-averaged precision, recall, F1-score,
incongruence detection metrics, and per-cluster breakdowns.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

from src.data.emotion_clusters import CLUSTER_LABELS, NUM_CLUSTERS


def evaluate_epoch(
    model,
    loader: DataLoader,
    device: torch.device,
    return_embeddings: bool = False,
) -> dict:
    """
    Full evaluation pass — computes all metrics reported in Table 4 and Table 5.

    Args:
        model: MindSync model (or unimodal model).
        loader: DataLoader with ground-truth labels.
        device: Compute device.
        return_embeddings: Whether to collect fused embeddings (for t-SNE).

    Returns:
        dict: accuracy, macro_f1, macro_precision, macro_recall,
              per_class_f1, confusion_matrix, incongruence_rate,
              (optionally) embeddings, labels
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_incong_scores = []
    all_incong_flags = []
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Eval]", dynamic_ncols=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            input_values = batch["input_values"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_values=input_values,
                labels=labels,
            )

            preds = outputs["logits_final"].argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_incong_scores.extend(outputs["incongruence_scores"].cpu().numpy().tolist())
            all_incong_flags.extend(outputs["incongruence_flags"].cpu().numpy().tolist())

            if return_embeddings:
                all_embeddings.append(outputs["embedding_fused"].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_incong_scores = np.array(all_incong_scores)
    all_incong_flags = np.array(all_incong_flags)

    # ── Classification Metrics (Table 4) ─────────────────────────────────────
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    per_class_f1 = f1_score(
        all_labels, all_preds, average=None, labels=list(range(NUM_CLUSTERS)), zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLUSTERS)))

    report = classification_report(
        all_labels, all_preds,
        target_names=CLUSTER_LABELS,
        digits=4,
        zero_division=0,
    )

    result = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "per_class_f1": {CLUSTER_LABELS[i]: float(per_class_f1[i]) for i in range(NUM_CLUSTERS)},
        "confusion_matrix": cm,
        "classification_report": report,
        "incongruence_rate": float(all_incong_flags.mean()),
        "mean_incongruence_score": float(all_incong_scores.mean()),
        "n_samples": len(all_labels),
        "predictions": all_preds,
        "true_labels": all_labels,
        "incongruence_scores": all_incong_scores,
        "incongruence_flags": all_incong_flags,
    }

    if return_embeddings and all_embeddings:
        result["embeddings"] = np.concatenate(all_embeddings, axis=0)

    return result


def evaluate_text_only(
    model,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate text-only baseline (RoBERTa-Large without audio)."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Eval Text-Only]", dynamic_ncols=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
        "per_class_f1": {
            CLUSTER_LABELS[i]: float(v)
            for i, v in enumerate(
                f1_score(all_labels, all_preds, average=None, labels=list(range(NUM_CLUSTERS)), zero_division=0)
            )
        },
        "n_samples": len(all_labels),
    }


def evaluate_audio_only(
    model,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate audio-only baseline (wav2vec 2.0 without text)."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Eval Audio-Only]", dynamic_ncols=True):
            input_values = batch["input_values"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_values=input_values)
            preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
        "per_class_f1": {
            CLUSTER_LABELS[i]: float(v)
            for i, v in enumerate(
                f1_score(all_labels, all_preds, average=None, labels=list(range(NUM_CLUSTERS)), zero_division=0)
            )
        },
        "n_samples": len(all_labels),
    }


def evaluate_incongruence_detection(
    model,
    incongruent_loader: DataLoader,
    congruent_loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate incongruence detection module (Table 5).

    Evaluates binary detection performance on human-annotated
    incongruent (n=520) and congruent samples.

    Returns:
        dict: precision/recall/F1 for congruent + incongruent classes,
              overall accuracy, and binary score breakdown.
    """
    model.eval()

    true_labels = []   # 1 = incongruent, 0 = congruent
    pred_labels = []   # from JSD threshold

    def collect(loader, true_label: int):
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                input_values = batch["input_values"].to(device)

                outputs = model(input_ids, attention_mask, input_values)
                flags = outputs["incongruence_flags"].cpu().numpy()

                true_labels.extend([true_label] * len(flags))
                pred_labels.extend(flags.astype(int).tolist())

    collect(incongruent_loader, true_label=1)
    collect(congruent_loader, true_label=0)

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    binary_acc = accuracy_score(true_labels, pred_labels)
    report = classification_report(
        true_labels, pred_labels,
        target_names=["Congruent", "Incongruent"],
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(binary_acc),
        "congruent": {
            "precision": report["Congruent"]["precision"],
            "recall": report["Congruent"]["recall"],
            "f1": report["Congruent"]["f1-score"],
        },
        "incongruent": {
            "precision": report["Incongruent"]["precision"],
            "recall": report["Incongruent"]["recall"],
            "f1": report["Incongruent"]["f1-score"],
        },
        "n_incongruent": int((true_labels == 1).sum()),
        "n_congruent": int((true_labels == 0).sum()),
    }
