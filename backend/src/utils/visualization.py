"""
Visualization utilities for MindSync paper figures.

Generates:
    Figure 4 — Figure2_ConfusionMatrix.png
    Figure 5 — Figure3_tSNE.png
    Additional: training curves, per-class F1, incongruence score distribution
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Dict

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

from src.data.emotion_clusters import CLUSTER_LABELS

# ── Style ────────────────────────────────────────────────────────────────────────
PALETTE = {
    "Distress": "#E74C3C",
    "Resilience": "#2ECC71",
    "Aggression": "#E67E22",
    "Ambiguity": "#9B59B6",
}
FONT_SIZE = 12
plt.rcParams.update({"font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE + 2})


# ── Figure 4: Confusion Matrix ───────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str] = CLUSTER_LABELS,
    title: str = "MindSync (CMAF) Confusion Matrix",
    save_path: Optional[str] = None,
    normalize: bool = True,
) -> plt.Figure:
    """
    Plot confusion matrix (Figure 4 — Figure2_ConfusionMatrix.png).

    Args:
        cm: (C, C) confusion matrix array.
        labels: Class label names.
        title: Plot title.
        save_path: File path to save the figure.
        normalize: Normalize by row (true counts → proportions).

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        fmt = ".2f"
        vmin, vmax = 0.0, 1.0
    else:
        cm_plot = cm
        fmt = "d"
        vmin, vmax = None, None

    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
    )

    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# ── Figure 5: t-SNE Embedding Visualisation ──────────────────────────────────────

def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = "t-SNE: MindSync Fused Emotion Embeddings",
    save_path: Optional[str] = None,
    perplexity: int = 40,
    random_state: int = 42,
    max_points: int = 3000,
) -> plt.Figure:
    """
    t-SNE visualisation of fused emotion cluster embeddings (Figure 5).

    Args:
        embeddings: (N, D) embedding matrix.
        labels: (N,) integer cluster indices.
        title: Plot title.
        save_path: File path to save the figure.
        perplexity: t-SNE perplexity parameter.
        random_state: Random seed for reproducibility.
        max_points: Subsample to this many points for speed.

    Returns:
        matplotlib Figure.
    """
    # Subsample
    if len(embeddings) > max_points:
        idx = np.random.choice(len(embeddings), max_points, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    print("Running t-SNE (this may take a moment)...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=1000,      # sklearn >= 1.3 renamed n_iter → max_iter
        verbose=0,
    )
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(9, 7))

    for cls_idx, cls_name in enumerate(CLUSTER_LABELS):
        mask = labels == cls_idx
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=PALETTE[cls_name],
            label=cls_name,
            s=10,
            alpha=0.65,
            edgecolors="none",
        )

    ax.set_title(title, fontweight="bold", pad=14)
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.legend(markerscale=3, title="Emotion Cluster", fontsize=10)
    ax.grid(True, alpha=0.2, linestyle="--")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# ── Training Curves ──────────────────────────────────────────────────────────────

def plot_training_curves(
    history: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training and validation loss / F1 curves.

    Args:
        history: dict with 'train' and 'val' lists of metric dicts.
        save_path: File path to save the figure.

    Returns:
        matplotlib Figure.
    """
    epochs = range(1, len(history["train"]) + 1)
    train_loss = [h["avg_loss"] for h in history["train"]]
    val_f1 = [h["macro_f1"] for h in history["val"]]
    val_acc = [h["accuracy"] for h in history["val"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, train_loss, "b-o", markersize=4, label="Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_f1, "g-o", markersize=4, label="Val Macro F1")
    axes[1].plot(epochs, val_acc, "r-s", markersize=4, label="Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Performance", fontweight="bold")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("MindSync Training Progress", fontweight="bold", fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# ── Per-Cluster F1 Bar Chart (Table 8 visualisation) ────────────────────────────

def plot_cluster_f1_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart comparing per-cluster F1 across model configurations (Table 8).

    Args:
        results: {model_name: {cluster_name: f1_score}}
        save_path: File path to save the figure.

    Returns:
        matplotlib Figure.
    """
    models = list(results.keys())
    clusters = CLUSTER_LABELS
    x = np.arange(len(clusters))
    bar_width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#3498DB", "#E74C3C", "#2ECC71", "#9B59B6", "#E67E22"]

    for i, (model_name, scores) in enumerate(results.items()):
        offset = (i - len(models) / 2 + 0.5) * bar_width
        f1_vals = [scores.get(c, 0.0) for c in clusters]
        bars = ax.bar(
            x + offset,
            f1_vals,
            bar_width,
            label=model_name,
            color=colors[i % len(colors)],
            alpha=0.85,
            edgecolor="white",
        )
        for bar, val in zip(bars, f1_vals):
            ax.annotate(
                f"{val:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Emotion Cluster")
    ax.set_ylabel("Macro F1-Score")
    ax.set_title("Per-Cluster F1 Score Comparison (Table 8)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(clusters)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# ── Incongruence Score Distribution ─────────────────────────────────────────────

def plot_incongruence_distribution(
    congruent_scores: np.ndarray,
    incongruent_scores: np.ndarray,
    threshold: float = 0.5,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Distribution of JSD incongruence scores for congruent vs. incongruent samples.

    Args:
        congruent_scores: JSD scores for congruent samples.
        incongruent_scores: JSD scores for incongruent samples.
        threshold: Decision boundary (default 0.5 per paper).
        save_path: File path to save.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(congruent_scores, bins=40, alpha=0.65, color="#2ECC71", label="Congruent", density=True)
    ax.hist(incongruent_scores, bins=40, alpha=0.65, color="#E74C3C", label="Incongruent", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold (δ={threshold})")

    ax.set_xlabel("JSD Incongruence Score (δ)")
    ax.set_ylabel("Density")
    ax.set_title("Incongruence Score Distribution by Class", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.2, linestyle="--")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig
