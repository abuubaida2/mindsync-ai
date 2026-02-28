"""
Jensen-Shannon Divergence-based Incongruence Detection Module.
Implements Section 3.5, Equation (10) of MindSync paper.

Incongruence score δ via JSD (Equation 10):
    δ = JSD(P_text ‖ P_audio)
      = ½ KL(P_text ‖ M) + ½ KL(P_audio ‖ M)

where M = ½(P_text + P_audio).

Threshold δ > 0.5 flags a sample as incongruent → clinical-risk alert.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import jensenshannon
from typing import List, Optional, Tuple


# ── JSD Utilities ───────────────────────────────────────────────────────────────

def jsd_torch(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Jensen-Shannon Divergence (Equation 10) in PyTorch — batched.

    JSD(P ‖ Q) = ½ KL(P ‖ M) + ½ KL(Q ‖ M),  M = ½(P + Q)

    Args:
        p: (B, C) text probability distribution (after softmax).
        q: (B, C) audio probability distribution (after softmax).
        eps: Small constant for numerical stability.

    Returns:
        jsd: (B,) JSD scores in [0, 1].
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    m = 0.5 * (p + q)

    kl_pm = F.kl_div(m.log(), p, reduction="none").sum(dim=-1)
    kl_qm = F.kl_div(m.log(), q, reduction="none").sum(dim=-1)

    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    # Normalise to [0, 1]:  JSD ≤ log(2) ≈ 0.693
    jsd = (jsd / torch.log(torch.tensor(2.0, device=p.device))).clamp(0.0, 1.0)
    return jsd


def jsd_numpy(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon Divergence via SciPy (numpy interface).
    Returns score in [0, 1].
    """
    # scipy.spatial.distance.jensenshannon returns sqrt(JSD)
    return float(jensenshannon(p, q) ** 2)


# ── Incongruence Detector ───────────────────────────────────────────────────────

class IncongruenceDetector(nn.Module):
    """
    Detects cross-modal incongruence between text and audio predictions.

    Given unimodal logits from the text and audio streams, computes the
    JSD-based incongruence score δ and applies a configurable threshold.

    Clinical interpretation (Section 3.5):
        δ > threshold  →  INCONGRUENT  (flag for clinical-risk review)
        δ ≤ threshold  →  CONGRUENT

    Args:
        threshold: Decision boundary for incongruence detection (default 0.5).
        temperature: Softmax temperature for probability calibration.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature

    def _to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to calibrated probabilities."""
        return F.softmax(logits / self.temperature, dim=-1)

    def forward(
        self,
        text_logits: torch.Tensor,
        audio_logits: torch.Tensor,
    ) -> dict:
        """
        Compute incongruence scores and binary flags.

        Args:
            text_logits: (B, C) raw logits from text classifier.
            audio_logits: (B, C) raw logits from audio classifier.

        Returns:
            dict with keys:
                'scores': (B,) JSD scores δ ∈ [0, 1]
                'flags': (B,) bool tensor — True if incongruent (δ > τ)
                'p_text': (B, C) text probability distribution P_text
                'p_audio': (B, C) audio probability distribution P_audio
        """
        p_text = self._to_probs(text_logits)    # (B, C)
        p_audio = self._to_probs(audio_logits)  # (B, C)

        scores = jsd_torch(p_text, p_audio)      # (B,)
        flags = scores > self.threshold

        return {
            "scores": scores,
            "flags": flags,
            "p_text": p_text,
            "p_audio": p_audio,
        }

    def detect(
        self,
        text_logits: torch.Tensor,
        audio_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method returning (scores, flags) tensors.

        Args:
            text_logits: (B, C) text logits.
            audio_logits: (B, C) audio logits.

        Returns:
            scores: (B,) incongruence scores.
            flags: (B,) boolean incongruence flags.
        """
        result = self.forward(text_logits, audio_logits)
        return result["scores"], result["flags"]

    @torch.no_grad()
    def predict_single(
        self,
        text_logits: torch.Tensor,
        audio_logits: torch.Tensor,
    ) -> dict:
        """
        Predict incongruence for a single sample (inference utility).

        Args:
            text_logits: (C,) or (1, C) text logits.
            audio_logits: (C,) or (1, C) audio logits.

        Returns:
            dict: score, flag, p_text, p_audio, dominant_text_class,
                  dominant_audio_class, clinical_alert
        """
        if text_logits.dim() == 1:
            text_logits = text_logits.unsqueeze(0)
        if audio_logits.dim() == 1:
            audio_logits = audio_logits.unsqueeze(0)

        result = self.forward(text_logits, audio_logits)
        score = result["scores"].item()
        flag = result["flags"].item()
        p_text = result["p_text"].squeeze(0).cpu().numpy()
        p_audio = result["p_audio"].squeeze(0).cpu().numpy()

        from src.data.emotion_clusters import IDX_TO_CLUSTER
        dominant_text = IDX_TO_CLUSTER[int(p_text.argmax())]
        dominant_audio = IDX_TO_CLUSTER[int(p_audio.argmax())]

        return {
            "incongruence_score": round(score, 4),
            "is_incongruent": flag,
            "clinical_alert": flag,
            "dominant_text_emotion": dominant_text,
            "dominant_audio_emotion": dominant_audio,
            "p_text": p_text.tolist(),
            "p_audio": p_audio.tolist(),
            "threshold": self.threshold,
        }


# ── Statistical Analysis Utilities ──────────────────────────────────────────────

def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> dict:
    """
    McNemar's test for paired predictions (Section 4.3).

    Tests whether two classifiers are significantly different on the
    same test set (n = 5,801).

    Args:
        y_true: (N,) ground-truth labels.
        y_pred_a: (N,) predictions from model A.
        y_pred_b: (N,) predictions from model B.

    Returns:
        dict: statistic, p_value, b, c (contingency table values)
    """
    from statsmodels.stats.contingency_tables import mcnemar

    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true

    # Discordant pairs
    b = int(np.sum(correct_a & ~correct_b))  # A correct, B wrong
    c = int(np.sum(~correct_a & correct_b))  # A wrong, B correct

    result = mcnemar([[0, b], [c, 0]], exact=False, correction=True)
    return {
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "b": b,
        "c": c,
        "significant": result.pvalue < 0.001,
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cohen's d effect size for practical significance (Section 4.3).

    d = (μ_a - μ_b) / pooled_std

    Args:
        a: Scores for group A (e.g., MindSync F1 per fold).
        b: Scores for group B (e.g., MulT F1 per fold).

    Returns:
        Cohen's d value (0.94 reported in paper = large effect).
    """
    mean_diff = np.mean(a) - np.mean(b)
    pooled_std = np.sqrt(
        (np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2
    )
    return float(mean_diff / pooled_std) if pooled_std > 0 else 0.0


def fleiss_kappa(ratings: np.ndarray) -> float:
    """
    Compute Fleiss' kappa for inter-rater agreement (Section 4.2).
    Reported value: κ = 0.81 on 520 human-annotated incongruent samples.

    Args:
        ratings: (N, k) matrix — N subjects, k raters, values are category indices.

    Returns:
        Fleiss' kappa coefficient.
    """
    import statsmodels.stats.inter_rater as ir
    agg, cats = ir.aggregate_raters(ratings)
    return ir.fleiss_kappa(agg, method="fleiss")
