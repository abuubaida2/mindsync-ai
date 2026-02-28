"""
MindSync: Full Multimodal Model
Integrates text stream, audio stream, CMAF, and incongruence detection.

This is the end-to-end trainable module described in Section 3 of the paper.
Implements multi-task loss (Equation 11):
    L_total = L_CE(ŷ_final, y) + λ1·L_CE(ŷ_text, y) + λ2·L_CE(ŷ_audio, y)
where λ1 = λ2 = 0.3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.models.text_model import MindSyncTextModel
from src.models.audio_model import MindSyncAudioModel
from src.models.multimodal_fusion import CMAFModule
from src.models.incongruence import IncongruenceDetector
from src.data.emotion_clusters import NUM_CLUSTERS


class MindSync(nn.Module):
    """
    MindSync: Multimodal AI Framework for Mental Health Monitoring.

    Full dual-stream architecture:
        Text Stream:  RoBERTa-Large → e_text ∈ R^1024
        Audio Stream: wav2vec 2.0   → e_audio ∈ R^768
        Fusion:       CMAF          → e_fused → ŷ_final
        Detection:    JSD-based     → δ, incongruence flag

    Multi-task loss (Eq. 11):
        L_total = L_CE(ŷ_final, y) + 0.3·L_CE(ŷ_text, y) + 0.3·L_CE(ŷ_audio, y)

    Args:
        text_model_name: HuggingFace model for RoBERTa-Large.
        audio_model_name: HuggingFace model for wav2vec 2.0.
        d_model: CMAF shared projection dimension.
        num_heads: CMAF cross-attention heads.
        num_cmaf_layers: Number of stacked CMAF blocks.
        num_classes: Number of emotion clusters (4).
        dropout: Dropout throughout the model.
        lambda_text: Text auxiliary loss weight (λ1 = 0.3).
        lambda_audio: Audio auxiliary loss weight (λ2 = 0.3).
        incongruence_threshold: JSD threshold for clinical alert (0.5).
        incongruence_temperature: Temperature for probability calibration.
        freeze_text_base: Freeze RoBERTa encoder weights.
        freeze_audio_encoder: Freeze wav2vec CNN feature extractor.
    """

    def __init__(
        self,
        text_model_name: str = "roberta-large",
        audio_model_name: str = "facebook/wav2vec2-large-960h",
        d_model: int = 512,
        num_heads: int = 8,
        num_cmaf_layers: int = 2,
        num_classes: int = NUM_CLUSTERS,
        dropout: float = 0.1,
        lambda_text: float = 0.3,
        lambda_audio: float = 0.3,
        incongruence_threshold: float = 0.5,
        incongruence_temperature: float = 1.0,
        freeze_text_base: bool = False,
        freeze_audio_encoder: bool = True,
    ) -> None:
        super().__init__()

        self.lambda_text = lambda_text
        self.lambda_audio = lambda_audio
        self.num_classes = num_classes

        # ── Text Stream ──────────────────────────────────────────────────────
        self.text_model = MindSyncTextModel(
            model_name=text_model_name,
            num_classes=num_classes,
            dropout=dropout,
            freeze_base=freeze_text_base,
        )

        # ── Audio Stream ─────────────────────────────────────────────────────
        self.audio_model = MindSyncAudioModel(
            model_name=audio_model_name,
            num_classes=num_classes,
            dropout=dropout,
            freeze_feature_encoder=freeze_audio_encoder,
        )

        # ── Cross-Modal Attention Fusion ─────────────────────────────────────
        self.cmaf = CMAFModule(
            text_dim=self.text_model.embedding_dim,    # 1024
            audio_dim=self.audio_model.embedding_dim,  # 1024
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_cmaf_layers,
            num_classes=num_classes,
            dropout=dropout,
        )

        # ── Incongruence Detector ────────────────────────────────────────────
        self.incongruence_detector = IncongruenceDetector(
            threshold=incongruence_threshold,
            temperature=incongruence_temperature,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Full forward pass through MindSync.

        Args:
            input_ids: (B, L) text token IDs.
            attention_mask: (B, L) text attention mask.
            input_values: (B, T) raw audio waveform values.
            labels: (B,) optional ground-truth cluster indices for loss computation.

        Returns:
            dict with keys:
                'logits_final': (B, C) fused prediction logits
                'logits_text': (B, C) text-only prediction logits
                'logits_audio': (B, C) audio-only prediction logits
                'embedding_fused': (B, d_model) fused embedding e_fused
                'embedding_text': (B, 1024) e_text
                'embedding_audio': (B, 1024) e_audio
                'incongruence_scores': (B,) JSD scores δ
                'incongruence_flags': (B,) bool flags (True = incongruent)
                'attn_weights': (B, 1, 1) CMAF text→audio attention weights
                'loss': scalar total loss (only if labels provided)
                'loss_final': scalar CE loss on fused prediction
                'loss_text': scalar CE loss on text prediction
                'loss_audio': scalar CE loss on audio prediction
        """
        # ── Text stream ──────────────────────────────────────────────────────
        text_out = self.text_model(input_ids, attention_mask)
        e_text = text_out["embedding"]          # (B, 1024)
        logits_text = text_out["logits"]        # (B, C)

        # ── Audio stream ─────────────────────────────────────────────────────
        audio_out = self.audio_model(input_values)
        e_audio = audio_out["embedding"]        # (B, 1024)
        logits_audio = audio_out["logits"]      # (B, C)

        # ── CMAF fusion ──────────────────────────────────────────────────────
        cmaf_out = self.cmaf(e_text, e_audio)
        logits_final = cmaf_out["logits"]       # (B, C)
        e_fused = cmaf_out["embedding"]         # (B, d_model)
        attn_weights = cmaf_out["attn_weights"]

        # ── Incongruence detection ───────────────────────────────────────────
        incong_out = self.incongruence_detector(logits_text, logits_audio)
        incong_scores = incong_out["scores"]    # (B,)
        incong_flags = incong_out["flags"]      # (B,) bool

        output = {
            "logits_final": logits_final,
            "logits_text": logits_text,
            "logits_audio": logits_audio,
            "embedding_fused": e_fused,
            "embedding_text": e_text,
            "embedding_audio": e_audio,
            "incongruence_scores": incong_scores,
            "incongruence_flags": incong_flags,
            "attn_weights": attn_weights,
        }

        # ── Multi-task loss (Equation 11) ────────────────────────────────────
        if labels is not None:
            loss_final = F.cross_entropy(logits_final, labels)
            loss_text = F.cross_entropy(logits_text, labels)
            loss_audio = F.cross_entropy(logits_audio, labels)

            loss_total = (
                loss_final
                + self.lambda_text * loss_text
                + self.lambda_audio * loss_audio
            )

            output["loss"] = loss_total
            output["loss_final"] = loss_final
            output["loss_text"] = loss_text
            output["loss_audio"] = loss_audio

        return output

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_values: torch.Tensor,
    ) -> dict:
        """
        Inference-mode prediction.

        Returns predicted cluster labels, probabilities, and incongruence info.
        """
        self.eval()
        output = self.forward(input_ids, attention_mask, input_values)

        probs_final = F.softmax(output["logits_final"], dim=-1)
        probs_text = F.softmax(output["logits_text"], dim=-1)
        probs_audio = F.softmax(output["logits_audio"], dim=-1)

        from src.data.emotion_clusters import IDX_TO_CLUSTER
        predictions = [
            IDX_TO_CLUSTER[int(i)] for i in probs_final.argmax(dim=-1).cpu()
        ]

        return {
            "predictions": predictions,
            "probabilities": probs_final.cpu().numpy(),
            "text_probabilities": probs_text.cpu().numpy(),
            "audio_probabilities": probs_audio.cpu().numpy(),
            "incongruence_scores": output["incongruence_scores"].cpu().numpy(),
            "incongruence_flags": output["incongruence_flags"].cpu().numpy(),
        }

    def count_parameters(self) -> dict:
        """Return trainable and total parameter counts per sub-module."""
        def count(module):
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable

        t_tot, t_train = count(self.text_model)
        a_tot, a_train = count(self.audio_model)
        c_tot, c_train = count(self.cmaf)

        return {
            "text_model": {"total": t_tot, "trainable": t_train},
            "audio_model": {"total": a_tot, "trainable": a_train},
            "cmaf": {"total": c_tot, "trainable": c_train},
            "total": {
                "total": t_tot + a_tot + c_tot,
                "trainable": t_train + a_train + c_train,
            },
        }
