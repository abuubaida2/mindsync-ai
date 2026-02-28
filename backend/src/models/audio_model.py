"""
Audio Stream: wav2vec 2.0 fine-tuned on RAVDESS + IEMOCAP.
Implements Section 3.3.2, Equation (5) of MindSync paper.

Architecture:
    Raw waveform → CNN encoder (z_t) → Transformer context network (c_t)
    → Mean pool → e_audio ∈ R^768

Contrastive pre-training objective (Eq. 5):
    L_w2v = -log [ exp(sim(c_t, q_t) / κ) / Σ exp(sim(c_t, q̃) / κ) ]

Key hyperparameters (from paper):
    lr = 1e-4, batch_size = 16, epochs = 10
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config

from src.data.emotion_clusters import NUM_CLUSTERS


class Wav2VecAudioEncoder(nn.Module):
    """
    wav2vec 2.0 audio encoder.

    Produces mean-pooled contextual representations e_audio ∈ R^768.
    Used as the audio stream in MindSync CMAF pipeline.

    Args:
        model_name: HuggingFace model identifier.
        hidden_size: Transformer hidden size (768 for wav2vec 2.0 base,
                     1024 for large).
        dropout: Dropout on output embedding.
        freeze_feature_encoder: Freeze the CNN feature encoder
                                (standard wav2vec 2.0 fine-tuning practice).
    """

    HIDDEN_SIZE = 768  # wav2vec 2.0 base / large-960h

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-large-960h",
        hidden_size: int = 768,
        dropout: float = 0.1,
        freeze_feature_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)

        if freeze_feature_encoder:
            # Freeze CNN feature extractor (standard practice).
            # transformers >=5.x exposes freeze_feature_encoder() directly;
            # older versions used feature_extractor._freeze_parameters().
            if hasattr(self.encoder, "freeze_feature_encoder"):
                self.encoder.freeze_feature_encoder()
            elif hasattr(self.encoder, "feature_extractor") and hasattr(
                self.encoder.feature_extractor, "_freeze_parameters"
            ):
                self.encoder.feature_extractor._freeze_parameters()
            else:
                for param in self.encoder.feature_extractor.parameters():
                    param.requires_grad = False

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_values: (B, T_samples) raw waveform values, normalised.

        Returns:
            e_audio: (B, 768) mean-pooled contextual representation.
        """
        outputs = self.encoder(input_values=input_values)
        # outputs.last_hidden_state: (B, T_frames, 768)
        # Mean pool over time dimension
        hidden_states = outputs.last_hidden_state           # (B, T, 768)
        e_audio = hidden_states.mean(dim=1)                 # (B, 768)
        return self.dropout(e_audio)


class AudioClassificationHead(nn.Module):
    """
    Classification head for audio stream.

    Args:
        hidden_size: Input embedding dimension (768).
        num_classes: Number of output classes (4 clusters).
        dropout: Dropout before linear projection.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_classes: int = NUM_CLUSTERS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: (B, hidden_size) e_audio.

        Returns:
            logits: (B, num_classes).
        """
        return self.classifier(embedding)


class MindSyncAudioModel(nn.Module):
    """
    Complete audio-stream model: wav2vec 2.0 + classification head.
    Used as standalone audio-only baseline and as audio sub-module in CMAF.

    Args:
        model_name: HuggingFace model name.
        num_classes: Output classes (default 4 clusters).
        hidden_size: Encoder hidden size.
        dropout: Dropout rate.
        freeze_feature_encoder: Freeze CNN feature encoder.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-large-960h",
        num_classes: int = NUM_CLUSTERS,
        hidden_size: int = 768,
        dropout: float = 0.1,
        freeze_feature_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = Wav2VecAudioEncoder(
            model_name=model_name,
            hidden_size=hidden_size,
            dropout=dropout,
            freeze_feature_encoder=freeze_feature_encoder,
        )
        self.classifier = AudioClassificationHead(
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout=dropout,
        )

    @property
    def embedding_dim(self) -> int:
        return self.encoder.hidden_size

    def forward(self, input_values: torch.Tensor) -> dict:
        """
        Args:
            input_values: (B, T_samples) raw waveform.

        Returns:
            dict with keys:
                'embedding': (B, 768) — e_audio for CMAF fusion
                'logits': (B, num_classes) — for audio auxiliary loss
        """
        embedding = self.encoder(input_values)
        logits = self.classifier(embedding)
        return {"embedding": embedding, "logits": logits}
