"""
Audio Stream: wav2vec 2.0 fine-tuned on RAVDESS.
Implements Section 3.3.2, Equation (5) of MindSync paper.

Architecture:
    Raw waveform → CNN encoder → Transformer context network
    → Mean pool → e_audio ∈ R^1024 → classification head
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

from src.data.emotion_clusters import NUM_CLUSTERS


class Wav2VecAudioEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-large-960h",
        dropout: float = 0.1,
        freeze_feature_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        if freeze_feature_encoder:
            if hasattr(self.encoder, "freeze_feature_encoder"):
                self.encoder.freeze_feature_encoder()
            else:
                for p in self.encoder.feature_extractor.parameters():
                    p.requires_grad = False
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_values=input_values).last_hidden_state
        e = out.mean(dim=1)
        return self.dropout(e)


class AudioClassificationHead(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
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
        return self.classifier(embedding)


class MindSyncAudioModel(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-large-960h",
        num_classes: int = NUM_CLUSTERS,
    ) -> None:
        super().__init__()
        self.encoder = Wav2VecAudioEncoder(model_name=model_name)
        self.classifier = AudioClassificationHead(
            hidden_size=self.encoder.hidden_size,
            num_classes=num_classes,
        )

    @property
    def embedding_dim(self) -> int:
        return self.encoder.hidden_size

    def forward(self, input_values: torch.Tensor) -> dict:
        embedding = self.encoder(input_values)
        logits = self.classifier(embedding)
        return {"embedding": embedding, "logits": logits}
