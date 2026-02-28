"""
Text Stream: RoBERTa-Large fine-tuned on GoEmotions.
Implements Section 3.3.1, Equations (3) and (4) of MindSync paper.

Architecture:
    Input tokens → RoBERTa-Large (24 layers, d=1024) → [CLS] embedding e_text ∈ R^1024
    Classification head: ŷ_text = softmax(W_c · e_text + b_c)

Key hyperparameters (from paper):
    lr = 2e-5, batch_size = 32, epochs = 5
"""

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

from src.data.emotion_clusters import NUM_CLUSTERS


class RoBERTaTextEncoder(nn.Module):
    """
    RoBERTa-Large text encoder.

    Returns [CLS] token embedding e_text ∈ R^1024 from the final layer.
    Used as the text stream in MindSync CMAF pipeline.

    Args:
        model_name: HuggingFace model identifier.
        hidden_size: Embedding dimension (1024 for RoBERTa-Large).
        dropout: Dropout rate on [CLS] embedding.
        freeze_base: Freeze base encoder weights (for ablation studies).
    """

    HIDDEN_SIZE = 1024  # RoBERTa-Large

    def __init__(
        self,
        model_name: str = "roberta-large",
        hidden_size: int = 1024,
        dropout: float = 0.1,
        freeze_base: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = RobertaModel.from_pretrained(model_name)

        if freeze_base:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: (B, L) token indices.
            attention_mask: (B, L) binary mask.

        Returns:
            e_text: (B, 1024) [CLS] token embeddings.
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # [CLS] token = first token of last hidden state
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (B, 1024)
        return self.dropout(cls_embedding)


class TextClassificationHead(nn.Module):
    """
    Classification head for text stream (Equation 4).

    ŷ_text = softmax(W_c · e_text + b_c)

    Args:
        hidden_size: Input embedding dimension (1024).
        num_classes: Number of output classes (4 clusters).
        dropout: Dropout before linear projection.
    """

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
        """
        Args:
            embedding: (B, hidden_size) e_text from encoder.

        Returns:
            logits: (B, num_classes) raw scores.
        """
        return self.classifier(embedding)


class MindSyncTextModel(nn.Module):
    """
    Complete text-stream model: RoBERTa-Large + classification head.
    Used as standalone text-only baseline and as text sub-module in CMAF.

    Args:
        model_name: HuggingFace model name.
        num_classes: Output classes (default 4 clusters).
        dropout: Dropout rate.
        freeze_base: Freeze base RoBERTa weights.
    """

    def __init__(
        self,
        model_name: str = "roberta-large",
        num_classes: int = NUM_CLUSTERS,
        dropout: float = 0.1,
        freeze_base: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = RoBERTaTextEncoder(
            model_name=model_name,
            dropout=dropout,
            freeze_base=freeze_base,
        )
        self.classifier = TextClassificationHead(
            hidden_size=self.encoder.HIDDEN_SIZE,
            num_classes=num_classes,
            dropout=dropout,
        )

    @property
    def embedding_dim(self) -> int:
        return self.encoder.HIDDEN_SIZE

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        """
        Args:
            input_ids: (B, L)
            attention_mask: (B, L)

        Returns:
            dict with keys:
                'embedding': (B, 1024) — e_text for CMAF fusion
                'logits': (B, num_classes) — for text auxiliary loss
        """
        embedding = self.encoder(input_ids, attention_mask)
        logits = self.classifier(embedding)
        return {"embedding": embedding, "logits": logits}
