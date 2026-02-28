"""
Cross-Modal Attention Fusion (CMAF) Module.
Implements Section 3.4, Equations (6)–(9) of MindSync paper.

Architecture:
    e_text ∈ R^1024, e_audio ∈ R^768
    ↓ Project to shared dimension d_model
    e_text' = Attention(Q_text, K_audio, V_audio)   (Eq. 6)
    e_audio' = Attention(Q_audio, K_text, V_text)    (Eq. 7)
    e_fused = FFN([e_text' ∥ e_audio'])               (Eq. 8)
    ŷ_final = softmax(W_f · e_fused + b_f)           (Eq. 9)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.data.emotion_clusters import NUM_CLUSTERS


class ProjectionLayer(nn.Module):
    """Linear projection to shared embedding space with LayerNorm."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class CrossModalAttentionBlock(nn.Module):
    """
    Single bidirectional cross-modal attention block.

    e_A' = Attention(Q_A, K_B, V_B)  — A attends over B
    e_B' = Attention(Q_B, K_A, V_A)  — B attends over A

    For global (non-sequential) embeddings the sequence length = 1,
    equivalent to scaled dot-product cross-attention.

    Args:
        d_model: Shared embedding dimension.
        num_heads: Number of attention heads.
        dropout: Attention dropout.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn_ab = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_ba = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)

    def forward(
        self,
        e_a: torch.Tensor,
        e_b: torch.Tensor,
    ):
        """
        Args:
            e_a: (B, d_model) — text embeddings
            e_b: (B, d_model) — audio embeddings

        Returns:
            e_a_prime: (B, d_model)
            e_b_prime: (B, d_model)
            attn_weight_a: (B, 1, 1) — attention weights (text→audio)
        """
        # Unsqueeze to (B, 1, d_model) for MultiheadAttention seq format
        e_a_seq = e_a.unsqueeze(1)   # (B, 1, d)
        e_b_seq = e_b.unsqueeze(1)   # (B, 1, d)

        # Text attends over audio: Q=text, K=V=audio
        e_a_prime, attn_w_a = self.attn_ab(
            query=e_a_seq,
            key=e_b_seq,
            value=e_b_seq,
        )  # e_a_prime: (B, 1, d)

        # Audio attends over text: Q=audio, K=V=text
        e_b_prime, attn_w_b = self.attn_ba(
            query=e_b_seq,
            key=e_a_seq,
            value=e_a_seq,
        )  # e_b_prime: (B, 1, d)

        # Residual + LayerNorm
        e_a_prime = self.norm_a(e_a + e_a_prime.squeeze(1))
        e_b_prime = self.norm_b(e_b + e_b_prime.squeeze(1))

        return e_a_prime, e_b_prime, attn_w_a


class FeedForwardFusion(nn.Module):
    """
    Feed-forward fusion network (Equation 8).

    e_fused = FFN([e_text' ∥ e_audio'])

    Two-layer MLP with GELU activation and residual connection.

    Args:
        in_dim: Concatenated input dimension (2 * d_model).
        out_dim: Output fused embedding dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, e_text_prime: torch.Tensor, e_audio_prime: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([e_text_prime, e_audio_prime], dim=-1)  # (B, 2*d_model)
        return self.ffn(concat)                                     # (B, out_dim)


class CMAFModule(nn.Module):
    """
    Cross-Modal Attention Fusion (CMAF) — MindSync core fusion module.
    Implements bidirectional cross-attention between text and audio streams.

    Full pipeline (Section 3.4):
        1. Project e_text (R^1024) and e_audio (R^768) → shared d_model
        2. Bidirectional cross-modal attention (multiple layers)
        3. Concatenate + FFN → e_fused
        4. Final classification head → ŷ_final

    Args:
        text_dim: Text encoder output dimension (1024 for RoBERTa-Large).
        audio_dim: Audio encoder output dimension (768 for wav2vec 2.0).
        d_model: Shared projection dimension.
        num_heads: Cross-attention heads.
        num_layers: Number of stacked cross-attention blocks.
        num_classes: Output classes (4 clusters).
        dropout: Dropout rate throughout.
    """

    def __init__(
        self,
        text_dim: int = 1024,
        audio_dim: int = 768,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        num_classes: int = NUM_CLUSTERS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # ── Projection layers ────────────────────────────────────────────────
        self.text_proj = ProjectionLayer(text_dim, d_model, dropout)
        self.audio_proj = ProjectionLayer(audio_dim, d_model, dropout)

        # ── Stacked cross-modal attention blocks ─────────────────────────────
        self.cross_attn_layers = nn.ModuleList(
            [CrossModalAttentionBlock(d_model, num_heads, dropout) for _ in range(num_layers)]
        )

        # ── Feed-forward fusion (Eq. 8) ──────────────────────────────────────
        fused_dim = d_model  # after FFN
        self.ffn_fusion = FeedForwardFusion(
            in_dim=d_model * 2,
            out_dim=fused_dim,
            dropout=dropout,
        )

        # ── Final classification head (Eq. 9) ────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, num_classes),
        )

    def forward(
        self,
        e_text: torch.Tensor,
        e_audio: torch.Tensor,
    ) -> dict:
        """
        Args:
            e_text: (B, 1024) text embeddings from RoBERTa encoder.
            e_audio: (B, 768) audio embeddings from wav2vec encoder.

        Returns:
            dict with keys:
                'logits': (B, num_classes) — ŷ_final
                'embedding': (B, d_model) — e_fused
                'attn_weights': (B, 1, 1) — attention weights (text→audio)
        """
        # ── Project to shared space ──────────────────────────────────────────
        h_text = self.text_proj(e_text)     # (B, d_model)
        h_audio = self.audio_proj(e_audio)  # (B, d_model)

        # ── Stacked cross-modal attention ────────────────────────────────────
        attn_weights = None
        for layer in self.cross_attn_layers:
            h_text, h_audio, attn_weights = layer(h_text, h_audio)

        # ── FFN Fusion ───────────────────────────────────────────────────────
        e_fused = self.ffn_fusion(h_text, h_audio)  # (B, fused_dim)

        # ── Final classification ─────────────────────────────────────────────
        logits = self.classifier(e_fused)  # (B, num_classes)

        return {
            "logits": logits,
            "embedding": e_fused,
            "attn_weights": attn_weights,
        }
