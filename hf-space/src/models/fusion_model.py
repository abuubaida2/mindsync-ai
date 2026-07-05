"""
Cross-Modal Attention Fusion (CMAF) + JSD for MindSync.

Architecture (Stage 3 notebook, matching exactly):
    e_text  ∈ R^1024  ──┐
                        ├─► text_proj / audio_proj → d=256
    e_audio ∈ R^1024  ──┘
                        │
    Bidirectional cross-attention:
        t_attn = MultiheadAttn(query=text, key=audio, value=audio)
        a_attn = MultiheadAttn(query=audio, key=text, value=text)
        t = LayerNorm(text + t_attn)
        a = LayerNorm(audio + a_attn)

    Fused classifier on cat([t, a]) → 4-cluster logits
    Per-stream auxiliary heads → text_aux_logits, audio_aux_logits
    JSD(text_aux_probs ‖ audio_aux_probs) → incongruence score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.emotion_clusters import NUM_CLUSTERS


class CMAFFusion(nn.Module):
    def __init__(
        self,
        text_dim: int = 1024,
        audio_dim: int = 1024,
        d: int = 256,
        num_classes: int = NUM_CLUSTERS,
        dropout: float = 0.1,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.text_proj = nn.Linear(text_dim, d)
        self.audio_proj = nn.Linear(audio_dim, d)
        self.t2a = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.a2t = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(d)
        self.norm_a = nn.LayerNorm(d)
        self.fuse = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, num_classes),
        )
        self.head_text = nn.Linear(d, num_classes)
        self.head_audio = nn.Linear(d, num_classes)

    def forward(self, e_text: torch.Tensor, e_audio: torch.Tensor) -> dict:
        """
        Args:
            e_text:  (B, 1024)
            e_audio: (B, 1024)
        Returns:
            dict with 'fused_logits', 'text_logits', 'audio_logits' — all (B, num_classes)
        """
        t = self.text_proj(e_text).unsqueeze(1)   # (B, 1, d)
        a = self.audio_proj(e_audio).unsqueeze(1) # (B, 1, d)
        t_attn, _ = self.t2a(t, a, a)
        a_attn, _ = self.a2t(a, t, t)
        t = self.norm_t(t + t_attn).squeeze(1)    # (B, d)
        a = self.norm_a(a + a_attn).squeeze(1)    # (B, d)
        fused = torch.cat([t, a], dim=-1)         # (B, 2d)
        return {
            "fused_logits": self.fuse(fused),
            "text_logits": self.head_text(t),
            "audio_logits": self.head_audio(a),
        }


def jensen_shannon_divergence(
    p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """JSD(P ‖ Q) in bits (log base 2). Returns shape (B,) in [0, 1]."""
    m = 0.5 * (p + q)
    kl_pm = (p * (p.clamp_min(eps).log2() - m.clamp_min(eps).log2())).sum(-1)
    kl_qm = (q * (q.clamp_min(eps).log2() - m.clamp_min(eps).log2())).sum(-1)
    return 0.5 * (kl_pm + kl_qm)
