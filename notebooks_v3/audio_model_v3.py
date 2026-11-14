"""Shared Stage 2 v3 audio model (wav2vec2-Large-960h, same checkpoint as v1).

Fixes relative to v1 (which mean-pooled the LAST layer over zero-padding):
  1. learned softmax-weighted sum over all 25 hidden states (each LayerNorm-ed first)
  2. mean+std pooling over VALID frames only (frame mask from true input lengths)
  3. (training side) waveform augmentation + light SpecAugment

Both the Stage 2 v3 training notebook and the Stage 3 v3 notebook paste this
class VERBATIM so the saved state_dict always loads.
"""
import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

AUDIO_MODEL_NAME = 'facebook/wav2vec2-large-960h'


class AudioModelV3(nn.Module):
    def __init__(self, model_name=AUDIO_MODEL_NAME, num_classes=4, freeze_layers=8,
                 dropout=0.1, spec_augment=True):
        super().__init__()
        cfg = Wav2Vec2Config.from_pretrained(model_name)
        cfg.output_hidden_states = True
        cfg.apply_spec_augment = spec_augment
        cfg.mask_time_prob = 0.05
        cfg.mask_time_length = 5
        cfg.mask_feature_prob = 0.0
        self.enc = Wav2Vec2Model.from_pretrained(model_name, config=cfg)
        self.enc.freeze_feature_encoder()
        for layer in self.enc.encoder.layers[:freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False
        # 'layer' norm models accept attention_mask; 'group' norm models (e.g. -960h) must not get one
        self.use_attention_mask = (cfg.feat_extract_norm == 'layer')
        H = cfg.hidden_size
        L = cfg.num_hidden_layers + 1
        self.layer_w = nn.Parameter(torch.zeros(L))
        self.layer_norm = nn.LayerNorm(H)
        self.proj = nn.Sequential(nn.LayerNorm(2 * H), nn.Linear(2 * H, H))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(H, H // 2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(H // 2, num_classes))

    def frames(self, input_values, lengths):
        """input_values: (B, S) zero-padded waveforms; lengths: (B,) true sample counts.
        Returns frame features (B, T, H) and a bool frame mask (B, T), True = valid."""
        if self.use_attention_mask:
            am = (torch.arange(input_values.shape[1], device=input_values.device)[None, :]
                  < lengths[:, None]).long()
            out = self.enc(input_values=input_values, attention_mask=am, output_hidden_states=True)
        else:
            out = self.enc(input_values=input_values, output_hidden_states=True)
        hs = torch.stack([self.layer_norm(h) for h in out.hidden_states], 0)   # (L, B, T, H)
        w = torch.softmax(self.layer_w, 0)
        f = (w[:, None, None, None] * hs).sum(0)                               # (B, T, H)
        flen = self.enc._get_feat_extract_output_lengths(lengths).clamp(min=1)
        mask = torch.arange(f.shape[1], device=f.device)[None, :] < flen[:, None]
        return f, mask

    @staticmethod
    def pool(f, mask):
        m = mask.unsqueeze(-1).to(f.dtype)
        n = m.sum(1).clamp_min(1.0)
        mean = (f * m).sum(1) / n
        var = (((f - mean[:, None, :]) ** 2) * m).sum(1) / n
        return torch.cat([mean, var.clamp_min(1e-6).sqrt()], -1)

    def forward(self, input_values, lengths):
        f, mask = self.frames(input_values, lengths)
        emb = self.proj(self.pool(f, mask))
        return {'embedding': emb, 'logits': self.head(emb)}
