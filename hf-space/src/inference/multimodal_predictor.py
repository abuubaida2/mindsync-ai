"""
Multimodal predictor for MindSync (paper version).

Loads the Stage 1 text classifier (RoBERTa-Large) and the Stage 2 v3 audio classifier
(wav2vec2-Large-960h with layer-weighted, padding-free pooling) and returns:
  - per-stream predictions and probabilities
  - a fused prediction by late fusion (mean of the two posteriors), which was at least
    as accurate as cross-modal attention on real conversational data (MELD) in the paper
  - the Jensen-Shannon divergence between the two posteriors as the incongruence score
  - an alert when the two streams pick different clusters or the divergence exceeds 0.5

Three modes via .predict(): text + audio, text only, audio only.
"""

from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor

from src.data.emotion_clusters import CLUSTER_LABELS, IDX_TO_CLUSTER
from src.data.text_preprocessing import TextPreprocessor
from src.models.text_model import MindSyncTextModel
from src.models.audio_model_v3 import AudioModelV3
from src.models.fusion_model import jensen_shannon_divergence

JSD_HIGH_THRESHOLD = 0.5


def _load_state_dict(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


class MultimodalPredictor:
    def __init__(
        self,
        text_ckpt: str,
        audio_ckpt: str,
        device: Union[str, torch.device] = "auto",
        text_model_name: str = "roberta-large",
        audio_model_name: str = "facebook/wav2vec2-large-960h",
        target_sr: int = 16_000,
        max_audio_samples: int = 80_000,
    ) -> None:
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.target_sr = target_sr
        self.max_audio_samples = max_audio_samples

        # Text (Stage 1)
        self.text_preprocessor = TextPreprocessor(model_name=text_model_name)
        self.text_model = MindSyncTextModel(model_name=text_model_name, num_classes=4)
        self.text_model.load_state_dict(_load_state_dict(text_ckpt))
        self.text_model.eval().to(self.device)

        # Audio (Stage 2 v3)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(audio_model_name)
        self.audio_model = AudioModelV3(model_name=audio_model_name, num_classes=4, spec_augment=False)
        self.audio_model.load_state_dict(_load_state_dict(audio_ckpt))
        self.audio_model.eval().to(self.device)

        print(f"MultimodalPredictor (Stage 1 + Stage 2 v3, late fusion) ready on {self.device}")

    # ── audio: same evaluation preprocessing as in the paper ───────────────────
    def _audio_inputs(self, path: Optional[str], array: Optional[np.ndarray], sr: int):
        """mono 16 kHz -> trim silence (30 dB) -> centre 5 s crop -> per-clip zero-mean/unit-variance.
        One clip at a time, so no zero padding reaches the group-normalised -960h encoder."""
        import librosa

        if array is not None:
            wav = np.asarray(array, dtype=np.float32)
            if wav.ndim > 1:
                wav = wav.mean(axis=-1)
            if sr != self.target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.target_sr)
        else:
            wav, _ = librosa.load(path, sr=self.target_sr, mono=True)
        wav, _ = librosa.effects.trim(wav, top_db=30)
        if len(wav) > self.max_audio_samples:
            start = (len(wav) - self.max_audio_samples) // 2
            wav = wav[start : start + self.max_audio_samples]
        if len(wav) < 400:
            wav = np.pad(wav, (0, 400 - len(wav)))
        x = self.feature_extractor(wav, sampling_rate=self.target_sr, return_tensors="np")["input_values"][0]
        iv = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        lengths = torch.tensor([iv.shape[1]], device=self.device)
        return iv, lengths

    def _text_probs(self, text: str) -> torch.Tensor:
        enc = self.text_preprocessor(text)
        input_ids = enc["input_ids"].unsqueeze(0).to(self.device)
        attn_mask = enc["attention_mask"].unsqueeze(0).to(self.device)
        out = self.text_model(input_ids, attn_mask)
        return F.softmax(out["logits"], dim=-1).squeeze(0)

    def _audio_probs(self, path, array, sr) -> torch.Tensor:
        iv, lengths = self._audio_inputs(path, array, sr)
        out = self.audio_model(iv, lengths)
        return F.softmax(out["logits"], dim=-1).squeeze(0)

    # ── prediction modes ────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        text: Optional[str] = None,
        audio_path: Optional[str] = None,
        audio_array: Optional[np.ndarray] = None,
        sr: int = 16_000,
    ) -> dict:
        has_text = bool(text and text.strip())
        has_audio = (audio_path is not None) or (audio_array is not None)
        if has_text and has_audio:
            return self._predict_multimodal(text, audio_path, audio_array, sr)
        if has_text:
            return self._single("text", self._text_probs(text))
        if has_audio:
            return self._single("audio", self._audio_probs(audio_path, audio_array, sr))
        raise ValueError("predict() requires at least one of text or audio")

    def _predict_multimodal(self, text, audio_path, audio_array, sr) -> dict:
        text_probs = self._text_probs(text)
        audio_probs = self._audio_probs(audio_path, audio_array, sr)
        fused_probs = 0.5 * (text_probs + audio_probs)                     # late fusion
        jsd = jensen_shannon_divergence(text_probs.unsqueeze(0), audio_probs.unsqueeze(0)).clamp(0.0, 1.0).item()

        fused_idx, text_idx, audio_idx = int(fused_probs.argmax()), int(text_probs.argmax()), int(audio_probs.argmax())
        alert = (text_idx != audio_idx) or jsd > JSD_HIGH_THRESHOLD
        if alert:
            msg = (
                f"⚠ Cross-modal incongruence detected (δ={jsd:.2f}). "
                f"Text suggests '{IDX_TO_CLUSTER[text_idx]}' while voice suggests "
                f"'{IDX_TO_CLUSTER[audio_idx]}'. Possible emotional masking — "
                "consider whether you want to talk to someone you trust."
            )
        else:
            msg = (
                f"✓ Text and voice are congruent (δ={jsd:.2f}). "
                f"Predicted emotional state: '{IDX_TO_CLUSTER[fused_idx]}'."
            )
        return {
            "predicted_emotion": IDX_TO_CLUSTER[fused_idx],
            "text_emotion": IDX_TO_CLUSTER[text_idx],
            "audio_emotion": IDX_TO_CLUSTER[audio_idx],
            "fused_probabilities": _to_dict(fused_probs),
            "text_probabilities": _to_dict(text_probs),
            "audio_probabilities": _to_dict(audio_probs),
            "incongruence_score": jsd,
            "clinical_alert": alert,
            "clinical_message": msg,
        }

    def _single(self, stream: str, probs: torch.Tensor) -> dict:
        idx = int(probs.argmax())
        cluster = IDX_TO_CLUSTER[idx]
        if stream == "text":
            msg = (f"📝 Text-only analysis. Predicted emotional state: '{cluster}'. "
                   "Audio stream not active — cross-modal incongruence not assessed.")
        else:
            msg = (f"🎙 Voice-only analysis. Predicted emotional state: '{cluster}'. "
                   "Text stream not active — cross-modal incongruence not assessed.")
        prob_d = _to_dict(probs)
        return {
            "predicted_emotion": cluster,
            "text_emotion": cluster,
            "audio_emotion": cluster,
            "fused_probabilities": prob_d,
            "text_probabilities": prob_d,
            "audio_probabilities": prob_d,
            "incongruence_score": 0.0,
            "clinical_alert": False,
            "clinical_message": msg,
        }


def _to_dict(probs: torch.Tensor) -> dict:
    return {CLUSTER_LABELS[i]: float(probs[i].cpu()) for i in range(4)}
