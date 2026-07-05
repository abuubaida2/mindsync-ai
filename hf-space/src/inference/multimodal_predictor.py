"""
Full multimodal predictor for MindSync.

Loads the trained Stage 1 (text), Stage 2 (audio), and Stage 3 (CMAF fusion)
checkpoints, runs the full pipeline, and returns:
  - fused prediction (CMAF) + probabilities
  - per-stream auxiliary predictions (text/audio aux heads from fusion module)
  - JSD between aux distributions → incongruence score
  - clinical alert flag when JSD exceeds the threshold

Three prediction modes are supported via .predict():
  1. text + audio   → real multimodal with JSD-based incongruence
  2. text only      → text-stream classifier (Stage 1 head), audio mirrored
  3. audio only     → audio-stream classifier (Stage 2 head), text mirrored
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from src.data.emotion_clusters import CLUSTER_LABELS, IDX_TO_CLUSTER
from src.data.text_preprocessing import TextPreprocessor
from src.models.text_model import MindSyncTextModel
from src.models.audio_model import MindSyncAudioModel
from src.models.fusion_model import CMAFFusion, jensen_shannon_divergence

# JSD is computed between the Stage 1 standalone text head and the Stage 2
# standalone audio head — i.e. the two single-modality classifiers, not the
# fusion's auxiliary heads. Those auxiliary heads share cross-attention so
# they no longer represent pure per-stream views and their JSD collapses to
# a tiny range. With true independent classifiers JSD naturally spans [0, 1]:
# congruent ≈ 0.0–0.1, clear cluster disagreement ≈ 0.4–1.0.
#
# Alert logic combines a categorical check (do the two modalities pick the
# same cluster at argmax?) with a magnitude check (how far apart are the
# distributions overall?). Either condition is sufficient — a categorical
# mismatch IS the clinically interesting signal, even when JSD is borderline.
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
        fusion_ckpt: str,
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

        # Text
        self.text_preprocessor = TextPreprocessor(model_name=text_model_name)
        self.text_model = MindSyncTextModel(model_name=text_model_name, num_classes=4)
        self.text_model.load_state_dict(_load_state_dict(text_ckpt))
        self.text_model.eval().to(self.device)

        # Audio
        self.audio_model = MindSyncAudioModel(model_name=audio_model_name, num_classes=4)
        self.audio_model.load_state_dict(_load_state_dict(audio_ckpt))
        self.audio_model.eval().to(self.device)

        # Fusion
        self.fusion = CMAFFusion()
        self.fusion.load_state_dict(_load_state_dict(fusion_ckpt))
        self.fusion.eval().to(self.device)

        print(f"MultimodalPredictor ready on {self.device}")

    # ── audio helpers ───────────────────────────────────────────────────────
    def _load_audio(self, path: Optional[str], array: Optional[np.ndarray], sr: int) -> torch.Tensor:
        """Read audio (path OR raw array) → mono 16-kHz waveform, trimmed/padded
        to max_audio_samples, normalised, as a (1, T) tensor on device."""
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
        else:
            wav = np.pad(wav, (0, self.max_audio_samples - len(wav)))
        wav = wav / (np.abs(wav).max() + 1e-8)
        return torch.tensor(wav, dtype=torch.float32, device=self.device).unsqueeze(0)

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
            return self._predict_text_only(text)
        if has_audio:
            return self._predict_audio_only(audio_path, audio_array, sr)
        raise ValueError("predict() requires at least one of text or audio")

    def _predict_multimodal(
        self, text: str, audio_path: Optional[str], audio_array: Optional[np.ndarray], sr: int
    ) -> dict:
        # Stage 1: independent text prediction (e_text + standalone classifier)
        enc = self.text_preprocessor(text)
        input_ids = enc["input_ids"].unsqueeze(0).to(self.device)
        attn_mask = enc["attention_mask"].unsqueeze(0).to(self.device)
        text_out = self.text_model(input_ids, attn_mask)
        e_text = text_out["embedding"]                                # (1, 1024)
        text_probs = F.softmax(text_out["logits"], dim=-1).squeeze(0)  # Stage 1 pure-text

        # Stage 2: independent audio prediction (e_audio + standalone classifier)
        iv = self._load_audio(audio_path, audio_array, sr)
        audio_out = self.audio_model(iv)
        e_audio = audio_out["embedding"]                                # (1, 1024)
        audio_probs = F.softmax(audio_out["logits"], dim=-1).squeeze(0) # Stage 2 pure-audio

        # Stage 3: CMAF fusion gives the integrated multimodal prediction
        f = self.fusion(e_text, e_audio)
        fused_probs = F.softmax(f["fused_logits"], dim=-1).squeeze(0)

        # JSD between the two truly independent classifiers — this is the
        # meaningful incongruence signal (no cross-attention contamination)
        jsd = jensen_shannon_divergence(
            text_probs.unsqueeze(0), audio_probs.unsqueeze(0)
        ).clamp(0.0, 1.0).item()

        fused_idx = int(fused_probs.argmax())
        text_idx = int(text_probs.argmax())
        audio_idx = int(audio_probs.argmax())

        categorical_mismatch = text_idx != audio_idx
        alert = categorical_mismatch or jsd > JSD_HIGH_THRESHOLD
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

    def _predict_text_only(self, text: str) -> dict:
        # Use the Stage-1 text classifier (NOT the fusion's aux head; that one
        # was trained on projected+attended embeddings and only makes sense
        # when audio is present).
        enc = self.text_preprocessor(text)
        input_ids = enc["input_ids"].unsqueeze(0).to(self.device)
        attn_mask = enc["attention_mask"].unsqueeze(0).to(self.device)
        out = self.text_model(input_ids, attn_mask)
        probs = F.softmax(out["logits"], dim=-1).squeeze(0)
        idx = int(probs.argmax())
        cluster = IDX_TO_CLUSTER[idx]
        msg = (
            f"📝 Text-only analysis. Predicted emotional state: '{cluster}'. "
            "Audio stream not active — cross-modal incongruence not assessed."
        )
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

    def _predict_audio_only(
        self, audio_path: Optional[str], audio_array: Optional[np.ndarray], sr: int
    ) -> dict:
        iv = self._load_audio(audio_path, audio_array, sr)
        out = self.audio_model(iv)
        probs = F.softmax(out["logits"], dim=-1).squeeze(0)
        idx = int(probs.argmax())
        cluster = IDX_TO_CLUSTER[idx]
        msg = (
            f"🎙 Voice-only analysis. Predicted emotional state: '{cluster}'. "
            "Text stream not active — cross-modal incongruence not assessed."
        )
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
