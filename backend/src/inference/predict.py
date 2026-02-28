"""
MindSync Inference Pipeline.
Real-time prediction from text + audio inputs.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union

from src.models.mindsync import MindSync
from src.data.text_preprocessing import TextPreprocessor
from src.data.audio_preprocessing import AudioPreprocessor
from src.data.emotion_clusters import IDX_TO_CLUSTER, CLUSTER_LABELS
from src.utils.seed import set_seed


class MindSyncPredictor:
    """
    End-to-end inference predictor for MindSync.

    Loads a trained MindSync checkpoint and produces:
        - Dominant emotion cluster prediction
        - Per-cluster probability distributions (text, audio, fused)
        - Incongruence score δ and clinical alert flag

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        device: Compute device ('cuda', 'cpu', or 'auto').
        text_model_name: HuggingFace model name for text preprocessor.
        audio_model_name: HuggingFace model name for audio preprocessor.
        incongruence_threshold: JSD threshold (default 0.5).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Union[str, torch.device] = "auto",
        text_model_name: str = "roberta-large",
        audio_model_name: str = "facebook/wav2vec2-large-960h",
        incongruence_threshold: float = 0.5,
    ) -> None:
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.text_preprocessor = TextPreprocessor(model_name=text_model_name)
        self.audio_preprocessor = AudioPreprocessor(model_name=audio_model_name)

        self.model = MindSync(
            text_model_name=text_model_name,
            audio_model_name=audio_model_name,
            incongruence_threshold=incongruence_threshold,
        )

        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print("No checkpoint loaded — using random weights (for testing only)")

        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def predict(
        self,
        text: str,
        audio_path: Optional[str] = None,
        audio_array: Optional[np.ndarray] = None,
        sr: int = 16_000,
    ) -> dict:
        """
        Run full multimodal inference on a text + audio input.

        Args:
            text: Input sentence/utterance text.
            audio_path: Path to audio file (.wav, .mp3, etc.).
            audio_array: Pre-loaded audio waveform (NumPy array).
            sr: Sample rate if providing audio_array.

        Returns:
            dict:
                predicted_emotion: Dominant fused cluster name
                text_emotion: Text-stream dominant cluster
                audio_emotion: Audio-stream dominant cluster
                fused_probabilities: {cluster: probability}
                text_probabilities: {cluster: probability}
                audio_probabilities: {cluster: probability}
                incongruence_score: JSD score δ ∈ [0, 1]
                clinical_alert: bool — True if δ > threshold
                clinical_message: Human-readable alert message
        """
        # ── Preprocess text ──────────────────────────────────────────────────
        text_enc = self.text_preprocessor(text)
        input_ids = text_enc["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = text_enc["attention_mask"].unsqueeze(0).to(self.device)

        # ── Preprocess audio ─────────────────────────────────────────────────
        if audio_path is not None:
            audio_enc = self.audio_preprocessor(audio_path)
        elif audio_array is not None:
            waveform = audio_array.astype(np.float32)
            if sr != self.audio_preprocessor.sample_rate:
                import librosa
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.audio_preprocessor.sample_rate)
            from src.data.audio_preprocessing import pad_or_truncate
            waveform = pad_or_truncate(waveform, self.audio_preprocessor.max_length_sec, self.audio_preprocessor.sample_rate)
            inputs = self.audio_preprocessor.feature_extractor(
                waveform, sampling_rate=self.audio_preprocessor.sample_rate,
                return_tensors="pt", padding=True,
            )
            audio_enc = {"input_values": inputs["input_values"].squeeze(0)}
        else:
            # Silent audio fallback when no audio is provided
            from src.data.audio_preprocessing import pad_or_truncate
            silence = np.zeros(int(self.audio_preprocessor.sample_rate * 5), dtype=np.float32)
            silence = pad_or_truncate(silence, self.audio_preprocessor.max_length_sec, self.audio_preprocessor.sample_rate)
            inputs = self.audio_preprocessor.feature_extractor(
                silence, sampling_rate=self.audio_preprocessor.sample_rate,
                return_tensors="pt", padding=True,
            )
            audio_enc = {"input_values": inputs["input_values"].squeeze(0)}

        input_values = audio_enc["input_values"].unsqueeze(0).to(self.device)

        # ── Forward pass ─────────────────────────────────────────────────────
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_values=input_values,
        )

        import torch.nn.functional as F
        probs_final = F.softmax(outputs["logits_final"], dim=-1).squeeze(0).cpu().numpy()
        probs_text = F.softmax(outputs["logits_text"], dim=-1).squeeze(0).cpu().numpy()
        probs_audio = F.softmax(outputs["logits_audio"], dim=-1).squeeze(0).cpu().numpy()

        pred_cluster = IDX_TO_CLUSTER[int(probs_final.argmax())]
        text_cluster = IDX_TO_CLUSTER[int(probs_text.argmax())]
        audio_cluster = IDX_TO_CLUSTER[int(probs_audio.argmax())]

        incong_score = float(outputs["incongruence_scores"].item())
        clinical_alert = bool(outputs["incongruence_flags"].item())

        if clinical_alert:
            clinical_message = (
                f"⚠ CLINICAL ALERT: Cross-modal incongruence detected (δ={incong_score:.3f}). "
                f"Text signals '{text_cluster}' but audio signals '{audio_cluster}'. "
                "Recommend review by a qualified mental health professional."
            )
        else:
            clinical_message = (
                f"✓ Congruent signals (δ={incong_score:.3f}). "
                f"Both modalities indicate '{pred_cluster}'."
            )

        return {
            "predicted_emotion": pred_cluster,
            "text_emotion": text_cluster,
            "audio_emotion": audio_cluster,
            "fused_probabilities": {CLUSTER_LABELS[i]: float(probs_final[i]) for i in range(4)},
            "text_probabilities": {CLUSTER_LABELS[i]: float(probs_text[i]) for i in range(4)},
            "audio_probabilities": {CLUSTER_LABELS[i]: float(probs_audio[i]) for i in range(4)},
            "incongruence_score": incong_score,
            "clinical_alert": clinical_alert,
            "clinical_message": clinical_message,
        }

    def predict_batch(self, texts: list, audio_paths: list) -> list:
        """Batch prediction (texts and audio_paths must be same length)."""
        assert len(texts) == len(audio_paths)
        return [self.predict(t, a) for t, a in zip(texts, audio_paths)]
