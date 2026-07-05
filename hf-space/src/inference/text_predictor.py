"""
Text-only inference predictor for MindSync.

Loads a trained MindSyncTextModel checkpoint (RoBERTa-Large fine-tuned on
GoEmotions). Used when only the text stream checkpoint is available — the
audio stream is reported as unavailable and incongruence cannot be computed.

This is the fallback predictor for the backend when a full multimodal
checkpoint is not yet ready.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Union

from src.models.text_model import MindSyncTextModel
from src.data.text_preprocessing import TextPreprocessor
from src.data.emotion_clusters import IDX_TO_CLUSTER, CLUSTER_LABELS


class TextOnlyPredictor:
    """
    Text-only emotion predictor backed by a trained MindSyncTextModel.

    For text input: produces real cluster predictions and probabilities.
    For audio / multimodal: text branch is real; audio fields mirror the text
    output (audio model not loaded), incongruence_score = 0, no clinical alert.

    Args:
        checkpoint_path: Path to best_text_model.pt (MindSyncTextModel state dict).
        device: 'cuda', 'cpu', or 'auto'.
        text_model_name: HuggingFace model name (must match the checkpoint).
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Union[str, torch.device] = "auto",
        text_model_name: str = "roberta-base",
    ) -> None:
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.text_preprocessor = TextPreprocessor(model_name=text_model_name)
        self.model = MindSyncTextModel(model_name=text_model_name, num_classes=4)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state)
        print(f"Loaded text-only checkpoint from {checkpoint_path}")

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
        Run text-only inference. Audio arguments are accepted for API
        compatibility but ignored (no audio model loaded).

        Returns the same dict shape as MindSyncPredictor.predict():
            predicted_emotion, text_emotion, audio_emotion,
            fused_probabilities, text_probabilities, audio_probabilities,
            incongruence_score, clinical_alert, clinical_message
        """
        text = text or ""
        enc = self.text_preprocessor(text)
        input_ids = enc["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = enc["attention_mask"].unsqueeze(0).to(self.device)

        out = self.model(input_ids, attention_mask)
        probs = F.softmax(out["logits"], dim=-1).squeeze(0).cpu().numpy()

        pred_idx = int(probs.argmax())
        pred_cluster = IDX_TO_CLUSTER[pred_idx]
        prob_dict = {CLUSTER_LABELS[i]: float(probs[i]) for i in range(4)}

        clinical_message = (
            f"\U0001F4DD Text-only analysis. Predicted emotional state: '{pred_cluster}'. "
            "Audio stream not active — cross-modal incongruence not assessed."
        )

        return {
            "predicted_emotion": pred_cluster,
            "text_emotion": pred_cluster,
            "audio_emotion": pred_cluster,          # mirrors text (no audio model)
            "fused_probabilities": prob_dict,
            "text_probabilities": prob_dict,
            "audio_probabilities": prob_dict,       # mirrors text
            "incongruence_score": 0.0,
            "clinical_alert": False,
            "clinical_message": clinical_message,
        }
