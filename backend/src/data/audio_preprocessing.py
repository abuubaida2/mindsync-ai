"""
Audio preprocessing pipeline for RAVDESS and IEMOCAP datasets.
Implements Section 3.2.2 and Equation (2) of MindSync paper.

Equation (2):
    z_t = f_CNN(x_{t:t+w}),  c_t = Transformer(z_{1:t})

where x = raw audio, w = context window, z_t = latent representations,
c_t = contextual representations (handled inside wav2vec 2.0).
"""

from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor

# ── Constants (from paper) ──────────────────────────────────────────────────────
SAMPLE_RATE: int = 16_000           # 16 kHz mono
N_MFCC: int = 40                   # MFCC feature dimension (T × 40)
WIN_LENGTH_MS: float = 25.0        # Hamming window = 25 ms
HOP_LENGTH_MS: float = 10.0        # 10 ms hop


def load_audio(path: str, sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load an audio file and resample to target sample rate."""
    waveform, orig_sr = librosa.load(path, sr=sr, mono=True)
    return waveform, sr


def extract_mfcc(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    win_length_ms: float = WIN_LENGTH_MS,
    hop_length_ms: float = HOP_LENGTH_MS,
) -> np.ndarray:
    """
    Compute MFCC feature matrix.

    Returns:
        mfcc: np.ndarray of shape (T, n_mfcc=40)
    """
    win_samples = int(win_length_ms * sr / 1000)
    hop_samples = int(hop_length_ms * sr / 1000)

    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=win_samples,
        hop_length=hop_samples,
        window="hamm",
    )
    # (n_mfcc, T) → (T, n_mfcc)
    return mfcc.T


def pad_or_truncate(
    waveform: np.ndarray,
    max_length_sec: float = 10.0,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Pad or truncate waveform to fixed number of samples."""
    max_samples = int(max_length_sec * sr)
    if len(waveform) < max_samples:
        waveform = np.pad(waveform, (0, max_samples - len(waveform)))
    else:
        waveform = waveform[:max_samples]
    return waveform


class AudioPreprocessor:
    """
    End-to-end audio preprocessing for wav2vec 2.0 input.

    Steps:
        1. Load and resample to 16 kHz mono
        2. Pad / truncate to max_length_sec
        3. Extract raw waveform values (feature extractor normalises)
        4. Optionally extract MFCC (for auxiliary analysis)

    Args:
        model_name: HuggingFace model name for feature extractor.
        max_length_sec: Maximum audio length in seconds.
        sample_rate: Target sample rate (default 16 kHz).
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-large-960h",
        max_length_sec: float = 10.0,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.max_length_sec = max_length_sec
        self.sample_rate = sample_rate
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    def __call__(
        self, path: str, label: Optional[int] = None, return_mfcc: bool = False
    ) -> dict:
        """
        Preprocess a single audio file.

        Returns dict with input_values (for wav2vec 2.0), optionally mfcc and label.
        """
        waveform, _ = load_audio(path, sr=self.sample_rate)
        waveform = pad_or_truncate(waveform, self.max_length_sec, self.sample_rate)

        inputs = self.feature_extractor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        result = {
            "input_values": inputs["input_values"].squeeze(0),
        }
        if return_mfcc:
            result["mfcc"] = torch.tensor(
                extract_mfcc(waveform, sr=self.sample_rate), dtype=torch.float32
            )
        if label is not None:
            result["label"] = label
        return result

    @staticmethod
    def from_array(
        waveform: np.ndarray,
        sr: int = SAMPLE_RATE,
        model_name: str = "facebook/wav2vec2-large-960h",
    ) -> dict:
        """Convenience method for in-memory waveform array."""
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        inputs = extractor(
            waveform, sampling_rate=sr, return_tensors="pt", padding=True
        )
        return {"input_values": inputs["input_values"].squeeze(0)}
