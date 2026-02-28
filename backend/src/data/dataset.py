"""
PyTorch Dataset classes for GoEmotions, RAVDESS, IEMOCAP,
and the synthetic paired multimodal dataset.
"""

import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from datasets import load_dataset

from src.data.emotion_clusters import (
    CLUSTER_TO_IDX,
    map_go_emotions_to_idx,
    map_ravdess_label,
    map_iemocap_label,
)
from src.data.text_preprocessing import TextPreprocessor
from src.data.audio_preprocessing import AudioPreprocessor


# ── GoEmotions Text Dataset ─────────────────────────────────────────────────────

class GoEmotionsDataset(Dataset):
    """
    GoEmotions dataset via HuggingFace Datasets Hub.
    58,009 Reddit comments, 27 emotion labels → 4 clinical clusters.

    Split: train/validation/test (80/10/10 stratified).
    Source: google-research-datasets/go_emotions

    Args:
        split: 'train', 'validation', or 'test'.
        preprocessor: TextPreprocessor instance.
        max_samples: Optional cap for debugging.
    """

    def __init__(
        self,
        split: str = "train",
        preprocessor: Optional[TextPreprocessor] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self.split = split
        self.preprocessor = preprocessor or TextPreprocessor()

        raw = load_dataset("google-research-datasets/go_emotions", "simplified", split=split)
        if max_samples:
            raw = raw.select(range(min(max_samples, len(raw))))
        self.data = raw

        # Build list of (text, cluster_idx) pairs
        # GoEmotions uses multi-label; take first label after confidence thresholding
        label_names = self.data.features["labels"].feature.names
        self.samples: List[Tuple[str, int]] = []
        for item in self.data:
            text = item["text"]
            labels = item["labels"]
            if not labels:
                continue
            # Use first (highest-confidence) label
            label_str = label_names[labels[0]]
            cluster_idx = map_go_emotions_to_idx(label_str)
            self.samples.append((text, cluster_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        text, label = self.samples[idx]
        encoding = self.preprocessor(text)
        encoding["label"] = torch.tensor(label, dtype=torch.long)
        return encoding

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for balanced sampling."""
        from collections import Counter
        counts = Counter(label for _, label in self.samples)
        total = len(self.samples)
        weights = torch.zeros(4)
        for cls, cnt in counts.items():
            weights[cls] = total / (4 * cnt)
        return weights


# ── RAVDESS Audio Dataset ───────────────────────────────────────────────────────

class RAVDESSDataset(Dataset):
    """
    RAVDESS speech dataset.
    1,440 speech files from 24 actors, 8 emotions → 4 clusters.

    Expected directory structure:
        ravdess_root/
            Actor_01/
                03-01-01-01-01-01-01.wav
                ...
            Actor_02/
                ...

    RAVDESS filename structure:
        Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
        Emotion codes: 01=neutral, 02=calm, 03=happy, 04=sad,
                       05=angry, 06=fearful, 07=disgust, 08=surprised

    Args:
        root_dir: Path to RAVDESS root directory.
        split: 'train', 'validation', or 'test'.
        preprocessor: AudioPreprocessor instance.
        split_ratios: (train, val, test) ratios.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        preprocessor: Optional[AudioPreprocessor] = None,
        split_ratios: Tuple[float, float, float] = (0.80, 0.10, 0.10),
        seed: int = 42,
    ) -> None:
        self.preprocessor = preprocessor or AudioPreprocessor()

        all_files = sorted(Path(root_dir).rglob("*.wav"))
        random.seed(seed)
        all_files = list(all_files)
        random.shuffle(all_files)

        n = len(all_files)
        n_train = int(n * split_ratios[0])
        n_val = int(n * split_ratios[1])

        if split == "train":
            self.files = all_files[:n_train]
        elif split == "validation":
            self.files = all_files[n_train: n_train + n_val]
        else:
            self.files = all_files[n_train + n_val:]

    def _parse_label(self, filepath: Path) -> int:
        """Parse emotion code from RAVDESS filename → cluster index."""
        parts = filepath.stem.split("-")
        emotion_code = int(parts[2])
        cluster = map_ravdess_label(emotion_code)
        return CLUSTER_TO_IDX[cluster]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        label = self._parse_label(path)
        encoding = self.preprocessor(str(path))
        encoding["label"] = torch.tensor(label, dtype=torch.long)
        return encoding


# ── IEMOCAP Audio Dataset ───────────────────────────────────────────────────────

class IEMOCAPDataset(Dataset):
    """
    IEMOCAP dataset (~10,039 utterances across 5 sessions).
    Audio files with categorical emotion labels → 4 clusters.

    Expected directory structure:
        iemocap_root/
            Session1/
                sentences/
                    wav/
                        Ses01F_impro01/
                            Ses01F_impro01_F000.wav
            ...
            labels.csv   (pre-generated: utterance_id, emotion, text)

    Args:
        root_dir: Path to IEMOCAP root directory.
        labels_csv: Path to pre-generated labels CSV.
        split: 'train', 'validation', or 'test'.
        preprocessor: AudioPreprocessor instance.
    """

    def __init__(
        self,
        root_dir: str,
        labels_csv: str,
        split: str = "train",
        preprocessor: Optional[AudioPreprocessor] = None,
        split_ratios: Tuple[float, float, float] = (0.80, 0.10, 0.10),
        seed: int = 42,
    ) -> None:
        import pandas as pd

        self.preprocessor = preprocessor or AudioPreprocessor()
        df = pd.read_csv(labels_csv)

        # Filter valid emotions
        valid = list(map_iemocap_label.__code__.co_consts)  # not ideal, use dict
        from src.data.emotion_clusters import IEMOCAP_LABEL_MAP
        df = df[df["emotion"].isin(IEMOCAP_LABEL_MAP.keys())].reset_index(drop=True)

        random.seed(seed)
        indices = list(range(len(df)))
        random.shuffle(indices)

        n = len(indices)
        n_train = int(n * split_ratios[0])
        n_val = int(n * split_ratios[1])

        if split == "train":
            split_idx = indices[:n_train]
        elif split == "validation":
            split_idx = indices[n_train: n_train + n_val]
        else:
            split_idx = indices[n_train + n_val:]

        self.df = df.iloc[split_idx].reset_index(drop=True)
        self.root_dir = root_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        path = os.path.join(self.root_dir, row["wav_path"])
        cluster = map_iemocap_label(row["emotion"])
        label = CLUSTER_TO_IDX[cluster]
        encoding = self.preprocessor(path)
        encoding["label"] = torch.tensor(label, dtype=torch.long)
        return encoding


# ── Synthetic Multimodal Paired Dataset ────────────────────────────────────────

class SyntheticMultimodalDataset(Dataset):
    """
    Synthetically paired text-audio dataset (Section 3.8 of MindSync paper).

    GoEmotions text samples are probabilistically matched to acoustically
    similar audio by shared emotion cluster label. This enables multimodal
    training at scale despite the lack of naturally aligned data.

    Args:
        text_dataset: GoEmotionsDataset instance.
        audio_dataset: RAVDESSDataset or IEMOCAPDataset instance.
        seed: Random seed for reproducible pairing.
    """

    def __init__(
        self,
        text_dataset: GoEmotionsDataset,
        audio_dataset: Dataset,
        seed: int = 42,
    ) -> None:
        self.text_dataset = text_dataset
        self.audio_dataset = audio_dataset

        # Group audio samples by cluster
        random.seed(seed)
        self._audio_by_cluster: Dict[int, List[int]] = {i: [] for i in range(4)}
        for i in range(len(audio_dataset)):
            lbl = int(audio_dataset[i]["label"].item())
            self._audio_by_cluster[lbl].append(i)

    def __len__(self) -> int:
        return len(self.text_dataset)

    def __getitem__(self, idx: int) -> dict:
        text_sample = self.text_dataset[idx]
        cluster = int(text_sample["label"].item())

        candidates = self._audio_by_cluster.get(cluster, [])
        if not candidates:
            # Fallback: random audio sample from any cluster
            audio_idx = random.randint(0, len(self.audio_dataset) - 1)
        else:
            audio_idx = random.choice(candidates)

        audio_sample = self.audio_dataset[audio_idx]

        return {
            "input_ids": text_sample["input_ids"],
            "attention_mask": text_sample["attention_mask"],
            "input_values": audio_sample["input_values"],
            "label": text_sample["label"],
        }


# ── DataLoader Factory ──────────────────────────────────────────────────────────

def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    use_weighted_sampler: bool = False,
) -> DataLoader:
    """
    Build a DataLoader with optional class-balanced weighted sampler.

    Args:
        dataset: Any torch Dataset.
        batch_size: Batch size.
        shuffle: Shuffle data (ignored when use_weighted_sampler=True).
        num_workers: Parallel data loading workers.
        use_weighted_sampler: Enable weighted random oversampling for
                              class imbalance mitigation (Section 3.8).
    """
    sampler = None
    if use_weighted_sampler and hasattr(dataset, "get_class_weights"):
        class_weights = dataset.get_class_weights()
        sample_weights = [
            float(class_weights[int(dataset[i]["label"].item())])
            for i in range(len(dataset))
        ]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
