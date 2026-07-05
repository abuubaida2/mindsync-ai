"""
Text preprocessing pipeline for GoEmotions dataset.
Implements Section 3.2.1 of MindSync paper.
"""

import re
from typing import List, Optional

from transformers import RobertaTokenizerFast

from src.data.emotion_clusters import map_go_emotions_to_idx


def clean_text(text: str) -> str:
    """Pass-through: training tokenised raw text with no preprocessing,
    so inference must match exactly. RoBERTa is case-sensitive BPE —
    lowercasing here would produce tokens the model never saw during training,
    causing it to default to the majority class (Ambiguity)."""
    return text


class TextPreprocessor:
    """
    End-to-end text preprocessing pipeline.

    Steps:
        1. Clean raw text
        2. Tokenize with RoBERTa-Large (max 128 tokens)
        3. Return input_ids, attention_mask, and label index

    Args:
        model_name: HuggingFace model name for tokenizer.
        max_length: Maximum token length (default 128 per paper).
    """

    def __init__(
        self,
        model_name: str = "roberta-large",
        max_length: int = 128,
    ) -> None:
        self.max_length = max_length
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

    def __call__(self, text: str, label: Optional[str] = None):
        """
        Preprocess a single text sample.

        Returns dict with input_ids, attention_mask, and optionally label.
        """
        cleaned = clean_text(text)
        encoding = self.tokenizer(
            cleaned,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if label is not None:
            result["label"] = map_go_emotions_to_idx(label)
        return result

    def batch_encode(self, texts: List[str]) -> dict:
        """Batch tokenize a list of texts."""
        cleaned = [clean_text(t) for t in texts]
        return self.tokenizer(
            cleaned,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
