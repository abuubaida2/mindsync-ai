"""Tests for text preprocessing and text model."""

import pytest
import torch

from src.data.emotion_clusters import (
    map_go_emotions_label,
    map_go_emotions_to_idx,
    map_ravdess_label,
    map_iemocap_label,
    CLUSTER_TO_IDX,
    NUM_CLUSTERS,
)
from src.data.text_preprocessing import clean_text, TextPreprocessor


# ── Emotion Cluster Mapping Tests ────────────────────────────────────────────────

class TestEmotionClusters:
    def test_distress_labels(self):
        for label in ["grief", "fear", "sadness", "nervousness", "remorse"]:
            assert map_go_emotions_label(label) == "Distress"

    def test_resilience_labels(self):
        for label in ["joy", "admiration", "excitement", "relief"]:
            assert map_go_emotions_label(label) == "Resilience"

    def test_aggression_labels(self):
        for label in ["anger", "annoyance", "disgust"]:
            assert map_go_emotions_label(label) == "Aggression"

    def test_neutral_maps_to_ambiguity(self):
        assert map_go_emotions_label("neutral") == "Ambiguity"

    def test_unknown_maps_to_ambiguity(self):
        assert map_go_emotions_label("unknown_xyz") == "Ambiguity"

    def test_idx_range(self):
        for label in ["grief", "joy", "anger", "neutral"]:
            idx = map_go_emotions_to_idx(label)
            assert 0 <= idx < NUM_CLUSTERS

    def test_ravdess_fear_is_distress(self):
        cluster = map_ravdess_label(6)  # 6 = fearful
        assert cluster == "Distress"

    def test_ravdess_happy_is_resilience(self):
        cluster = map_ravdess_label(3)  # 3 = happy
        assert cluster == "Resilience"

    def test_iemocap_ang_is_aggression(self):
        assert map_iemocap_label("ang") == "Aggression"

    def test_iemocap_sad_is_distress(self):
        assert map_iemocap_label("sad") == "Distress"

    def test_iemocap_hap_is_resilience(self):
        assert map_iemocap_label("hap") == "Resilience"


# ── Text Preprocessing Tests ─────────────────────────────────────────────────────

class TestTextPreprocessing:
    def test_clean_removes_url(self):
        text = "Check this out http://example.com amazing"
        cleaned = clean_text(text)
        assert "http" not in cleaned
        assert "amazing" in cleaned

    def test_clean_lowercase(self):
        text = "THIS IS ALL UPPERCASE"
        cleaned = clean_text(text)
        assert cleaned == cleaned.lower()

    def test_clean_removes_html(self):
        text = "<b>Hello</b> World"
        cleaned = clean_text(text)
        assert "<b>" not in cleaned
        assert "Hello" in cleaned

    def test_clean_normalizes_whitespace(self):
        text = "too    many    spaces"
        cleaned = clean_text(text)
        assert "  " not in cleaned

    def test_clean_deduplicates_punctuation(self):
        text = "Really!!! So good???"
        cleaned = clean_text(text)
        assert "!!!" not in cleaned

    @pytest.mark.slow
    def test_preprocessor_output_shape(self):
        proc = TextPreprocessor(max_length=64)
        result = proc("I feel anxious and worried today.", label="nervousness")
        assert result["input_ids"].shape == (64,)
        assert result["attention_mask"].shape == (64,)
        assert "label" in result
        assert isinstance(result["label"], int)

    @pytest.mark.slow
    def test_preprocessor_label_mapping(self):
        proc = TextPreprocessor(max_length=32)
        result = proc("I am so sad", label="grief")
        assert result["label"] == CLUSTER_TO_IDX["Distress"]


# ── Text Model Tests ─────────────────────────────────────────────────────────────

class TestTextModel:
    @pytest.fixture
    def model(self):
        """Use a small BERT model for fast unit tests."""
        from src.models.text_model import MindSyncTextModel
        # Use distilroberta for speed in tests
        return MindSyncTextModel(
            model_name="distilroberta-base",
            num_classes=4,
        )

    def test_output_keys(self, model):
        B, L = 2, 32
        input_ids = torch.randint(0, 1000, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids, attention_mask)
        assert "embedding" in out
        assert "logits" in out

    def test_logits_shape(self, model):
        B, L = 3, 16
        input_ids = torch.randint(0, 1000, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids, attention_mask)
        assert out["logits"].shape == (B, 4)

    def test_embedding_shape(self, model):
        B, L = 2, 16
        input_ids = torch.randint(0, 1000, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids, attention_mask)
        # distilroberta has hidden_size=768, roberta-large=1024
        assert out["embedding"].dim() == 2
        assert out["embedding"].shape[0] == B
