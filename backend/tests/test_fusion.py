"""Tests for CMAF fusion and MindSync full model."""

import pytest
import torch
import numpy as np

from src.models.multimodal_fusion import CMAFModule, ProjectionLayer
from src.models.incongruence import IncongruenceDetector, jsd_torch, cohens_d, mcnemar_test


# ── CMAF Tests ───────────────────────────────────────────────────────────────────

class TestCMAFModule:
    @pytest.fixture
    def cmaf(self):
        return CMAFModule(
            text_dim=64,
            audio_dim=48,
            d_model=32,
            num_heads=4,
            num_layers=1,
            num_classes=4,
            dropout=0.0,
        )

    def test_output_keys(self, cmaf):
        B = 3
        e_text = torch.randn(B, 64)
        e_audio = torch.randn(B, 48)
        with torch.no_grad():
            out = cmaf(e_text, e_audio)
        assert "logits" in out
        assert "embedding" in out
        assert "attn_weights" in out

    def test_logits_shape(self, cmaf):
        B = 4
        e_text = torch.randn(B, 64)
        e_audio = torch.randn(B, 48)
        with torch.no_grad():
            out = cmaf(e_text, e_audio)
        assert out["logits"].shape == (B, 4)

    def test_embedding_shape(self, cmaf):
        B = 2
        e_text = torch.randn(B, 64)
        e_audio = torch.randn(B, 48)
        with torch.no_grad():
            out = cmaf(e_text, e_audio)
        assert out["embedding"].shape == (B, 32)

    def test_different_input_dims(self):
        """Test that projection correctly handles different text/audio dims."""
        cmaf = CMAFModule(text_dim=1024, audio_dim=768, d_model=512, num_heads=8,
                          num_layers=2, num_classes=4)
        B = 2
        e_text = torch.randn(B, 1024)
        e_audio = torch.randn(B, 768)
        with torch.no_grad():
            out = cmaf(e_text, e_audio)
        assert out["logits"].shape == (B, 4)


# ── Incongruence Detection Tests ─────────────────────────────────────────────────

class TestIncongruenceDetector:
    @pytest.fixture
    def detector(self):
        return IncongruenceDetector(threshold=0.5, temperature=1.0)

    def test_congruent_samples_low_score(self, detector):
        """Identical distributions should have JSD = 0."""
        B, C = 4, 4
        logits = torch.randn(B, C)
        with torch.no_grad():
            out = detector(logits, logits)
        # JSD(P, P) = 0
        assert torch.allclose(out["scores"], torch.zeros(B), atol=1e-5)
        assert not out["flags"].any()

    def test_incongruent_samples_high_score(self, detector):
        """Opposite one-hot distributions should have JSD ≈ 1."""
        B, C = 2, 4
        # Cluster 0 text, cluster 3 audio
        text_logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]] * B)
        audio_logits = torch.tensor([[-10.0, -10.0, -10.0, 10.0]] * B)
        with torch.no_grad():
            out = detector(text_logits, audio_logits)
        assert (out["scores"] > 0.5).all()
        assert out["flags"].all()

    def test_output_shapes(self, detector):
        B, C = 8, 4
        text = torch.randn(B, C)
        audio = torch.randn(B, C)
        with torch.no_grad():
            out = detector(text, audio)
        assert out["scores"].shape == (B,)
        assert out["flags"].shape == (B,)
        assert out["p_text"].shape == (B, C)
        assert out["p_audio"].shape == (B, C)

    def test_scores_in_range(self, detector):
        B, C = 16, 4
        text = torch.randn(B, C)
        audio = torch.randn(B, C)
        with torch.no_grad():
            out = detector(text, audio)
        assert (out["scores"] >= 0.0).all()
        assert (out["scores"] <= 1.0).all()

    def test_predict_single_returns_dict(self, detector):
        text_logits = torch.tensor([2.0, -1.0, -1.0, -1.0])
        audio_logits = torch.tensor([-1.0, -1.0, -1.0, 2.0])
        result = detector.predict_single(text_logits, audio_logits)
        assert "incongruence_score" in result
        assert "is_incongruent" in result
        assert "clinical_alert" in result
        assert "dominant_text_emotion" in result
        assert "dominant_audio_emotion" in result


# ── JSD Function Tests ───────────────────────────────────────────────────────────

class TestJSDFunction:
    def test_self_divergence_is_zero(self):
        p = torch.softmax(torch.randn(5, 4), dim=-1)
        scores = jsd_torch(p, p)
        assert torch.allclose(scores, torch.zeros(5), atol=1e-5)

    def test_symmetry(self):
        p = torch.softmax(torch.randn(10, 4), dim=-1)
        q = torch.softmax(torch.randn(10, 4), dim=-1)
        assert torch.allclose(jsd_torch(p, q), jsd_torch(q, p), atol=1e-5)

    def test_output_in_unit_interval(self):
        for _ in range(20):
            p = torch.softmax(torch.randn(8, 4), dim=-1)
            q = torch.softmax(torch.randn(8, 4), dim=-1)
            scores = jsd_torch(p, q)
            assert (scores >= 0.0).all() and (scores <= 1.0).all()


# ── Statistical Tests ────────────────────────────────────────────────────────────

class TestStatisticalTests:
    def test_cohens_d_large_effect(self):
        """MindSync vs MulT F1 gap → large effect (paper reports d=0.94)."""
        a = np.array([0.88, 0.89, 0.87, 0.88, 0.90])
        b = np.array([0.79, 0.80, 0.78, 0.79, 0.80])
        d = cohens_d(a, b)
        assert d > 0.8  # large effect

    def test_cohens_d_zero_for_equal(self):
        a = np.array([0.80, 0.80, 0.80])
        b = np.array([0.80, 0.80, 0.80])
        assert cohens_d(a, b) == 0.0
