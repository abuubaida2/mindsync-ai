"""Tests for full MindSync end-to-end model."""

import pytest
import torch

from src.models.mindsync import MindSync


@pytest.fixture(scope="module")
def mindsync_model():
    """Lightweight MindSync with small encoder stubs for CI speed."""
    return MindSync(
        text_model_name="distilroberta-base",
        audio_model_name="facebook/wav2vec2-base",
        d_model=64,
        num_heads=4,
        num_cmaf_layers=1,
        dropout=0.0,
    )


class TestMindSyncModel:
    def test_forward_output_keys(self, mindsync_model):
        B, L, T = 2, 32, 8000
        input_ids = torch.randint(0, 1000, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)
        input_values = torch.randn(B, T)
        labels = torch.randint(0, 4, (B,))

        with torch.no_grad():
            out = mindsync_model(input_ids, attention_mask, input_values, labels)

        expected_keys = [
            "logits_final", "logits_text", "logits_audio",
            "embedding_fused", "embedding_text", "embedding_audio",
            "incongruence_scores", "incongruence_flags",
            "loss", "loss_final", "loss_text", "loss_audio",
        ]
        for k in expected_keys:
            assert k in out, f"Missing key: {k}"

    def test_logits_shapes(self, mindsync_model):
        B, L, T = 3, 16, 4000
        input_ids = torch.randint(0, 500, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)
        input_values = torch.randn(B, T)

        with torch.no_grad():
            out = mindsync_model(input_ids, attention_mask, input_values)

        assert out["logits_final"].shape == (B, 4)
        assert out["logits_text"].shape == (B, 4)
        assert out["logits_audio"].shape == (B, 4)

    def test_incongruence_output_shapes(self, mindsync_model):
        B, L, T = 4, 16, 4000
        input_ids = torch.randint(0, 500, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)
        input_values = torch.randn(B, T)

        with torch.no_grad():
            out = mindsync_model(input_ids, attention_mask, input_values)

        assert out["incongruence_scores"].shape == (B,)
        assert out["incongruence_flags"].shape == (B,)

    def test_loss_is_scalar(self, mindsync_model):
        B, L, T = 2, 16, 4000
        input_ids = torch.randint(0, 500, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)
        input_values = torch.randn(B, T)
        labels = torch.randint(0, 4, (B,))

        with torch.no_grad():
            out = mindsync_model(input_ids, attention_mask, input_values, labels)

        assert out["loss"].dim() == 0
        assert out["loss"].item() > 0

    def test_multi_task_loss_formula(self, mindsync_model):
        """Verify L_total = L_final + 0.3*L_text + 0.3*L_audio (Eq. 11)."""
        B, L, T = 2, 16, 4000
        input_ids = torch.randint(0, 500, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)
        input_values = torch.randn(B, T)
        labels = torch.randint(0, 4, (B,))

        with torch.no_grad():
            out = mindsync_model(input_ids, attention_mask, input_values, labels)

        expected = (
            out["loss_final"]
            + mindsync_model.lambda_text * out["loss_text"]
            + mindsync_model.lambda_audio * out["loss_audio"]
        )
        assert torch.allclose(out["loss"], expected, atol=1e-5)

    def test_no_loss_without_labels(self, mindsync_model):
        B, L, T = 2, 16, 4000
        input_ids = torch.randint(0, 500, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)
        input_values = torch.randn(B, T)

        with torch.no_grad():
            out = mindsync_model(input_ids, attention_mask, input_values)

        assert "loss" not in out

    def test_count_parameters(self, mindsync_model):
        params = mindsync_model.count_parameters()
        assert "total" in params
        assert params["total"]["trainable"] > 0
        assert params["total"]["total"] >= params["total"]["trainable"]


class TestMindSyncPredict:
    def test_predict_returns_dict(self, mindsync_model):
        B, L, T = 1, 16, 4000
        input_ids = torch.randint(0, 500, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)
        input_values = torch.randn(B, T)

        result = mindsync_model.predict(input_ids, attention_mask, input_values)
        assert "predictions" in result
        assert "probabilities" in result
        assert "incongruence_scores" in result
        assert len(result["predictions"]) == B

    def test_predictions_are_valid_clusters(self, mindsync_model):
        from src.data.emotion_clusters import CLUSTER_LABELS
        B, L, T = 3, 16, 4000
        input_ids = torch.randint(0, 500, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)
        input_values = torch.randn(B, T)

        result = mindsync_model.predict(input_ids, attention_mask, input_values)
        for pred in result["predictions"]:
            assert pred in CLUSTER_LABELS
