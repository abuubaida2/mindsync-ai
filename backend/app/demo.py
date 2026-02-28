"""
MindSync Interactive Demo — Gradio Web Interface.

Provides real-time multimodal emotion analysis with incongruence detection.

Usage:
    python app/demo.py
    python app/demo.py --checkpoint checkpoints/mindsync_cmaf/best_model.pt
    python app/demo.py --share   # generate public URL
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import numpy as np
import torch
import soundfile as sf
import tempfile
import os

from src.inference.predict import MindSyncPredictor
from src.data.emotion_clusters import CLUSTER_LABELS
from src.utils.visualization import PALETTE as CLUSTER_COLORS


# ── Colour Mapping ───────────────────────────────────────────────────────────────
GRADIO_COLORS = {
    "Distress": "#E74C3C",
    "Resilience": "#2ECC71",
    "Aggression": "#E67E22",
    "Ambiguity": "#9B59B6",
}


def build_predictor(checkpoint_path=None):
    """Instantiate predictor (loads checkpoint if provided)."""
    return MindSyncPredictor(
        checkpoint_path=checkpoint_path,
        device="auto",
        incongruence_threshold=0.5,
    )


def analyze(text: str, audio, predictor: MindSyncPredictor):
    """
    Gradio prediction callback.

    Args:
        text: User-entered text.
        audio: Gradio audio tuple (sr, ndarray) or path string.
        predictor: MindSyncPredictor instance.

    Returns:
        Tuple of Gradio UI component outputs.
    """
    if not text or text.strip() == "":
        return "Please enter some text.", "", "", "", "", "", ""

    audio_path = None
    if audio is not None:
        try:
            if isinstance(audio, tuple):
                sr, wave = audio
                wave = wave.astype(np.float32)
                if wave.ndim > 1:
                    wave = wave.mean(axis=1)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, wave, sr)
                    audio_path = tmp.name
            else:
                audio_path = audio
        except Exception as e:
            audio_path = None

    result = predictor.predict(text=text, audio_path=audio_path)

    if audio_path and audio_path.startswith(tempfile.gettempdir()):
        try:
            os.unlink(audio_path)
        except Exception:
            pass

    # ── Format outputs ───────────────────────────────────────────────────────
    pred_emotion = result["predicted_emotion"]
    incong_score = result["incongruence_score"]
    clinical_alert = result["clinical_alert"]

    alert_html = (
        f"<div style='background:#E74C3C22;border:2px solid #E74C3C;border-radius:8px;"
        f"padding:12px;color:#E74C3C;font-weight:bold;'>"
        f"⚠ CLINICAL ALERT (δ={incong_score:.3f}): "
        f"Text signals <em>{result['text_emotion']}</em> but audio signals <em>{result['audio_emotion']}</em>. "
        f"Please consult a qualified mental health professional.</div>"
    ) if clinical_alert else (
        f"<div style='background:#2ECC7122;border:2px solid #2ECC71;border-radius:8px;"
        f"padding:12px;color:#2ECC71;'>"
        f"✓ Congruent (δ={incong_score:.3f}) — Both modalities indicate {pred_emotion}</div>"
    )

    fused_bar = make_prob_bar(result["fused_probabilities"])
    text_bar = make_prob_bar(result["text_probabilities"])
    audio_bar = make_prob_bar(result["audio_probabilities"])
    incong_gauge = make_score_gauge(incong_score)

    return (
        f"**{pred_emotion}**",
        alert_html,
        fused_bar,
        text_bar,
        audio_bar,
        incong_gauge,
        result["clinical_message"],
    )


def make_prob_bar(probs: dict) -> str:
    """Build HTML probability bar visualization."""
    html = "<div style='font-size:13px;'>"
    for label, prob in sorted(probs.items(), key=lambda x: -x[1]):
        pct = prob * 100
        color = GRADIO_COLORS.get(label, "#888")
        html += (
            f"<div style='margin-bottom:5px;'>"
            f"<span style='width:90px;display:inline-block;font-weight:bold;color:{color}'>{label}</span>"
            f"<div style='display:inline-block;width:{pct:.1f}%;background:{color};"
            f"height:16px;border-radius:3px;vertical-align:middle;margin:0 6px'/>"
            f"<span style='color:#aaa'>{pct:.1f}%</span></div>"
        )
    return html + "</div>"


def make_score_gauge(score: float) -> str:
    """Build HTML gauge for incongruence score."""
    pct = score * 100
    color = "#E74C3C" if score > 0.5 else ("#F39C12" if score > 0.3 else "#2ECC71")
    return (
        f"<div style='font-size:13px;padding:8px;'>"
        f"<b>JSD Incongruence Score (δ): {score:.4f}</b><br/>"
        f"<div style='background:#333;border-radius:6px;overflow:hidden;height:22px;margin-top:8px;'>"
        f"<div style='width:{pct:.1f}%;background:{color};height:22px;border-radius:6px;'/>"
        f"</div>"
        f"<span style='color:#aaa;font-size:11px;'>Threshold: 0.5 | "
        f"{'⚠ INCONGRUENT' if score > 0.5 else '✓ CONGRUENT'}</span></div>"
    )


EXAMPLE_INPUTS = [
    ["I'm fine, everything is great today!", None],
    ["I honestly don't even care anymore.", None],
    ["This makes me so incredibly angry!", None],
    ["I'm not sure how I feel about this...", None],
]

CSS = """
.gradio-container { max-width: 1100px; margin: auto; }
h1 { text-align: center; color: #4A90D9; }
"""


def build_interface(predictor: MindSyncPredictor) -> gr.Blocks:
    with gr.Blocks(title="MindSync — Mental Health AI Demo") as demo:
        gr.Markdown(
            "# 🧠 MindSync\n"
            "**Multimodal AI Framework for Mental Health Monitoring**  \n"
            "Fine-Grained Emotion Recognition + Cross-Modal Incongruence Detection  \n"
            "> ⚠ This is a research prototype. Not a clinical diagnostic tool.  "
            "All high-incongruence alerts must be reviewed by a qualified professional."
        )

        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="📝 Enter text (what the user said)",
                    placeholder="e.g. I'm fine, don't worry about me...",
                    lines=4,
                )
                audio_input = gr.Audio(
                    label="🎙 Upload/Record audio (how they said it)",
                    type="numpy",
                )
                analyze_btn = gr.Button("🔍 Analyze", variant="primary")

            with gr.Column(scale=1):
                pred_output = gr.Markdown(label="Predicted Emotion Cluster")
                alert_output = gr.HTML(label="Clinical Alert")
                score_output = gr.HTML(label="Incongruence Score")

        with gr.Row():
            fused_output = gr.HTML(label="Fused Probabilities")
            text_prob_output = gr.HTML(label="Text-Stream Probabilities")
            audio_prob_output = gr.HTML(label="Audio-Stream Probabilities")

        message_output = gr.Textbox(label="Clinical Message", interactive=False)

        gr.Examples(
            examples=EXAMPLE_INPUTS,
            inputs=[text_input, audio_input],
            label="Example Inputs",
        )

        analyze_btn.click(
            fn=lambda t, a: analyze(t, a, predictor),
            inputs=[text_input, audio_input],
            outputs=[
                pred_output, alert_output, fused_output,
                text_prob_output, audio_prob_output, score_output, message_output,
            ],
        )

        gr.Markdown(
            "---\n"
            "**References:** MindSync paper | GoEmotions | RAVDESS | IEMOCAP | wav2vec 2.0 | RoBERTa-Large"
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="MindSync Gradio Demo")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print("Loading MindSync predictor...")
    predictor = build_predictor(args.checkpoint)

    demo = build_interface(predictor)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
        css=CSS,
    )


if __name__ == "__main__":
    main()
