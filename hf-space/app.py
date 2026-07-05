"""
MindSync Backend — Hugging Face Space (FastAPI, port 7860).

Full multimodal deployment: loads Stage 1 (text RoBERTa-Large), Stage 2 (audio
wav2vec2-Large), and Stage 3 (CMAF + JSD fusion) checkpoints from HF Model repos.

Endpoints:
    GET  /health
    GET  /clusters
    POST /predict/text       — text-only (Stage 1 classifier)
    POST /predict/audio      — voice-only (Stage 2 classifier)
    POST /predict            — multimodal (text + audio → CMAF fusion + JSD)
    POST /narrative          — Groq llama-3.3 reflection (requires GROQ_API_KEY)
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.inference.multimodal_predictor import MultimodalPredictor
from src.data.emotion_clusters import CLUSTER_LABELS

# ── Download trained checkpoints from HF Model repos ────────────────────────────
from huggingface_hub import hf_hub_download

TEXT_REPO   = os.getenv("MINDSYNC_TEXT_REPO",   "Ubaida1/mindsync-text-model")
AUDIO_REPO  = os.getenv("MINDSYNC_AUDIO_REPO",  "Ubaida1/mindsync-audio-model")
FUSION_REPO = os.getenv("MINDSYNC_FUSION_REPO", "Ubaida1/mindsync-fusion-model")

print(f"Downloading text checkpoint from {TEXT_REPO} ...")
TEXT_CKPT = hf_hub_download(repo_id=TEXT_REPO, filename="best_text_model.pt")
print(f"Downloading audio checkpoint from {AUDIO_REPO} ...")
AUDIO_CKPT = hf_hub_download(repo_id=AUDIO_REPO, filename="best_audio_model.pt")
print(f"Downloading fusion checkpoint from {FUSION_REPO} ...")
FUSION_CKPT = hf_hub_download(repo_id=FUSION_REPO, filename="best_fusion_model.pt")
print("Checkpoints ready.")

# ── App ─────────────────────────────────────────────────────────────────────────
app = FastAPI(title="MindSync API (HF Space, multimodal)", version="2.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

predictor: Optional[MultimodalPredictor] = None
PREDICTOR_MODE = "uninitialized"


@app.on_event("startup")
def load_model():
    global predictor, PREDICTOR_MODE
    text_name = os.getenv("MINDSYNC_TEXT_MODEL", "roberta-large")
    audio_name = os.getenv("MINDSYNC_AUDIO_MODEL", "facebook/wav2vec2-large-960h")
    print(f"Loading multimodal predictor ({text_name} + {audio_name} + CMAF) ...")
    predictor = MultimodalPredictor(
        text_ckpt=TEXT_CKPT,
        audio_ckpt=AUDIO_CKPT,
        fusion_ckpt=FUSION_CKPT,
        text_model_name=text_name,
        audio_model_name=audio_name,
    )
    PREDICTOR_MODE = f"multimodal ({text_name} + wav2vec2 + CMAF)"
    print("Model ready.")


# ── Schemas ─────────────────────────────────────────────────────────────────────
class TextOnlyRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    predicted_emotion: str
    text_emotion: str
    audio_emotion: str
    fused_probabilities: dict
    text_probabilities: dict
    audio_probabilities: dict
    incongruence_score: float
    clinical_alert: bool
    clinical_message: str


class NarrativeRequest(BaseModel):
    predicted_emotion: str
    text_emotion: Optional[str] = None
    audio_emotion: Optional[str] = None
    incongruence_score: float = 0.0
    clinical_alert: bool = False
    user_text: Optional[str] = None


class NarrativeResponse(BaseModel):
    narrative: str
    model: str


NARRATIVE_SYSTEM_PROMPT = (
    "You are an empathetic mental-health wellness companion embedded in a research "
    "prototype called MindSync. The app classifies a user's emotional state from "
    "text and/or voice into one of four clusters: Distress, Resilience, Aggression, "
    "or Ambiguity, and detects cross-modal incongruence (when text says one thing "
    "but voice tone suggests another).\n\n"
    "Given a single analysis result, write a short (2-3 sentence) supportive note:\n"
    "1. Reflect what the model detected, in warm plain language.\n"
    "2. Offer ONE small, gentle, evidence-aligned coping suggestion.\n"
    "3. If clinical_alert is true (incongruence detected), gently invite reflection "
    "on whether they want to talk to someone they trust.\n\n"
    "RULES: Never diagnose. Never replace professional help. No emojis, no bullets, "
    "no headings. Plain prose only. Max 60 words."
)


# ── Routes ──────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None, "mode": PREDICTOR_MODE}


@app.get("/clusters")
def clusters():
    return {"clusters": CLUSTER_LABELS}


@app.post("/predict/text", response_model=PredictResponse)
def predict_text(req: TextOnlyRequest):
    if not predictor:
        raise HTTPException(503, "Model not loaded")
    if not req.text.strip():
        raise HTTPException(400, "text must not be empty")
    return predictor.predict(text=req.text)


@app.post("/predict/audio", response_model=PredictResponse)
async def predict_audio(audio: UploadFile = File(...)):
    if not predictor:
        raise HTTPException(503, "Model not loaded")
    suffix = Path(audio.filename or "in.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        return predictor.predict(audio_path=tmp_path)
    except FileNotFoundError as e:
        # ffmpeg missing -> librosa cannot decode m4a/aac/caf
        raise HTTPException(415, f"Audio decoder unavailable: {e}")
    except RuntimeError as e:
        raise HTTPException(415, f"Could not decode audio ({suffix}): {e}")
    except Exception as e:
        raise HTTPException(500, f"Audio analysis failed: {type(e).__name__}: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/predict", response_model=PredictResponse)
async def predict_multimodal(
    text: str = Form(...),
    audio: Optional[UploadFile] = File(None),
):
    if not predictor:
        raise HTTPException(503, "Model not loaded")
    if not text.strip():
        raise HTTPException(400, "text must not be empty")
    if audio is None:
        return predictor.predict(text=text)
    suffix = Path(audio.filename or "in.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        return predictor.predict(text=text, audio_path=tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/narrative", response_model=NarrativeResponse)
def narrative(req: NarrativeRequest):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or "REPLACE" in api_key:
        raise HTTPException(503, "GROQ_API_KEY not configured on the server.")
    from groq import Groq

    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    user_lines = [
        f"Predicted emotion cluster: {req.predicted_emotion}",
        f"Incongruence score: {req.incongruence_score:.2f}",
        f"Clinical alert raised: {req.clinical_alert}",
    ]
    if req.text_emotion and req.text_emotion != req.predicted_emotion:
        user_lines.append(f"Text-only stream said: {req.text_emotion}")
    if req.audio_emotion and req.audio_emotion != req.predicted_emotion:
        user_lines.append(f"Voice-only stream said: {req.audio_emotion}")
    if req.user_text:
        user_lines.append(f'User said: "{req.user_text}"')

    client = Groq(api_key=api_key)
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
                {"role": "user", "content": "\n".join(user_lines)},
            ],
            temperature=0.6,
            max_tokens=180,
        )
    except Exception as e:
        raise HTTPException(502, f"Groq call failed: {e}")

    text = (completion.choices[0].message.content or "").strip()
    if not text:
        raise HTTPException(502, "Groq returned an empty response.")
    return NarrativeResponse(narrative=text, model=model_name)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860)
