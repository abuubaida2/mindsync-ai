"""
MindSync FastAPI Backend
Exposes the MindSync model as a REST API for the Expo mobile app.

Endpoints:
    POST /predict       — text + optional audio → emotion + incongruence
    GET  /health        — health check
    GET  /clusters      — list emotion cluster labels
"""

import sys
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predict import MindSyncPredictor
from src.inference.text_predictor import TextOnlyPredictor
from src.data.emotion_clusters import CLUSTER_LABELS

# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MindSync API",
    description="Multimodal AI for mental health monitoring",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load predictor once at startup ───────────────────────────────────────────
predictor = None
PREDICTOR_MODE = "uninitialized"


@app.on_event("startup")
def load_model():
    global predictor, PREDICTOR_MODE
    project_root = Path(__file__).parent.parent

    # 1. Full multimodal checkpoint?
    mm_ckpt = os.getenv("MINDSYNC_CHECKPOINT")
    if mm_ckpt is None:
        default_mm = project_root / "checkpoints" / "multimodal" / "best_multimodal.pt"
        if default_mm.exists():
            mm_ckpt = str(default_mm)

    if mm_ckpt and Path(mm_ckpt).exists():
        print(f"Loading full MindSync multimodal model from {mm_ckpt} ...")
        predictor = MindSyncPredictor(checkpoint_path=mm_ckpt)
        PREDICTOR_MODE = "multimodal"
        print("Multimodal model ready.")
        return

    # 2. Text-only checkpoint? (prefer the verified roberta-base checkpoint)
    text_ckpt = os.getenv("MINDSYNC_TEXT_CHECKPOINT")
    text_model_name = "roberta-base"
    if text_ckpt is None:
        candidates = [
            (project_root / "checkpoints" / "text_only_local" / "best_text_model.pt", "roberta-base"),
            (project_root / "checkpoints" / "text" / "best_text_model.pt", "roberta-large"),
        ]
        for path, mname in candidates:
            if path.exists():
                text_ckpt = str(path)
                text_model_name = mname
                break

    if text_ckpt and Path(text_ckpt).exists():
        print(f"Loading text-only model ({text_model_name}) from {text_ckpt} ...")
        predictor = TextOnlyPredictor(checkpoint_path=text_ckpt, text_model_name=text_model_name)
        PREDICTOR_MODE = f"text-only ({text_model_name})"
        print("Text-only model ready.")
        return

    # 3. Fallback — full multimodal with random weights (testing only)
    print("WARNING: no checkpoint found — loading MindSync with random weights (testing only).")
    predictor = MindSyncPredictor(checkpoint_path=None)
    PREDICTOR_MODE = "random-weights"
    print("Random-weights model ready.")


# ── Schemas ──────────────────────────────────────────────────────────────────
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


# ── Gemini narrative generator ───────────────────────────────────────────────
NARRATIVE_SYSTEM_PROMPT = (
    "You are an empathetic mental-health wellness companion embedded in a research "
    "prototype called MindSync. The app classifies a user's emotional state from "
    "their text and voice into one of four clusters: Distress, Resilience, "
    "Aggression, or Ambiguity, and computes an incongruence score (0–1) measuring "
    "mismatch between what the user said and how they said it.\n\n"
    "Given a single analysis result, write a short (2–3 sentence) supportive note:\n"
    "1. Reflect what the model detected, in warm plain language.\n"
    "2. Offer ONE small, gentle, evidence-aligned coping suggestion "
    "(grounding, breathing, journaling, reaching out, etc.).\n"
    "3. If clinical_alert is true OR incongruence > 0.5, gently invite reflection "
    "on whether they want to talk to someone they trust.\n\n"
    "RULES — never break:\n"
    "• Never diagnose, never claim certainty about the user's state.\n"
    "• Never replace professional help. If you mention distress, add that a qualified "
    "professional is the right next step.\n"
    "• No emojis. No bullet points. No headings. Plain prose only.\n"
    "• Maximum 60 words."
)


def _build_user_prompt(req: NarrativeRequest) -> str:
    lines = [
        f"Predicted emotion cluster: {req.predicted_emotion}",
        f"Incongruence score: {req.incongruence_score:.2f} (threshold 0.5)",
        f"Clinical alert raised: {req.clinical_alert}",
    ]
    if req.text_emotion:
        lines.append(f"Text-stream emotion: {req.text_emotion}")
    if req.audio_emotion:
        lines.append(f"Voice-stream emotion: {req.audio_emotion}")
    if req.user_text:
        lines.append(f'User said: "{req.user_text}"')
    return "\n".join(lines)


def generate_narrative(req: NarrativeRequest) -> tuple[str, str]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or "REPLACE" in api_key:
        raise HTTPException(503, "GROQ_API_KEY not configured on the server.")

    from groq import Groq

    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    client = Groq(api_key=api_key)
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(req)},
            ],
            temperature=0.6,
            max_tokens=180,
        )
    except Exception as e:
        raise HTTPException(502, f"Groq call failed: {e}")

    text = (completion.choices[0].message.content or "").strip()
    if not text:
        raise HTTPException(502, "Groq returned an empty response.")
    return text, model_name


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None, "mode": PREDICTOR_MODE}


@app.get("/clusters")
def clusters():
    return {"clusters": CLUSTER_LABELS}


@app.post("/predict/text", response_model=PredictResponse)
def predict_text(req: TextOnlyRequest):
    """Predict from text only (no audio)."""
    if not predictor:
        raise HTTPException(503, "Model not loaded")
    if not req.text.strip():
        raise HTTPException(400, "text must not be empty")
    result = predictor.predict(text=req.text)
    return result


@app.post("/predict/audio", response_model=PredictResponse)
async def predict_audio(audio: UploadFile = File(...)):
    """Predict from audio only (no text)."""
    if not predictor:
        raise HTTPException(503, "Model not loaded")

    tmp_file = None
    try:
        contents = await audio.read()
        suffix = Path(audio.filename).suffix if audio.filename else ".wav"
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_file.write(contents)
        tmp_file.close()
        result = predictor.predict(text="", audio_path=tmp_file.name)
    except Exception as e:
        raise HTTPException(422, f"Audio prediction failed: {e}")
    finally:
        if tmp_file and os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
    return result


@app.post("/predict", response_model=PredictResponse)
async def predict_multimodal(
    text: str = Form(...),
    audio: Optional[UploadFile] = File(None),
):
    """Predict from text + optional audio file upload."""
    if not predictor:
        raise HTTPException(503, "Model not loaded")
    if not text.strip():
        raise HTTPException(400, "text must not be empty")

    audio_path = None
    tmp_file = None

    if audio is not None:
        try:
            contents = await audio.read()
            suffix = Path(audio.filename).suffix if audio.filename else ".wav"
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp_file.write(contents)
            tmp_file.close()
            audio_path = tmp_file.name
        except Exception as e:
            raise HTTPException(400, f"Audio processing error: {e}")

    try:
        result = predictor.predict(text=text, audio_path=audio_path)
    except Exception as e:
        raise HTTPException(422, f"Prediction failed: {e}")
    finally:
        if tmp_file and os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)

    return result


@app.post("/narrative", response_model=NarrativeResponse)
def narrative(req: NarrativeRequest):
    """Generate an empathetic clinical narrative for a prediction result."""
    text, model_name = generate_narrative(req)
    return NarrativeResponse(narrative=text, model=model_name)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
