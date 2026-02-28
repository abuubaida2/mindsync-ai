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

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predict import MindSyncPredictor
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
predictor: Optional[MindSyncPredictor] = None


@app.on_event("startup")
def load_model():
    global predictor
    checkpoint = os.getenv("MINDSYNC_CHECKPOINT", None)
    print("Loading MindSync model...")
    predictor = MindSyncPredictor(checkpoint_path=checkpoint)
    print("Model ready.")


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


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
