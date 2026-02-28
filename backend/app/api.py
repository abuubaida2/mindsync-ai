"""
MindSync FastAPI Backend — for mobile/web clients.

Endpoints:
    GET  /health          — health check
    POST /predict         — multimodal emotion prediction

Usage:
    python app/api.py
    # or
    uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
"""

import argparse
import base64
import sys
import tempfile
import os
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.inference.predict import MindSyncPredictor

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MindSync API",
    description="Multimodal AI Framework for Mental Health Monitoring",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor (loaded once at startup)
predictor: Optional[MindSyncPredictor] = None


@app.on_event("startup")
def load_model():
    global predictor
    checkpoint = os.environ.get("MINDSYNC_CHECKPOINT", None)
    print("Loading MindSync predictor...")
    predictor = MindSyncPredictor(checkpoint_path=checkpoint)
    print("MindSync predictor ready.")


# ── Request / Response models ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str
    audio_base64: Optional[str] = None   # base64-encoded WAV/PCM audio bytes
    audio_sr: int = 16000               # sample rate if providing raw PCM float32


class EmotionProbabilities(BaseModel):
    Distress: float
    Resilience: float
    Aggression: float
    Ambiguity: float


class PredictResponse(BaseModel):
    predicted_emotion: str
    text_emotion: str
    audio_emotion: str
    fused_probabilities: EmotionProbabilities
    text_probabilities: EmotionProbabilities
    audio_probabilities: EmotionProbabilities
    incongruence_score: float
    clinical_alert: bool
    clinical_message: str


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text field is required")

    audio_path = None
    tmp_path = None

    try:
        if req.audio_base64:
            # Decode base64 → write temp WAV file
            audio_bytes = base64.b64decode(req.audio_base64)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(audio_bytes)
            tmp.close()
            tmp_path = tmp.name
            audio_path = tmp_path

        result = predictor.predict(text=req.text, audio_path=audio_path)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return PredictResponse(
        predicted_emotion=result["predicted_emotion"],
        text_emotion=result["text_emotion"],
        audio_emotion=result["audio_emotion"],
        fused_probabilities=EmotionProbabilities(**result["fused_probabilities"]),
        text_probabilities=EmotionProbabilities(**result["text_probabilities"]),
        audio_probabilities=EmotionProbabilities(**result["audio_probabilities"]),
        incongruence_score=result["incongruence_score"],
        clinical_alert=result["clinical_alert"],
        clinical_message=result["clinical_message"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MindSync API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    if args.checkpoint:
        os.environ["MINDSYNC_CHECKPOINT"] = args.checkpoint

    uvicorn.run("app.api:app", host=args.host, port=args.port, reload=args.reload)
