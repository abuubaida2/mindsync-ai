"""
MindSync Mock API — runs without PyTorch/ML dependencies.
Use this for local development when the full model isn't available.
"""

import os
import random
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

app = FastAPI(
    title="MindSync Mock API",
    description="Mock backend for local development",
    version="1.0.0-mock",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLUSTER_LABELS = ["Distress", "Resilience", "Aggression", "Ambiguity"]


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
    "their text and voice into one of four clusters: Distress, Resilience, "
    "Aggression, or Ambiguity, and computes an incongruence score (0–1) measuring "
    "mismatch between what the user said and how they said it.\n\n"
    "Given a single analysis result, write a short (2–3 sentence) supportive note:\n"
    "1. Reflect what the model detected, in warm plain language.\n"
    "2. Offer ONE small, gentle, evidence-aligned coping suggestion.\n"
    "3. If clinical_alert is true OR incongruence > 0.5, gently invite reflection "
    "on whether they want to talk to someone they trust.\n\n"
    "RULES: Never diagnose. Never replace professional help. No emojis, no bullets, "
    "no headings. Plain prose only. Max 60 words."
)


def _make_mock_response(seed_text: str = "") -> dict:
    random.seed(hash(seed_text) % (2**31))
    probs = [random.random() for _ in range(4)]
    total = sum(probs)
    probs = [round(p / total, 4) for p in probs]
    fused = dict(zip(CLUSTER_LABELS, probs))

    text_probs = [random.random() for _ in range(4)]
    tt = sum(text_probs)
    text_probs_dict = dict(zip(CLUSTER_LABELS, [round(p / tt, 4) for p in text_probs]))

    audio_probs = [random.random() for _ in range(4)]
    at = sum(audio_probs)
    audio_probs_dict = dict(zip(CLUSTER_LABELS, [round(p / at, 4) for p in audio_probs]))

    predicted = max(fused, key=fused.get)
    text_em = max(text_probs_dict, key=text_probs_dict.get)
    audio_em = max(audio_probs_dict, key=audio_probs_dict.get)

    incongruence = round(random.uniform(0.0, 0.6), 4)
    alert = incongruence > 0.45

    return {
        "predicted_emotion": predicted,
        "text_emotion": text_em,
        "audio_emotion": audio_em,
        "fused_probabilities": fused,
        "text_probabilities": text_probs_dict,
        "audio_probabilities": audio_probs_dict,
        "incongruence_score": incongruence,
        "clinical_alert": alert,
        "clinical_message": "⚠️ Emotional incongruence detected." if alert else "No significant incongruence detected.",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/clusters")
def clusters():
    return {"clusters": CLUSTER_LABELS}


@app.post("/predict/text", response_model=PredictResponse)
def predict_text(req: TextOnlyRequest):
    return _make_mock_response(req.text)


@app.post("/predict/audio", response_model=PredictResponse)
async def predict_audio(audio: UploadFile = File(...)):
    return _make_mock_response("audio")


@app.post("/predict", response_model=PredictResponse)
async def predict_multimodal(
    text: str = Form(...),
    audio: Optional[UploadFile] = File(None),
):
    return _make_mock_response(text)


@app.post("/narrative", response_model=NarrativeResponse)
def narrative(req: NarrativeRequest):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or "REPLACE" in api_key:
        raise HTTPException(503, "GROQ_API_KEY not configured on the server.")

    from groq import Groq

    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    user_lines = [
        f"Predicted emotion cluster: {req.predicted_emotion}",
        f"Incongruence score: {req.incongruence_score:.2f} (threshold 0.5)",
        f"Clinical alert raised: {req.clinical_alert}",
    ]
    if req.text_emotion:
        user_lines.append(f"Text-stream emotion: {req.text_emotion}")
    if req.audio_emotion:
        user_lines.append(f"Voice-stream emotion: {req.audio_emotion}")
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
    import uvicorn
    uvicorn.run("mock_main:app", host="0.0.0.0", port=8000, reload=True)
