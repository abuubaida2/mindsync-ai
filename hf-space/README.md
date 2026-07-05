---
title: MindSync Backend
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# MindSync Backend (HF Space)

FastAPI backend for the MindSync mobile app — multimodal AI for mental-health
monitoring. This deployment runs the **text stream** (RoBERTa-base fine-tuned on
GoEmotions, mapped to four clinical clusters: Distress / Resilience / Aggression /
Ambiguity) plus a Groq-powered empathetic "AI Wellness Note" generator.

## Endpoints

- `GET  /health` — service + model status
- `GET  /clusters` — cluster label list
- `POST /predict/text` — `{"text": "..."}` → emotion prediction
- `POST /predict/audio` — multipart `audio` → text-only-style response (audio model not in this build)
- `POST /predict` — multipart `text` + optional `audio` → text-branch prediction
- `POST /narrative` — `{predicted_emotion, incongruence_score, clinical_alert, user_text}` → empathetic note (needs `GROQ_API_KEY` secret)

## Notes

Research prototype — **not a clinical diagnostic tool**. The trained checkpoint is
pulled from the model repo `Ubaida1/mindsync-text-model` at startup.
