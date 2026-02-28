# MindSync AI 🧠

**A Multimodal AI Mobile App for Mental Health Monitoring**

> ⚠️ **Disclaimer:** MindSync is a research prototype. It is **not a clinical diagnostic tool**. All high-incongruence flags must be reviewed by a qualified mental health professional.

---

## Overview

MindSync is a full-stack AI-powered mental health monitoring application consisting of:

- **Mobile App** — React Native (Expo) frontend for voice & text input
- **AI Backend** — Python FastAPI server with a multimodal deep learning model

The AI combines **text analysis** (RoBERTa-Large) and **voice analysis** (wav2vec 2.0) to detect emotional states and cross-modal incongruence (e.g., "I'm fine" said with a trembling voice).

---

## Key Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Text-Only (RoBERTa-Large) | 85.0% | 0.83 |
| Audio-Only (wav2vec 2.0) | 78.0% | 0.75 |
| Early Fusion | 87.0% | 0.857 |
| **MindSync (CMAF)** | **90.0%** | **0.88** |

**Incongruence Detection:** 92% accuracy (Fleiss' κ = 0.81)

---

## Project Structure

```
mindsync-ai/
├── app/                        # Expo Router screens
│   ├── (tabs)/
│   │   ├── index.tsx           # Home screen
│   │   ├── analyze.tsx         # Text + Voice analysis
│   │   └── history.tsx         # Past analyses
│   ├── results.tsx             # Analysis results
│   └── _layout.tsx
├── src/                        # Frontend utilities
│   ├── api.ts                  # Backend API calls
│   ├── store.ts                # AsyncStorage (history)
│   ├── colors.ts               # Theme colors
│   └── types.ts                # TypeScript types
├── assets/                     # Images & fonts
├── backend/                    # Python AI Backend
│   ├── api/
│   │   └── main.py             # FastAPI app (endpoints)
│   ├── src/
│   │   ├── models/             # AI model definitions
│   │   ├── data/               # Data preprocessing
│   │   ├── training/           # Training scripts
│   │   └── inference/          # Prediction pipeline
│   ├── configs/                # YAML config files
│   ├── tests/                  # Unit tests
│   ├── requirements.txt        # Python dependencies
│   ├── start_all.ps1           # Windows start script
│   └── start_api_public.py     # API launcher
├── package.json                # JS dependencies
├── app.json                    # Expo config
└── tsconfig.json
```

---

## Emotion Clusters

| Cluster | Meaning | Examples |
|---|---|---|
| 😢 Distress | Negative/sad states | sadness, fear, grief |
| 😊 Resilience | Positive/coping states | joy, gratitude, relief |
| ⚡ Aggression | High arousal negative | anger, disgust, annoyance |
| ❓ Ambiguity | Mixed/unclear states | surprise, confusion |

---

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.10+
- Expo Go app on your phone

---

### 1. Run the Backend

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Start the API server
python start_api_public.py
```

The server runs on `http://0.0.0.0:8000`

API Endpoints:
- `GET  /health` — Server health check
- `POST /predict/text` — Text-only prediction
- `POST /predict/audio` — Audio-only prediction
- `POST /predict/multimodal` — Text + audio prediction

---

### 2. Configure the Mobile App

Update the API URL in [src/api.ts](src/api.ts) to your machine's local IP:

```ts
export const API_BASE = 'http://YOUR_LOCAL_IP:8000';
```

---

### 3. Run the Mobile App

```bash
# Install dependencies
npm install

# Start Expo
npx expo start --lan
```

Scan the QR code with **Expo Go** on your phone.

---

## Analysis Modes

| Mode | Description |
|---|---|
| 📝 Text Only | Type how you're feeling |
| 🎤 Voice Only | Record your voice |
| 🔀 Both | Text + Voice (most accurate) |

---

## Tech Stack

### Frontend
- React Native + Expo (SDK 54)
- Expo Router (file-based navigation)
- expo-av (audio recording)
- AsyncStorage (local history)
- React Native Reanimated

### Backend
- FastAPI (REST API)
- PyTorch (deep learning)
- RoBERTa-Large (text model)
- wav2vec 2.0 (audio model)
- Cross-Modal Attention Fusion (CMAF)
- Jensen-Shannon Divergence (incongruence detection)

---

## Developer

**Abu Ubaida**
Researcher & Developer — MindSync AI
📧 abuubaida.202202778@gcuf.edu.pk

---

## License

This project is for research and educational purposes only.
