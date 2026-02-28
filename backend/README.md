# MindSync 🧠

**A Multimodal AI Framework for Mental Health Monitoring via Fine-Grained Emotion Recognition and Cross-Modal Incongruence Detection**

> ⚠️ **Disclaimer:** MindSync is a research prototype. It is **not a clinical diagnostic tool**. All high-incongruence flags must be reviewed by a qualified mental health professional. No claims of clinical efficacy are made.

---

## Overview

MindSync is a dual-stream multimodal AI framework that combines:

- **Text Stream** — RoBERTa-Large fine-tuned on GoEmotions (27 labels → 4 clinical clusters)
- **Audio Stream** — wav2vec 2.0 fine-tuned on RAVDESS and IEMOCAP
- **Cross-Modal Attention Fusion (CMAF)** — Bidirectional cross-attention integrating both streams
- **Incongruence Detection** — Jensen-Shannon Divergence-based module flagging cases where verbal content contradicts vocal affect (e.g., "I'm fine" with a trembling voice)

### Key Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Text-Only (RoBERTa-Large) | 85.0% | 0.83 |
| Audio-Only (wav2vec 2.0) | 78.0% | 0.75 |
| Early Fusion | 87.0% | 0.857 |
| **MindSync (CMAF)** | **90.0%** | **0.88** |

**Incongruence Detection:** 92% accuracy (Fleiss' κ = 0.81, n=520 human-annotated samples)

---

## Project Structure

```
mindsync/
├── src/
│   ├── data/
│   │   ├── emotion_clusters.py       # 27-label → 4-cluster mapping (Eq. 1)
│   │   ├── text_preprocessing.py     # RoBERTa tokenization pipeline
│   │   ├── audio_preprocessing.py    # wav2vec feature extraction (Eq. 2)
│   │   └── dataset.py               # GoEmotions, RAVDESS, IEMOCAP datasets
│   ├── models/
│   │   ├── text_model.py            # RoBERTa-Large encoder + head (Eqs. 3–4)
│   │   ├── audio_model.py           # wav2vec 2.0 encoder + head (Eq. 5)
│   │   ├── multimodal_fusion.py     # CMAF module (Eqs. 6–9)
│   │   ├── incongruence.py          # JSD incongruence detection (Eq. 10)
│   │   └── mindsync.py              # Full model + multi-task loss (Eq. 11)
│   ├── training/
│   │   ├── train.py                 # AdamW + linear warm-up training loop
│   │   └── evaluate.py              # Metrics: accuracy, macro F1, confusion matrix
│   ├── inference/
│   │   └── predict.py               # End-to-end inference pipeline
│   └── utils/
│       ├── seed.py                  # Reproducibility (seed=42)
│       └── visualization.py        # Confusion matrix, t-SNE, training curves
├── scripts/
│   ├── train_multimodal.py          # Train full MindSync CMAF model
│   ├── train_text.py                # Train text-only baseline
│   ├── train_audio.py               # Train audio-only baseline
│   └── generate_figures.py         # Generate all paper figures
├── configs/
│   ├── multimodal_config.yaml
│   ├── text_config.yaml
│   └── audio_config.yaml
├── app/
│   └── demo.py                      # Gradio interactive demo
├── tests/
│   ├── test_text_model.py
│   ├── test_fusion.py
│   └── test_mindsync.py
└── docs/
    └── figures/                     # Generated paper figures
```

---

## Quick Start

### 1. Environment Setup

```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate mindsync

# Or pip
pip install -r requirements.txt
pip install -e .
```

**Hardware:** NVIDIA GPU with ≥16GB VRAM recommended (paper: A100 40GB)  
**Software:** Python 3.9.13 · PyTorch 2.0.1 + CUDA 11.8 · HuggingFace Transformers 4.30.2

### 2. Generate Paper Figures (no data required)

```bash
python scripts/generate_figures.py
```

Produces all 5 paper figures in `docs/figures/`.

### 3. Run the Demo

```bash
python app/demo.py
# or with a checkpoint:
python app/demo.py --checkpoint checkpoints/mindsync_cmaf/best_model.pt
# with public URL:
python app/demo.py --share
```

---

## Training

### Full Multimodal Model (CMAF)

```bash
# Requires GoEmotions (auto-downloaded from HuggingFace) 
# and RAVDESS audio files in data/ravdess/
python scripts/train_multimodal.py --config configs/multimodal_config.yaml
```

### Text-Only Baseline

```bash
python scripts/train_text.py --config configs/text_config.yaml
```

### Audio-Only Baseline

```bash
python scripts/train_audio.py --config configs/audio_config.yaml
```

### Debug Mode (small dataset, fast)

```bash
python scripts/train_multimodal.py --config configs/multimodal_config.yaml --debug
```

---

## Data Setup

### GoEmotions (Text — Auto-downloaded)
```python
from datasets import load_dataset
ds = load_dataset("google-research-datasets/go_emotions", "simplified")
```

### RAVDESS (Audio)
Download from [Zenodo](https://zenodo.org/record/1188976). Extract to `data/ravdess/`:
```
data/ravdess/
    Actor_01/
        03-01-01-01-01-01-01.wav
        ...
    Actor_24/
        ...
```

### IEMOCAP (Audio — Institutional)
Request from [USC SAIL Lab](https://sail.usc.edu/iemocap/). 
Place in `data/iemocap/` with a `labels.csv` (columns: `wav_path`, `emotion`).

---

## Architecture Details

### Emotion Cluster Mapping (Equation 1)

| Cluster | GoEmotions Labels |
|---|---|
| **Distress** | grief, nervousness, fear, sadness, remorse |
| **Resilience** | joy, admiration, excitement, relief, amusement, approval, gratitude, love, optimism, pride |
| **Aggression** | anger, annoyance, disgust, disapproval, embarrassment |
| **Ambiguity** | confusion, disappointment, surprise, caring, neutral |

### Multi-task Loss (Equation 11)

$$\mathcal{L}_{total} = \mathcal{L}_{CE}(\hat{y}_{final}, y) + \lambda_1 \mathcal{L}_{CE}(\hat{y}_{text}, y) + \lambda_2 \mathcal{L}_{CE}(\hat{y}_{audio}, y)$$

where $\lambda_1 = \lambda_2 = 0.3$.

### JSD Incongruence Detection (Equation 10)

$$\delta = \text{JSD}(P_{text} \| P_{audio}) = \frac{1}{2}\text{KL}(P_{text} \| M) + \frac{1}{2}\text{KL}(P_{audio} \| M)$$

where $M = \frac{1}{2}(P_{text} + P_{audio})$. Threshold $\delta > 0.5$ triggers a clinical-risk alert.

---

## Inference Example

```python
from src.inference.predict import MindSyncPredictor

predictor = MindSyncPredictor(
    checkpoint_path="checkpoints/mindsync_cmaf/best_model.pt",
    incongruence_threshold=0.5,
)

result = predictor.predict(
    text="I'm fine, everything is great.",
    audio_path="audio_samples/anxious_voice.wav",
)

print(result["predicted_emotion"])       # "Distress"
print(result["incongruence_score"])       # 0.73
print(result["clinical_alert"])          # True
print(result["clinical_message"])
# ⚠ CLINICAL ALERT: Incongruence detected (δ=0.730).
# Text signals 'Resilience' but audio signals 'Distress'.
```

---

## Running Tests

```bash
# All tests (excluding slow HuggingFace download tests)
pytest -m "not slow"

# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html
```

---

## Reproducibility

| Item | Value |
|---|---|
| Seed | 42 (PyTorch, NumPy, Python random) |
| Python | 3.9.13 |
| PyTorch | 2.0.1 + CUDA 11.8 |
| HuggingFace Transformers | 4.30.2 |
| librosa | 0.10.1 |
| SciPy | 1.11.1 |
| Hardware | NVIDIA A100 40GB, 64GB RAM |
| Training Runtime | ~4.5 hours |

---

## Configuration

All hyperparameters are controlled via YAML files in `configs/`:

```yaml
# configs/multimodal_config.yaml (key settings)
model:
  text_model_name: "roberta-large"
  audio_model_name: "facebook/wav2vec2-large-960h"
  d_model: 512
  num_heads: 8
  lambda_text: 0.3      # Eq. 11 auxiliary weight
  lambda_audio: 0.3
incongruence:
  threshold: 0.5        # JSD threshold
training:
  text_lr: 2.0e-5       # RoBERTa differential LR
  audio_lr: 1.0e-4
  num_epochs: 10
seed: 42
```

---

## Limitations

1. **Synthetic cross-dataset alignment** — GoEmotions (text) and RAVDESS/IEMOCAP (audio) are probabilistically paired; no naturally co-occurring speech-text pairs
2. **Demographically narrow** — English-language, North American/European populations only
3. **Offline evaluation only** — Not tested under real-time streaming constraints
4. **Small pilot study** — n=30, 2 weeks; insufficient for clinical evidence

---

## Ethical Considerations

- Audio-based emotion analysis constitutes **sensitive biometric data** under GDPR Article 9
- Real-world deployment requires explicit informed consent and E2E-encrypted processing
- Outputs must be framed as **probabilistic indicators**, not clinical diagnoses
- Demographic parity audits required before any clinical deployment
- IRB-approved prospective studies required before clinical use

---

## Citation

```bibtex
@article{mindsync2024,
  title     = {MindSync: A Multimodal AI Framework for Mental Health Monitoring 
               via Fine-Grained Emotion Recognition and Cross-Modal Incongruence Detection},
  author    = {[Author Name]},
  journal   = {[Journal Name]},
  year      = {2024},
  note      = {Code: https://github.com/[author]/mindsync-mental-health}
}
```

---

## License

MIT License — See [LICENSE](LICENSE) for details.

---

## References

Key references:
- **GoEmotions** — Demszky et al., ACL 2020
- **RoBERTa** — Liu et al., arXiv 2019
- **wav2vec 2.0** — Baevski et al., NeurIPS 2020
- **RAVDESS** — Livingstone & Russo, PLoS ONE 2018
- **MentalLLaMA** — Yang et al., arXiv 2024
- **MulT** — Tsai et al., ACL 2019

Full reference list in paper Section 6.
