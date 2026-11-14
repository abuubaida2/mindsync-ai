# MindSync — experiment notebooks (paper version)

Notebooks and results behind the leakage-free evaluation reported in the paper.
All runs used a single NVIDIA Tesla T4 (Kaggle); each notebook installs `transformers==4.45.2`,
`datasets==2.21.0`, `huggingface_hub==0.25.2` and downloads RAVDESS (Zenodo) and GoEmotions (Hugging Face).

| Order | Notebook | What it produces |
|---|---|---|
| 1 | `../mindsync_colab_stage1_text.ipynb` | Stage 1 text model (RoBERTa-Large, GoEmotions → 4 clusters), published run |
| 1b | `stage1_multiseed/mindsync-stage1-multiseed.ipynb` | Two further Stage 1 runs (seeds 43, 44) → `results/stage1_multiseed_results.json` |
| 2 | `stage2_v3_audio/mindsync-stage2-v3-audio.ipynb` | Stage 2 v3 audio model (wav2vec2-Large-960h, layer-weighted padding-free pooling), seeds 42–44, speaker-independent split → `results/stage2_v3_results.json` and the checkpoint used downstream |
| 3 | `stage3_v3_seqcmaf/mindsync-stage3-v3-seqcmaf.ipynb` | Leakage-free fusion evaluation (SeqCMAF, pooled CMAF, baselines, cluster bootstrap, ECE, JSD, attention, errors) → `results/results_stage3_v3.json`. Needs the Stage 2 checkpoint attached as a kernel input. |
| 4 | `cv_a/`, `cv_b/` | 8-fold speaker cross-validation (folds 0–3 and 4–7): audio, fusion, rule-based and learned alerts, RAVDESS same-utterance test |

`audio_model_v3.py` holds the audio model class that is pasted verbatim into notebooks 2–4.
`build_stage3_v3.py` and `build_cv.py` regenerate notebooks 3 and 4.

Earlier versions kept for transparency: `../mindsync_colab_stage3_fusion.ipynb` (first fusion run, whose test text
overlapped with the Stage 1 training data and over-reported 94.83%) and `../mindsync_stage3_v2_leakfree.ipynb`
(first leakage-free re-evaluation with the v1 audio model).
