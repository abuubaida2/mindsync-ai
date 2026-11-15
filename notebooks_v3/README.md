# MindSync — experiment notebooks (paper version)

Notebooks and result files behind the paper. Every reported number comes from these runs on real data.
All runs used a single NVIDIA Tesla T4 on Kaggle; each notebook installs `transformers==4.45.2`,
`datasets==2.21.0`, `huggingface_hub==0.25.2`.

## Used in the paper

| Notebook | What it produces |
|---|---|
| `../mindsync_colab_stage1_text.ipynb` | Stage 1 text model (RoBERTa-Large, GoEmotions → 4 clusters), published run |
| `stage1_multiseed/` | Two further Stage 1 runs (seeds 43, 44) → `results/stage1_multiseed_results.json` |
| `stage2_v3_audio/` | Stage 2 v3 audio model (wav2vec2-Large-960h, layer-weighted padding-free pooling), seeds 42–44 on the speaker-independent split → `results/stage2_v3_results.json` and the checkpoint used downstream |
| `cv_all/` | Eight-fold speaker-independent cross-validation of the audio model (all 24 RAVDESS actors) and the RAVDESS same-utterance incongruence test (real transcript + its own clip) → `results/cv_results_all.json` |
| `adapted/` | Encoders fine-tuned on MELD (text: 3 seeds; audio: 1 seed), late fusion and a concatenation head; MUStARD with dialogue context, frozen vs adapted encoders → `results/results_adapted.json` |
| `real_pairs/` | MELD (real text + audio of the same utterance): zero-shot transfer and fusion trained on MELD; MUStARD (real sarcasm) incongruence test → `results/results_real_pairs.json`. Needs the Stage 2 v3 checkpoint attached as an input. |

Datasets: GoEmotions, RAVDESS, MELD (`zaber666/meld-dataset` on Kaggle), MUStARD labels from the official repository and
MUStARD++ utterance audio (`kirti1210/mustard-multimodal-research` on Kaggle). Three MUStARD clips whose audio does not match
the official transcript are excluded (`1_5617`, `1_7089`, `2_235`).

`audio_model_v3.py` holds the audio model class pasted verbatim into the notebooks; `build_real.py` and `build_cv.py`
regenerate the notebooks (`KAGGLE_USER=<your username> python build_real.py`).

Live deployment check and latency: `results/latency_hf_space.json` (180 multimodal + 50 text-only requests to the Hugging Face Space).

## Not used in the paper

`stage3_v3_seqcmaf/`, `cv_a/`, `cv_b/` and the synthetic-pair parts of `cv_all/` evaluate fusion on artificially paired
GoEmotions × RAVDESS data. They were exploratory and are kept only for transparency; the paper reports real paired data (MELD)
instead. The same applies to `../mindsync_colab_stage3_fusion.ipynb` and `../mindsync_stage3_v2_leakfree.ipynb`.
