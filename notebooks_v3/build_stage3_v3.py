"""Builds notebooks_v3/stage3_v3_seqcmaf/{notebook, kernel-metadata.json}.
Stage 3 v3: leakage-free evaluation with the Stage 2 v3 audio model and a
sequence-level CMAF (text tokens x audio frames) next to the pooled CMAF."""
import json, pathlib, ast, os
KAGGLE_USER = os.environ.get('KAGGLE_USER', 'YOUR_KAGGLE_USERNAME')   # set before building to push under your account

HERE = pathlib.Path(__file__).parent
OUT = HERE / 'stage3_v3_seqcmaf'; OUT.mkdir(exist_ok=True)
AUDIO_CLASS = (HERE / 'audio_model_v3.py').read_text(encoding='utf8')
AUDIO_CLASS = AUDIO_CLASS[AUDIO_CLASS.index('import torch'):]   # drop module docstring

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# MindSync Stage 3 v3 — Stage 2 v3 audio + sequence-level CMAF (leakage-free)
Test set identical to Stage 3 v2 (GoEmotions test × RAVDESS actors 22–24). PRIMARY fusion protocol trains on out-of-sample encoder outputs (see pair-set cell).
New: (1) audio = Stage 2 v3 model (kernel source `mindsync-stage2-v3-audio`); (2) **SeqCMAF**: cross-attention
between RoBERTa token states (≤64 tokens) and wav2vec2 frame features (≤250 frames), so attention is non-trivial;
(3) pooled CMAF kept for comparison. 5 seeds per configuration. Ends with a SUMMARY block.
""")

code("""
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.45.2', 'datasets==2.21.0',
                'huggingface_hub==0.25.2'], check=True)
import os, json, time, random, pathlib, urllib.request, zipfile, glob, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
WORK = pathlib.Path('/kaggle/working'); OUT_DIR = WORK / 'stage3_v3'; OUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda')
print(torch.__version__, torch.cuda.get_device_name(0), flush=True)
hits = glob.glob('/kaggle/input/**/audio_v3_best.pt', recursive=True)
assert hits, 'audio_v3_best.pt not found under /kaggle/input — attach kernel mindsync-stage2-v3-audio'
AUDIO_CKPT = hits[0]; print('audio ckpt:', AUDIO_CKPT)
""")

code("""
CLUSTER_NAMES = ['Distress', 'Resilience', 'Aggression', 'Ambiguity']
C2I = {c: i for i, c in enumerate(CLUSTER_NAMES)}
EMOTION_TO_CLUSTER = {
    'grief': 'Distress', 'nervousness': 'Distress', 'fear': 'Distress', 'sadness': 'Distress', 'remorse': 'Distress',
    'joy': 'Resilience', 'admiration': 'Resilience', 'excitement': 'Resilience', 'relief': 'Resilience',
    'amusement': 'Resilience', 'approval': 'Resilience', 'curiosity': 'Resilience', 'desire': 'Resilience',
    'gratitude': 'Resilience', 'love': 'Resilience', 'optimism': 'Resilience', 'pride': 'Resilience',
    'realization': 'Resilience',
    'anger': 'Aggression', 'annoyance': 'Aggression', 'disgust': 'Aggression', 'disapproval': 'Aggression',
    'embarrassment': 'Aggression',
    'confusion': 'Ambiguity', 'disappointment': 'Ambiguity', 'surprise': 'Ambiguity', 'caring': 'Ambiguity',
    'neutral': 'Ambiguity',
}
RAVDESS_EMOTION_TO_CLUSTER = {1: 3, 2: 1, 3: 1, 4: 0, 5: 2, 6: 0, 7: 2, 8: 3}

DATA_DIR = pathlib.Path('/tmp/ravdess'); DATA_DIR.mkdir(parents=True, exist_ok=True); ZIP = DATA_DIR / 'ravdess.zip'
if not ZIP.exists():
    urllib.request.urlretrieve('https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip', ZIP)
AUDIO_ROOT = DATA_DIR / 'extracted'
if not AUDIO_ROOT.exists():
    with zipfile.ZipFile(ZIP) as z: z.extractall(AUDIO_ROOT)
wavs = sorted(AUDIO_ROOT.rglob('*.wav')); print('RAVDESS files:', len(wavs))
assert len(wavs) == 1440, f'expected 1440 RAVDESS speech clips, found {len(wavs)}'
""")

md("## Models: Stage 1 text (unchanged, from HF) + Stage 2 v3 audio (verbatim class)")
code(AUDIO_CLASS + """

from transformers import RobertaModel, AutoTokenizer, Wav2Vec2FeatureExtractor
from huggingface_hub import hf_hub_download

class MindSyncTextModel(nn.Module):
    def __init__(self, model_name='roberta-large', num_classes=4, dropout=0.1):
        super().__init__()
        self.encoder = nn.Module(); self.encoder.encoder = RobertaModel.from_pretrained(model_name)
        self.encoder.hidden_size = self.encoder.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Module(); h = self.encoder.hidden_size
        self.classifier.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, h // 2), nn.GELU(),
                                                   nn.Dropout(dropout), nn.Linear(h // 2, num_classes))

def freeze(m):
    m.eval()
    for p in m.parameters(): p.requires_grad = False
    return m

sd = torch.load(hf_hub_download('Ubaida1/mindsync-text-model', 'best_text_model.pt'), map_location='cpu', weights_only=False)
sd = sd.get('model_state_dict', sd) if isinstance(sd, dict) else sd
text_model = MindSyncTextModel(); text_model.load_state_dict(sd); text_model = freeze(text_model.to(device))

ack = torch.load(AUDIO_CKPT, map_location='cpu', weights_only=False)
audio_model = AudioModelV3(model_name=ack.get('model_name', AUDIO_MODEL_NAME), freeze_layers=ack.get('freeze_layers', 8))
audio_model.load_state_dict(ack['model_state_dict']); audio_model = freeze(audio_model.to(device))
print('audio v3 ckpt: seed', ack.get('seed'), '| val F1', ack.get('val_f1'), '| test acc', ack.get('test_acc'), '| test F1', ack.get('test_f1'))
""")

md("## Audio: frame features, pooled embedding, standalone posterior (eval preprocessing = Stage 2 v3)")
code("""
import librosa
fe = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)
def load_clip(path):
    wav, _ = librosa.load(path, sr=16_000, mono=True)
    wav, _ = librosa.effects.trim(wav, top_db=30)
    if len(wav) > 80_000:
        s = (len(wav) - 80_000) // 2; wav = wav[s:s + 80_000]
    return wav.astype(np.float32)

frames, fmask_len, a_emb, a_logits, a_cluster, a_actor = [], [], [], [], [], []
t0 = time.time()
for i, wp in enumerate(wavs):
    parts = wp.stem.split('-')
    if len(parts) != 7: continue
    emo, actor = int(parts[2]), int(parts[6])
    iv = fe(load_clip(str(wp)), sampling_rate=16_000, return_tensors='pt').input_values.to(device)
    ln = torch.tensor([iv.shape[1]], device=device)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        f, m = audio_model.frames(iv, ln)
        emb = audio_model.proj(audio_model.pool(f.float(), m))
        lg = audio_model.head(emb)
    n = int(m.sum())
    frames.append(f[0, :n].half().cpu()); fmask_len.append(n)
    a_emb.append(emb[0].float().cpu()); a_logits.append(lg[0].float().cpu())
    a_cluster.append(RAVDESS_EMOTION_TO_CLUSTER[emo]); a_actor.append(actor)
    if (i + 1) % 200 == 0: print(f'  {i+1}/{len(wavs)} {time.time()-t0:.0f}s', flush=True)
TMAX = max(fmask_len)
AF = torch.zeros(len(frames), TMAX, 1024, dtype=torch.float16)
AM = torch.zeros(len(frames), TMAX, dtype=torch.bool)
for k, f in enumerate(frames): AF[k, :len(f)] = f; AM[k, :len(f)] = True
del frames
audio_embs = torch.stack(a_emb); audio_probs = F.softmax(torch.stack(a_logits), -1)
audio_clusters = torch.tensor(a_cluster); audio_actors = torch.tensor(a_actor)
a_split = {'train': audio_actors <= 18, 'val': (audio_actors >= 19) & (audio_actors <= 21), 'test': audio_actors >= 22}
for k, msk in a_split.items():
    print(f"audio {k}: n={int(msk.sum())} standalone acc {(audio_probs[msk].argmax(-1) == audio_clusters[msk]).float().mean():.4f}")
print('frames: TMAX', TMAX, '| mean valid', float(np.mean(fmask_len)))
_te = float((audio_probs[a_split['test']].argmax(-1) == audio_clusters[a_split['test']]).float().mean())
assert ack.get('test_acc') is None or abs(_te - ack['test_acc']) < 0.012, f'audio test acc {_te:.4f} != checkpoint {ack[\"test_acc\"]}: wrong ckpt or preprocessing'
""")

md("## Text: CLS embedding, token states (first 64 tokens), standalone posterior")
code("""
from datasets import load_dataset
ds = load_dataset('google-research-datasets/go_emotions', 'simplified')
names = ds['train'].features['labels'].feature.names
to_c = lambda ids: C2I[EMOTION_TO_CLUSTER.get(names[ids[0]], 'Ambiguity')] if ids else None
tok = AutoTokenizer.from_pretrained('roberta-large')
NTOK = 64; rng = random.Random(42); T = {}
for split in ['train', 'validation', 'test']:
    items = [(it['text'], to_c(it['labels'])) for it in ds[split]]
    items = [x for x in items if x[1] is not None]
    if split == 'train':
        by_c = {c: [x for x in items if x[1] == c] for c in range(4)}
        items = [x for c in range(4) for x in rng.sample(by_c[c], min(1500, len(by_c[c])))]
    texts = [x[0] for x in items]; labels = torch.tensor([x[1] for x in items])
    cls, toks, tmask = [], [], []
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i + 32], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        with torch.no_grad():
            h = text_model.encoder.encoder(input_ids=enc['input_ids'].to(device),
                                           attention_mask=enc['attention_mask'].to(device)).last_hidden_state
        cls.append(h[:, 0].float().cpu()); toks.append(h[:, :NTOK].half().cpu())
        tmask.append(enc['attention_mask'][:, :NTOK].bool())
    emb = torch.cat(cls)
    with torch.no_grad():
        probs = F.softmax(text_model.classifier.classifier(emb.to(device)).float(), -1).cpu()
    T[split] = {'emb': emb, 'tok': torch.cat(toks), 'tmask': torch.cat(tmask), 'cluster': labels,
                'texts': texts, 'probs': probs}
    print(split, len(texts), flush=True)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
yt_ = T['test']['cluster']; pt_ = T['test']['probs'].argmax(-1)
print(f"text test acc {accuracy_score(yt_, pt_):.4f} F1 {f1_score(yt_, pt_, average='macro'):.4f} (expect ~0.7076 / 0.6822 in fp32)")
ytr_ = T['train']['cluster']; ptr_ = T['train']['probs'].argmax(-1)
print(f"text TRAIN-subset acc {accuracy_score(ytr_, ptr_):.4f} (in-sample for Stage 1)")
""")

md("## Real Stage 2 v3 audio report (actors 22–24)")
code("""
def report(name, y, p, fname):
    cm = confusion_matrix(y, p, labels=range(4))
    fig, ax = plt.subplots(figsize=(6, 5)); im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(CLUSTER_NAMES, rotation=20); ax.set_yticklabels(CLUSTER_NAMES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(f'{name} (n={len(y)})')
    for i in range(4):
        for j in range(4): ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.colorbar(im, ax=ax, fraction=0.04); plt.tight_layout(); plt.savefig(OUT_DIR / fname, dpi=200); plt.close()
    return {'report': classification_report(y, p, labels=range(4), target_names=CLUSTER_NAMES, digits=4, output_dict=True, zero_division=0),
            'confusion': cm.tolist(), 'n': int(len(y))}
m = a_split['test']
STAGE2_V3 = report('Stage 2 v3 audio — confusion matrix', audio_clusters[m].numpy(), audio_probs[m].argmax(-1).numpy(), 'stage2_v3_confusion.png')
print(json.dumps({k: v for k, v in STAGE2_V3['report'].items() if k in ('accuracy', 'macro avg')}, indent=1))
""")

md("""
## Pair sets and protocols (pre-specified before any result was seen)
* **test** (shared): every GoEmotions *test* comment × a random same-cluster clip from actors 22–24.
* **out_of_sample** (PRIMARY): the fusion is trained on encoder outputs the encoders were never trained on —
  GoEmotions *validation* split 80/20 (fixed permutation) → fusion-train texts × actors 19–20, fusion-val texts × actor 21.
* **in_sample** (secondary, comparable to Stage 3 v2): GoEmotions *train* (1,500/cluster) × actors 1–18, GoEmotions *validation* × actors 19–21.
""")
code("""
def build_pairs(tsplit, text_idx, audio_mask, seed, congruent=True):
    r = random.Random(seed)
    pool = {c: [i for i in torch.nonzero(audio_mask).squeeze(-1).tolist() if audio_clusters[i].item() == c] for c in range(4)}
    labels = T[tsplit]['cluster']; out = []
    for ti in text_idx:
        c = int(labels[ti]); ac = c if congruent else r.choice([k for k in range(4) if k != c])
        out.append((ti, r.choice(pool[ac]), c))
    return {'ts': tsplit, 'pairs': out}

ACT = {a: audio_actors == a for a in range(1, 25)}
m_19_20 = ACT[19] | ACT[20]; m_21 = ACT[21]
perm = torch.randperm(len(T['validation']['cluster']), generator=torch.Generator().manual_seed(7)).tolist()
cut = int(0.8 * len(perm)); VAL_A, VAL_B = sorted(perm[:cut]), sorted(perm[cut:])
ALL = lambda ts: list(range(len(T[ts]['cluster'])))
TEST = build_pairs('test', ALL('test'), a_split['test'], 3)
PROTOCOLS = {
    'out_of_sample': {'train': build_pairs('validation', VAL_A, m_19_20, 4), 'val': build_pairs('validation', VAL_B, m_21, 5)},
    'in_sample':     {'train': build_pairs('train', ALL('train'), a_split['train'], 1), 'val': build_pairs('validation', ALL('validation'), a_split['val'], 2)},
}
PRIMARY = 'out_of_sample'
for p_, d_ in PROTOCOLS.items(): print(p_, {k: len(v['pairs']) for k, v in d_.items()}, '| test', len(TEST['pairs']))

def fetch(ps, idx):
    ts = T[ps['ts']]; pairs = ps['pairs']
    ti = torch.tensor([pairs[i][0] for i in idx]); ai = torch.tensor([pairs[i][1] for i in idx])
    return {'cls': ts['emb'][ti].to(device), 'tok': ts['tok'][ti].to(device).float(), 'tm': ts['tmask'][ti].to(device),
            'aemb': audio_embs[ai].to(device), 'fr': AF[ai].to(device).float(), 'fm': AM[ai].to(device),
            'y': torch.tensor([pairs[i][2] for i in idx], device=device)}

def stream_probs(ps):
    ts = T[ps['ts']]; pairs = ps['pairs']
    ti = torch.tensor([p[0] for p in pairs]); ai = torch.tensor([p[1] for p in pairs])
    return ts['probs'][ti], audio_probs[ai], torch.tensor([p[2] for p in pairs])

def metrics(y, p): return {'acc': float(accuracy_score(y, p)), 'f1': float(f1_score(y, p, average='macro'))}
pt_test, pa_test, y_test = stream_probs(TEST)
late_pred = (pt_test + pa_test).argmax(-1)
BASE = {'text_only': metrics(y_test, pt_test.argmax(-1)), 'audio_only': metrics(y_test, pa_test.argmax(-1)),
        'late_fusion_avg': metrics(y_test, late_pred)}
for k, v in BASE.items(): print(f"{k:16s} acc {v['acc']:.4f} F1 {v['f1']:.4f}")
# per-stream accuracy on each protocol's fusion-train pairs (shows how over-confident in-sample features are)
TRAIN_STREAM_ACC = {}
for p_, d_ in PROTOCOLS.items():
    a_, b_, y_ = stream_probs(d_['train'])
    TRAIN_STREAM_ACC[p_] = {'text': float((a_.argmax(-1) == y_).float().mean()), 'audio': float((b_.argmax(-1) == y_).float().mean())}
print('stream accuracy on fusion-train pairs:', TRAIN_STREAM_ACC)
""")

md("## Fusion models: pooled CMAF (v2 design) and sequence-level SeqCMAF")
code("""
class PooledCMAF(nn.Module):
    def __init__(self, d=256, heads=4, dropout=0.1, attention=True, aux=True):
        super().__init__(); self.attention, self.aux = attention, aux
        self.tp, self.ap = nn.Linear(1024, d), nn.Linear(1024, d)
        if attention:
            self.t2a = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
            self.a2t = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
            self.nt, self.na = nn.LayerNorm(d), nn.LayerNorm(d)
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 4))
        self.ht, self.ha = nn.Linear(d, 4), nn.Linear(d, 4)
    def forward(self, b, need_weights=False):
        t, a = self.tp(b['cls']).unsqueeze(1), self.ap(b['aemb']).unsqueeze(1)
        if self.attention:
            ta, _ = self.t2a(t, a, a); at, _ = self.a2t(a, t, t)
            t, a = self.nt(t + ta), self.na(a + at)
        t, a = t.squeeze(1), a.squeeze(1)
        return {'fused_logits': self.fuse(torch.cat([t, a], -1)), 'text_logits': self.ht(t), 'audio_logits': self.ha(a)}

def mpool(x, m):
    mm = m.unsqueeze(-1).float(); return (x * mm).sum(1) / mm.sum(1).clamp_min(1.0)

class SeqCMAF(nn.Module):
    \"\"\"Text tokens attend over audio frames and vice versa (key padding masks), then masked mean pooling.\"\"\"
    def __init__(self, d=256, heads=4, dropout=0.1, aux=True):
        super().__init__(); self.aux = aux
        self.tp, self.ap = nn.Linear(1024, d), nn.Linear(1024, d)
        self.t2a = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.a2t = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.nt, self.na = nn.LayerNorm(d), nn.LayerNorm(d)
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 4))
        self.ht, self.ha = nn.Linear(d, 4), nn.Linear(d, 4)
    def forward(self, b, need_weights=False):
        t, a = self.tp(b['tok']), self.ap(b['fr'])
        ta, w = self.t2a(t, a, a, key_padding_mask=~b['fm'], need_weights=need_weights)
        at, _ = self.a2t(a, t, t, key_padding_mask=~b['tm'], need_weights=False)
        t, a = self.nt(t + ta), self.na(a + at)
        tp, ap = mpool(t, b['tm']), mpool(a, b['fm'])
        out = {'fused_logits': self.fuse(torch.cat([tp, ap], -1)), 'text_logits': self.ht(tp), 'audio_logits': self.ha(ap)}
        if need_weights: out['attn'] = w
        return out

CONFIGS = {
    'pooled_full':   (PooledCMAF, {}),
    'pooled_concat': (PooledCMAF, {'attention': False}),
    'seq_full':      (SeqCMAF, {}),
    'seq_no_aux':    (SeqCMAF, {'aux': False}),
    'seq_d128':      (SeqCMAF, {'d': 128}),
}
RUN_PLAN = {'out_of_sample': list(CONFIGS), 'in_sample': ['pooled_full', 'pooled_concat', 'seq_full']}
SEEDS = [42, 43, 44, 45, 46]

@torch.no_grad()
def predict(model, ps, bs=256):
    model.eval(); outs = {'fused_logits': [], 'text_logits': [], 'audio_logits': []}
    n = len(ps['pairs'])
    for i in range(0, n, bs):
        o = model(fetch(ps, range(i, min(i + bs, n))))
        for k in outs: outs[k].append(o[k].float().cpu())
    return {k: torch.cat(v) for k, v in outs.items()}

def train_one(cls_, cfg, seed, proto, epochs=10, lr=3e-4, alpha=0.3, bs=64):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = cls_(**cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    tr, va = PROTOCOLS[proto]['train'], PROTOCOLS[proto]['val']
    y = torch.tensor([p[2] for p in tr['pairs']]); yv = torch.tensor([p[2] for p in va['pairs']])
    w = (1.0 / torch.bincount(y, minlength=4).float())[y]
    g = torch.Generator().manual_seed(seed)
    best, best_state = -1, None
    for ep in range(epochs):
        model.train()
        idx = torch.multinomial(w, len(y), replacement=True, generator=g).tolist()
        for i in range(0, len(idx), bs):
            b = fetch(tr, idx[i:i + bs]); out = model(b)
            loss = F.cross_entropy(out['fused_logits'], b['y'])
            if model.aux:
                loss = loss + alpha * (F.cross_entropy(out['text_logits'], b['y']) + F.cross_entropy(out['audio_logits'], b['y']))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        vf1 = f1_score(yv, predict(model, va)['fused_logits'].argmax(-1), average='macro')
        if vf1 > best: best, best_state = vf1, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, float(best)

RESULTS, MODELS, PRED = {}, {}, {}
for proto, names_ in RUN_PLAN.items():
    RESULTS[proto] = {}
    for name in names_:
        cls_, cfg = CONFIGS[name]; runs = []; t0 = time.time()
        for s in SEEDS:
            m_, vf1 = train_one(cls_, cfg, s, proto)
            pred = predict(m_, TEST)['fused_logits'].argmax(-1)
            runs.append({'seed': s, 'val_f1': vf1, **metrics(y_test, pred)})
            if s == 42: MODELS[(proto, name)], PRED[(proto, name)] = m_, pred
            else: del m_
        a = np.array([r['acc'] for r in runs]); f = np.array([r['f1'] for r in runs])
        RESULTS[proto][name] = {'runs': runs, 'acc_mean': float(a.mean()), 'acc_std': float(a.std(ddof=1)),
                                'f1_mean': float(f.mean()), 'f1_std': float(f.std(ddof=1)),
                                'params': sum(p.numel() for p in cls_(**cfg).parameters())}
        print(f"[{proto}] {name:14s} acc {a.mean()*100:.2f} ± {a.std(ddof=1)*100:.2f} | F1 {f.mean():.4f} ± {f.std(ddof=1):.4f} | {time.time()-t0:.0f}s", flush=True)
    with open(OUT_DIR / 'fusion_results_partial.json', 'w', encoding='utf-8') as fh: json.dump(RESULTS, fh, indent=1)
""")

md("""
## Significance: cluster bootstrap over the 180 test clips (pairs re-use clips, so pairs are not independent)
Pair-level McNemar is kept for reference only and labelled as optimistic.
""")
code("""
from scipy.stats import binomtest
yt = y_test.numpy()
clip_of_pair = np.array([p[1] for p in TEST['pairs']])
clips = np.unique(clip_of_pair); clip_pos = {c: i for i, c in enumerate(clips)}
cidx = np.array([clip_pos[c] for c in clip_of_pair])
n_per_clip = np.bincount(cidx, minlength=len(clips)).astype(float)
def correct_per_clip(pred): return np.bincount(cidx, weights=(np.asarray(pred) == yt).astype(float), minlength=len(clips))
PREDS_FOR_SIG = {'text_only': pt_test.argmax(-1).numpy(), 'audio_only': pa_test.argmax(-1).numpy(), 'late_fusion': late_pred.numpy()}
for (proto, name), pr in PRED.items(): PREDS_FOR_SIG[f'{proto}/{name}'] = pr.numpy()
K = {k: correct_per_clip(v) for k, v in PREDS_FOR_SIG.items()}
rng_ = np.random.default_rng(0); B = 5000
W = rng_.multinomial(len(clips), np.full(len(clips), 1.0 / len(clips)), size=B).astype(float)   # (B, n_clips)
den = W @ n_per_clip
BOOT = {k: (W @ v) / den for k, v in K.items()}
def ci(v): return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
SIG = {'n_test_clips': int(len(clips)), 'n_test_pairs': int(len(yt)), 'bootstrap_resamples': B,
       'acc_ci95_cluster': {k: ci(v) for k, v in BOOT.items()}}
P_ = f'{PRIMARY}/'
COMPARISONS = [(P_ + 'seq_full', 'late_fusion'), (P_ + 'pooled_full', 'late_fusion'), (P_ + 'seq_full', P_ + 'pooled_full'),
               (P_ + 'seq_full', 'text_only'), (P_ + 'pooled_full', P_ + 'pooled_concat'), (P_ + 'seq_full', P_ + 'pooled_concat')]
def mcnemar(a, b):
    ca, cb = (a == yt), (b == yt); b01, b10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return {'first_only_correct': b01, 'second_only_correct': b10,
            'p_value_pair_level_optimistic': float(binomtest(b01, b01 + b10, 0.5).pvalue) if b01 + b10 else 1.0}
SIG['differences'] = {}
for a_, b_ in COMPARISONS:
    d = BOOT[a_] - BOOT[b_]
    SIG['differences'][f'{a_} - {b_}'] = {'acc_diff': float((PREDS_FOR_SIG[a_] == yt).mean() - (PREDS_FOR_SIG[b_] == yt).mean()),
        'ci95_cluster': ci(d), 'p_two_sided_cluster': float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
        **mcnemar(PREDS_FOR_SIG[a_], PREDS_FOR_SIG[b_])}
print(json.dumps(SIG, indent=1))
PER_CLASS = {k: classification_report(yt, PRED[(PRIMARY, k)].numpy(), target_names=CLUSTER_NAMES, digits=4, output_dict=True)
             for k in ['seq_full', 'pooled_full']}
CONF = {k: confusion_matrix(yt, PRED[(PRIMARY, k)].numpy()).tolist() for k in ['seq_full', 'pooled_full']}

def ece(probs, y, n_bins=15):
    conf, pred = probs.max(-1); acc = (pred == y).float(); e = 0.0
    edges = torch.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mm = (conf > lo) & (conf <= hi)
        if mm.any(): e += mm.float().mean().item() * abs(acc[mm].mean().item() - conf[mm].mean().item())
    return float(e)
ECE = {'text_only': ece(pt_test, y_test), 'audio_only': ece(pa_test, y_test), 'late_fusion_avg': ece((pt_test + pa_test) / 2, y_test)}
for k in ['seq_full', 'pooled_full']:
    ECE[f'{PRIMARY}/{k}'] = ece(F.softmax(predict(MODELS[(PRIMARY, k)], TEST)['fused_logits'], -1), y_test)
print('ECE', json.dumps(ECE, indent=1))
""")

md("## JSD incongruence (test speakers only; threshold calibrated on GoEmotions-validation × actors 19–21)")
code("""
def jsd(p, q, eps=1e-8):
    m = 0.5 * (p + q)
    kl = lambda x: (x * (x.clamp_min(eps).log2() - m.clamp_min(eps).log2())).sum(-1)
    return 0.5 * (kl(p) + kl(q))
AUXM = MODELS[(PRIMARY, 'seq_full')]
def jsd_scores(tsplit, audio_mask, congruent, seed):
    ps = build_pairs(tsplit, ALL(tsplit), audio_mask, seed, congruent)
    pt, pa, _ = stream_probs(ps); o = predict(AUXM, ps)
    return {'standalone': jsd(pt, pa).numpy(),
            'aux': jsd(F.softmax(o['text_logits'], -1), F.softmax(o['audio_logits'], -1)).numpy(),
            'mismatch': (pt.argmax(-1) != pa.argmax(-1)).numpy()}
JSDS = {'val': {'cong': jsd_scores('validation', a_split['val'], True, 10), 'incong': jsd_scores('validation', a_split['val'], False, 11)},
        'test': {'cong': jsd_scores('test', a_split['test'], True, 20), 'incong': jsd_scores('test', a_split['test'], False, 21)}}
JSD_RES = {}
for kind in ['standalone', 'aux']:
    c, i = JSDS['test']['cong'][kind], JSDS['test']['incong'][kind]
    vc, vi = JSDS['val']['cong'][kind], JSDS['val']['incong'][kind]
    best_t, best_j = 0.5, -1
    for t_ in np.unique(np.r_[vc, vi]):
        j = (vi > t_).mean() - (vc > t_).mean()
        if j > best_j: best_j, best_t = j, float(t_)
    JSD_RES[kind] = {'cong_mean': float(c.mean()), 'cong_std': float(c.std()), 'cong_median': float(np.median(c)),
                     'incong_mean': float(i.mean()), 'incong_std': float(i.std()), 'incong_median': float(np.median(i)),
                     'ratio': float(i.mean() / max(c.mean(), 1e-9)),
                     'roc_auc': float(roc_auc_score(np.r_[np.zeros(len(c)), np.ones(len(i))], np.r_[c, i])),
                     'tau_val': best_t, 'detect@tau': float((i > best_t).mean()), 'false_alarm@tau': float((c > best_t).mean()),
                     'detect@0.5': float((i > 0.5).mean()), 'false_alarm@0.5': float((c > 0.5).mean())}
mc, mi = JSDS['test']['cong']['mismatch'], JSDS['test']['incong']['mismatch']
sc, si = JSDS['test']['cong']['standalone'], JSDS['test']['incong']['standalone']
tau = JSD_RES['standalone']['tau_val']
JSD_RES['alert_deployed_tau0.5'] = {'detect': float((mi | (si > 0.5)).mean()), 'false_alarm': float((mc | (sc > 0.5)).mean())}
JSD_RES['alert_calibrated'] = {'tau': tau, 'detect': float((mi | (si > tau)).mean()), 'false_alarm': float((mc | (sc > tau)).mean())}
JSD_RES['mismatch_only'] = {'detect': float(mi.mean()), 'false_alarm': float(mc.mean())}
print(json.dumps(JSD_RES, indent=1))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
for k, kind in enumerate(['standalone', 'aux']):
    ax[k].hist(JSDS['test']['cong'][kind], bins=40, alpha=.6, label='congruent')
    ax[k].hist(JSDS['test']['incong'][kind], bins=40, alpha=.6, label='incongruent')
    ax[k].set_title(f"JSD ({kind}) — AUC {JSD_RES[kind]['roc_auc']:.3f}"); ax[k].set_xlabel('JSD'); ax[k].legend()
plt.tight_layout(); plt.savefig(OUT_DIR / 'jsd_v3.png', dpi=200); plt.close()
""")

md("## Attention analysis (SeqCMAF, primary protocol, seed 42) + error analysis")
code("""
sub = {'ts': 'test', 'pairs': TEST['pairs'][:512]}
with torch.no_grad():
    AUXM.eval(); b = fetch(sub, range(len(sub['pairs']))); o = AUXM(b, need_weights=True)
w = o['attn'].float()
nvalid = b['fm'].sum(1).float()
H = -(w.clamp_min(1e-12) * w.clamp_min(1e-12).log()).sum(-1) / nvalid.log()[:, None]
tm = b['tm'].float()
ATTN = {'mean_normalised_entropy': float((H * tm).sum() / tm.sum()),
        'mean_max_weight': float((w.max(-1).values * tm).sum() / tm.sum()),
        'uniform_max_weight_ref': float((1.0 / nvalid).mean())}
print('attention:', ATTN)
k = 0
ids = tok(T['test']['texts'][sub['pairs'][k][0]], truncation=True, max_length=128)['input_ids'][:NTOK]
toks_ = tok.convert_ids_to_tokens(ids); nt, nf = len(toks_), int(b['fm'][k].sum())
fig, ax = plt.subplots(figsize=(10, max(2.5, nt * 0.25)))
ax.imshow(w[k, :nt, :nf].cpu(), aspect='auto', cmap='viridis')
ax.set_yticks(range(nt)); ax.set_yticklabels([t.replace('Ġ', '') for t in toks_], fontsize=7)
ax.set_xlabel('audio frame (20 ms)'); ax.set_title('Text-to-audio attention (SeqCMAF, one test pair)')
plt.tight_layout(); plt.savefig(OUT_DIR / 'attention_example.png', dpi=200); plt.close()

fp_ = PRED[(PRIMARY, 'seq_full')]; tp_ = pt_test.argmax(-1); ap_ = pa_test.argmax(-1)
err = (fp_ != y_test).nonzero().squeeze(-1).tolist()
ERR = {'n_errors': len(err), 'errors_text_correct': int(sum(int(tp_[i] == y_test[i]) for i in err)),
       'errors_audio_correct': int(sum(int(ap_[i] == y_test[i]) for i in err)),
       'errors_both_wrong': int(sum(int(tp_[i] != y_test[i] and ap_[i] != y_test[i]) for i in err)),
       'fixed_vs_text': int(((fp_ == y_test) & (tp_ != y_test)).sum()), 'broken_vs_text': int(((fp_ != y_test) & (tp_ == y_test)).sum())}
r_ = random.Random(0)
ERR['examples'] = [{'text': T['test']['texts'][TEST['pairs'][i][0]], 'true': CLUSTER_NAMES[int(y_test[i])],
                    'seq_cmaf': CLUSTER_NAMES[int(fp_[i])], 'text_only': CLUSTER_NAMES[int(tp_[i])],
                    'audio_only': CLUSTER_NAMES[int(ap_[i])]} for i in r_.sample(err, min(25, len(err)))]
print(json.dumps({k: v for k, v in ERR.items() if k != 'examples'}, indent=1))
""")

md("## Save + SUMMARY")
code("""
ALLRES = {'audio_ckpt': {k: ack.get(k) for k in ('seed', 'val_f1', 'test_acc', 'test_f1')}, 'stage2_v3_report': STAGE2_V3,
          'primary_protocol': PRIMARY, 'train_stream_acc': TRAIN_STREAM_ACC, 'baselines': BASE, 'fusion': RESULTS,
          'significance': SIG, 'per_class_seed42': PER_CLASS, 'confusion_seed42': CONF, 'ece': ECE, 'jsd': JSD_RES,
          'attention': ATTN, 'error_analysis': ERR,
          'n_pairs': {p_: {k: len(v['pairs']) for k, v in d_.items()} for p_, d_ in PROTOCOLS.items()} | {'test': len(TEST['pairs'])}}
with open(OUT_DIR / 'results_stage3_v3.json', 'w', encoding='utf-8') as f: json.dump(ALLRES, f, indent=1, ensure_ascii=False)
print('===== SUMMARY START ====='); print(json.dumps(ALLRES, indent=1, ensure_ascii=False)); print('===== SUMMARY END =====')
""")

for c in cells:
    if c['cell_type'] == 'code': ast.parse(c['source'])
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
for c in nb['cells']:
    src = c['source'].split('\n'); c['source'] = [l + '\n' for l in src[:-1]] + [src[-1]]
NAME = 'mindsync-stage3-v3-seqcmaf'
json.dump(nb, open(OUT / f'{NAME}.ipynb', 'w', encoding='utf8'), indent=1, ensure_ascii=False)
json.dump({"id": f"{KAGGLE_USER}/{NAME}", "title": "MindSync Stage3 v3 SeqCMAF", "code_file": f"{NAME}.ipynb",
           "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
           "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [],
           "kernel_sources": [f"{KAGGLE_USER}/mindsync-stage2-v3-audio"]},
          open(OUT / 'kernel-metadata.json', 'w'), indent=1)
print('built', len(cells), 'cells ->', OUT)
