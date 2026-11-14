"""Builds notebooks_v3/real_pairs/: Stage 4 real-pair evaluation.
MELD (real text+audio of the same utterance): zero-shot transfer of Stage 1/2, and fusion trained on MELD train,
selected on dev, tested on test. MUStARD (real sarcasm, words vs tone): JSD-based incongruence detection.
All analysis choices fixed before results were seen."""
import json, pathlib, ast, os
KAGGLE_USER = os.environ.get('KAGGLE_USER', 'YOUR_KAGGLE_USERNAME')
HERE = pathlib.Path(__file__).parent
AUDIO_CLASS = (HERE / 'audio_model_v3.py').read_text(encoding='utf8')
AUDIO_CLASS = AUDIO_CLASS[AUDIO_CLASS.index('import torch'):]
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# MindSync Stage 4 — real paired data (MELD) and real incongruence (MUStARD sarcasm)
Encoders are frozen: Stage 1 text model (GoEmotions) and Stage 2 v3 audio model (RAVDESS, kernel source).
**MELD**: emotion → cluster (joy→Resilience; sadness, fear→Distress; anger, disgust→Aggression; neutral, surprise→Ambiguity).
(a) zero-shot: standalone Stage 1/2 posteriors on MELD test; (b) trained on real pairs: text head, audio head, late fusion of the
two heads, pooled CMAF, pooled concatenation, SeqCMAF — 5 seeds, trained on MELD train, selected on dev (macro-F1), tested on test;
cluster bootstrap over test dialogues. **MUStARD**: sarcastic vs non-sarcastic utterances; primary metric = ROC-AUC of the JSD between
standalone Stage 1 and Stage 2 posteriors (no training on MUStARD); secondary: JSD of MELD-trained heads and SeqCMAF auxiliary heads,
and a speaker-grouped 5-fold logistic probe. Pre-specified before any result was seen.
""")

code("""
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.45.2', 'datasets==2.21.0',
                'huggingface_hub==0.25.2', 'ftfy'], check=True)
import os, gc, json, time, random, pathlib, glob, shutil, urllib.request, traceback
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
T_START = time.time(); hours = lambda: (time.time() - T_START) / 3600
WORK = pathlib.Path('/kaggle/working'); OUT = WORK / 'real_pairs'; OUT.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda'); print(torch.__version__, torch.cuda.get_device_name(0), flush=True)
FFMPEG = shutil.which('ffmpeg')
if FFMPEG is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'imageio-ffmpeg'], check=True)
    import imageio_ffmpeg; FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
print('ffmpeg:', FFMPEG)
def find_one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True)); assert hits, f'not found: {pattern}'; return hits[0]
AUDIO_CKPT = find_one('/kaggle/input/**/audio_v3_best.pt')
MELD_ROOT = pathlib.Path(find_one('/kaggle/input/**/MELD.Raw/dev_sent_emo.csv')).parent
MUST_AUDIO = pathlib.Path(find_one('/kaggle/input/**/audio/*_u.wav')).parent
print('MELD root:', MELD_ROOT, '| MUStARD audio:', MUST_AUDIO, '| audio ckpt:', AUDIO_CKPT, flush=True)
""")

md("## Models (frozen Stage 1 text, Stage 2 v3 audio)")
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
text_model = MindSyncTextModel(); text_model.load_state_dict(sd); text_model = freeze(text_model.to(device)); del sd
ack = torch.load(AUDIO_CKPT, map_location='cpu', weights_only=False)
audio_model = AudioModelV3(model_name=ack.get('model_name', AUDIO_MODEL_NAME), freeze_layers=ack.get('freeze_layers', 8))
audio_model.load_state_dict(ack['model_state_dict']); audio_model = freeze(audio_model.to(device))
print('audio ckpt seed', ack.get('seed'), 'RAVDESS test acc', ack.get('test_acc'), flush=True); del ack
tok = AutoTokenizer.from_pretrained('roberta-large'); fe = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)
CLUSTER_NAMES = ['Distress', 'Resilience', 'Aggression', 'Ambiguity']; C2I = {c: i for i, c in enumerate(CLUSTER_NAMES)}
NTOK, SR, MAXS, MINS = 64, 16_000, 80_000, 400
""")

md("## Feature extraction (text: fp32; audio: same eval preprocessing as Stage 2 v3, one clip at a time)")
code("""
import ftfy, librosa
def encode_texts(texts):
    cls, toks, tmask = [], [], []
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i + 32], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        with torch.no_grad():
            h = text_model.encoder.encoder(input_ids=enc['input_ids'].to(device), attention_mask=enc['attention_mask'].to(device)).last_hidden_state
        cls.append(h[:, 0].float().cpu()); toks.append(h[:, :NTOK].half().cpu()); tmask.append(enc['attention_mask'][:, :NTOK].bool())
    emb = torch.cat(cls)
    with torch.no_grad(): probs = F.softmax(text_model.classifier.classifier(emb.to(device)).float(), -1).cpu()
    return {'emb': emb, 'tok': torch.cat(toks), 'tmask': torch.cat(tmask), 'probs': probs}

def decode(path):
    try:
        if str(path).endswith('.wav'):
            w, _ = librosa.load(str(path), sr=SR, mono=True)
        else:
            r = subprocess.run([FFMPEG, '-nostdin', '-v', 'error', '-i', str(path), '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                               capture_output=True, timeout=120)
            if r.returncode != 0: return None
            w = np.frombuffer(r.stdout, np.float32).copy()
        if len(w) < MINS: return None
        w, _ = librosa.effects.trim(w, top_db=30)
        if len(w) > MAXS: s = (len(w) - MAXS) // 2; w = w[s:s + MAXS]
        if len(w) < MINS: w = np.pad(w, (0, MINS - len(w)))
        return w.astype(np.float32)
    except Exception:
        return None

def encode_audio(paths):
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex: wavs = list(ex.map(decode, paths))
    print(f'  decoded {sum(w is not None for w in wavs)}/{len(paths)} in {time.time()-t0:.0f}s', flush=True)
    TM = int(audio_model.enc._get_feat_extract_output_lengths(torch.tensor(MAXS)))     # frames of a full 5 s crop
    AF = torch.zeros(len(paths), TM, 1024, dtype=torch.float16); AM = torch.zeros(len(paths), TM, dtype=torch.bool)
    emb, lg, ok = [], [], []
    for k in range(len(wavs)):
        w = wavs[k]; wavs[k] = None                                  # free each waveform once used
        if w is None:
            ok.append(False); emb.append(torch.zeros(1024)); lg.append(torch.zeros(4)); continue
        x = fe(w, sampling_rate=SR, return_tensors='np')['input_values'][0].astype(np.float32)
        iv = torch.from_numpy(x)[None].to(device); ln = torch.tensor([x.shape[0]], device=device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            f_, m_ = audio_model.frames(iv, ln); e_ = audio_model.proj(audio_model.pool(f_.float(), m_)); l_ = audio_model.head(e_)
        n_ = int(m_.sum()); assert n_ <= TM
        AF[k, :n_] = f_[0, :n_].half().cpu(); AM[k, :n_] = True
        emb.append(e_[0].float().cpu()); lg.append(l_[0].float().cpu()); ok.append(True)
    print(f'  audio encoded in {time.time()-t0:.0f}s | frame capacity {TM}', flush=True)
    return {'fr': AF, 'fm': AM, 'aemb': torch.stack(emb), 'aprobs': F.softmax(torch.stack(lg), -1), 'ok': np.array(ok)}

MELD_MAP = {'joy': 'Resilience', 'sadness': 'Distress', 'fear': 'Distress', 'anger': 'Aggression', 'disgust': 'Aggression',
            'neutral': 'Ambiguity', 'surprise': 'Ambiguity'}
MELD_DIRS = {'train': ['train/train_splits', 'train_splits'], 'dev': ['dev/dev_splits_complete', 'dev_splits_complete'],
             'test': ['test/output_repeated_splits_test', 'output_repeated_splits_test']}
D = {}
for split, csvname in [('train', 'train_sent_emo.csv'), ('dev', 'dev_sent_emo.csv'), ('test', 'test_sent_emo.csv')]:
    df = pd.read_csv(find_one(str(MELD_ROOT) + f'/**/{csvname}'))
    vdir = next(MELD_ROOT / d for d in MELD_DIRS[split] if (MELD_ROOT / d).exists())
    df['path'] = [vdir / f'dia{a}_utt{b}.mp4' for a, b in zip(df.Dialogue_ID, df.Utterance_ID)]
    df['text'] = [ftfy.fix_text(str(u)) for u in df.Utterance]
    df['y'] = [C2I[MELD_MAP[e]] for e in df.Emotion]
    print(f'MELD {split}: {len(df)} rows, video files present: {sum(p.exists() for p in df.path)}', flush=True)
    A_ = encode_audio(list(df.path)); T_ = encode_texts(list(df.text))
    keep = A_['ok']
    assert keep.mean() > 0.99, (split, int((~keep).sum()), 'too many unreadable clips')
    sel = (lambda t: t) if keep.all() else (lambda t: t[torch.from_numpy(keep)] if torch.is_tensor(t) else t[keep])
    D[split] = {'tok': sel(T_['tok']), 'tm': sel(T_['tmask']), 'cls': sel(T_['emb']), 'pt': sel(T_['probs']),
                'fr': sel(A_['fr']), 'fm': sel(A_['fm']), 'aemb': sel(A_['aemb']), 'pa': sel(A_['aprobs']),
                'y': torch.tensor(df.y.values[keep]), 'dia': df.Dialogue_ID.values[keep], 'text': list(df.text[keep]),
                'emotion': list(df.Emotion[keep]), 'n_total': int(len(df)), 'n_dropped': int((~keep).sum())}
    del A_, T_; gc.collect()
    print(f'  kept {int(keep.sum())}, dropped {int((~keep).sum())} unreadable clips | elapsed {hours():.2f} h', flush=True)
""")

md("## MELD (a): zero-shot transfer of the standalone encoders")
code("""
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
def met(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {'acc': float(accuracy_score(y, p)), 'macro_f1': float(f1_score(y, p, average='macro')),
            'weighted_f1': float(f1_score(y, p, average='weighted'))}
te = D['test']; yte = te['y'].numpy()
ZERO = {'text_only': met(yte, te['pt'].argmax(-1)), 'audio_only': met(yte, te['pa'].argmax(-1)),
        'late_fusion': met(yte, (te['pt'] + te['pa']).argmax(-1)), 'majority_class': met(yte, np.full_like(yte, np.bincount(D['train']['y'].numpy()).argmax()))}
print(json.dumps(ZERO, indent=1)); print('test label distribution:', {CLUSTER_NAMES[i]: int(c) for i, c in enumerate(np.bincount(yte, minlength=4))})
""")

md("## MELD (b): models trained on real pairs (train → dev selection → test), 5 seeds")
code("""
def mpool(x, m):
    mm = m.unsqueeze(-1).float(); return (x * mm).sum(1) / mm.sum(1).clamp_min(1.0)
class Head(nn.Module):
    def __init__(self, src, d=256, dropout=0.1):
        super().__init__(); self.src, self.aux = src, False
        self.net = nn.Sequential(nn.Linear(1024, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 4))
    def forward(self, b): return {'fused_logits': self.net(b[self.src])}
class PooledCMAF(nn.Module):
    def __init__(self, d=256, heads=4, dropout=0.1, attention=True):
        super().__init__(); self.attention, self.aux = attention, True
        self.tp, self.ap = nn.Linear(1024, d), nn.Linear(1024, d)
        if attention:
            self.t2a = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
            self.a2t = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
            self.nt, self.na = nn.LayerNorm(d), nn.LayerNorm(d)
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 4))
        self.ht, self.ha = nn.Linear(d, 4), nn.Linear(d, 4)
    def forward(self, b):
        t, a = self.tp(b['cls']).unsqueeze(1), self.ap(b['aemb']).unsqueeze(1)
        if self.attention:
            ta, _ = self.t2a(t, a, a); at, _ = self.a2t(a, t, t); t, a = self.nt(t + ta), self.na(a + at)
        t, a = t.squeeze(1), a.squeeze(1)
        return {'fused_logits': self.fuse(torch.cat([t, a], -1)), 'text_logits': self.ht(t), 'audio_logits': self.ha(a)}
class SeqCMAF(nn.Module):
    def __init__(self, d=256, heads=4, dropout=0.1):
        super().__init__(); self.aux = True
        self.tp, self.ap = nn.Linear(1024, d), nn.Linear(1024, d)
        self.t2a = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.a2t = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.nt, self.na = nn.LayerNorm(d), nn.LayerNorm(d)
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 4))
        self.ht, self.ha = nn.Linear(d, 4), nn.Linear(d, 4)
    def forward(self, b):
        t, a = self.tp(b['tok']), self.ap(b['fr'])
        ta, _ = self.t2a(t, a, a, key_padding_mask=~b['fm'], need_weights=False)
        at, _ = self.a2t(a, t, t, key_padding_mask=~b['tm'], need_weights=False)
        t, a = self.nt(t + ta), self.na(a + at); tp, ap = mpool(t, b['tm']), mpool(a, b['fm'])
        return {'fused_logits': self.fuse(torch.cat([tp, ap], -1)), 'text_logits': self.ht(tp), 'audio_logits': self.ha(ap)}

NEEDS = {Head: None, PooledCMAF: ('cls', 'aemb'), SeqCMAF: ('tok', 'tm', 'fr', 'fm')}
def fetch(S, idx, keys=('cls', 'aemb', 'tok', 'tm', 'fr', 'fm')):
    idx = torch.as_tensor(list(idx)); b = {'y': S['y'][idx].to(device)}
    for k in keys:
        v = S[k][idx].to(device); b[k] = v.float() if k in ('tok', 'fr') else v
    return b
def keys_for(model):
    return (model.src,) if isinstance(model, Head) else NEEDS[type(model)]
@torch.no_grad()
def predict(model, S, bs=128):
    model.eval(); outs = {}; n = len(S['y'])
    for i in range(0, n, bs):
        o = model(fetch(S, range(i, min(i + bs, n)), keys_for(model)))
        for k, v in o.items(): outs.setdefault(k, []).append(v.float().cpu())
    return {k: torch.cat(v) for k, v in outs.items()}
def train_model(make, seed, S_tr, S_va, epochs=10, lr=3e-4, alpha=0.3, bs=64):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = make().to(device); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    y = S_tr['y']; w = (1.0 / torch.bincount(y, minlength=4).float())[y]; g = torch.Generator().manual_seed(seed)
    yv = S_va['y'].numpy(); best, best_state = -1, None
    for ep in range(epochs):
        model.train(); idx = torch.multinomial(w, len(y), replacement=True, generator=g).tolist()
        for i in range(0, len(idx), bs):
            b = fetch(S_tr, idx[i:i + bs], keys_for(model)); out = model(b); loss = F.cross_entropy(out['fused_logits'], b['y'])
            if model.aux: loss = loss + alpha * (F.cross_entropy(out['text_logits'], b['y']) + F.cross_entropy(out['audio_logits'], b['y']))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        vf1 = f1_score(yv, predict(model, S_va)['fused_logits'].argmax(-1).numpy(), average='macro')
        if vf1 > best: best, best_state = vf1, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); return model, float(best)

MAKERS = {'text_head': lambda: Head('cls'), 'audio_head': lambda: Head('aemb'), 'pooled_cmaf': lambda: PooledCMAF(),
          'pooled_concat': lambda: PooledCMAF(attention=False), 'seq_cmaf': lambda: SeqCMAF()}
SEEDS = [42, 43, 44, 45, 46]
RES, PROBS, KEEP = {}, {}, {}
for name, mk in MAKERS.items():
    runs = []; t0 = time.time()
    for s in SEEDS:
        m_, vf1 = train_model(mk, s, D['train'], D['dev'])
        p_ = F.softmax(predict(m_, D['test'])['fused_logits'], -1); PROBS[(name, s)] = p_
        runs.append({'seed': s, 'dev_macro_f1': vf1, **met(yte, p_.argmax(-1))})
        if s == 42: KEEP[name] = m_
        else: del m_
    RES[name] = {'runs': runs, **{f'{k}_mean': float(np.mean([r[k] for r in runs])) for k in ('acc', 'macro_f1', 'weighted_f1')},
                 **{f'{k}_std': float(np.std([r[k] for r in runs], ddof=1)) for k in ('acc', 'macro_f1', 'weighted_f1')}}
    print(f"{name:14s} acc {RES[name]['acc_mean']*100:.2f}±{RES[name]['acc_std']*100:.2f} | macroF1 {RES[name]['macro_f1_mean']:.4f}±{RES[name]['macro_f1_std']:.4f} | wF1 {RES[name]['weighted_f1_mean']:.4f} | {time.time()-t0:.0f}s", flush=True)
late = []
for s in SEEDS:
    pr = (PROBS[('text_head', s)] + PROBS[('audio_head', s)]).argmax(-1); PROBS[('late_heads', s)] = None
    late.append({'seed': s, **met(yte, pr)})
RES['late_fusion_heads'] = {'runs': late, **{f'{k}_mean': float(np.mean([r[k] for r in late])) for k in ('acc', 'macro_f1', 'weighted_f1')},
                            **{f'{k}_std': float(np.std([r[k] for r in late], ddof=1)) for k in ('acc', 'macro_f1', 'weighted_f1')}}
print('late_fusion_heads', RES['late_fusion_heads']['acc_mean'], RES['late_fusion_heads']['macro_f1_mean'])
with open(OUT / 'meld_partial.json', 'w', encoding='utf-8') as f: json.dump({'zero_shot': ZERO, 'trained': RES}, f, indent=1)
""")

md("## MELD significance: cluster bootstrap over test dialogues (seed-42 models)")
code("""
dia = te['dia']; ud = np.unique(dia); pos = {d: i for i, d in enumerate(ud)}; cid = np.array([pos[d] for d in dia])
n_c = np.bincount(cid, minlength=len(ud)).astype(float)
P42 = {k: PROBS[(k, 42)].argmax(-1).numpy() for k in MAKERS}
P42['late_fusion_heads'] = (PROBS[('text_head', 42)] + PROBS[('audio_head', 42)]).argmax(-1).numpy()
P42['zero_shot_late'] = (te['pt'] + te['pa']).argmax(-1).numpy()
rng = np.random.default_rng(0); W = rng.multinomial(len(ud), np.full(len(ud), 1 / len(ud)), size=5000).astype(float); den = W @ n_c
BOOT = {k: (W @ np.bincount(cid, weights=(v == yte).astype(float), minlength=len(ud))) / den for k, v in P42.items()}
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
SIG = {'n_test_dialogues': int(len(ud)), 'acc_ci95': {k: ci(v) for k, v in BOOT.items()}, 'differences': {}}
for a, b in [('seq_cmaf', 'late_fusion_heads'), ('seq_cmaf', 'text_head'), ('seq_cmaf', 'pooled_cmaf'), ('pooled_cmaf', 'pooled_concat'),
             ('late_fusion_heads', 'text_head'), ('seq_cmaf', 'audio_head')]:
    d = BOOT[a] - BOOT[b]
    SIG['differences'][f'{a} - {b}'] = {'acc_diff': float((P42[a] == yte).mean() - (P42[b] == yte).mean()), 'ci95': ci(d),
                                        'p_two_sided': float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
print(json.dumps(SIG, indent=1))
PER_CLASS = {k: classification_report(yte, P42[k], target_names=CLUSTER_NAMES, digits=4, output_dict=True, zero_division=0) for k in ['seq_cmaf', 'text_head', 'audio_head']}
CONF = {k: confusion_matrix(yte, P42[k], labels=range(4)).tolist() for k in ['seq_cmaf', 'text_head', 'audio_head']}
""")

md("## MUStARD: real word–tone incongruence (sarcasm)")
code("""
MJ = OUT / 'sarcasm_data.json'
if not MJ.exists():
    urllib.request.urlretrieve('https://raw.githubusercontent.com/soujanyaporia/MUStARD/master/data/sarcasm_data.json', MJ)
mus = json.load(open(MJ, encoding='utf-8'))
# Pre-specified exclusion: in the MUStARD++ audio release these 3 clips were re-cut, so the audio does not match the
# official utterance text (1_5617: different speaker and sentence; 1_7089, 2_235: shorter segment). Only true pairs are kept.
MISMATCHED = {'1_5617', '1_7089', '2_235'}
ids = [k for k in mus if (MUST_AUDIO / f'{k}_u.wav').exists() and k not in MISMATCHED]
print(f'MUStARD: {len(mus)} labelled utterances, {len(ids)} with utterance audio', flush=True)
ys = np.array([int(bool(mus[k]['sarcasm'])) for k in ids]); spk = np.array([mus[k]['speaker'] for k in ids]); show = np.array([mus[k]['show'] for k in ids])
A_ = encode_audio([MUST_AUDIO / f'{k}_u.wav' for k in ids]); T_ = encode_texts([ftfy.fix_text(mus[k]['utterance']) for k in ids])
ok = A_['ok']; print('MUStARD audio readable:', int(ok.sum()), '/', len(ids))
M = {'tok': T_['tok'][ok], 'tm': T_['tmask'][ok], 'cls': T_['emb'][ok], 'pt': T_['probs'][ok], 'fr': A_['fr'][ok], 'fm': A_['fm'][ok],
     'aemb': A_['aemb'][ok], 'pa': A_['aprobs'][ok], 'y': torch.zeros(int(ok.sum()), dtype=torch.long)}
ys, spk, show = ys[ok], spk[ok], show[ok]; ids = [i for i, o in zip(ids, ok) if o]
def jsd(p, q, eps=1e-8):
    m = 0.5 * (p + q); kl = lambda x: (x * (x.clamp_min(eps).log2() - m.clamp_min(eps).log2())).sum(-1); return 0.5 * (kl(p) + kl(q))
def ent(p): return -(p.clamp_min(1e-8) * p.clamp_min(1e-8).log()).sum(-1)
pt_h = F.softmax(predict(KEEP['text_head'], M)['fused_logits'], -1); pa_h = F.softmax(predict(KEEP['audio_head'], M)['fused_logits'], -1)
o = predict(KEEP['seq_cmaf'], M); pt_x, pa_x = F.softmax(o['text_logits'], -1), F.softmax(o['audio_logits'], -1)
FEAT = {'jsd_standalone': jsd(M['pt'], M['pa']), 'jsd_meld_heads': jsd(pt_h, pa_h), 'jsd_seq_aux': jsd(pt_x, pa_x),
        'mismatch_standalone': (M['pt'].argmax(-1) != M['pa'].argmax(-1)).float(), 'mismatch_meld_heads': (pt_h.argmax(-1) != pa_h.argmax(-1)).float(),
        'maxp_text': M['pt'].max(-1).values, 'maxp_audio': M['pa'].max(-1).values, 'ent_text': ent(M['pt']), 'ent_audio': ent(M['pa'])}
FEAT = {k: v.numpy() for k, v in FEAT.items()}
spk_u = np.unique(spk); spos = {s: i for i, s in enumerate(spk_u)}; scid = np.array([spos[s] for s in spk])
rng2 = np.random.default_rng(1)
def auc_ci(score):
    vals = []
    for _ in range(2000):
        pick = rng2.choice(len(spk_u), len(spk_u), replace=True); idx = np.concatenate([np.nonzero(scid == c)[0] for c in pick])
        if len(np.unique(ys[idx])) == 2: vals.append(roc_auc_score(ys[idx], score[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
MUS = {'n': int(len(ys)), 'n_sarcastic': int(ys.sum()), 'n_speakers': int(len(spk_u)), 'shows': {s: int((show == s).sum()) for s in np.unique(show)}, 'auc': {}}
for k in ['jsd_standalone', 'jsd_meld_heads', 'jsd_seq_aux', 'mismatch_standalone', 'mismatch_meld_heads']:
    MUS['auc'][k] = {'auc': float(roc_auc_score(ys, FEAT[k])), 'ci95_speaker_bootstrap': auc_ci(FEAT[k])}
MUS['rule_rates'] = {k: {'sarcastic_flagged': float(FEAT[k][ys == 1].mean()), 'non_sarcastic_flagged': float(FEAT[k][ys == 0].mean())}
                     for k in ['mismatch_standalone', 'mismatch_meld_heads']}
from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline; from sklearn.model_selection import GroupKFold
X = np.stack([FEAT[k] for k in ['jsd_standalone', 'jsd_meld_heads', 'jsd_seq_aux', 'maxp_text', 'maxp_audio', 'ent_text', 'ent_audio']], 1)
oof = np.zeros(len(ys))
for tr, va in GroupKFold(5).split(X, ys, spk):
    oof[va] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], ys[tr]).predict_proba(X[va])[:, 1]
MUS['auc']['probe_speaker_grouped_cv'] = {'auc': float(roc_auc_score(ys, oof)), 'ci95_speaker_bootstrap': auc_ci(oof)}
MUS['text_pred_dist'] = {c: int(n) for c, n in zip(CLUSTER_NAMES, np.bincount(M['pt'].argmax(-1).numpy(), minlength=4))}
MUS['audio_pred_dist'] = {c: int(n) for c, n in zip(CLUSTER_NAMES, np.bincount(M['pa'].argmax(-1).numpy(), minlength=4))}
print(json.dumps(MUS, indent=1))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for k_, key in enumerate(['jsd_standalone', 'jsd_meld_heads']):
    ax[k_].hist(FEAT[key][ys == 0], bins=30, alpha=.6, label='non-sarcastic'); ax[k_].hist(FEAT[key][ys == 1], bins=30, alpha=.6, label='sarcastic')
    ax[k_].set_title(f"MUStARD {key} — AUC {MUS['auc'][key]['auc']:.3f}"); ax[k_].set_xlabel('JSD'); ax[k_].legend()
plt.tight_layout(); plt.savefig(OUT / 'mustard_jsd.png', dpi=200); plt.close()
np.savez_compressed(OUT / 'mustard_arrays.npz', ys=ys, spk=spk, show=show, oof=oof, **FEAT)
np.savez_compressed(OUT / 'meld_test_arrays.npz', y=yte, dia=dia, **{f'p42_{k}': v for k, v in P42.items()},
                    pt=te['pt'].numpy(), pa=te['pa'].numpy())
""")

md("## Save + SUMMARY")
code("""
ALLR = {'meld': {'n': {s: {'total': D[s]['n_total'], 'dropped_unreadable': D[s]['n_dropped'], 'used': int(len(D[s]['y']))} for s in D},
                 'test_label_dist': {CLUSTER_NAMES[i]: int(c) for i, c in enumerate(np.bincount(yte, minlength=4))},
                 'zero_shot': ZERO, 'trained': RES, 'significance': SIG, 'per_class_seed42': PER_CLASS, 'confusion_seed42': CONF},
        'mustard': MUS, 'hours': hours()}
with open(OUT / 'results_real_pairs.json', 'w', encoding='utf-8') as f: json.dump(ALLR, f, indent=1, ensure_ascii=False)
print('===== SUMMARY START ====='); print(json.dumps(ALLR, indent=1, ensure_ascii=False)); print('===== SUMMARY END =====')
""")

for c in cells:
    if c['cell_type'] == 'code': ast.parse(c['source'])
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
for c in nb['cells']:
    s = c['source'].split('\n'); c['source'] = [l + '\n' for l in s[:-1]] + [s[-1]]
OUTD = HERE / 'real_pairs'; OUTD.mkdir(exist_ok=True); KID = 'mindsync-real-pairs'
json.dump(nb, open(OUTD / f'{KID}.ipynb', 'w', encoding='utf8'), indent=1, ensure_ascii=False)
json.dump({"id": f"{KAGGLE_USER}/{KID}", "title": "MindSync Real Pairs", "code_file": f"{KID}.ipynb", "language": "python",
           "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
           "dataset_sources": ["zaber666/meld-dataset", "kirti1210/mustard-multimodal-research"]
                              + ([os.environ['CKPT_DATASET']] if os.environ.get('CKPT_DATASET') else []),
           "competition_sources": [],
           "kernel_sources": [] if os.environ.get('CKPT_DATASET') else [f"{KAGGLE_USER}/mindsync-stage2-v3-audio"]},
          open(OUTD / 'kernel-metadata.json', 'w'), indent=1)
print('built', OUTD, len(cells), 'cells')
