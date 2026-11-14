"""Builds notebooks_v3/adapted/: encoders adapted to MELD + MUStARD with context. Pre-specified before results.
A) Fine-tune the Stage 1 text model (3 seeds) and the Stage 2 v3 audio model (seed 42) on MELD train, select on dev macro-F1,
   test once; late fusion of adapted posteriors; fusion head on adapted features (5 seeds); dialogue bootstrap.
B) MUStARD: four JSD scores (utterance text vs audio, context+utterance text vs audio, context audio vs utterance audio,
   context text vs utterance text) with frozen and MELD-adapted encoders; speaker bootstrap; speaker-grouped probe;
   sensitivity analysis without FRIENDS clips (MELD is also from Friends)."""
import json, pathlib, ast, os
KAGGLE_USER = os.environ.get('KAGGLE_USER', 'YOUR_KAGGLE_USERNAME')
HERE = pathlib.Path(__file__).parent
AUDIO_CLASS = (HERE / 'audio_model_v3.py').read_text(encoding='utf8'); AUDIO_CLASS = AUDIO_CLASS[AUDIO_CLASS.index('import torch'):]
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# MindSync — encoders adapted to MELD, and MUStARD with dialogue context (pre-specified)
**A** fine-tunes the Stage 1 text model (seeds 42–44) and the Stage 2 v3 audio model (seed 42) on MELD train (4 clusters),
selects on dev macro-F1 and tests once; reports adapted text, adapted audio, late fusion of the adapted posteriors, and a
pooled-concatenation fusion head on adapted features (5 seeds), with a bootstrap over test dialogues.
**B** scores MUStARD sarcasm with four JSD variants, using frozen (Stage 1/2) and MELD-adapted encoders; speaker bootstrap;
speaker-grouped logistic probe; sensitivity analysis excluding FRIENDS clips.
""")
code("""
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.45.2', 'huggingface_hub==0.25.2', 'ftfy'], check=True)
import os, gc, json, time, random, math, pathlib, glob, shutil, urllib.request, copy
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
T_START = time.time(); hours = lambda: (time.time() - T_START) / 3600
WORK = pathlib.Path('/kaggle/working'); OUT = WORK / 'adapted'; OUT.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda'); print(torch.__version__, torch.cuda.get_device_name(0), flush=True)
FFMPEG = shutil.which('ffmpeg')
if FFMPEG is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'imageio-ffmpeg'], check=True)
    import imageio_ffmpeg; FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
def find_one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True)); assert hits, f'not found: {pattern}'; return hits[0]
AUDIO_CKPT = find_one('/kaggle/input/**/audio_v3_best.pt')
MELD_ROOT = pathlib.Path(find_one('/kaggle/input/**/MELD.Raw/dev_sent_emo.csv')).parent
MUST_AUDIO = pathlib.Path(find_one('/kaggle/input/**/audio/*_u.wav')).parent
def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def write(name, obj):
    with open(OUT / name, 'w', encoding='utf-8') as f: json.dump(obj, f, indent=1, ensure_ascii=False)
""")
code(AUDIO_CLASS + """

from transformers import RobertaModel, AutoTokenizer, Wav2Vec2FeatureExtractor, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
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
    def forward(self, ids, mask):
        cls = self.encoder.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return cls, self.classifier.classifier(self.dropout(cls))
TEXT_SD = torch.load(hf_hub_download('Ubaida1/mindsync-text-model', 'best_text_model.pt'), map_location='cpu', weights_only=False)
TEXT_SD = TEXT_SD.get('model_state_dict', TEXT_SD) if isinstance(TEXT_SD, dict) else TEXT_SD
AUDIO_SD = torch.load(AUDIO_CKPT, map_location='cpu', weights_only=False)['model_state_dict']
tok = AutoTokenizer.from_pretrained('roberta-large'); fe = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)
CLUSTER_NAMES = ['Distress', 'Resilience', 'Aggression', 'Ambiguity']; C2I = {c: i for i, c in enumerate(CLUSTER_NAMES)}
SR, MAXS, MINS = 16_000, 80_000, 400
def new_text_model():
    m = MindSyncTextModel(); m.load_state_dict(TEXT_SD); return m.to(device)
def new_audio_model(train=False):
    m = AudioModelV3(model_name=AUDIO_MODEL_NAME, freeze_layers=8, spec_augment=train); m.load_state_dict(AUDIO_SD); return m.to(device)
""")

md("## MELD data (same exclusions as the real-pairs notebook) and decoded audio")
code("""
import ftfy, librosa
def decode_raw(path):
    try:
        if str(path).endswith('.wav'):
            w, _ = librosa.load(str(path), sr=SR, mono=True)
        else:
            r = subprocess.run([FFMPEG, '-nostdin', '-v', 'error', '-i', str(path), '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'], capture_output=True, timeout=120)
            if r.returncode != 0: return None
            w = np.frombuffer(r.stdout, np.float32).copy()
        if len(w) < MINS: return None
        w, _ = librosa.effects.trim(w, top_db=30)
        if len(w) < MINS: w = np.pad(w, (0, MINS - len(w)))
        return w.astype(np.float32)          # full trimmed clip; cropping happens later (random in training, centre in eval)
    except Exception:
        return None
def crop(w, train):
    if len(w) <= MAXS: return w
    s = random.randint(0, len(w) - MAXS) if train else (len(w) - MAXS) // 2
    return w[s:s + MAXS]
MELD_MAP = {'joy': 'Resilience', 'sadness': 'Distress', 'fear': 'Distress', 'anger': 'Aggression', 'disgust': 'Aggression',
            'neutral': 'Ambiguity', 'surprise': 'Ambiguity'}
MELD_DIRS = {'train': ['train/train_splits', 'train_splits'], 'dev': ['dev/dev_splits_complete', 'dev_splits_complete'],
             'test': ['test/output_repeated_splits_test', 'output_repeated_splits_test']}
D = {}
for split, csvname in [('train', 'train_sent_emo.csv'), ('dev', 'dev_sent_emo.csv'), ('test', 'test_sent_emo.csv')]:
    df = pd.read_csv(find_one(str(MELD_ROOT) + f'/**/{csvname}'))
    vdir = next(MELD_ROOT / d for d in MELD_DIRS[split] if (MELD_ROOT / d).exists())
    paths = [vdir / f'dia{a}_utt{b}.mp4' for a, b in zip(df.Dialogue_ID, df.Utterance_ID)]
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex: wavs = list(ex.map(decode_raw, paths))
    keep = np.array([w is not None for w in wavs]); assert keep.mean() > 0.99, (split, int((~keep).sum()))
    D[split] = {'wav': [w for w, k in zip(wavs, keep) if k], 'text': [ftfy.fix_text(str(u)) for u, k in zip(df.Utterance, keep) if k],
                'y': torch.tensor([C2I[MELD_MAP[e]] for e, k in zip(df.Emotion, keep) if k]), 'dia': df.Dialogue_ID.values[keep]}
    print(f'MELD {split}: kept {int(keep.sum())}/{len(df)} | decode {time.time()-t0:.0f}s | elapsed {hours():.2f} h', flush=True)
yte = D['test']['y'].numpy()
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
def met(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {'acc': float(accuracy_score(y, p)), 'macro_f1': float(f1_score(y, p, average='macro')), 'weighted_f1': float(f1_score(y, p, average='weighted'))}
""")

md("## A1 · Fine-tune the text model on MELD (seeds 42–44)")
code("""
def tok_all(texts):
    enc = tok(texts, padding='max_length', truncation=True, max_length=128, return_tensors='pt'); return enc['input_ids'], enc['attention_mask']
TOK = {s: tok_all(D[s]['text']) for s in D}
@torch.no_grad()
def text_infer(model, ids, mask, bs=64):
    model.eval(); E, P = [], []
    for i in range(0, len(ids), bs):
        with torch.autocast('cuda', dtype=torch.float16):
            L = int(mask[i:i + bs].sum(1).max()); cls, lg = model(ids[i:i + bs, :L].to(device), mask[i:i + bs, :L].to(device))
        E.append(cls.float().cpu()); P.append(F.softmax(lg.float(), -1).cpu())
    return torch.cat(E), torch.cat(P)
def ft_text(seed, epochs=4, lr=2e-5, bs=16):
    set_seed(seed); model = new_text_model()
    ids, mask = TOK['train']; y = D['train']['y']
    w = (1.0 / torch.bincount(y, minlength=4).float())[y]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(y) / bs); sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler('cuda'); g = torch.Generator().manual_seed(seed); best, best_state, hist = -1, None, []
    for ep in range(epochs):
        model.train(); idx = torch.multinomial(w, len(y), replacement=True, generator=g)
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            with torch.autocast('cuda', dtype=torch.float16):
                L = int(mask[b].sum(1).max()); _, lg = model(ids[b][:, :L].to(device), mask[b][:, :L].to(device))
            loss = F.cross_entropy(lg.float(), y[b].to(device))
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
        _, pd_ = text_infer(model, *TOK['dev']); vf1 = f1_score(D['dev']['y'].numpy(), pd_.argmax(-1).numpy(), average='macro')
        hist.append(float(vf1)); print(f'  text seed {seed} epoch {ep+1}: dev macro-F1 {vf1:.4f} | {hours():.2f} h', flush=True)
        if vf1 > best: best, best_state = vf1, {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); return model, float(best), hist
TXT = {}
for s in [42, 43, 44]:
    m_, vf1, hist = ft_text(s)
    e_te, p_te = text_infer(m_, *TOK['test'])
    TXT[s] = {'dev_macro_f1': vf1, 'dev_hist': hist, 'test': met(yte, p_te.argmax(-1)), 'p_test': p_te}
    if s == 42:
        E_T = {sp: text_infer(m_, *TOK[sp])[0] for sp in ['train', 'dev']}; E_T['test'] = e_te
        TEXT_FT = m_.cpu()
    del m_; torch.cuda.empty_cache(); gc.collect()
    print(f"text seed {s}: test {TXT[s]['test']}", flush=True)
write('partial_text.json', {s: {k: v for k, v in r.items() if k != 'p_test'} for s, r in TXT.items()})
""")

md("## A2 · Fine-tune the audio model on MELD (seed 42; one clip per step, accumulation 16)")
code("""
def augment(w):
    speed = random.uniform(0.9, 1.1); w = librosa.resample(w, orig_sr=SR, target_sr=int(round(SR / speed)))
    w = w * random.uniform(0.8, 1.2)
    if random.random() < 0.5:
        rms = float(np.sqrt(np.mean(w ** 2))) + 1e-9; w = w + np.random.normal(0.0, rms / 10 ** (random.uniform(15, 30) / 20), len(w))
    return w.astype(np.float32)
def to_input(w):
    x = fe(w, sampling_rate=SR, return_tensors='np')['input_values'][0].astype(np.float32)
    return torch.from_numpy(x)[None].to(device), torch.tensor([x.shape[0]], device=device)
@torch.no_grad()
def audio_infer(model, wavs, tail=False):
    model.eval(); E, P = [], []
    for w in wavs:
        iv, ln = to_input(w[-MAXS:] if tail else crop(w, False))
        with torch.autocast('cuda', dtype=torch.float16):
            o = model(iv, ln)
        E.append(o['embedding'][0].float().cpu()); P.append(F.softmax(o['logits'][0].float(), -1).cpu())
    return torch.stack(E), torch.stack(P)
def ft_audio(seed, max_epochs=4, patience=2, accum=16):
    set_seed(seed); model = new_audio_model(train=True)
    new = [model.layer_w, *model.layer_norm.parameters(), *model.proj.parameters(), *model.head.parameters()]
    newid = {id(p) for p in new}; back = [p for p in model.enc.parameters() if p.requires_grad and id(p) not in newid]
    opt = torch.optim.AdamW([{'params': back, 'lr': 1e-5}, {'params': new, 'lr': 1e-4}], weight_decay=0.01)
    y = D['train']['y']; w = (1.0 / torch.bincount(y, minlength=4).float())[y]; n = len(y)
    steps = max_epochs * math.ceil(n / accum); sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler('cuda'); crit = nn.CrossEntropyLoss(label_smoothing=0.05); g = torch.Generator().manual_seed(seed)
    best, best_state, bad, hist = -1, None, 0, []
    trainable = [p for p in model.parameters() if p.requires_grad]
    for ep in range(max_epochs):
        if hours() > 9.5: print('time budget: stop audio fine-tuning', flush=True); break
        model.train(); idx = torch.multinomial(w, n, replacement=True, generator=g).tolist(); opt.zero_grad(set_to_none=True); t0 = time.time()
        for i, k in enumerate(idx):
            iv, ln = to_input(crop(augment(D['train']['wav'][k]), True))
            # SpecAugment needs more frames than its mask length; very short MELD clips (< 20 frames, ~0.4 s) skip it
            model.enc.config.apply_spec_augment = int(model.enc._get_feat_extract_output_lengths(ln)[0]) >= 20
            with torch.autocast('cuda', dtype=torch.float16):
                lg = model(iv, ln)['logits']
            loss = crit(lg.float(), y[k:k + 1].to(device)) / accum; scaler.scale(loss).backward()
            if (i + 1) % accum == 0 or i + 1 == n:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(trainable, 1.0); scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True); sched.step()
            if (i + 1) % 2000 == 0: print(f'    audio ep {ep+1} step {i+1}/{n} | {time.time()-t0:.0f}s', flush=True)
        _, pd_ = audio_infer(model, D['dev']['wav']); vf1 = f1_score(D['dev']['y'].numpy(), pd_.argmax(-1).numpy(), average='macro')
        hist.append(float(vf1)); print(f'  audio epoch {ep+1}: dev macro-F1 {vf1:.4f} | {time.time()-t0:.0f}s | {hours():.2f} h', flush=True)
        if vf1 > best: best, best_state, bad = vf1, {k_: v.detach().to('cpu', copy=True) for k_, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state); return model, float(best), hist
am_ft, a_vf1, a_hist = ft_audio(42)
E_A, P_A = {}, {}
for sp in ['train', 'dev', 'test']: E_A[sp], P_A[sp] = audio_infer(am_ft, D[sp]['wav'])
AUD = {'dev_macro_f1': a_vf1, 'dev_hist': a_hist, 'test': met(yte, P_A['test'].argmax(-1))}
print('adapted audio:', AUD, flush=True); write('partial_audio.json', AUD)
""")

md("## A3 · Late fusion of adapted posteriors and a fusion head on adapted features; dialogue bootstrap")
code("""
LATE = {s: met(yte, (TXT[s]['p_test'] + P_A['test']).argmax(-1)) for s in TXT}
class ConcatHead(nn.Module):
    def __init__(self, d=256, dropout=0.1):
        super().__init__(); self.tp, self.ap = nn.Linear(1024, d), nn.Linear(1024, d)
        self.net = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 4))
    def forward(self, t, a): return self.net(torch.cat([self.tp(t), self.ap(a)], -1))
# The encoders were fine-tuned on MELD train, so their train features are over-confident; the head is therefore trained on
# DEV features (never used for encoder training), with epoch selection on 20% of dev dialogues held out (fixed split, seed 7).
_dd = np.unique(D['dev']['dia']); _r = np.random.default_rng(7); _hold = set(_r.choice(_dd, int(0.2 * len(_dd)), replace=False).tolist())
DEV_TR = torch.tensor([i for i, d_ in enumerate(D['dev']['dia']) if d_ not in _hold]); DEV_SEL = torch.tensor([i for i, d_ in enumerate(D['dev']['dia']) if d_ in _hold])
print('head training on dev:', len(DEV_TR), 'utterances | selection:', len(DEV_SEL), flush=True)
def train_head(seed, epochs=30, lr=3e-4, bs=32):
    set_seed(seed); m = ConcatHead().to(device); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    y = D['dev']['y'][DEV_TR]; ET, EA = E_T['dev'][DEV_TR], E_A['dev'][DEV_TR]
    w = (1.0 / torch.bincount(y, minlength=4).float().clamp_min(1))[y]; g = torch.Generator().manual_seed(seed); best, bs_ = -1, None
    for ep in range(epochs):
        m.train(); idx = torch.multinomial(w, len(y), replacement=True, generator=g)
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]; loss = F.cross_entropy(m(ET[b].to(device), EA[b].to(device)), y[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): pv = m(E_T['dev'][DEV_SEL].to(device), E_A['dev'][DEV_SEL].to(device)).argmax(-1).cpu().numpy()
        vf1 = f1_score(D['dev']['y'][DEV_SEL].numpy(), pv, average='macro')
        if vf1 > best: best, bs_ = vf1, copy.deepcopy(m.state_dict())
    m.load_state_dict(bs_); m.eval()
    with torch.no_grad(): return F.softmax(m(E_T['test'].to(device), E_A['test'].to(device)), -1).cpu()
HEAD = {s: train_head(s) for s in [42, 43, 44, 45, 46]}
def agg(d):
    ks = ['acc', 'macro_f1', 'weighted_f1']
    return {**{f'{k}_mean': float(np.mean([v[k] for v in d.values()])) for k in ks}, **{f'{k}_std': float(np.std([v[k] for v in d.values()], ddof=1)) if len(d) > 1 else 0.0 for k in ks}, 'runs': d}
RES_A = {'adapted_text': agg({s: TXT[s]['test'] for s in TXT}), 'adapted_audio_seed42': AUD['test'],
         'late_fusion_adapted': agg(LATE), 'concat_head_adapted': agg({s: met(yte, p.argmax(-1)) for s, p in HEAD.items()})}
for k, v in RES_A.items(): print(k, {kk: (round(vv, 4) if isinstance(vv, float) else '') for kk, vv in v.items() if kk != 'runs'})
dia = D['test']['dia']; ud = np.unique(dia); pos = {d_: i for i, d_ in enumerate(ud)}; cid = np.array([pos[d_] for d_ in dia]); n_c = np.bincount(cid).astype(float)
P42 = {'adapted_text': TXT[42]['p_test'].argmax(-1).numpy(), 'adapted_audio': P_A['test'].argmax(-1).numpy(),
       'late_fusion_adapted': (TXT[42]['p_test'] + P_A['test']).argmax(-1).numpy(), 'concat_head_adapted': HEAD[42].argmax(-1).numpy()}
rng = np.random.default_rng(0); W = rng.multinomial(len(ud), np.full(len(ud), 1 / len(ud)), size=5000).astype(float); den = W @ n_c
BOOT = {k: (W @ np.bincount(cid, weights=(v == yte).astype(float), minlength=len(ud))) / den for k, v in P42.items()}
def mf1_boot(pred, B=1000):
    r = np.random.default_rng(1); vals = []
    for _ in range(B):
        pick = r.choice(len(ud), len(ud), replace=True); idx = np.concatenate([np.nonzero(cid == c)[0] for c in pick])
        vals.append(f1_score(yte[idx], pred[idx], average='macro'))
    return vals
MF1 = {k: np.array(mf1_boot(v)) for k, v in P42.items()}
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
SIG_A = {'acc_ci95': {k: ci(v) for k, v in BOOT.items()}, 'macro_f1_ci95': {k: ci(v) for k, v in MF1.items()}, 'differences': {}}
for a, b in [('late_fusion_adapted', 'adapted_text'), ('concat_head_adapted', 'adapted_text'), ('late_fusion_adapted', 'concat_head_adapted')]:
    SIG_A['differences'][f'{a} - {b}'] = {'acc_ci95': ci(BOOT[a] - BOOT[b]), 'macro_f1_diff': float(f1_score(yte, P42[a], average='macro') - f1_score(yte, P42[b], average='macro')),
                                          'macro_f1_ci95': ci(MF1[a] - MF1[b])}
print(json.dumps(SIG_A, indent=1))
PER_CLASS_A = {k: classification_report(yte, v, target_names=CLUSTER_NAMES, digits=4, output_dict=True, zero_division=0) for k, v in P42.items()}
write('partial_A.json', {'results': RES_A, 'significance': SIG_A})
""")

md("## B · MUStARD with dialogue context, frozen vs MELD-adapted encoders")
code("""
tok.truncation_side = 'left'   # context+utterance: keep the utterance and the context closest to it
MJ = OUT / 'sarcasm_data.json'
if not MJ.exists(): urllib.request.urlretrieve('https://raw.githubusercontent.com/soujanyaporia/MUStARD/master/data/sarcasm_data.json', MJ)
mus = json.load(open(MJ, encoding='utf-8'))
MISMATCHED = {'1_5617', '1_7089', '2_235'}
ids = [k for k in mus if k not in MISMATCHED and (MUST_AUDIO / f'{k}_u.wav').exists() and (MUST_AUDIO / f'{k}_c.wav').exists()]
with ThreadPoolExecutor(8) as ex:
    wu = list(ex.map(decode_raw, [MUST_AUDIO / f'{k}_u.wav' for k in ids])); wc = list(ex.map(decode_raw, [MUST_AUDIO / f'{k}_c.wav' for k in ids]))
ok = [a is not None and b is not None for a, b in zip(wu, wc)]
ids = [k for k, o in zip(ids, ok) if o]; wu = [w for w, o in zip(wu, ok) if o]; wc = [w for w, o in zip(wc, ok) if o]
ys = np.array([int(bool(mus[k]['sarcasm'])) for k in ids]); spk = np.array([mus[k]['speaker'] for k in ids]); show = np.array([mus[k]['show'] for k in ids])
utt = [ftfy.fix_text(mus[k]['utterance']) for k in ids]; ctx = [ftfy.fix_text(' '.join(mus[k]['context'])) for k in ids]
both = [c + ' ' + u for c, u in zip(ctx, utt)]
print(f'MUStARD: {len(ids)} utterances with utterance and context audio, {int(ys.sum())} sarcastic', flush=True)
def jsd(p, q, eps=1e-8):
    m = 0.5 * (p + q); kl = lambda x: (x * (x.clamp_min(eps).log2() - m.clamp_min(eps).log2())).sum(-1); return 0.5 * (kl(p) + kl(q))
def text_probs(model, texts):
    ids_, mask_ = tok_all(texts); return text_infer(model, ids_, mask_)[1]
FEATS = {}
for enc_name in ['frozen', 'adapted']:
    tm = new_text_model() if enc_name == 'frozen' else TEXT_FT.to(device)
    amod = new_audio_model() if enc_name == 'frozen' else am_ft
    pu, pc, pb = text_probs(tm, utt), text_probs(tm, ctx), text_probs(tm, both)
    au, ac = audio_infer(amod, wu)[1], audio_infer(amod, wc, tail=True)[1]   # context audio: last 5 s, the turn just before the utterance
    FEATS[enc_name] = {'jsd_utt_text_vs_utt_audio': jsd(pu, au), 'jsd_ctxutt_text_vs_utt_audio': jsd(pb, au),
                       'jsd_ctx_audio_vs_utt_audio': jsd(ac, au), 'jsd_ctx_text_vs_utt_text': jsd(pc, pu),
                       'maxp_text': pu.max(-1).values, 'maxp_audio': au.max(-1).values}
    FEATS[enc_name] = {k: v.numpy() for k, v in FEATS[enc_name].items()}
    if enc_name == 'frozen': del tm, amod
    torch.cuda.empty_cache(); gc.collect()
spk_u = np.unique(spk)
def auc_ci(score, sel, B=2000):
    r = np.random.default_rng(2); su = np.unique(spk[sel]); vals = []
    for _ in range(B):
        pick = r.choice(su, len(su), replace=True); idx = np.concatenate([np.nonzero(sel & (spk == s_))[0] for s_ in pick])
        if len(np.unique(ys[idx])) == 2: vals.append(roc_auc_score(ys[idx], score[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline; from sklearn.model_selection import GroupKFold
MUS = {'n': int(len(ys)), 'n_sarcastic': int(ys.sum()), 'shows': {s_: int((show == s_).sum()) for s_ in np.unique(show)}, 'subsets': {}}
for sub_name, sel in [('all', np.ones(len(ys), bool)), ('without_FRIENDS', show != 'FRIENDS')]:
    R_ = {}
    for enc_name, Fs in FEATS.items():
        for k, v in Fs.items():
            if k.startswith('jsd'): R_[f'{enc_name}/{k}'] = {'auc': float(roc_auc_score(ys[sel], v[sel])), 'ci95': auc_ci(v, sel)}
        X = np.stack([v for v in Fs.values()], 1)[sel]; oof = np.zeros(int(sel.sum()))
        for tr, va in GroupKFold(5).split(X, ys[sel], spk[sel]):
            oof[va] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], ys[sel][tr]).predict_proba(X[va])[:, 1]
        full = np.zeros(len(ys)); full[sel] = oof
        R_[f'{enc_name}/probe_speaker_grouped_cv'] = {'auc': float(roc_auc_score(ys[sel], oof)), 'ci95': auc_ci(full, sel)}
    MUS['subsets'][sub_name] = {'n': int(sel.sum()), 'n_sarcastic': int(ys[sel].sum()), 'auc': R_}
print(json.dumps(MUS, indent=1))
np.savez_compressed(OUT / 'mustard_context_arrays.npz', ys=ys, spk=spk, show=show, **{f'{e}__{k}': v for e, Fs in FEATS.items() for k, v in Fs.items()})
""")
md("## Save + SUMMARY")
code("""
ALLR = {'meld_adapted': {'results': RES_A, 'significance': SIG_A, 'per_class_seed42': PER_CLASS_A,
                         'text_dev': {s: {'best': TXT[s]['dev_macro_f1'], 'hist': TXT[s]['dev_hist']} for s in TXT},
                         'audio_dev': {'best': AUD['dev_macro_f1'], 'hist': AUD['dev_hist']},
                         'n': {s: int(len(D[s]['y'])) for s in D}},
        'mustard_context': MUS, 'hours': hours()}
write('results_adapted.json', ALLR)
print('===== SUMMARY START ====='); print(json.dumps(ALLR, indent=1, ensure_ascii=False)); print('===== SUMMARY END =====')
""")

for c in cells:
    if c['cell_type'] == 'code': ast.parse(c['source'])
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
for c in nb['cells']:
    s = c['source'].split('\n'); c['source'] = [l + '\n' for l in s[:-1]] + [s[-1]]
OUTD = HERE / 'adapted'; OUTD.mkdir(exist_ok=True); KID = 'mindsync-adapted'
json.dump(nb, open(OUTD / f'{KID}.ipynb', 'w', encoding='utf8'), indent=1, ensure_ascii=False)
json.dump({"id": f"{KAGGLE_USER}/{KID}", "title": "MindSync Adapted", "code_file": f"{KID}.ipynb", "language": "python",
           "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
           "dataset_sources": ["zaber666/meld-dataset", "kirti1210/mustard-multimodal-research"] + ([os.environ['CKPT_DATASET']] if os.environ.get('CKPT_DATASET') else []),
           "competition_sources": [], "kernel_sources": [] if os.environ.get('CKPT_DATASET') else [f"{KAGGLE_USER}/mindsync-stage2-v3-audio"]},
          open(OUTD / 'kernel-metadata.json', 'w'), indent=1)
print('built', OUTD, len(cells), 'cells')
