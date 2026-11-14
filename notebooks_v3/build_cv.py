"""Builds two Kaggle notebooks (folds 0-3 and 4-7) for 8-fold speaker cross-validation:
per fold: Stage 2 v3 audio training (seed 42) -> audio features -> out-of-sample SeqCMAF / pooled CMAF (3 seeds)
-> JSD + learned alert -> RAVDESS same-utterance incongruence test. Per-pair arrays saved for pooled analysis."""
import json, pathlib, ast, os
KAGGLE_USER = os.environ.get('KAGGLE_USER', 'YOUR_KAGGLE_USERNAME')   # set before building to push under your account

HERE = pathlib.Path(__file__).parent
S2 = json.load(open(HERE / 'stage2_v3_audio' / 'mindsync-stage2-v3-audio.ipynb', encoding='utf8'))
S2C = [''.join(c['source']) for c in S2['cells'] if c['cell_type'] == 'code']
# S2C: 0 setup, 1 model class, 2 data download, 3 mapping/split, 4 dataset/collate, 5 config, 6 helpers, 7 loop, 8 summary
assert 'class AudioModelV3' in S2C[1] and 'def run_seed' in S2C[6] and 'class RavdessDS' in S2C[4]


def build(folds, name):
    cells = []
    md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
    code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

    md(f"""
# MindSync — 8-fold speaker cross-validation (this kernel: folds {folds})
Speaker groups G_k = actors 3k+1..3k+3 (k = 0..7). Fold k: test = G_k, validation = G_(k-1 mod 8), train = the other 18 actors.
Fold 7 reproduces the original split (test 22–24, val 19–21). Per fold: Stage 2 v3 audio (seed 42) → frame features →
out-of-sample fusion (GoEmotions-validation 80% × first two val actors; selection on the other 20% × third val actor) →
test on GoEmotions test × fold test actors → JSD, rule-based and learned alerts → RAVDESS same-utterance test.
All analysis choices below were fixed before any CV result was seen.
""")
    setup = S2C[0].replace("RESULTS_PATH = WORK / 'stage2_v3_results.json'", f"RESULTS_PATH = WORK / 'cv_results_{name}.json'")
    setup += f"\nFOLDS = {folds}\nARR_DIR = WORK / 'cv_arrays'; ARR_DIR.mkdir(exist_ok=True)\n"
    code(setup)
    code(S2C[1])
    code(S2C[2])
    code("""
CLUSTER_NAMES = ['Distress', 'Resilience', 'Aggression', 'Ambiguity']
C2I = {c: i for i, c in enumerate(CLUSTER_NAMES)}
RAVDESS_TO_CLUSTER = {1: 'Ambiguity', 2: 'Resilience', 3: 'Resilience', 4: 'Distress',
                      5: 'Aggression', 6: 'Distress', 7: 'Aggression', 8: 'Ambiguity'}
by_name = {}
for p in wavs:
    if len(p.stem.split('-')) == 7:
        by_name.setdefault(p.name, p)
items, EMO, STMT = [], [], []                     # items: (path, cluster_idx, actor)
for nm, p in sorted(by_name.items()):
    parts = p.stem.split('-')
    emo, stmt, actor = int(parts[2]), int(parts[4]), int(parts[6])
    items.append((str(p), C2I[RAVDESS_TO_CLUSTER[emo]], actor)); EMO.append(emo); STMT.append(stmt)
assert len(items) == 1440
EMO, STMT = np.array(EMO), np.array(STMT)
GROUP = lambda k: {3 * k + 1, 3 * k + 2, 3 * k + 3}
def fold_splits(k):
    te, va = GROUP(k), GROUP((k - 1) % 8)
    return {'train': [s for s in items if s[2] not in te | va], 'val': [s for s in items if s[2] in va],
            'test': [s for s in items if s[2] in te]}
for k in range(8):
    sp = fold_splits(k); A = {kk: {s[2] for s in v} for kk, v in sp.items()}
    assert [len(sp[x]) for x in ('train', 'val', 'test')] == [1080, 180, 180]
    assert not (A['train'] & A['val']) and not (A['train'] & A['test']) and not (A['val'] & A['test'])
SPLITS = fold_splits(7)                            # default (= original split) for the helper cells below
print('8 speaker-disjoint folds OK; fold 7 =', {k: sorted({s[2] for s in v}) for k, v in SPLITS.items() if k != 'train'})
""")
    code(S2C[4])
    code(S2C[5].replace('SEEDS          = [42, 43, 44]', 'SEEDS          = [42]               # one audio seed per fold (fold-to-fold spread replaces seed spread)')
         .replace("TIME_BUDGET_H  = 11.0", "TIME_BUDGET_H  = 11.3"))
    code(S2C[6])

    md("## Text features (Stage 1, fp32): GoEmotions validation + test, and the two RAVDESS transcripts")
    code("""
from transformers import RobertaModel, AutoTokenizer
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
device = DEVICE

class MindSyncTextModel(nn.Module):
    def __init__(self, model_name='roberta-large', num_classes=4, dropout=0.1):
        super().__init__()
        self.encoder = nn.Module(); self.encoder.encoder = RobertaModel.from_pretrained(model_name)
        self.encoder.hidden_size = self.encoder.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Module(); h = self.encoder.hidden_size
        self.classifier.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, h // 2), nn.GELU(),
                                                   nn.Dropout(dropout), nn.Linear(h // 2, num_classes))
sd = torch.load(hf_hub_download('Ubaida1/mindsync-text-model', 'best_text_model.pt'), map_location='cpu', weights_only=False)
sd = sd.get('model_state_dict', sd) if isinstance(sd, dict) else sd
text_model = MindSyncTextModel(); text_model.load_state_dict(sd); text_model.to(device).eval()
for p_ in text_model.parameters(): p_.requires_grad = False
del sd

EMOTION_TO_CLUSTER = {
    'grief': 'Distress', 'nervousness': 'Distress', 'fear': 'Distress', 'sadness': 'Distress', 'remorse': 'Distress',
    'joy': 'Resilience', 'admiration': 'Resilience', 'excitement': 'Resilience', 'relief': 'Resilience',
    'amusement': 'Resilience', 'approval': 'Resilience', 'curiosity': 'Resilience', 'desire': 'Resilience',
    'gratitude': 'Resilience', 'love': 'Resilience', 'optimism': 'Resilience', 'pride': 'Resilience',
    'realization': 'Resilience',
    'anger': 'Aggression', 'annoyance': 'Aggression', 'disgust': 'Aggression', 'disapproval': 'Aggression',
    'embarrassment': 'Aggression',
    'confusion': 'Ambiguity', 'disappointment': 'Ambiguity', 'surprise': 'Ambiguity', 'caring': 'Ambiguity',
    'neutral': 'Ambiguity'}
tok = AutoTokenizer.from_pretrained('roberta-large'); NTOK = 64
def encode_texts(texts):
    cls, toks, tmask = [], [], []
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i + 32], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        with torch.no_grad():
            h = text_model.encoder.encoder(input_ids=enc['input_ids'].to(device), attention_mask=enc['attention_mask'].to(device)).last_hidden_state
        cls.append(h[:, 0].float().cpu()); toks.append(h[:, :NTOK].half().cpu()); tmask.append(enc['attention_mask'][:, :NTOK].bool())
    emb = torch.cat(cls)
    with torch.no_grad():
        probs = F.softmax(text_model.classifier.classifier(emb.to(device)).float(), -1).cpu()
    return {'emb': emb, 'tok': torch.cat(toks), 'tmask': torch.cat(tmask), 'probs': probs}

ds = load_dataset('google-research-datasets/go_emotions', 'simplified')
names = ds['train'].features['labels'].feature.names
to_c = lambda ids: C2I[EMOTION_TO_CLUSTER.get(names[ids[0]], 'Ambiguity')] if ids else None
T = {}
for split in ['validation', 'test']:
    its = [(it['text'], to_c(it['labels'])) for it in ds[split]]; its = [x for x in its if x[1] is not None]
    T[split] = encode_texts([x[0] for x in its]); T[split]['cluster'] = torch.tensor([x[1] for x in its]); T[split]['texts'] = [x[0] for x in its]
    print(split, len(its), flush=True)
SENTENCES = ['Kids are talking by the door', 'Dogs are sitting by the door']      # RAVDESS statements 01 / 02
T['sent'] = encode_texts(SENTENCES); T['sent']['cluster'] = torch.tensor([C2I['Ambiguity']] * 2); T['sent']['texts'] = SENTENCES
yt_ = T['test']['cluster']; print(f"text test acc {accuracy_score(yt_, T['test']['probs'].argmax(-1)):.4f} (expect 0.7076)")
print('RAVDESS transcripts -> text posterior:', {s: [round(float(v), 3) for v in p] for s, p in zip(SENTENCES, T['sent']['probs'])})
text_model.cpu(); del text_model; torch.cuda.empty_cache()
""")

    md("## Fusion models, pair construction, metrics")
    code("""
def mpool(x, m):
    mm = m.unsqueeze(-1).float(); return (x * mm).sum(1) / mm.sum(1).clamp_min(1.0)

class PooledCMAF(nn.Module):
    def __init__(self, d=256, heads=4, dropout=0.1, aux=True):
        super().__init__(); self.aux = aux
        self.tp, self.ap = nn.Linear(1024, d), nn.Linear(1024, d)
        self.t2a = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.a2t = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.nt, self.na = nn.LayerNorm(d), nn.LayerNorm(d)
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 4))
        self.ht, self.ha = nn.Linear(d, 4), nn.Linear(d, 4)
    def forward(self, b):
        t, a = self.tp(b['cls']).unsqueeze(1), self.ap(b['aemb']).unsqueeze(1)
        ta, _ = self.t2a(t, a, a); at, _ = self.a2t(a, t, t)
        t, a = self.nt(t + ta).squeeze(1), self.na(a + at).squeeze(1)
        return {'fused_logits': self.fuse(torch.cat([t, a], -1)), 'text_logits': self.ht(t), 'audio_logits': self.ha(a)}

class SeqCMAF(nn.Module):
    def __init__(self, d=256, heads=4, dropout=0.1, aux=True):
        super().__init__(); self.aux = aux
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
        t, a = self.nt(t + ta), self.na(a + at)
        tp, ap = mpool(t, b['tm']), mpool(a, b['fm'])
        return {'fused_logits': self.fuse(torch.cat([tp, ap], -1)), 'text_logits': self.ht(tp), 'audio_logits': self.ha(ap)}

def build_pairs(tsplit, text_idx, audio_mask, seed, congruent=True):
    r = random.Random(seed)
    pool = {c: [i for i in np.nonzero(audio_mask)[0].tolist() if A_CL[i] == c] for c in range(4)}
    labels = T[tsplit]['cluster']; out = []
    for ti in text_idx:
        c = int(labels[ti]); ac = c if congruent else r.choice([k for k in range(4) if k != c])
        out.append((ti, r.choice(pool[ac]), c))
    return {'ts': tsplit, 'pairs': out}

def fetch(ps, idx):
    ts = T[ps['ts']]; pairs = ps['pairs']
    ti = torch.tensor([pairs[i][0] for i in idx]); ai = torch.tensor([pairs[i][1] for i in idx])
    return {'cls': ts['emb'][ti].to(device), 'tok': ts['tok'][ti].to(device).float(), 'tm': ts['tmask'][ti].to(device),
            'aemb': AEMB[ai].to(device), 'fr': AF[ai].to(device).float(), 'fm': AM[ai].to(device),
            'y': torch.tensor([pairs[i][2] for i in idx], device=device)}

def stream_probs(ps):
    ts = T[ps['ts']]; pairs = ps['pairs']
    ti = torch.tensor([p[0] for p in pairs]); ai = torch.tensor([p[1] for p in pairs])
    return ts['probs'][ti], APROB[ai], torch.tensor([p[2] for p in pairs])

@torch.no_grad()
def predict(model, ps, bs=256):
    model.eval(); outs = {'fused_logits': [], 'text_logits': [], 'audio_logits': []}; n = len(ps['pairs'])
    for i in range(0, n, bs):
        o = model(fetch(ps, range(i, min(i + bs, n))))
        for k in outs: outs[k].append(o[k].float().cpu())
    return {k: torch.cat(v) for k, v in outs.items()}

def train_fusion(cls_, seed, tr, va, epochs=10, lr=3e-4, alpha=0.3, bs=64):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = cls_().to(device); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    y = torch.tensor([p[2] for p in tr['pairs']]); yv = np.array([p[2] for p in va['pairs']])
    w = (1.0 / torch.bincount(y, minlength=4).float())[y]; g = torch.Generator().manual_seed(seed)
    best, best_state = -1, None
    for ep in range(epochs):
        model.train(); idx = torch.multinomial(w, len(y), replacement=True, generator=g).tolist()
        for i in range(0, len(idx), bs):
            b = fetch(tr, idx[i:i + bs]); out = model(b)
            loss = F.cross_entropy(out['fused_logits'], b['y']) + alpha * (F.cross_entropy(out['text_logits'], b['y']) + F.cross_entropy(out['audio_logits'], b['y']))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        vf1 = f1_score(yv, predict(model, va)['fused_logits'].argmax(-1).numpy(), average='macro')
        if vf1 > best: best, best_state = vf1, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); return model, float(best)

def jsd(p, q, eps=1e-8):
    m = 0.5 * (p + q); kl = lambda x: (x * (x.clamp_min(eps).log2() - m.clamp_min(eps).log2())).sum(-1)
    return 0.5 * (kl(p) + kl(q))
def ent(p): return -(p.clamp_min(1e-8) * p.clamp_min(1e-8).log()).sum(-1)
def alert_features(ps, model):
    pt, pa, _ = stream_probs(ps); o = predict(model, ps)
    aux = jsd(F.softmax(o['text_logits'], -1), F.softmax(o['audio_logits'], -1))
    X = torch.stack([jsd(pt, pa), pt.max(-1).values, pa.max(-1).values, ent(pt), ent(pa),
                     (pt.argmax(-1) != pa.argmax(-1)).float(), aux], 1).numpy()
    return X       # columns: jsd, max_pt, max_pa, ent_t, ent_a, mismatch, aux_jsd
def rates(score, thr, y):  # y: 1 = incongruent
    a = score >= thr; return {'detect': float(a[y == 1].mean()), 'false_alarm': float(a[y == 0].mean())}
def youden(score, y):
    best_t, best_j = 0.5, -9
    for t_ in np.unique(score):
        j = (score[y == 1] >= t_).mean() - (score[y == 0] >= t_).mean()
        if j > best_j: best_j, best_t = j, float(t_)
    return best_t
print('fusion/alert helpers ready', flush=True)
""")

    md("## Cross-validation loop")
    code("""
def write_cv():
    tmp = WORK / (RESULTS_PATH.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f: json.dump(CV, f, indent=1, default=_json_default)
    os.replace(tmp, RESULTS_PATH)
CV = {'folds': {}, 'errors': []}
SEEDS_FUSION = [42, 43, 44]
perm = torch.randperm(len(T['validation']['cluster']), generator=torch.Generator().manual_seed(7)).tolist()
cut = int(0.8 * len(perm)); VAL_A, VAL_B = sorted(perm[:cut]), sorted(perm[cut:])
ALL = lambda ts: list(range(len(T[ts]['cluster'])))
A_CL = np.array([s[1] for s in items]); A_ACT = np.array([s[2] for s in items])
ds_all = RavdessDS(items, False)

for fold in FOLDS:
    if hours() > TIME_BUDGET_H:
        CV['errors'].append({'fold': fold, 'error': 'time budget'}); write_cv(); continue
    t_f = time.time(); print(f'\\n================ fold {fold} ================', flush=True)
    try:
        SPLITS = fold_splits(fold)
        te_act, va_act = sorted(GROUP(fold)), sorted(GROUP((fold - 1) % 8))
        run, state = run_seed(42)
        am_ = AudioModelV3(model_name=AUDIO_MODEL_NAME, num_classes=4, freeze_layers=FREEZE_LAYERS).to(DEVICE)
        am_.load_state_dict(state); am_.eval(); del state
        frames, flen, emb_l, lg_l = [], [], [], []
        for k in range(len(items)):
            x, _ = ds_all[k]
            iv = torch.from_numpy(x)[None].to(DEVICE); ln = torch.tensor([x.shape[0]], device=DEVICE)
            with torch.no_grad(), autocast():
                f_, m_ = am_.frames(iv, ln); e_ = am_.proj(am_.pool(f_.float(), m_)); l_ = am_.head(e_)
            n_ = int(m_.sum()); frames.append(f_[0, :n_].half().cpu()); flen.append(n_)
            emb_l.append(e_[0].float().cpu()); lg_l.append(l_[0].float().cpu())
        del am_; torch.cuda.empty_cache(); gc.collect()
        TM = max(flen); AF = torch.zeros(len(items), TM, 1024, dtype=torch.float16); AM = torch.zeros(len(items), TM, dtype=torch.bool)
        for k, f_ in enumerate(frames): AF[k, :len(f_)] = f_; AM[k, :len(f_)] = True
        del frames
        AEMB = torch.stack(emb_l); APROB = F.softmax(torch.stack(lg_l), -1)
        te_mask = np.isin(A_ACT, te_act)
        a_test_acc = float((APROB[te_mask].argmax(-1).numpy() == A_CL[te_mask]).mean())
        assert abs(a_test_acc - run['test']['acc']) < 0.012, f'extraction acc {a_test_acc} vs training eval {run["test"]["acc"]}'

        TEST = build_pairs('test', ALL('test'), te_mask, 3)
        TR = build_pairs('validation', VAL_A, np.isin(A_ACT, va_act[:2]), 4)
        VA = build_pairs('validation', VAL_B, np.isin(A_ACT, va_act[2:]), 5)
        pt, pa, yt = stream_probs(TEST); late = (pt + pa).argmax(-1)
        res = {'test_actors': te_act, 'val_actors': va_act, 'audio_run': {k: run[k] for k in ('best_epoch', 'epochs_run', 'stop_reason', 'initial_loss_check')},
               'audio_val': {k: run['val'][k] for k in ('acc', 'f1')}, 'audio_test': {k: run['test'][k] for k in ('acc', 'f1', 'confusion')},
               'audio_test_report': run['test']['report'],
               'pairs_text_acc': float((pt.argmax(-1) == yt).float().mean()), 'pairs_audio_acc': float((pa.argmax(-1) == yt).float().mean()),
               'pairs_late_acc': float((late == yt).float().mean()), 'fusion': {}}
        preds = {'y': yt.numpy(), 'clip': np.array([p[1] for p in TEST['pairs']]), 'text': pt.argmax(-1).numpy(),
                 'audio': pa.argmax(-1).numpy(), 'late': late.numpy()}
        MODELS_F = {}
        for nm_, cls_ in [('seq', SeqCMAF), ('pooled', PooledCMAF)]:
            accs, f1s = [], []
            for s_ in SEEDS_FUSION:
                mdl, vf1 = train_fusion(cls_, s_, TR, VA)
                pr = predict(mdl, TEST)['fused_logits'].argmax(-1)
                accs.append(float((pr == yt).float().mean())); f1s.append(float(f1_score(yt.numpy(), pr.numpy(), average='macro')))
                preds[f'{nm_}_s{s_}'] = pr.numpy()
                if s_ == 42: MODELS_F[nm_] = mdl
                else: del mdl
            res['fusion'][nm_] = {'acc': accs, 'f1': f1s, 'acc_mean': float(np.mean(accs)), 'f1_mean': float(np.mean(f1s))}
            print(f"  fold {fold} {nm_}: acc {np.mean(accs)*100:.2f} (seeds {[round(a*100,2) for a in accs]})", flush=True)
        print(f"  fold {fold} text {res['pairs_text_acc']*100:.2f} | audio {res['pairs_audio_acc']*100:.2f} | late {res['pairs_late_acc']*100:.2f}", flush=True)

        # ---- incongruence: synthetic pairs (val for fitting, test for evaluation) ----
        seqm = MODELS_F['seq']
        va_all = np.isin(A_ACT, va_act)
        Xv = np.r_[alert_features(build_pairs('validation', ALL('validation'), va_all, 10, True), seqm),
                   alert_features(build_pairs('validation', ALL('validation'), va_all, 11, False), seqm)]
        yv = np.r_[np.zeros(len(ALL('validation'))), np.ones(len(ALL('validation')))]
        Xt = np.r_[alert_features(build_pairs('test', ALL('test'), te_mask, 20, True), seqm),
                   alert_features(build_pairs('test', ALL('test'), te_mask, 21, False), seqm)]
        ytt = np.r_[np.zeros(len(ALL('test'))), np.ones(len(ALL('test')))]
        lr_ = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xv, yv)
        sv, st = lr_.predict_proba(Xv)[:, 1], lr_.predict_proba(Xt)[:, 1]
        thr_l, thr_j = youden(sv, yv), youden(Xv[:, 0], yv)
        res['incongruence_synthetic'] = {
            'auc_jsd': float(roc_auc_score(ytt, Xt[:, 0])), 'auc_aux_jsd': float(roc_auc_score(ytt, Xt[:, 6])),
            'auc_learned': float(roc_auc_score(ytt, st)),
            'rule_mismatch_or_jsd05': rates(np.maximum(Xt[:, 5], (Xt[:, 0] >= 0.5).astype(float)), 0.5, ytt),
            'jsd_calibrated': {'thr': thr_j, **rates(Xt[:, 0], thr_j, ytt)},
            'learned_youden': {'thr': thr_l, **rates(st, thr_l, ytt)},
            'learned_coef': lr_[-1].coef_[0].tolist()}
        # ---- RAVDESS same-utterance test: real transcript + its own clip (fold test actors only) ----
        sel = np.nonzero(te_mask & np.isin(EMO, [1, 3, 4, 5, 6, 7]))[0]          # neutral = congruent; happy/sad/angry/fearful/disgust = incongruent
        SU = {'ts': 'sent', 'pairs': [(int(STMT[i]) - 1, int(i), int(A_CL[i])) for i in sel]}
        Xs = alert_features(SU, seqm); ys = (EMO[sel] != 1).astype(int)
        ss = lr_.predict_proba(Xs)[:, 1]
        res['same_utterance'] = {'n_congruent': int((ys == 0).sum()), 'n_incongruent': int((ys == 1).sum()),
            'auc_jsd': float(roc_auc_score(ys, Xs[:, 0])), 'auc_learned': float(roc_auc_score(ys, ss)),
            'rule_mismatch_or_jsd05': rates(np.maximum(Xs[:, 5], (Xs[:, 0] >= 0.5).astype(float)), 0.5, ys),
            'jsd_calibrated': rates(Xs[:, 0], thr_j, ys), 'learned_youden': rates(ss, thr_l, ys),
            'text_pred_on_transcripts': [CLUSTER_NAMES[int(i)] for i in T['sent']['probs'].argmax(-1)]}
        np.savez_compressed(ARR_DIR / f'fold{fold}.npz', **preds, Xt=Xt, yt_inc=ytt, st=st, Xs=Xs, ys=ys, ss=ss,
                            su_emo=EMO[sel], su_actor=A_ACT[sel], test_clip_actor=A_ACT, aprob=APROB.numpy())
        res['minutes'] = (time.time() - t_f) / 60
        CV['folds'][str(fold)] = res; write_cv()
        print(f"  fold {fold} JSD AUC {res['incongruence_synthetic']['auc_jsd']:.3f} | learned AUC {res['incongruence_synthetic']['auc_learned']:.3f} | "
              f"same-utterance JSD AUC {res['same_utterance']['auc_jsd']:.3f} | {res['minutes']:.0f} min", flush=True)
        del AF, AM, AEMB, APROB, MODELS_F, seqm; torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        traceback.print_exc(); CV['errors'].append({'fold': fold, 'error': repr(e)}); write_cv()
print('===== SUMMARY START ====='); print(json.dumps(CV, indent=1, default=_json_default)); print('===== SUMMARY END =====')
""")
    for c in cells:
        if c['cell_type'] == 'code':
            ast.parse('\n'.join(l for l in c['source'].split('\n') if not l.lstrip().startswith('!')))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
    for c in nb['cells']:
        s = c['source'].split('\n'); c['source'] = [l + '\n' for l in s[:-1]] + [s[-1]]
    out = HERE / f'cv_{name}'; out.mkdir(exist_ok=True)
    kid = f'mindsync-cv-{name}'
    json.dump(nb, open(out / f'{kid}.ipynb', 'w', encoding='utf8'), indent=1, ensure_ascii=False)
    json.dump({"id": f"{KAGGLE_USER}/{kid}", "title": f"MindSync CV {name}", "code_file": f"{kid}.ipynb", "language": "python",
               "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
               "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": []},
              open(out / 'kernel-metadata.json', 'w'), indent=1)
    print('built', out, len(cells), 'cells')


build([0, 1, 2, 3], 'a')
build([4, 5, 6, 7], 'b')
