"""
Test 5 — multi-seed robustness pass over Tests 1–4.

Every earlier test was single seed. Here each test is re-run across several seeds
(default 5) and reported as mean ± std, so the numbers are formal, not a one-off.
Faithfully reuses the same models/tasks as 01–04. CPU, single thread — this is a
long run; launch it in the background. Set MS_QUICK=1 for a fast smoke test.

Outputs: multiseed_results.json + one error-bar figure per test (05_test{1..4}.png).
"""
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, math
import matplotlib.pyplot as plt
torch.set_num_threads(4)
QUICK = os.environ.get("MS_QUICK") == "1"
N_SEEDS = 1 if QUICK else 5
NTRAIN  = 400 if QUICK else 2500
NTEST   = 80  if QUICK else 300
EP = (lambda n: max(2, n//12)) if QUICK else (lambda n: n)   # epoch scaler for smoke test

# ── tasks ────────────────────────────────────────────────────────────────────
def bbs_step(s):
    out = np.zeros_like(s); c = 0
    for i in range(len(s)):
        if s[i] == 1: c += 1; out[i] = 0
        elif c > 0:   out[i] = 1; c -= 1
    return out
def fc_step(s, K):
    out = np.zeros_like(s); u = 0
    for i in range(len(s)):
        if s[i] == 1:
            if u < K: u += 1; out[i] = 0
            else:     out[i] = 1
        else:
            if u > 0: out[i] = 1; u -= 1
            else:     out[i] = 0
    return out
def run(s, T, kind, K):
    s = s.copy()
    for _ in range(T): s = bbs_step(s) if kind == "bbs" else fc_step(s, K)
    return s
def cfg_std(L, rng):                       # Tests 1–3
    s = np.zeros(L, dtype=np.int64); usable = int(L*0.55); i = 0; n = 0
    while i < usable - 4:
        if rng.random() < 0.5 and n < max(3, L//7):
            sz = rng.integers(1, 4); s[i:i+sz] = 1; i += sz + rng.integers(2, 5); n += 1
        else: i += 1
    if s.sum() == 0: s[1:3] = 1
    return s
def cfg_dense(L, rng):                      # Test 4 (bigger/denser, MAXSOL=6)
    s = np.zeros(L, dtype=np.int64); usable = int(L*0.50); i = 0
    while i < usable - 7:
        if rng.random() < 0.6:
            sz = rng.integers(1, 7); s[i:i+sz] = 1; i += sz + rng.integers(1, 4)
        else: i += 1
    if s.sum() == 0: s[1:3] = 1
    return s
def data(L, N, rng, T, kind, K, cfg):
    X = np.zeros((N, L), np.int64); Y = np.zeros((N, L), np.int64)
    for j in range(N):
        x = cfg(L, rng); X[j] = x; Y[j] = run(x, T, kind, K)
    return torch.tensor(X), torch.tensor(Y)

# ── models (copied verbatim from 01–04) ──────────────────────────────────────
class PE(nn.Module):
    def __init__(s, d, maxlen=8192):
        super().__init__(); pe = torch.zeros(maxlen, d); pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float()*(-math.log(10000.0)/d))
        pe[:, 0::2] = torch.sin(pos*div); pe[:, 1::2] = torch.cos(pos*div); s.register_buffer("pe", pe)
    def forward(s, x): return x + s.pe[:x.size(1)].unsqueeze(0)
class TF(nn.Module):
    def __init__(s, d=64, heads=4, layers=3):
        super().__init__(); s.emb = nn.Embedding(2, d); s.pe = PE(d)
        enc = nn.TransformerEncoderLayer(d, heads, 4*d, batch_first=True, dropout=0.0, activation="gelu")
        s.enc = nn.TransformerEncoder(enc, layers); s.head = nn.Linear(d, 2)
    def forward(s, x): return s.head(s.enc(s.pe(s.emb(x))))
class LearnedCarrier(nn.Module):           # Test 1/2 plain carrier
    def __init__(s, H=24, hid=64):
        super().__init__(); s.H = H
        s.net = nn.Sequential(nn.Linear(1+H, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU())
        s.out = nn.Linear(hid, 1); s.hnext = nn.Linear(hid, H); s.T = 2
    def one(s, c):
        B, L = c.shape; h = torch.zeros(B, s.H); outs = []
        for i in range(L):
            z = s.net(torch.cat([c[:, i:i+1], h], 1)); outs.append(s.out(z)); h = torch.tanh(s.hnext(z))
        return torch.sigmoid(torch.cat(outs, 1))
    def forward(s, x):
        c = x.float()
        for _ in range(s.T): c = s.one(c)
        return c
class ReversibleConservativeCA(nn.Module): # Test 2 swap automaton
    def __init__(s, n_layers=16, hid=48):
        super().__init__()
        def mk(): return nn.Sequential(nn.Linear(5, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
        s.gates = nn.ModuleList([mk() for _ in range(n_layers)]); s.temp = 1.0
        cfgs = [(0, 0), (0, 1), (1, 0), (1, 1)]; s.configs = [cfgs[i % 4] for i in range(n_layers)]
    def _layer(s, state, o, parity, gate, hard):
        B, L = state.shape; K = (L - o)//2
        if K == 0: return state
        left = o + 2*torch.arange(K); kk = torch.arange(K); act = (kk % 2 == parity)
        if act.sum() == 0: return state
        la = left[act]; ra = la + 1; sp = F.pad(state, (2, 2))
        lval = state[:, la]; rval = state[:, ra]; own = lval + rval
        feat = torch.stack([own, sp[:, la], sp[:, la+1], sp[:, ra+3], sp[:, ra+4]], -1)
        a = torch.sigmoid(gate(feat).squeeze(-1) / s.temp)
        if hard: ah = (a > 0.5).float(); a = ah + (a - a.detach())
        out = state.clone(); out[:, la] = lval + a*(rval-lval); out[:, ra] = rval + a*(lval-rval)
        return out
    def forward(s, x, hard=False):
        state = x.float()
        for i, (o, p) in enumerate(s.configs): state = s._layer(state, o, p, s.gates[i], hard)
        return state
class ConservingCarrier(nn.Module):        # Test 3/4 (blind=True → fixed emit=1-cell)
    def __init__(s, hid=48, Kmax=8.0, blind=False):
        super().__init__(); s.Kmax = Kmax; s.blind = blind; s.T = 2
        s.gate = nn.Sequential(nn.Linear(4, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
    def one(s, xf, hard):
        B, L = xf.shape; k = torch.zeros(B); xp = F.pad(xf, (1, 1)); outs = []
        for i in range(L):
            c = xf[:, i]
            if s.blind: e = 1.0 - c
            else:
                e = torch.sigmoid(s.gate(torch.stack([c, k/s.Kmax, xp[:, i], xp[:, i+2]], -1)).squeeze(-1))
                if hard: eh = (e > 0.5).float(); e = eh + (e - e.detach())
            t = c + k; o = e*torch.clamp(t, 0, 1); k = t - o; outs.append(o)
        return torch.stack(outs, 1)
    def forward(s, x, hard=False):
        c = x.float()
        for _ in range(s.T): c = s.one(c, hard)
        return c
class GRUCarrier(nn.Module):               # Test 4 composing GRU
    def __init__(s, d=96, layers=1):
        super().__init__(); s.gru = nn.GRU(1, d, layers, batch_first=True); s.head = nn.Linear(d, 1); s.T = 2
    def one(s, c): h, _ = s.gru(c.unsqueeze(-1)); return torch.sigmoid(s.head(h)).squeeze(-1)
    def forward(s, x):
        c = x.float()
        for _ in range(s.T): c = s.one(c)
        return c
class LSTMNet(nn.Module):
    def __init__(s, d=96, layers=2):
        super().__init__(); s.emb = nn.Embedding(2, d); s.lstm = nn.LSTM(d, d, layers, batch_first=True); s.head = nn.Linear(d, 2)
    def forward(s, x): y, _ = s.lstm(s.emb(x)); return s.head(y)
class SSMLayer(nn.Module):
    def __init__(s, d):
        super().__init__(); s.dec = nn.Parameter(torch.randn(d)*0.5); s.inp = nn.Linear(d, d); s.out = nn.Linear(d, d); s.norm = nn.LayerNorm(d)
    def forward(s, x):
        a = torch.sigmoid(s.dec); u = s.inp(x); B, L, d = x.shape; st = torch.zeros(B, d); outs = []
        for t in range(L): st = a*st + (1-a)*u[:, t]; outs.append(st)
        return s.norm(x + s.out(F.gelu(torch.stack(outs, 1))))
class SSMNet(nn.Module):
    def __init__(s, d=96, layers=2):
        super().__init__(); s.emb = nn.Embedding(2, d); s.layers = nn.ModuleList([SSMLayer(d) for _ in range(layers)]); s.head = nn.Linear(d, 2)
    def forward(s, x):
        h = s.emb(x)
        for l in s.layers: h = l(h)
        return s.head(h)

# ── training / eval helpers ──────────────────────────────────────────────────
def prob(m, X):
    if isinstance(m, (LearnedCarrier, GRUCarrier)): return m(X)
    if isinstance(m, ConservingCarrier):            return m(X, hard=True)
    if isinstance(m, ReversibleConservativeCA):     return m(X, hard=True)
    return m(X).softmax(-1)[..., 1]
def is_prob(m): return isinstance(m, (LearnedCarrier, GRUCarrier, ConservingCarrier, ReversibleConservativeCA))
def fit(m, Xtr, Ytr, epochs, lr, clip=False, anneal=False):
    opt = torch.optim.AdamW(m.parameters(), lr=lr); ce = nn.CrossEntropyLoss(); bce = nn.BCELoss(); B = 256
    logits = not is_prob(m)
    for ep in range(epochs):
        if anneal: m.temp = 1.0*(0.12)**(ep/max(1, epochs-1))
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), B):
            idx = perm[i:i+B]
            if logits: loss = ce(m(Xtr[idx]).reshape(-1, 2), Ytr[idx].reshape(-1))
            else:      loss = bce((m(Xtr[idx], hard=False) if isinstance(m, (ConservingCarrier, ReversibleConservativeCA)) else m(Xtr[idx])).clamp(1e-6, 1-1e-6), Ytr[idx].float())
            opt.zero_grad(); loss.backward()
            if clip: torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
    return m
def evaluate(m, lengths, rng, T, kind, K, cfg):
    m.eval(); out = {}
    with torch.no_grad():
        for L in lengths:
            Xte, Yte = data(L, NTEST, rng, T, kind, K, cfg)
            pred = (prob(m, Xte) > 0.5).long()
            acc = (pred == Yte).float().mean().item()*100
            cons = (pred.sum(1) == Xte.sum(1)).float().mean().item()*100
            out[L] = (acc, cons)
    return out

# ── the four tests (one seed each) ───────────────────────────────────────────
LEN123 = [32, 48, 64, 96, 128]; LEN4 = [32, 64, 128, 256]
def seed_all(sd): torch.manual_seed(sd); np.random.seed(sd)

def test1(sd):
    seed_all(sd); rng = np.random.default_rng(sd); Xtr, Ytr = data(32, NTRAIN, rng, 2, "bbs", 0, cfg_std)
    r = {}
    r["conserving? no · plain carrier"] = evaluate(fit(LearnedCarrier(), Xtr, Ytr, EP(40), 3e-3), LEN123, rng, 2, "bbs", 0, cfg_std)
    r["Transformer"] = evaluate(fit(TF(), Xtr, Ytr, EP(28), 1e-3), LEN123, rng, 2, "bbs", 0, cfg_std)
    return r
def test2(sd):
    seed_all(sd); rng = np.random.default_rng(sd); Xtr, Ytr = data(32, NTRAIN, rng, 2, "bbs", 0, cfg_std)
    r = {}
    r["swap-automaton"] = evaluate(fit(ReversibleConservativeCA(), Xtr, Ytr, EP(140), 3e-3, anneal=True), LEN123, rng, 2, "bbs", 0, cfg_std)
    r["plain carrier"]  = evaluate(fit(LearnedCarrier(), Xtr, Ytr, EP(40), 3e-3), LEN123, rng, 2, "bbs", 0, cfg_std)
    r["Transformer"]    = evaluate(fit(TF(), Xtr, Ytr, EP(28), 1e-3), LEN123, rng, 2, "bbs", 0, cfg_std)
    return r
def test3(sd):
    seed_all(sd); rng = np.random.default_rng(sd); Xtr, Ytr = data(32, NTRAIN, rng, 2, "fc", 2, cfg_std)
    r = {}
    r["conserving carrier"] = evaluate(fit(ConservingCarrier(), Xtr, Ytr, EP(40), 3e-3), LEN123, rng, 2, "fc", 2, cfg_std)
    r["carrier-blind"]      = evaluate(ConservingCarrier(blind=True), LEN123, rng, 2, "fc", 2, cfg_std)
    r["Transformer"]        = evaluate(fit(TF(), Xtr, Ytr, EP(28), 1e-3), LEN123, rng, 2, "fc", 2, cfg_std)
    return r
def test4(sd):
    seed_all(sd); rng = np.random.default_rng(sd); Xtr, Ytr = data(32, NTRAIN, rng, 2, "fc", 6, cfg_dense)
    r = {}
    r["conserving carrier"]    = evaluate(fit(ConservingCarrier(), Xtr, Ytr, EP(45), 3e-3), LEN4, rng, 2, "fc", 6, cfg_dense)
    r["GRU carrier (compose)"] = evaluate(fit(GRUCarrier(), Xtr, Ytr, EP(50), 2e-3, clip=True), LEN4, rng, 2, "fc", 6, cfg_dense)
    r["LSTM"]                  = evaluate(fit(LSTMNet(), Xtr, Ytr, EP(45), 2e-3, clip=True), LEN4, rng, 2, "fc", 6, cfg_dense)
    r["SSM (Mamba)"]           = evaluate(fit(SSMNet(), Xtr, Ytr, EP(45), 2e-3, clip=True), LEN4, rng, 2, "fc", 6, cfg_dense)
    r["Transformer"]           = evaluate(fit(TF(), Xtr, Ytr, EP(30), 1e-3), LEN4, rng, 2, "fc", 6, cfg_dense)
    return r

TESTS = [("Test 1", test1, LEN123), ("Test 2", test2, LEN123), ("Test 3", test3, LEN123), ("Test 4", test4, LEN4)]
allres = {}
for tname, fn, lengths in TESTS:
    print(f"\n===== {tname} — {N_SEEDS} seeds =====", flush=True)
    per = [fn(sd) for sd in range(N_SEEDS)]
    models = list(per[0].keys()); agg = {}
    for mo in models:
        agg[mo] = {}
        for L in lengths:
            accs = [per[s][mo][L][0] for s in range(N_SEEDS)]; cons = [per[s][mo][L][1] for s in range(N_SEEDS)]
            agg[mo][L] = (float(np.mean(accs)), float(np.std(accs)), float(np.mean(cons)), float(np.std(cons)))
        print(f"  {mo:26s} " + "  ".join(f"L{L}:{agg[mo][L][0]:4.0f}±{agg[mo][L][1]:3.0f}/{agg[mo][L][2]:4.0f}±{agg[mo][L][3]:3.0f}" for L in lengths), flush=True)
    allres[tname] = {"lengths": lengths, "agg": agg}
    # error-bar figure (accuracy + conservation)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for mo in models:
        am = [agg[mo][L][0] for L in lengths]; asd = [agg[mo][L][1] for L in lengths]
        cm = [agg[mo][L][2] for L in lengths]; csd = [agg[mo][L][3] for L in lengths]
        ax[0].errorbar(lengths, am, yerr=asd, marker="o", capsize=3, label=mo)
        ax[1].errorbar(lengths, cm, yerr=csd, marker="o", capsize=3, label=mo)
    ax[0].set_title(f"{tname} · accuracy (mean±std, {N_SEEDS} seeds)"); ax[0].set_ylabel("acc %"); ax[0].set_ylim(40, 103)
    ax[1].set_title(f"{tname} · conservation"); ax[1].set_ylabel("cons %"); ax[1].set_ylim(-3, 105)
    for a in ax: a.set_xlabel("length"); a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"05_{tname.replace(' ', '').lower()}.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    json.dump(allres, open("multiseed_results.json", "w"), indent=1)   # save incrementally
print("\nDONE — multiseed_results.json + 05_test*.png")
