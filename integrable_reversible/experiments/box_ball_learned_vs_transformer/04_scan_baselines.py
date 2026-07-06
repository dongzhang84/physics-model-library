"""
Test 4 — the fair comparison: conserving carrier vs RNN / LSTM / Mamba(SSM),
all with the left→right scan prior; Transformer kept only as a no-structure
background reference.

Claim under test: among scan models, only the STRUCTURALLY-conserving one keeps
the invariant exact — free-emit scan models drift once their accuracy is < 100%.
Conservation drift is a symptom of imperfect accuracy, so this only shows up in a
regime where the free-emit models CAN'T learn the rule perfectly. Knob used here:
a larger, harder-to-track carrier (bigger K + denser/bigger solitons), tested out
to long lengths.

SINGLE SEED — trend signal only, not a conclusion (multi-seed is deferred to a
final batch, to match Tests 1–3). If the models don't separate, that is reported
as-is; no cherry-picking.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, math
import matplotlib.pyplot as plt
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)

# ── difficulty knobs (regime) ────────────────────────────────────────────────
K_CAP   = 6        # harder: bigger bounded carrier
T_STEPS = 2
MAXSOL  = 6        # bigger solitons to fill the carrier
GAP_LO, GAP_HI = 1, 3
DENSITY = 0.50     # keep room so the target still conserves
TRAIN_L = 32
TEST_L  = [32, 64, 128, 256]

def fc_step(s):
    out = np.zeros_like(s); u = 0
    for i in range(len(s)):
        if s[i] == 1:
            if u < K_CAP: u += 1; out[i] = 0
            else:         out[i] = 1
        else:
            if u > 0: out[i] = 1; u -= 1
            else:     out[i] = 0
    return out
def fc_run(s, T):
    s = s.copy()
    for _ in range(T): s = fc_step(s)
    return s
def make_config(L, rng):
    s = np.zeros(L, dtype=np.int64); usable = int(L*DENSITY); i = 0
    while i < usable - MAXSOL - 1:
        if rng.random() < 0.6:
            size = rng.integers(1, MAXSOL+1); s[i:i+size] = 1
            i += size + rng.integers(GAP_LO, GAP_HI+1)
        else: i += 1
    if s.sum() == 0: s[1:3] = 1
    return s
def dataset(L, N, rng):
    X = np.zeros((N, L), dtype=np.int64); Y = np.zeros((N, L), dtype=np.int64)
    for j in range(N):
        x = make_config(L, rng); X[j] = x; Y[j] = fc_run(x, T_STEPS)
    return torch.tensor(X), torch.tensor(Y)

# self-check: the TARGET must conserve (no balls fall off the right edge)
_rng = np.random.default_rng(1); _ok = True
for L in TEST_L:
    for _ in range(200):
        x = make_config(L, _rng)
        if fc_run(x, T_STEPS).sum() != x.sum(): _ok = False
assert _ok, "target not conserved at these lengths — reduce density / raise usable room"
print(f"regime: K={K_CAP} T={T_STEPS} MAXSOL={MAXSOL} density={DENSITY} — target conserved at all test lengths: {_ok}")

# ── scan models (all have the left→right prior) ──────────────────────────────
class PlainCarrier(nn.Module):        # custom RNN, FREE-emit output
    def __init__(s, H=32, hid=64):
        super().__init__(); s.H = H
        s.net = nn.Sequential(nn.Linear(1+H, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU())
        s.out = nn.Linear(hid, 1); s.hnext = nn.Linear(hid, H)
    def one(s, c):
        B, L = c.shape; h = torch.zeros(B, s.H); outs = []
        for i in range(L):
            z = s.net(torch.cat([c[:, i:i+1], h], 1)); outs.append(s.out(z)); h = torch.tanh(s.hnext(z))
        return torch.sigmoid(torch.cat(outs, 1))
    def forward(s, x):
        c = x.float()
        for _ in range(T_STEPS): c = s.one(c)
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
        for t in range(L):
            st = a*st + (1-a)*u[:, t]; outs.append(st)
        return s.norm(x + s.out(F.gelu(torch.stack(outs, 1))))
class SSMNet(nn.Module):
    def __init__(s, d=96, layers=2):
        super().__init__(); s.emb = nn.Embedding(2, d); s.layers = nn.ModuleList([SSMLayer(d) for _ in range(layers)]); s.head = nn.Linear(d, 2)
    def forward(s, x):
        h = s.emb(x)
        for l in s.layers: h = l(h)
        return s.head(h)
class ConservingCarrier(nn.Module):   # scan + STRUCTURAL conservation (k' = t - out)
    def __init__(s, hid=48, Kmax=8.0):
        super().__init__(); s.Kmax = Kmax
        s.gate = nn.Sequential(nn.Linear(4, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
    def one(s, xf, hard):
        B, L = xf.shape; k = torch.zeros(B); xp = F.pad(xf, (1, 1)); outs = []
        for i in range(L):
            c = xf[:, i]; feat = torch.stack([c, k/s.Kmax, xp[:, i], xp[:, i+2]], -1)
            e = torch.sigmoid(s.gate(feat).squeeze(-1))
            if hard: eh = (e > 0.5).float(); e = eh + (e - e.detach())
            t = c + k; out = e*torch.clamp(t, 0, 1); k = t - out; outs.append(out)
        return torch.stack(outs, 1)
    def forward(s, x, hard=False):
        c = x.float()
        for _ in range(T_STEPS): c = s.one(c, hard)
        return c
class PE(nn.Module):
    def __init__(s, d, maxlen=8192):
        super().__init__(); pe = torch.zeros(maxlen, d); pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float()*(-math.log(10000.0)/d))
        pe[:, 0::2] = torch.sin(pos*div); pe[:, 1::2] = torch.cos(pos*div); s.register_buffer("pe", pe)
    def forward(s, x): return x + s.pe[:x.size(1)].unsqueeze(0)
class TF(nn.Module):
    def __init__(s, d=96, heads=4, layers=3):
        super().__init__(); s.emb = nn.Embedding(2, d); s.pe = PE(d)
        enc = nn.TransformerEncoderLayer(d, heads, 4*d, batch_first=True, dropout=0.0, activation="gelu")
        s.enc = nn.TransformerEncoder(enc, layers); s.head = nn.Linear(d, 2)
    def forward(s, x): return s.head(s.enc(s.pe(s.emb(x))))

# ── train each on L=32 ───────────────────────────────────────────────────────
rng = np.random.default_rng(0); Xtr, Ytr = dataset(TRAIN_L, 3000, rng); B = 256
ce = nn.CrossEntropyLoss(); bce = nn.BCELoss()
def prob(m, X):
    if isinstance(m, PlainCarrier):       return m(X)
    if isinstance(m, ConservingCarrier):  return m(X, hard=True)
    return m(X).softmax(-1)[..., 1]
def fit(m, epochs, lr, logits):
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), B):
            idx = perm[i:i+B]
            loss = ce(m(Xtr[idx]).reshape(-1, 2), Ytr[idx].reshape(-1)) if logits \
                   else bce(m(Xtr[idx]).clamp(1e-6, 1-1e-6), Ytr[idx].float())
            opt.zero_grad(); loss.backward(); opt.step()
    return m
models = {}
for name, ctor, ep, lr, lg in [
    ("plain carrier (RNN, small)", lambda: PlainCarrier(32, 64),  45, 3e-3, False),
    ("plain carrier (RNN, big)",   lambda: PlainCarrier(64, 256), 80, 3e-3, False),  # fairness check
    ("LSTM",                       lambda: LSTMNet(),      45, 2e-3, True),
    ("SSM (Mamba-family)",         lambda: SSMNet(),       45, 2e-3, True),
    ("conserving carrier",         lambda: ConservingCarrier(), 45, 3e-3, False),
    ("Transformer (ref)",          lambda: TF(),           30, 1e-3, True)]:
    print(f"training {name} ..."); models[name] = fit(ctor(), ep, lr, lg)

# ── evaluate: accuracy + conservation vs length ──────────────────────────────
rng_te = np.random.default_rng(123)
res = {n: {"acc": [], "cons": []} for n in models}
for m in models.values(): m.eval()
Xtes = {L: dataset(L, 200, rng_te) for L in TEST_L}
print("\n" + " "*22 + "".join(f"{L:>12}" for L in TEST_L) + "   (acc% / cons%)")
with torch.no_grad():
    for name, m in models.items():
        row = ""
        for L in TEST_L:
            Xte, Yte = Xtes[L]; pred = (prob(m, Xte) > 0.5).long()
            acc = (pred == Yte).float().mean().item()*100
            cons = (pred.sum(1) == Xte.sum(1)).float().mean().item()*100
            res[name]["acc"].append(acc); res[name]["cons"].append(cons)
            row += f"   {acc:5.1f}/{cons:5.1f}"
        print(f"{name:22s}{row}")

# ── figure ───────────────────────────────────────────────────────────────────
style = {"conserving carrier": ("#1e8449","D-",2.2),
         "plain carrier (RNN, small)": ("#e67e22","o-",1.3), "plain carrier (RNN, big)": ("#d35400","s-",1.7),
         "LSTM": ("#8e44ad","o-",1.6), "SSM (Mamba-family)": ("#2980b9","o-",1.6), "Transformer (ref)": ("#95a5a6","o--",1.4)}
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for a, key, ttl, ylab, ylim in [(0,"acc","Accuracy vs length","per-position accuracy (%)",(40,102)),
                                 (1,"cons","Conservation vs length","ball-count conserved (%)",(-3,105))]:
    for n in style:
        c, ls, lw = style[n]; ax[a].plot(TEST_L, res[n][key], ls, color=c, lw=lw, label=n)
    ax[a].axvline(TRAIN_L, ls=":", color="gray"); ax[a].set_xlabel("lattice length (32 = trained, rest = OOD)")
    ax[a].set_ylabel(ylab); ax[a].set_ylim(*ylim); ax[a].set_title(ttl, fontsize=11); ax[a].legend(fontsize=8); ax[a].grid(alpha=0.3)
fig.suptitle(f"Test 4 — fair comparison on finite-carrier BBS (K={K_CAP}, single seed)", fontsize=12, y=1.01)
fig.tight_layout(); fig.savefig("04_scan_baselines.png", dpi=130, bbox_inches="tight")
print("\nsaved 04_scan_baselines.png")
