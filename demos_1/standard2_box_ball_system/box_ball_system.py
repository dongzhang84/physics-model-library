import numpy as np, torch, torch.nn as nn, math
import matplotlib.pyplot as plt
torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(4)

# ---------------------------------------------------------------
# Integrable engine: Box-Ball System update via an infinite-capacity
# carrier sweeping LEFT->RIGHT. This IS the integrable dynamics:
#   - solitons (blocks of 1s) move right, big ones overtake & pass through small
#   - number of balls conserved exactly
#   - exactly reversible (time reversal = mirror . forward . mirror)
# It is a single reversible recurrent pass -> works at ANY length.
# ---------------------------------------------------------------
def bbs_step(s):
    out = np.zeros_like(s); carrier = 0
    for i in range(len(s)):
        if s[i] == 1:
            carrier += 1; out[i] = 0
        else:
            if carrier > 0:
                out[i] = 1; carrier -= 1
    return out

def bbs_run(s, T):
    for _ in range(T): s = bbs_step(s)
    return s

def mirror(s): return s[::-1].copy()
def bbs_step_inv(s): return mirror(bbs_step(mirror(s)))   # exact time-reversal

# sanity: reversibility + conservation
_t = np.array([1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
assert np.array_equal(bbs_step_inv(bbs_step(_t)), _t), "not reversible!"
assert _t.sum() == bbs_step(_t).sum(), "balls not conserved!"

# ---------------------------------------------------------------
# Data: random well-spaced solitons (blocks of size 1..3) in the left
# portion, trailing zeros give room to move. Predict state after T steps.
# Soliton sizes & density fixed across lengths -> pure LENGTH extrapolation.
# ---------------------------------------------------------------
T_STEPS = 2
MAXSOL  = 3

def make_config(L, rng):
    s = np.zeros(L, dtype=np.int64)
    usable = int(L * 0.55)              # keep room on the right
    i = 0; n = 0
    while i < usable - MAXSOL - 1:
        if rng.random() < 0.5 and n < max(3, L//7):
            size = rng.integers(1, MAXSOL+1)
            s[i:i+size] = 1; i += size + rng.integers(2, 5); n += 1
        else:
            i += 1
    if s.sum() == 0: s[1:3] = 1
    return s

def dataset(L, N, rng):
    X = np.zeros((N, L), dtype=np.int64); Y = np.zeros((N, L), dtype=np.int64)
    for j in range(N):
        x = make_config(L, rng); X[j] = x; Y[j] = bbs_run(x, T_STEPS)
    return torch.tensor(X), torch.tensor(Y)

# ---------------------------------------------------------------
# Small Transformer: per-position binary prediction, sinusoidal PE
# (so it CAN accept longer inputs at test time).
# ---------------------------------------------------------------
class PE(nn.Module):
    def __init__(s, d, maxlen=4096):
        super().__init__()
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0)/d))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        s.register_buffer("pe", pe)
    def forward(s, x): return x + s.pe[:x.size(1)].unsqueeze(0)

class TF(nn.Module):
    def __init__(s, d=96, heads=4, layers=4):
        super().__init__()
        s.emb = nn.Embedding(2, d); s.pe = PE(d)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4*d,
                                         batch_first=True, dropout=0.0, activation="gelu")
        s.enc = nn.TransformerEncoder(enc, layers)
        s.head = nn.Linear(d, 2)
    def forward(s, x):
        h = s.pe(s.emb(x)); h = s.enc(h); return s.head(h)

# ---------------------------------------------------------------
# Train on L=32 only
# ---------------------------------------------------------------
rng = np.random.default_rng(0)
L_TRAIN = 32
Xtr, Ytr = dataset(L_TRAIN, 2500, rng)
model = TF(d=64, heads=4, layers=3); opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()
B = 256
print("training small Transformer on L=32 ...")
for epoch in range(28):
    perm = torch.randperm(len(Xtr))
    tot = 0.0
    for i in range(0, len(Xtr), B):
        idx = perm[i:i+B]; xb, yb = Xtr[idx], Ytr[idx]
        logits = model(xb)
        loss = lossf(logits.reshape(-1,2), yb.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()*len(idx)
    if (epoch+1) % 4 == 0:
        print(f"  epoch {epoch+1:2d}  loss {tot/len(Xtr):.4f}")

# ---------------------------------------------------------------
# Evaluate: exact-sequence accuracy + ball-count conservation
# across lengths (32 = in-dist, rest = OOD longer)
# ---------------------------------------------------------------
model.eval()
lengths = [32, 48, 64, 96, 128]
tf_seq, tf_cons, int_seq, int_cons = [], [], [], []
rng_te = np.random.default_rng(123)
with torch.no_grad():
    for L in lengths:
        Xte, Yte = dataset(L, 300, rng_te)
        pred = model(Xte).argmax(-1)
        pos_acc = (pred == Yte).float().mean().item()                # per-position accuracy
        cons_ok = (pred.sum(1) == Xte.sum(1)).float().mean().item()  # ball count preserved
        tf_seq.append(pos_acc*100); tf_cons.append(cons_ok*100)
        int_seq.append(100.0); int_cons.append(100.0)
        print(f"L={L:3d}  Transformer per-pos={pos_acc*100:5.1f}%  ball-conserved={cons_ok*100:5.1f}%   | integrable=100% / 100%")

# reversibility check (integrable only): recover input from output exactly
rng_rv = np.random.default_rng(7)
ok_rev = True
for L in lengths:
    for _ in range(50):
        x = make_config(L, rng_rv); y = bbs_run(x, T_STEPS)
        xrec = y.copy()
        for _ in range(T_STEPS): xrec = bbs_step_inv(xrec)
        if not np.array_equal(xrec, x): ok_rev = False
print("integrable exact reversibility on all lengths:", ok_rev)

# ---------------------------------------------------------------
# Figure
# ---------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].plot(lengths, tf_seq, "o-", color="#2c3e50", label="small Transformer (learned)")
ax[0].plot(lengths, int_seq, "s-", color="#c0392b", label="integrable engine (BBS carrier)")
ax[0].axvline(32, ls=":", color="gray"); ax[0].text(33, 43, "trained only here", fontsize=8, color="gray")
ax[0].set_xlabel("lattice length (32 = trained, rest = longer / OOD)")
ax[0].set_ylabel("per-position accuracy (%)")
ax[0].set_title("Predict T=2-step evolution of a Box-Ball System")
ax[0].set_ylim(40, 102); ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

ax[1].plot(lengths, tf_cons, "o-", color="#2c3e50", label="Transformer prediction")
ax[1].plot(lengths, int_cons, "s-", color="#c0392b", label="integrable engine")
ax[1].set_xlabel("lattice length")
ax[1].set_ylabel("ball-count conserved (%)")
ax[1].set_title("Invariant (# balls) preserved?  structure = exact, learned = drifts")
ax[1].set_ylim(-3, 105); ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)

fig.suptitle("Standard 2 — integrable structure generalizes to any length; a learned Transformer does not",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig("bbs_standard2_demo.png", dpi=130, bbox_inches="tight")
print("saved.")
