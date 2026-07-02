"""
Route C — first prototype: a LEARNED integrable-style model vs a Transformer,
on the Box-Ball System (BBS) length-extrapolation task.

Demo 2's hard-coded BBS rule is a cheat: it IS the label generator, so its 100%
is "structure by construction", not learning. Here we KEEP that line as the
ceiling and ADD a genuinely TRAINED model that:
  - carries the integrable computational FORM as structure — a weight-shared,
    left-to-right recurrent "carrier" scan (length-independent by construction),
  - but LEARNS the collision rule from data (it never sees the BBS rule),
  - applies the learned single step T times (BBS = a repeated single step).
Both models are trained on L=32 ONLY, then tested on 48/64/96/128 (OOD by length;
soliton size & density fixed across lengths -> pure length extrapolation).

Three lines: hard-coded integrable (ceiling) / learned carrier (this) / Transformer.
Also shown: the all-zeros baseline, since the lattice is sparse and per-position
accuracy is inflated by it. Does NOT modify Demo 2 (demos/box_ball_system/).

Honest scope: this removes the Demo-2 cheat and tests "can a length-independent
structure LEARN the rule and extrapolate where attention can't?". The learned
carrier's reversibility here is approximate (a forward scan), NOT welded-in; the
exact-reversible coupling-layer version is the next iteration.
"""
import numpy as np, torch, torch.nn as nn, math
import matplotlib.pyplot as plt
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)

# ── integrable engine = ground-truth generator (also the ceiling line) ───────
def bbs_step(s):
    out = np.zeros_like(s); carrier = 0
    for i in range(len(s)):
        if s[i] == 1: carrier += 1; out[i] = 0
        else:
            if carrier > 0: out[i] = 1; carrier -= 1
    return out
def bbs_run(s, T):
    for _ in range(T): s = bbs_step(s)
    return s

T_STEPS = 2; MAXSOL = 3
def make_config(L, rng):
    s = np.zeros(L, dtype=np.int64); usable = int(L*0.55); i = 0; n = 0
    while i < usable - MAXSOL - 1:
        if rng.random() < 0.5 and n < max(3, L//7):
            size = rng.integers(1, MAXSOL+1); s[i:i+size] = 1
            i += size + rng.integers(2, 5); n += 1
        else: i += 1
    if s.sum() == 0: s[1:3] = 1
    return s
def dataset(L, N, rng):
    X = np.zeros((N, L), dtype=np.int64); Y = np.zeros((N, L), dtype=np.int64)
    for j in range(N):
        x = make_config(L, rng); X[j] = x; Y[j] = bbs_run(x, T_STEPS)
    return torch.tensor(X), torch.tensor(Y)

# ── baseline: small Transformer (same recipe as Demo 2) ──────────────────────
class PE(nn.Module):
    def __init__(s, d, maxlen=4096):
        super().__init__(); pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float()*(-math.log(10000.0)/d))
        pe[:, 0::2] = torch.sin(pos*div); pe[:, 1::2] = torch.cos(pos*div)
        s.register_buffer("pe", pe)
    def forward(s, x): return x + s.pe[:x.size(1)].unsqueeze(0)
class TF(nn.Module):
    def __init__(s, d=64, heads=4, layers=3):
        super().__init__(); s.emb = nn.Embedding(2, d); s.pe = PE(d)
        enc = nn.TransformerEncoderLayer(d, heads, 4*d, batch_first=True,
                                         dropout=0.0, activation="gelu")
        s.enc = nn.TransformerEncoder(enc, layers); s.head = nn.Linear(d, 2)
    def forward(s, x): return s.head(s.enc(s.pe(s.emb(x))))

# ── learned integrable-style model: weight-shared L->R carrier scan ──────────
class LearnedCarrier(nn.Module):
    """One BBS-like step as a shared recurrent cell (bit, carrier h) -> (out prob, h');
    apply the SAME learned step T_STEPS times. Weights are position- and
    length-independent, so it runs at any length."""
    def __init__(s, H=24, hid=64):
        super().__init__(); s.H = H
        s.net   = nn.Sequential(nn.Linear(1+H, hid), nn.GELU(),
                                nn.Linear(hid, hid), nn.GELU())
        s.out   = nn.Linear(hid, 1)   # output-bit logit for this position
        s.hnext = nn.Linear(hid, H)    # updated carrier state
    def one_step(s, c):                # c: (B,L) in [0,1] -> p: (B,L)
        B, L = c.shape
        h = torch.zeros(B, s.H)
        outs = []
        for i in range(L):
            z = s.net(torch.cat([c[:, i:i+1], h], dim=1))
            outs.append(s.out(z))
            h = torch.tanh(s.hnext(z))
        return torch.sigmoid(torch.cat(outs, dim=1))
    def forward(s, x_bits, steps=T_STEPS):
        c = x_bits.float()
        for _ in range(steps):         # compose the learned single step
            c = s.one_step(c)
        return c

# ── train both on L=32 only ──────────────────────────────────────────────────
rng = np.random.default_rng(0)
L_TRAIN = 32
Xtr, Ytr = dataset(L_TRAIN, 2500, rng)
B = 256

print("training Transformer on L=32 ...")
tf = TF(); opt = torch.optim.AdamW(tf.parameters(), lr=1e-3); ce = nn.CrossEntropyLoss()
for epoch in range(28):
    perm = torch.randperm(len(Xtr)); tot = 0.0
    for i in range(0, len(Xtr), B):
        idx = perm[i:i+B]; logits = tf(Xtr[idx])
        loss = ce(logits.reshape(-1, 2), Ytr[idx].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()*len(idx)
    if (epoch+1) % 7 == 0: print(f"  TF epoch {epoch+1:2d}  loss {tot/len(Xtr):.4f}")

print("training LearnedCarrier on L=32 ...")
lc = LearnedCarrier(); optc = torch.optim.Adam(lc.parameters(), lr=3e-3)
bce = nn.BCELoss()
for epoch in range(40):
    perm = torch.randperm(len(Xtr)); tot = 0.0
    for i in range(0, len(Xtr), B):
        idx = perm[i:i+B]; p = lc(Xtr[idx])
        loss = bce(p.clamp(1e-6, 1-1e-6), Ytr[idx].float())
        optc.zero_grad(); loss.backward(); optc.step(); tot += loss.item()*len(idx)
    if (epoch+1) % 8 == 0: print(f"  LC epoch {epoch+1:2d}  loss {tot/len(Xtr):.4f}")

# ── evaluate across lengths (32 = in-dist, rest = OOD longer) ─────────────────
tf.eval(); lc.eval()
lengths = [32, 48, 64, 96, 128]
tf_acc, tf_cons, lc_acc, lc_cons, zero_acc = [], [], [], [], []
rng_te = np.random.default_rng(123)
print("\n L    all-0   Transformer(acc/cons)   LearnedCarrier(acc/cons)   integrable")
with torch.no_grad():
    for L in lengths:
        Xte, Yte = dataset(L, 300, rng_te)
        # all-zeros baseline (sparsity reference)
        z = (Yte == 0).float().mean().item()*100
        # Transformer
        pt = tf(Xte).argmax(-1)
        ta = (pt == Yte).float().mean().item()*100
        tc = (pt.sum(1) == Xte.sum(1)).float().mean().item()*100
        # LearnedCarrier
        pc = (lc(Xte) > 0.5).long()
        la = (pc == Yte).float().mean().item()*100
        lcc = (pc.sum(1) == Xte.sum(1)).float().mean().item()*100
        zero_acc.append(z); tf_acc.append(ta); tf_cons.append(tc); lc_acc.append(la); lc_cons.append(lcc)
        print(f"{L:3d}  {z:5.1f}%   {ta:5.1f}% / {tc:5.1f}%          {la:5.1f}% / {lcc:5.1f}%         100% / 100%")

int_line = [100.0]*len(lengths)

# ── figure: three model lines + all-zeros reference ──────────────────────────
CG, CB, CR, CZ = "#27ae60", "#2c3e50", "#c0392b", "#95a5a6"
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
ax[0].plot(lengths, int_line, "s-", color=CR, label="hard-coded integrable (ceiling / cheat)")
ax[0].plot(lengths, lc_acc,  "o-", color=CG, label="learned carrier (trained, structure)")
ax[0].plot(lengths, tf_acc,  "o-", color=CB, label="Transformer (trained, no structure)")
ax[0].plot(lengths, zero_acc, "--", color=CZ, lw=1, label="all-zeros baseline (sparsity)")
ax[0].axvline(32, ls=":", color="gray"); ax[0].text(33, 55, "trained only here", fontsize=8, color="gray")
ax[0].set_xlabel("lattice length (32 = trained, rest = OOD longer)")
ax[0].set_ylabel("per-position accuracy (%)"); ax[0].set_ylim(50, 102)
ax[0].set_title("Predict T=2 BBS evolution — accuracy vs length")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

ax[1].plot(lengths, int_line, "s-", color=CR, label="hard-coded integrable")
ax[1].plot(lengths, lc_cons, "o-", color=CG, label="learned carrier")
ax[1].plot(lengths, tf_cons, "o-", color=CB, label="Transformer")
ax[1].set_xlabel("lattice length"); ax[1].set_ylabel("ball-count conserved (%)")
ax[1].set_ylim(-3, 105); ax[1].set_title("Invariant (# balls) preserved?")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

fig.suptitle("Route C prototype — a LEARNED length-independent structure vs a Transformer "
             "(hard-coded integrable = ceiling)", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig("01_carrier_vs_transformer.png", dpi=130, bbox_inches="tight")
print("\nsaved 01_carrier_vs_transformer.png")
