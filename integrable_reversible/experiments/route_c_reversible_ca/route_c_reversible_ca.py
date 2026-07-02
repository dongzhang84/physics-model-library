"""
Route C (full) — four-way comparison on the Box-Ball System length task.

Four lines, each isolating one thing:
  1. hard-coded integrable (BBS rule)         -> ceiling / cheat reference
  2. learned carrier (no coupling)            -> recurrence learns, but conservation drifts
  3. learned reversible+conservative CA (THIS)-> reversibility & conservation WELDED by structure
  4. Transformer                              -> lower bound / no structure

Model #3 = "gated-swap reversible CA":
  - pair up adjacent cells; a shared local gate decides whether to SWAP each pair.
  - a swap conserves the pair's ball-count exactly and is its own inverse (involution).
  - the gate reads only (a) the pair's SUM (invariant under its own swap) and (b) neighbor
    cells that live in pairs NOT updated this layer (frozen) -> the whole layer is EXACTLY
    invertible, coupling-layer style.
  => ball-count conservation and reversibility are STRUCTURAL (hold at any length, no matter
     what the gate learns); only the collision rule (when to swap) is learned from data.

Trained on L=32 only; tested on 48/64/96/128 (OOD by length). Does NOT modify Demo 2 or the
earlier prototype experiment.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, math
import matplotlib.pyplot as plt
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)

# ── integrable engine = ground-truth generator (ceiling line) ────────────────
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

# ── #4 Transformer (same recipe as Demo 2) ───────────────────────────────────
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
        enc = nn.TransformerEncoderLayer(d, heads, 4*d, batch_first=True, dropout=0.0, activation="gelu")
        s.enc = nn.TransformerEncoder(enc, layers); s.head = nn.Linear(d, 2)
    def forward(s, x): return s.head(s.enc(s.pe(s.emb(x))))

# ── #2 learned carrier (no coupling): weight-shared L->R scan ─────────────────
class LearnedCarrier(nn.Module):
    def __init__(s, H=24, hid=64):
        super().__init__(); s.H = H
        s.net = nn.Sequential(nn.Linear(1+H, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU())
        s.out = nn.Linear(hid, 1); s.hnext = nn.Linear(hid, H)
    def one_step(s, c):
        B, L = c.shape; h = torch.zeros(B, s.H); outs = []
        for i in range(L):
            z = s.net(torch.cat([c[:, i:i+1], h], dim=1))
            outs.append(s.out(z)); h = torch.tanh(s.hnext(z))
        return torch.sigmoid(torch.cat(outs, dim=1))
    def forward(s, x_bits, steps=T_STEPS):
        c = x_bits.float()
        for _ in range(steps): c = s.one_step(c)
        return c

# ── #3 reversible + conservative gated-swap CA (the full Route C) ─────────────
class ReversibleConservativeCA(nn.Module):
    """Stack of gated-swap layers. Each layer pairs cells at a given offset, updates the
    'active' pairs (every other pair) by a possible swap whose gate reads only frozen context
    -> exactly invertible; swaps conserve ball count exactly."""
    def __init__(s, n_layers=16, hid=48):
        super().__init__()
        # one UNtied gate per layer (own_sum invariant + 4 frozen context cells from pairs k±1)
        def mk(): return nn.Sequential(nn.Linear(5, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
        s.gates = nn.ModuleList([mk() for _ in range(n_layers)])
        s.temp = 1.0                                    # gate temperature (annealed toward hard during training)
        # cycle through (offset, active_parity) so every cell is updated and balls can flow
        cfgs = [(0, 0), (0, 1), (1, 0), (1, 1)]
        s.configs = [cfgs[i % 4] for i in range(n_layers)]

    def _layer(s, state, o, parity, gate, hard):
        B, L = state.shape
        K = (L - o) // 2
        if K == 0: return state
        left = o + 2 * torch.arange(K)                 # left cell of each pair
        kk = torch.arange(K); act = (kk % 2 == parity)
        if act.sum() == 0: return state
        la = left[act]; ra = la + 1                     # active pair indices (Ka,)
        sp = F.pad(state, (2, 2))                        # pad 2 so a ±2 window is always valid
        lval = state[:, la]; rval = state[:, ra]        # (B,Ka)
        own = lval + rval                               # swap-invariant
        # 4 frozen context cells (pairs k-1 and k+1, unchanged this layer): la-2, la-1, ra+1, ra+2
        c1 = sp[:, la]; c2 = sp[:, la + 1]; c3 = sp[:, ra + 3]; c4 = sp[:, ra + 4]
        feat = torch.stack([own, c1, c2, c3, c4], dim=-1)  # (B,Ka,5) — all frozen/invariant
        a_soft = torch.sigmoid(gate(feat).squeeze(-1) / s.temp)  # (B,Ka)
        if hard:                                          # straight-through: forward hard, gradient soft
            a_hard = (a_soft > 0.5).float()
            a_use = a_hard + (a_soft - a_soft.detach())   # value == a_hard (binary preserved -> exact conservation)
        else:
            a_use = a_soft
        new_l = lval + a_use * (rval - lval)
        new_r = rval + a_use * (lval - rval)
        out = state.clone()
        out[:, la] = new_l; out[:, ra] = new_r
        return out

    def forward(s, x_bits, hard=False):     # soft for training (gradient flows); hard=True at eval
        state = x_bits.float()
        for i, (o, p) in enumerate(s.configs): state = s._layer(state, o, p, s.gates[i], hard)
        return state
    def invert(s, state, hard=True):
        state = state.clone()
        for i in reversed(range(len(s.configs))):
            o, p = s.configs[i]; state = s._layer(state, o, p, s.gates[i], hard)  # each layer self-inverse
        return state

# ── self-check: structural conservation + reversibility at RANDOM init ───────
print("structural self-check (random init, no training):")
_ca = ReversibleConservativeCA()
_rng = np.random.default_rng(1)
_ok_cons = _ok_rev = True
for L in [32, 48, 64, 96, 128]:
    Xc, _ = dataset(L, 40, _rng)
    with torch.no_grad():
        y = _ca(Xc, hard=True)
        yb = (y > 0.5).long()
        if not torch.equal(yb.sum(1), Xc.sum(1)): _ok_cons = False
        xrec = (_ca.invert(y.float(), hard=True) > 0.5).long()
        if not torch.equal(xrec, Xc): _ok_rev = False
print(f"  ball-count conserved exactly at all lengths: {_ok_cons}")
print(f"  exactly reversible (invert(forward(x))==x)  : {_ok_rev}")
assert _ok_cons and _ok_rev, "structural guarantees broken — fix construction before training"

# ── train #2, #3, #4 on L=32 ─────────────────────────────────────────────────
rng = np.random.default_rng(0); L_TRAIN = 32
Xtr, Ytr = dataset(L_TRAIN, 2500, rng); B = 256

def train(model, kind, epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss(); bce = nn.BCELoss()
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr)); tot = 0.0
        for i in range(0, len(Xtr), B):
            idx = perm[i:i+B]
            if kind == "tf":
                loss = ce(model(Xtr[idx]).reshape(-1, 2), Ytr[idx].reshape(-1))
            else:
                p = model(Xtr[idx]); loss = bce(p.clamp(1e-6, 1-1e-6), Ytr[idx].float())
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()*len(idx)
        if (ep+1) % max(1, epochs//4) == 0: print(f"  {kind} ep {ep+1:2d} loss {tot/len(Xtr):.4f}")

print("training Transformer ...");     tf = TF();                    train(tf, "tf", 28, 1e-3)
print("training learned carrier ...");  lc = LearnedCarrier();        train(lc, "lc", 40, 3e-3)

print("training reversible CA (temperature-annealed soft -> hard) ...")
ca = ReversibleConservativeCA()
optca = torch.optim.AdamW(ca.parameters(), lr=3e-3); bce = nn.BCELoss(); CA_EP = 140
for ep in range(CA_EP):
    ca.temp = 1.0 * (0.12) ** (ep / (CA_EP - 1))          # anneal gate temperature 1.0 -> 0.12
    perm = torch.randperm(len(Xtr)); tot = 0.0
    for i in range(0, len(Xtr), B):
        idx = perm[i:i+B]
        p = ca(Xtr[idx], hard=False)                      # soft forward (gradients), sharpened by temp
        loss = bce(p.clamp(1e-6, 1-1e-6), Ytr[idx].float())
        optca.zero_grad(); loss.backward(); optca.step(); tot += loss.item()*len(idx)
    if (ep+1) % (CA_EP//7) == 0: print(f"  ca ep {ep+1:3d}  temp {ca.temp:.2f}  loss {tot/len(Xtr):.4f}")

# ── evaluate four lines across lengths ───────────────────────────────────────
tf.eval(); lc.eval(); ca.eval()
lengths = [32, 48, 64, 96, 128]
res = {k: {"acc": [], "cons": []} for k in ["int", "ca", "lc", "tf"]}
zero_acc = []
rng_te = np.random.default_rng(123)
print("\n L    all-0    TF acc/cons     carrier acc/cons     revCA acc/cons     integrable")
with torch.no_grad():
    for L in lengths:
        Xte, Yte = dataset(L, 300, rng_te)
        zero_acc.append((Yte == 0).float().mean().item()*100)
        def acc_cons(pred):
            return (pred == Yte).float().mean().item()*100, (pred.sum(1) == Xte.sum(1)).float().mean().item()*100
        ta, tc = acc_cons(tf(Xte).argmax(-1))
        la, lco = acc_cons((lc(Xte) > 0.5).long())
        ca_pred = (ca(Xte, hard=True) > 0.5).long()
        ca_a, ca_c = acc_cons(ca_pred)
        res["int"]["acc"].append(100.0); res["int"]["cons"].append(100.0)
        res["tf"]["acc"].append(ta);   res["tf"]["cons"].append(tc)
        res["lc"]["acc"].append(la);   res["lc"]["cons"].append(lco)
        res["ca"]["acc"].append(ca_a); res["ca"]["cons"].append(ca_c)
        print(f"{L:3d}  {zero_acc[-1]:5.1f}%   {ta:5.1f}/{tc:5.1f}     {la:5.1f}/{lco:5.1f}       {ca_a:5.1f}/{ca_c:5.1f}       100/100")

# ── figure: four lines, two panels ───────────────────────────────────────────
CI, CCA, CLC, CTF, CZ = "#c0392b", "#8e44ad", "#27ae60", "#2c3e50", "#95a5a6"
fig, ax = plt.subplots(1, 2, figsize=(13, 4.9))
ax[0].plot(lengths, res["int"]["acc"], "s-", color=CI,  label="1. hard-coded integrable (cheat)")
ax[0].plot(lengths, res["ca"]["acc"],  "D-", color=CCA, label="3. learned reversible+conserv. CA (Route C)")
ax[0].plot(lengths, res["lc"]["acc"],  "o-", color=CLC, label="2. learned carrier (no coupling)")
ax[0].plot(lengths, res["tf"]["acc"],  "o-", color=CTF, label="4. Transformer")
ax[0].plot(lengths, zero_acc, "--", color=CZ, lw=1, label="all-zeros baseline")
ax[0].axvline(32, ls=":", color="gray"); ax[0].text(33, 55, "trained only here", fontsize=8, color="gray")
ax[0].set_xlabel("lattice length (32 = trained, rest = OOD)"); ax[0].set_ylabel("per-position accuracy (%)")
ax[0].set_ylim(50, 102); ax[0].set_title("Accuracy vs length"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

ax[1].plot(lengths, res["int"]["cons"], "s-", color=CI,  label="1. hard-coded integrable")
ax[1].plot(lengths, res["ca"]["cons"],  "D-", color=CCA, label="3. reversible+conserv. CA (Route C)")
ax[1].plot(lengths, res["lc"]["cons"],  "o-", color=CLC, label="2. learned carrier")
ax[1].plot(lengths, res["tf"]["cons"],  "o-", color=CTF, label="4. Transformer")
ax[1].set_xlabel("lattice length"); ax[1].set_ylabel("ball-count conserved (%)")
ax[1].set_ylim(-3, 105); ax[1].set_title("Invariant (# balls) preserved?  #3 is welded -> exact")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
fig.suptitle("Route C — welding reversibility+conservation into a LEARNED model (four-way)", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig("route_c_reversible_ca.png", dpi=130, bbox_inches="tight")
print("\nsaved route_c_reversible_ca.png")
