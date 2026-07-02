"""
The conserving carrier (the proposal's Route C) — the reversible, mass-CONSERVING carrier.

The previous full-Route-C attempt (local gated-swap CA) had the guarantees but
couldn't learn BBS. The fix: keep the CARRIER (left->right reach = expressive)
but constrain its per-site update to CONSERVE ball count by construction.

Per-site update during the scan, with carrier count k and cell c:
  total t = c + k  (the resource passing through this site)
  the only two count-preserving outcomes (for t>=1) are:
     "emit": out=1, k' = t-1        "hold": out=0, k' = t
  a learned gate picks emit vs hold (for t=0 it is forced to hold).
=> cell + carrier is preserved at every site, so sum(output) = sum(input) minus
   whatever remains in the carrier at the end. With trailing zeros the carrier
   empties, giving EXACT ball-count conservation. (BBS itself is just the fixed
   rule "emit iff cell==0"; here that rule is learned.)

Four lines: hard-coded integrable / conserving carrier (THIS) / plain carrier
(free-emit, from the prototype) / Transformer. Trained on L=32, tested OOD.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, math
import matplotlib.pyplot as plt
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)

# ── integrable engine + data (same as the other experiments) ─────────────────
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

# ── #4 Transformer ───────────────────────────────────────────────────────────
class PE(nn.Module):
    def __init__(s, d, maxlen=4096):
        super().__init__(); pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float()*(-math.log(10000.0)/d))
        pe[:, 0::2] = torch.sin(pos*div); pe[:, 1::2] = torch.cos(pos*div); s.register_buffer("pe", pe)
    def forward(s, x): return x + s.pe[:x.size(1)].unsqueeze(0)
class TF(nn.Module):
    def __init__(s, d=64, heads=4, layers=3):
        super().__init__(); s.emb = nn.Embedding(2, d); s.pe = PE(d)
        enc = nn.TransformerEncoderLayer(d, heads, 4*d, batch_first=True, dropout=0.0, activation="gelu")
        s.enc = nn.TransformerEncoder(enc, layers); s.head = nn.Linear(d, 2)
    def forward(s, x): return s.head(s.enc(s.pe(s.emb(x))))

# ── #2 plain carrier (free emit, no conservation) — the prototype baseline ───
class PlainCarrier(nn.Module):
    def __init__(s, H=24, hid=64):
        super().__init__(); s.H = H
        s.net = nn.Sequential(nn.Linear(1+H, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU())
        s.out = nn.Linear(hid, 1); s.hnext = nn.Linear(hid, H)
    def one(s, c):
        B, L = c.shape; h = torch.zeros(B, s.H); outs = []
        for i in range(L):
            z = s.net(torch.cat([c[:, i:i+1], h], dim=1)); outs.append(s.out(z)); h = torch.tanh(s.hnext(z))
        return torch.sigmoid(torch.cat(outs, dim=1))
    def forward(s, x, steps=T_STEPS):
        c = x.float()
        for _ in range(steps): c = s.one(c)
        return c

# ── #3 conserving carrier (Route C v2): per-site count-preserving emit/hold ──
class ConservingCarrier(nn.Module):
    def __init__(s, hid=32, Kmax=6.0):
        super().__init__(); s.Kmax = Kmax
        s.gate = nn.Sequential(nn.Linear(4, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
    def one(s, xf, hard):
        B, L = xf.shape
        k = torch.zeros(B)                       # carrier count (integer in hard mode)
        xp = F.pad(xf, (1, 1))
        outs = []
        for i in range(L):
            c = xf[:, i]
            feat = torch.stack([c, k / s.Kmax, xp[:, i], xp[:, i + 2]], dim=-1)  # cell, carrier, L/R neighbor
            e = torch.sigmoid(s.gate(feat).squeeze(-1))
            if hard:
                eh = (e > 0.5).float(); e = eh + (e - e.detach())   # straight-through (forward hard)
            t = c + k
            avail = torch.clamp(t, 0.0, 1.0)      # can emit at most min(t,1)
            out_i = e * avail                     # emit -> 1, hold -> 0 (hard); conserves: k' = t - out
            k = t - out_i
            outs.append(out_i)
        return torch.stack(outs, dim=1), k        # (B,L), final carrier
    def forward(s, x, steps=T_STEPS, hard=False):
        c = x.float()
        for _ in range(steps): c, _ = s.one(c, hard)
        return c
    def final_carrier(s, x, hard=True):           # for diagnostics
        c = x.float(); tot = 0.0
        for _ in range(T_STEPS): c, k = s.one(c, hard); tot = k
        return tot

# ── train #2,#3,#4 on L=32 ───────────────────────────────────────────────────
rng = np.random.default_rng(0); L_TRAIN = 32
Xtr, Ytr = dataset(L_TRAIN, 2500, rng); B = 256
ce = nn.CrossEntropyLoss(); bce = nn.BCELoss()

def fit(model, kind, epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr)); tot = 0.0
        for i in range(0, len(Xtr), B):
            idx = perm[i:i+B]
            if kind == "tf":   loss = ce(model(Xtr[idx]).reshape(-1, 2), Ytr[idx].reshape(-1))
            elif kind == "cc": loss = bce(model(Xtr[idx], hard=False).clamp(1e-6, 1-1e-6), Ytr[idx].float())
            else:              loss = bce(model(Xtr[idx]).clamp(1e-6, 1-1e-6), Ytr[idx].float())
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()*len(idx)
        if (ep+1) % max(1, epochs//4) == 0: print(f"  {kind} ep {ep+1:2d} loss {tot/len(Xtr):.4f}")

print("training Transformer ...");         tf = TF();               fit(tf, "tf", 28, 1e-3)
print("training plain carrier ...");       pc = PlainCarrier();     fit(pc, "pc", 40, 3e-3)
print("training conserving carrier ...");  cc = ConservingCarrier(); fit(cc, "cc", 40, 3e-3)

# ── evaluate four lines + reversibility of the conserving carrier ─────────────
tf.eval(); pc.eval(); cc.eval()
lengths = [32, 48, 64, 96, 128]
R = {k: {"acc": [], "cons": []} for k in ["int", "cc", "pc", "tf"]}; zero = []
rng_te = np.random.default_rng(123)
print("\n L    all-0    TF acc/cons     plainCarrier acc/cons    consCarrier acc/cons     integrable")
with torch.no_grad():
    for L in lengths:
        Xte, Yte = dataset(L, 300, rng_te); zero.append((Yte == 0).float().mean().item()*100)
        def ac(pred): return (pred == Yte).float().mean().item()*100, (pred.sum(1) == Xte.sum(1)).float().mean().item()*100
        ta, tc = ac(tf(Xte).argmax(-1))
        pa, pco = ac((pc(Xte) > 0.5).long())
        ca, cco = ac((cc(Xte, hard=True) > 0.5).long())
        R["int"]["acc"].append(100.0); R["int"]["cons"].append(100.0)
        R["tf"]["acc"].append(ta); R["tf"]["cons"].append(tc)
        R["pc"]["acc"].append(pa); R["pc"]["cons"].append(pco)
        R["cc"]["acc"].append(ca); R["cc"]["cons"].append(cco)
        print(f"{L:3d}  {zero[-1]:5.1f}%   {ta:5.1f}/{tc:5.1f}     {pa:5.1f}/{pco:5.1f}         {ca:5.1f}/{cco:5.1f}         100/100")

    # reversibility of the conserving carrier via the BBS mirror trick: inv = mirror.f.mirror
    def mirror(z): return torch.flip(z, dims=[1])
    ok = tot = 0
    rng_rv = np.random.default_rng(7)
    for L in lengths:
        Xrv, _ = dataset(L, 100, rng_rv)
        y = (cc(Xrv, hard=True) > 0.5).long()
        # invert 2 steps: each step inverse = mirror . step . mirror
        z = y.float()
        for _ in range(T_STEPS):
            zk, _ = cc.one(mirror(z), hard=True); z = mirror(zk)
        xrec = (z > 0.5).long()
        ok += (xrec == Xrv).all(dim=1).sum().item(); tot += len(Xrv)
    print(f"\nconserving-carrier exact reversibility (mirror trick), whole-sequence: {100*ok/tot:.1f}%")

# ── figure ───────────────────────────────────────────────────────────────────
CI, CCC, CPC, CTF, CZ = "#c0392b", "#e67e22", "#27ae60", "#2c3e50", "#95a5a6"
fig, ax = plt.subplots(1, 2, figsize=(13, 4.9))
for a, key in [(0, "acc"), (1, "cons")]:
    ax[a].plot(lengths, R["int"][key], "s-", color=CI,  label="1. hard-coded integrable (cheat)")
    ax[a].plot(lengths, R["cc"][key],  "D-", color=CCC, label="3. conserving carrier (Route C v2)")
    ax[a].plot(lengths, R["pc"][key],  "o-", color=CPC, label="2. plain carrier (free emit)")
    ax[a].plot(lengths, R["tf"][key],  "o-", color=CTF, label="4. Transformer")
ax[0].plot(lengths, zero, "--", color=CZ, lw=1, label="all-zeros baseline")
ax[0].axvline(32, ls=":", color="gray"); ax[0].text(33, 55, "trained only here", fontsize=8, color="gray")
ax[0].set_xlabel("lattice length (32 = trained, rest = OOD)"); ax[0].set_ylabel("per-position accuracy (%)")
ax[0].set_ylim(50, 102); ax[0].set_title("Accuracy vs length"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].set_xlabel("lattice length"); ax[1].set_ylabel("ball-count conserved (%)"); ax[1].set_ylim(-3, 105)
ax[1].set_title("Invariant (# balls) preserved?  #3 conserves by construction"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
fig.suptitle("Route C v2 — a LEARNED carrier that conserves by construction (four-way)", fontsize=12, y=1.02)
fig.tight_layout(); fig.savefig("03_conserving_carrier.png", dpi=130, bbox_inches="tight")
print("saved 03_conserving_carrier.png")
