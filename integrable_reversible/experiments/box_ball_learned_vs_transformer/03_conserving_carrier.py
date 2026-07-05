"""
Test 3 — conserving carrier on the FINITE-CARRIER Box-Ball System.

Plain BBS was dropped: welding conservation into the carrier makes its rule
trivial (a hard-coded emit = 1 - cell already scores 100% — see the appendix in
this folder's README). Finite-carrier BBS keeps the SAME conserving-carrier
structure but bounds the carrier at K balls: when the carrier is FULL, an
arriving ball must PASS THROUGH instead of being picked up. That makes the emit
decision depend on the carrier count k, so a carrier-blind gate (emit = 1 - cell)
can no longer solve it — the model must genuinely learn to use the carrier.

Four lines:
  1. hard-coded finite-carrier rule    -> ceiling
  2. carrier-blind gate (emit=1-cell)   -> proves the residual is non-trivial (~89%)
  3. conserving carrier (learned gate)  -> learns to use the carrier
  4. Transformer                        -> no structure, collapses OOD

Conservation is structural (k' = t - out, holds for any gate). The finite-carrier
BBS is verified conserved + reversible (mirror trick) before training. Trained on
L=32, tested on 48/64/96/128.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, math
import matplotlib.pyplot as plt
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)

K_CAP = 2; T_STEPS = 2; MAXSOL = 3

# ── finite-carrier BBS: the carrier holds at most K balls ────────────────────
def fc_step(s):
    out = np.zeros_like(s); u = 0
    for i in range(len(s)):
        if s[i] == 1:
            if u < K_CAP: u += 1; out[i] = 0    # pick up (carrier has room)
            else:         out[i] = 1            # carrier FULL -> ball passes through
        else:
            if u > 0: out[i] = 1; u -= 1        # drop one
            else:     out[i] = 0
    return out
def fc_run(s, T):
    s = s.copy()
    for _ in range(T): s = fc_step(s)
    return s
def mirror(s): return s[::-1].copy()
def fc_step_inv(s): return mirror(fc_step(mirror(s)))   # candidate inverse (mirror trick)

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
        x = make_config(L, rng); X[j] = x; Y[j] = fc_run(x, T_STEPS)
    return torch.tensor(X), torch.tensor(Y)

# ── verify the target dynamics is conserved + reversible (before any training) ─
print("finite-carrier BBS self-check (random configs, K=%d):" % K_CAP)
_rng = np.random.default_rng(1); _cons = _rev = True
for L in [32, 48, 64, 96, 128]:
    for _ in range(200):
        x = make_config(L, _rng); y = fc_step(x)
        if y.sum() != x.sum(): _cons = False
        if not np.array_equal(fc_step_inv(y), x): _rev = False
print(f"  ball-count conserved: {_cons}   reversible (mirror trick): {_rev}")
assert _cons and _rev, "finite-carrier BBS is not conserved/reversible — fix the rule"

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

# ── conserving carrier: conservation structural (k'=t-out); gate = learned OR blind ─
class ConservingCarrier(nn.Module):
    def __init__(s, hid=32, Kmax=6.0, blind=False):
        super().__init__(); s.Kmax = Kmax; s.blind = blind
        s.gate = nn.Sequential(nn.Linear(4, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
    def one(s, xf, hard):
        B, L = xf.shape; k = torch.zeros(B); xp = F.pad(xf, (1, 1)); outs = []
        for i in range(L):
            c = xf[:, i]
            if s.blind:
                e = 1.0 - c                        # carrier-blind fixed rule: no params, ignores k & neighbors
            else:
                feat = torch.stack([c, k / s.Kmax, xp[:, i], xp[:, i + 2]], dim=-1)
                e = torch.sigmoid(s.gate(feat).squeeze(-1))
                if hard:
                    eh = (e > 0.5).float(); e = eh + (e - e.detach())   # straight-through
            t = c + k; avail = torch.clamp(t, 0.0, 1.0)
            out_i = e * avail; k = t - out_i                            # conserves cell+carrier
            outs.append(out_i)
        return torch.stack(outs, dim=1), k
    def forward(s, x, hard=False):
        c = x.float()
        for _ in range(T_STEPS): c, _ = s.one(c, hard)
        return c

# ── train Transformer + conserving carrier on L=32 ───────────────────────────
rng = np.random.default_rng(0); Xtr, Ytr = dataset(32, 2500, rng); B = 256
ce = nn.CrossEntropyLoss(); bce = nn.BCELoss()

print("training Transformer ...")
tf = TF(); opt = torch.optim.AdamW(tf.parameters(), lr=1e-3)
for ep in range(28):
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), B):
        idx = perm[i:i+B]; loss = ce(tf(Xtr[idx]).reshape(-1, 2), Ytr[idx].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    if (ep+1) % 7 == 0: print(f"  TF ep {ep+1:2d} loss {loss.item():.4f}")

print("training conserving carrier ...")
cc = ConservingCarrier(); optc = torch.optim.Adam(cc.parameters(), lr=3e-3)
for ep in range(40):
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), B):
        idx = perm[i:i+B]; p = cc(Xtr[idx], hard=False)
        loss = bce(p.clamp(1e-6, 1-1e-6), Ytr[idx].float())
        optc.zero_grad(); loss.backward(); optc.step()
    if (ep+1) % 10 == 0: print(f"  CC ep {ep+1:2d} loss {loss.item():.4f}")

blind = ConservingCarrier(blind=True)   # no training

# ── evaluate four lines across lengths ───────────────────────────────────────
tf.eval(); cc.eval()
lengths = [32, 48, 64, 96, 128]
R = {k: {"acc": [], "cons": []} for k in ["ceil", "cc", "blind", "tf"]}
rng_te = np.random.default_rng(123)
print("\n L    ceiling      carrier-blind      conserving-carrier      Transformer   (acc / cons)")
with torch.no_grad():
    for L in lengths:
        Xte, Yte = dataset(L, 300, rng_te)
        def ac(pred): return (pred == Yte).float().mean().item()*100, (pred.sum(1) == Xte.sum(1)).float().mean().item()*100
        ba = ac((blind(Xte, hard=True) > 0.5).long())
        ca = ac((cc(Xte, hard=True) > 0.5).long())
        ta = ac(tf(Xte).argmax(-1))
        R["ceil"]["acc"].append(100.0); R["ceil"]["cons"].append(100.0)
        R["blind"]["acc"].append(ba[0]); R["blind"]["cons"].append(ba[1])
        R["cc"]["acc"].append(ca[0]);    R["cc"]["cons"].append(ca[1])
        R["tf"]["acc"].append(ta[0]);    R["tf"]["cons"].append(ta[1])
        print(f"{L:3d}   100/100     {ba[0]:5.1f}/{ba[1]:5.1f}      {ca[0]:5.1f}/{ca[1]:5.1f}         {ta[0]:5.1f}/{ta[1]:5.1f}")

    # reversibility of the learned conserving carrier (mirror trick, whole sequence)
    def tmirror(z): return torch.flip(z, dims=[1])
    ok = tot = 0; rng_rv = np.random.default_rng(7)
    for L in lengths:
        Xrv, _ = dataset(L, 100, rng_rv); y = (cc(Xrv, hard=True) > 0.5).float()
        z = y
        for _ in range(T_STEPS):
            zk, _ = cc.one(tmirror(z), hard=True); z = tmirror(zk)
        ok += ((z > 0.5).long() == Xrv).all(dim=1).sum().item(); tot += len(Xrv)
    print(f"\nconserving-carrier exact reversibility (mirror trick), whole-sequence: {100*ok/tot:.1f}%")

# ── figure ───────────────────────────────────────────────────────────────────
CI, CCC, CB, CTF = "#c0392b", "#1e8449", "#e67e22", "#2c3e50"
fig, ax = plt.subplots(1, 2, figsize=(13, 4.9))
for a, key in [(0, "acc"), (1, "cons")]:
    ax[a].plot(lengths, R["ceil"][key], "s-", color=CI,  label="hard-coded finite-carrier rule (ceiling)")
    ax[a].plot(lengths, R["cc"][key],   "D-", color=CCC, label="conserving carrier (learned)")
    ax[a].plot(lengths, R["blind"][key],"o--",color=CB,  label="carrier-blind gate (emit=1−cell)")
    ax[a].plot(lengths, R["tf"][key],   "o-", color=CTF, label="Transformer")
ax[0].axvline(32, ls=":", color="gray"); ax[0].text(33, 62, "trained only here", fontsize=8, color="gray")
ax[0].set_xlabel("lattice length (32 = trained, rest = OOD)"); ax[0].set_ylabel("per-position accuracy (%)")
ax[0].set_ylim(60, 102); ax[0].set_title("Accuracy vs length"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].set_xlabel("lattice length"); ax[1].set_ylabel("ball-count conserved (%)"); ax[1].set_ylim(-3, 105)
ax[1].set_title("Invariant preserved?  conserving models = structural"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
fig.suptitle("Test 3 — finite-carrier BBS: the carrier-blind gate can't solve it (~89%); the learned gate must use the carrier",
             fontsize=11.5, y=1.02)
fig.tight_layout(); fig.savefig("03_conserving_carrier.png", dpi=130, bbox_inches="tight")
print("saved 03_conserving_carrier.png")
