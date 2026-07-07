"""
Step 3 (multi-seed rigor) — the formal Exp-2 numbers.

Reruns the probe across N_SEEDS seeds and against THREE fair free-form baselines
(bidirectional GRU, bidirectional LSTM, a small Transformer as the single-step
learner) — to show the drift is not GRU-specific and to report mean ± std.

Setup (same as the probe): everything is trained ONLY on single (phase-aware)
steps on a small ring, then COMPOSED to T = L/2 on growing rings. The structural
block-CA masks its output to same-ball-count states, so conservation is exact by
construction; every free-form model is free to emit any bit.

Claim under test: as L (and T) grow, structural conservation stays exact (100 ± 0)
while every fair free-form model's ball-count drifts — because it learns the local
step *well but not exactly*, and T ∝ L composition amplifies the residual.

Env: MS_QUICK=1 for a fast smoke run (2 seeds, fewer epochs, shorter lengths).
"""
import os, json, numpy as np, torch, torch.nn as nn, math
import matplotlib.pyplot as plt

QUICK = os.environ.get("MS_QUICK") == "1"
N_SEEDS = 2 if QUICK else 5
TEST_L  = [48, 96, 192] if QUICK else [48, 96, 192, 384]
N_EVAL  = 120
TRAIN_L = 48
BLK = 3
torch.set_num_threads(4)

# ── the rule (in sync with 01_margolus_system.py) ────────────────────────────
def _bits(v): return ((v >> 2) & 1, (v >> 1) & 1, v & 1)
def _rot(cycle): return {v: cycle[(i + 1) % len(cycle)] for i, v in enumerate(cycle)}
PHI = {v: v for v in range(8)}; PHI.update(_rot([4, 2, 1])); PHI.update(_rot([6, 5, 3]))
COUNT = [sum(_bits(v)) for v in range(8)]
def apply_blocks(s, offset, table):
    L = len(s); out = s.copy()
    for start in range(offset, offset + L, BLK):
        idx = [(start + j) % L for j in range(BLK)]
        nb = _bits(table[(s[idx[0]] << 2) | (s[idx[1]] << 1) | s[idx[2]]])
        for j in range(BLK): out[idx[j]] = nb[j]
    return out
def run(s, T):
    s = s.copy()
    for t in range(T): s = apply_blocks(s, t % BLK, PHI)
    return s
def cfg(L, rng, d=0.45): return (rng.random(L) < d).astype(np.int64)

def block_onehot(x, offset):
    B, L = x.shape
    idx = (torch.arange(offset, offset + L) % L)
    v = (x[:, idx].view(B, L // BLK, BLK) * torch.tensor([4, 2, 1])).sum(-1).long()
    return torch.nn.functional.one_hot(v, 8).float(), idx

# ── models ───────────────────────────────────────────────────────────────────
class StructCA(nn.Module):
    """learns block map φ; output masked to same-count states ⇒ conservation structural."""
    def __init__(s):
        super().__init__(); s.W = nn.Parameter(torch.zeros(8, 8))
        mask = torch.tensor([[1.0 if COUNT[i] == COUNT[j] else 0.0 for j in range(8)] for i in range(8)])
        s.register_buffer("mask", mask)
        s.register_buffer("bitsT", torch.tensor([_bits(j) for j in range(8)], dtype=torch.float))
    def step(s, x, offset, hard=False):
        B, L = x.shape
        oh, idx = block_onehot((x > 0.5).long() if x.dtype == torch.float else x, offset)
        outdist = oh @ torch.softmax(s.W.masked_fill(s.mask == 0, -1e9), 1)
        bits = s.bitsT[outdist.argmax(-1)] if hard else outdist @ s.bitsT
        out = torch.zeros(B, L); out[:, idx] = bits.reshape(B, L)
        return out

class ComposingRNN(nn.Module):
    """bidirectional GRU/LSTM as a phase-aware single-step learner; composed at eval."""
    def __init__(s, kind="gru", d=64):
        super().__init__()
        R = nn.GRU if kind == "gru" else nn.LSTM
        s.rnn = R(1 + BLK, d, batch_first=True, bidirectional=True); s.head = nn.Linear(2 * d, 1)
    def step(s, x, offset, hard=False):
        B, L = x.shape
        ph = torch.zeros(B, L, BLK); ph[..., offset] = 1.0
        h, _ = s.rnn(torch.cat([x.float().unsqueeze(-1), ph], -1))
        return torch.sigmoid(s.head(h)).squeeze(-1)

class PE(nn.Module):
    def __init__(s, d, maxlen=8192):
        super().__init__(); pe = torch.zeros(maxlen, d); pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div); s.register_buffer("pe", pe)
    def forward(s, x): return x + s.pe[:x.size(1)].unsqueeze(0)
class ComposingTransformer(nn.Module):
    """small Transformer as a phase-aware single-step learner (sees the whole block via attention)."""
    def __init__(s, d=64, heads=4, layers=2):
        super().__init__(); s.emb = nn.Embedding(2, d); s.ph = nn.Embedding(BLK, d); s.pe = PE(d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, dropout=0.0, activation="gelu")
        s.enc = nn.TransformerEncoder(enc, layers); s.head = nn.Linear(d, 1)
    def step(s, x, offset, hard=False):
        B, L = x.shape
        h = s.pe(s.emb((x > 0.5).long()) + s.ph(torch.full((B, L), offset)))
        return torch.sigmoid(s.head(s.enc(h))).squeeze(-1)

MODELS = {
    "structural block-CA": lambda: StructCA(),
    "bi-GRU (free-form)":  lambda: ComposingRNN("gru"),
    "bi-LSTM (free-form)": lambda: ComposingRNN("lstm"),
    "Transformer (free-form)": lambda: ComposingTransformer(),
}
# per-model training budget. The free-form baselines get AMPLE (equal, larger) budget on
# purpose — tuned so each reaches ~99% SINGLE-STEP accuracy (a fair shot at the local rule).
# The structural CA needs only 40 epochs to hit an exact block map. Point: even with 3× the
# training budget and a near-perfect single step, free-form conservation drifts under T∝L.
CFG = {
    "structural block-CA":     dict(epochs=40,  lr=5e-2, clip=False),
    "bi-GRU (free-form)":      dict(epochs=120, lr=3e-3, clip=True),
    "bi-LSTM (free-form)":     dict(epochs=120, lr=2e-3, clip=True),
    "Transformer (free-form)": dict(epochs=120, lr=1e-3, clip=True),
}
if QUICK:
    for c in CFG.values(): c["epochs"] = min(c["epochs"], 20)

def step_data(N, rng):
    X = np.zeros((N, TRAIN_L), np.int64); Y = np.zeros((N, TRAIN_L), np.int64); P = np.zeros(N, np.int64)
    for j in range(N):
        x = cfg(TRAIN_L, rng); p = int(rng.integers(0, BLK))
        X[j] = x; P[j] = p; Y[j] = apply_blocks(x, p, PHI)
    return torch.tensor(X), torch.tensor(Y), torch.tensor(P)

def compose(m, X, T, hard=True):
    c = X.float()
    for t in range(T): c = (m.step(c, t % BLK, hard=hard) > 0.5).float()
    return c.long()

def compose_conserving(m, X, T):
    """free-form rollout with a BOLTED-ON conservation projection: each step keep the N
    highest-probability cells (N = input ball-count, the conserved quantity) → total count
    is forced exact every step. Same trained model as compose(), only the emit rule differs —
    isolates 'what does bolting conservation onto a free-form model actually buy?'."""
    n = X.sum(1); c = X.float()
    for t in range(T):
        prob = m.step(c, t % BLK)
        ranks = prob.argsort(1, descending=True).argsort(1)   # 0 = highest prob
        c = (ranks < n.unsqueeze(1)).float()                  # exactly n ones per row
    return c.long()

def one_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed); Xtr, Ytr, Ptr = step_data(4000, rng); B = 256
    bce = nn.BCELoss()
    out = {}
    for name, ctor in MODELS.items():
        cf = CFG[name]; m = ctor(); opt = torch.optim.Adam(m.parameters(), lr=cf["lr"])
        for ep in range(cf["epochs"]):
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), B):
                idx = perm[i:i + B]; xb, yb, pb = Xtr[idx], Ytr[idx].float(), Ptr[idx]
                loss = 0.0
                for p in range(BLK):
                    mp = pb == p
                    if mp.sum() == 0: continue
                    loss = loss + bce(m.step(xb[mp], p).clamp(1e-6, 1 - 1e-6), yb[mp])
                opt.zero_grad(); loss.backward()
                if cf["clip"]: torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
        m.eval(); is_ff = not isinstance(m, StructCA)
        rng_te = np.random.default_rng(1000 + seed)
        rec = {"acc": [], "cons": [], "acc_pc": [], "cons_pc": [], "ss_acc": None, "ss_cons": None}
        with torch.no_grad():
            # single-step diagnostic: did the model learn ONE (phase-aware) step? (honesty:
            # free-form drift below is from composing this residual, not from failing to learn)
            Xd, Yd, Pd = step_data(1500, np.random.default_rng(500 + seed)); sa = sc = 0.0
            for p in range(BLK):
                mp = Pd == p
                if mp.sum() == 0: continue
                sp = (m.step(Xd[mp], p, hard=True) > 0.5).long(); w = mp.float().mean().item()
                sa += (sp == Yd[mp]).float().mean().item() * w
                sc += (sp.sum(1) == Xd[mp].sum(1)).float().mean().item() * w
            rec["ss_acc"] = sa * 100; rec["ss_cons"] = sc * 100
            for L in TEST_L:
                T = L // 2
                Xte = torch.tensor(np.stack([cfg(L, rng_te) for _ in range(N_EVAL)]))
                Yte = torch.tensor(np.stack([run(x.numpy(), T) for x in Xte]))
                pred = compose(m, Xte, T)
                rec["acc"].append((pred == Yte).float().mean().item() * 100)
                rec["cons"].append((pred.sum(1) == Xte.sum(1)).float().mean().item() * 100)
                if is_ff:                                   # same model, + bolted-on conservation
                    pc = compose_conserving(m, Xte, T)
                    rec["acc_pc"].append((pc == Yte).float().mean().item() * 100)
                    rec["cons_pc"].append((pc.sum(1) == Xte.sum(1)).float().mean().item() * 100)
        out[name] = rec
        print(f"  seed {seed}  {name:24s}  [1-step {rec['ss_acc']:5.1f}/{rec['ss_cons']:5.1f}]  " +
              "  ".join(f"{a:5.1f}/{c:5.1f}" for a, c in zip(rec["acc"], rec["cons"])))
    return out

# ── run all seeds, aggregate ─────────────────────────────────────────────────
print(f"Margolus multi-seed: {N_SEEDS} seeds, compose T=L/2, lengths {TEST_L}\n")
per_seed = []
for sd in range(N_SEEDS):
    print(f"seed {sd}:"); per_seed.append(one_seed(sd))

agg = {}
for name in MODELS:
    accs = np.array([[ps[name]["acc"] for ps in per_seed]]).reshape(N_SEEDS, len(TEST_L))
    cons = np.array([[ps[name]["cons"] for ps in per_seed]]).reshape(N_SEEDS, len(TEST_L))
    ssa = np.array([ps[name]["ss_acc"] for ps in per_seed]); ssc = np.array([ps[name]["ss_cons"] for ps in per_seed])
    a = {"acc_mean": accs.mean(0).tolist(), "acc_std": accs.std(0).tolist(),
         "cons_mean": cons.mean(0).tolist(), "cons_std": cons.std(0).tolist(),
         "ss_acc_mean": float(ssa.mean()), "ss_acc_std": float(ssa.std()),
         "ss_cons_mean": float(ssc.mean()), "ss_cons_std": float(ssc.std())}
    if per_seed[0][name]["acc_pc"]:      # free-form models also have a +conservation variant
        apc = np.array([[ps[name]["acc_pc"] for ps in per_seed]]).reshape(N_SEEDS, len(TEST_L))
        cpc = np.array([[ps[name]["cons_pc"] for ps in per_seed]]).reshape(N_SEEDS, len(TEST_L))
        a.update(acc_pc_mean=apc.mean(0).tolist(), acc_pc_std=apc.std(0).tolist(),
                 cons_pc_mean=cpc.mean(0).tolist(), cons_pc_std=cpc.std(0).tolist())
    agg[name] = a

print(f"\n{'model':24s}  " + "   ".join(f"L={L}" for L in TEST_L) + "   (acc% mean±std)")
for name in MODELS:
    print(f"{name:24s}  " + "   ".join(f"{m:4.0f}±{s:<3.0f}" for m, s in zip(agg[name]["acc_mean"], agg[name]["acc_std"])))
print(f"\n{'model':24s}  " + "   ".join(f"L={L}" for L in TEST_L) + "   (conserved% mean±std)")
for name in MODELS:
    print(f"{name:24s}  " + "   ".join(f"{m:4.0f}±{s:<3.0f}" for m, s in zip(agg[name]["cons_mean"], agg[name]["cons_std"])))
print(f"\n{'model':24s}  SINGLE-STEP acc / cons (mean±std)  — free-form learns the step well; composing T∝L is what drifts")
for name in MODELS:
    a = agg[name]; print(f"{name:24s}  {a['ss_acc_mean']:5.1f}±{a['ss_acc_std']:<4.1f} / {a['ss_cons_mean']:5.1f}±{a['ss_cons_std']:<4.1f}")

print(f"\n+CONSERVATION bolted onto the SAME free-form model (top-N emit) — cons is forced ~100; does ACCURACY recover?")
print(f"{'model':24s}  " + "   ".join(f"L={L}" for L in TEST_L) + "   (acc% mean±std)   [cons L48→L{}]".format(TEST_L[-1]))
for name in MODELS:
    a = agg[name]
    if "acc_pc_mean" not in a: continue
    accs = "   ".join(f"{m:4.0f}±{s:<3.0f}" for m, s in zip(a["acc_pc_mean"], a["acc_pc_std"]))
    print(f"{name:24s}  {accs}   [{a['cons_pc_mean'][0]:.0f}→{a['cons_pc_mean'][-1]:.0f}]")

json.dump({"lengths": TEST_L, "n_seeds": N_SEEDS, "agg": agg},
          open("margolus_multiseed_results.json", "w"), indent=1)

# ── figures ──────────────────────────────────────────────────────────────────
style = {"structural block-CA": ("#1e8449", "D-", 2.2), "bi-GRU (free-form)": ("#e67e22", "s-", 1.7),
         "bi-LSTM (free-form)": ("#8e44ad", "o-", 1.7), "Transformer (free-form)": ("#2980b9", "o--", 1.5)}
for key, ylab, ylim, fname, ttl in [
    ("acc",  "per-position accuracy (%)", (40, 103), "03_accuracy.png",     "Accuracy vs length"),
    ("cons", "ball-count conserved (%)",  (-3, 105), "03_conservation.png", "Conservation vs length")]:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in MODELS:
        c, ls, lw = style[name]
        ax.errorbar(TEST_L, agg[name][f"{key}_mean"], yerr=agg[name][f"{key}_std"],
                    fmt=ls, color=c, lw=lw, capsize=3, label=name)
    ax.axvline(TRAIN_L, ls=":", color="gray"); ax.text(TRAIN_L + 2, ylim[0] + 5, "train length", fontsize=8, color="gray")
    ax.set_xlabel("ring length L  (compose T = L/2 steps)"); ax.set_ylabel(ylab); ax.set_ylim(*ylim)
    ax.set_title(f"Margolus block CA — {ttl} ({N_SEEDS} seeds)", fontsize=11); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fname, dpi=130, bbox_inches="tight"); print(f"saved {fname}")

# bolt-on figure (accuracy): does forcing conservation onto a free-form model rescue accuracy?
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(TEST_L, agg["structural block-CA"]["acc_mean"], yerr=agg["structural block-CA"]["acc_std"],
            fmt="D-", color="#1e8449", lw=2.2, capsize=3, label="structural block-CA (conservation intrinsic)")
for name, col in [("bi-GRU (free-form)", "#e67e22"), ("bi-LSTM (free-form)", "#8e44ad"), ("Transformer (free-form)", "#2980b9")]:
    if "acc_pc_mean" not in agg[name]: continue
    ax.plot(TEST_L, agg[name]["acc_mean"], "o:", color=col, lw=1.2, alpha=0.6, label=f"{name.split(' ')[0]} — free")
    ax.errorbar(TEST_L, agg[name]["acc_pc_mean"], yerr=agg[name]["acc_pc_std"], fmt="s-", color=col, lw=1.7, capsize=3, label=f"{name.split(' ')[0]} — + bolted-on conservation")
ax.axvline(TRAIN_L, ls=":", color="gray"); ax.text(TRAIN_L + 2, 44, "train length", fontsize=8, color="gray")
ax.set_xlabel("ring length L  (compose T = L/2 steps)"); ax.set_ylabel("per-position accuracy (%)"); ax.set_ylim(40, 103)
ax.set_title(f"Bolting conservation onto a free-form model ({N_SEEDS} seeds)\n(+conservation forces ball-count ~100 for all; accuracy shown)", fontsize=10)
ax.legend(fontsize=7.5); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig("03_bolt_on.png", dpi=130, bbox_inches="tight"); print("saved 03_bolt_on.png")
