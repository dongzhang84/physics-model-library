"""
Step 2 (probe, single seed) — does the discriminator appear under T ∝ L?

Margolus' one-step block rule is LOCAL, so unlike BBS a fixed small horizon would
let any model nail it. To recreate a length-generalization challenge we let the
horizon grow with the ring: at test length L we compose T = L/2 steps. Everything
is trained ONLY on single-step (phase-aware) supervision — cheap, no deep unroll —
then COMPOSED to large T at eval. This is the purest test of "does composition
extrapolate": learn one step, apply it many times on a bigger ring.

Two models, matched at single-step training:
  * structural block-CA  — learns the 8-entry block map φ, but its output is
    MASKED to same-ball-count states, so conservation is exact BY CONSTRUCTION
    for any learned map. Offset (the Margolus phase) is structural, not learned.
  * composing free-form GRU — a plain GRU given the phase as an input channel,
    free to emit any bit. No structural conservation.

Question: as L (and T) grow, does the free-form model's ball-count drift while the
structural one stays exact — reproducing the Box-Ball finding on a different
(block, no-carrier) reversible system? Single seed → trend signal only.
"""
import numpy as np, torch, torch.nn as nn
from importlib import import_module
sys_mod = import_module("01_margolus_system") if False else None  # (01 has a leading digit; inline the rule)

torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)

# ── the rule (kept in sync with 01_margolus_system.py) ───────────────────────
BLK = 3
def _bits(v): return ((v >> 2) & 1, (v >> 1) & 1, v & 1)
def _val(t):  return (t[0] << 2) | (t[1] << 1) | t[2]
def _rot(cycle):
    return {v: cycle[(i + 1) % len(cycle)] for i, v in enumerate(cycle)}
PHI = {v: v for v in range(8)}; PHI.update(_rot([4, 2, 1])); PHI.update(_rot([6, 5, 3]))
PHI_INV = {o: i for i, o in PHI.items()}
COUNT = [sum(_bits(v)) for v in range(8)]

def apply_blocks(s, offset, table):
    L = len(s); out = s.copy()
    for start in range(offset, offset + L, BLK):
        idx = [(start + j) % L for j in range(BLK)]
        nb = _bits(table[_val((s[idx[0]], s[idx[1]], s[idx[2]]))])
        for j in range(BLK): out[idx[j]] = nb[j]
    return out
def run(s, T):
    s = s.copy()
    for t in range(T): s = apply_blocks(s, t % BLK, PHI)
    return s
def cfg(L, rng, d=0.45): return (rng.random(L) < d).astype(np.int64)

# ── single-step (phase-aware) training data on a small ring ──────────────────
TRAIN_L = 48
def step_data(N, rng):
    X = np.zeros((N, TRAIN_L), np.int64); Y = np.zeros((N, TRAIN_L), np.int64); P = np.zeros(N, np.int64)
    for j in range(N):
        x = cfg(TRAIN_L, rng); p = int(rng.integers(0, BLK))
        X[j] = x; P[j] = p; Y[j] = apply_blocks(x, p, PHI)
    return torch.tensor(X), torch.tensor(Y), torch.tensor(P)

# ── models ───────────────────────────────────────────────────────────────────
def block_onehot(x, offset):
    """x: (B,L) hard 0/1 -> (B, L/BLK, 8) one-hot of each block at this offset, and the block index map."""
    B, L = x.shape
    idx = (torch.arange(offset, offset + L) % L)                      # (L,)
    xr = x[:, idx].view(B, L // BLK, BLK)                             # (B, nblk, 3)
    v = (xr[..., 0] * 4 + xr[..., 1] * 2 + xr[..., 2]).long()         # (B, nblk)
    return torch.nn.functional.one_hot(v, 8).float(), idx

class StructCA(nn.Module):
    """learns block map φ; output masked to same-count states ⇒ conservation structural."""
    def __init__(s):
        super().__init__()
        s.W = nn.Parameter(torch.zeros(8, 8))
        mask = torch.zeros(8, 8)
        for i in range(8):
            for j in range(8):
                mask[i, j] = 1.0 if COUNT[i] == COUNT[j] else 0.0     # only same ball-count outputs
        s.register_buffer("mask", mask)
        # decode: state j -> its 3 bits
        s.register_buffer("bitsT", torch.tensor([_bits(j) for j in range(8)], dtype=torch.float))  # (8,3)
    def map_soft(s):
        logit = s.W.masked_fill(s.mask == 0, -1e9)
        return torch.softmax(logit, dim=1)                            # (8,8) rows sum to 1 over same-count
    def one(s, x, offset, hard):
        B, L = x.shape
        oh, idx = block_onehot(x, offset)                             # (B,nblk,8)
        P = s.map_soft()
        outdist = oh @ P                                              # (B,nblk,8) distribution over output states
        if hard:
            j = outdist.argmax(-1)                                    # (B,nblk)
            bits = s.bitsT[j]                                         # (B,nblk,3)
        else:
            bits = outdist @ s.bitsT                                  # (B,nblk,3) soft prob of 1 per cell
        flat = bits.reshape(B, L)                                     # blocks are contiguous in permuted order
        out = torch.zeros(B, L, device=x.device)
        out[:, idx] = flat                                           # scatter back to original positions
        return out
    def rollout(s, x, T, hard):
        c = x.float()
        for t in range(T):
            c = s.one((c > 0.5).long() if hard else c, t % BLK, hard) if hard else s.one(c, t % BLK, False)
        return c

class ComposingGRU(nn.Module):
    """free-form single-step learner; BIDIRECTIONAL (sees the whole block); phase fed as
    an input channel; composed at eval. Bidirectional is the FAIR prior here — a Margolus
    block map needs intra-block context, unlike BBS's naturally-causal left→right carrier."""
    def __init__(s, d=64):
        super().__init__(); s.gru = nn.GRU(1 + BLK, d, batch_first=True, bidirectional=True); s.head = nn.Linear(2 * d, 1)
    def one(s, x, offset):
        B, L = x.shape
        ph = torch.zeros(B, L, BLK, device=x.device); ph[..., offset] = 1.0
        inp = torch.cat([x.unsqueeze(-1), ph], -1)
        h, _ = s.gru(inp); return torch.sigmoid(s.head(h)).squeeze(-1)
    def rollout(s, x, T):
        c = x.float()
        for t in range(T): c = s.one((c > 0.5).float(), t % BLK)
        return c

# ── train both on single-step supervision ────────────────────────────────────
rng = np.random.default_rng(0); Xtr, Ytr, Ptr = step_data(4000, rng); B = 256
bce = nn.BCELoss()
def fit_step(m, kind, epochs, lr):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), B):
            idx = perm[i:i+B]; xb, yb, pb = Xtr[idx], Ytr[idx].float(), Ptr[idx]
            # single step: group by phase (small #phases) to reuse the offset machinery
            loss = 0.0
            for p in range(BLK):
                m_p = pb == p
                if m_p.sum() == 0: continue
                if kind == "ca":  pred = m.one(xb[m_p], p, hard=False)
                else:             pred = m.one(xb[m_p].float(), p)
                loss = loss + bce(pred.clamp(1e-6, 1-1e-6), yb[m_p])
            opt.zero_grad(); loss.backward(); opt.step()
    return m
print(f"training on single-step (phase-aware) supervision, ring L={TRAIN_L} ...")
ca  = fit_step(StructCA(),      "ca",  40, 5e-2)
gru = fit_step(ComposingGRU(),  "gru", 40, 3e-3)

# did StructCA recover the true permutation?
learned = ca.map_soft().argmax(1).tolist()
true    = [PHI[i] for i in range(8)]
print("  StructCA learned block map == true φ :", learned == true, "  (learned:", learned, ")")

# ── diagnostic: SINGLE-STEP accuracy (is the GRU even learning one step?) ─────
rng_d = np.random.default_rng(77); Xd, Yd, Pd = step_data(1500, rng_d)
with torch.no_grad():
    ca_ss = gru_ss = ca_ssc = gru_ssc = 0.0
    for p in range(BLK):
        m = Pd == p
        if m.sum() == 0: continue
        cp = (ca.one(Xd[m], p, hard=True) > 0.5).long()
        gp = (gru.one(Xd[m].float(), p) > 0.5).long()
        w = m.float().mean().item()
        ca_ss  += (cp == Yd[m]).float().mean().item() * w
        gru_ss += (gp == Yd[m]).float().mean().item() * w
        ca_ssc  += (cp.sum(1) == Xd[m].sum(1)).float().mean().item() * w
        gru_ssc += (gp.sum(1) == Xd[m].sum(1)).float().mean().item() * w
print(f"  SINGLE-STEP  per-cell acc / conserved:  StructCA {ca_ss*100:5.1f}/{ca_ssc*100:5.1f}   "
      f"GRU {gru_ss*100:5.1f}/{gru_ssc*100:5.1f}")

# ── eval: compose T = L/2 on growing rings ───────────────────────────────────
TEST_L = [48, 96, 192, 384]
rng_te = np.random.default_rng(123)
print(f"\n{'L':>5} {'T=L/2':>6}   {'StructCA acc/cons':>20}   {'GRU acc/cons':>16}")
ca.eval(); gru.eval()
with torch.no_grad():
    for L in TEST_L:
        T = L // 2
        Xte = torch.tensor(np.stack([cfg(L, rng_te) for _ in range(200)]))
        Yte = torch.tensor(np.stack([run(x.numpy(), T) for x in Xte]))
        for name, pred in [("ca", (ca.rollout(Xte, T, hard=True) > 0.5).long()),
                           ("gru", (gru.rollout(Xte, T) > 0.5).long())]:
            acc = (pred == Yte).float().mean().item() * 100
            cons = (pred.sum(1) == Xte.sum(1)).float().mean().item() * 100
            if name == "ca":  ca_s = f"{acc:5.1f}/{cons:5.1f}"
            else:             gru_s = f"{acc:5.1f}/{cons:5.1f}"
        print(f"{L:>5} {T:>6}   {ca_s:>20}   {gru_s:>16}")
print("\n(structural conservation should stay exact for any learned map; the GRU's "
      "ball-count is only as good as its single-step accuracy, compounded over T steps.)")
