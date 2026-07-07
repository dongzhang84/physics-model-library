"""
Exp 3 · Step 2 (probe, single seed) — does the integrable structure buy time
extrapolation that even the strongest bolt-on can't?

Periodic Toda (bounded quasi-periodic motion). Task: given s(0), predict s(t).
Train on t ≤ T_train, test on t ∈ (T_train, T_max]  —  TIME extrapolation (the
analogue of BBS/Margolus length extrapolation).

Three models, matched training budget:
  • structural  = learnable ACTION–ANGLE model: encode s0 → (I, φ0); evolve I fixed,
                  φ(t)=φ0+ω(I)·t; decode (I, cosφ, sinφ) → s(t). The integrable
                  inductive bias (conserved actions + linear angles) is baked into
                  the architecture, but the change of variables is LEARNED — no
                  eigendecomposition, no hard-coded IST.
  • free-form   = direct map (s0, t) → s(t), no structure.
  • bolt-on     = free-form + the STRONGEST conservation constraint: penalise all N
                  conserved quantities tr(L^k), k=1..N (polynomial in the state,
                  differentiable — the practical "pin every conserved quantity").

Pre-registered claim: structural error stays ~flat as t grows; free-form drifts;
and bolt-on — even with every conserved quantity pinned — ALSO drifts, because it
gets the conserved values right but the ANGLE (phase) wrong. If bolt-on catches up
(F1) or structural also drifts (F2), that is the honest negative result.
"""
import numpy as np, torch, torch.nn as nn
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)

N = 8; DT = 0.01; T_TRAIN = 5.0; T_MAX = 20.0
QAMP, PAMP = 0.3, 0.6            # moderate amplitude → genuinely nonlinear (not just phonons)

# ── periodic Toda ground truth ───────────────────────────────────────────────
def force_p(q):
    b = np.exp(q - np.roll(q, -1))          # b_n = e^{q_n - q_{n+1}} (cyclic)
    return np.roll(b, 1) - b                # F_n = b_{n-1} - b_n
def verlet_p(q, p, steps):
    Q = np.empty((steps+1, N)); P = np.empty((steps+1, N)); Q[0]=q; P[0]=p; F = force_p(q)
    for k in range(steps):
        p = p + 0.5*DT*F; q = q + DT*p; F = force_p(q); p = p + 0.5*DT*F
        Q[k+1]=q; P[k+1]=p
    return Q, P
def state(q, p):                            # translation-invariant: (bond, momentum), dim 2N
    return np.concatenate([q - np.roll(q, -1), p], -1)

def make_ic(rng):
    q = QAMP*rng.standard_normal(N); q -= q.mean()
    p = PAMP*rng.standard_normal(N); p -= p.mean()      # zero total momentum
    return q, p

def dataset(n_ic, rng):
    S0=[]; T=[]; ST=[]
    steps = int(T_MAX/DT)
    for _ in range(n_ic):
        q, p = make_ic(rng); Q, P = verlet_p(q, p, steps); s0 = state(Q[0], P[0])
        ks = rng.integers(1, steps+1, size=25)
        for k in ks:
            S0.append(s0); T.append(k*DT); ST.append(state(Q[k], P[k]))
    return (torch.tensor(np.array(S0), dtype=torch.float32),
            torch.tensor(np.array(T),  dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(ST), dtype=torch.float32))

# ── conserved quantities tr(L^k) in torch (differentiable; no eig) ────────────
def trLk_batch(s):
    d = s[:, :N]; p = s[:, N:]                          # bond, momentum
    a = 0.5*torch.exp(d/2); b = -0.5*p; B = s.shape[0]
    L = torch.zeros(B, N, N)
    idx = torch.arange(N)
    L[:, idx, idx] = b
    for n in range(N):
        L[:, n, (n+1) % N] += a[:, n]; L[:, (n+1) % N, n] += a[:, n]
    M = L.clone(); outs = []
    for _ in range(N):                                  # tr(L^1..L^N)
        outs.append(M.diagonal(dim1=1, dim2=2).sum(-1)); M = M @ L
    return torch.stack(outs, 1)                         # (B, N)

# ── models ───────────────────────────────────────────────────────────────────
def mlp(i, o, h=128, n=3):
    layers = [nn.Linear(i, h), nn.GELU()]
    for _ in range(n-1): layers += [nn.Linear(h, h), nn.GELU()]
    return nn.Sequential(*layers, nn.Linear(h, o))

class Structural(nn.Module):                            # learnable action-angle
    def __init__(s, K=N):
        super().__init__(); s.K = K
        s.enc = mlp(2*N, 2*K); s.omega = mlp(K, K, h=64, n=2); s.dec = mlp(3*K, 2*N)
    def forward(s, s0, t):
        z = s.enc(s0); I, phi0 = z[:, :s.K], z[:, s.K:]
        phi = phi0 + s.omega(I)*t                       # linear angle motion → extrapolates in t
        return s.dec(torch.cat([I, torch.cos(phi), torch.sin(phi)], -1))
class FreeForm(nn.Module):                              # direct map, no structure
    def __init__(s): super().__init__(); s.net = mlp(2*N+1, 2*N)
    def forward(s, s0, t): return s.net(torch.cat([s0, t], -1))

# ── train / eval ─────────────────────────────────────────────────────────────
def fit(m, S0, T, ST, epochs=250, lr=2e-3, cons_pen=0.0):
    opt = torch.optim.Adam(m.parameters(), lr=lr); mse = nn.MSELoss(); Btr = 512
    tr = T.squeeze(1) <= T_TRAIN                        # train only on t <= T_train
    S0, T, ST = S0[tr], T[tr], ST[tr]
    trg = trLk_batch(S0).detach() if cons_pen > 0 else None
    for ep in range(epochs):
        perm = torch.randperm(len(S0))
        for i in range(0, len(S0), Btr):
            idx = perm[i:i+Btr]; pred = m(S0[idx], T[idx]); loss = mse(pred, ST[idx])
            if cons_pen > 0:                            # pin ALL N conserved quantities (strongest bolt-on)
                loss = loss + cons_pen*mse(trLk_batch(pred), trg[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return m

def evaluate(m, S0, T, ST):
    m.eval()
    with torch.no_grad():
        pred = m(S0, T); tt = T.squeeze(1)
        serr = ((pred-ST)**2).mean(1)                  # per-sample state MSE
        cerr = ((trLk_batch(pred)-trLk_batch(S0))**2).mean(1)   # conserved-quantity error
        buckets = [(0, T_TRAIN, "train t≤5"), (T_TRAIN, 10, "t 5–10"), (10, 15, "t 10–15"), (15, T_MAX, "t 15–20")]
        rows = []
        for lo, hi, name in buckets:
            msk = (tt > lo) & (tt <= hi) if lo > 0 else (tt <= hi)
            rows.append((name, serr[msk].mean().item(), cerr[msk].mean().item()))
    return rows

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    S0, T, ST = dataset(300, rng); S0te, Tte, STte = dataset(80, np.random.default_rng(1))
    print(f"periodic Toda N={N}, train t≤{T_TRAIN}, test t≤{T_MAX}, moderate amplitude "
          f"(state var={ST.var().item():.3f})\n")
    models = {
        "structural (action-angle)": (Structural(), 0.0),
        "free-form":                 (FreeForm(),    0.0),
        "bolt-on (all N cons pinned)":(FreeForm(),   1.0),
    }
    res = {}
    for name, (m, pen) in models.items():
        print(f"training {name} ...")
        fit(m, S0, T, ST, cons_pen=pen); res[name] = evaluate(m, S0te, Tte, STte)
    print(f"\n{'model':30s} " + "".join(f"{n:>12}" for n,_,_ in res['free-form']) + "   (state MSE; predict-mean ≈ {:.3f})".format(ST.var().item()))
    for name in res:
        print(f"{name:30s} " + "".join(f"{s:12.4f}" for _, s, _ in res[name]))
    print(f"\n{'model':30s} " + "".join(f"{n:>12}" for n,_,_ in res['free-form']) + "   (conserved-qty error)")
    for name in res:
        print(f"{name:30s} " + "".join(f"{c:12.4f}" for _, _, c in res[name]))
    print("\nRead: does structural state-MSE stay flat while free-form/bolt-on grow with t?  "
          "Does bolt-on keep conserved-qty error low yet still drift in state?")
