"""
Exp 3 · Step 2b (probe, single seed) — Neural-Lax: learn the isospectral Lax-flow
generator instead of going to spectral coordinates (attempt 2, after v1's F2).

State ↔ Lax matrix L (periodic Jacobi) via Flaschka — trivial both ways, so NO
inverse-spectral decoder (v1's R2 obstacle is sidestepped). The step is an exactly
ISOSPECTRAL orthogonal conjugation:
        L(t+Δ) = Q L(t) Qᵀ,   Q = exp(Δ · B_θ(L)),   B_θ skew
so every conserved quantity tr(L^k) is preserved by construction. The learnable part
is the Lax-flow generator B_θ(L) = Σ_k c_k (P₊(L^k) − P₋(L^k))  (Toda-hierarchy form,
tridiagonal-preserving); the model learns the coefficients c_k. HONEST caveat (see the
pre-registration): staying tridiagonal nearly pins the generator, so the learned part
is small — structure does most of the work.

Three composing steppers, matched (one-step training, compose at eval, time extrapolation
train t≤5 → test t≤20):
  • Neural-Lax  — isospectral conjugation step (conserves all tr(L^k) by construction).
  • free-form   — residual MLP step, no structure.
  • bolt-on     — free-form + penalise all N conserved quantities tr(L^k) (strongest litmus).
"""
import numpy as np, torch, torch.nn as nn
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)

N = 8; DT = 0.01; DELTA = 0.25; T_TRAIN = 5.0; T_MAX = 20.0
QAMP, PAMP = 0.3, 0.6; K = 2                     # hierarchy order (c_1..c_K learned)

# ── periodic Toda ground truth ───────────────────────────────────────────────
def force_p(q):
    b = np.exp(q - np.roll(q, -1)); return np.roll(b, 1) - b
def verlet_p(q, p, steps):
    F = force_p(q)
    for _ in range(steps):
        p = p + 0.5*DT*F; q = q + DT*p; F = force_p(q); p = p + 0.5*DT*F
    return q, p
def make_ic(rng):
    q = QAMP*rng.standard_normal(N); q -= q.mean(); p = PAMP*rng.standard_normal(N); p -= p.mean()
    return q, p
def state(q, p): return np.concatenate([q - np.roll(q, -1), p])      # (bond, momentum), 2N
def trajectory(rng):                                                 # states at 0, Δ, 2Δ, ...
    q, p = make_ic(rng); sub = int(round(DELTA/DT)); out = [state(q, p)]
    for _ in range(int(round(T_MAX/DELTA))):
        q, p = verlet_p(q, p, sub); out.append(state(q, p))
    return np.array(out)                                             # (n_t+1, 2N)

# ── state <-> Lax matrix (periodic Jacobi), and tr(L^k) ──────────────────────
_bandmask = torch.zeros(N, N)
for n in range(N):
    _bandmask[n, n] = 1; _bandmask[n, (n+1) % N] = 1; _bandmask[(n+1) % N, n] = 1
def to_L(s):                                                         # s:(B,2N) -> L:(B,N,N)
    d = s[:, :N]; p = s[:, N:]; a = 0.5*torch.exp(d/2); B = s.shape[0]
    L = torch.zeros(B, N, N)
    L[:, torch.arange(N), torch.arange(N)] = -0.5*p
    for n in range(N):
        L[:, n, (n+1) % N] += a[:, n]; L[:, (n+1) % N, n] += a[:, n]
    return L
def from_L(L):                                                      # L -> s (trivial inverse Flaschka)
    p = -2*L[:, torch.arange(N), torch.arange(N)]
    a = torch.stack([L[:, n, (n+1) % N] for n in range(N)], 1).clamp_min(1e-6)
    d = 2*torch.log(2*a)
    return torch.cat([d, p], 1)
def trLk(s):                                                        # (B, N) conserved quantities
    L = to_L(s); M = L.clone(); outs = []
    for _ in range(N):
        outs.append(M.diagonal(dim1=1, dim2=2).sum(-1)); M = M @ L
    return torch.stack(outs, 1)

# ── models (composing steppers) ──────────────────────────────────────────────
def mlp(i, o, h=128, n=3):
    L = [nn.Linear(i, h), nn.GELU()]
    for _ in range(n-1): L += [nn.Linear(h, h), nn.GELU()]
    return nn.Sequential(*L, nn.Linear(h, o))

class NeuralLax(nn.Module):                                         # learn isospectral flow generator
    def __init__(s): super().__init__(); s.c = nn.Parameter(0.1*torch.randn(K))
    def B(s, L):
        acc = torch.zeros_like(L); Lp = L
        for k in range(K):
            acc = acc + s.c[k]*(torch.triu(Lp, 1) - torch.tril(Lp, -1))   # P₊ − P₋ (skew)
            Lp = Lp @ L
        return acc
    def step_L(s, L):
        Q = torch.matrix_exp(DELTA*s.B(L))                          # orthogonal ⇒ exactly isospectral
        Ln = Q @ L @ Q.transpose(-1, -2)
        return 0.5*(Ln + Ln.transpose(-1, -2))*_bandmask            # reproject to periodic-Jacobi
    def step(s, x): return from_L(s.step_L(to_L(x)))
class FreeStep(nn.Module):                                          # residual MLP step, no structure
    def __init__(s): super().__init__(); s.net = mlp(2*N, 2*N)
    def step(s, x): return x + s.net(x)

def rollout(m, x0, n):
    x = x0; xs = [x]
    for _ in range(n): x = m.step(x); xs.append(x)
    return xs

# ── data ─────────────────────────────────────────────────────────────────────
def make_pairs(n_ic, rng):
    S, S1 = [], []; ntr = int(round(T_TRAIN/DELTA))
    for _ in range(n_ic):
        tr = trajectory(rng)
        for k in range(ntr): S.append(tr[k]); S1.append(tr[k+1])    # one-step pairs, t<T_train
    return torch.tensor(np.array(S), dtype=torch.float32), torch.tensor(np.array(S1), dtype=torch.float32)
def test_traj(n_ic, rng):
    return [torch.tensor(trajectory(rng), dtype=torch.float32) for _ in range(n_ic)]

# ── train / eval ─────────────────────────────────────────────────────────────
def fit(m, S, S1, epochs=200, lr=3e-3, cons_pen=0.0):
    opt = torch.optim.Adam(m.parameters(), lr=lr); mse = nn.MSELoss(); B = 512
    trg = trLk(S).detach() if cons_pen > 0 else None
    for _ in range(epochs):
        perm = torch.randperm(len(S))
        for i in range(0, len(S), B):
            idx = perm[i:i+B]; pred = m.step(S[idx]); loss = mse(pred, S1[idx])
            if cons_pen > 0: loss = loss + cons_pen*mse(trLk(pred), trg[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return m

def evaluate(m, trajs):
    m.eval(); nb = int(round(T_MAX/DELTA)); buckets = [(0, 5, "train t≤5"), (5, 10, "t 5–10"), (10, 15, "t 10–15"), (15, 20, "t 15–20")]
    serr = {b[2]: [] for b in buckets}; cerr = {b[2]: [] for b in buckets}
    with torch.no_grad():
        for tr in trajs:
            xs = rollout(m, tr[0:1], nb); c0 = trLk(tr[0:1])
            for k in range(1, nb+1):
                t = k*DELTA
                for lo, hi, nm in buckets:
                    if (t > lo or lo == 0) and t <= hi:
                        serr[nm].append(((xs[k]-tr[k:k+1])**2).mean().item())
                        cerr[nm].append(((trLk(xs[k])-c0)**2).mean().item())
    return [(nm, float(np.mean(serr[nm])), float(np.mean(cerr[nm]))) for _, _, nm in buckets]

if __name__ == "__main__":
    rng = np.random.default_rng(0); S, S1 = make_pairs(250, rng); trajs = test_traj(60, np.random.default_rng(1))
    var = torch.stack([t for t in trajs]).var().item()
    print(f"periodic Toda N={N}, Δ={DELTA}, compose train t≤{T_TRAIN}→test t≤{T_MAX}, "
          f"isospectral step, hierarchy K={K}  (predict-mean MSE≈{var:.3f})\n")
    models = {"Neural-Lax (isospectral, learn c_k)": (NeuralLax(), 0.0),
              "free-form":                           (FreeStep(), 0.0),
              "bolt-on (all N cons pinned)":         (FreeStep(), 1.0)}
    res = {}
    for nm, (m, pen) in models.items():
        print(f"training {nm} ..."); fit(m, S, S1, cons_pen=pen); res[nm] = evaluate(m, trajs)
    cols = [r[0] for r in res["free-form"]]
    print(f"\n{'model':36s}" + "".join(f"{c:>11}" for c in cols) + "   (state MSE)")
    for nm in res: print(f"{nm:36s}" + "".join(f"{s:11.4f}" for _, s, _ in res[nm]))
    print(f"\n{'model':36s}" + "".join(f"{c:>11}" for c in cols) + "   (conserved-qty error)")
    for nm in res: print(f"{nm:36s}" + "".join(f"{c:11.4f}" for _, _, c in res[nm]))
    nl = models["Neural-Lax (isospectral, learn c_k)"][0]
    print(f"\nlearned c_k = {nl.c.detach().numpy().round(3)}   (true Toda ≈ [1, 0])")
    print("Read: does Neural-Lax state-MSE stay flat + conserved-qty ~0, while bolt-on conserves but drifts in state?")
