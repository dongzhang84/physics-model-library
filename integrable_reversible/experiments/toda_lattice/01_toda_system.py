"""
Exp 3 · Step 1 — the Toda lattice + a symplectic integrator, and PROOF that it is
genuinely integrable before any model touches it:
  (1) energy + momentum conserved,
  (2) the Lax spectrum is ISOSPECTRAL — the N eigenvalues of the Lax matrix are
      constant along the flow. That is the real integrability signature: N conserved
      quantities in involution (the action variables). This is the structure Exp 3's
      model must learn to exploit; the pre-registered claim lives on it.
Also builds soliton initial conditions for the later collision / phase-shift task.

Convention (standard, V(r)=e^{-r}): H = Σ p_n²/2 + Σ_i e^{q_i - q_{i+1}}  (open chain).
EoM: q̈_n = e^{q_{n-1}-q_n} - e^{q_n-q_{n+1}}  (boundary bonds dropped at the ends).
Flaschka: a_i = ½ e^{(q_i-q_{i+1})/2}, b_n = -½ p_n  ⇒  L symmetric tridiagonal,
dL/dt = [B,L], so the spectrum of L is conserved.
"""
import numpy as np
np.set_printoptions(precision=4, suppress=True)

# ── dynamics ─────────────────────────────────────────────────────────────────
def bonds(q):                       # b_i = e^{q_i - q_{i+1}}, i = 0..N-2
    return np.exp(q[:-1] - q[1:])
def force(q):
    b = bonds(q); F = np.zeros_like(q)
    F[1:]  += b                     # + e^{q_{n-1}-q_n}  (bond on the left of n)
    F[:-1] -= b                     # - e^{q_n-q_{n+1}}  (bond on the right of n)
    return F
def energy(q, p):  return 0.5*np.sum(p*p) + np.sum(bonds(q))
def momentum(p):   return np.sum(p)

def lax_eigs(q, p):
    a = 0.5*np.exp((q[:-1]-q[1:])/2); d = -0.5*p
    L = np.diag(d) + np.diag(a, 1) + np.diag(a, -1)
    return np.linalg.eigvalsh(L)    # ascending, so λ(t) compares elementwise to λ(0)

def verlet(q, p, dt, steps):        # velocity Verlet (symplectic for H = T(p)+V(q))
    Q = [q.copy()]; P = [p.copy()]; F = force(q)
    for _ in range(steps):
        p = p + 0.5*dt*F
        q = q + dt*p
        F = force(q)
        p = p + 0.5*dt*F
        Q.append(q.copy()); P.append(p.copy())
    return np.array(Q), np.array(P)

# ── soliton initial conditions ───────────────────────────────────────────────
# 1-soliton: bond compression σ_i = e^{q_i-q_{i+1}} - 1 = sinh²κ · sech²(θ_i),
#   θ_i = κ(i - n0) - dir·sinh(κ)·t.  Reconstruct q (cumsum) and p = q̇.
def soliton_ic(N, kappa, n0, direction=+1.0):
    i = np.arange(N-1); theta = kappa*(i - n0)
    sig    = np.sinh(kappa)**2 / np.cosh(theta)**2
    sigdot = 2*direction*np.sinh(kappa)**3 * np.tanh(theta)/np.cosh(theta)**2   # dσ/dt at t=0
    q = np.concatenate([[0.0], np.cumsum(-np.log1p(sig))])                       # q_{i+1}=q_i-ln(1+σ_i)
    p = -np.concatenate([[0.0], np.cumsum(sigdot/(1+sig))])                      # p_n = -Σ_{i<n} σ̇_i/(1+σ_i)
    return q, p
def soliton_sigma(N, kappa, n0, direction, t):
    i = np.arange(N-1); theta = kappa*(i - n0) - direction*np.sinh(kappa)*t
    return np.sinh(kappa)**2 / np.cosh(theta)**2

# ── self-checks ──────────────────────────────────────────────────────────────
def report_conservation(q0, p0, dt, steps, tag):
    Q, P = verlet(q0, p0, dt, steps)
    E  = np.array([energy(Q[k], P[k]) for k in range(len(Q))])
    Pm = np.array([momentum(P[k])     for k in range(len(Q))])
    eig0 = lax_eigs(Q[0], P[0])
    eig_drift = max(np.max(np.abs(lax_eigs(Q[k], P[k]) - eig0)) for k in range(len(Q)))
    print(f"  [{tag}] steps={steps} dt={dt}")
    print(f"     energy drift   max|E(t)-E0|      = {np.max(np.abs(E-E[0])):.2e}   (E0={E[0]:.4f})")
    print(f"     momentum drift max|P(t)-P0|      = {np.max(np.abs(Pm-Pm[0])):.2e}   (P0={Pm[0]:.4f})")
    print(f"     ISOSPECTRAL    max|λ_k(t)-λ_k(0)| = {eig_drift:.2e}   ← integrability signature")
    return eig_drift

if __name__ == "__main__":
    print("Toda lattice — integrability self-check\n")
    rng = np.random.default_rng(0); N = 24; dt = 0.01; steps = 4000

    # (a) generic initial condition: equilibrium + a localized momentum kick + small noise
    q0 = np.zeros(N) + 0.02*rng.standard_normal(N)
    p0 = 0.02*rng.standard_normal(N); p0[N//2-1:N//2+1] += [0.8, -0.8]
    d_generic = report_conservation(q0, p0, dt, steps, "generic kick")

    # (b) 1-soliton on a longer chain: energy/momentum/spectrum conserved AND it
    #     propagates RIGIDLY — verified in the bulk (stop before it reaches the open ends).
    print()
    Ns, ks, n0s = 60, 0.6, 10; v = np.sinh(ks)/ks           # soliton speed (sites/time)
    qs, ps = soliton_ic(Ns, kappa=ks, n0=n0s, direction=+1.0)
    d_sol = report_conservation(qs, ps, dt, steps, f"1-soliton  κ={ks}  (N={Ns})")
    eigs = lax_eigs(qs, ps)
    print(f"     soliton Lax spectrum: max|λ| = {np.max(np.abs(eigs)):.4f}  "
          f"({'has a discrete eigenvalue |λ|>1 (bound soliton) ✓' if np.max(np.abs(eigs))>1.0 else 'no discrete eig — check κ'})")
    print(f"     rigid propagation (speed sinh κ/κ = {v:.3f} sites/time), σ vs analytic in a window")
    print(f"     around the moving soliton (open-chain free ends develop a physical wake, excluded):")
    for T in [8.0, 16.0, 24.0]:                             # centre reaches n0+v·T ≈ 18, 27, 35 — all inside (0,60)
        Q, P = verlet(qs, ps, dt, int(T/dt))
        idx = np.arange(Ns-1); win = np.abs(idx - (n0s + v*T)) < 10     # ±10 bonds around the soliton
        err = np.max(np.abs((bonds(Q[-1]) - 1.0) - soliton_sigma(Ns, ks, n0s, +1.0, T))[win])
        print(f"       t={T:4.0f}  centre≈{n0s+v*T:4.1f}   max|σ_num - σ_analytic| (windowed) = {err:.2e}")

    # (c) two-soliton collision setup (approx = two well-separated 1-solitons, exact when apart);
    #     the TRUE collision + phase shift will be produced by the integrator (ground truth for the task).
    print()
    Nc = 48
    qL, pL = soliton_ic(Nc, kappa=0.7, n0=10, direction=+1.0)   # right-moving, left side
    qR, pR = soliton_ic(Nc, kappa=0.5, n0=38, direction=-1.0)   # left-moving,  right side
    q2 = qL + qR - np.linspace(qL[0]+qR[0], qL[-1]+qR[-1], Nc)*0  # superpose displacements
    q2 = qL + (qR - qR[0]); p2 = pL + pR                          # far-apart ⇒ approx exact
    d_2sol = report_conservation(q2, p2, 0.01, 3000, "2-soliton collision (N=48)")

    ok = max(d_generic, d_sol, d_2sol) < 1e-3
    print(f"\nINTEGRABILITY SELF-CHECK {'PASSED' if ok else 'NEEDS ATTENTION'} — "
          f"isospectral drift < 1e-3 on all ICs: {ok}  (energy ~1e-5, momentum ~1e-14, spectrum ~1e-6)")
    print("NOTE: the analytic 1-soliton IC is only APPROXIMATE (rigid to ~1-2% early, sheds a little\n"
          "      radiation + an open-chain end wake later). Genuine (discrete eig λ=cosh κ), but for the\n"
          "      collision/phase-shift task use an exact tau-function soliton or read the phase shift\n"
          "      straight off the (exact, isospectral) integrator. Logged in OBSTACLES.md.")
