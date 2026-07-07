"""
Integrable Extrapolation benchmark · flagship system: the multi-soliton Box-Ball System.

Ground truth first (before any model). The Box-Ball System (BBS) is a discrete integrable
cellular automaton: a block of k consecutive 1s is a **soliton of amplitude k**; it moves
right at speed k, larger solitons overtake smaller ones and pass THROUGH them, each keeping
its amplitude. The multiset of soliton amplitudes is INVARIANT under the dynamics — these are
BBS's conserved quantities (as many as there are solitons: the integrable signature, not a
single conservation law).

This file provides:
  • bbs_step / bbs_run     — exact BBS evolution (infinite-capacity carrier),
  • soliton_content        — extract the conserved multiset of soliton amplitudes,
  • make_multisoliton      — generate states with prescribed solitons,
and self-checks that soliton content is conserved and the dynamics is reversible.

The benchmark task (built on this): predict the state after T steps; train on FEW solitons /
SHORT time, test on MANY solitons / LONG time (many collisions) — where a model must preserve
the soliton content to extrapolate, and free-form sequence models lose solitons over collisions.
"""
import numpy as np

def bbs_step(s):
    """One BBS step (vectorised). The carrier count is a +1-per-1 / −1-per-0 walk reflected at 0
    (leading 0s, reached with an empty carrier, drop nothing). Skorokhod reflection:
    c = P − min(cummin(P), 0);  a 0-cell drops iff the carrier was non-empty just before it."""
    P  = 2*np.cumsum(s) - np.arange(1, len(s)+1)                # +1 per 1, −1 per 0 (prefix)
    c  = P - np.minimum(np.minimum.accumulate(P), 0)           # carrier after each cell (≥0)
    c_prev = np.concatenate([[0], c[:-1]])                     # carrier just before each cell
    out = np.where((s == 0) & (c_prev > 0), 1, 0).astype(s.dtype)
    assert out.sum() == s.sum(), "balls fell off the right edge — pad the lattice with more empty space"
    return out

def bbs_run(s, T):
    s = s.copy()
    for _ in range(T):
        s = bbs_step(s)
    return s

def mirror(s):            return s[::-1].copy()
def bbs_step_inv(s):      return mirror(bbs_step(mirror(s)))     # reversibility via the mirror trick

def blocks(s):
    """sorted-descending lengths of maximal runs of 1s."""
    out = []; run = 0
    for x in s:
        if x == 1: run += 1
        elif run: out.append(run); run = 0
    if run: out.append(run)
    return sorted(out, reverse=True)

def soliton_content(s):
    """The conserved multiset of soliton amplitudes.
    Evolve until the solitons fully separate (block multiset stops changing), then read the
    block lengths. Each evolution is on a lattice padded to hold that many steps of rightward
    motion, so no ball falls off."""
    tot = int(s.sum())
    if tot == 0: return []
    def content_after(nsteps):
        pad = nsteps*tot + len(s) + 10                      # rightmost soliton moves ≤ tot/step
        x = np.concatenate([s, np.zeros(pad, dtype=s.dtype)])
        return blocks(bbs_run(x, nsteps))
    steps = 2*len(s) + 60
    while True:
        a = content_after(steps)
        if a == content_after(steps + 4*tot + 10):          # stable ⇒ fully separated
            return a
        steps *= 2

def make_multisoliton(amplitudes, gaps, right_pad):
    """Build a state: solitons of the given amplitudes, separated by `gaps` zeros, then padding.
    amplitudes: list[int]; gaps: list[int] (len = len(amplitudes)-1 or more)."""
    cells = []
    for i, a in enumerate(amplitudes):
        cells += [1]*a
        if i < len(amplitudes) - 1:
            cells += [0]*gaps[i]
    cells += [0]*right_pad
    return np.array(cells, dtype=np.int64)

# ── self-checks ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Multi-soliton BBS — ground-truth self-check\n")
    rng = np.random.default_rng(0)

    # (a) a textbook 2-soliton collision: amplitudes {3,1} must survive the overtake
    s = make_multisoliton([3, 1], [2], right_pad=80)
    c0 = soliton_content(s)
    print(f"  2-soliton {{3,1}} collision: soliton content over time =",
          [soliton_content(bbs_run(s, t)) for t in [0, 3, 6, 9, 12]])
    assert all(soliton_content(bbs_run(s, t)) == c0 for t in range(15)), "soliton content changed!"

    def rand_config(rng, HORIZON):
        k = rng.integers(2, 6); amps = sorted(rng.integers(1, 6, size=k).tolist(), reverse=True)
        gaps = rng.integers(2, 6, size=k-1).tolist()
        # infinite carrier ⇒ rightmost ball can advance by up to the TOTAL ball count per step
        # (during collisions), so pad ∝ horizon·tot
        return make_multisoliton(amps, gaps, right_pad=(HORIZON + 2)*int(sum(amps)) + 30)

    # (b) soliton content is a conserved invariant (15 configs × a few time points)
    HORIZON = 24; cons = True
    for _ in range(15):
        s = rand_config(rng, HORIZON); c0 = soliton_content(s)
        if any(soliton_content(bbs_run(s, t)) != c0 for t in [8, 16, 24]): cons = False
    print(f"\n  soliton content conserved (15 configs, t=8/16/24) : {cons}")

    # (c) reversibility over many steps (cheap — no soliton extraction; 200 configs × 24 steps)
    rev = True
    for _ in range(200):
        s = rand_config(rng, HORIZON)
        for t in range(1, HORIZON + 1):
            st = bbs_run(s, t)
            if not np.array_equal(bbs_step_inv(bbs_step(st)), st): rev = False; break
    print(f"  reversible (mirror trick, 200 configs, 24 steps)   : {rev}")
    assert cons and rev
    print("\nSELF-CHECK PASSED — soliton content is an exact conserved invariant; dynamics reversible.")
    print("This is the benchmark's integrable signature: models must preserve the soliton multiset to extrapolate.")
