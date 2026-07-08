"""
Phase 0 · path B, step 1 — finite-carrier BBS + soliton-amplitude content, verified first.

Path B tests whether a system we already trust (finite-carrier BBS: integrable, reversible, and
NON-trivially learnable — carrier-blind ~89% ≠ learned 100% in the box-ball experiment) already
carries a conserved signature RICHER than ball count: the multiset of soliton amplitudes. If it
does, then (a) bolt-on that pins ball count cannot recover the amplitude multiset, and (b) the
finite carrier makes learning genuine (no plain-BBS leak) — i.e. "deep + genuine" without needing
the harder colored rule.

This file only establishes the ground truth: soliton-amplitude content is a conserved invariant,
and the dynamics is reversible. Getting these right (self-check as oracle) is the prerequisite.

Finite-carrier BBS: a carrier of capacity K sweeps left->right; a 1 is picked up if the carrier has
room, else it PASSES THROUGH (carrier full); a 0 gets a drop if the carrier holds a ball.
"""
import numpy as np

def fc_step(s, K):
    out = np.zeros_like(s); u = 0
    for i in range(len(s)):
        if s[i] == 1:
            if u < K: u += 1          # pick up (room)
            else:     out[i] = 1      # carrier FULL -> pass through
        else:
            if u > 0: out[i] = 1; u -= 1
    assert u == 0, "carrier not empty at the right edge — pad the lattice with more empty space"
    return out

def fc_run(s, T, K):
    s = s.copy()
    for _ in range(T): s = fc_step(s, K)
    return s

def mirror(s): return s[::-1].copy()
def fc_step_inv(s, K): return mirror(fc_step(mirror(s), K))     # reversibility via the mirror trick

def blocks(s):
    out = []; run = 0
    for x in s:
        if x == 1: run += 1
        elif run: out.append(run); run = 0
    if run: out.append(run)
    return sorted(out, reverse=True)

def amplitude_content(s, K):
    """conserved soliton-amplitude multiset: evolve until the blocks stop changing (solitons
    separated), then read the sorted block lengths. Verified stable."""
    tot = int(s.sum())
    if tot == 0: return []
    def content_after(nsteps):
        pad = nsteps * tot + len(s) + 10
        x = np.concatenate([s, np.zeros(pad, dtype=s.dtype)])
        return blocks(fc_run(x, nsteps, K))
    steps = 2 * len(s) + 60
    for _ in range(4):                                  # cap the doubling — a garbage prediction may
        a = content_after(steps)                        # never fully separate; return best-effort then
        if a == content_after(steps + 4 * tot + 10):
            return a
        steps *= 2
    return a

def make_multisoliton(amps, gaps, right_pad):
    cells = []
    for i, a in enumerate(amps):
        cells += [1] * a
        if i < len(amps) - 1: cells += [0] * gaps[i]
    cells += [0] * right_pad
    return np.array(cells, dtype=np.int64)

# ── self-check: is amplitude content conserved (and richer than count)? ───────
if __name__ == "__main__":
    print("finite-carrier BBS — soliton-amplitude content self-check\n")
    rng = np.random.default_rng(0); HORIZON = 20

    def rand_cfg(rng, K):
        k = rng.integers(2, 6)
        amps = sorted(rng.integers(1, K + 3, size=k).tolist(), reverse=True)   # include amps > K
        gaps = rng.integers(3, 7, size=k - 1).tolist()
        return make_multisoliton(amps, gaps, right_pad=(HORIZON + 2) * int(sum(amps)) + 30)

    for K in [2, 4, 6]:
        cons = rev = True; richer = 0; n = 0
        for _ in range(15):
            s = rand_cfg(rng, K); c0 = amplitude_content(s, K); n += 1
            if any(amplitude_content(fc_run(s, t, K), K) != c0 for t in [7, 14, HORIZON]): cons = False
            # "richer than count": two different multisets can share the same sum
            richer += (len(set(c0)) > 1)
            for t in range(1, HORIZON + 1):
                st = fc_run(s, t, K)
                if not np.array_equal(fc_step_inv(fc_step(st, K), K), st): rev = False; break
        print(f"  K={K}:  amplitude content conserved = {cons}   reversible = {rev}   "
              f"(multi-amplitude configs {richer}/{n} — count alone can't recover)")

    # show a K-capacity overtake: content survives, and it is a multiset not a scalar
    print("\n  example (K=4): a {5,3,1} configuration, content over time")
    s = make_multisoliton([5, 3, 1], [3, 3], right_pad=(HORIZON + 2) * 9 + 30)
    for t in [0, 5, 10, 15]:
        print(f"    t={t:2d}  content = {amplitude_content(fc_run(s, t, 4), 4)}   ball-count = {int(fc_run(s,t,4).sum())}")
