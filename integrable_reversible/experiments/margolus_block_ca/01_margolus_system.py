"""
Step 1 (打样) — the 1D Margolus block CA itself: define it, and PROVE it is
conserving + reversible + a NON-TRIVIAL rule, before any model touches it.

Why a second system at all: the Box-Ball experiment showed that welding
*conservation* into a network's structure keeps the invariant exact where
free-emit scan models drift. That was ONE system whose natural prior is a
left→right carrier scan. Margolus is deliberately a *different* kind of
reversible system — a block/partitioning CA with no long-range carrier — so it
tests whether the "structural conservation + learned residual" recipe transfers,
and the proposal's claim that different systems want different structures.

The 1D Margolus-style rule used here (cheap CPU, clean parallel to BBS):
  * lattice is a ring of length L (L divisible by 3), periodic boundary.
  * BLOCK size 3; the partition OFFSET cycles 0 → 1 → 2 with the step index
    (this is the 1D analogue of Margolus' alternating neighbourhood).
  * each 3-cell block is mapped by a fixed permutation φ that
      - fixes 000 and 111,
      - rotates the 1-ball class  {100, 010, 001}  by a 3-cycle,
      - rotates the 2-ball class  {110, 101, 011}  by a 3-cycle.
    φ is a bijection  → reversibility is STRUCTURAL.
    φ preserves ball-count within every block → conservation is STRUCTURAL.
    φ is NOT the identity within the count classes → the rotation must be
    LEARNED (a conservation-respecting-but-"blind" identity gate can't match it).

This file only establishes the ground truth: conservation, reversibility (via
the block-inverse in reversed phase order), and a leak audit showing a trivial
count-preserving guess (identity) is far from 100%.
"""
import numpy as np

BLK = 3            # block size
T_STEPS = 6        # prediction horizon (input -> +T)

# ── the block permutation φ on the 8 states of a 3-cell block ─────────────────
# encode a block as b0*4 + b1*2 + b2  (b0 is the left cell)
def _bits(v):  return ((v >> 2) & 1, (v >> 1) & 1, v & 1)
def _val(t):   return (t[0] << 2) | (t[1] << 1) | t[2]

# 1-ball class in this integer encoding: 100=4, 010=2, 001=1
# 2-ball class:                          110=6, 101=5, 011=3
ONE = [4, 2, 1]     # rotate 4→2→1→4
TWO = [6, 5, 3]     # rotate 6→5→3→6
def _rot(cycle):
    m = {}
    for i, v in enumerate(cycle):
        m[v] = cycle[(i + 1) % len(cycle)]
    return m
PHI = {v: v for v in range(8)}          # start from identity (fixes 000=0, 111=7)
PHI.update(_rot(ONE)); PHI.update(_rot(TWO))
PHI_INV = {out: inp for inp, out in PHI.items()}

# sanity: φ is a bijection and preserves ball-count per block
assert sorted(PHI.values()) == list(range(8)), "φ is not a bijection"
for v in range(8):
    assert sum(_bits(v)) == sum(_bits(PHI[v])), "φ does not preserve block ball-count"

def _apply_blocks(s, offset, table):
    """Partition the ring into size-3 blocks starting at `offset`, map each by `table`."""
    L = len(s); out = s.copy()
    for start in range(offset, offset + L, BLK):
        idx = [(start + j) % L for j in range(BLK)]
        b = _val((s[idx[0]], s[idx[1]], s[idx[2]]))
        nb = _bits(table[b])
        for j in range(BLK):
            out[idx[j]] = nb[j]
    return out

def margolus_step(s, t):
    """One step; the partition offset at step t is (t mod 3)."""
    return _apply_blocks(s, t % BLK, PHI)

def margolus_run(s, T):
    s = s.copy()
    for t in range(T):
        s = margolus_step(s, t)
    return s

def margolus_inv(y, T):
    """Invert T steps: apply φ⁻¹ with the offsets in REVERSE order."""
    s = y.copy()
    for t in reversed(range(T)):
        s = _apply_blocks(s, t % BLK, PHI_INV)
    return s

def make_config(L, rng, density=0.45):
    return (rng.random(L) < density).astype(np.int64)

# ── self-check: conservation + reversibility on random rings ──────────────────
if __name__ == "__main__":
    print(f"1D Margolus block CA — BLK={BLK}, offsets cycle 0→1→2, horizon T={T_STEPS}")
    print("φ (block map):  " + "  ".join(f"{''.join(map(str,_bits(v)))}→{''.join(map(str,_bits(PHI[v])))}" for v in range(8)))

    rng = np.random.default_rng(1)
    cons = rev = True
    for L in [12, 24, 48, 96, 192]:
        for _ in range(300):
            x = make_config(L, rng)
            y = margolus_run(x, T_STEPS)
            if y.sum() != x.sum():                 cons = False
            if not np.array_equal(margolus_inv(y, T_STEPS), x): rev = False
    print(f"\n  ball-count conserved : {cons}")
    print(f"  reversible (block-inverse, reversed phase order) : {rev}")
    assert cons and rev, "system is not conserving/reversible — fix φ"

    # ── leak audit: is the rule non-trivial? ─────────────────────────────────
    # A trivial count-preserving guess = identity (predict output == input).
    # If identity already scored ~100%, the rotation would be vacuous to "learn".
    rng = np.random.default_rng(2)
    tot = same = 0
    for L in [24, 48, 96]:
        for _ in range(500):
            x = make_config(L, rng); y = margolus_run(x, T_STEPS)
            same += int((x == y).sum()); tot += L
    print(f"\n  leak audit — identity (blind) per-cell accuracy: {100*same/tot:.1f}%")
    print("  (well below 100% ⇒ the count-preserving rotation is a non-trivial residual to learn)")
