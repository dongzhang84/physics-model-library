"""
Phase 0 · step 1 — define a colored Box-Ball System and verify it before any model.

Goal of this file only: get the rule and the labeled-content extractor RIGHT, checked by a
self-check (labeled soliton content conserved + reversible). Getting these right is the first
risk of the whole probe; the self-check is the oracle. Finite carrier capacity (which makes the
learned gate non-trivial) is added in the next step, on top of a verified colored rule.

Colored BBS: cells are 0 (empty) or a color in 1..C. A carrier sweeps left->right, picks up every
ball, and at each empty cell emits one ball chosen by a fixed color priority. We do NOT assume a
rule is integrable — we test candidate emit priorities and keep whichever conserves a labeled
soliton content.
"""
import numpy as np
from collections import Counter

def colored_step(s, emit="max"):
    """One colored-BBS step. Carrier picks up every ball; at each empty cell it emits the
    highest- (emit='max') or lowest- (emit='min') color it holds. emit='max' is the candidate
    integrable rule (larger colors move faster); the self-check decides."""
    carrier = {}            # color -> count
    out = np.zeros_like(s)
    for i in range(len(s)):
        c = int(s[i])
        if c > 0:
            carrier[c] = carrier.get(c, 0) + 1
        elif carrier:
            k = max(carrier) if emit == "max" else min(carrier)
            out[i] = k
            carrier[k] -= 1
            if carrier[k] == 0:
                del carrier[k]
    assert not carrier, "balls fell off the right edge — pad more"
    return out

def colored_run(s, T, emit="max"):
    s = s.copy()
    for _ in range(T):
        s = colored_step(s, emit)
    return s

def mirror(s): return s[::-1].copy()
def colored_step_inv(s, emit="max"):
    """inverse of one colored step: mirror + the OPPOSITE emit priority (verified)."""
    return mirror(colored_step(mirror(s), "min" if emit == "max" else "max"))

def colored_blocks(s):
    """left->right list of maximal nonzero runs, each as a tuple of its colors."""
    out = []; run = []
    for x in s:
        if x > 0: run.append(int(x))
        elif run: out.append(tuple(run)); run = []
    if run: out.append(tuple(run))
    return out

def labeled_content(s, emit="max"):
    """conserved labeled soliton content: evolve until the colored blocks stop changing
    (solitons separated), then return the sorted multiset of colored solitons."""
    tot = int((s > 0).sum())
    if tot == 0: return ()
    def content_after(nsteps):
        pad = nsteps * tot + len(s) + 10
        x = np.concatenate([s, np.zeros(pad, dtype=s.dtype)])
        return tuple(sorted(colored_blocks(colored_run(x, nsteps, emit))))
    steps = 2 * len(s) + 60
    while True:
        a = content_after(steps)
        if a == content_after(steps + 4 * tot + 10):
            return a
        steps *= 2

def make_colored(solitons, gaps, right_pad):
    """solitons: list of color-tuples (e.g. (2,2,1) = a length-3 soliton); gaps between them."""
    cells = []
    for i, sol in enumerate(solitons):
        cells += list(sol)
        if i < len(solitons) - 1:
            cells += [0] * gaps[i]
    cells += [0] * right_pad
    return np.array(cells, dtype=np.int64)

# ── self-check: which emit rule conserves a labeled soliton content? ──────────
if __name__ == "__main__":
    print("Colored BBS — testing candidate emit rules for a conserved labeled content\n")
    rng = np.random.default_rng(0); C = 3; HORIZON = 20

    def rand_cfg(rng):
        k = rng.integers(2, 5)
        sols = [tuple(int(x) for x in rng.integers(1, C + 1, size=int(rng.integers(1, 4)))) for _ in range(k)]
        gaps = rng.integers(3, 7, size=k - 1).tolist()
        tot = sum(len(s) for s in sols)
        return make_colored(sols, gaps, right_pad=(HORIZON + 2) * tot + 30)

    for emit in ["max", "min"]:
        cons = rev = True; ex_change = None
        for _ in range(20):
            s = rand_cfg(rng); c0 = labeled_content(s, emit)
            for t in [7, 14, HORIZON]:
                if labeled_content(colored_run(s, t, emit), emit) != c0:
                    cons = False
                    if ex_change is None: ex_change = (c0, labeled_content(colored_run(s, t, emit), emit))
            for t in range(1, HORIZON + 1):
                st = colored_run(s, t, emit)
                if not np.array_equal(colored_step_inv(colored_step(st, emit), emit), st):
                    rev = False; break
        print(f"  emit='{emit}':  labeled content conserved = {cons}   reversible = {rev}")
        if not cons and ex_change: print(f"      example change: {ex_change[0]}  ->  {ex_change[1]}")

    # show the textbook colored 2-soliton overtake under the winning rule (if any)
    print("\n  colored 2-soliton overtake, emit='max':")
    s = make_colored([(2, 2, 1), (3,)], [3], right_pad=(HORIZON + 2) * 4 + 30)
    for t in [0, 4, 8, 12]:
        print(f"    t={t:2d}  content = {labeled_content(colored_run(s, t, 'max'), 'max')}")
