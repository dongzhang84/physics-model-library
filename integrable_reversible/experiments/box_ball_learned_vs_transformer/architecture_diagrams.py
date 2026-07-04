"""
Architecture schematics for the three models in this folder (static, no training):

  architecture_01_transformer_vs_carrier.png  — Test 1: Transformer  vs  plain carrier
  architecture_02_swap_automaton.png          — Test 2: reversible gated-swap automaton
  architecture_03_conserving_carrier.png      — Test 3: conserving carrier (emit / hold)

Same task throughout: given a 0/1 Box-Ball state, predict it 2 steps later.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# The input row and its 2-step target shown on the diagrams use the REAL Box-Ball
# rule (not hand-made numbers), so the input→output is checkable. The internal
# boxes/arrows are schematic. IN is chosen so the 2-step target conserves within
# the 5 displayed cells (no ball falls off the right edge).
def _bbs_step(s):
    out = np.zeros_like(s); c = 0
    for i in range(len(s)):
        if s[i] == 1: c += 1; out[i] = 0
        elif c > 0:   out[i] = 1; c -= 1
    return out
def _bbs_run(s, T):
    s = s.copy()
    for _ in range(T): s = _bbs_step(s)
    return s

BLUE, GREEN, PURPLE, RED = "#2c3e50", "#1e8449", "#8e44ad", "#c0392b"
TOK = [1.3, 2.9, 4.5, 6.1, 7.7]
IN_BITS = [1, 0, 1, 0, 0]
IN  = [str(b) for b in IN_BITS]
OUT = [str(b) for b in _bbs_run(np.array(IN_BITS), 2)]   # = true BBS state 2 steps later

def box(ax, x, y, w, h, text, fc, ec="#333", fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.09", fc=fc, ec=ec, lw=1.3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#111")

def up(ax, x, y0, y1, c="#555", lw=1.6):
    ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=lw))

def tokens(ax, y, labels, fc):
    for x, t in zip(TOK, labels):
        box(ax, x, y, 1.15, 0.72, t, fc, fs=11, bold=True)

# ══════════════════════════ Test 1 ══════════════════════════
def draw_01():
    fig, (aL, aR) = plt.subplots(1, 2, figsize=(15, 7.6))
    for ax in (aL, aR):
        ax.set_xlim(0, 9); ax.set_ylim(0, 10); ax.axis("off")
    aL.set_title("Transformer  ·  global, parallel, no built-in structure", fontsize=13, color=BLUE, pad=12)
    tokens(aL, 0.8, IN, "#dfe6ec"); aL.text(4.5, 0.05, "input: 0/1 Box-Ball state", ha="center", fontsize=9, color="#666")
    box(aL, 4.5, 2.15, 7.4, 0.7, "Embed (0/1 → vector) + sinusoidal position encoding", "#eaf0f6")
    box(aL, 4.5, 3.75, 7.4, 1.05, "Self-attention — every position attends to EVERY position", "#d6e4f0")
    for x in TOK: aL.plot(x, 3.75, "o", ms=5, color=BLUE, zorder=5)
    for i in range(len(TOK)):
        for j in range(i+1, len(TOK)):
            aL.plot([TOK[i], TOK[j]], [3.75, 3.75], color=BLUE, lw=0.5, alpha=0.35, zorder=4)
    box(aL, 4.5, 5.35, 7.4, 0.7, "Feed-forward  (each position, independently)", "#eaf0f6")
    aL.text(8.55, 4.55, "× 3\nlayers", ha="center", va="center", fontsize=9, color=BLUE, style="italic")
    box(aL, 4.5, 6.95, 7.4, 0.7, "Linear → 0/1 per position", "#eaf0f6")
    tokens(aL, 8.5, OUT, "#dfe6ec"); aL.text(4.5, 9.35, "output: predicted state 2 steps later", ha="center", fontsize=9, color="#666")
    for y0, y1 in [(1.2, 1.8), (2.5, 3.22), (4.27, 5.0), (5.7, 6.6), (7.3, 8.14)]: up(aL, 4.5, y0, y1)
    aL.text(4.5, -0.55, "every output computed from the whole sequence at once —\nno left→right order, no carried state, no conserved quantity",
            ha="center", va="top", fontsize=9.5, color=BLUE)

    aR.set_title("LearnedCarrier  ·  left→right recurrence carrying a state  h", fontsize=13, color=GREEN, pad=12)
    tokens(aR, 0.8, IN, "#e2f0e6"); aR.text(4.5, 0.05, "input: 0/1 Box-Ball state", ha="center", fontsize=9, color="#666")
    cy = 3.4
    for x in TOK:
        box(aR, x, cy, 1.15, 1.0, "cell", "#d7efdd", ec=GREEN, fs=10, bold=True)
        up(aR, x, 1.2, cy - 0.52); up(aR, x, cy + 0.52, 5.05)
    aR.text(0.2, cy, "h₀=0", ha="center", va="center", fontsize=9, color=GREEN)
    aR.annotate("", xy=(TOK[0]-0.58, cy), xytext=(0.55, cy), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.6))
    for i in range(len(TOK)-1):
        aR.annotate("", xy=(TOK[i+1]-0.58, cy), xytext=(TOK[i]+0.58, cy), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.6))
    aR.text((TOK[0]+TOK[1])/2, cy+0.42, "carrier  h", ha="center", fontsize=9, color=GREEN, style="italic")
    tokens(aR, 5.55, OUT, "#e2f0e6"); aR.text(4.5, 6.15, "output: predicted state 2 steps later", ha="center", fontsize=9, color="#666")
    box(aR, 4.5, 8.3, 7.9, 1.15, "zoom · one cell (weights shared across all positions):\n[ bitᵢ ,  hᵢ ]  →  small MLP  →  ( outᵢ ,  hᵢ₊₁ )", "#f0faf2", ec=GREEN, fs=10)
    aR.annotate("", xy=(4.5, 7.7), xytext=(TOK[2], cy + 0.55), arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.9, ls=":"))
    aR.text(4.5, -0.55, "same cell weights at every position → runs at ANY length (extrapolates) ·\napplied ×2 steps · emits 0/1 freely → accurate but does NOT conserve",
            ha="center", va="top", fontsize=9.5, color=GREEN)
    fig.suptitle("Test 1 — two models, same task: predict the Box-Ball state 2 steps ahead", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96]); fig.savefig("architecture_01_transformer_vs_carrier.png", dpi=140, bbox_inches="tight")
    print("saved architecture_01_transformer_vs_carrier.png"); plt.close(fig)

# ══════════════════════════ Test 2 ══════════════════════════
def pair(ax, xa, xb, y0, y1, swap, c):
    if swap:
        ax.annotate("", xy=(xb, y1), xytext=(xa, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.7))
        ax.annotate("", xy=(xa, y1), xytext=(xb, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.7))
    else:
        up(ax, xa, y0, y1, c=c); up(ax, xb, y0, y1, c=c)
    xm, ym = (xa+xb)/2, (y0+y1)/2
    ax.plot(xm, ym, marker="D", ms=11, color="white", mec=c, mew=1.5, zorder=6)
    ax.text(xm, ym, "g", fontsize=7.5, ha="center", va="center", color=c, zorder=7)

def draw_02():
    fig, ax = plt.subplots(figsize=(11.5, 7.4)); ax.set_xlim(-0.5, 9.5); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Test 2 — reversible gated-swap automaton  ·  exact guarantees, but LOCAL", fontsize=13, color=PURPLE, pad=12)
    tokens(ax, 0.9, IN, "#efe3f5"); ax.text(4.5, 0.12, "input: 0/1 Box-Ball state", ha="center", fontsize=9, color="#666")
    pair(ax, TOK[0], TOK[1], 1.28, 2.7, True,  PURPLE)
    pair(ax, TOK[2], TOK[3], 1.28, 2.7, False, PURPLE)
    up(ax, TOK[4], 1.28, 2.7, c=PURPLE)
    ax.text(9.15, 2.0, "layer 1\npair (0,1)(2,3)…", ha="center", va="center", fontsize=8, color=PURPLE, style="italic")
    for x in TOK: box(ax, x, 3.1, 1.15, 0.62, "", "#f6eefb", ec=PURPLE, fs=10)
    up(ax, TOK[0], 3.5, 4.9, c=PURPLE)
    pair(ax, TOK[1], TOK[2], 3.5, 4.9, True,  PURPLE)
    pair(ax, TOK[3], TOK[4], 3.5, 4.9, False, PURPLE)
    ax.text(9.15, 4.2, "layer 2\nshift pairing", ha="center", va="center", fontsize=8, color=PURPLE, style="italic")
    ax.text(4.5, 5.35, "⋮   × N layers   (pairing offset alternates each layer, so balls can move across)",
            ha="center", fontsize=9.5, color=PURPLE, style="italic")
    tokens(ax, 6.4, OUT, "#efe3f5"); ax.text(4.5, 7.05, "output: predicted state 2 steps later", ha="center", fontsize=9, color="#666")
    for x in TOK: up(ax, x, 5.7, 6.02, c=PURPLE)
    box(ax, 4.5, 8.5, 9.0, 1.15,
        "each layer: pair up neighbors; a gate  g  swaps the pair or not.\n"
        "a swap conserves the pair's count + is its own inverse, and  g  reads only FROZEN cells\n"
        "⇒ the whole stack is EXACTLY reversible & conservative — for any learned gate.",
        "#f7f0fb", ec=PURPLE, fs=9.5)
    ax.text(4.5, -0.35, "but swaps are LOCAL — they can't express BBS's nonlocal left→right carrier  ⇒  couldn't learn it (accuracy stuck at the trivial baseline)",
            ha="center", va="top", fontsize=9.5, color=RED)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97]); fig.savefig("architecture_02_swap_automaton.png", dpi=140, bbox_inches="tight")
    print("saved architecture_02_swap_automaton.png"); plt.close(fig)

# ══════════════════════════ Test 3 ══════════════════════════
def draw_03():
    fig, ax = plt.subplots(figsize=(11.5, 7.4)); ax.set_xlim(-0.3, 9.3); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Test 3 — conserving carrier  ·  same left→right scan, but each step CONSERVES", fontsize=13, color=GREEN, pad=12)
    tokens(ax, 0.9, IN, "#e2f0e6"); ax.text(4.5, 0.12, "input: 0/1 Box-Ball state", ha="center", fontsize=9, color="#666")
    cy = 3.3
    for x in TOK:
        box(ax, x, cy, 1.15, 1.0, "cell", "#d7efdd", ec=GREEN, fs=10, bold=True)
        up(ax, x, 1.3, cy - 0.52); up(ax, x, cy + 0.52, 5.05)
    ax.text(0.05, cy, "k₀=0", ha="center", va="center", fontsize=9, color=GREEN)
    ax.annotate("", xy=(TOK[0]-0.58, cy), xytext=(0.5, cy), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.7))
    for i in range(len(TOK)-1):
        ax.annotate("", xy=(TOK[i+1]-0.58, cy), xytext=(TOK[i]+0.58, cy), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.7))
    ax.text((TOK[0]+TOK[1])/2, cy+0.42, "carrier  k  =  #balls held  (integer)", ha="center", fontsize=8.5, color=GREEN, style="italic")
    tokens(ax, 5.55, OUT, "#e2f0e6"); ax.text(4.5, 6.2, "output: predicted state 2 steps later", ha="center", fontsize=9, color="#666")
    box(ax, 4.5, 8.35, 9.0, 1.35,
        "zoom · one cell —  total  t = cell + k  (a learned gate picks one of the only two count-preserving moves):\n"
        "EMIT → out=1, k→t−1        or        HOLD → out=0, k→t          (t=0 ⇒ forced HOLD)",
        "#f0faf2", ec=GREEN, fs=9.5)
    ax.annotate("", xy=(4.5, 7.68), xytext=(TOK[2], cy + 0.55), arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.9, ls=":"))
    ax.text(4.5, -0.35, "cell + carrier preserved at EVERY step ⇒ conserves by construction ·  'emit iff cell==0' = BBS (learned)  ·  100% acc / 100% conserved / 100% reversible",
            ha="center", va="top", fontsize=9.5, color=GREEN)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97]); fig.savefig("architecture_03_conserving_carrier.png", dpi=140, bbox_inches="tight")
    print("saved architecture_03_conserving_carrier.png"); plt.close(fig)

if __name__ == "__main__":
    draw_01(); draw_02(); draw_03()
