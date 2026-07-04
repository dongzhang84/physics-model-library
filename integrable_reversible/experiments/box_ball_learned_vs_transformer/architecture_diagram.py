"""
Architecture diagram: the two models in 01_plain_carrier.py, side by side.

Left  = Transformer      (global self-attention, no built-in structure)
Right = LearnedCarrier   (left->right recurrent scan carrying a state h)

Same task for both: given a 0/1 Box-Ball state, predict it 2 steps later.
Outputs architecture_diagram.png (static schematic; no data / training needed).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, GREEN, GRY, ARR = "#2c3e50", "#1e8449", "#eef1f4", "#555"
TOK = [1.3, 2.9, 4.5, 6.1, 7.7]           # x-centres of the 5 example tokens
IN  = ["1", "1", "0", "1", "0"]
OUT = ["0", "1", "1", "0", "1"]

def box(ax, x, y, w, h, text, fc, ec="#333", fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.09", fc=fc, ec=ec, lw=1.3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#111")

def up(ax, x, y0, y1, c=ARR, lw=1.6):
    ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=lw))

def tokens(ax, y, labels, fc):
    for x, t in zip(TOK, labels):
        box(ax, x, y, 1.15, 0.72, t, fc, fs=11, bold=True)

fig, (aL, aR) = plt.subplots(1, 2, figsize=(15, 7.6))
for ax in (aL, aR):
    ax.set_xlim(0, 9); ax.set_ylim(0, 10); ax.axis("off")

# ───────────────────────── Transformer ─────────────────────────
aL.set_title("Transformer  ·  global, parallel, no built-in structure", fontsize=13, color=BLUE, pad=12)
tokens(aL, 0.8, IN, "#dfe6ec")
aL.text(4.5, 0.05, "input: 0/1 Box-Ball state", ha="center", fontsize=9, color="#666")
box(aL, 4.5, 2.15, 7.4, 0.7, "Embed (0/1 → vector) + sinusoidal position encoding", "#eaf0f6")
box(aL, 4.5, 3.75, 7.4, 1.05, "Self-attention — every position attends to EVERY position", "#d6e4f0")
# all-to-all hint: dots + a few connectors inside the attention box
ys = 3.75
for x in TOK: aL.plot(x, ys, "o", ms=5, color=BLUE, zorder=5)
for i in range(len(TOK)):
    for j in range(i+1, len(TOK)):
        aL.plot([TOK[i], TOK[j]], [ys, ys], color=BLUE, lw=0.5, alpha=0.35, zorder=4)
box(aL, 4.5, 5.35, 7.4, 0.7, "Feed-forward  (each position, independently)", "#eaf0f6")
aL.text(8.55, 4.55, "× 3\nlayers", ha="center", va="center", fontsize=9, color=BLUE, style="italic")
box(aL, 4.5, 6.95, 7.4, 0.7, "Linear → 0/1 per position", "#eaf0f6")
tokens(aL, 8.5, OUT, "#dfe6ec")
aL.text(4.5, 9.35, "output: predicted state 2 steps later", ha="center", fontsize=9, color="#666")
for y0, y1 in [(1.2, 1.8), (2.5, 3.22), (4.27, 5.0), (5.7, 6.6), (7.3, 8.14)]:
    up(aL, 4.5, y0, y1)
aL.text(4.5, -0.55, "every output computed from the whole sequence at once —\nno left→right order, no carried state, no conserved quantity",
        ha="center", va="top", fontsize=9.5, color=BLUE)

# ───────────────────────── LearnedCarrier ─────────────────────────
aR.set_title("LearnedCarrier  ·  left→right recurrence carrying a state  h", fontsize=13, color=GREEN, pad=12)
tokens(aR, 0.8, IN, "#e2f0e6")
aR.text(4.5, 0.05, "input: 0/1 Box-Ball state", ha="center", fontsize=9, color="#666")
# a row of shared cells, one above each token
cy = 3.4
for x in TOK:
    box(aR, x, cy, 1.15, 1.0, "cell", "#d7efdd", ec=GREEN, fs=10, bold=True)
    up(aR, x, 1.2, cy - 0.52)               # input -> cell
    up(aR, x, cy + 0.52, 5.05)              # cell  -> output
# carrier arrows left->right between cells (the state h)
aR.text(0.2, cy, "h₀=0", ha="center", va="center", fontsize=9, color=GREEN)
aR.annotate("", xy=(TOK[0]-0.58, cy), xytext=(0.55, cy), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.6))
for i in range(len(TOK)-1):
    aR.annotate("", xy=(TOK[i+1]-0.58, cy), xytext=(TOK[i]+0.58, cy),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.6))
aR.text((TOK[0]+TOK[1])/2, cy+0.42, "carrier  h", ha="center", fontsize=9, color=GREEN, style="italic")
tokens(aR, 5.55, OUT, "#e2f0e6")
aR.text(4.5, 6.15, "output: predicted state 2 steps later", ha="center", fontsize=9, color="#666")
# inside-one-cell callout (a zoom, not part of the vertical flow)
box(aR, 4.5, 8.3, 7.9, 1.15,
    "zoom · one cell (weights shared across all positions):\n[ bitᵢ ,  hᵢ ]  →  small MLP  →  ( outᵢ ,  hᵢ₊₁ )",
    "#f0faf2", ec=GREEN, fs=10)
aR.annotate("", xy=(4.5, 7.7), xytext=(TOK[2], cy + 0.55),
            arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.9, ls=":"))
aR.text(4.5, -0.55, "same cell weights at every position → runs at ANY length (extrapolates) ·\napplied ×2 steps · emits 0/1 freely → accurate but does NOT conserve",
        ha="center", va="top", fontsize=9.5, color=GREEN)

fig.suptitle("Two models, same task (01_plain_carrier.py): predict the Box-Ball state 2 steps ahead",
             fontsize=14, y=0.99)
fig.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.savefig("architecture_diagram.png", dpi=140, bbox_inches="tight")
print("saved architecture_diagram.png")
