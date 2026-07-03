"""
Visualize what the L=32 Box-Ball System data looks like (animation).

Top strip  = the current 0/1 lattice (balls filled, colored by soliton/block size).
Bottom     = a space-time diagram building up row by row (time downward), so you
             see the soliton worldlines: big blocks move faster and PASS THROUGH
             small ones, and the ball-count stays constant.

The learning task uses t -> t+2 (input row and the "+2 = target" row are marked);
we run a few more steps here just to show the dynamics.

Outputs bbs_data_l32.gif (committed) and bbs_data_l32.mp4 (local, git-ignored).
Requires ffmpeg on PATH.
"""
import os, subprocess, tempfile, shutil
import numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def bbs_step(s):
    out = np.zeros_like(s); c = 0
    for i in range(len(s)):
        if s[i] == 1: c += 1; out[i] = 0
        else:
            if c > 0: out[i] = 1; c -= 1
    return out

L = 32
s0 = np.zeros(L, dtype=int)
for a, b in [(2, 5), (10, 11), (16, 18)]:   # a size-3, a size-1, a size-2 soliton
    s0[a:b] = 1
NB = 8                                       # rows t=0..7 (all 6 balls stay on the lattice)
states = [s0.copy()]
for _ in range(NB - 1): states.append(bbs_step(states[-1]))

def size_index(s):                           # each ball cell -> the size of its block (for coloring)
    idx = np.zeros(len(s), int); i = 0
    while i < len(s):
        if s[i] == 1:
            j = i
            while j < len(s) and s[j] == 1: j += 1
            idx[i:j] = j - i; i = j
        else: i += 1
    return idx
IDX = np.array([size_index(s) for s in states])   # (NB, L)

# 0 = empty, 1/2/3 = soliton size
cmap = ListedColormap(["#f2f2f2", "#2980b9", "#e67e22", "#c0392b", "#8e44ad"])
xg, yg = np.arange(L + 1), np.arange(NB + 1)

frames_dir = tempfile.mkdtemp(prefix="bbs_frames_")
HOLD = 5; fi = 0
for t in range(NB):
    fig, (axT, axS) = plt.subplots(2, 1, figsize=(11, 6.2), gridspec_kw={"height_ratios": [1, 4.2]})
    # top: current lattice
    axT.pcolormesh(xg, np.arange(2), IDX[t][None, :], cmap=cmap, vmin=0, vmax=4,
                   edgecolors="white", linewidth=0.6)
    axT.set_yticks([]); axT.set_xticks(np.arange(0, L + 1, 4)); axT.set_xlim(0, L); axT.invert_yaxis()
    axT.set_title(f"Box-Ball System · L=32 · step t = {t}     (balls = {int(states[t].sum())}, conserved)",
                  fontsize=12)
    # bottom: space-time, revealed up to t
    grid = np.zeros((NB, L)); grid[:t + 1] = IDX[:t + 1]
    axS.pcolormesh(xg, yg, grid, cmap=cmap, vmin=0, vmax=4, edgecolors="white", linewidth=0.4)
    axS.invert_yaxis()
    axS.set_xlabel("position along the lattice  (0 … 31)"); axS.set_ylabel("time step")
    axS.set_yticks(np.arange(NB) + 0.5); axS.set_yticklabels([f"t={i}" for i in range(NB)])
    axS.set_xticks(np.arange(0, L + 1, 4))
    axS.annotate("input", xy=(0, 0.5), xytext=(-3.6, 0.5), va="center", ha="right", fontsize=9, color="#333")
    axS.annotate("+2 = target", xy=(0, 2.5), xytext=(-3.6, 2.5), va="center", ha="right", fontsize=9, color="#c0392b")
    axS.set_xlim(-0.5, L)
    axS.legend(handles=[Patch(facecolor="#2980b9", label="soliton size 1"),
                        Patch(facecolor="#e67e22", label="size 2"),
                        Patch(facecolor="#c0392b", label="size 3")],
               loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle("What the L=32 data looks like: bigger solitons move faster, overtake and pass through smaller ones",
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    hold = HOLD * (3 if t == NB - 1 else 1)          # linger on the final frame
    for _ in range(hold):
        fig.savefig(os.path.join(frames_dir, f"f{fi:03d}.png"), dpi=110); fi += 1
    plt.close(fig)

FPS = 6; seq = os.path.join(frames_dir, "f%03d.png"); pal = os.path.join(frames_dir, "pal.png")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", seq,
                "-vf", "palettegen=stats_mode=full", pal], check=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", seq, "-i", pal,
                "-lavfi", "paletteuse=dither=none", "bbs_data_l32.gif"], check=True)
print("saved bbs_data_l32.gif")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", seq,
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "bbs_data_l32.mp4"], check=True)
print("saved bbs_data_l32.mp4  (local only; git-ignored)")
shutil.rmtree(frames_dir, ignore_errors=True)
