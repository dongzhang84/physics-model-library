"""Animated version of the soliton-channel demo.

Same physics as soliton_channel.py, but instead of one static figure it plays
the full time evolution as a movie:

  TOP  — INTEGRABLE channel (KdV): 4 solitons overtake, collide, and pass
         through each other with shape/height intact.
  BOT  — NON-integrable control (linear dispersion): the same 4 pulses smear
         and delocalize; the symbols are lost.

Outputs soliton_channel.gif — palette-optimized (~1 MB) so it embeds and
auto-plays in the README. Renders a temporary mp4 as an intermediate and
deletes it (we do not keep an mp4). Requires ffmpeg on PATH.
Does not touch soliton_channel.py or soliton_channel_demo.png.
"""

import os, subprocess, tempfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ── physics (identical to soliton_channel.py) ───────────────────────────────
Lx, N = 120.0, 1024
x = np.linspace(-Lx/2, Lx/2, N, endpoint=False)
dx = x[1]-x[0]
k = 2*np.pi*np.fft.fftfreq(N, d=dx)
ik3 = 1j*k**3

def make_E(dt):
    return np.exp(ik3*dt/2), np.exp(ik3*dt)

def kdv_ifrk4(v, Eh, E2, g):
    a = g*np.fft.fft(np.real(np.fft.ifft(v))**2)
    b = g*np.fft.fft(np.real(np.fft.ifft(Eh*(v+a/2)))**2)
    c = g*np.fft.fft(np.real(np.fft.ifft(Eh*v + b/2))**2)
    d = g*np.fft.fft(np.real(np.fft.ifft(E2*v + Eh*c))**2)
    return E2*v + (E2*a + 2*Eh*(b+c) + d)/6

def soliton(c, x0):
    return (c/2.0)/np.cosh(np.sqrt(c)/2.0*(x-x0))**2

symbol_to_c = {0:1.0, 1:2.0, 2:3.0, 3:4.0}
message = [3,1,2,0]
starts  = [-40,-28,-16,-4]

u0 = np.zeros(N)
for s, x0 in zip(message, starts):
    u0 += soliton(symbol_to_c[s], x0)

dt = 2e-4; T = 40000
Eh, E2 = make_E(dt)
g = -3j*k*dt

# ── collect frames of the full evolution (both channels) ────────────────────
FRAMES = 200
stride = T // FRAMES

vk = np.fft.fft(u0.copy())          # KdV state (Fourier)
vl = np.fft.fft(u0.copy())          # linear-dispersion state (Fourier)
kdv_frames = [u0.copy()]
lin_frames = [u0.copy()]
times = [0.0]
for n in range(T):
    vk = kdv_ifrk4(vk, Eh, E2, g)
    vl = E2*vl                       # pure linear dispersion, no nonlinearity
    if (n+1) % stride == 0:
        kdv_frames.append(np.real(np.fft.ifft(vk)))
        lin_frames.append(np.real(np.fft.ifft(vl)))
        times.append((n+1)*dt)
print(f"collected {len(kdv_frames)} frames")

# ── figure ──────────────────────────────────────────────────────────────────
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
XL = (-55, 55); YL = (-0.4, 2.3)
for ax in (ax0, ax1):
    ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_ylabel("amplitude")
    ax.grid(alpha=0.15)

(line0,) = ax0.plot([], [], color="#1f77b4", lw=1.8)
fill0 = [ax0.fill_between(x, 0, 0, color="#1f77b4", alpha=0.15)]
(line1,) = ax1.plot([], [], color="#d62728", lw=1.8)
fill1 = [ax1.fill_between(x, 0, 0, color="#d62728", alpha=0.12)]

ax0.set_title("INTEGRABLE channel (KdV solitons): 4 symbols collide and pass through INTACT", fontsize=11)
ax1.set_title("NON-integrable control (linear dispersion): the same 4 pulses SMEAR away", fontsize=11)
ax1.set_xlabel("position along the shared channel")
tt = ax0.text(0.985, 0.90, "", transform=ax0.transAxes, ha="right", fontsize=10,
              bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))
fig.suptitle("Integrable-system engine as an AI-style task: interference-free multi-symbol channel",
             fontsize=12, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])

def draw(i):
    uk, ul = kdv_frames[i], lin_frames[i]
    line0.set_data(x, uk)
    line1.set_data(x, ul)
    fill0[0].remove(); fill0[0] = ax0.fill_between(x, 0, uk, color="#1f77b4", alpha=0.15)
    fill1[0].remove(); fill1[0] = ax1.fill_between(x, 0, ul, color="#d62728", alpha=0.12)
    tt.set_text(f"t = {times[i]:4.1f}     peak |u|:  KdV={np.abs(uk).max():.2f}   disp={np.abs(ul).max():.2f}")
    return line0, line1, fill0[0], fill1[0], tt

anim = FuncAnimation(fig, draw, frames=len(kdv_frames), interval=40, blit=False)

# Render a temporary high-quality mp4, then convert to a palette-optimized GIF
# via ffmpeg (much smaller than a direct pillow GIF for this busy content).
tmp_mp4 = os.path.join(tempfile.gettempdir(), "soliton_channel_tmp.mp4")
pal     = os.path.join(tempfile.gettempdir(), "soliton_palette.png")
anim.save(tmp_mp4, writer=FFMpegWriter(fps=25, bitrate=2400), dpi=130)

vf = "fps=10,scale=560:-1:flags=lanczos"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_mp4,
                "-vf", f"{vf},palettegen=stats_mode=diff", pal], check=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_mp4, "-i", pal,
                "-lavfi", f"{vf}[x];[x][1:v]paletteuse=dither=none",
                "soliton_channel.gif"], check=True)
os.remove(tmp_mp4); os.remove(pal)
print("saved soliton_channel.gif")
