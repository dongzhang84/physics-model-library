import numpy as np
import matplotlib.pyplot as plt

# Task (standard 1): encode K symbols -> K solitons -> send through ONE shared
# channel where they overtake & collide -> decode symbols at the output.
# Engine = integrable system (KdV). Signature: solitons emerge from collisions
# with shape intact (only a phase shift), so the symbols survive. Control =
# a non-integrable (linear dispersive) channel where pulses smear into mush.

# u_t + 6 u u_x + u_xxx = 0 ; soliton: u=(c/2) sech^2( sqrt(c)/2 (x-ct-x0) ), speed c
Lx, N = 120.0, 1024
x = np.linspace(-Lx/2, Lx/2, N, endpoint=False)
dx = x[1]-x[0]
k = 2*np.pi*np.fft.fftfreq(N, d=dx)

ik3 = 1j*k**3
E  = np.exp(ik3*0)          # placeholder, set with dt below
def make_E(dt):
    return np.exp(ik3*dt/2), np.exp(ik3*dt)

g = None
def kdv_ifrk4(v, dt, Eh, E2, g):
    a = g*np.fft.fft(np.real(np.fft.ifft(v))**2)
    b = g*np.fft.fft(np.real(np.fft.ifft(Eh*(v+a/2)))**2)
    c = g*np.fft.fft(np.real(np.fft.ifft(Eh*v + b/2))**2)
    d = g*np.fft.fft(np.real(np.fft.ifft(E2*v + Eh*c))**2)
    return E2*v + (E2*a + 2*Eh*(b+c) + d)/6

def soliton(c, x0):
    return (c/2.0)/np.cosh(np.sqrt(c)/2.0*(x-x0))**2

symbol_to_c = {0:1.0, 1:2.0, 2:3.0, 3:4.0}
message = [3,1,2,0]
starts  = [-40,-28,-16,-4]     # faster (taller) ones behind -> they overtake & collide

def encode(msg):
    u = np.zeros(N)
    for s,x0 in zip(msg,starts):
        u += soliton(symbol_to_c[s], x0)
    return u

u0 = encode(message)
dt = 2e-4; T = 40000
Eh, E2 = make_E(dt)
g = -3j*k*dt   # nonlinear coeff for -6 u u_x = -3 (u^2)_x

def run_kdv():
    v = np.fft.fft(u0.copy()); snaps=[u0.copy()]
    for n in range(T):
        v = kdv_ifrk4(v, dt, Eh, E2, g)
        if (n+1) % (T//5)==0: snaps.append(np.real(np.fft.ifft(v)))
    return np.real(np.fft.ifft(v)), snaps

def run_linear():           # non-integrable control: dispersion only, no soliton stability
    v = np.fft.fft(u0.copy()); snaps=[u0.copy()]
    for n in range(T):
        v = E2*v
        if (n+1) % (T//5)==0: snaps.append(np.real(np.fft.ifft(v)))
    return np.real(np.fft.ifft(v)), snaps

u_kdv, snaps_kdv = run_kdv()
u_lin, snaps_lin = run_linear()

def decode(u):
    peaks=[]
    for i in range(2,N-2):
        if u[i]>0.3 and u[i]>=u[i-1] and u[i]>u[i+1]:
            if not peaks or x[i]-peaks[-1][0]>1.0:
                peaks.append((x[i],u[i]))
    syms=[min(symbol_to_c,key=lambda s:abs(symbol_to_c[s]-2*h)) for _,h in peaks]
    return syms, peaks

dec_kdv,pk_kdv = decode(u_kdv)
dec_lin,pk_lin = decode(u_lin)
ok_kdv = sorted(dec_kdv)==sorted(message)
ok_lin = sorted(dec_lin)==sorted(message)
print("sent      :", message, "c=",[symbol_to_c[s] for s in message])
print("INTEGRABLE:", dec_kdv, "recovered all?", ok_kdv)
print("non-integ :", dec_lin, "recovered all?", ok_lin)
print("max|u| kdv:", np.max(np.abs(u_kdv)))

fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
cols=plt.cm.viridis(np.linspace(0,0.85,len(snaps_kdv)))
for s,col in zip(snaps_kdv,cols): axes[0].plot(x,s,color=col,lw=1.1)
for xp,h in pk_kdv: axes[0].annotate(f"c≈{2*h:.1f}",(xp,h),textcoords="offset points",xytext=(0,4),fontsize=8,ha="center")
axes[0].set_title(f"INTEGRABLE channel (KdV solitons): 4 symbols sent down ONE line, collide, exit intact  "
                  f"→ decoded {dec_kdv}  {'✓ all recovered' if ok_kdv else '✗'}",fontsize=10)
axes[0].set_ylabel("amplitude")
for s,col in zip(snaps_lin,cols): axes[1].plot(x,s,color=col,lw=1.1)
axes[1].set_title(f"NON-integrable channel (linear dispersion): same 4 symbols smear into mush  "
                  f"→ decoded {len(dec_lin)} bumps  {'✓' if ok_lin else '✗ symbols lost'}",fontsize=10)
axes[1].set_ylabel("amplitude"); axes[1].set_xlabel("position along shared channel   (color = time, dark→bright)")
axes[1].set_xlim(-55,55); axes[0].set_xlim(-55,55)
fig.suptitle("Integrable-system engine doing an AI-style task: interference-free multi-symbol channel",fontsize=12,y=0.97)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("soliton_channel_demo.png",dpi=130,bbox_inches="tight")
print("saved.")
