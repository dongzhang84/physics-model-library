"""
Phase 0 · path B, step 2 — the gate. Single seed.

On finite-carrier BBS (verified: reversible, amplitude content conserved), train models to do one
step and compose over a long multi-soliton horizon. Score the SOLITON-AMPLITUDE content (a multiset,
richer than ball count). The two questions the gate must answer YES to:

  (genuine learning)  does the carrier-blind audit LOSE amplitude-content fidelity vs the learned
                      structural model? (if blind is also high → hard-coded leak → FAIL)
  (depth)             does the bolt-on (ball-count pinned) FAIL to recover amplitude content?
                      (ball count is bolt-on-able; the amplitude multiset should not be)
"""
import numpy as np, torch, torch.nn as nn, math
from collections import Counter
from fc_bbs import fc_step, fc_run, amplitude_content, make_multisoliton
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)

K = 4; AMP_MAX = 7; T_TEST = 6   # K / AMP_MAX are reset per sweep iteration below

def rand_state(rng, n_lo, n_hi, L_min, horizon):
    k = rng.integers(n_lo, n_hi + 1)
    amps = sorted(rng.integers(1, AMP_MAX + 1, size=k).tolist(), reverse=True)   # many amps exceed K
    gaps = rng.integers(3, 6, size=k - 1).tolist()
    pad = (horizon + 1) * int(sum(amps)) + 6
    return make_multisoliton(amps, gaps, right_pad=max(pad, L_min - (sum(amps) + sum(gaps))))

def onestep_pairs(n, rng, **kw):
    X, Y = [], []
    for _ in range(n):
        s = rand_state(rng, **kw); X.append(s); Y.append(fc_step(s, K))
    L = max(len(x) for x in X); pad = lambda a: np.concatenate([a, np.zeros(L - len(a), dtype=a.dtype)])
    return (torch.tensor(np.stack([pad(x) for x in X]), dtype=torch.float32),
            torch.tensor(np.stack([pad(y) for y in Y]), dtype=torch.float32))

# ── models ───────────────────────────────────────────────────────────────────
def mlp(i, o, h=128, n=3):
    L = [nn.Linear(i, h), nn.GELU()]
    for _ in range(n - 1): L += [nn.Linear(h, h), nn.GELU()]
    return nn.Sequential(*L, nn.Linear(h, o))
class ConservingCarrier(nn.Module):                # structural: k'=t-out, learned emit gate
    def __init__(s, hid=48, Kmax=8.0, blind=False):
        super().__init__(); s.Kmax = Kmax; s.blind = blind
        s.gate = nn.Sequential(nn.Linear(4, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
    def one(s, xf, hard):
        B, L = xf.shape; k = torch.zeros(B); xp = torch.nn.functional.pad(xf, (1, 1)); outs = []
        for i in range(L):
            c = xf[:, i]
            if s.blind:
                e = 1.0 - c                        # leak audit: emit=1-cell, ignores the carrier
            else:
                feat = torch.stack([c, k / s.Kmax, xp[:, i], xp[:, i + 2]], -1)
                e = torch.sigmoid(s.gate(feat).squeeze(-1))
                if hard: e = (e > 0.5).float() + (e - e.detach())
            t = c + k; out = e * torch.clamp(t, 0, 1); k = t - out; outs.append(out)
        return torch.stack(outs, 1)
    def one_step(s, x, hard=True): return s.one(x, hard)
class GRUStep(nn.Module):
    def __init__(s, d=96): super().__init__(); s.gru = nn.GRU(1, d, batch_first=True); s.head = nn.Linear(d, 1)
    def one_step(s, x, hard=True): h, _ = s.gru(x.unsqueeze(-1)); return torch.sigmoid(s.head(h)).squeeze(-1)
class LSTMStep(nn.Module):
    def __init__(s, d=96): super().__init__(); s.lstm = nn.LSTM(1, d, batch_first=True); s.head = nn.Linear(d, 1)
    def one_step(s, x, hard=True): h, _ = s.lstm(x.unsqueeze(-1)); return torch.sigmoid(s.head(h)).squeeze(-1)
class PE(nn.Module):
    def __init__(s, d, m=8192):
        super().__init__(); pe = torch.zeros(m, d); pos = torch.arange(m).unsqueeze(1).float()
        dv = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * dv); pe[:, 1::2] = torch.cos(pos * dv); s.register_buffer("pe", pe)
    def forward(s, x): return x + s.pe[:x.size(1)].unsqueeze(0)
class TFStep(nn.Module):
    def __init__(s, d=96, heads=4, layers=3):
        super().__init__(); s.emb = nn.Linear(1, d); s.pe = PE(d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, dropout=0.0, activation="gelu")
        s.enc = nn.TransformerEncoder(enc, layers); s.head = nn.Linear(d, 1)
    def one_step(s, x, hard=True): return torch.sigmoid(s.head(s.enc(s.pe(s.emb(x.unsqueeze(-1)))))).squeeze(-1)

def compose(m, x0, T, project=False):
    c = x0
    for _ in range(T):
        p = m.one_step(c)
        if project:
            n = x0.sum(1); r = p.argsort(1, descending=True).argsort(1); c = (r < n.unsqueeze(1)).float()
        else:
            c = (p > 0.5).float()
    return c

def fit(m, X, Y, epochs=30, lr=2e-3, pos_weight=5.0):
    opt = torch.optim.Adam(m.parameters(), lr=lr); B = 256
    def wbce(p, y): p = p.clamp(1e-6, 1 - 1e-6); return -(pos_weight * y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), B):
            idx = perm[i:i + B]; loss = wbce(m.one_step(X[idx], hard=False), Y[idx])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m

def iou(a, b):
    ca, cb = Counter(a), Counter(b); i = sum((ca & cb).values()); u = sum((ca | cb).values())
    return i / u if u else 1.0
def evaluate(m, states, T, project=False):
    m.eval(); acc = cons = c_exact = c_iou = 0.0; n = len(states)
    with torch.no_grad():
        for s in states:
            x0 = torch.tensor(s[None], dtype=torch.float32)
            pred = compose(m, x0, T, project)[0].numpy().astype(np.int64); true = fc_run(s, T, K)
            acc += (pred == true).mean(); cons += float(pred.sum() == s.sum())
            ct = amplitude_content(s, K); cp = amplitude_content(pred, K)
            c_exact += float(cp == ct); c_iou += iou(cp, ct)
    return dict(acc=100 * acc / n, cons=100 * cons / n, amp_exact=100 * c_exact / n, amp_iou=100 * c_iou / n)

if __name__ == "__main__":
    entrants = [("conserving carrier (structural)", lambda: ConservingCarrier(), False, True),
                ("  carrier-blind (leak audit)",    lambda: ConservingCarrier(blind=True), False, False),
                ("GRU (free-form)",                 GRUStep, False, True),
                ("bolt-on = GRU + count pinned",    GRUStep, True, True)]
    for Kval in [2, 3]:
        K = Kval; AMP_MAX = Kval + 4          # more amps exceed capacity -> carrier-blind should drop more
        rng = np.random.default_rng(0)
        Xtr, Ytr = onestep_pairs(1000, rng, n_lo=2, n_hi=4, L_min=36, horizon=1)
        tstates = [rand_state(np.random.default_rng(7 + i), n_lo=3, n_hi=5, L_min=44, horizon=T_TEST) for i in range(30)]
        print(f"\n==== K={K}, amps up to {AMP_MAX}, compose T={T_TEST}, single seed ====", flush=True)
        rows = []
        for name, ctor, proj, train in entrants:
            print(f"  {'train' if train else 'eval '} {name} ...", flush=True)
            m = fit(ctor(), Xtr, Ytr, epochs=25) if train else ctor()
            rows.append((name, evaluate(m, tstates, T_TEST, project=proj)))
        print(f"  {'entrant':34s}{'acc%':>8}{'count%':>9}{'amp-exact%':>12}{'amp-IoU%':>10}")
        for name, r in rows: print(f"  {name:34s}{r['acc']:8.1f}{r['cons']:9.1f}{r['amp_exact']:12.1f}{r['amp_iou']:10.1f}")
        st = dict(rows)["conserving carrier (structural)"]; bl = dict(rows)["  carrier-blind (leak audit)"]
        print(f"  >> genuine-learning margin (structural − blind, amp-exact): {st['amp_exact'] - bl['amp_exact']:.1f} pts", flush=True)
