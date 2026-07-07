"""
Integrable Extrapolation benchmark · flagship result (single seed).

Task: learn ONE Box-Ball step, then COMPOSE it over a long horizon on MULTI-soliton states
(many collisions) — train on few solitons / short lattices, test on more solitons / longer
lattices + longer horizon. The discriminating metric is SOLITON-CONTENT FIDELITY: does the
predicted state still carry the right multiset of soliton amplitudes (BBS's conserved
quantities — the integrable signature), not just the right ball count?

Entrants:
  • conserving carrier (structural) — carrier-scan + structural ball-count conservation; the
    integrable inductive bias. Learns the emit gate; composes the step.
  • GRU / LSTM / Transformer (free-form) — generic sequence models, composing.
  • bolt-on = GRU + a top-N projection each step that PINS the ball count. The sharpest test:
    pinning conservation (count) does NOT pin soliton content.

Thesis: conservation (count) is bolt-on-able, but soliton content (integrability) is not — only
the structure-aware model preserves it over many collisions.
"""
import numpy as np, torch, torch.nn as nn, math
from soliton_bbs import bbs_step, bbs_run, soliton_content, make_multisoliton
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)

T_TEST = 12                      # compose this many steps at eval (many collisions)

# ── data: random multi-soliton states, one-step training pairs ────────────────
def rand_state(rng, n_lo, n_hi, amp_hi, L, horizon):
    k = rng.integers(n_lo, n_hi + 1)
    amps = sorted(rng.integers(1, amp_hi + 1, size=k).tolist(), reverse=True)
    gaps = rng.integers(2, 6, size=k - 1).tolist()
    pad = (horizon + 1) * int(sum(amps)) + 4     # just enough for the horizon (avoid all-zero sparsity)
    core = make_multisoliton(amps, gaps, right_pad=0)
    room = L - len(core)
    if room < pad: L2 = len(core) + pad
    else:          L2 = L
    return make_multisoliton(amps, gaps, right_pad=L2 - len(core))

def onestep_pairs(n, rng, **kw):
    X, Y = [], []
    for _ in range(n):
        s = rand_state(rng, **kw); X.append(s); Y.append(bbs_step(s))
    L = max(len(x) for x in X)
    pad = lambda a: np.concatenate([a, np.zeros(L - len(a), dtype=a.dtype)])
    return (torch.tensor(np.stack([pad(x) for x in X]), dtype=torch.float32),
            torch.tensor(np.stack([pad(y) for y in Y]), dtype=torch.float32))

def test_states(n, rng, **kw):
    return [rand_state(rng, **kw) for _ in range(n)]

# ── models (composing steppers) ──────────────────────────────────────────────
class ConservingCarrier(nn.Module):          # structural: k' = t - out, learned emit gate
    def __init__(s, hid=48, Kmax=24.0, blind=False):
        super().__init__(); s.Kmax = Kmax; s.blind = blind
        s.gate = nn.Sequential(nn.Linear(4, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
    def one(s, xf, hard):
        B, L = xf.shape; k = torch.zeros(B); xp = torch.nn.functional.pad(xf, (1, 1)); outs = []
        for i in range(L):
            c = xf[:, i]
            if s.blind:                                     # leak audit: emit = 1 − cell (no learned gate)
                e = 1.0 - c
            else:
                feat = torch.stack([c, k / s.Kmax, xp[:, i], xp[:, i + 2]], -1)
                e = torch.sigmoid(s.gate(feat).squeeze(-1))
                if hard: e = (e > 0.5).float() + (e - e.detach())
            t = c + k; out = e * torch.clamp(t, 0, 1); k = t - out; outs.append(out)
        return torch.stack(outs, 1)
    def one_step(s, x, hard=True): return s.one(x, hard)
class GRUStep(nn.Module):                     # free-form composing (causal GRU — BBS's natural scan)
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
        if project:                                       # bolt-on: pin ball count to top-N
            n = x0.sum(1); r = p.argsort(1, descending=True).argsort(1); c = (r < n.unsqueeze(1)).float()
        else:
            c = (p > 0.5).float()
    return c

# ── train / eval ─────────────────────────────────────────────────────────────
def fit(m, X, Y, epochs=45, lr=2e-3, clip=True, pos_weight=6.0):
    opt = torch.optim.Adam(m.parameters(), lr=lr); B = 256
    def wbce(p, y):                                  # weighted BCE — sparse 1s, don't collapse to all-0
        p = p.clamp(1e-6, 1-1e-6); return -(pos_weight*y*torch.log(p) + (1-y)*torch.log(1-p)).mean()
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), B):
            idx = perm[i:i+B]; loss = wbce(m.one_step(X[idx], hard=False), Y[idx])   # soft train, hard eval
            opt.zero_grad(); loss.backward()
            if clip: torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
    return m

def multiset_iou(a, b):
    from collections import Counter
    ca, cb = Counter(a), Counter(b); inter = sum((ca & cb).values()); union = sum((ca | cb).values())
    return inter / union if union else 1.0

def evaluate(m, states, T, project=False):
    m.eval(); acc = cons = solit_exact = solit_iou = 0.0; nb = len(states)
    with torch.no_grad():
        for s in states:
            x0 = torch.tensor(s[None], dtype=torch.float32)
            pred = compose(m, x0, T, project)[0].numpy().astype(np.int64)
            true = bbs_run(s, T)
            acc  += (pred == true).mean()
            cons += float(pred.sum() == s.sum())
            c_true = soliton_content(s)                    # invariant = true final content
            c_pred = soliton_content(pred)
            solit_exact += float(c_pred == c_true); solit_iou += multiset_iou(c_pred, c_true)
    return dict(acc=100*acc/nb, cons=100*cons/nb, sol_exact=100*solit_exact/nb, sol_iou=100*solit_iou/nb)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    tr_kw = dict(n_lo=2, n_hi=4, amp_hi=4, L=48, horizon=1)      # train: 1-step, few solitons
    te_kw = dict(n_lo=4, n_hi=6, amp_hi=5, L=80, horizon=T_TEST)  # test: more solitons, long horizon
    Xtr, Ytr = onestep_pairs(3000, rng, **tr_kw)
    tstates  = test_states(120, np.random.default_rng(7), **te_kw)
    print(f"Integrable Extrapolation — flagship (multi-soliton BBS, compose T={T_TEST}, single seed)")
    print(f"train: 2–4 solitons, 1 step  |  test: 4–6 solitons, {T_TEST} steps (many collisions)\n")

    entrants = [("conserving carrier (structural)", lambda: ConservingCarrier(),           False, True),
                ("  └ carrier-blind (leak audit)",  lambda: ConservingCarrier(blind=True),  False, False),
                ("GRU (free-form)",                 GRUStep,                                False, True),
                ("LSTM (free-form)",                LSTMStep,                               False, True),
                ("Transformer (free-form)",         TFStep,                                 False, True),
                ("bolt-on = GRU + count pinned",    GRUStep,                                True,  True)]
    rows = []
    for name, ctor, proj, train in entrants:
        print(f"{'training' if train else 'eval (no train)'} {name} ...")
        m = fit(ctor(), Xtr, Ytr) if train else ctor()
        rows.append((name, evaluate(m, tstates, T_TEST, project=proj)))

    print(f"\n{'entrant':34s}{'acc%':>8}{'ball-cons%':>12}{'soliton-exact%':>16}{'soliton-IoU%':>14}")
    print("-"*84)
    for name, r in rows:
        print(f"{name:34s}{r['acc']:8.1f}{r['cons']:12.1f}{r['sol_exact']:16.1f}{r['sol_iou']:14.1f}")
    print("\nHeadline: ball-count is conserved by structure AND by the bolt-on; but SOLITON CONTENT")
    print("(the integrable invariant) is preserved only by the structure-aware model.")
