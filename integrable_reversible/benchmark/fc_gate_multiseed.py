"""
Phase 0 · path B — multi-seed, full-scale confirmation (K=3). Formal numbers for the candidate.

Reuses the models / fit / evaluate from fc_gate_probe.py. Reports mean ± std over N_SEEDS seeds
for all six entrants on finite-carrier BBS with the soliton-amplitude-content metric.
Until this passes, it is a candidate/signal — not yet a benchmark.
"""
import numpy as np, torch, json
import fc_gate_probe as P

P.K = 3; P.AMP_MAX = 7                       # K=3 was the cleanest in the sweep
N_SEEDS = 5; T = 8; N_TEST = 60; EPOCHS = 40

ENTRANTS = [("conserving carrier (structural)", lambda: P.ConservingCarrier(),            False, True),
            ("carrier-blind (leak audit)",      lambda: P.ConservingCarrier(blind=True),  False, False),
            ("GRU (free-form)",                 P.GRUStep,                                 False, True),
            ("LSTM (free-form)",                P.LSTMStep,                                False, True),
            ("Transformer (free-form)",         P.TFStep,                                  False, True),
            ("bolt-on = GRU + count pinned",    P.GRUStep,                                 True,  True)]
KEYS = ["acc", "cons", "amp_exact", "amp_iou"]

def one_seed(sd):
    torch.manual_seed(sd); np.random.seed(sd); rng = np.random.default_rng(sd)
    Xtr, Ytr = P.onestep_pairs(2000, rng, n_lo=2, n_hi=4, L_min=40, horizon=1)
    tstates = [P.rand_state(np.random.default_rng(1000 + sd * 100 + i), n_lo=3, n_hi=6, L_min=64, horizon=T)
               for i in range(N_TEST)]
    out = {}
    for name, ctor, proj, train in ENTRANTS:
        m = P.fit(ctor(), Xtr, Ytr, epochs=EPOCHS) if train else ctor()
        out[name] = P.evaluate(m, tstates, T, project=proj)
        r = out[name]
        print(f"  seed {sd}  {name:32s} acc {r['acc']:5.1f}  cons {r['cons']:5.1f}  "
              f"amp-exact {r['amp_exact']:5.1f}  amp-IoU {r['amp_iou']:5.1f}", flush=True)
    json.dump(out, open(f"fc_seed_{sd}.json", "w"))            # save incrementally (resumable)
    return out

if __name__ == "__main__":
    import os
    print(f"Phase 0 path B — multi-seed (K={P.K}, amps<= {P.AMP_MAX}, T={T}, {N_SEEDS} seeds, {N_TEST} test states)\n", flush=True)
    for sd in range(N_SEEDS):
        if os.path.exists(f"fc_seed_{sd}.json"):
            print(f"seed {sd}: cached, skipping", flush=True); continue
        print(f"seed {sd}:", flush=True); one_seed(sd)
    per = [json.load(open(f"fc_seed_{sd}.json")) for sd in range(N_SEEDS) if os.path.exists(f"fc_seed_{sd}.json")]
    N_SEEDS = len(per)

    agg = {}
    for name, *_ in ENTRANTS:
        agg[name] = {k: (float(np.mean([per[s][name][k] for s in range(N_SEEDS)])),
                         float(np.std([per[s][name][k] for s in range(N_SEEDS)]))) for k in KEYS}
    print(f"\n{'entrant':34s}{'acc%':>11}{'count%':>11}{'amp-exact%':>13}{'amp-IoU%':>12}   (mean±std)")
    print("-" * 91)
    for name, *_ in ENTRANTS:
        a = agg[name]
        print(f"{name:34s}" + "".join(f"{a[k][0]:6.1f}±{a[k][1]:<4.1f}" for k in KEYS))
    st = agg["conserving carrier (structural)"]["amp_exact"][0]; bl = agg["carrier-blind (leak audit)"]["amp_exact"][0]
    bo = agg["bolt-on = GRU + count pinned"]
    print(f"\ngenuine-learning margin (structural − blind, amp-exact): {st - bl:.1f} pts")
    print(f"depth: bolt-on count {bo['cons'][0]:.0f}% but amp-exact {bo['amp_exact'][0]:.0f}%")
    json.dump({"K": P.K, "T": T, "n_seeds": N_SEEDS, "agg": agg}, open("fc_gate_multiseed.json", "w"), indent=1)
    print("saved fc_gate_multiseed.json")
