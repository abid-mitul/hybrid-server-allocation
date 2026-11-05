"""
Validation of fast_env.py against (a) the independent SimPy implementation in
factory_env.py and (b) the closed-form M/M/c/K results, at a fixed server
count. Output: data/fast_env_validation.csv

Usage:
    python validate_fast_env.py            # full check (~10-20 min)
    python validate_fast_env.py --quick    # ~2 min
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd

from parameters import SERVER, SLA_AND_WEIGHTS
from factory_env import FactoryEnvironment
from fast_env import make_fast_env

T_SLA = SLA_AND_WEIGHTS["T_SLA"]
ACTION = {3: 0, 4: 1, 5: 2, 6: 3}


def mmck(lam, mu, c, K):
    """Closed-form M/M/c/K: blocking probability and mean wait in queue."""
    a = lam / mu
    p = np.zeros(K + 1)
    for n in range(K + 1):
        if n <= c:
            p[n] = a ** n / math.factorial(n)
        else:
            p[n] = a ** c / math.factorial(c) * (a / c) ** (n - c)
    p /= p.sum()
    p_b = p[K]
    lq = sum((n - c) * p[n] for n in range(c, K + 1))
    lam_eff = lam * (1 - p_b)
    return p_b, lq / lam_eff


def run_original(n, c, network, T, seed):
    f = FactoryEnvironment(n_machines=n, c=c, network_mode=network,
                           run_time=T, warmup_time=0.0, seed=seed)
    f.run()
    m = f.metrics
    offered = len(m.task_arrivals)          # this list already includes drops
    drops = len(m.task_drops)
    comp = m.task_completions
    soj = np.array([x[2] - x[0] for x in comp])
    wait = np.array([x[1] - x[0] for x in comp])
    return dict(lam=offered / T, P_B=drops / offered, W_q_ms=1000 * wait.mean(),
                sojourn_ms=1000 * soj.mean(), sla=(soj <= T_SLA).mean(),
                throughput=len(comp) / T)


def run_fast(n, c, network, T, seed):
    env = make_fast_env(n_machines=n, network_mode=network, episode_duration=T,
                        step_duration=60.0, seed=seed)
    env.reset(seed=seed)
    tot = dict(off=0, drop=0, comp=0, sla=0, soj=0.0, wait=0.0)
    done = False
    while not done:
        _, _, done, _, info = env.step(ACTION[c])
        nc = info["n_completions"]
        tot["off"] += info["n_arrivals"]
        tot["drop"] += info["n_drops"]
        tot["comp"] += nc
        tot["sla"] += info["sla_compliance"] * nc
        tot["soj"] += info["mean_sojourn"] * nc
        tot["wait"] += info["mean_wait"] * nc
    return dict(lam=tot["off"] / T, P_B=tot["drop"] / tot["off"],
                W_q_ms=1000 * tot["wait"] / tot["comp"],
                sojourn_ms=1000 * tot["soj"] / tot["comp"],
                sla=tot["sla"] / tot["comp"], throughput=tot["comp"] / T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    T = 600.0 if args.quick else 1800.0
    seeds = [1, 2] if args.quick else [1, 2, 3]
    cases = [(200, 3, "stable"), (200, 4, "stable"), (200, 6, "stable"),
             (350, 4, "stable"), (200, 4, "degraded"), (200, 4, "noisy")]
    if args.quick:
        cases = [(200, 4, "stable"), (350, 4, "stable")]

    rows = []
    for n, c, net in cases:
        for sim, fn in (("original_simpy", run_original), ("fast", run_fast)):
            t0 = time.time()
            res = [fn(n, c, net, T, s) for s in seeds]
            wall = time.time() - t0
            avg = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
            sd = {k + "_sd": float(np.std([r[k] for r in res], ddof=1)) for k in res[0]}
            pb_th, wq_th = mmck(avg["lam"], SERVER["mu"], c, SERVER["K"])
            row = dict(N=n, c=c, network=net, simulator=sim, seeds=len(seeds),
                       sim_seconds=T, wall_seconds=round(wall, 1), **avg, **sd,
                       P_B_theory=pb_th, W_q_theory_ms=1000 * wq_th)
            rows.append(row)
            print(f"N={n} c={c} {net:8s} {sim:15s} lam={avg['lam']:7.1f} "
                  f"P_B={avg['P_B']:.5f} (th {pb_th:.5f})  "
                  f"W_q={avg['W_q_ms']:.3f}ms (th {1000*wq_th:.3f})  "
                  f"soj={avg['sojourn_ms']:.1f}ms  SLA={avg['sla']:.4f}  "
                  f"wall={wall:.1f}s", flush=True)
    os.makedirs("data", exist_ok=True)
    pd.DataFrame(rows).to_csv("data/fast_env_validation.csv", index=False)
    print("Saved data/fast_env_validation.csv")


if __name__ == "__main__":
    main()
