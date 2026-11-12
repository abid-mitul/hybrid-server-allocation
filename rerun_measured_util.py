"""Revision item 1: remove the utilisation oracle.

Re-evaluates every non-structural scenario (seen, unseen and parameter mismatch;
13 per network, 39 in total) with the utilisation computed from the recorded
processing times (measured_util=True), exactly as the structural scenarios
already do.  Training (PPO agents, GA rule, alarm thresholds) is unchanged.

    python rerun_measured_util.py --network stable      # repeat for degraded, noisy
    python rerun_measured_util.py --network stable --oracle --only slowservers_stable
                                                          # reproduction check

Writes data/measured/eval_<net>.csv with the same columns as the published
result files, so analysis/build_results_measured.py can rebuild the tables.
"""
import argparse, json, time
from pathlib import Path
import pandas as pd
from fast_env import make_fast_env, env_options, ACTION_TO_C
from fast_evaluate import (scenarios_for, extra_scenarios_for, env_kwargs_for,
                           run_episode, PPOPolicy, FixedCPolicy, fit_ga)
from baselines import FixedPriorityPolicy
from model_policy import QueueingModelPolicy
from hybrid_policy import HybridPolicy, load_thresholds
from variants import EnsemblePolicy, AdaptiveModelPolicy, HybridAdaptivePolicy


def build_policies(net, cfg):
    tr = cfg["train"]; opt = env_options(tr); thr = load_thresholds()
    kw = dict(w_c=opt["w_c"], w_switch=opt["w_switch"], deg_cong=opt["degraded_congestion"])
    ppos = [PPOPolicy(Path("models") / net / f"ppo_seed_{s}.zip", "ppo", s) for s in tr["seeds"]]
    t0 = time.time(); ga = fit_ga(net, cfg)
    print(f"    GA fitted in {time.time() - t0:.0f}s, thresholds "
          f"{[round(x, 3) for x in ga.thresholds]}", flush=True)
    hyb_ens = HybridPolicy(net, EnsemblePolicy(ppos), threshold=thr[net], **kw)
    hyb_ens.name = "hybrid_ensemble"
    pols = [FixedCPolicy(c) for c in ACTION_TO_C] + [FixedPriorityPolicy(), ga,
            QueueingModelPolicy(net, **kw), AdaptiveModelPolicy(net, **kw)] + ppos
    pols += [EnsemblePolicy(ppos)]
    pols += [HybridPolicy(net, p, threshold=thr[net], train_seed=p.train_seed, **kw) for p in ppos]
    pols += [hyb_ens,
             HybridAdaptivePolicy(net, EnsemblePolicy(ppos), threshold=thr["adaptive_" + net], **kw)]
    return pols


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True)
    ap.add_argument("--oracle", action="store_true", help="keep the old utilisation (check)")
    ap.add_argument("--only", default=None, help="evaluate a single scenario")
    ap.add_argument("--policies", default=None, help="comma-separated policy names")
    a = ap.parse_args(); net = a.network
    cfg = json.load(open("experiment_config.json")); ev, tr = cfg["eval"], cfg["train"]
    scen = {**scenarios_for(net), **extra_scenarios_for(net)}
    if a.only:
        scen = {a.only: scen[a.only]}
    pols = build_policies(net, cfg)
    if a.policies:
        keep = a.policies.split(",")
        pols = [p for p in pols if p.name in keep]
    rows = []
    for sname, scfg in scen.items():
        kw = env_kwargs_for(scfg, tr, ev["episode_duration"])
        kw["measured_util"] = not a.oracle
        env = make_fast_env(**kw); t0 = time.time()
        for pol in pols:
            for ep in range(ev["episodes"]):
                seed = ev["seed_base"] + ep
                res, _ = run_episode(env, pol, seed)
                extra = ({"ppo_fraction": pol.ppo_steps / env.n_steps_per_episode,
                          "handovers": pol.n_handover} if hasattr(pol, "ppo_steps") else {})
                rows.append(dict(scenario=sname, group=scfg["group"], network=net,
                                 n_machines=kw["n_machines"],
                                 n_machines_end=kw.get("burst_n_end", kw["n_machines"]),
                                 policy=pol.name, train_seed=getattr(pol, "train_seed", ""),
                                 episode=ep, eval_seed=seed, **extra, **res))
        print(f"    {sname}: {len(pols)} policies in {time.time() - t0:.0f}s", flush=True)
    out = Path("data") / ("oracle_check" if a.oracle else "measured")
    out.mkdir(parents=True, exist_ok=True)
    name = f"eval_{net}{'_' + a.only if a.only else ''}.csv"
    pd.DataFrame(rows).to_csv(out / name, index=False)
    print("done", out / name, flush=True)
