"""Structural-mismatch evaluation.

  part 1: all controllers on the 6 structural scenarios of the network
          -> data/eval_structural_<net>.csv
  part 2: the self-calibrating hybrid on the 13 earlier scenarios of the network
          -> data/eval_hybadapt_<net>.csv
Same evaluation seeds as fast_evaluate.py, so all results are paired.
"""
import argparse, json, time
from pathlib import Path
import pandas as pd
from fast_env import make_fast_env, env_options, ACTION_TO_C
from fast_evaluate import (scenarios_for, extra_scenarios_for, structural_scenarios_for,
                           structural2_scenarios_for,
                           env_kwargs_for, run_episode, PPOPolicy, FixedCPolicy, fit_ga)
from baselines import RoundRobinPolicy, FixedPriorityPolicy
from model_policy import QueueingModelPolicy
from hybrid_policy import HybridPolicy, load_thresholds
from variants import EnsemblePolicy, AdaptiveModelPolicy, HybridAdaptivePolicy


def run(net, scen, pols, ev, tr, out):
    rows = []
    for sname, scfg in scen.items():
        env = make_fast_env(**env_kwargs_for(scfg, tr, ev["episode_duration"])); t0 = time.time()
        for pol in pols:
            for ep in range(ev["episodes"]):
                seed = ev["seed_base"] + ep
                res, _ = run_episode(env, pol, seed)
                extra = ({"ppo_fraction": pol.ppo_steps / env.n_steps_per_episode,
                          "handovers": pol.n_handover} if hasattr(pol, "ppo_steps") else {})
                rows.append(dict(scenario=sname, group=scfg["group"], network=net,
                                 n_machines=scfg.get("N", scfg.get("N_start")), policy=pol.name,
                                 train_seed=getattr(pol, "train_seed", ""), episode=ep,
                                 eval_seed=seed, **extra, **res))
        print(f"    {sname}: {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--network", required=True)
    ap.add_argument("--set2", action="store_true", help="second structural set only")
    args = ap.parse_args(); net = args.network
    cfg = json.load(open("experiment_config.json")); ev, tr = cfg["eval"], cfg["train"]
    opt = env_options(tr); thr = load_thresholds()
    kw = dict(w_c=opt["w_c"], w_switch=opt["w_switch"], deg_cong=opt["degraded_congestion"])
    ppos = [PPOPolicy(Path("models") / net / f"ppo_seed_{s}.zip", "ppo", s) for s in tr["seeds"]]
    ens = EnsemblePolicy(ppos)
    hyb_ens = HybridPolicy(net, EnsemblePolicy(ppos), threshold=thr[net], **kw)
    hyb_ens.name = "hybrid_ensemble"
    hyb_adp = HybridAdaptivePolicy(net, EnsemblePolicy(ppos), threshold=thr["adaptive_" + net], **kw)

    t0 = time.time(); ga = fit_ga(net, cfg); print(f"    GA fitted in {time.time() - t0:.0f}s", flush=True)
    pols = [FixedCPolicy(c) for c in ACTION_TO_C] + [RoundRobinPolicy(), FixedPriorityPolicy(),
            ga, QueueingModelPolicy(net, **kw), AdaptiveModelPolicy(net, **kw)] + ppos + [ens]
    pols += [HybridPolicy(net, p, threshold=thr[net], train_seed=p.train_seed, **kw) for p in ppos]
    pols += [hyb_ens, hyb_adp]
    Path("data").mkdir(exist_ok=True)
    if args.set2:
        run(net, structural2_scenarios_for(net), pols, ev, tr,
            Path("data") / f"eval_structural2_{net}.csv")
    else:
        run(net, structural_scenarios_for(net), pols, ev, tr, Path("data") / f"eval_structural_{net}.csv")
        run(net, {**scenarios_for(net), **extra_scenarios_for(net)}, [hyb_adp], ev, tr,
            Path("data") / f"eval_hybadapt_{net}.csv")
    print("done", flush=True)
