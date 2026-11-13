"""Revision item 2: "model + c_max fallback" baseline.

Same controller as the proposed hybrid (self-calibrating model by default,
the same mismatch test with the same calibrated threshold, n_on = 2 and
n_off = 3), but when the test fires it falls back to c_max = 6 servers instead
of handing the decision to the PPO ensemble.  This isolates what the learned
mode adds beyond the alarm itself (a simplex-style safe fallback).

Evaluated on all 69 scenarios with measured utilisation, using the same
evaluation seeds as every other controller (paired with the published and the
revised results).

    python evaluate_fallback.py --network stable      # repeat for degraded, noisy

Writes data/fallback/eval_fallback_<net>.csv.
"""
import argparse, json, time
from pathlib import Path
import pandas as pd
from fast_env import make_fast_env, env_options, C_MAX
from fast_evaluate import (scenarios_for, extra_scenarios_for, structural_scenarios_for,
                           structural2_scenarios_for, env_kwargs_for, run_episode,
                           FixedCPolicy)
from hybrid_policy import load_thresholds
from variants import HybridAdaptivePolicy


def make_fallback(net, opt, thr):
    pol = HybridAdaptivePolicy(net, FixedCPolicy(C_MAX), w_c=opt["w_c"],
                               w_switch=opt["w_switch"], deg_cong=opt["degraded_congestion"],
                               threshold=thr["adaptive_" + net])
    pol.name = "model_fallback_c6"
    return pol


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--network", required=True)
    net = ap.parse_args().network
    cfg = json.load(open("experiment_config.json")); ev, tr = cfg["eval"], cfg["train"]
    pol = make_fallback(net, env_options(tr), load_thresholds())
    scen = {**scenarios_for(net), **extra_scenarios_for(net),
            **structural_scenarios_for(net), **structural2_scenarios_for(net)}
    rows = []
    for sname, scfg in scen.items():
        kw = env_kwargs_for(scfg, tr, ev["episode_duration"])
        kw["measured_util"] = True
        env = make_fast_env(**kw); t0 = time.time()
        for ep in range(ev["episodes"]):
            seed = ev["seed_base"] + ep
            res, _ = run_episode(env, pol, seed)
            rows.append(dict(scenario=sname, group=scfg["group"], network=net,
                             n_machines=kw["n_machines"],
                             n_machines_end=kw.get("burst_n_end", kw["n_machines"]),
                             policy=pol.name, train_seed="", episode=ep, eval_seed=seed,
                             ppo_fraction=pol.ppo_steps / env.n_steps_per_episode,
                             handovers=pol.n_handover, **res))
        print(f"    {sname}: {time.time() - t0:.0f}s", flush=True)
    out = Path("data") / "fallback"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"eval_fallback_{net}.csv", index=False)
    print("done", flush=True)
