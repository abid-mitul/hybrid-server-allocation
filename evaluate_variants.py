"""Evaluate ensemble, hybrid-with-ensemble and adaptive model on all scenarios.
Same evaluation seeds as fast_evaluate.py (paired). Writes data/eval_variants_<net>.csv."""
import argparse, json, time
from pathlib import Path
import pandas as pd
from fast_env import make_fast_env, env_options
from fast_evaluate import scenarios_for, extra_scenarios_for, env_kwargs_for, run_episode, PPOPolicy
from hybrid_policy import HybridPolicy, load_thresholds
from variants import EnsemblePolicy, AdaptiveModelPolicy

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--network", required=True)
    a = ap.parse_args(); net = a.network
    cfg = json.load(open("experiment_config.json")); ev, tr = cfg["eval"], cfg["train"]
    opt = env_options(tr)
    members = [PPOPolicy(Path("models") / net / f"ppo_seed_{s}.zip", "ppo", s) for s in tr["seeds"]]
    ens = EnsemblePolicy(members)
    hyb = HybridPolicy(net, EnsemblePolicy(members), w_c=opt["w_c"], w_switch=opt["w_switch"],
                       deg_cong=opt["degraded_congestion"], threshold=load_thresholds()[net])
    hyb.name = "hybrid_ensemble"
    adm = AdaptiveModelPolicy(net, w_c=opt["w_c"], w_switch=opt["w_switch"],
                              deg_cong=opt["degraded_congestion"])
    rows = []
    for sname, scfg in {**scenarios_for(net), **extra_scenarios_for(net)}.items():
        env = make_fast_env(**env_kwargs_for(scfg, tr, ev["episode_duration"])); t0 = time.time()
        for pol in (ens, hyb, adm):
            for ep in range(ev["episodes"]):
                seed = ev["seed_base"] + ep
                res, _ = run_episode(env, pol, seed)
                extra = dict(ppo_fraction=pol.ppo_steps / env.n_steps_per_episode,
                             handovers=pol.n_handover) if pol is hyb else {}
                rows.append(dict(scenario=sname, group=scfg["group"], network=net,
                                 policy=pol.name, train_seed="", episode=ep, eval_seed=seed,
                                 **extra, **res))
        print(f"    {sname}: {time.time() - t0:.0f}s", flush=True)
    Path("data").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(Path("data") / f"eval_variants_{net}.csv", index=False)
    print("done")
