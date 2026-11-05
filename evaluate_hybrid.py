"""Evaluate the hybrid controller with the fixed-parameter model on the seen,
unseen and parameter-mismatch scenarios (13 per network).

Uses the same evaluation seeds as fast_evaluate.py, so the results are paired
episode by episode with data/eval_<net>.csv and data/eval_extra_<net>.csv.
One hybrid per trained PPO agent (5 per network). Writes data/eval_hybrid_<net>.csv.
"""
import argparse, json, time
from pathlib import Path
import pandas as pd
from fast_env import make_fast_env, env_options
from fast_evaluate import scenarios_for, extra_scenarios_for, env_kwargs_for, run_episode, PPOPolicy
from hybrid_policy import HybridPolicy, load_thresholds


def evaluate(network, cfg, root="."):
    root = Path(root); ev, tr = cfg["eval"], cfg["train"]; opt = env_options(tr)
    thr = load_thresholds()[network]
    scen = {**scenarios_for(network), **extra_scenarios_for(network)}
    pols = []
    for s in tr["seeds"]:
        p = root / "models" / network / f"ppo_seed_{s}.zip"
        pols.append(HybridPolicy(network, PPOPolicy(p, "ppo", s), w_c=opt["w_c"],
                                 w_switch=opt["w_switch"], deg_cong=opt["degraded_congestion"],
                                 threshold=thr, train_seed=s))
    rows = []
    for sname, scfg in scen.items():
        env = make_fast_env(**env_kwargs_for(scfg, tr, ev["episode_duration"]))
        t0 = time.time()
        for pol in pols:
            for ep in range(ev["episodes"]):
                seed = ev["seed_base"] + ep
                res, _ = run_episode(env, pol, seed)
                rows.append(dict(scenario=sname, group=scfg["group"], network=network,
                                 policy="hybrid", train_seed=pol.train_seed, episode=ep,
                                 eval_seed=seed, handovers=pol.n_handover,
                                 ppo_fraction=pol.ppo_steps / env.n_steps_per_episode, **res))
        print(f"    {sname}: {time.time() - t0:.0f}s", flush=True)
    (root / "data").mkdir(exist_ok=True)
    out = root / "data" / f"eval_hybrid_{network}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--network", required=True)
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    cfg = json.load(open(Path(a.root) / "experiment_config.json"))
    print(evaluate(a.network, cfg, a.root))
