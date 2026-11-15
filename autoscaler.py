"""Industry baseline: target-utilisation autoscaler (Kubernetes HPA rule).

Each decision interval the controller computes the recommendation
    c_rec = ceil( c_k * U_hat_k / U_target ),   clamped to [c_min, c_max],
where U_hat_k is the measured utilisation of the window just ended.  As in the
Kubernetes Horizontal Pod Autoscaler:
  * tolerance: no change while |U_hat_k / U_target - 1| <= 0.1;
  * scale-down stabilisation: the applied count is the largest recommendation
    of the last 5 decision intervals (5 min, the HPA default of 300 s), so
    servers are added at once and removed only after the load has stayed low.
The target U_target is tuned on the nominal training conditions only (training
loads, seeds 5000-5009, per network regime), the same protocol as the GA rule:
the candidate targets 0.50, 0.55, ..., 0.90 are compared on the mean episode
reward and the best one is kept.

    python autoscaler.py --tune                     # w_C = 0.05, per network
    python autoscaler.py --network stable           # evaluate, repeat per network
    python autoscaler.py --server-cost-x2           # tune and evaluate at w_C = 0.10 (stable)
"""
import argparse, json, math, time
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
from fast_env import make_fast_env, env_options, C_MIN, C_MAX, ACTION_TO_C
from fast_evaluate import (scenarios_for, extra_scenarios_for, structural_scenarios_for,
                           structural2_scenarios_for, env_kwargs_for, run_episode)

TARGETS = [round(x, 2) for x in np.arange(0.50, 0.901, 0.05)]
FILE = "autoscaler_targets.json"


class HPAPolicy:
    def __init__(self, target, tolerance=0.1, window=5, name="autoscaler_hpa"):
        self.target, self.tol, self.window, self.name = target, tolerance, window, name
        self.train_seed = ""
        self.reset()

    def reset(self):
        self.recs = deque(maxlen=self.window)

    def predict(self, obs, deterministic=True):
        c = int(round(C_MIN + float(obs[5]) * (C_MAX - C_MIN)))
        u = float(obs[2])
        ratio = u / self.target
        rec = c if abs(ratio - 1.0) <= self.tol else math.ceil(c * ratio - 1e-9)
        rec = min(max(rec, C_MIN), C_MAX)
        self.recs.append(rec)
        return ACTION_TO_C.index(max(self.recs)), None


def tune(cfg, nets):
    tr = cfg["train"]; opt = env_options(tr); out = {}
    for net in nets:
        score = {}
        for tgt in TARGETS:
            pol, rew = HPAPolicy(tgt), []
            for N in tr["n_machines"]:
                env = make_fast_env(n_machines=N, network_mode=net, episode_duration=1800,
                                    step_duration=tr["step_duration"], measured_util=True, **opt)
                for s in range(5000, 5010):
                    res, _ = run_episode(env, pol, s)
                    rew.append(res["total_reward"])
            score[tgt] = float(np.mean(rew))
        best = max(score, key=score.get)
        out[net] = {"target": best, "scores": {str(k): round(v, 4) for k, v in score.items()}}
        print(net, "best target", best, flush=True)
    return out


def evaluate(cfg, net, target, tag):
    ev, tr = cfg["eval"], cfg["train"]
    pol = HPAPolicy(target)
    scen = {**scenarios_for(net), **extra_scenarios_for(net),
            **structural_scenarios_for(net), **structural2_scenarios_for(net)}
    rows = []
    for sname, scfg in scen.items():
        k = env_kwargs_for(scfg, tr, ev["episode_duration"]); k["measured_util"] = True
        env = make_fast_env(**k); t0 = time.time()
        for ep in range(ev["episodes"]):
            seed = ev["seed_base"] + ep
            res, _ = run_episode(env, pol, seed)
            rows.append(dict(scenario=sname, group=scfg["group"], network=net, policy=pol.name,
                             train_seed="", episode=ep, eval_seed=seed, target=target, **res))
        print(f"    {sname}: {time.time() - t0:.0f}s", flush=True)
    out = Path("data") / "autoscaler"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"eval_hpa_{tag}.csv", index=False)
    print("done", tag, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true"); ap.add_argument("--network")
    ap.add_argument("--server-cost-x2", action="store_true")
    a = ap.parse_args()
    if a.tune:
        cfg = json.load(open("experiment_config.json"))
        json.dump(tune(cfg, cfg["train"]["networks"]), open(FILE, "w"), indent=2)
    if a.network:
        cfg = json.load(open("experiment_config.json"))
        evaluate(cfg, a.network, json.load(open(FILE))[a.network]["target"], a.network)
    if a.server_cost_x2:
        cfg = json.load(open("experiment_config_server_cost_x2.json"))
        t = tune(cfg, ["stable"])["stable"]
        json.dump(t, open("autoscaler_target_server_cost_x2.json", "w"), indent=2)
        evaluate(cfg, "stable", t["target"], "wc010")
