"""Model-shielded learned fallback ("safety floor").

Same controller as model + CUSUM + PPO-ensemble fallback (cusum_test.py), with
one change: while the fallback is active, the number of servers is the larger
of the ensemble's choice and the self-calibrating model's choice,
    c_k = max(c_PPO, c_model).
The stochastic model thus acts as a shield on the learned policy: PPO may add
servers beyond the model's recommendation but never go below it.  This targets
the failure seen at doubled server cost, where the ensemble under-provisions
under bursty traffic.  No retraining.

    python safety_floor.py --network stable            # w_C = 0.05, repeat per network
    python safety_floor.py --server-cost-x2            # w_C = 0.10, stable network

Writes data/safety_floor/eval_floor_<net>.csv and eval_floor_wc010.csv.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from fast_env import make_fast_env, env_options
from fast_evaluate import (scenarios_for, extra_scenarios_for, structural_scenarios_for,
                           structural2_scenarios_for, env_kwargs_for, run_episode, PPOPolicy)
from variants import EnsemblePolicy
from cusum_test import CusumHybridPolicy


class CusumFloorPolicy(CusumHybridPolicy):
    def predict(self, obs, deterministic=True):
        r = self.ratio(obs)
        self.S = max(0.0, self.S + np.log(max(r, 1e-9)) - self.k_ref)
        self.run_lo = 0 if r > self.threshold else self.run_lo + 1
        if not self.use_ppo and self.S > self.h:
            self.use_ppo = True
            self.n_handover += 1
            if self.first_alarm is None:
                self.first_alarm = self.k
        elif self.use_ppo and self.run_lo >= self.n_off:
            self.use_ppo, self.S = False, 0.0
        self.k += 1
        a_model, _ = self.model.predict(obs)
        if self.use_ppo:
            self.ppo_steps += 1
            return max(int(self.ppo.predict(obs)[0]), int(a_model)), None   # actions ordered by c
        return a_model, None


def evaluate(cfg_file, models, thr_file, net, tag, scen_fn):
    cfg = json.load(open(cfg_file)); ev, tr = cfg["eval"], cfg["train"]; opt = env_options(tr)
    t = json.load(open(thr_file)); t = t.get(net, t)
    ens = EnsemblePolicy([PPOPolicy(Path(models) / net / f"ppo_seed_{s}.zip", "ppo", s)
                          for s in tr["seeds"]])
    pol = CusumFloorPolicy(net, ens, w_c=opt["w_c"], w_switch=opt["w_switch"],
                           deg_cong=opt["degraded_congestion"], threshold=t["tau"], h=t["h"],
                           name="cusum_floor_ppo")
    rows = []
    for sname, scfg in scen_fn(net).items():
        k = env_kwargs_for(scfg, tr, ev["episode_duration"]); k["measured_util"] = True
        env = make_fast_env(**k); t0 = time.time()
        for ep in range(ev["episodes"]):
            seed = ev["seed_base"] + ep
            res, _ = run_episode(env, pol, seed)
            rows.append(dict(scenario=sname, group=scfg["group"], network=net, policy=pol.name,
                             train_seed="", episode=ep, eval_seed=seed,
                             ppo_fraction=pol.ppo_steps / env.n_steps_per_episode,
                             handovers=pol.n_handover, **res))
        print(f"    {sname}: {time.time() - t0:.0f}s", flush=True)
    out = Path("data") / "safety_floor"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"eval_floor_{tag}.csv", index=False)
    print("done", tag, flush=True)


ALL = lambda n: {**scenarios_for(n), **extra_scenarios_for(n),
                 **structural_scenarios_for(n), **structural2_scenarios_for(n)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--network"); ap.add_argument("--server-cost-x2", action="store_true")
    a = ap.parse_args()
    if a.network:
        evaluate("experiment_config.json", "models", "cusum_thresholds.json", a.network, a.network, ALL)
    if a.server_cost_x2:
        evaluate("experiment_config_server_cost_x2.json", "models_server_cost_x2",
                 "server_cost_x2_thresholds.json", "stable", "wc010", ALL)
