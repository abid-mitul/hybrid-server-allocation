"""Three-component hybrid: stochastic model + PPO + autoscaler.

Normal operation: the self-calibrating queueing model decides, and a CUSUM
test on its own predictions checks whether it still holds (cusum_test.py).
After an alarm the PPO ensemble decides, but never below a safety floor:

  variant "A"  floor = autoscaler (model-free, reacts to measured utilisation)
               c_k = max(c_PPO, c_HPA)
  variant "B"  floor = autoscaler and model
               c_k = max(c_PPO, c_HPA, c_model)

The autoscaler (autoscaler.py, HPA rule with its tuned target) is updated in
every window, so its 5-minute scale-down memory is current when an alarm
fires.  No retraining.

    python hybrid3.py --network stable            # w_C = 0.05, repeat per network
    python hybrid3.py --server-cost-x2            # w_C = 0.10, stable network
Writes data/hybrid3/eval_h3_<tag>.csv.
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
from autoscaler import HPAPolicy


class Hybrid3Policy(CusumHybridPolicy):
    def __init__(self, *a, hpa_target, use_model_floor, **kw):
        super().__init__(*a, **kw)
        self.hpa, self.use_model_floor = HPAPolicy(hpa_target), use_model_floor

    def reset(self):
        super().reset()
        self.hpa.reset()

    def predict(self, obs, deterministic=True):
        a_hpa, _ = self.hpa.predict(obs)                 # keep its history current
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
            a = max(int(self.ppo.predict(obs)[0]), int(a_hpa))       # actions ordered by c
            if self.use_model_floor:
                a = max(a, int(a_model))
            return a, None
        return a_model, None


ALL = lambda n: {**scenarios_for(n), **extra_scenarios_for(n),
                 **structural_scenarios_for(n), **structural2_scenarios_for(n)}


def evaluate(cfg_file, models, thr_file, hpa_target, net, tag):
    cfg = json.load(open(cfg_file)); ev, tr = cfg["eval"], cfg["train"]; opt = env_options(tr)
    t = json.load(open(thr_file)); t = t.get(net, t)
    ens = EnsemblePolicy([PPOPolicy(Path(models) / net / f"ppo_seed_{s}.zip", "ppo", s)
                          for s in tr["seeds"]])
    kw = dict(w_c=opt["w_c"], w_switch=opt["w_switch"], deg_cong=opt["degraded_congestion"],
              threshold=t["tau"], h=t["h"], hpa_target=hpa_target)
    pols = [Hybrid3Policy(net, ens, name="hybrid3_A", use_model_floor=False, **kw),
            Hybrid3Policy(net, ens, name="hybrid3_B", use_model_floor=True, **kw)]
    rows = []
    for sname, scfg in ALL(net).items():
        k = env_kwargs_for(scfg, tr, ev["episode_duration"]); k["measured_util"] = True
        env = make_fast_env(**k); t0 = time.time()
        for pol in pols:
            for ep in range(ev["episodes"]):
                seed = ev["seed_base"] + ep
                res, _ = run_episode(env, pol, seed)
                rows.append(dict(scenario=sname, group=scfg["group"], network=net,
                                 policy=pol.name, train_seed="", episode=ep, eval_seed=seed,
                                 ppo_fraction=pol.ppo_steps / env.n_steps_per_episode,
                                 handovers=pol.n_handover, **res))
        print(f"    {sname}: {time.time() - t0:.0f}s", flush=True)
    out = Path("data") / "hybrid3"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"eval_h3_{tag}.csv", index=False)
    print("done", tag, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--network"); ap.add_argument("--server-cost-x2", action="store_true")
    a = ap.parse_args()
    if a.network:
        tgt = json.load(open("autoscaler_targets.json"))[a.network]["target"]
        evaluate("experiment_config.json", "models", "cusum_thresholds.json", tgt, a.network, a.network)
    if a.server_cost_x2:
        tgt = json.load(open("autoscaler_target_server_cost_x2.json"))["target"]
        evaluate("experiment_config_server_cost_x2.json", "models_server_cost_x2",
                 "server_cost_x2_thresholds.json", tgt, "stable", "wc010")
