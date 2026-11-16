"""Network regime switching during operation.

In every other experiment the network regime is fixed for the whole episode.
Here it changes halfway (after 15 of 30 decision windows):
    stable -> degraded,  stable -> noisy,  degraded -> stable
at low, medium and high load (N = 80, 200, 350), 20 episodes each.

Controllers (no retraining, no recalibration):
  fixed c = 3..6, autoscaler (HPA), self-calibrating model,
  model + CUSUM + c_max (told the initial regime),
  Hybrid B (told the initial regime; model + CUSUM + PPO with autoscaler and
            model floor; PPO ensemble of the initial regime),
  regime-agnostic Hybrid B (estimates the regime online, stable-trained PPO).

    python regime_switch.py
Writes data/regime_switch/eval_switch.csv.
"""
import json, time
from pathlib import Path
import pandas as pd
from fast_env import make_fast_env, env_options, C_MAX, ACTION_TO_C, NETWORK_CODE
from fast_evaluate import run_episode, FixedCPolicy, PPOPolicy
from variants import EnsemblePolicy, AdaptiveModelPolicy
from autoscaler import HPAPolicy
from cusum_test import CusumHybridPolicy
from hybrid3 import Hybrid3Policy
from final_checks import RAHybrid3

SWITCHES = [("stable", "degraded"), ("stable", "noisy"), ("degraded", "stable")]
LOADS = [80, 200, 350]
SWITCH_STEP = 15


class SwitchEnv:
    """Changes the simulator's network regime before decision window SWITCH_STEP."""

    def __init__(self, env, start, end):
        self._env, self._start, self._end = env, start, end

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, **kw):
        self._env.mode = NETWORK_CODE[self._start]; self._env.network_mode = self._start
        self._k = 0
        return self._env.reset(**kw)

    def step(self, a):
        if self._k == SWITCH_STEP:
            self._env.mode = NETWORK_CODE[self._end]; self._env.network_mode = self._end
        self._k += 1
        return self._env.step(a)


def main():
    cfg = json.load(open("experiment_config.json")); ev, tr = cfg["eval"], cfg["train"]
    opt = env_options(tr)
    kw = dict(w_c=opt["w_c"], w_switch=opt["w_switch"], deg_cong=opt["degraded_congestion"])
    cus = json.load(open("cusum_thresholds.json")); ra = json.load(open("regime_agnostic_cusum.json"))
    tg = json.load(open("autoscaler_targets.json"))
    ens = {n: EnsemblePolicy([PPOPolicy(Path("models") / n / f"ppo_seed_{s}.zip", "ppo", s)
                              for s in tr["seeds"]]) for n in ("stable", "degraded")}
    rows = []
    for start, end in SWITCHES:
        c = cus[start]
        pols = [FixedCPolicy(x) for x in ACTION_TO_C] + [
            HPAPolicy(tg[start]["target"]),
            AdaptiveModelPolicy(start, **kw),
            CusumHybridPolicy(start, FixedCPolicy(C_MAX), threshold=c["tau"], h=c["h"],
                              name="cusum_fallback_c6", **kw),
            Hybrid3Policy(start, ens[start], threshold=c["tau"], h=c["h"],
                          hpa_target=tg[start]["target"], use_model_floor=True,
                          name="hybrid3_B", **kw),
            RAHybrid3(ens["stable"], "ra_hybrid3_B", threshold=ra["tau"], h=ra["h"],
                      hpa_target=tg["stable"]["target"], **kw)]
        for N in LOADS:
            env = SwitchEnv(make_fast_env(n_machines=N, network_mode=start,
                                          episode_duration=ev["episode_duration"],
                                          step_duration=tr["step_duration"],
                                          measured_util=True, **opt), start, end)
            t0 = time.time()
            for pol in pols:
                for ep in range(ev["episodes"]):
                    seed = ev["seed_base"] + ep
                    res, _ = run_episode(env, pol, seed)
                    rows.append(dict(scenario=f"{start}_to_{end}_N{N}", switch=f"{start}->{end}",
                                     n_machines=N, policy=pol.name, eval_seed=seed,
                                     fallback_share=(pol.ppo_steps / 30 if hasattr(pol, "ppo_steps") else None),
                                     **res))
            print(f"    {start}->{end} N={N}: {time.time() - t0:.0f}s", flush=True)
    out = Path("data") / "regime_switch"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "eval_switch.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
