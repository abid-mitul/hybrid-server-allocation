"""When does the learned fallback pay?  Doubled server cost (w_C = 0.10).

At the default cost (w_C = 0.05) a fixed fallback to c_max = 6 servers is as
good as the PPO ensemble.  This study repeats the comparison when servers cost
twice as much, so that the conservative fallback is expensive.  Stable network,
all 23 scenarios (8 seen, 3 unseen, 6 parameter mismatch... as defined in
fast_evaluate), measured utilisation, common seeds.

Everything is recalibrated for w_C = 0.10 with the nominal-only rules:
  tau : 99th percentile of the measured/predicted sojourn ratio (training loads,
        seeds 5000-5009), as calibrate_hybrid.py --adaptive
  h   : CUSUM level with the same nominal false-alarm count as the persistence
        test, as cusum_test.py
The PPO agents are the ones trained at w_C = 0.10 (models_server_cost_x2).

    python server_cost_study.py --calibrate
    python server_cost_study.py --evaluate
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from fast_env import make_fast_env, env_options, C_MAX, ACTION_TO_C
from fast_evaluate import (scenarios_for, extra_scenarios_for, structural_scenarios_for,
                           structural2_scenarios_for, env_kwargs_for, run_episode,
                           FixedCPolicy, PPOPolicy)
from variants import HybridAdaptivePolicy, EnsemblePolicy, AdaptiveModelPolicy
from cusum_test import CusumHybridPolicy

CFG, MODELS, NET = "experiment_config_server_cost_x2.json", "models_server_cost_x2", "stable"
FILE = "server_cost_x2_thresholds.json"


def calibrate(cfg):
    tr = cfg["train"]; opt = env_options(tr)
    probe = HybridAdaptivePolicy(NET, None, opt["w_c"], opt["w_switch"],
                                 opt["degraded_congestion"], threshold=1e9)
    traces = []
    for N in tr["n_machines"]:
        env = make_fast_env(n_machines=N, network_mode=NET, episode_duration=1800,
                            step_duration=tr["step_duration"], measured_util=True, **opt)
        for s in range(5000, 5010):
            obs, _ = env.reset(seed=s); probe.reset(); done = False; x = []
            while not done:
                x.append(probe.ratio(obs))
                a, _ = probe.predict(obs)
                obs, _, done, _, _ = env.step(a)
            traces.append(np.array(x))
    tau = round(float(np.quantile(np.concatenate(traces), 0.99)), 4)
    # persistence alarms on the same runs (closed loop with the c = 6 fallback)
    p = HybridAdaptivePolicy(NET, FixedCPolicy(C_MAX), opt["w_c"], opt["w_switch"],
                             opt["degraded_congestion"], threshold=tau)
    logx, n_persist = [], 0
    for N in tr["n_machines"]:
        env = make_fast_env(n_machines=N, network_mode=NET, episode_duration=1800,
                            step_duration=tr["step_duration"], measured_util=True, **opt)
        for s in range(5000, 5010):
            obs, _ = env.reset(seed=s); p.reset(); done = False; x = []
            while not done:
                x.append(np.log(max(p.ratio(obs), 1e-9)))
                a, _ = p.predict(obs)
                obs, _, done, _, _ = env.step(a)
            logx.append(np.array(x)); n_persist += p.n_handover
    k_ref = 0.5 * np.log(tau)

    def n_alarms(h):
        n = 0
        for x in logx:
            S = 0.0
            for v in x:
                S = max(0.0, S + v - k_ref)
                if S > h:
                    n, S = n + 1, 0.0
        return n
    h = next(float(g) for g in np.round(np.arange(0.0, 3.0, 0.005), 3) if n_alarms(g) <= n_persist)
    out = {"tau": tau, "k_ref": round(float(k_ref), 4), "h": h,
           "nominal_alarms_persistence": int(n_persist), "nominal_alarms_cusum": int(n_alarms(h))}
    json.dump(out, open(FILE, "w"), indent=2)
    print(out, flush=True)


def evaluate(cfg):
    ev, tr = cfg["eval"], cfg["train"]; opt = env_options(tr)
    c = json.load(open(FILE)); tau, h = c["tau"], c["h"]
    kw = dict(w_c=opt["w_c"], w_switch=opt["w_switch"], deg_cong=opt["degraded_congestion"])
    ppos = [PPOPolicy(Path(MODELS) / NET / f"ppo_seed_{s}.zip", "ppo", s) for s in tr["seeds"]]
    ens = EnsemblePolicy(ppos)
    pols = [FixedCPolicy(cc) for cc in ACTION_TO_C] + [
        AdaptiveModelPolicy(NET, **kw), ens,
        CusumHybridPolicy(NET, FixedCPolicy(C_MAX), threshold=tau, h=h, name="cusum_fallback_c6", **kw),
        CusumHybridPolicy(NET, ens, threshold=tau, h=h, name="cusum_hybrid_ppo", **kw)]
    pers = HybridAdaptivePolicy(NET, FixedCPolicy(C_MAX), threshold=tau, **kw)
    pers.name = "model_fallback_c6"
    pers_ppo = HybridAdaptivePolicy(NET, ens, threshold=tau, **kw)
    pols += [pers, pers_ppo]
    scen = {**scenarios_for(NET), **extra_scenarios_for(NET),
            **structural_scenarios_for(NET), **structural2_scenarios_for(NET)}
    rows = []
    for sname, scfg in scen.items():
        k = env_kwargs_for(scfg, tr, ev["episode_duration"]); k["measured_util"] = True
        env = make_fast_env(**k); t0 = time.time()
        for pol in pols:
            for ep in range(ev["episodes"]):
                seed = ev["seed_base"] + ep
                res, _ = run_episode(env, pol, seed)
                extra = ({"ppo_fraction": pol.ppo_steps / env.n_steps_per_episode,
                          "handovers": pol.n_handover} if hasattr(pol, "ppo_steps") else {})
                rows.append(dict(scenario=sname, group=scfg["group"], network=NET,
                                 policy=pol.name, train_seed="", episode=ep, eval_seed=seed,
                                 **extra, **res))
        print(f"    {sname}: {time.time() - t0:.0f}s", flush=True)
    out = Path("data") / "server_cost_x2"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "eval_stable.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true"); ap.add_argument("--evaluate", action="store_true")
    a = ap.parse_args(); cfg = json.load(open(CFG))
    if a.calibrate:
        calibrate(cfg)
    if a.evaluate:
        evaluate(cfg)
