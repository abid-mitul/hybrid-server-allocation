"""Final checks of the proposed three-component hybrid (Hybrid B).

1. --ra : regime-agnostic Hybrid B (normal cost, all 69 scenarios).
   The network regime is estimated online (regime_agnostic.py); the mismatch
   ratio uses the regime estimated before each window; the threshold is the
   pooled regime-agnostic one (1.4064); the CUSUM level h is calibrated on the
   same pooled nominal runs to the persistence test's false-alarm count; the
   PPO ensemble is the stable-trained one; the autoscaler target is the tuned
   0.55 (the same for every regime).

2. --x2 NET : doubled server cost (w_C = 0.10) on the degraded or noisy network.
   Agents trained at w_C = 0.10 exist for the stable network only and are used
   unchanged (transfer, as in item 4).  tau and h are recalibrated for NET at
   w_C = 0.10 with the nominal-only rules, and the autoscaler target is tuned
   for NET at w_C = 0.10.  Controllers: fixed c = 3..6, autoscaler, PPO
   ensemble, model + CUSUM + c_max, model + CUSUM + PPO (model floor) and
   Hybrid B, on all 23 scenarios of NET.

    python final_checks.py --ra-calibrate
    python final_checks.py --ra --network stable        # repeat per network
    python final_checks.py --x2 degraded                # and noisy
Writes data/final_checks/*.csv and the calibration JSON files.
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from fast_env import make_fast_env, env_options, C_MAX, ACTION_TO_C
from fast_evaluate import (scenarios_for, extra_scenarios_for, structural_scenarios_for,
                           structural2_scenarios_for, env_kwargs_for, run_episode,
                           FixedCPolicy, PPOPolicy)
from variants import HybridAdaptivePolicy, EnsemblePolicy
from regime_agnostic import RegimeAgnosticHybrid, MODES, THRESH_FILE
from cusum_test import CusumHybridPolicy
from autoscaler import HPAPolicy, tune as tune_hpa
from hybrid3 import Hybrid3Policy
from safety_floor import CusumFloorPolicy

OUT = Path("data") / "final_checks"
ALL = lambda n: {**scenarios_for(n), **extra_scenarios_for(n),
                 **structural_scenarios_for(n), **structural2_scenarios_for(n)}


def h_for_equal_false_alarms(logx, n_persist, k_ref):
    def n_alarms(h):
        n = 0
        for x in logx:
            S = 0.0
            for v in x:
                S = max(0.0, S + v - k_ref)
                if S > h:
                    n, S = n + 1, 0.0
        return n
    return next(float(g) for g in np.round(np.arange(0.0, 3.0, 0.005), 3) if n_alarms(g) <= n_persist)


def run_rows(pols, net, cfg):
    ev, tr = cfg["eval"], cfg["train"]; rows = []
    for sname, scfg in ALL(net).items():
        k = env_kwargs_for(scfg, tr, ev["episode_duration"]); k["measured_util"] = True
        env = make_fast_env(**k); t0 = time.time()
        for pol in pols:
            for ep in range(ev["episodes"]):
                seed = ev["seed_base"] + ep
                res, _ = run_episode(env, pol, seed)
                extra = ({"ppo_fraction": pol.ppo_steps / env.n_steps_per_episode,
                          "handovers": pol.n_handover} if hasattr(pol, "ppo_steps") else {})
                rows.append(dict(scenario=sname, group=scfg["group"], network=net, policy=pol.name,
                                 train_seed="", episode=ep, eval_seed=seed, **extra, **res))
        print(f"    {sname}: {time.time() - t0:.0f}s", flush=True)
    return rows


# ---------------------------------------------------------------- 1. regime-agnostic
class RAHybrid3(RegimeAgnosticHybrid):
    """Regime-agnostic model + CUSUM + PPO with autoscaler and model floor."""

    def __init__(self, fallback, name, w_c, w_switch, deg_cong, threshold, h, hpa_target):
        self.hpa, self.h, self.k_ref = HPAPolicy(hpa_target), h, 0.5 * math.log(threshold)
        super().__init__(fallback, name, w_c, w_switch, deg_cong, threshold)

    def reset(self):
        super().reset()
        self.hpa.reset(); self.S = 0.0

    def predict(self, obs, deterministic=True):
        a_hpa, _ = self.hpa.predict(obs)
        r = self.ratio(obs, self.model.mode)               # regime estimated before this window
        self.model.observe(obs)
        self.S = max(0.0, self.S + math.log(max(r, 1e-9)) - self.k_ref)
        self.run_lo = 0 if r > self.threshold else self.run_lo + 1
        if not self.use_fb and self.S > self.h:
            self.use_fb = True; self.n_handover += 1
        elif self.use_fb and self.run_lo >= self.n_off:
            self.use_fb, self.S = False, 0.0
        a_model, _ = self.model.predict(obs, update=False)
        if self.use_fb:
            self.ppo_steps += 1
            return max(int(self.fallback.predict(obs)[0]), int(a_hpa), int(a_model)), None
        return a_model, None


def ra_calibrate(cfg):
    tr = cfg["train"]; opt = env_options(tr); tau = json.load(open(THRESH_FILE))["threshold"]
    p = RegimeAgnosticHybrid(FixedCPolicy(C_MAX), "cal", opt["w_c"], opt["w_switch"],
                             opt["degraded_congestion"], threshold=tau)
    logx, n_persist = [], 0
    for net in MODES:
        for N in tr["n_machines"]:
            env = make_fast_env(n_machines=N, network_mode=net, episode_duration=1800,
                                step_duration=tr["step_duration"], measured_util=True, **opt)
            for s in range(5000, 5010):
                obs, _ = env.reset(seed=s); p.reset(); done = False
                while not done:
                    a, _ = p.predict(obs)
                    obs, _, done, _, _ = env.step(a)
                logx.append(np.log(np.maximum(np.array(p.ratios[1:]), 1e-9)))
                n_persist += p.n_handover
    k_ref = 0.5 * math.log(tau)
    h = h_for_equal_false_alarms(logx, n_persist, k_ref)
    out = {"tau": tau, "k_ref": round(k_ref, 4), "h": h, "nominal_alarms_persistence": n_persist}
    json.dump(out, open("regime_agnostic_cusum.json", "w"), indent=2); print(out, flush=True)


def ra_evaluate(cfg, net):
    tr = cfg["train"]; opt = env_options(tr); c = json.load(open("regime_agnostic_cusum.json"))
    tgt = json.load(open("autoscaler_targets.json"))["stable"]["target"]
    ens = EnsemblePolicy([PPOPolicy(Path("models") / "stable" / f"ppo_seed_{s}.zip", "ppo", s)
                          for s in tr["seeds"]])
    pol = RAHybrid3(ens, "ra_hybrid3_B", opt["w_c"], opt["w_switch"], opt["degraded_congestion"],
                    threshold=c["tau"], h=c["h"], hpa_target=tgt)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run_rows([pol], net, cfg)).to_csv(OUT / f"eval_ra_h3_{net}.csv", index=False)
    print("done ra", net, flush=True)


# ---------------------------------------------------------------- 2. doubled server cost
def x2(net):
    cfg = json.load(open("experiment_config_server_cost_x2.json")); tr = cfg["train"]; opt = env_options(tr)
    kw = dict(w_c=opt["w_c"], w_switch=opt["w_switch"], deg_cong=opt["degraded_congestion"])
    # tau: 99th percentile of the nominal ratio; h: equal false alarms
    probe = HybridAdaptivePolicy(net, FixedCPolicy(C_MAX), threshold=1e9, **kw)
    ratios = []
    envs = [make_fast_env(n_machines=N, network_mode=net, episode_duration=1800,
                          step_duration=tr["step_duration"], measured_util=True, **opt)
            for N in tr["n_machines"]]
    for env in envs:
        for s in range(5000, 5010):
            obs, _ = env.reset(seed=s); probe.reset(); done = False
            while not done:
                ratios.append(probe.ratio(obs)); a, _ = probe.predict(obs)
                obs, _, done, _, _ = env.step(a)
    tau = round(float(np.quantile(ratios, 0.99)), 4)
    p = HybridAdaptivePolicy(net, FixedCPolicy(C_MAX), threshold=tau, **kw)
    logx, n_persist = [], 0
    for env in envs:
        for s in range(5000, 5010):
            obs, _ = env.reset(seed=s); p.reset(); done = False; x = []
            while not done:
                x.append(math.log(max(p.ratio(obs), 1e-9))); a, _ = p.predict(obs)
                obs, _, done, _, _ = env.step(a)
            logx.append(np.array(x)); n_persist += p.n_handover
    k_ref = 0.5 * math.log(tau)
    h = h_for_equal_false_alarms(logx, n_persist, k_ref)
    tgt = tune_hpa(cfg, [net])[net]["target"]
    cal = {"tau": tau, "h": h, "hpa_target": tgt, "nominal_alarms_persistence": n_persist,
           "agents": "trained on the stable network at w_C = 0.10 (transfer)"}
    json.dump(cal, open(f"server_cost_x2_{net}_calibration.json", "w"), indent=2); print(net, cal, flush=True)

    ens = EnsemblePolicy([PPOPolicy(Path("models_server_cost_x2") / "stable" / f"ppo_seed_{s}.zip", "ppo", s)
                          for s in tr["seeds"]])
    hpa = HPAPolicy(tgt)
    pols = [FixedCPolicy(c) for c in ACTION_TO_C] + [hpa, ens,
            CusumHybridPolicy(net, FixedCPolicy(C_MAX), threshold=tau, h=h, name="cusum_fallback_c6", **kw),
            CusumFloorPolicy(net, ens, threshold=tau, h=h, name="cusum_floor_ppo", **kw),
            Hybrid3Policy(net, ens, threshold=tau, h=h, hpa_target=tgt, use_model_floor=True,
                          name="hybrid3_B", **kw)]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run_rows(pols, net, cfg)).to_csv(OUT / f"eval_x2_{net}.csv", index=False)
    print("done x2", net, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ra-calibrate", action="store_true"); ap.add_argument("--ra", action="store_true")
    ap.add_argument("--network"); ap.add_argument("--x2")
    a = ap.parse_args(); cfg = json.load(open("experiment_config.json"))
    if a.ra_calibrate:
        ra_calibrate(cfg)
    if a.ra and a.network:
        ra_evaluate(cfg, a.network)
    if a.x2:
        x2(a.x2)
