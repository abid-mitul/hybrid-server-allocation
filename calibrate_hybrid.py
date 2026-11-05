"""Calibrate the hybrid alarm thresholds under NOMINAL conditions only.

For each network: run the controller's model part on the training loads
(80, 200, 350) with calibration seeds 5000-5009 (evaluation uses 1000-1019),
record the ratio measured / predicted mean sojourn time at every decision, and
store its 99th percentile in hybrid_thresholds.json. No test scenario is used.

    python calibrate_hybrid.py              # hybrid with the fixed-parameter model
    python calibrate_hybrid.py --adaptive   # hybrid with the self-calibrating model
"""
import json
import sys

import numpy as np

from fast_env import make_fast_env, env_options, T_SLA, C_MIN
from model_policy import QueueingModelPolicy, expected_sojourn

FILE = "hybrid_thresholds.json"


def _load():
    try:
        return json.load(open(FILE))
    except FileNotFoundError:
        return {}


def calibrate_nominal(cfg):
    tr = cfg["train"]; opt = env_options(tr); out = _load()
    for net in tr["networks"]:
        r = []
        for N in tr["n_machines"]:
            env = make_fast_env(n_machines=N, network_mode=net, episode_duration=1800,
                                step_duration=tr["step_duration"], **opt)
            qp = QueueingModelPolicy(net, w_c=opt["w_c"], w_switch=opt["w_switch"],
                                     deg_cong=opt["degraded_congestion"])
            for s in range(5000, 5010):
                obs, _ = env.reset(seed=s); qp.reset(); done = False
                while not done:
                    lam = qp.estimate_lambda(obs)
                    c = int(round(C_MIN + obs[5] * 3))
                    r.append(obs[1] * T_SLA / expected_sojourn(lam, c, net,
                                                               opt["degraded_congestion"]))
                    a, _ = qp.predict(obs)
                    obs, _, done, _, _ = env.step(a)
        out[net] = round(float(np.quantile(r, 0.99)), 4)
        print(net, out[net], flush=True)
    json.dump(out, open(FILE, "w"), indent=2)


def calibrate_adaptive(cfg):
    from variants import HybridAdaptivePolicy
    tr = cfg["train"]; opt = env_options(tr); out = _load()
    for net in tr["networks"]:
        h = HybridAdaptivePolicy(net, None, opt["w_c"], opt["w_switch"],
                                 opt["degraded_congestion"], threshold=1e9)
        r = []
        for N in tr["n_machines"]:
            env = make_fast_env(n_machines=N, network_mode=net, episode_duration=1800,
                                step_duration=tr["step_duration"], measured_util=True, **opt)
            for s in range(5000, 5010):
                obs, _ = env.reset(seed=s); h.reset(); done = False
                while not done:
                    r.append(h.ratio(obs))
                    a, _ = h.predict(obs)
                    obs, _, done, _, _ = env.step(a)
        out["adaptive_" + net] = round(float(np.quantile(r, 0.99)), 4)
        print("adaptive", net, out["adaptive_" + net], flush=True)
    json.dump(out, open(FILE, "w"), indent=2)


if __name__ == "__main__":
    cfg = json.load(open("experiment_config.json"))
    if "--adaptive" in sys.argv:
        calibrate_adaptive(cfg)
    else:
        calibrate_nominal(cfg)
