"""Revision item 3: oracle adaptive baseline.

OracleModelPolicy applies the queueing-model rule (21) with PERFECT information
instead of estimates: at every decision it reads from the simulator
  * the true arrival rate of the next window, lambda_0 n0 + lambda_1 n1 from the
    true machine counts (a population change scheduled for that window is
    applied first, as the simulator will apply it);
  * the true processing rate of the c servers that will be active
    (mu_true times their mean speed);
  * the true congestion coefficient eta of the delay model.
It keeps the planning horizon H and the switching cost of (21).

This is the best the queueing analysis can do when nothing is estimated.  It is
an upper reference where the model's structure holds (seen, unseen and
parameter-mismatch scenarios).  Under structural mismatch it shows what perfect
parameters can and cannot repair, and it is not an upper bound there.

Bursty traffic and bursty service keep the same mean, so the oracle uses the
mean arrival and processing rates; batches keep the same mean task rate.

    python evaluate_oracle.py --network stable      # repeat for degraded, noisy

Evaluated on all 69 scenarios with measured utilisation (the oracle does not use
it) and the common evaluation seeds.  Writes data/oracle/eval_oracle_<net>.csv.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from fast_env import (make_fast_env, env_options, ACTION_TO_C, C_MIN, C_MAX,
                      S_N0, S_N1, S_N2)
from fast_evaluate import (scenarios_for, extra_scenarios_for, structural_scenarios_for,
                           structural2_scenarios_for, env_kwargs_for, run_episode)
from model_policy import expected_step_reward


def _base_env(env):
    """make_fast_env may wrap the simulator; return the simulator itself."""
    e = env
    while not hasattr(e, "state") and hasattr(e, "env"):
        e = e.env
    return e


class OracleModelPolicy:
    name = "oracle_model"

    def __init__(self, env, network_mode, w_c, w_switch, deg_cong, horizon=10.0):
        self.env, self.mode = _base_env(env), network_mode
        self.w_c, self.w_switch, self.deg_cong, self.horizon = w_c, w_switch, deg_cong, horizon
        self.train_seed = ""
        self.reset()

    def reset(self):
        self.current_c = self.env.initial_c

    def true_lambda(self):
        e = self.env
        n = np.array([e.state[S_N0], e.state[S_N1], e.state[S_N2]], dtype=float)
        # population change that the simulator applies at the start of this window
        k = e.step_count + e.offset
        t_end = (k + 1) * e.step_duration
        b_time = e.burst_time + e.offset * e.step_duration
        if e.burst_n_end is not None and not e._burst_done and b_time < t_end:
            extra = e.burst_n_end - e.n_machines_start
            if extra >= 0:
                n[0] += extra
            else:
                n *= (n.sum() + extra) / n.sum()     # removal in expectation
        return e.lam0 * n[0] + e.lam1 * n[1]

    def predict(self, obs, deterministic=True):
        e = self.env
        if len(obs) > 5:
            self.current_c = int(round(C_MIN + float(obs[5]) * (C_MAX - C_MIN)))
        lam = self.true_lambda()
        eta = e.lat[1]
        scores = np.array([self.horizon * expected_step_reward(
            lam, c, self.mode, self.w_c, self.deg_cong,
            mu=e.mu * float(np.mean(e.srv_speed[:c])), eta=eta) for c in ACTION_TO_C])
        for a, c in enumerate(ACTION_TO_C):
            if c != self.current_c:
                scores[a] -= self.w_switch
        a = int(np.argmax(scores))
        self.current_c = ACTION_TO_C[a]
        return a, None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--network", required=True)
    ap.add_argument("--only", default=None)
    net = ap.parse_args().network; only = ap.parse_args().only
    cfg = json.load(open("experiment_config.json")); ev, tr = cfg["eval"], cfg["train"]
    opt = env_options(tr)
    scen = {**scenarios_for(net), **extra_scenarios_for(net),
            **structural_scenarios_for(net), **structural2_scenarios_for(net)}
    if only:
        scen = {only: scen[only]}
    rows = []
    for sname, scfg in scen.items():
        kw = env_kwargs_for(scfg, tr, ev["episode_duration"])
        kw["measured_util"] = True
        env = make_fast_env(**kw); t0 = time.time()
        pol = OracleModelPolicy(env, net, w_c=opt["w_c"], w_switch=opt["w_switch"],
                                deg_cong=opt["degraded_congestion"])
        for ep in range(ev["episodes"]):
            seed = ev["seed_base"] + ep
            res, _ = run_episode(env, pol, seed)
            rows.append(dict(scenario=sname, group=scfg["group"], network=net,
                             n_machines=kw["n_machines"],
                             n_machines_end=kw.get("burst_n_end", kw["n_machines"]),
                             policy=pol.name, train_seed="", episode=ep, eval_seed=seed, **res))
        print(f"    {sname}: {time.time() - t0:.0f}s", flush=True)
    out = Path("data") / "oracle"; out.mkdir(parents=True, exist_ok=True)
    name = f"eval_oracle_{net}{'_' + only if only else ''}.csv"
    pd.DataFrame(rows).to_csv(out / name, index=False)
    print("done", flush=True)
