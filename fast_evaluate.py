"""
Evaluation of all policies on the 12 scenarios, using the fast simulator.

Policies evaluated in every scenario:
  fixed_c3 .. fixed_c6   static server counts (the best one is the static optimum)
  round_robin            cycles c = 3,4,5,6
  fixed_priority         hand-tuned threshold rule (baselines.py)
  ga                     GA-tuned threshold rule, fitted on the scenario's network
  ppo                    PPO trained on the scenario's network (one row per seed)
  ppo_stable_transfer    PPO trained on the stable network, tested on degraded/noisy

All policies see the SAME random seeds per episode (common random numbers),
so results are paired and suitable for Wilcoxon signed-rank tests.

In burst scenarios N changes from N_start to N_end in the middle of the episode.

Usage (normally called by run_experiments.py):
    python fast_evaluate.py --network stable
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from parameters import SCENARIOS
from baselines import RoundRobinPolicy, FixedPriorityPolicy, GeneticAlgorithmPolicy
from fast_env import make_fast_env, ACTION_TO_C, env_options
from model_policy import QueueingModelPolicy


def scenarios_for(network):
    """Seen, unseen and first mismatch scenarios of this network (8 per network).

    group = "seen"      loads used in training (80, 200, 350) and the 200->350 burst
            "unseen"    loads never used in training (120, 280) and a 350->200 drop
            "mismatch"  servers 30% slower than the nominal rate the policies assume
    """
    out = {}
    for k, v in SCENARIOS.items():
        if v["network"] == network:
            out[k] = {**v, "group": "seen"}
    out[f"unseen120_{network}"] = {"N": 120, "network": network, "group": "unseen"}
    out[f"unseen280_{network}"] = {"N": 280, "network": network, "group": "unseen"}
    out[f"drop_{network}"] = {"N_start": 350, "N_end": 200, "network": network,
                              "group": "unseen"}
    out[f"slowservers_{network}"] = {"N": 200, "network": network, "mu_scale": 0.7,
                                     "group": "mismatch"}
    return out


def extra_scenarios_for(network):
    """Further parameter-mismatch scenarios (5 per network).

    The policies (queueing model, GA rule, PPO) all assume the nominal
    parameters; the simulated system differs as follows:
      slowservers85   servers 15% slower (milder version of slowservers)
      fastservers     servers 30% faster, at N = 280 (over-provisioning risk)
      congestion      network congestion sensitivity doubled
      failures        machines degrade and fail twice as fast, at N = 280
      combined        servers 30% slower and congestion sensitivity doubled
    """
    return {
        f"slowservers85_{network}": {"N": 200, "network": network, "mu_scale": 0.85,
                                     "group": "mismatch"},
        f"fastservers_{network}": {"N": 280, "network": network, "mu_scale": 1.3,
                                   "group": "mismatch"},
        f"congestion_{network}": {"N": 200, "network": network, "eta_scale": 2.0,
                                  "group": "mismatch"},
        f"failures_{network}": {"N": 280, "network": network, "fail_scale": 2.0,
                                "group": "mismatch"},
        f"combined_{network}": {"N": 200, "network": network, "mu_scale": 0.7,
                                "eta_scale": 2.0, "group": "mismatch"},
    }


def structural_scenarios_for(network):
    """Structural model mismatch, first set (6 per network).

    The true system departs from the STRUCTURE of the M/M/c/K model, not only
    from a parameter value. Utilisation is measured from recorded processing
    times (as in a server log). Each at N = 200 and N = 280:
      highvar     service times hyperexponential, same mean, SCV = 4
      bursty      common sending-rate factor alternates 1.5 / 0.5,
                  exponential holding times with mean 10 s (same mean load)
      mixedspeed  server speeds 1.4, 0.6, 1.4, 0.6, 1.4, 0.6 (mean = nominal)
    """
    out = {}
    for N in (200, 280):
        base = {"N": N, "network": network, "group": "structural", "measured_util": True}
        out[f"highvar{N}_{network}"] = {**base, "svc_scv": 4.0}
        out[f"bursty{N}_{network}"] = {**base, "burst_factors": (1.5, 0.5),
                                       "burst_switch_mean": 10.0}
        out[f"mixedspeed{N}_{network}"] = {**base,
                                           "server_speeds": [1.4, 0.6, 1.4, 0.6, 1.4, 0.6]}
    return out


def structural2_scenarios_for(network):
    """Structural model mismatch, second set (4 per network).

      batch     related tasks arrive in batches, geometric size with mean 3;
                batch rate lowered so that the mean load is unchanged
      svcburst  processing time alternates between 0.5 and 1.5 times its
                nominal value, exponential holding times with mean 10 s
    Each at N = 200 and N = 280; utilisation measured from processing times.
    """
    out = {}
    for N in (200, 280):
        base = {"N": N, "network": network, "group": "structural", "measured_util": True}
        out[f"batch{N}_{network}"] = {**base, "batch_mean": 3.0}
        out[f"svcburst{N}_{network}"] = {**base, "service_burst": (0.5, 1.5),
                                         "service_switch_mean": 10.0}
    return out


class FixedCPolicy:
    def __init__(self, c):
        self.c = c
        self.name = f"fixed_c{c}"
        self.action = ACTION_TO_C.index(c)

    def predict(self, obs, deterministic=True):
        return self.action, None

    def reset(self):
        pass


class PPOPolicy:
    def __init__(self, path, name, seed):
        from stable_baselines3 import PPO
        self.model = PPO.load(path, device="cpu")
        self.name = name
        self.train_seed = seed

    def predict(self, obs, deterministic=True):
        a, _ = self.model.predict(obs, deterministic=True)
        return int(a), None

    def reset(self):
        pass


def env_kwargs_for(scen_cfg, train_cfg, episode_duration):
    kw = dict(network_mode=scen_cfg["network"], episode_duration=episode_duration,
              step_duration=train_cfg["step_duration"],
              **env_options(train_cfg))
    if "N" in scen_cfg:
        kw["n_machines"] = scen_cfg["N"]
    else:
        kw["n_machines"] = scen_cfg["N_start"]
        kw["burst_n_end"] = scen_cfg["N_end"]
        kw["burst_time"] = episode_duration / 2
    for key in ("mu_scale", "eta_scale", "fail_scale", "svc_scv", "server_speeds",
                "burst_factors", "burst_switch_mean", "measured_util", "batch_mean",
                "service_burst", "service_switch_mean"):
        if key in scen_cfg:
            kw[key] = scen_cfg[key]
    return kw


def run_episode(env, policy, seed, trace=False):
    obs, _ = env.reset(seed=seed)
    policy.reset()
    rew, cs, steps = [], [], []
    tot = dict(off=0, drop=0, comp=0, sla=0.0, soj=0.0, wait=0.0, delay=0.0)
    done = False
    while not done:
        a, _ = policy.predict(obs, deterministic=True)
        obs, r, done, _, info = env.step(a)
        rew.append(r)
        cs.append(info["c"])
        nc = info["n_completions"]
        tot["off"] += info["n_arrivals"]
        tot["drop"] += info["n_drops"]
        tot["comp"] += nc
        tot["sla"] += info["sla_compliance"] * nc
        tot["soj"] += info["mean_sojourn"] * nc
        tot["wait"] += info["mean_wait"] * nc
        tot["delay"] += info["mean_delay"] * nc
        if trace:
            steps.append(dict(step=info["step"], c=info["c"], reward=r,
                              queue_length=info["queue_length"],
                              throughput=info["throughput"],
                              mean_sojourn_ms=1000 * info["mean_sojourn"],
                              p_b=info["p_b"], sla=info["sla_compliance"],
                              n_machines=info["n_machines"]))
    comp = max(tot["comp"], 1)
    T = env.episode_duration
    res = dict(
        total_reward=float(np.sum(rew)), mean_reward=float(np.mean(rew)),
        throughput=tot["comp"] / T,
        mean_sojourn_ms=1000 * tot["soj"] / comp,
        mean_wait_ms=1000 * tot["wait"] / comp,
        mean_delay_ms=1000 * tot["delay"] / comp,
        sla_compliance=tot["sla"] / comp,
        p_b=tot["drop"] / tot["off"] if tot["off"] else 0.0,
        mean_c=float(np.mean(cs)), n_switches=int(np.sum(np.diff(cs) != 0)),
        n_offered=tot["off"], n_drops=tot["drop"], n_completions=tot["comp"],
        **{f"frac_c{c}": float(np.mean(np.array(cs) == c)) for c in ACTION_TO_C},
    )
    return res, steps


def fit_ga(network, cfg, seed=42):
    tr = cfg["train"]
    loads = tr["n_machines"] if isinstance(tr["n_machines"], list) else [tr["n_machines"]]
    n_eps = max(2, len(loads))
    counter = {"k": 0}

    def factory():
        # each candidate is scored on every training load with the same seeds
        k = counter["k"] % n_eps
        counter["k"] += 1
        return make_fast_env(n_machines=loads[k % len(loads)], network_mode=network,
                             episode_duration=tr["episode_duration"],
                             step_duration=tr["step_duration"], seed=seed + k,
                             **env_options(tr))
    ga = GeneticAlgorithmPolicy()
    ga.fit(factory, population_size=20, n_generations=10, n_eval_episodes=n_eps,
           seed=seed, verbose=False)
    return ga


def evaluate_network(network, cfg, root=".", extra=False):
    """Evaluate all policies on the 8 scenarios of this network
    (or, with extra=True, on the additional mismatch scenarios)."""
    root = Path(root)
    ev, tr = cfg["eval"], cfg["train"]
    scen = extra_scenarios_for(network) if extra else scenarios_for(network)

    policies = [FixedCPolicy(c) for c in ACTION_TO_C]
    policies += [RoundRobinPolicy(), FixedPriorityPolicy()]
    opt = env_options(tr)
    policies.append(QueueingModelPolicy(network, w_c=opt["w_c"], w_switch=opt["w_switch"],
                                        deg_cong=opt["degraded_congestion"]))
    if ev.get("include_ga", True):
        t0 = time.time()
        ga = fit_ga(network, cfg)
        print(f"    GA fitted in {time.time() - t0:.0f}s, thresholds="
              f"{[round(x, 3) for x in ga.thresholds]}", flush=True)
        policies.append(ga)
    for s in tr["seeds"]:
        p = root / "models" / network / f"ppo_seed_{s}.zip"
        if p.exists():
            policies.append(PPOPolicy(p, "ppo", s))
        if network != "stable":
            q = root / "models" / "stable" / f"ppo_seed_{s}.zip"
            if q.exists():
                policies.append(PPOPolicy(q, "ppo_stable_transfer", s))

    rows, traces = [], []
    for sname, scfg in scen.items():
        kw = env_kwargs_for(scfg, tr, ev["episode_duration"])
        env = make_fast_env(**kw)
        t0 = time.time()
        for pol in policies:
            for ep in range(ev["episodes"]):
                seed = ev["seed_base"] + ep
                res, steps = run_episode(env, pol, seed, trace=(ep == 0))
                base = dict(scenario=sname, group=scfg["group"], network=network,
                            n_machines=kw["n_machines"],
                            n_machines_end=kw.get("burst_n_end", kw["n_machines"]),
                            policy=pol.name,
                            train_seed=getattr(pol, "train_seed", ""),
                            episode=ep, eval_seed=seed)
                rows.append({**base, **res})
                for st in steps:
                    traces.append({**base, **st})
        print(f"    {sname}: {len(policies)} policies x {ev['episodes']} episodes "
              f"in {time.time() - t0:.0f}s", flush=True)

    out = root / "data"
    out.mkdir(exist_ok=True)
    tag = "extra_" if extra else ""
    pd.DataFrame(rows).to_csv(out / f"eval_{tag}{network}.csv", index=False)
    pd.DataFrame(traces).to_csv(out / f"traces_{tag}{network}.csv", index=False)
    return out / f"eval_{tag}{network}.csv"


def summarize(root="."):
    """Merge per-network results and write a per-scenario/per-policy summary."""
    out = Path(root) / "data"
    files = sorted(f for f in out.glob("eval_*.csv") if not f.name.startswith("eval_extra_"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(out / "evaluation_results.csv", index=False)
    metrics = ["total_reward", "throughput", "mean_sojourn_ms", "sla_compliance",
               "p_b", "mean_c", "n_switches"]
    g = df.groupby(["scenario", "policy"])[metrics]
    summ = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd"))
    summ["n"] = g.size()
    summ.reset_index().to_csv(out / "summary_by_scenario_policy.csv", index=False)
    from stats import write_statistics
    write_statistics(df, out)
    return out / "summary_by_scenario_policy.csv"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True, choices=["stable", "degraded", "noisy"])
    ap.add_argument("--config", default="experiment_config.json")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)
    evaluate_network(a.network, cfg)
    summarize()
