"""v1.3 tasks C1 and C2: what do the PPO agents add to CHAMPS?

C1  "CUSUM + floors, no agents": identical to CHAMPS (same self-calibrating
    model, CUSUM test, psi*, h, exit rule, autoscaler), but in learned mode
    c_k = max(c_hpa, c_model); the ensemble is never queried.
C2  CHAMPS re-run with logging: in every learned-mode decision the three
    candidates c_ppo, c_hpa, c_model are recorded and the component that set the
    final c_k = max(...) is counted (sole maximum, or a tie).

Both controllers are run on the same seeds as every earlier study:
    --study normal  NET   69 main scenarios (23 per regime), w_C = 0.05
    --study x2      NET   doubled server cost, w_C = 0.10 (stable: retrained
                          agents; degraded/noisy: transferred stable agents,
                          recalibrated psi*, h and autoscaler target)
    --study switch        9 regime-switch scenarios (regime_switch.py)
Each row also stores the action and mode sequence of the episode, so that the
identity of the two controllers in model mode can be verified offline.
Writes data/v13/ppo_role_<study>_<net>.csv
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from fast_env import make_fast_env, env_options, ACTION_TO_C
from fast_evaluate import (scenarios_for, extra_scenarios_for, structural_scenarios_for,
                           structural2_scenarios_for, env_kwargs_for, run_episode, PPOPolicy)
from variants import EnsemblePolicy
from hybrid3 import Hybrid3Policy

ALL = lambda n: {**scenarios_for(n), **extra_scenarios_for(n),
                 **structural_scenarios_for(n), **structural2_scenarios_for(n)}
OUT = Path("data") / "v13"


class LoggedHybrid3(Hybrid3Policy):
    """CHAMPS (both floors) or, with use_agents=False, CUSUM + floors without agents."""

    def __init__(self, *a, use_agents=True, **kw):
        self.use_agents = use_agents
        super().__init__(*a, use_model_floor=True, **kw)

    def reset(self):
        super().reset()
        self.log = dict(n_learned=0, ppo_sole=0, hpa_sole=0, model_sole=0, tie=0,
                        ppo_in_max=0, ppo_below_floor=0)
        self.actions, self.modes = [], []

    def predict(self, obs, deterministic=True):
        a_hpa, _ = self.hpa.predict(obs)
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
        a_model, a_hpa = int(a_model), int(a_hpa)
        if self.use_ppo:
            self.ppo_steps += 1
            L = self.log; L["n_learned"] += 1
            if self.use_agents:
                a_ppo = int(self.ppo.predict(obs)[0])
                a = max(a_ppo, a_hpa, a_model)
                top = [n for n, v in (("ppo", a_ppo), ("hpa", a_hpa), ("model", a_model)) if v == a]
                if len(top) > 1:
                    L["tie"] += 1
                else:
                    L[top[0] + "_sole"] += 1
                L["ppo_in_max"] += a_ppo == a
                L["ppo_below_floor"] += a_ppo < max(a_hpa, a_model)
            else:
                a = max(a_hpa, a_model)
                top = [n for n, v in (("hpa", a_hpa), ("model", a_model)) if v == a]
                if len(top) > 1:
                    L["tie"] += 1
                else:
                    L[top[0] + "_sole"] += 1
            self.actions.append(a); self.modes.append("L")
            return a, None
        self.actions.append(a_model); self.modes.append("M")
        return a_model, None


def setup(study, net):
    if study == "x2":
        cfg = json.load(open("experiment_config_server_cost_x2.json")); models = "models_server_cost_x2"
        if net == "stable":
            t = json.load(open("server_cost_x2_thresholds.json"))
            tgt = json.load(open("autoscaler_target_server_cost_x2.json"))["target"]
        else:
            t = json.load(open(f"server_cost_x2_{net}_calibration.json")); tgt = t["hpa_target"]
        agent_net = "stable"
    else:
        cfg = json.load(open("experiment_config.json")); models = "models"
        t = json.load(open("cusum_thresholds.json"))[net]
        tgt = json.load(open("autoscaler_targets.json"))[net]["target"]
        agent_net = net
    tr = cfg["train"]; opt = env_options(tr)
    ens = EnsemblePolicy([PPOPolicy(Path(models) / agent_net / f"ppo_seed_{s}.zip", "ppo", s)
                          for s in tr["seeds"]])
    kw = dict(w_c=opt["w_c"], w_switch=opt["w_switch"], deg_cong=opt["degraded_congestion"],
              threshold=t["tau"], h=t["h"], hpa_target=tgt)
    pols = [LoggedHybrid3(net, ens, name="champs", use_agents=True, **kw),
            LoggedHybrid3(net, ens, name="no_agents", use_agents=False, **kw)]
    return cfg, opt, pols


def row(pol, env, **base):
    n = env.n_steps_per_episode
    return dict(**base, policy=pol.name, ppo_fraction=pol.ppo_steps / n, handovers=pol.n_handover,
                first_alarm=(-1 if pol.first_alarm is None else pol.first_alarm),
                actions="".join(str(ACTION_TO_C[a]) for a in pol.actions), modes="".join(pol.modes),
                **pol.log)


def run_main(study, net):
    cfg, opt, pols = setup(study, net); ev, tr = cfg["eval"], cfg["train"]; rows = []
    for sname, scfg in ALL(net).items():
        k = env_kwargs_for(scfg, tr, ev["episode_duration"]); k["measured_util"] = True
        env = make_fast_env(**k); t0 = time.time()
        for pol in pols:
            for ep in range(ev["episodes"]):
                seed = ev["seed_base"] + ep
                res, _ = run_episode(env, pol, seed)
                rows.append(row(pol, env, scenario=sname, group=scfg["group"], network=net,
                                episode=ep, eval_seed=seed, **res))
        print(f"    {study} {net} {sname}: {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"ppo_role_{study}_{net}.csv", index=False)
    print("done", study, net, flush=True)


def run_switch():
    from regime_switch import SwitchEnv, SWITCHES, LOADS
    rows = []
    for start, end in SWITCHES:
        cfg, opt, pols = setup("normal", start); ev, tr = cfg["eval"], cfg["train"]
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
                    rows.append(row(pol, env, scenario=f"{start}_to_{end}_N{N}",
                                    switch=f"{start}->{end}", n_machines=N,
                                    episode=ep, eval_seed=seed, **res))
            print(f"    switch {start}->{end} N={N}: {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "ppo_role_switch.csv", index=False)
    print("done switch", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["normal", "x2", "switch"]); ap.add_argument("--network")
    a = ap.parse_args()
    if a.study == "switch":
        run_switch()
    else:
        run_main(a.study, a.network)
