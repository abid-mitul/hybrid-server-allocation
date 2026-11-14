"""Revision item 5: CUSUM change detection as the validity test.

The proposed controller enters its fallback when the ratio r_k of measured to
predicted mean sojourn time exceeds the calibrated threshold tau in n_on = 2
consecutive windows (persistence test).  Here only that entry rule is replaced
by a one-sided CUSUM on x_k = log r_k:

    S_k = max(0, S_{k-1} + x_k - k_ref),   alarm when S_k > h,

with allowance k_ref = log(tau) / 2 (half the log-shift the original test is
built to detect, the classical choice).  Everything else is unchanged: the same
self-calibrating model, the same ratio, the same exit rule (n_off = 3
consecutive windows with r_k <= tau), and S is reset to zero on exit.

h is calibrated for EQUAL FALSE ALARMS: on the same nominal calibration runs as
tau (training loads, seeds 5000-5009, no test scenario), h is the smallest value
at which the CUSUM raises no more alarms than the persistence test does.  The
two tests are therefore compared at the same nominal false-alarm rate, and
the comparison is about how quickly each detects real mismatch.

    python cusum_test.py --calibrate
    python cusum_test.py --network stable        # repeat for degraded, noisy
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from fast_env import make_fast_env, env_options, C_MAX
from fast_evaluate import (scenarios_for, extra_scenarios_for, structural_scenarios_for,
                           structural2_scenarios_for, env_kwargs_for, run_episode,
                           FixedCPolicy, PPOPolicy)
from hybrid_policy import load_thresholds
from variants import HybridAdaptivePolicy, EnsemblePolicy

FILE = "cusum_thresholds.json"


class CusumHybridPolicy(HybridAdaptivePolicy):
    def __init__(self, network, fallback, w_c, w_switch, deg_cong, threshold, h,
                 name="cusum_hybrid", n_off=3):
        super().__init__(network, fallback, w_c, w_switch, deg_cong, threshold, n_off=n_off)
        self.k_ref, self.h, self.name = 0.5 * np.log(threshold), h, name

    def reset(self):
        super().reset()
        self.S, self.first_alarm, self.k = 0.0, None, 0

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
            return self.ppo.predict(obs)[0], None
        return a_model, None


class PersistenceProbe(HybridAdaptivePolicy):
    """The original test, recording the window of its first alarm."""
    def reset(self):
        super().reset()
        self.first_alarm, self.k = None, 0

    def predict(self, obs, deterministic=True):
        before = self.n_handover
        out = super().predict(obs, deterministic)
        if self.n_handover > before and self.first_alarm is None:
            self.first_alarm = self.k
        self.k += 1
        return out


def calibrate(cfg):
    tr = cfg["train"]; opt = env_options(tr); thr = load_thresholds(); out = {}
    for net in tr["networks"]:
        tau = thr["adaptive_" + net]; k_ref = 0.5 * np.log(tau)
        p = HybridAdaptivePolicy(net, FixedCPolicy(C_MAX), opt["w_c"], opt["w_switch"],
                                 opt["degraded_congestion"], threshold=tau)
        traces, n_persist = [], 0
        for N in tr["n_machines"]:
            env = make_fast_env(n_machines=N, network_mode=net, episode_duration=1800,
                                step_duration=tr["step_duration"], measured_util=True, **opt)
            for s in range(5000, 5010):
                obs, _ = env.reset(seed=s); p.reset(); done = False; x = []
                while not done:
                    x.append(np.log(max(p.ratio(obs), 1e-9)))
                    a, _ = p.predict(obs)
                    obs, _, done, _, _ = env.step(a)
                traces.append(np.array(x)); n_persist += p.n_handover

        def n_alarms(h):                    # alarms of the CUSUM entry rule alone
            n = 0
            for x in traces:
                S = 0.0
                for v in x:
                    S = max(0.0, S + v - k_ref)
                    if S > h:
                        n, S = n + 1, 0.0
            return n
        grid = np.round(np.arange(0.0, 3.0, 0.005), 3)
        h = next(float(g) for g in grid if n_alarms(g) <= n_persist)
        out[net] = {"tau": tau, "k_ref": round(float(k_ref), 4), "h": h,
                    "nominal_alarms_persistence": int(n_persist),
                    "nominal_alarms_cusum": int(n_alarms(h)), "decisions": int(sum(map(len, traces)))}
        print(net, out[net], flush=True)
    json.dump(out, open(FILE, "w"), indent=2)


def evaluate(net, cfg):
    ev, tr = cfg["eval"], cfg["train"]; opt = env_options(tr)
    c = json.load(open(FILE))[net]; tau, h = c["tau"], c["h"]
    kw = dict(w_c=opt["w_c"], w_switch=opt["w_switch"], deg_cong=opt["degraded_congestion"])
    ens = EnsemblePolicy([PPOPolicy(Path("models") / net / f"ppo_seed_{s}.zip", "ppo", s)
                          for s in tr["seeds"]])
    pols = [CusumHybridPolicy(net, FixedCPolicy(C_MAX), threshold=tau, h=h,
                              name="cusum_fallback_c6", **kw),
            CusumHybridPolicy(net, ens, threshold=tau, h=h, name="cusum_hybrid_ppo", **kw),
            PersistenceProbe(net, FixedCPolicy(C_MAX), threshold=tau, **kw)]
    pols[2].name = "persistence_probe"          # = model_fallback_c6, re-run to log alarm times
    scen = {**scenarios_for(net), **extra_scenarios_for(net),
            **structural_scenarios_for(net), **structural2_scenarios_for(net)}
    rows = []
    for sname, scfg in scen.items():
        k = env_kwargs_for(scfg, tr, ev["episode_duration"]); k["measured_util"] = True
        env = make_fast_env(**k); t0 = time.time()
        for pol in pols:
            for ep in range(ev["episodes"]):
                seed = ev["seed_base"] + ep
                res, _ = run_episode(env, pol, seed)
                rows.append(dict(scenario=sname, group=scfg["group"], network=net,
                                 n_machines=k["n_machines"],
                                 n_machines_end=k.get("burst_n_end", k["n_machines"]),
                                 policy=pol.name, train_seed="", episode=ep, eval_seed=seed,
                                 ppo_fraction=pol.ppo_steps / env.n_steps_per_episode,
                                 handovers=pol.n_handover,
                                 first_alarm=(-1 if pol.first_alarm is None else pol.first_alarm),
                                 **res))
        print(f"    {sname}: {time.time() - t0:.0f}s", flush=True)
    out = Path("data") / "cusum"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"eval_cusum_{net}.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true"); ap.add_argument("--network")
    a = ap.parse_args(); cfg = json.load(open("experiment_config.json"))
    if a.calibrate:
        calibrate(cfg)
    if a.network:
        evaluate(a.network, cfg)
