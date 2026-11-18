"""v1.3 task C4: long nominal run for the false-alarm rate of the validity test.

Nominal model in each network regime, training loads N = 80, 200, 350,
112 episodes per load (3 x 112 x 30 = 10 080 decisions per regime), seeds
20000-20111, disjoint from training (agent seeds), calibration (5000-5009),
the main evaluation (1000-1019) and every other study.

Two controllers run on the same seeds:
  champs       CHAMPS exactly as evaluated (CUSUM entry rule, both floors)
  persistence  identical, except that the entry rule is the persistence test
               (psi_k > psi* in two consecutive windows), with the same psi*

A false alarm is a switch from model mode to learned mode (an event).  The rate
is alarms divided by the decisions at which the test was evaluated in model
mode.  95% Clopper-Pearson intervals.

    python c4_false_alarm.py --network stable      # repeat for degraded, noisy
    python c4_false_alarm.py --summary
Writes data/c4/c4_<network>.csv and results/v13/c4_false_alarm.csv
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from fast_env import make_fast_env, env_options
from fast_evaluate import run_episode, PPOPolicy
from variants import EnsemblePolicy
from hybrid3 import Hybrid3Policy

LOADS, EPISODES, SEED0 = [80, 200, 350], 112, 20000


class PersistenceHybrid3(Hybrid3Policy):
    """CHAMPS with the persistence entry rule instead of the CUSUM."""

    def predict(self, obs, deterministic=True):
        a_hpa, _ = self.hpa.predict(obs)
        r = self.ratio(obs)
        alarm = r > self.threshold
        self.run_hi = self.run_hi + 1 if alarm else 0
        self.run_lo = 0 if alarm else self.run_lo + 1
        if not self.use_ppo and self.run_hi >= self.n_on:
            self.use_ppo = True
            self.n_handover += 1
        elif self.use_ppo and self.run_lo >= self.n_off:
            self.use_ppo = False
        self.k += 1
        a_model, _ = self.model.predict(obs)
        if self.use_ppo:
            self.ppo_steps += 1
            a = max(int(self.ppo.predict(obs)[0]), int(a_hpa))
            return max(a, int(a_model)), None
        return a_model, None


def run(net):
    cfg = json.load(open("experiment_config.json")); tr = cfg["train"]; opt = env_options(tr)
    t = json.load(open("cusum_thresholds.json"))[net]
    tgt = json.load(open("autoscaler_targets.json"))[net]["target"]
    ens = EnsemblePolicy([PPOPolicy(Path("models") / net / f"ppo_seed_{s}.zip", "ppo", s)
                          for s in tr["seeds"]])
    kw = dict(w_c=opt["w_c"], w_switch=opt["w_switch"], deg_cong=opt["degraded_congestion"],
              threshold=t["tau"], h=t["h"], hpa_target=tgt, use_model_floor=True)
    pols = [Hybrid3Policy(net, ens, name="champs", **kw),
            PersistenceHybrid3(net, ens, name="persistence", **kw)]
    rows = []
    for N in LOADS:
        env = make_fast_env(n_machines=N, network_mode=net, episode_duration=1800,
                            step_duration=tr["step_duration"], measured_util=True, **opt)
        t0 = time.time()
        for pol in pols:
            for ep in range(EPISODES):
                seed = SEED0 + ep
                res, _ = run_episode(env, pol, seed)
                n = env.n_steps_per_episode
                rows.append(dict(network=net, n_machines=N, policy=pol.name, eval_seed=seed,
                                 decisions=n, learned=pol.ppo_steps, alarms=pol.n_handover,
                                 model_mode_tests=n - pol.ppo_steps + pol.n_handover))
        print(f"  {net} N={N}: {time.time() - t0:.0f}s", flush=True)
    out = Path("data") / "c4"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"c4_{net}.csv", index=False)
    print("done", net, flush=True)


def summary():
    from scipy.stats import binomtest
    d = pd.concat([pd.read_csv(f"data/c4/c4_{n}.csv") for n in ("stable", "degraded", "noisy")])
    rows = []
    for (net, pol), g in d.groupby(["network", "policy"], sort=False):
        k, n = int(g.alarms.sum()), int(g.model_mode_tests.sum())
        ci = binomtest(k, n).proportion_ci(method="exact")
        rows.append(dict(network=net, test=pol, decisions=int(g.decisions.sum()), model_mode_tests=n,
                         alarms=k, rate_pct=100 * k / n, ci_low_pct=100 * ci.low, ci_high_pct=100 * ci.high,
                         learned_share_pct=100 * g.learned.sum() / g.decisions.sum(),
                         episodes_with_alarm_pct=100 * (g.alarms > 0).mean()))
    s = pd.DataFrame(rows)
    Path("results/v13").mkdir(parents=True, exist_ok=True)
    s.to_csv("results/v13/c4_false_alarm.csv", index=False)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--network"); ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.network:
        run(a.network)
    if a.summary:
        summary()
