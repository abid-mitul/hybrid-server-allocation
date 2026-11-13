"""Revision item 4: controllers that do NOT know the true network regime.

In the original controllers the queueing model (21) and the mismatch test use
the true regime (stable, degraded or noisy) and the PPO agents are trained per
regime.  Here the regime is estimated online from the measurements the
controller already has:

  For every window with a valid sojourn measurement (0 < T_hat < T_SLA, no
  blocking, and the same number of servers as in the window before, so that
  the transient after a change of c is skipped) and every regime r, the log-ratio
        l_{k,r} = log( T_hat_k / T_r(Lambda_hat_k, c_k, mu_hat_k) )
  of the measured to the predicted mean sojourn time is computed.  The
  estimate is the regime whose running mean log-ratio is closest to zero.
  Under nominal conditions the three regimes are 0.21-0.49 apart in this
  log-ratio and the window-to-window spread is 0.02-0.10, so one window is
  usually enough.  Before the first valid window the middle regime (noisy) is
  assumed.

  The mismatch test compares a window with the prediction under the regime
  estimated BEFORE that window, so a sudden departure from the model cannot
  explain itself away by moving the regime estimate.

Controllers
  RegimeAgnosticModel    self-calibrating model with the estimated regime
  RegimeAgnosticHybrid   the same model plus the mismatch test; on alarm it
                         falls back to c_max (fallback="c6") or to the PPO
                         ensemble trained on the stable network only
                         (fallback="ppo_stable"), so no regime knowledge
                         enters anywhere

    python regime_agnostic.py --calibrate               # one threshold, nominal runs only
    python regime_agnostic.py --network stable          # evaluate (repeat for degraded, noisy)
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from fast_env import make_fast_env, env_options, T_SLA, LAMBDA_MAX, C_MIN, C_MAX, ACTION_TO_C
from model_policy import expected_sojourn
from variants import AdaptiveModelPolicy, EnsemblePolicy
from fast_evaluate import (scenarios_for, extra_scenarios_for, structural_scenarios_for,
                           structural2_scenarios_for, env_kwargs_for, run_episode,
                           FixedCPolicy, PPOPolicy)

MODES = ["stable", "degraded", "noisy"]
DEFAULT_MODE = "noisy"
THRESH_FILE = "regime_agnostic_threshold.json"


class RegimeEstimator:
    def __init__(self, deg_cong):
        self.deg_cong = deg_cong
        self.reset()

    def reset(self):
        self.sum = dict.fromkeys(MODES, 0.0)
        self.n = 0
        self.prev_c = None

    @property
    def mode(self):
        if self.n == 0:
            return DEFAULT_MODE
        return min(MODES, key=lambda m: abs(self.sum[m] / self.n))

    def update(self, obs, lam, mu):
        T, p_b = float(obs[1]) * T_SLA, float(obs[4])
        c = int(round(C_MIN + float(obs[5]) * (C_MAX - C_MIN)))
        settled = c == self.prev_c            # skip the transient after a change of c
        self.prev_c = c
        if not settled or not (0.0 < float(obs[1]) < 0.999) or p_b > 0.0 or lam <= 1.0:
            return
        for m in MODES:
            self.sum[m] += math.log(T / expected_sojourn(lam, c, m, self.deg_cong, mu=mu))
        self.n += 1


class RegimeAgnosticModel(AdaptiveModelPolicy):
    """Self-calibrating model; regime estimated from the sojourn measurements."""
    name = "ra_model"

    def __init__(self, w_c, w_switch, deg_cong):
        super().__init__(DEFAULT_MODE, w_c=w_c, w_switch=w_switch, deg_cong=deg_cong)
        self.est = RegimeEstimator(deg_cong)
        self.train_seed = ""

    def reset(self):
        super().reset()
        self.est.reset()
        self.mode = DEFAULT_MODE
        self.mode_log = []

    def observe(self, obs):
        """Update mu_hat and the regime estimate with the window just ended."""
        thr, util = float(obs[3]) * LAMBDA_MAX, float(obs[2])
        c = int(round(C_MIN + float(obs[5]) * (C_MAX - C_MIN)))
        mu = thr / (c * util) if (thr > 1.0 and 0.0 < util < 0.999) else self.mu_hat
        self.est.update(obs, self.estimate_lambda(obs), mu)
        self.mode = self.est.mode

    def predict(self, obs, deterministic=True, update=True):
        if update:
            self.observe(obs)
        self.mode_log.append(self.mode)
        return super().predict(obs, deterministic)


class RegimeAgnosticHybrid:
    """Regime-agnostic model plus the mismatch test (n_on = 2, n_off = 3)."""

    def __init__(self, fallback_policy, name, w_c, w_switch, deg_cong, threshold,
                 n_on=2, n_off=3):
        self.model = RegimeAgnosticModel(w_c, w_switch, deg_cong)
        self.fallback, self.name = fallback_policy, name
        self.deg_cong, self.threshold, self.n_on, self.n_off = deg_cong, threshold, n_on, n_off
        self.train_seed = ""
        self.reset()

    def reset(self):
        self.model.reset()
        if hasattr(self.fallback, "reset"):
            self.fallback.reset()
        self.use_fb = False
        self.run_hi = self.run_lo = self.n_handover = self.ppo_steps = 0
        self.ratios = []

    def ratio(self, obs, mode):
        c = int(round(C_MIN + float(obs[5]) * (C_MAX - C_MIN)))
        thr, util = float(obs[3]) * LAMBDA_MAX, float(obs[2])
        mu = thr / (c * util) if (thr > 1.0 and 0.0 < util < 0.999) else self.model.mu_hat
        lam = self.model.estimate_lambda(obs)
        return float(obs[1]) * T_SLA / expected_sojourn(lam, c, mode, self.deg_cong, mu=mu)

    def predict(self, obs, deterministic=True):
        r = self.ratio(obs, self.model.mode)          # regime estimated before this window
        self.ratios.append(r)
        self.model.observe(obs)
        alarm = r > self.threshold
        self.run_hi = self.run_hi + 1 if alarm else 0
        self.run_lo = 0 if alarm else self.run_lo + 1
        if not self.use_fb and self.run_hi >= self.n_on:
            self.use_fb = True
            self.n_handover += 1
        elif self.use_fb and self.run_lo >= self.n_off:
            self.use_fb = False
        a_model, _ = self.model.predict(obs, update=False)
        if self.use_fb:
            self.ppo_steps += 1
            return self.fallback.predict(obs)[0], None
        return a_model, None


def calibrate(cfg):
    """99th percentile of the ratio under NOMINAL conditions, pooled over the
    three regimes and the three training loads (seeds 5000-5009, as before)."""
    tr = cfg["train"]; opt = env_options(tr)
    h = RegimeAgnosticHybrid(FixedCPolicy(C_MAX), "cal", opt["w_c"], opt["w_switch"],
                             opt["degraded_congestion"], threshold=1e9)
    r, acc = [], []
    for net in MODES:
        for N in tr["n_machines"]:
            env = make_fast_env(n_machines=N, network_mode=net, episode_duration=1800,
                                step_duration=tr["step_duration"], measured_util=True, **opt)
            for s in range(5000, 5010):
                obs, _ = env.reset(seed=s); h.reset(); done = False
                while not done:
                    a, _ = h.predict(obs)
                    obs, _, done, _, _ = env.step(a)
                r += h.ratios[1:]                      # the first window has no estimate yet
                acc += [m == net for m in h.model.mode_log[1:]]
    thr = round(float(np.quantile(r, 0.99)), 4)
    json.dump({"threshold": thr}, open(THRESH_FILE, "w"), indent=2)
    print(f"threshold {thr}; regime correctly identified in {100 * np.mean(acc):.1f}% "
          f"of nominal decisions after the first window", flush=True)


def evaluate(net, cfg):
    ev, tr = cfg["eval"], cfg["train"]; opt = env_options(tr)
    thr = json.load(open(THRESH_FILE))["threshold"]
    kw = dict(w_c=opt["w_c"], w_switch=opt["w_switch"], deg_cong=opt["degraded_congestion"])
    ens = EnsemblePolicy([PPOPolicy(Path("models") / "stable" / f"ppo_seed_{s}.zip", "ppo", s)
                          for s in tr["seeds"]])
    pols = [RegimeAgnosticModel(**kw),
            RegimeAgnosticHybrid(FixedCPolicy(C_MAX), "ra_fallback_c6", threshold=thr, **kw),
            RegimeAgnosticHybrid(ens, "ra_hybrid_ppo_stable", threshold=thr, **kw)]
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
                m = pol.model if hasattr(pol, "model") else pol
                extra = dict(regime_correct=float(np.mean([x == net for x in m.mode_log[1:]])))
                if hasattr(pol, "ppo_steps"):
                    extra.update(ppo_fraction=pol.ppo_steps / env.n_steps_per_episode,
                                 handovers=pol.n_handover)
                rows.append(dict(scenario=sname, group=scfg["group"], network=net,
                                 n_machines=k["n_machines"],
                                 n_machines_end=k.get("burst_n_end", k["n_machines"]),
                                 policy=pol.name, train_seed="", episode=ep, eval_seed=seed,
                                 **extra, **res))
        print(f"    {sname}: {time.time() - t0:.0f}s", flush=True)
    out = Path("data") / "regime_agnostic"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"eval_ra_{net}.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--network", default=None)
    a = ap.parse_args()
    cfg = json.load(open("experiment_config.json"))
    if a.calibrate:
        calibrate(cfg)
    if a.network:
        evaluate(a.network, cfg)
