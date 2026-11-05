"""
Model-guided hybrid controller (proposed method).

The queueing-model policy is in control by default. At every decision the
model also predicts the mean sojourn time it expects for the estimated load
and the current number of servers. If the measured sojourn time exceeds the
prediction by more than a threshold in `n_on` consecutive windows, the model
is taken to be invalid and the PPO agent decides. Control returns to the
model after `n_off` consecutive windows in which the measurement is again
consistent with the prediction.

The thresholds are calibrated ONLY under nominal conditions (training loads,
calibration seeds that are never used in evaluation), as the 99th percentile
of the ratio measured / predicted while the model policy is in control
(see calibrate_hybrid.py). No test scenario is used for calibration.
"""
import json
from pathlib import Path

from fast_env import C_MIN, T_SLA
from model_policy import QueueingModelPolicy, expected_sojourn

THRESHOLDS_FILE = Path(__file__).resolve().parent / "hybrid_thresholds.json"


class HybridPolicy:
    name = "hybrid"

    def __init__(self, network, ppo_policy, w_c, w_switch, deg_cong, threshold,
                 n_on=2, n_off=3, train_seed=""):
        self.network, self.deg_cong = network, deg_cong
        self.model = QueueingModelPolicy(network, w_c=w_c, w_switch=w_switch,
                                         deg_cong=deg_cong)
        self.ppo = ppo_policy
        self.threshold, self.n_on, self.n_off = threshold, n_on, n_off
        self.train_seed = train_seed
        self.reset()

    def reset(self):
        self.model.reset()
        self.use_ppo = False
        self.run_hi = 0
        self.run_lo = 0
        self.n_handover = 0
        self.ppo_steps = 0

    def ratio(self, obs):
        lam = self.model.estimate_lambda(obs)
        c = int(round(C_MIN + float(obs[5]) * 3))
        pred = expected_sojourn(lam, c, self.network, self.deg_cong)
        return float(obs[1]) * T_SLA / pred

    def predict(self, obs, deterministic=True):
        alarm = self.ratio(obs) > self.threshold
        self.run_hi = self.run_hi + 1 if alarm else 0
        self.run_lo = 0 if alarm else self.run_lo + 1
        if not self.use_ppo and self.run_hi >= self.n_on:
            self.use_ppo = True
            self.n_handover += 1
        elif self.use_ppo and self.run_lo >= self.n_off:
            self.use_ppo = False
        a_model, _ = self.model.predict(obs)   # keeps the model's state current
        if self.use_ppo:
            self.ppo_steps += 1
            a, _ = self.ppo.predict(obs)
            return a, None
        return a_model, None


def load_thresholds():
    with open(THRESHOLDS_FILE) as f:
        return json.load(f)
