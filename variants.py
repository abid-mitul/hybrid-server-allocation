"""Ensemble, self-calibrating and hybrid controllers.

EnsemblePolicy        median of the server counts chosen by the five PPO agents
                      trained on the same network
AdaptiveModelPolicy   queueing-model policy that re-estimates the service rate at
                      every decision from the measured utilisation,
                      mu_hat = throughput / (c * utilisation), and plans with it
"""
import numpy as np
from fast_env import ACTION_TO_C, C_MIN, C_MAX, LAMBDA_MAX
from model_policy import QueueingModelPolicy, expected_step_reward
from parameters import SERVER


class EnsemblePolicy:
    name = "ppo_ensemble"

    def __init__(self, members):
        self.members = members
        self.train_seed = ""

    def reset(self):
        pass

    def predict(self, obs, deterministic=True):
        cs = [ACTION_TO_C[int(m.predict(obs)[0])] for m in self.members]
        return ACTION_TO_C.index(int(np.median(cs))), None


class AdaptiveModelPolicy(QueueingModelPolicy):
    name = "adaptive_model"

    def __init__(self, network_mode, w_c, w_switch, deg_cong, **kw):
        super().__init__(network_mode, w_c=w_c, w_switch=w_switch, deg_cong=deg_cong, **kw)
        self.mode, self.w_c, self.deg_cong = network_mode, w_c, deg_cong
        self.mu_hat = SERVER["mu"]

    def reset(self):
        super().reset()
        self.mu_hat = SERVER["mu"]

    def predict(self, obs, deterministic=True):
        if len(obs) > 5:
            self.current_c = int(round(C_MIN + float(obs[5]) * (C_MAX - C_MIN)))
        thr = float(obs[3]) * LAMBDA_MAX
        util = float(obs[2])
        if thr > 1.0 and 0.0 < util < 0.999:
            self.mu_hat = thr / (self.current_c * util)
        lam = self.estimate_lambda(obs)
        scores = np.array([self.horizon * expected_step_reward(lam, c, self.mode, self.w_c,
                                                               self.deg_cong, mu=self.mu_hat)
                           for c in ACTION_TO_C])
        for a, c in enumerate(ACTION_TO_C):
            if c != self.current_c:
                scores[a] -= self.w_switch
        a = int(np.argmax(scores))
        self.current_c = ACTION_TO_C[a]
        return a, None


class HybridAdaptivePolicy:
    """Self-calibrating model by default; hands over to the PPO ensemble when the
    measured sojourn time exceeds the calibrated model's prediction (with mu_hat)
    by more than the threshold in n_on consecutive windows; returns after n_off
    consistent windows. Threshold calibrated under nominal conditions only."""
    name = "hybrid_adaptive"

    def __init__(self, network, ensemble, w_c, w_switch, deg_cong, threshold,
                 n_on=2, n_off=3):
        from model_policy import expected_sojourn
        self._pred = expected_sojourn
        self.network, self.deg_cong = network, deg_cong
        self.model = AdaptiveModelPolicy(network, w_c=w_c, w_switch=w_switch, deg_cong=deg_cong)
        self.ppo = ensemble
        self.threshold, self.n_on, self.n_off = threshold, n_on, n_off
        self.train_seed = ""
        self.reset()

    def reset(self):
        self.model.reset()
        self.use_ppo = False
        self.run_hi = self.run_lo = self.n_handover = self.ppo_steps = 0

    def ratio(self, obs):
        from fast_env import T_SLA
        c = int(round(C_MIN + float(obs[5]) * 3))
        thr = float(obs[3]) * LAMBDA_MAX
        util = float(obs[2])
        mu = thr / (c * util) if (thr > 1.0 and 0.0 < util < 0.999) else self.model.mu_hat
        lam = self.model.estimate_lambda(obs)
        return float(obs[1]) * T_SLA / self._pred(lam, c, self.network, self.deg_cong, mu=mu)

    def predict(self, obs, deterministic=True):
        alarm = self.ratio(obs) > self.threshold
        self.run_hi = self.run_hi + 1 if alarm else 0
        self.run_lo = 0 if alarm else self.run_lo + 1
        if not self.use_ppo and self.run_hi >= self.n_on:
            self.use_ppo = True
            self.n_handover += 1
        elif self.use_ppo and self.run_lo >= self.n_off:
            self.use_ppo = False
        a_model, _ = self.model.predict(obs)
        if self.use_ppo:
            self.ppo_steps += 1
            return self.ppo.predict(obs)[0], None
        return a_model, None
