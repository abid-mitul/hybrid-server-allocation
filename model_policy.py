"""
Model-based queueing policy (analytical baseline).

At every decision it
  1. estimates the current task arrival rate from the last observation window:
         lambda_hat = throughput / (1 - P_B)
  2. predicts, for each c in {3,4,5,6}, the expected one-step reward from the
     M/M/c/K stationary distribution and the congestion-coupled delay model
     (same reward as the environment, including server and switching costs),
  3. picks the c with the highest predicted reward over a planning horizon of
     H decisions, so a one-off switching cost is weighed against the gain it
     brings in the following minutes. H = 1 / (1 - gamma) = 10 for gamma = 0.9,
     the same effective horizon as the PPO agent.

It knows the true model structure and parameters (including the network type),
so it is the policy a queueing analyst would build. It needs no training.

The SLA term uses the approximation
    P(sojourn <= T) ~ sum_k q_k * P(D_k <= T - E[W | k] - 1/mu),
where q_k is the probability that an accepted task finds k tasks in the system
(PASTA), E[W | k] its mean wait and D_k the delay given the waiting tasks it
saw. Accuracy is checked in validate_model_policy() against simulation.
"""
import math

import numpy as np

from parameters import SERVER, LATENCY, SLA_AND_WEIGHTS
from fast_env import ACTION_TO_C, C_MIN, C_MAX, LAMBDA_MAX, T_SLA


def mmck_probs(lam, mu, c, K):
    """Stationary distribution p_0..p_K of M/M/c/K."""
    a = lam / mu
    logp = np.empty(K + 1)
    for k in range(K + 1):
        if k <= c:
            logp[k] = k * math.log(a) - math.lgamma(k + 1) if a > 0 else (0.0 if k == 0 else -np.inf)
        else:
            logp[k] = (c * math.log(a) - math.lgamma(c + 1) + (k - c) * math.log(a / c)
                       if a > 0 else -np.inf)
    logp -= logp.max()
    p = np.exp(logp)
    return p / p.sum()


def delay_cdf(t, mean_delay, mode, deg_cong):
    """P(D <= t) for the delay model used in fast_env (vectorised over arrays)."""
    t = np.asarray(t, dtype=float)
    d0 = LATENCY["delta_0"]
    alpha, pmin = LATENCY["pareto_shape"], LATENCY["pareto_min"]
    pos = np.maximum(t, 0.0)
    f_exp = np.where(t > 0, 1.0 - np.exp(-pos / mean_delay), 0.0)
    xm = pmin * (mean_delay / d0 if deg_cong else 1.0)
    f_par = np.where(t > xm, 1.0 - (xm / np.maximum(pos, 1e-12)) ** alpha, 0.0)
    if mode == "stable":
        return f_exp
    if mode == "degraded":
        return f_par
    return 0.5 * f_exp + 0.5 * f_par


def expected_step_reward(lam, c, mode, w_c, deg_cong, mu=None):
    """Predicted mean reward of one decision step at arrival rate lam with c servers
    (mu: service rate to assume; default the nominal rate)."""
    K = SERVER["K"]
    mu = SERVER["mu"] if mu is None else mu
    p = mmck_probs(lam, mu, c, K)
    p_b = p[K]
    acc = p[:K] / max(1.0 - p_b, 1e-12)            # what accepted arrivals see
    k = np.arange(K)
    n_wait = np.maximum(k - c, 0)                   # waiting tasks seen
    e_wait = np.where(k >= c, (k - c + 1) / (c * mu), 0.0)
    mean_delay = LATENCY["delta_0"] + LATENCY["eta"] * (n_wait / c) ** LATENCY["beta"]
    budget = T_SLA - e_wait - 1.0 / mu
    eta_d = float(np.sum(acc * delay_cdf(budget, mean_delay, mode, deg_cong)))
    eta_x = lam * (1.0 - p_b) / LAMBDA_MAX
    r = (SLA_AND_WEIGHTS["w_X"] * eta_x - SLA_AND_WEIGHTS["w_D"] * (1.0 - eta_d)
         - SLA_AND_WEIGHTS["w_B"] * p_b)
    return r - w_c * (c - C_MIN) / (C_MAX - C_MIN)


class QueueingModelPolicy:
    """Analytical policy: argmax of predicted one-step reward."""

    name = "queueing_model"

    def __init__(self, network_mode, w_c=0.0, w_switch=0.0, deg_cong=False,
                 lam_max=2500.0, lam_step=5.0, initial_c=4, horizon=10.0):
        self.w_switch = w_switch
        self.horizon = horizon
        self.initial_c = initial_c
        self.grid = np.arange(0.0, lam_max + lam_step, lam_step)
        # lookup table: predicted reward for every (lambda, c)
        self.table = np.array([[expected_step_reward(l, c, network_mode, w_c, deg_cong)
                                for c in ACTION_TO_C] for l in self.grid])
        self.current_c = initial_c

    def reset(self):
        self.current_c = self.initial_c

    def estimate_lambda(self, obs):
        thr = float(obs[3]) * LAMBDA_MAX
        p_b = min(float(obs[4]), 0.99)
        return thr / (1.0 - p_b)

    def predict(self, obs, deterministic=True):
        if len(obs) > 5:  # current c is part of the observation
            self.current_c = int(round(C_MIN + float(obs[5]) * (C_MAX - C_MIN)))
        lam = self.estimate_lambda(obs)
        i = int(min(round(lam / (self.grid[1] - self.grid[0])), len(self.grid) - 1))
        scores = self.horizon * self.table[i]
        for a, c in enumerate(ACTION_TO_C):
            if c != self.current_c:
                scores[a] -= self.w_switch
        a = int(np.argmax(scores))
        self.current_c = ACTION_TO_C[a]
        return a, None


def expected_sojourn(lam, c, mode, deg_cong, mu=None):
    """Model-predicted mean sojourn time (wait + service + delay) of accepted
    tasks at arrival rate lam with c servers (mu: service rate to assume)."""
    K = SERVER["K"]
    mu = SERVER["mu"] if mu is None else mu
    p = mmck_probs(lam, mu, c, K)
    acc = p[:K] / max(1.0 - p[K], 1e-12)
    k = np.arange(K)
    n_wait = np.maximum(k - c, 0)
    e_wait = np.where(k >= c, (k - c + 1) / (c * mu), 0.0)
    md = LATENCY["delta_0"] + LATENCY["eta"] * (n_wait / c) ** LATENCY["beta"]
    xi, pmin, d0 = LATENCY["pareto_shape"], LATENCY["pareto_min"], LATENCY["delta_0"]
    par = xi * pmin * (md / d0 if deg_cong else 1.0) / (xi - 1.0)
    e_delay = {"stable": md, "degraded": par, "noisy": 0.5 * md + 0.5 * par}[mode]
    return float(np.sum(acc * (e_wait + 1.0 / mu + e_delay)))
