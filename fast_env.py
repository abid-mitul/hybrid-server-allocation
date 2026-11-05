"""
Event-driven simulator of the industrial multi-server system:
CTMC machine conditions + M/M/c/K server pool + congestion-coupled delay.

  * The machine population is simulated by its occupancy counts (n0, n1, n2).
    Because all machines follow the same CTMC, the counts form an exact
    continuous-time Markov chain (lumpability), so this is exact, not an
    approximation, and much faster than one process per machine.
  * Task arrivals are a Poisson process with rate lambda_0*n0 + lambda_1*n1,
    resampled whenever a machine changes condition (memoryless property).
  * The FCFS c-server pool with capacity K is an event-driven loop compiled
    with numba (plain Python is used if numba is unavailable).
  * Changes of c are non-preemptive: tasks in service finish, and new tasks
    start only while the number of busy servers is below c.
  * The blocking probability is measured as drops / offered tasks.

Model options (all off by default):
  * init_state="stationary": initial machine mix drawn from pi*.
  * burst_n_end / burst_time: the population changes during an episode.
  * w_c, w_switch: server and reconfiguration costs in the reward.
  * degraded_congestion: Pareto delay scaled by the congestion term.
  * mu_scale, eta_scale, fail_scale: true parameters differ from the nominal
    ones assumed by the controllers (parameter mismatch).
  * svc_scv, server_speeds, burst_factors, batch_mean, service_burst:
    departures from the structure of the model (structural mismatch).
  * measured_util: utilisation computed from recorded processing times.

Author: Md Abid Hassan Mitul
"""

import math
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from parameters import CTMC, TASK_GENERATION, SERVER, LATENCY, SLA_AND_WEIGHTS

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # plain-Python fallback (much slower, same results model)
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
ACTION_TO_C = (3, 4, 5, 6)
C_MAX = max(ACTION_TO_C)
C_MIN = min(ACTION_TO_C)
LAMBDA_MAX = 350 * TASK_GENERATION["lambda_0"]
T_SLA = SLA_AND_WEIGHTS["T_SLA"]
NETWORK_CODE = {"stable": 0, "degraded": 1, "noisy": 2}

# State vector layout (float64 array, persists between kernel calls)
(S_T, S_N0, S_N1, S_N2, S_NEXT_CTMC, S_NEXT_ARR, S_QHEAD, S_QLEN, S_BUSY, S_MOD, S_NEXT_MOD,
 S_SMOD, S_NEXT_SMOD) = range(13)
S_SIZE = 13

# Per-step bin layout (columns of the bins array)
B_OFFERED, B_DROPS, B_COMP, B_SLA, B_SOJ, B_WAIT, B_DELAY, B_SVC = range(8)
B_SIZE = 8


@njit(cache=True)
def _seed(seed):
    np.random.seed(seed)


@njit(cache=True)
def _sample_delay(mode, seen_q, c, delta0, eta, beta, p_shape, p_min, deg_cong):
    mean_delay = delta0 + eta * (seen_q / c) ** beta
    use_pareto = False
    if mode == 1:
        use_pareto = True
    elif mode == 2:
        use_pareto = np.random.random() < 0.5
    if not use_pareto:
        return np.random.exponential(mean_delay)
    u = np.random.random()
    x = p_min * (1.0 - u) ** (-1.0 / p_shape)
    if deg_cong:
        # optional: scale the heavy-tailed delay by the same congestion factor
        x = x * (mean_delay / delta0)
    return x


@njit(cache=True)
def _arrival_rate(n0, n1, lam0, lam1):
    return lam0 * n0 + lam1 * n1


@njit(cache=True)
def _svc(mu, speed, scv):
    """Service time. scv == 1: exponential (nominal model). scv > 1: two-phase
    hyperexponential with balanced means, same mean 1/(mu*speed), given SCV."""
    rate = mu * speed
    if scv == 1.0:
        return np.random.exponential(1.0 / rate)
    p1 = 0.5 * (1.0 + np.sqrt((scv - 1.0) / (scv + 1.0)))
    if np.random.random() < p1:
        return np.random.exponential(1.0 / (2.0 * p1 * rate))
    return np.random.exponential(1.0 / (2.0 * (1.0 - p1) * rate))


@njit(cache=True)
def _advance(state, srv_dep, srv_arr, srv_start, srv_seen, srv_c,
             q_arr, q_seen, q_c, bins, t_end, c, step_dur,
             K, mu, lam0, lam1, a01, a12, b10, b21,
             mode, delta0, eta, beta, p_shape, p_min, deg_cong, t_sla,
             svc_scv, srv_speed, mod_hi, mod_lo, mod_rate,
             batch_mean, smod_a, smod_b, smod_rate):
    """Advance the simulation from state[S_T] to t_end with c active servers."""
    n_srv = srv_dep.shape[0]
    n_bins = bins.shape[0]
    t = state[S_T]
    n0 = state[S_N0]
    n1 = state[S_N1]
    n2 = state[S_N2]
    next_ctmc = state[S_NEXT_CTMC]
    next_arr = state[S_NEXT_ARR]
    qhead = int(state[S_QHEAD])
    qlen = int(state[S_QLEN])
    busy = int(state[S_BUSY])
    mod = state[S_MOD]
    next_mod = state[S_NEXT_MOD]
    smod = state[S_SMOD]
    next_smod = state[S_NEXT_SMOD]

    # If c was increased, waiting tasks may start right now
    while qlen > 0 and busy < c:
        for j in range(n_srv):
            if srv_dep[j] == np.inf:
                break
        srv_arr[j] = q_arr[qhead]
        srv_seen[j] = q_seen[qhead]
        srv_c[j] = q_c[qhead]
        srv_start[j] = t
        srv_dep[j] = t + _svc(mu, srv_speed[j] / smod, svc_scv)
        qhead = (qhead + 1) % K
        qlen -= 1
        busy += 1

    while True:
        # next departure
        jd = 0
        t_dep = srv_dep[0]
        for j in range(1, n_srv):
            if srv_dep[j] < t_dep:
                t_dep = srv_dep[j]
                jd = j

        t_next = t_end
        ev = 0  # 0=end, 1=ctmc, 2=arrival, 3=departure
        if next_ctmc < t_next:
            t_next = next_ctmc
            ev = 1
        if next_arr < t_next:
            t_next = next_arr
            ev = 2
        if t_dep < t_next:
            t_next = t_dep
            ev = 3
        if next_mod < t_next:
            t_next = next_mod
            ev = 4
        if next_smod < t_next:
            t_next = next_smod
            ev = 5
        if ev == 0:
            t = t_end
            break
        t = t_next

        if ev == 1:
            # one machine changes state (lumped CTMC on counts)
            r0 = n0 * a01
            r1 = n1 * (a12 + b10)
            r2 = n2 * b21
            tot = r0 + r1 + r2
            u = np.random.random() * tot
            if u < r0:
                n0 -= 1
                n1 += 1
            elif u < r0 + r1:
                if np.random.random() < b10 / (a12 + b10):
                    n1 -= 1
                    n0 += 1
                else:
                    n1 -= 1
                    n2 += 1
            else:
                n2 -= 1
                n1 += 1
            tot = n0 * a01 + n1 * (a12 + b10) + n2 * b21
            next_ctmc = t + np.random.exponential(1.0 / tot) if tot > 0 else np.inf
            lam = mod * _arrival_rate(n0, n1, lam0, lam1) / batch_mean
            next_arr = t + np.random.exponential(1.0 / lam) if lam > 0 else np.inf

        elif ev == 2:
            # batch arrivals: a batch of related tasks arrives together
            nb = 1 if batch_mean == 1.0 else np.random.geometric(1.0 / batch_mean)
            for _ib in range(nb):
                b = int(t / step_dur)
                if b < n_bins:
                    bins[b, B_OFFERED] += 1.0
                if busy + qlen >= K:
                    if b < n_bins:
                        bins[b, B_DROPS] += 1.0
                else:
                    seen = qlen  # waiting tasks ahead (excludes itself)
                    if busy < c:
                        for j in range(n_srv):
                            if srv_dep[j] == np.inf:
                                break
                        srv_arr[j] = t
                        srv_seen[j] = seen
                        srv_c[j] = c
                        srv_start[j] = t
                        srv_dep[j] = t + _svc(mu, srv_speed[j] / smod, svc_scv)
                        busy += 1
                    else:
                        tail = (qhead + qlen) % K
                        q_arr[tail] = t
                        q_seen[tail] = seen
                        q_c[tail] = c
                        qlen += 1
            lam = mod * _arrival_rate(n0, n1, lam0, lam1) / batch_mean
            next_arr = t + np.random.exponential(1.0 / lam) if lam > 0 else np.inf

        elif ev == 5:
            # bursty service demand: processing-time multiplier switches level
            smod = smod_b if smod == smod_a else smod_a
            next_smod = t + np.random.exponential(1.0 / smod_rate)

        elif ev == 4:
            # bursty traffic: the common sending-rate factor switches level
            mod = mod_lo if mod == mod_hi else mod_hi
            next_mod = t + np.random.exponential(1.0 / mod_rate)
            lam = mod * _arrival_rate(n0, n1, lam0, lam1) / batch_mean
            next_arr = t + np.random.exponential(1.0 / lam) if lam > 0 else np.inf

        else:
            # service completion at server jd; delay does not hold the server
            j = jd
            delay = _sample_delay(mode, srv_seen[j], srv_c[j], delta0, eta,
                                  beta, p_shape, p_min, deg_cong)
            complete = t + delay
            b = int(complete / step_dur)
            if b < n_bins:
                soj = complete - srv_arr[j]
                bins[b, B_COMP] += 1.0
                if soj <= t_sla:
                    bins[b, B_SLA] += 1.0
                bins[b, B_SOJ] += soj
                bins[b, B_WAIT] += srv_start[j] - srv_arr[j]
                bins[b, B_DELAY] += delay
            bt = int(t / step_dur)
            if bt < n_bins:
                bins[bt, B_SVC] += t - srv_start[j]
            srv_dep[j] = np.inf
            busy -= 1
            if qlen > 0 and busy < c:
                srv_arr[j] = q_arr[qhead]
                srv_seen[j] = q_seen[qhead]
                srv_c[j] = q_c[qhead]
                srv_start[j] = t
                srv_dep[j] = t + _svc(mu, srv_speed[j] / smod, svc_scv)
                qhead = (qhead + 1) % K
                qlen -= 1
                busy += 1

    state[S_T] = t
    state[S_N0] = n0
    state[S_N1] = n1
    state[S_N2] = n2
    state[S_NEXT_CTMC] = next_ctmc
    state[S_NEXT_ARR] = next_arr
    state[S_QHEAD] = qhead
    state[S_QLEN] = qlen
    state[S_BUSY] = busy
    state[S_MOD] = mod
    state[S_NEXT_MOD] = next_mod
    state[S_SMOD] = smod
    state[S_NEXT_SMOD] = next_smod


class FastFactoryGymEnv(gym.Env):
    """Same interface, observation, action and reward as FactoryGymEnv."""

    metadata = {"render_modes": []}
    ACTION_TO_C = {i: c for i, c in enumerate(ACTION_TO_C)}

    def __init__(
        self,
        n_machines: int = 200,
        network_mode: str = "stable",
        episode_duration: float = 1800.0,
        step_duration: float = 60.0,
        initial_c: int = 4,
        seed: Optional[int] = None,
        init_state: str = "operational",
        burst_n_end: Optional[int] = None,
        burst_time: Optional[float] = None,
        w_c: float = 0.0,
        degraded_congestion: bool = False,
        observe_first: bool = False,
        obs_include_c: bool = False,
        w_switch: float = 0.0,
        mu_scale: float = 1.0,
        eta_scale: float = 1.0,
        fail_scale: float = 1.0,
        svc_scv: float = 1.0,
        server_speeds=None,
        burst_factors=None,
        burst_switch_mean: float = 10.0,
        measured_util: bool = False,
        batch_mean: float = 1.0,
        service_burst=None,
        service_switch_mean: float = 10.0,
    ):
        super().__init__()
        # n_machines may be a list: a load level is then drawn at every reset
        # (training across loads, so the policy has a reason to adapt c)
        if isinstance(n_machines, (list, tuple)):
            self.n_choices = [int(x) for x in n_machines]
        else:
            self.n_choices = [int(n_machines)]
        self.n_machines_start = self.n_choices[0]
        self.network_mode = network_mode
        self.mode = NETWORK_CODE[network_mode]
        self.episode_duration = float(episode_duration)
        self.step_duration = float(step_duration)
        self.initial_c = initial_c
        self.n_steps_per_episode = int(episode_duration // step_duration)
        self.init_state = init_state
        self.burst_n_end = burst_n_end
        self.burst_time = burst_time if burst_time is not None else episode_duration / 2
        self.w_c = float(w_c)
        self.deg_cong = bool(degraded_congestion)
        # observe_first: run one observation window at initial_c before the
        # first decision, so the agent never acts on an empty observation
        self.offset = 1 if observe_first else 0

        # obs_include_c: append the server count currently in force (scaled to
        # [0, 1]); w_switch: reward penalty each time c is changed
        self.obs_include_c = bool(obs_include_c)
        self.w_switch = float(w_switch)
        self.obs_dim = 6 if self.obs_include_c else 5
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.obs_dim,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTION_TO_C))

        self.w_X = SLA_AND_WEIGHTS["w_X"]
        self.w_D = SLA_AND_WEIGHTS["w_D"]
        self.w_B = SLA_AND_WEIGHTS["w_B"]
        self.K = int(SERVER["K"])
        # mu_scale / eta_scale: the TRUE system differs from the nominal
        # parameters (model-mismatch tests); policies still assume nominal values
        self.mu = float(SERVER["mu"]) * float(mu_scale)
        self.lam0 = float(TASK_GENERATION["lambda_0"])
        self.lam1 = float(TASK_GENERATION["lambda_1"])
        # CTMC rates converted from 1/hour to 1/second
        # fail_scale: machines degrade and fail faster than the nominal rates
        # (model-mismatch test); repair rates are unchanged
        self.fail_scale = float(fail_scale)
        self.a01 = CTMC["alpha_01"] * self.fail_scale / 3600.0
        self.a12 = CTMC["alpha_12"] * self.fail_scale / 3600.0
        self.b10 = CTMC["beta_10"] / 3600.0
        self.b21 = CTMC["beta_21"] / 3600.0
        self.lat = (LATENCY["delta_0"], LATENCY["eta"] * float(eta_scale), LATENCY["beta"],
                    LATENCY["pareto_shape"], LATENCY["pareto_min"])

        # structural-mismatch options (all off by default = nominal model)
        self.svc_scv = float(svc_scv)
        self.srv_speed = (np.ones(C_MAX) if server_speeds is None
                          else np.array(server_speeds, dtype=np.float64))
        self.burst_factors = burst_factors
        self.mod_rate = 1.0 / float(burst_switch_mean)
        self.measured_util = bool(measured_util)
        self.batch_mean = float(batch_mean)
        self.service_burst = service_burst
        self.smod_rate = 1.0 / float(service_switch_mean)

        self._rng = np.random.default_rng(seed)
        self._seeded = False
        if seed is not None:
            _seed(int(seed))
            self._seeded = True
        self.current_c = initial_c
        self.step_count = 0
        self._burst_done = False

    # ---------------- internals ----------------
    def _initial_counts(self, n):
        if self.init_state == "stationary":
            if self.fail_scale == 1.0:
                p = np.array(CTMC["pi_star_theoretical"], dtype=float)
            else:  # stationary distribution of the true (scaled) chain
                Q = np.array([[-self.a01, self.a01, 0.0],
                              [self.b10, -(self.b10 + self.a12), self.a12],
                              [0.0, self.b21, -self.b21]])
                A = np.vstack([Q.T, np.ones(3)])
                p = np.linalg.lstsq(A, np.array([0.0, 0.0, 0.0, 1.0]), rcond=None)[0]
                p = np.clip(p, 0.0, None)
            p = p / p.sum()
            return self._rng.multinomial(n, p)
        return np.array([n, 0, 0])

    def _resample_clocks(self):
        s = self.state
        t = s[S_T]
        n0, n1, n2 = s[S_N0], s[S_N1], s[S_N2]
        tot = n0 * self.a01 + n1 * (self.a12 + self.b10) + n2 * self.b21
        s[S_NEXT_CTMC] = t + self._rng.exponential(1.0 / tot) if tot > 0 else np.inf
        lam = s[S_MOD] * (self.lam0 * n0 + self.lam1 * n1) / self.batch_mean
        s[S_NEXT_ARR] = t + self._rng.exponential(1.0 / lam) if lam > 0 else np.inf

    def _run_until(self, t_end):
        d0, eta, beta, ps, pm = self.lat
        _advance(self.state, self.srv_dep, self.srv_arr, self.srv_start,
                 self.srv_seen, self.srv_c, self.q_arr, self.q_seen, self.q_c,
                 self.bins, float(t_end), int(self.current_c), self.step_duration,
                 self.K, self.mu, self.lam0, self.lam1,
                 self.a01, self.a12, self.b10, self.b21,
                 self.mode, d0, eta, beta, ps, pm, self.deg_cong, T_SLA,
                 self.svc_scv, self.srv_speed, self._mod_hi, self._mod_lo, self.mod_rate,
                 self.batch_mean, self._smod_a, self._smod_b, self.smod_rate)

    # ---------------- gym API ----------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            _seed(int(seed))
        elif not self._seeded:
            _seed(int(self._rng.integers(0, 2**31 - 1)))
            self._seeded = True
        self.current_c = self.initial_c
        self.step_count = 0
        self._burst_done = False

        self.state = np.zeros(S_SIZE, dtype=np.float64)
        if len(self.n_choices) > 1:
            self.n_machines_start = int(self._rng.choice(self.n_choices))
        n0, n1, n2 = self._initial_counts(self.n_machines_start)
        self.state[S_N0], self.state[S_N1], self.state[S_N2] = n0, n1, n2
        if self.burst_factors is None:
            self._mod_hi = self._mod_lo = 1.0
            self.state[S_MOD], self.state[S_NEXT_MOD] = 1.0, np.inf
        else:
            self._mod_hi, self._mod_lo = (float(x) for x in self.burst_factors)
            self.state[S_MOD] = self._mod_hi if self._rng.random() < 0.5 else self._mod_lo
            self.state[S_NEXT_MOD] = self._rng.exponential(1.0 / self.mod_rate)
        if self.service_burst is None:
            self._smod_a = self._smod_b = 1.0
            self.state[S_SMOD], self.state[S_NEXT_SMOD] = 1.0, np.inf
        else:
            self._smod_a, self._smod_b = (float(x) for x in self.service_burst)
            self.state[S_SMOD] = self._smod_a if self._rng.random() < 0.5 else self._smod_b
            self.state[S_NEXT_SMOD] = self._rng.exponential(1.0 / self.smod_rate)
        self.srv_dep = np.full(C_MAX, np.inf)
        self.srv_arr = np.zeros(C_MAX)
        self.srv_start = np.zeros(C_MAX)
        self.srv_seen = np.zeros(C_MAX)
        self.srv_c = np.ones(C_MAX)
        self.q_arr = np.zeros(self.K)
        self.q_seen = np.zeros(self.K)
        self.q_c = np.ones(self.K)
        self.bins = np.zeros((self.n_steps_per_episode + 1 + self.offset, B_SIZE))
        self._resample_clocks()
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        if self.obs_include_c:
            obs[5] = (self.current_c - C_MIN) / (C_MAX - C_MIN)
        if self.offset:
            self._run_until(self.step_duration)
            obs = self._observe(self.bins[0])[0]
        return obs, {"c": self.current_c, "step": 0}

    def _observe(self, row):
        offered, drops, n_comp = row[B_OFFERED], row[B_DROPS], row[B_COMP]
        mean_sojourn = row[B_SOJ] / n_comp if n_comp > 0 else 0.0
        throughput = n_comp / self.step_duration
        p_b = drops / offered if offered > 0 else 0.0
        obs = np.array([
            min(self.state[S_QLEN] / self.K, 1.0),
            min(mean_sojourn / T_SLA, 1.0),
            min((row[B_SVC] / (self.current_c * self.step_duration)) if self.measured_util
                else throughput / (self.current_c * self.mu), 1.0),
            min(throughput / LAMBDA_MAX, 1.0),
            p_b,
        ], dtype=np.float32)
        if self.obs_include_c:
            obs = np.append(obs, np.float32((self.current_c - C_MIN) / (C_MAX - C_MIN)))
        return np.clip(obs, 0.0, 1.0), mean_sojourn, throughput, p_b

    def step(self, action: int):
        prev_c = self.current_c
        self.current_c = self.ACTION_TO_C[int(action)]
        k = self.step_count + self.offset
        t_end = min((k + 1) * self.step_duration,
                    self.episode_duration + self.offset * self.step_duration)
        burst_time = self.burst_time + self.offset * self.step_duration

        # optional burst: extra machines join (Operational) at burst_time
        if (self.burst_n_end is not None and not self._burst_done
                and burst_time < t_end):
            self._run_until(max(burst_time, self.state[S_T]))
            extra = self.burst_n_end - self.n_machines_start
            if extra >= 0:  # new machines join Operational
                self.state[S_N0] += extra
            else:           # machines leave, drawn at random across states
                counts = np.array([self.state[S_N0], self.state[S_N1],
                                   self.state[S_N2]], dtype=np.int64)
                gone = self._rng.multivariate_hypergeometric(counts, -extra)
                self.state[S_N0] -= gone[0]
                self.state[S_N1] -= gone[1]
                self.state[S_N2] -= gone[2]
            self._resample_clocks()
            self._burst_done = True

        self._run_until(t_end)

        row = self.bins[k]
        offered, drops, n_comp = row[B_OFFERED], row[B_DROPS], row[B_COMP]
        obs, mean_sojourn, throughput, p_b = self._observe(row)
        eta_D = row[B_SLA] / n_comp if n_comp > 0 else 1.0
        eta_X = throughput / LAMBDA_MAX
        queue_len = int(self.state[S_QLEN])

        reward = self.w_X * eta_X - self.w_D * (1 - eta_D) - self.w_B * p_b
        if self.w_switch > 0 and self.current_c != prev_c:
            reward -= self.w_switch
        if self.w_c > 0:
            reward -= self.w_c * (self.current_c - C_MIN) / (C_MAX - C_MIN)

        self.step_count += 1
        done = self.step_count >= self.n_steps_per_episode
        info = {
            "c": self.current_c,
            "step": self.step_count,
            "throughput": throughput,
            "mean_sojourn": mean_sojourn,
            "mean_wait": row[B_WAIT] / n_comp if n_comp > 0 else 0.0,
            "mean_delay": row[B_DELAY] / n_comp if n_comp > 0 else 0.0,
            "p_b": p_b,
            "sla_compliance": eta_D,
            "queue_length": queue_len,
            "n_arrivals": int(offered),
            "n_completions": int(n_comp),
            "n_drops": int(drops),
            "n_machines": int(self.state[S_N0] + self.state[S_N1] + self.state[S_N2]),
            "machine_counts": (int(self.state[S_N0]), int(self.state[S_N1]),
                               int(self.state[S_N2])),
        }
        return obs, float(reward), done, False, info

    def close(self):
        pass


def make_fast_env(n_machines=200, network_mode="stable", episode_duration=1800.0,
                  step_duration=60.0, seed=None, **kwargs) -> FastFactoryGymEnv:
    return FastFactoryGymEnv(n_machines=n_machines, network_mode=network_mode,
                             episode_duration=episode_duration,
                             step_duration=step_duration, seed=seed, **kwargs)


if __name__ == "__main__":
    import time
    print(f"numba available: {HAVE_NUMBA}")
    env = make_fast_env(n_machines=200, network_mode="stable",
                        episode_duration=300, step_duration=60, seed=42)
    env.reset(seed=42)
    env.step(1)  # compile
    t0 = time.time()
    n = 0
    for ep in range(20):
        env.reset()
        done = False
        while not done:
            _, r, done, _, info = env.step(env.action_space.sample())
            n += 1
    dt = time.time() - t0
    print(f"{n} RL steps in {dt:.2f}s  ->  {1000 * dt / n:.1f} ms per step")


def env_options(train_cfg: dict) -> dict:
    """Model options from the 'train' block of experiment_config.json."""
    return dict(init_state=train_cfg.get("init_state", "operational"),
                w_c=train_cfg.get("w_c", 0.0),
                degraded_congestion=train_cfg.get("degraded_congestion", False),
                observe_first=train_cfg.get("observe_first", False),
                obs_include_c=train_cfg.get("obs_include_c", False),
                w_switch=train_cfg.get("w_switch", 0.0))
