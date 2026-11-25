"""Held-out structural departures (v1.3, task C3), defined before evaluation.

The three departures were specified in the v1.3 review package before any
controller was run on them, and they played no role in the design of CHAMPS.
Every controller still assumes the nominal model.

H1  Correlated service times.  On each server, successive tasks have
    S_j = (1/mu) * exp(Y_j - sigma^2/2) * E_j, E_j ~ Exp(1),
    Y_j = phi * Y_{j-1} + sigma * sqrt(1 - phi^2) * Z_j,  phi = 0.9, sigma = 0.7.
    The mean service time stays 1/mu; the correlation between consecutive
    tasks on the same server is the departure.
H2  Server failures.  Each active (allocated and working) server fails after an
    exponential time with mean 30 min and is down for an exponential time with
    mean 60 s.  A task in service at a failure returns to the head of the queue
    and restarts.  A failed server still counts as allocated (cost, c_k, and the
    utilisation denominator); the controller is not told about the failure.
H3  Periodic polling.  Each machine in state i issues tasks periodically with
    period 1/lambda_i, a uniform random phase and +/-10% uniform jitter on every
    interval; failed machines issue none.  A machine that changes condition
    restarts its cycle with a new random phase at the new rate.

Implementation: a separate event kernel (_advance_h) that reproduces the
nominal kernel of fast_env.py when all three options are off; the population
size is fixed (no bursts) in every held-out scenario.
"""
import numpy as np
from fast_env import (FastFactoryGymEnv, njit, _svc, _sample_delay, C_MAX, T_SLA,
                      S_T, S_N0, S_N1, S_N2, S_NEXT_CTMC, S_NEXT_ARR, S_QHEAD, S_QLEN, S_BUSY,
                      B_OFFERED, B_DROPS, B_COMP, B_SLA, B_SOJ, B_WAIT, B_DELAY, B_SVC)

PHI, SIGMA = 0.9, 0.7
MTTF, MTTR = 1800.0, 60.0
JITTER = 0.10

# extra state (float64 array): number of failed servers, next failure time,
# number of failures in the episode
H_DOWN, H_NEXT_FAIL, H_NFAIL = range(3)


@njit(cache=True)
def _svc_h(j, mu, h1, srv_y, phi, sigma):
    if not h1:
        return np.random.exponential(1.0 / mu)
    y = phi * srv_y[j] + sigma * np.sqrt(1.0 - phi * phi) * np.random.standard_normal()
    srv_y[j] = y
    return np.exp(y - 0.5 * sigma * sigma) * np.random.exponential(1.0 / mu)


@njit(cache=True)
def _fail_clock(t, c, n_down, mttf):
    up = c - n_down
    return t + np.random.exponential(mttf / up) if up > 0 else np.inf


@njit(cache=True)
def _next_poll(m_next):
    jm = 0
    tm = m_next[0]
    for m in range(1, m_next.shape[0]):
        if m_next[m] < tm:
            tm = m_next[m]
            jm = m
    return jm, tm


@njit(cache=True)
def _advance_h(state, hs, srv_dep, srv_arr, srv_start, srv_seen, srv_c, srv_y, rep,
               q_arr, q_seen, q_c, m_state, m_next, bins, t_end, c, step_dur,
               K, mu, lam0, lam1, a01, a12, b10, b21,
               mode, delta0, eta, beta, p_shape, p_min, deg_cong, t_sla,
               h1, phi, sigma, h2, mttf, mttr, h3, jitter):
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
    n_down = int(hs[H_DOWN])
    next_fail = hs[H_NEXT_FAIL]
    if h2:  # the allocation may have changed: the clock is memoryless, resample
        next_fail = _fail_clock(t, c, n_down, mttf)

    # start waiting tasks if capacity allows
    c_eff = c - n_down if c - n_down > 0 else 0
    while qlen > 0 and busy < c_eff:
        for j in range(n_srv):
            if srv_dep[j] == np.inf:
                break
        srv_arr[j] = q_arr[qhead]
        srv_seen[j] = q_seen[qhead]
        srv_c[j] = q_c[qhead]
        srv_start[j] = t
        srv_dep[j] = t + _svc_h(j, mu, h1, srv_y, phi, sigma)
        qhead = (qhead + 1) % K
        qlen -= 1
        busy += 1

    while True:
        jd = 0
        t_dep = srv_dep[0]
        for j in range(1, n_srv):
            if srv_dep[j] < t_dep:
                t_dep = srv_dep[j]
                jd = j
        jr = 0
        t_rep = rep[0]
        for j in range(1, n_srv):
            if rep[j] < t_rep:
                t_rep = rep[j]
                jr = j
        jm = -1
        if h3:
            jm, next_arr = _next_poll(m_next)

        t_next = t_end
        ev = 0  # 0 end, 1 ctmc, 2 arrival, 3 departure, 6 failure, 7 repair
        if next_ctmc < t_next:
            t_next = next_ctmc
            ev = 1
        if next_arr < t_next:
            t_next = next_arr
            ev = 2
        if t_dep < t_next:
            t_next = t_dep
            ev = 3
        if next_fail < t_next:
            t_next = next_fail
            ev = 6
        if t_rep < t_next:
            t_next = t_rep
            ev = 7
        if ev == 0:
            t = t_end
            break
        t = t_next

        if ev == 1:
            r0 = n0 * a01
            r1 = n1 * (a12 + b10)
            r2 = n2 * b21
            u = np.random.random() * (r0 + r1 + r2)
            if u < r0:
                s_from, s_to = 0, 1
            elif u < r0 + r1:
                if np.random.random() < b10 / (a12 + b10):
                    s_from, s_to = 1, 0
                else:
                    s_from, s_to = 1, 2
            else:
                s_from, s_to = 2, 1
            if s_from == 0:
                n0 -= 1
            elif s_from == 1:
                n1 -= 1
            else:
                n2 -= 1
            if s_to == 0:
                n0 += 1
            elif s_to == 1:
                n1 += 1
            else:
                n2 += 1
            tot = n0 * a01 + n1 * (a12 + b10) + n2 * b21
            next_ctmc = t + np.random.exponential(1.0 / tot) if tot > 0 else np.inf
            if h3:
                # the changing machine restarts its cycle with a new random phase
                nm = m_state.shape[0]
                while True:
                    m = np.random.randint(0, nm)
                    if m_state[m] == s_from:
                        break
                m_state[m] = s_to
                if s_to == 0:
                    m_next[m] = t + np.random.random() / lam0
                elif s_to == 1:
                    m_next[m] = t + np.random.random() / lam1
                else:
                    m_next[m] = np.inf
            else:
                lam = lam0 * n0 + lam1 * n1
                next_arr = t + np.random.exponential(1.0 / lam) if lam > 0 else np.inf

        elif ev == 2:
            b = int(t / step_dur)
            if b < n_bins:
                bins[b, B_OFFERED] += 1.0
            if busy + qlen >= K:
                if b < n_bins:
                    bins[b, B_DROPS] += 1.0
            else:
                seen = qlen
                c_eff = c - n_down if c - n_down > 0 else 0
                if busy < c_eff:
                    for j in range(n_srv):
                        if srv_dep[j] == np.inf:
                            break
                    srv_arr[j] = t
                    srv_seen[j] = seen
                    srv_c[j] = c
                    srv_start[j] = t
                    srv_dep[j] = t + _svc_h(j, mu, h1, srv_y, phi, sigma)
                    busy += 1
                else:
                    tail = (qhead + qlen) % K
                    q_arr[tail] = t
                    q_seen[tail] = seen
                    q_c[tail] = c
                    qlen += 1
            if h3:
                period = 1.0 / lam0 if m_state[jm] == 0 else 1.0 / lam1
                m_next[jm] = t + period * (1.0 - jitter + 2.0 * jitter * np.random.random())
            else:
                lam = lam0 * n0 + lam1 * n1
                next_arr = t + np.random.exponential(1.0 / lam) if lam > 0 else np.inf

        elif ev == 6:
            # an active working server fails
            up = c - n_down
            hs[H_NFAIL] += 1.0
            if busy > 0 and np.random.random() < min(busy, up) / up:
                # the failed server was busy: its task returns to the queue head
                k = np.random.randint(0, busy)
                for j in range(n_srv):
                    if srv_dep[j] != np.inf:
                        if k == 0:
                            break
                        k -= 1
                bt = int(t / step_dur)
                if bt < n_bins:
                    bins[bt, B_SVC] += t - srv_start[j]
                qhead = (qhead - 1) % K
                q_arr[qhead] = srv_arr[j]
                q_seen[qhead] = srv_seen[j]
                q_c[qhead] = srv_c[j]
                qlen += 1
                srv_dep[j] = np.inf
                busy -= 1
            n_down += 1
            for j in range(n_srv):
                if rep[j] == np.inf:
                    rep[j] = t + np.random.exponential(mttr)
                    break
            next_fail = _fail_clock(t, c, n_down, mttf)

        elif ev == 7:
            rep[jr] = np.inf
            n_down -= 1
            next_fail = _fail_clock(t, c, n_down, mttf)
            c_eff = c - n_down if c - n_down > 0 else 0
            while qlen > 0 and busy < c_eff:
                for j in range(n_srv):
                    if srv_dep[j] == np.inf:
                        break
                srv_arr[j] = q_arr[qhead]
                srv_seen[j] = q_seen[qhead]
                srv_c[j] = q_c[qhead]
                srv_start[j] = t
                srv_dep[j] = t + _svc_h(j, mu, h1, srv_y, phi, sigma)
                qhead = (qhead + 1) % K
                qlen -= 1
                busy += 1

        else:
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
            c_eff = c - n_down if c - n_down > 0 else 0
            if qlen > 0 and busy < c_eff:
                srv_arr[j] = q_arr[qhead]
                srv_seen[j] = q_seen[qhead]
                srv_c[j] = q_c[qhead]
                srv_start[j] = t
                srv_dep[j] = t + _svc_h(j, mu, h1, srv_y, phi, sigma)
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
    hs[H_DOWN] = n_down
    hs[H_NEXT_FAIL] = next_fail


class HeldoutEnv(FastFactoryGymEnv):
    """FastFactoryGymEnv with the held-out departure `heldout` in {"H1","H2","H3"}."""

    def __init__(self, heldout=None, **kw):
        for k in ("burst_n_end", "burst_factors", "service_burst", "server_speeds"):
            assert kw.get(k) is None, f"{k} is not supported by the held-out kernel"
        assert kw.get("batch_mean", 1.0) == 1.0 and kw.get("svc_scv", 1.0) == 1.0
        self.heldout = heldout
        super().__init__(**kw)

    def _resample_clocks(self):
        # called once by reset() before the first window: set up the extra state
        super()._resample_clocks()
        n0, n1, n2 = (int(self.state[i]) for i in (S_N0, S_N1, S_N2))
        self.hs = np.zeros(3)
        self.hs[H_NEXT_FAIL] = np.inf
        self.srv_y = self._rng.normal(0.0, SIGMA, C_MAX)     # stationary AR(1) start
        self.rep = np.full(C_MAX, np.inf)
        if self.heldout == "H3":
            self.m_state = self._rng.permutation(np.repeat([0, 1, 2], [n0, n1, n2])).astype(np.int64)
            per = np.where(self.m_state == 0, 1.0 / self.lam0,
                           np.where(self.m_state == 1, 1.0 / self.lam1, np.inf))
            self.m_next = np.where(np.isfinite(per), self._rng.random(len(per)) * per, np.inf)
        else:
            self.m_state = np.zeros(1, dtype=np.int64)
            self.m_next = np.full(1, np.inf)

    def _run_until(self, t_end):
        d0, eta, beta, ps, pm = self.lat
        h = self.heldout
        _advance_h(self.state, self.hs, self.srv_dep, self.srv_arr, self.srv_start,
                   self.srv_seen, self.srv_c, self.srv_y, self.rep,
                   self.q_arr, self.q_seen, self.q_c, self.m_state, self.m_next,
                   self.bins, float(t_end), int(self.current_c), self.step_duration,
                   self.K, self.mu, self.lam0, self.lam1,
                   self.a01, self.a12, self.b10, self.b21,
                   self.mode, d0, eta, beta, ps, pm, self.deg_cong, T_SLA,
                   h == "H1", PHI, SIGMA, h == "H2", MTTF, MTTR, h == "H3", JITTER)

    @property
    def n_failures(self):
        return int(self.hs[H_NFAIL])


HELDOUT_LOADS = (200, 280)
HELDOUT_TYPES = ("H1", "H2", "H3")
HELDOUT_SEED_BASE = 30000      # disjoint from training, calibration (5000-5009),
                               # main evaluation (1000-1019) and C4 (20000-20111)


def heldout_scenarios(net):
    return {f"{h.lower()}_{N}_{net}": dict(heldout=h, n_machines=N, network=net)
            for h in HELDOUT_TYPES for N in HELDOUT_LOADS}


def make_heldout_env(heldout, n_machines, network_mode, opt, step_duration=60.0,
                     episode_duration=1800.0):
    return HeldoutEnv(heldout=heldout, n_machines=n_machines, network_mode=network_mode,
                      episode_duration=episode_duration, step_duration=step_duration,
                      measured_util=True, **opt)
