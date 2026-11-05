"""
Reference implementation of the system model in SimPy (one process per
machine, one event chain per task). It is slower than fast_env.py and is used
only to validate it (validate_fast_env.py).

Implements:
  - CTMC machine population (N machines, states {0,1,2})
  - M/M/c/K server pool (event-driven)
  - Congestion-coupled communication delay
  - Metric collection with warm-up filtering

Author: Md Abid Hassan Mitul
"""

import simpy
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from parameters import CTMC, TASK_GENERATION, SERVER, LATENCY


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Task:
    """Represents a single MES task submitted to the SCADA server."""
    arrival_time: float
    source_machine: int
    source_state: int


@dataclass
class Metrics:
    """
    Accumulated metrics during simulation.
    All timestamped so we can filter by warmup period during analysis.
    """
    # State occupancy per (machine_id, state) -> total time
    # Uses (time_recorded, duration) tuples for warmup filtering
    state_time_records: list = field(default_factory=list)  # list of (t_start, machine_id, state, duration)
    
    # Task-level records (each with timestamp for filtering)
    task_arrivals: list = field(default_factory=list)    # (t, source_machine, source_state)
    task_drops: list = field(default_factory=list)       # (t, source_machine, source_state)
    task_completions: list = field(default_factory=list) # (t_arrival, t_service_start, t_complete, delay, service_time)
    
    # Queue length samples for time-average
    queue_samples: list = field(default_factory=list)    # (t, queue_length, in_service)


# ============================================================
# LATENCY MODEL
# ============================================================

class LatencyModel:
    """
    Congestion-coupled communication delay model.
    Mean delay: E[delta|n] = delta_0 + eta * (n/c)^beta
    where n = current queue occupancy (waiting tasks, not counting in-service).
    
    Network modes:
      - stable:   delay ~ Exponential(mean=E[delta|n])
      - degraded: delay ~ Pareto(shape=alpha_P, min=delta_min), heavy-tailed
      - noisy:    50/50 mixture of stable and degraded
    """
    
    def __init__(self, network_mode: str = "stable", c: int = 4):
        self.network_mode = network_mode
        self.c = c
        self.delta_0 = LATENCY["delta_0"]
        self.eta = LATENCY["eta"]
        self.beta = LATENCY["beta"]
        self.pareto_shape = LATENCY["pareto_shape"]
        self.pareto_min = LATENCY["pareto_min"]
    
    def sample_delay(self, queue_occupancy: int) -> float:
        """Sample delay given current queue occupancy (waiting tasks only)."""
        
        congestion_factor = (queue_occupancy / self.c) ** self.beta
        mean_delay = self.delta_0 + self.eta * congestion_factor
        
        if self.network_mode == "stable":
            return float(np.random.exponential(mean_delay))
        
        elif self.network_mode == "degraded":
            # Pareto: X = delta_min * (1-U)^(-1/alpha)
            u = np.random.uniform(0, 1)
            return float(self.pareto_min * (1 - u) ** (-1.0 / self.pareto_shape))
        
        elif self.network_mode == "noisy":
            if np.random.rand() < 0.5:
                return float(np.random.exponential(mean_delay))
            else:
                u = np.random.uniform(0, 1)
                return float(self.pareto_min * (1 - u) ** (-1.0 / self.pareto_shape))
        
        else:
            raise ValueError(f"Unknown network mode: {self.network_mode}")


# ============================================================
# SERVER QUEUE (M/M/c/K)
# ============================================================

class ServerQueue:
    """
    M/M/c/K queue with c servers, buffer capacity K, service rate mu per server.
    Event-driven: machines call submit_task() directly (no polling).
    
    System capacity K = max total tasks in system (waiting + in-service).
    When system_size >= K, new arrivals are blocked (dropped).
    """
    
    def __init__(self, env, c, mu, K, metrics, latency_model):
        self.env = env
        self.c = c
        self.mu = mu
        self.K = K
        self.metrics = metrics
        self.latency_model = latency_model
        
        self.servers = simpy.Resource(env, capacity=c)
        self.system_size = 0  # Total in system (waiting + in-service)
        self.in_service = 0   # Currently being processed
    
    @property
    def waiting_count(self) -> int:
        """Tasks currently waiting (queue length excluding those in service)."""
        return self.system_size - self.in_service
    
    def submit_task(self, task: Task):
        """
        Called by machine when generating a task.
        Returns immediately (non-blocking). Task is either accepted or dropped.
        
        Records queue occupancy the task SEES on arrival (not counting itself)
        for the congestion-coupled delay calculation.
        """
        # Record arrival
        self.metrics.task_arrivals.append(
            (self.env.now, task.source_machine, task.source_state)
        )
        
        # Blocking check
        if self.system_size >= self.K:
            self.metrics.task_drops.append(
                (self.env.now, task.source_machine, task.source_state)
            )
            return
        
        # Queue occupancy the task sees ahead of it (excludes itself)
        queue_len_seen = self.waiting_count
        
        # Accept: increment system size and process
        self.system_size += 1
        self.env.process(self._process_task(task, queue_len_seen))
    
    def _process_task(self, task: Task, queue_len_at_arrival: int):
        """
        Task lifecycle:
          arrival -> queue wait -> service (server busy for service_time) 
          -> communication delay (server FREE during this) -> departure
        
        The communication delay is a network-level end-to-end latency and does
        NOT block the server. This matches the M/M/c/K analysis
        where service rate mu governs queue dynamics and delta is added to
        end-to-end sojourn but is outside the server occupation window.
        """
        arrival_time = task.arrival_time
        self.metrics.queue_samples.append(
            (self.env.now, queue_len_at_arrival, self.in_service)
        )
        
        # Wait for a server
        with self.servers.request() as req:
            yield req
            
            service_start = self.env.now
            self.in_service += 1
            
            # Actual service (server busy only for this)
            service_time = float(np.random.exponential(1.0 / self.mu))
            yield self.env.timeout(service_time)
            
            # Service done: release server, decrement system size
            self.in_service -= 1
            self.system_size -= 1
        
        # Communication delay AFTER service: server is free during this window
        # Represents response transmission back to machine over the network
        delay = self.latency_model.sample_delay(queue_len_at_arrival)
        yield self.env.timeout(delay)
        
        complete_time = self.env.now
        
        self.metrics.task_completions.append((
            arrival_time,
            service_start,
            complete_time,
            delay,
            service_time,
        ))


# ============================================================
# MACHINE PROCESS (CTMC)
# ============================================================

class MachineProcess:
    """
    One machine running CTMC state evolution.
    State space: {0=Operational, 1=Degraded, 2=Failed}
    Transitions driven by exponential holding times (memoryless).
    
    While in state i, generates tasks at Poisson rate lambda_i.
    """
    
    # Exit rates from each state (sum of outgoing rates)
    EXIT_RATES = {
        0: CTMC["alpha_01"],                        # 0 -> 1 only
        1: CTMC["alpha_12"] + CTMC["beta_10"],      # 1 -> 0 or 1 -> 2
        2: CTMC["beta_21"],                          # 2 -> 1 only
    }
    
    # Task generation rates (tasks/sec)
    TASK_RATES = {
        0: TASK_GENERATION["lambda_0"],
        1: TASK_GENERATION["lambda_1"],
        2: 0,  # Failed machines generate no tasks
    }
    
    def __init__(self, env, machine_id, server_queue, metrics):
        self.env = env
        self.machine_id = machine_id
        self.server_queue = server_queue
        self.metrics = metrics
        
        self.current_state = 0  # Start Operational
        self._task_gen_process: Optional[simpy.events.Process] = None
    
    def run(self):
        """
        Main CTMC loop.
        Convert transition rates from hr^-1 to s^-1 by dividing by 3600.
        """
        while True:
            # Exit rate in per-second units
            q_per_sec = self.EXIT_RATES[self.current_state] / 3600.0
            
            # Sample exponential holding time (in seconds)
            hold_time = float(np.random.exponential(1.0 / q_per_sec))
            
            # Record state occupancy
            t_start = self.env.now
            self.metrics.state_time_records.append(
                (t_start, self.machine_id, self.current_state, hold_time)
            )
            
            # Spawn task generator for this state (dies when hold ends)
            self._task_gen_process = self.env.process(
                self._task_generator(self.current_state, hold_time)
            )
            
            # Sleep through the holding time
            yield self.env.timeout(hold_time)
            
            # Interrupt task generator if still alive (usually already done)
            if self._task_gen_process is not None and self._task_gen_process.is_alive:
                try:
                    self._task_gen_process.interrupt()
                except RuntimeError:
                    pass  # Already terminated between check and interrupt
            
            # Transition to next state
            self.current_state = self._sample_next_state()
    
    def _sample_next_state(self) -> int:
        """One-step transition: P[i,j] = q[i,j] / exit_rate[i]."""
        state = self.current_state
        if state == 0:
            return 1  # Only outgoing edge
        elif state == 1:
            p_to_0 = CTMC["beta_10"] / self.EXIT_RATES[1]
            return 0 if np.random.rand() < p_to_0 else 2
        else:  # state == 2
            return 1  # Only outgoing edge
    
    def _task_generator(self, state: int, duration: float):
        """
        Emit tasks at Poisson rate lambda_state during 'duration' seconds.
        Terminates naturally when duration elapses or on interrupt.
        """
        lambda_rate = self.TASK_RATES[state]
        if lambda_rate <= 0:
            # State 2 (Failed): no tasks
            try:
                yield self.env.timeout(duration)
            except simpy.Interrupt:
                pass
            return
        
        time_elapsed = 0.0
        try:
            while time_elapsed < duration:
                inter_arrival = float(np.random.exponential(1.0 / lambda_rate))
                
                if time_elapsed + inter_arrival > duration:
                    # No more time — sleep the remainder and return
                    yield self.env.timeout(duration - time_elapsed)
                    return
                
                yield self.env.timeout(inter_arrival)
                time_elapsed += inter_arrival
                
                # Submit task to server queue
                task = Task(
                    arrival_time=self.env.now,
                    source_machine=self.machine_id,
                    source_state=state,
                )
                self.server_queue.submit_task(task)
        
        except simpy.Interrupt:
            pass  # Cleanly exit on interrupt


# ============================================================
# FACTORY ENVIRONMENT (TOP-LEVEL)
# ============================================================

class FactoryEnvironment:
    """
    Top-level simulator orchestrating N machines + M/M/c/K queue + latency.
    """
    
    def __init__(
        self,
        n_machines: int = 200,
        c: int = 4,
        network_mode: str = "stable",
        run_time: float = 3600.0,
        warmup_time: float = 600.0,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            np.random.seed(seed)
        
        self.env = simpy.Environment()
        self.n_machines = n_machines
        self.c = c
        self.network_mode = network_mode
        self.run_time = run_time
        self.warmup_time = warmup_time
        
        self.metrics = Metrics()
        self.latency_model = LatencyModel(network_mode=network_mode, c=c)
        self.server_queue = ServerQueue(
            self.env,
            c=c,
            mu=SERVER["mu"],
            K=SERVER["K"],
            metrics=self.metrics,
            latency_model=self.latency_model,
        )
        
        # Spawn N machines
        self.machines = []
        for i in range(n_machines):
            machine = MachineProcess(self.env, i, self.server_queue, self.metrics)
            self.machines.append(machine)
            self.env.process(machine.run())
    
    def run(self):
        """Execute simulation to completion."""
        self.env.run(until=self.run_time)
    
    def compute_metrics(self) -> dict:
        """
        Compute performance metrics with warmup filtering.
        Only records after warmup_time are included in stats.
        """
        t_warm = self.warmup_time
        t_end = self.run_time
        analysis_duration = t_end - t_warm
        
        # ---- 1. Steady-state distribution pi* ----
        # For each state record, count only the time inside [warmup, end]
        state_time = {0: 0.0, 1: 0.0, 2: 0.0}
        for t_start, machine_id, state, duration in self.metrics.state_time_records:
            t_stop = t_start + duration
            # Compute overlap with [t_warm, t_end]
            overlap = min(t_stop, t_end) - max(t_start, t_warm)
            if overlap > 0:
                state_time[state] += overlap
        
        total_state_time = sum(state_time.values())
        if total_state_time > 0:
            pi_empirical = tuple(state_time[s] / total_state_time for s in range(3))
        else:
            pi_empirical = (0.0, 0.0, 0.0)
        
        # ---- 2. Task-level metrics (filtered by warmup) ----
        arrivals_post_warmup = [a for a in self.metrics.task_arrivals if a[0] >= t_warm]
        drops_post_warmup = [d for d in self.metrics.task_drops if d[0] >= t_warm]
        completions_post_warmup = [
            c for c in self.metrics.task_completions if c[0] >= t_warm
        ]
        
        n_arrivals = len(arrivals_post_warmup)
        n_drops = len(drops_post_warmup)
        n_completions = len(completions_post_warmup)
        
        # Queue wait times (service_start - arrival_time)
        queue_wait_times = [c[1] - c[0] for c in completions_post_warmup]
        w_q_mean = float(np.mean(queue_wait_times)) if queue_wait_times else 0.0
        
        # Delays
        delays = [c[3] for c in completions_post_warmup]
        mean_delay = float(np.mean(delays)) if delays else 0.0
        
        # Blocking probability
        p_b = n_drops / (n_arrivals + n_drops) if (n_arrivals + n_drops) > 0 else 0.0
        
        # Throughput (tasks completed per second, post-warmup)
        throughput = n_completions / analysis_duration if analysis_duration > 0 else 0.0
        
        # End-to-end sojourn time (arrival to complete)
        sojourn_times = [c[2] - c[0] for c in completions_post_warmup]
        mean_sojourn = float(np.mean(sojourn_times)) if sojourn_times else 0.0
        
        # SLA compliance: fraction of tasks with sojourn <= T_SLA (300 ms)
        from parameters import SLA_AND_WEIGHTS
        t_sla = SLA_AND_WEIGHTS["T_SLA"]
        sla_compliant = sum(1 for s in sojourn_times if s <= t_sla)
        sla_compliance = sla_compliant / len(sojourn_times) if sojourn_times else 0.0
        
        # Utilization (fraction of time servers were busy on average)
        # Approximation: throughput / (c * mu)
        utilization = throughput / (self.c * SERVER["mu"])
        
        return {
            "pi_empirical": pi_empirical,
            "pi_theoretical": CTMC["pi_star_theoretical"],
            "W_q": w_q_mean,
            "mean_delay": mean_delay,
            "mean_sojourn": mean_sojourn,
            "P_B": p_b,
            "throughput": throughput,
            "utilization": utilization,
            "sla_compliance": sla_compliance,
            "n_arrivals": n_arrivals,
            "n_completions": n_completions,
            "n_drops": n_drops,
            "run_time": self.run_time,
            "warmup_time": self.warmup_time,
        }


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def run_simulation(
    n_machines: int = 200,
    c: int = 4,
    network_mode: str = "stable",
    run_time: float = 3600.0,
    warmup_time: float = 600.0,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """
    Run one full simulation and return metrics dict.
    
    Args:
        n_machines: population size
        c: number of servers (3, 4, 5, or 6)
        network_mode: 'stable', 'degraded', or 'noisy'
        run_time: total simulated time in seconds
        warmup_time: warmup period to exclude from metrics
        seed: random seed for reproducibility
        verbose: print formatted output
    """
    env = FactoryEnvironment(
        n_machines=n_machines,
        c=c,
        network_mode=network_mode,
        run_time=run_time,
        warmup_time=warmup_time,
        seed=seed,
    )
    env.run()
    metrics = env.compute_metrics()
    
    if verbose:
        print(f"\n{'=' * 72}")
        print(f"SIMULATION RESULTS: N={n_machines}, c={c}, network={network_mode}")
        print(f"Run: {run_time:.0f}s | Warmup: {warmup_time:.0f}s | Analyzed: {run_time - warmup_time:.0f}s")
        print(f"{'=' * 72}")
        pi_e = metrics["pi_empirical"]
        pi_t = metrics["pi_theoretical"]
        print(f"\nSteady-State Distribution (pi*):")
        print(f"  Empirical:   ({pi_e[0]:.3f}, {pi_e[1]:.3f}, {pi_e[2]:.3f})")
        print(f"  Theoretical: ({pi_t[0]:.3f}, {pi_t[1]:.3f}, {pi_t[2]:.3f})")
        err = [abs(pi_e[i] - pi_t[i]) / pi_t[i] * 100 for i in range(3)]
        print(f"  Errors:      ({err[0]:.2f}%, {err[1]:.2f}%, {err[2]:.2f}%)")
        
        print(f"\nQueue Metrics:")
        print(f"  W_q (mean wait):      {metrics['W_q']*1000:.2f} ms")
        print(f"  P_B (blocking prob):  {metrics['P_B']:.6f}")
        print(f"  Utilization:          {metrics['utilization']:.3f}")
        
        print(f"\nCommunication:")
        print(f"  Mean delay:           {metrics['mean_delay']*1000:.2f} ms")
        print(f"  Mean sojourn:         {metrics['mean_sojourn']*1000:.2f} ms")
        print(f"  SLA compliance:       {metrics['sla_compliance']*100:.2f}%")
        
        print(f"\nThroughput:")
        print(f"  Tasks/sec:            {metrics['throughput']:.2f}")
        print(f"  Arrivals:             {metrics['n_arrivals']}")
        print(f"  Completions:          {metrics['n_completions']}")
        print(f"  Drops:                {metrics['n_drops']}")
        print(f"{'=' * 72}\n")
    
    return metrics


if __name__ == "__main__":
    print("Phase 1: SimPy Environment Verification")
    print("Target: Verify pi*, W_q, P_B match theoretical formulas\n")
    
    # Verification run
    metrics = run_simulation(
        n_machines=200,
        c=4,
        network_mode="stable",
        run_time=3600,        # 1 hour of simulated time
        warmup_time=600,      # 10 min warmup
        seed=42,
        verbose=True,
    )
