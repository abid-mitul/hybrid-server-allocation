"""
Model parameters of the case study (a representative SCADA server layer of a
textile plant).

The values are engineering estimates representative of such installations,
not measurements from a particular plant; reward weights and costs are design
choices. The sensitivity of the results to these values is examined in the
paper (sensitivity analysis and model-mismatch scenarios).

Author: Md Abid Hassan Mitul
"""

# ============================================================
# CTMC TRANSITION RATES
# ============================================================
# Machine state transitions: {Operational → Degraded → Failed}
# Units: hr^-1 (hourly rates). Divided by 3600 for rates per second.

CTMC = {
    "alpha_01": 0.1,           # Operational → Degraded (estimate) [1/10 hr MTTD]
    "alpha_12": 0.2,           # Degraded → Failed (estimate) [1/5 hr]
    "beta_10": 0.5,            # Degraded → Operational (estimate) [1/2 hr repair]
    "beta_21": 0.333,          # Failed → Degraded (estimate) [1/3 hr repair initiation]
    
    # Steady-state verification (closed form)
    "pi_star_theoretical": (0.758, 0.151, 0.091),  # (π_0*, π_1*, π_2*)
}

# ============================================================
# TASK GENERATION RATES
# ============================================================
# Each machine generates tasks at rate λ_i depending on state.
# Units: tasks/s per machine.

TASK_GENERATION = {
    "lambda_0": 5,             # Operational state (estimate)
    "lambda_1": 2,             # Degraded state (estimate) [40% of normal]
}

# ============================================================
# SERVER PARAMETERS
# ============================================================
# M/M/c/K queue model for SCADA server layer.
# Service rate in tasks/s per server (inverse of 3 ms mean service time).

SERVER = {
    "mu": 333,                 # Service rate tasks/s per server (estimate) [1/0.003 s]
    "c_action_space": [3, 4, 5, 6],  # Available server counts (DRL agent actions)
    "K": 50,                   # Buffer capacity (estimate) [SCADA queue limit]
}

# ============================================================
# LATENCY PARAMETERS
# ============================================================
# Communication delay model with congestion coupling.
# E[δ|n] = δ_0 + η(n/c)^β, where n = queue occupancy.

LATENCY = {
    "delta_0": 0.050,          # Baseline delay (estimate) [50 ms, network RTT]
    "eta": 0.807,              # Congestion sensitivity (estimate)
    "beta": 1.4,               # Congestion exponent (estimate)
    
    # Pareto distribution for degraded network
    "pareto_shape": 2.5,       # Shape parameter α_P (estimate)
    "pareto_min": 0.050,       # Minimum delay δ_min (estimate) [50 ms]
    # Mean delay under Pareto: 2.5 * 50 / (2.5 - 1) = 83.3 ms
}

# ============================================================
# SLA AND OBJECTIVE WEIGHTS
# ============================================================
# Service Level Agreement threshold and composite reward function weights.
# Φ = w_X·η_X - w_D·(1-η_D) - w_B·P_B

SLA_AND_WEIGHTS = {
    "T_SLA": 0.300,            # SLA threshold (estimate) [300 ms]
    
    # Composite objective weights (sum to 1.0)
    "w_X": 0.5,                # Throughput weight (design choice)
    "w_D": 0.3,                # Latency/delay weight
    "w_B": 0.2,                # Blocking probability weight
}

# ============================================================
# EVALUATION SCENARIOS
# ============================================================
# Base scenarios: 4 load conditions × 3 network regimes (further scenarios
# are defined in fast_evaluate.py).
# Each specifies: number of machines N, resulting Λ, ρ at c=4, network type.

SCENARIOS = {
    # LOW LOAD (N=80 machines)
    "low_stable": {
        "N": 80,
        "Lambda": 327,         # = 80 * (0.758*5 + 0.151*2)
        "rho_at_c4": 0.245,
        "network": "stable",
    },
    "low_degraded": {
        "N": 80,
        "Lambda": 327,
        "rho_at_c4": 0.245,
        "network": "degraded",
    },
    "low_noisy": {
        "N": 80,
        "Lambda": 327,
        "rho_at_c4": 0.245,
        "network": "noisy",
    },
    
    # MEDIUM LOAD (N=200 machines)
    "medium_stable": {
        "N": 200,
        "Lambda": 818,         # = 200 * (0.758*5 + 0.151*2)
        "rho_at_c4": 0.614,
        "network": "stable",
    },
    "medium_degraded": {
        "N": 200,
        "Lambda": 818,
        "rho_at_c4": 0.614,
        "network": "degraded",
    },
    "medium_noisy": {
        "N": 200,
        "Lambda": 818,
        "rho_at_c4": 0.614,
        "network": "noisy",
    },
    
    # HIGH LOAD (N=350 machines) — UNSTABLE at c=4
    "high_stable": {
        "N": 350,
        "Lambda": 1432,        # = 350 * (0.758*5 + 0.151*2)
        "rho_at_c4": 1.074,    # > 1.0 → queue unstable if c=4
        "network": "stable",
    },
    "high_degraded": {
        "N": 350,
        "Lambda": 1432,
        "rho_at_c4": 1.074,
        "network": "degraded",
    },
    "high_noisy": {
        "N": 350,
        "Lambda": 1432,
        "rho_at_c4": 1.074,
        "network": "noisy",
    },
    
    # BURST LOAD (N: 200→350 during simulation)
    "burst_stable": {
        "N_start": 200,
        "N_end": 350,
        "Lambda_start": 818,
        "Lambda_end": 1432,
        "network": "stable",
    },
    "burst_degraded": {
        "N_start": 200,
        "N_end": 350,
        "Lambda_start": 818,
        "Lambda_end": 1432,
        "network": "degraded",
    },
    "burst_noisy": {
        "N_start": 200,
        "N_end": 350,
        "Lambda_start": 818,
        "Lambda_end": 1432,
        "network": "noisy",
    },
}

# ============================================================
# TRAINING HYPERPARAMETERS (Phases 3–4)
# ============================================================
# PPO training configuration and evaluation settings.

TRAINING = {
    "total_timesteps": 1_000_000,  # Total environment steps per seed
    "n_seeds": 5,                   # Number of random seeds for robustness
    "ppo_network": [128, 128, 64],  # Policy/value network architecture
    "discount_factor": 0.99,         # γ for PPO (standard RL practice)
}

EVALUATION = {
    "episodes_per_policy_scenario": 10,  # 4 policies × 12 scenarios × 10 episodes = 480 runs
    "episode_length_hours": 8,            # Simulated shift duration
    "statistical_test": "wilcoxon",       # Pairwise policy comparison
    "significance_threshold": 0.001,      # p < 0.001 for claim of superiority
}

# ============================================================
# SIMULATION CONTROL
# ============================================================
# Phase 1 (this phase) runs with fixed c to verify math.
# Phases 2–4 will make c adaptive via DRL.

SIMULATION = {
    "phase": 1,                     # Phase 1: verify CTMC + queue + latency
    "fixed_server_count": 4,        # c = 4 servers (medium load baseline)
    "warmup_time": 3600,            # Seconds to run before recording metrics (1 hour)
    "run_time": 86400,              # Total simulated time (1 shift = 8 hrs, but we run longer for stats)
}
