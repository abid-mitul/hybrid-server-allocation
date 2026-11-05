"""
Baseline allocation policies (actions 0..3 correspond to c = 3, 4, 5, 6):

  1. RoundRobinPolicy       -- cycles through c = 3, 4, 5, 6
  2. FixedPriorityPolicy    -- threshold rule on the observed congestion
  3. GeneticAlgorithmPolicy -- same rule with thresholds tuned by a GA (DEAP)

All policies share the predict(obs) interface of the PPO models.

Author: Md Abid Hassan Mitul
"""

import numpy as np
import random
from typing import Tuple, Optional, List
from deap import base, creator, tools, algorithms


# ============================================================
# BASE POLICY INTERFACE
# ============================================================

class BasePolicy:
    """
    Common interface matching Stable-Baselines3 model predict method,
    so baselines can be substituted for PPO in evaluate.py.
    """
    name: str = "base"
    
    def predict(self, observation: np.ndarray, deterministic: bool = True
                ) -> Tuple[int, None]:
        """Return (action, state). State is unused for stateless baselines."""
        raise NotImplementedError
    
    def reset(self):
        """Reset any internal state at start of episode."""
        pass


# ============================================================
# 1. ROUND-ROBIN POLICY
# ============================================================

class RoundRobinPolicy(BasePolicy):
    """
    Cycles through c in {3, 4, 5, 6} in order, one server count per step.
    This baseline serves as a naive non-adaptive control.
    """
    name = "round_robin"
    
    def __init__(self):
        self.counter = 0
    
    def predict(self, observation: np.ndarray, deterministic: bool = True
                ) -> Tuple[int, None]:
        action = self.counter % 4
        self.counter += 1
        return int(action), None
    
    def reset(self):
        self.counter = 0


# ============================================================
# 2. FIXED-PRIORITY (THRESHOLD) POLICY
# ============================================================

class FixedPriorityPolicy(BasePolicy):
    """
    Threshold-based adaptive control:
      - If congestion (n_hat) exceeds high threshold, use c=6
      - Otherwise moderate/low thresholds determine c=5/4/3
    
    Uses the first observation dimension (n_hat: normalized queue occupancy)
    plus utilization U to decide c.
    
    Default thresholds are hand-tuned for the "medium" load scenario.
    """
    name = "fixed_priority"
    
    def __init__(self, thresholds: Optional[List[float]] = None):
        # thresholds[i] = boundary for c = 3 + i (i.e., threshold for going up)
        # If congestion > thresholds[2] -> c=6 (highest)
        # If congestion > thresholds[1] -> c=5
        # If congestion > thresholds[0] -> c=4
        # else                          -> c=3
        if thresholds is None:
            self.thresholds = [0.10, 0.30, 0.60]  # low, medium, high
        else:
            self.thresholds = list(thresholds)
    
    def predict(self, observation: np.ndarray, deterministic: bool = True
                ) -> Tuple[int, None]:
        # observation = (n_hat, delta_hat, U, X_hat, P_B)
        # Use a combined congestion signal: max of n_hat and U
        n_hat, delta_hat, u, x_hat, p_b = observation[:5]
        congestion = max(n_hat, u * 0.8)  # utilization weighted
        
        if congestion >= self.thresholds[2]:
            action = 3  # c=6
        elif congestion >= self.thresholds[1]:
            action = 2  # c=5
        elif congestion >= self.thresholds[0]:
            action = 1  # c=4
        else:
            action = 0  # c=3
        
        return int(action), None


# ============================================================
# 3. GENETIC ALGORITHM POLICY
# ============================================================

class GeneticAlgorithmPolicy(BasePolicy):
    """
    Genetic algorithm-optimized threshold policy.
    Same threshold structure as FixedPriorityPolicy, but thresholds
    are evolved by DEAP to maximize composite reward on evaluation episodes.
    
    Called offline (before evaluation) via `fit(env_factory, ...)`.
    """
    name = "ga"
    
    def __init__(self, thresholds: Optional[List[float]] = None):
        if thresholds is None:
            # Default before fitting
            self.thresholds = [0.10, 0.30, 0.60]
        else:
            self.thresholds = list(thresholds)
        self.fitness_history: List[float] = []
    
    def predict(self, observation: np.ndarray, deterministic: bool = True
                ) -> Tuple[int, None]:
        n_hat, delta_hat, u, x_hat, p_b = observation[:5]
        congestion = max(n_hat, u * 0.8)
        
        if congestion >= self.thresholds[2]:
            action = 3
        elif congestion >= self.thresholds[1]:
            action = 2
        elif congestion >= self.thresholds[0]:
            action = 1
        else:
            action = 0
        
        return int(action), None
    
    @staticmethod
    def _evaluate_thresholds(thresholds, env_factory, n_eval_episodes: int = 2
                             ) -> float:
        """
        Run n_eval_episodes with the given thresholds and return mean total reward.
        env_factory is a callable that returns a fresh env.
        """
        # Ensure thresholds are sorted ascending and in (0, 1)
        t = sorted([max(0.01, min(0.99, x)) for x in thresholds])
        policy = FixedPriorityPolicy(thresholds=t)
        
        total_rewards = []
        for _ in range(n_eval_episodes):
            env = env_factory()
            obs, _ = env.reset()
            policy_reward = 0.0
            done = False
            while not done:
                action, _ = policy.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                policy_reward += reward
            total_rewards.append(policy_reward)
        return float(np.mean(total_rewards))
    
    def fit(
        self,
        env_factory,
        population_size: int = 20,
        n_generations: int = 10,
        n_eval_episodes: int = 2,
        seed: int = 42,
        verbose: bool = True,
    ):
        """
        Run GA to evolve optimal thresholds.
        
        Args:
            env_factory: callable that returns a fresh env (with fixed seed)
            population_size: number of candidate threshold sets per generation
            n_generations: GA generations
            n_eval_episodes: episodes per fitness evaluation
            seed: RNG seed
            verbose: print progress
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # DEAP setup — re-create only if not already registered
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, 0.05, 0.95)
        toolbox.register("individual", tools.initRepeat,
                         creator.Individual, toolbox.attr_float, 3)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def eval_fn(ind):
            return (self._evaluate_thresholds(ind, env_factory, n_eval_episodes),)
        
        toolbox.register("evaluate", eval_fn)
        toolbox.register("mate", tools.cxBlend, alpha=0.3)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.3)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        population = toolbox.population(n=population_size)
        
        if verbose:
            print(f"GA training: pop={population_size}, gen={n_generations}, "
                  f"eval_eps={n_eval_episodes}")
        
        for gen in range(n_generations):
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.3)
            fits = list(map(toolbox.evaluate, offspring))
            for ind, fit in zip(offspring, fits):
                ind.fitness.values = fit
            population = toolbox.select(offspring, k=len(population))
            
            best = tools.selBest(population, k=1)[0]
            best_fit = best.fitness.values[0]
            self.fitness_history.append(best_fit)
            
            if verbose:
                print(f"  Gen {gen+1}/{n_generations}: best fit = {best_fit:.3f}, "
                      f"thresholds = ({best[0]:.3f}, {best[1]:.3f}, {best[2]:.3f})")
        
        # Set final thresholds from best individual
        best = tools.selBest(population, k=1)[0]
        self.thresholds = sorted(
            [max(0.01, min(0.99, x)) for x in best]
        )
        
        if verbose:
            print(f"GA final thresholds: ({self.thresholds[0]:.3f}, "
                  f"{self.thresholds[1]:.3f}, {self.thresholds[2]:.3f})")


# ============================================================
# QUICK SANITY TEST
# ============================================================

if __name__ == "__main__":
    from fast_env import make_fast_env as make_env
    
    print("Testing baseline policies...\n")
    
    for policy_cls, name in [
        (RoundRobinPolicy, "Round-Robin"),
        (FixedPriorityPolicy, "Fixed-Priority"),
    ]:
        policy = policy_cls()
        policy.reset()
        env = make_env(n_machines=50, network_mode="stable",
                       episode_duration=300, step_duration=60, seed=42)
        obs, _ = env.reset()
        total_r = 0.0
        done = False
        step = 0
        while not done:
            action, _ = policy.predict(obs)
            obs, reward, done, _, info = env.step(action)
            total_r += reward
            step += 1
        print(f"{name:20s}: total reward = {total_r:.2f} over {step} steps")
    
    # GA quick test
    print("\nQuick GA fit (population=8, generations=3)...")
    def env_factory():
        return make_env(n_machines=50, network_mode="stable",
                        episode_duration=300, step_duration=60, seed=42)
    ga = GeneticAlgorithmPolicy()
    ga.fit(env_factory, population_size=8, n_generations=3,
           n_eval_episodes=1, verbose=True)
    print(f"GA final thresholds: {ga.thresholds}")
