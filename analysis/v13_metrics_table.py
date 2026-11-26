"""Per-metric table of the v1.3 paper (appendix): the individual performance
measures named in the thesis proposal, for the main controllers.

For each controller, averaged over the 20 evaluation episodes of each scenario
and then over scenarios:
  throughput      completed tasks per second
  latency         mean sojourn time of completed tasks (ms)
  utilisation     throughput / (mean servers x nominal service rate mu0)
  blocking        blocked / offered tasks (%)
  SLA violation   share of completed tasks with sojourn above T_SLA (%)
  servers         time-average number of allocated servers
  reward          mean episode reward (the objective; higher is better)
Single PPO agents are averaged over the five agents.  Rows: the 69-scenario
main study (nominal cost) and the 18 held-out scenarios.
Writes results/v13/table_metrics_{main,heldout}.csv and .md
"""
import os, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT)
sys.path[:0] = [str(ROOT), str(ROOT / "analysis")]
from v13_tables import main_df, OUT
from fast_env import SERVER  # nominal server parameters

MU0 = float(SERVER["mu"])
NAMES = [("fixed_c6", "Fixed c_max = 6"), ("round_robin", "Round-Robin"), ("fixed_priority", "Threshold rule"),
         ("ga", "GA-tuned rule"), ("autoscaler_hpa", "Autoscaler"), ("adaptive_model", "Self-calibrating model"),
         ("ppo", "PPO (single agent)"), ("ppo_ensemble", "PPO ensemble"), ("no_agents", "CUSUM + floors, no agents"),
         ("hybrid3_B", "CHAMPS"), ("champs", "CHAMPS"), ("oracle_model", "Perfect-information model policy")]
COLS = ["throughput", "latency_ms", "utilisation_pct", "blocking_pct", "sla_violation_pct", "servers", "reward"]


def metrics(df):
    df = df[df.policy.isin(dict(NAMES))].copy()
    df["latency_ms"] = df.mean_sojourn_ms
    df["utilisation_pct"] = 100 * df.throughput / (df.mean_c * MU0)
    df["blocking_pct"] = 100 * df.n_drops / df.n_offered.clip(lower=1)
    df["sla_violation_pct"] = 100 * (1 - df.sla_compliance)
    df["servers"] = df.mean_c; df["reward"] = df.total_reward
    ep = df.groupby(["scenario", "policy", "eval_seed"])[COLS].mean()          # PPO: mean of 5 agents
    tab = ep.groupby(["scenario", "policy"]).mean().groupby("policy").mean()
    order = [p for p, _ in NAMES if p in tab.index]
    tab = tab.loc[order]; tab.index = [dict(NAMES)[p] for p in order]
    return tab


def write(tab, name):
    tab.round(3).to_csv(OUT / f"{name}.csv")
    fmt = dict(throughput="{:.1f}", latency_ms="{:.1f}", utilisation_pct="{:.1f}", blocking_pct="{:.3f}",
               sla_violation_pct="{:.2f}", servers="{:.2f}", reward="{:.3f}")
    with open(OUT / f"{name}.md", "w") as f:
        f.write("| Controller | " + " | ".join(COLS) + " |\n|" + "---|" * (len(COLS) + 1) + "\n")
        for i, r in tab.iterrows():
            f.write(f"| {i} | " + " | ".join(fmt[c].format(r[c]) for c in COLS) + " |\n")
    print(name); print(tab.round(3).to_string()); print()


if __name__ == "__main__":
    write(metrics(main_df()), "table_metrics_main")
    ho = pd.concat([pd.read_csv(f) for f in sorted(Path("data/v13").glob("c3_*.csv"))], ignore_index=True)
    write(metrics(ho), "table_metrics_heldout")
