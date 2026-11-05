"""
Statistics for the evaluation results (called by fast_evaluate.summarize).

All policies are evaluated on the same evaluation seeds (common random
numbers), so comparisons are paired by episode.

Writes to data/:
  summary_ci.csv          per scenario and policy: mean, sd, n, 95% CI of the mean
                          (PPO first averaged over its training seeds per episode)
  ppo_seed_variation.csv  PPO mean reward per training seed, per scenario
  statistical_tests.csv   PPO vs every other policy, per scenario:
                          mean paired difference with 95% CI, win rate,
                          Wilcoxon signed-rank p-value, Holm-adjusted p-value
"""
import numpy as np
import pandas as pd
from scipy import stats as st

METRIC = "total_reward"


def _ci(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = x.mean()
    if n < 2:
        return m, np.nan, np.nan, np.nan
    sd = x.std(ddof=1)
    h = st.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return m, sd, m - h, m + h


def _wilcoxon(d):
    d = np.asarray(d, dtype=float)
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(st.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def _holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def write_statistics(df, out):
    # one value per (scenario, policy, episode): PPO averaged over training seeds
    per_ep = (df.groupby(["scenario", "group", "network", "policy", "eval_seed"])[METRIC]
                .mean().reset_index())

    rows = []
    for (scen, grp, net, pol), g in per_ep.groupby(["scenario", "group", "network", "policy"]):
        m, sd, lo, hi = _ci(g[METRIC])
        rows.append(dict(scenario=scen, group=grp, network=net, policy=pol, mean=m,
                         sd=sd, ci95_low=lo, ci95_high=hi, n_episodes=len(g)))
    pd.DataFrame(rows).to_csv(out / "summary_ci.csv", index=False)

    ppo = df[df.policy == "ppo"]
    if len(ppo):
        (ppo.groupby(["scenario", "group", "network", "train_seed"])[METRIC].mean()
            .reset_index().rename(columns={METRIC: "mean_total_reward"})
            .to_csv(out / "ppo_seed_variation.csv", index=False))

    tests = []
    for scen, g in per_ep.groupby("scenario"):
        wide = g.pivot(index="eval_seed", columns="policy", values=METRIC)
        if "ppo" not in wide:
            continue
        block = []
        for other in wide.columns:
            if other == "ppo":
                continue
            pair = wide[["ppo", other]].dropna()
            d = pair["ppo"] - pair[other]
            m, sd, lo, hi = _ci(d)
            block.append(dict(scenario=scen, group=g["group"].iloc[0],
                              network=g["network"].iloc[0], comparison=f"ppo vs {other}",
                              mean_diff=m, ci95_low=lo, ci95_high=hi,
                              ppo_win_rate=float((d > 1e-12).mean()),
                              p_wilcoxon=_wilcoxon(d), n_pairs=len(d)))
        if block:
            adj = _holm([b["p_wilcoxon"] for b in block])
            for b, a in zip(block, adj):
                b["p_holm"] = a
            tests.extend(block)
    pd.DataFrame(tests).to_csv(out / "statistical_tests.csv", index=False)
