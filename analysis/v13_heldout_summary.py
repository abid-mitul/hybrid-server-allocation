"""Summary of the held-out evaluation (v1.3 task C3, data/v13/c3_*.csv).

Shortfall relative to the best fixed allocation (mean episode reward of the
best of c = 3..6 in each scenario minus that of the controller).  Per
controller: mean over the six scenarios of each type, overall mean, worst
scenario, and the paired comparison with CHAMPS (two-sided Wilcoxon on the 20
paired episodes, Holm-adjusted over the comparisons of each scenario,
p < 0.05).  Single PPO agents are reported as the mean over the five agents of
each regime, as in the main study.  Also: learned-mode share and binding shares
of CHAMPS per type, failures per episode (H2), and mean servers.
Writes results/v13/heldout_table.csv and results/v13/heldout_summary.json
"""
import json, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT)
OUT = Path("results/v13"); OUT.mkdir(parents=True, exist_ok=True)
ORDER = ["fixed_c6", "fixed_priority", "ga", "autoscaler_hpa", "queueing_model", "adaptive_model",
         "ppo", "ppo_ensemble", "cusum_fallback_c6", "no_agents", "champs", "oracle_model"]


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); adj = np.empty(n); run = 0.0
    for i, k in enumerate(o):
        run = max(run, (n - i) * p[k]); adj[k] = min(run, 1.0)
    return adj


def main():
    d = pd.concat([pd.read_csv(f"data/v13/c3_{n}.csv") for n in ("stable", "degraded", "noisy")])
    ep = d.groupby(["scenario", "policy", "eval_seed"]).total_reward.mean().reset_index()   # PPO: mean of 5
    w = ep.groupby(["scenario", "policy"]).total_reward.mean().unstack()
    bf = w[[f"fixed_c{c}" for c in (3, 4, 5, 6)]].max(axis=1)
    sf = w.rsub(bf, axis=0)
    typ = d.drop_duplicates("scenario").set_index("scenario")["heldout"]
    # paired tests, Holm within each scenario
    cnt = {p: [0, 0, 0] for p in ORDER if p != "champs"}
    for sc, g in ep.groupby("scenario"):
        x = g[g.policy == "champs"].sort_values("eval_seed").total_reward.values
        ps, sg = [], []
        for p in cnt:
            y = g[g.policy == p].sort_values("eval_seed").total_reward.values
            diff = x - y
            ps.append(1.0 if np.allclose(diff, 0) else wilcoxon(x, y).pvalue); sg.append(np.mean(diff))
        for p, q, s in zip(cnt, holm(ps), sg):
            cnt[p][0 if (q < 0.05 and s > 0) else 2 if (q < 0.05 and s < 0) else 1] += 1
    rows = []
    for p in ORDER:
        r = {"controller": p, **{h: sf.loc[typ.index[typ == h], p].mean() for h in ("H1", "H2", "H3")},
             "mean": sf[p].mean(), "worst": sf[p].max(),
             "tests": "---" if p == "champs" else "/".join(map(str, cnt[p]))}
        rows.append(r)
    tab = pd.DataFrame(rows); tab.to_csv(OUT / "heldout_table.csv", index=False)
    print(tab.round(3).to_string(index=False))
    c = d[d.policy == "champs"]
    summ = {"learned_share_pct": (100 * c.groupby("heldout").ppo_fraction.mean()).round(1).to_dict(),
            "binding_pct": {}, "failures_per_episode_H2": round(d[d.heldout == "H2"].n_failures.mean(), 2),
            "mean_servers": d.groupby(["heldout", "policy"]).mean_c.mean().unstack()[
                ["champs", "no_agents", "autoscaler_hpa", "adaptive_model", "fixed_c6"]].round(2).to_dict("index"),
            "best_fixed_mean_by_type": {h: float(bf[typ.index[typ == h]].mean()) for h in ("H1", "H2", "H3")},
            "best_fixed_c": w[[f"fixed_c{c}" for c in (3, 4, 5, 6)]].idxmax(axis=1).to_dict(),
            "within_0.05": (sf[ORDER] <= 0.05).sum().to_dict()}
    for h, g in c.groupby("heldout"):
        L = g[["n_learned", "ppo_sole", "hpa_sole", "model_sole", "tie"]].sum()
        summ["binding_pct"][h] = {k: round(100 * L[k] / max(L["n_learned"], 1), 1)
                                  for k in ("ppo_sole", "hpa_sole", "model_sole", "tie")} | {
                                  "learned_decisions": int(L["n_learned"])}
    json.dump(summ, open(OUT / "heldout_summary.json", "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))
    sf.assign(type=typ).to_csv(OUT / "heldout_shortfall_by_scenario.csv")


if __name__ == "__main__":
    main()
