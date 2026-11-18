"""Summary of v1.3 tasks C1/C2 (data/v13/ppo_role_*.csv).

* shortfall of CHAMPS and of "CUSUM + floors, no agents" relative to the best
  fixed allocation, by group, mean and worst, for each study
* paired Wilcoxon test CHAMPS vs no-agents per scenario (Holm over the
  comparisons of each scenario is applied later with all controllers; here the
  raw two-sided p < 0.05 count is reported alongside)
* binding-component shares of CHAMPS in learned mode
* identity check: in every episode the two controllers take identical actions
  up to the first learned-mode decision
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT)
OUT = Path("results/v13"); OUT.mkdir(parents=True, exist_ok=True)
NETS = ["stable", "degraded", "noisy"]


def best_fixed(study):
    """Mean episode reward of the best fixed allocation per scenario."""
    if study == "normal":
        sys.path.insert(0, str(ROOT / "analysis"))
        from build_results_measured import revised
        d = revised()
    elif study == "x2":
        d = pd.concat([pd.read_csv("results/revision/server_cost_x2/eval_stable.csv")] +
                      [pd.read_csv(f"data/final_checks/eval_x2_{n}.csv") for n in ("degraded", "noisy")])
    else:
        d = pd.read_csv("data/regime_switch/eval_switch.csv")
    d = d[d.policy.str.startswith("fixed_c")]
    return d.groupby(["scenario", "policy"]).total_reward.mean().unstack().max(axis=1)


def load(study):
    if study == "switch":
        return pd.read_csv("data/v13/ppo_role_switch.csv")
    return pd.concat([pd.read_csv(f"data/v13/ppo_role_{study}_{n}.csv") for n in NETS])


def identity(d):
    a = d[d.policy == "champs"].set_index(["scenario", "eval_seed"])
    b = d[d.policy == "no_agents"].set_index(["scenario", "eval_seed"])
    bad = 0
    for key in a.index:
        ma, mb = a.loc[key, "modes"], b.loc[key, "modes"]
        k = min([i for i, m in enumerate(ma) if m == "L"] + [i for i, m in enumerate(mb) if m == "L"] + [len(ma)])
        bad += a.loc[key, "actions"][:k] != b.loc[key, "actions"][:k]
    return bad, len(a)


def summarise(study):
    d = load(study); bf = best_fixed(study)
    m = d.groupby(["scenario", "policy"]).total_reward.mean().unstack()
    sf = m.rsub(bf.reindex(m.index), axis=0)
    grp = d.drop_duplicates("scenario").set_index("scenario")
    grp = grp["group"] if "group" in grp else pd.Series("switch", index=grp.index)
    out = {}
    for p in ("champs", "no_agents"):
        s = sf[p]
        out[p] = {**{g: round(s[grp == g].mean(), 4) for g in grp.unique()},
                  "mean": round(s.mean(), 4), "worst": round(s.max(), 4)}
    # paired tests
    better = worse = 0
    for sc, g in d.groupby("scenario"):
        x = g[g.policy == "champs"].sort_values("eval_seed").total_reward.values
        y = g[g.policy == "no_agents"].sort_values("eval_seed").total_reward.values
        if np.allclose(x, y):
            continue
        p = wilcoxon(x, y).pvalue
        if p < 0.05:
            better += x.mean() > y.mean(); worse += x.mean() < y.mean()
    c = d[d.policy == "champs"]
    L = c[["n_learned", "ppo_sole", "hpa_sole", "model_sole", "tie", "ppo_in_max", "ppo_below_floor"]].sum()
    bind = {k: round(100 * L[k] / max(L["n_learned"], 1), 1) for k in
            ("ppo_sole", "hpa_sole", "model_sole", "tie", "ppo_in_max", "ppo_below_floor")}
    bind["learned_decisions"] = int(L["n_learned"])
    bad, n = identity(d)
    servers = d.groupby("policy").mean_c.mean().round(3).to_dict() if "mean_c" in d else {}
    return dict(study=study, shortfall=out, champs_better=int(better), champs_worse=int(worse),
                n_scenarios=int(d.scenario.nunique()), binding_pct=bind,
                identity_mismatch=f"{bad}/{n}", mean_servers=servers), sf, grp


if __name__ == "__main__":
    import json
    allres = {}
    for st in ("normal", "x2", "switch"):
        if st != "switch" and not all(Path(f"data/v13/ppo_role_{st}_{n}.csv").exists() for n in NETS):
            continue
        if st == "switch" and not Path("data/v13/ppo_role_switch.csv").exists():
            continue
        r, sf, grp = summarise(st); allres[st] = r
        sf.assign(group=grp).to_csv(OUT / f"ppo_role_shortfall_{st}.csv")
        print(json.dumps(r, indent=1))
    json.dump(allres, open(OUT / "ppo_role_summary.json", "w"), indent=1)
