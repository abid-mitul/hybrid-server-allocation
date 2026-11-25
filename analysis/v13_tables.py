"""Tables of the v1.3 paper, with the no-agents ablation (C1) added to every
comparison and the Holm correction recomputed over the enlarged families.

  table_main     69 scenarios, nominal cost: all controllers, group means,
                 mean, worst, CHAMPS better / same / worse (Holm within scenario)
  table_bytype   shortfall per scenario type (Appendix B)
  table_x2       doubled server cost, per network: mean, worst and tests
  table_switch   regime change during operation
Shortfall = mean episode reward of the best fixed allocation minus that of the
controller.  Wilcoxon signed-rank on 20 paired episodes, p < 0.05.
Writes results/v13/table_*.csv and table_*.md
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "analysis"))
from build_results_measured import revised, grp
from stats import _holm

OUT = Path("results/v13"); OUT.mkdir(parents=True, exist_ok=True)
NETS = ["stable", "degraded", "noisy"]; RV = Path("results/revision")
FIX = [f"fixed_c{c}" for c in (3, 4, 5, 6)]
REF = "hybrid3_B"


def rd(sub, pat, nets=NETS):
    return pd.concat([pd.read_csv(RV / sub / pat.format(n)) for n in nets], ignore_index=True)


def v13(study, nets=NETS):
    d = pd.concat([pd.read_csv(f"data/v13/ppo_role_{study}_{n}.csv", dtype={"actions": str, "modes": str})
                   for n in nets], ignore_index=True)
    return d[d.policy == "no_agents"]


def compare(df, comps, ref=REF):
    """Shortfall table and Holm-adjusted paired tests of ref against comps."""
    pe = df.groupby(["scenario", "policy", "eval_seed"]).total_reward.mean().reset_index()
    w = pe.groupby(["scenario", "policy"]).total_reward.mean().unstack()
    bf = w[FIX].max(axis=1)
    sf = w.rsub(bf, axis=0)
    res = {c: [0, 0, 0] for c in comps}
    for sc in w.index:
        a = pe[(pe.scenario == sc) & (pe.policy == ref)].set_index("eval_seed").total_reward
        ps, ds = [], []
        for c in comps:
            b = pe[(pe.scenario == sc) & (pe.policy == c)].set_index("eval_seed").total_reward
            d = (a - b).dropna(); ds.append(d.mean())
            ps.append(1.0 if np.allclose(d, 0) else st.wilcoxon(d).pvalue)
        for c, d, p in zip(comps, ds, _holm(np.array(ps))):
            res[c][0 if (p < .05 and d > 0) else 2 if p < .05 else 1] += 1
    tests = {c: "/".join(map(str, v)) for c, v in res.items()}
    tests[ref] = "---"
    return sf, tests


def write_md(tab, name, cols):
    with open(OUT / f"{name}.md", "w") as f:
        f.write("| Controller | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n")
        for i, r in tab.iterrows():
            f.write(f"| {i} | " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], (float, np.floating)) else str(r[c])
                                            for c in cols) + " |\n")


# ------------------------------------------------------------------ main study
NAMES = [("fixed_c6", "Fixed c_max = 6"), ("fixed_priority", "Threshold rule"), ("ga", "GA-tuned rule"),
         ("autoscaler_hpa", "Autoscaler"), ("queueing_model", "Queueing model"),
         ("adaptive_model", "Self-calibrating model"), ("ppo", "PPO (single agent)"),
         ("ppo_ensemble", "PPO ensemble"), ("hybrid_adaptive", "+ persistence + PPO ensemble"),
         ("model_fallback_c6", "+ persistence + c_max"), ("cusum_fallback_c6", "+ CUSUM + c_max"),
         ("cusum_hybrid_ppo", "+ CUSUM + PPO ensemble"), ("cusum_floor_ppo", "+ CUSUM + PPO, model floor"),
         ("hybrid3_A", "+ CUSUM + PPO, autoscaler floor"), ("no_agents", "+ CUSUM + floors, no agents"),
         ("hybrid3_B", "CHAMPS"), ("oracle_model", "Perfect-information model policy")]


def main_tables():
    cu = rd("cusum", "eval_cusum_{}.csv"); cu = cu[cu.policy != "persistence_probe"]
    df = pd.concat([revised(), rd("fallback", "eval_fallback_{}.csv"), cu,
                    rd("safety_floor", "eval_floor_{}.csv"), rd("autoscaler", "eval_hpa_{}.csv"),
                    rd("hybrid3", "eval_h3_{}.csv"), rd("oracle", "eval_oracle_{}.csv"), v13("normal")],
                   ignore_index=True)
    P = [p for p, _ in NAMES]
    sf, tests = compare(df, [p for p in P if p != REF])
    sf = sf[P]; g = sf.index.map(grp)
    tab = pd.concat([sf.groupby(g).mean().loc[["seen", "unseen", "parameter", "structural"]].T,
                     sf.mean().rename("mean"), sf.max().rename("worst")], axis=1)
    tab["within_0.05"] = (sf <= 0.05).sum()
    tab["tests"] = [tests[p] for p in P]
    tab.index = [dict(NAMES)[p] for p in P]
    tab.round(4).to_csv(OUT / "table_main.csv")
    write_md(tab.round(3), "table_main", ["seen", "unseen", "parameter", "structural", "mean", "worst",
                                          "within_0.05", "tests"])
    print(tab.round(3).to_string())
    # Appendix B: by scenario type
    cols = ["fixed_c6", "fixed_priority", "ga", "autoscaler_hpa", "queueing_model", "adaptive_model",
            "ppo_ensemble", "cusum_fallback_c6", "no_agents", "hybrid3_B", "oracle_model"]
    bt = sf[cols].copy(); bt["type"] = [s.rsplit("_", 1)[0] for s in bt.index]
    bt.groupby("type")[cols].mean().round(3).to_csv(OUT / "table_bytype.csv")


# ------------------------------------------------------------------ doubled cost
def x2_table():
    rows, P = [], ["autoscaler_hpa", "ppo_ensemble", "cusum_fallback_c6", "cusum_floor_ppo", "no_agents", REF]
    st_df = pd.concat([pd.read_csv(RV / "server_cost_x2" / "eval_stable.csv"),
                       pd.read_csv(RV / "autoscaler" / "eval_hpa_wc010.csv"),
                       pd.read_csv(RV / "safety_floor" / "eval_floor_wc010.csv"),
                       pd.read_csv(RV / "hybrid3" / "eval_h3_wc010.csv"), v13("x2", ["stable"])],
                      ignore_index=True)
    out = {}
    for net in NETS:
        df = st_df if net == "stable" else pd.concat(
            [pd.read_csv(RV / "final_checks" / f"eval_x2_{net}.csv"), v13("x2", [net])], ignore_index=True)
        df = df[df.policy.isin(FIX + P)]
        sf, tests = compare(df, [p for p in P if p != REF])
        for p in P:
            out.setdefault(p, {})
            out[p].update({f"{net}_mean": sf[p].mean(), f"{net}_worst": sf[p].max(), f"{net}_tests": tests[p]})
    tab = pd.DataFrame(out).T
    tab.to_csv(OUT / "table_x2.csv"); print(tab.to_string())


# ------------------------------------------------------------------ regime switch
def switch_table():
    d = pd.read_csv("data/regime_switch/eval_switch.csv")
    na = pd.read_csv("data/v13/ppo_role_switch.csv", dtype={"actions": str, "modes": str})
    df = pd.concat([d, na[na.policy == "no_agents"]], ignore_index=True)
    P = ["autoscaler_hpa", "adaptive_model", "cusum_fallback_c6", "no_agents", REF, "ra_hybrid3_B"]
    df = df[df.policy.isin(FIX + P)]
    sf, tests = compare(df, [p for p in P if p != REF])
    sw = df.drop_duplicates("scenario").set_index("scenario")["switch"]
    tab = pd.concat([sf[P].groupby(sw.reindex(sf.index)).mean().T, sf[P].mean().rename("mean"),
                     sf[P].max().rename("worst")], axis=1)
    tab["tests"] = [tests[p] for p in P]
    tab.to_csv(OUT / "table_switch.csv"); print(tab.round(3).to_string())


if __name__ == "__main__":
    main_tables(); x2_table(); switch_table()
