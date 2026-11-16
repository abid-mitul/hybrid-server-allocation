"""Final summary table of the revised paper: every controller on the 69
scenarios (normal server cost, measured utilisation, common seeds), mean
shortfall per scenario group, overall mean and worst case, and paired
Wilcoxon tests of the proposed hybrid (Hybrid B) against each controller
(better / no significant difference / worse at p < 0.05, Holm-adjusted within
each scenario).  Output: results/analysis_measured/final_summary.csv and .md
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_results_measured import revised, grp, OUT, RES
from stats import _holm

NETS = ['stable', 'degraded', 'noisy']; RV = os.path.join(RES, 'revision')
rd = lambda sub, pat: pd.concat([pd.read_csv(os.path.join(RV, sub, pat.format(n))) for n in NETS], ignore_index=True)
cu = rd('cusum', 'eval_cusum_{}.csv'); cu = cu[cu.policy != 'persistence_probe']
df = pd.concat([revised(), rd('fallback', 'eval_fallback_{}.csv'), cu,
                rd('safety_floor', 'eval_floor_{}.csv'), rd('autoscaler', 'eval_hpa_{}.csv'),
                rd('hybrid3', 'eval_h3_{}.csv'), rd('oracle', 'eval_oracle_{}.csv')], ignore_index=True)
NAMES = [('fixed_c6', 'Fixed $c_{\\max}=6$'), ('fixed_priority', 'Threshold rule'), ('ga', 'GA-tuned rule'),
         ('autoscaler_hpa', 'Autoscaler (HPA)'), ('queueing_model', 'Queueing model'),
         ('adaptive_model', 'Self-calibrating model'), ('ppo', 'PPO (single agent)'),
         ('ppo_ensemble', 'PPO ensemble'), ('hybrid_adaptive', 'v1.1 hybrid (persistence + PPO)'),
         ('model_fallback_c6', 'Model + persistence + $c_{\\max}$'), ('cusum_fallback_c6', 'Model + CUSUM + $c_{\\max}$'),
         ('cusum_hybrid_ppo', 'Model + CUSUM + PPO'), ('cusum_floor_ppo', 'Model + CUSUM + PPO, model floor'),
         ('hybrid3_A', 'Model + CUSUM + PPO, autoscaler floor'), ('hybrid3_B', 'Proposed hybrid (Hybrid B)'),
         ('oracle_model', 'Oracle (perfect information)')]
REF = 'hybrid3_B'; COMP = [p for p, _ in NAMES if p != REF]
pe = df.groupby(['scenario', 'policy', 'eval_seed']).total_reward.mean().reset_index()
w = pe.groupby(['scenario', 'policy']).total_reward.mean().unstack()
bf = w[['fixed_c3', 'fixed_c4', 'fixed_c5', 'fixed_c6']].max(axis=1)
P = [p for p, _ in NAMES]
sf = w[P].rsub(bf, axis=0); sf['group'] = sf.index.map(grp)
rows = []
for sc in w.index:
    a = pe[(pe.scenario == sc) & (pe.policy == REF)].set_index('eval_seed').total_reward
    ps, ds = [], []
    for c in COMP:
        b = pe[(pe.scenario == sc) & (pe.policy == c)].set_index('eval_seed').total_reward
        d = (a - b).dropna(); ds.append(d.mean())
        ps.append(1.0 if np.allclose(d, 0) else st.wilcoxon(d).pvalue)
    adj = _holm(np.array(ps))
    for c, d, p in zip(COMP, ds, adj):
        rows.append(dict(scenario=sc, comp=c, res='+' if p < .05 and d > 0 else '-' if p < .05 else '='))
T = pd.DataFrame(rows)
cnt = T.groupby('comp').res.value_counts().unstack(fill_value=0)
tab = pd.concat([sf.groupby('group')[P].mean().loc[['seen', 'unseen', 'parameter', 'structural']].T,
                 sf[P].mean().rename('mean'), sf[P].max().rename('worst')], axis=1).round(3)
tab['hybridB_vs'] = [('—' if p == REF else f"{cnt.loc[p].get('+', 0)}/{cnt.loc[p].get('=', 0)}/{cnt.loc[p].get('-', 0)}") for p in P]
tab.index = [dict(NAMES)[p] for p in P]
tab.to_csv(OUT + 'final_summary.csv')
with open(OUT + 'final_summary.md', 'w') as f:
    f.write('| Controller | Seen | Unseen | Param. | Struct. | Mean | Worst | Hybrid B better/same/worse |\n|---|---|---|---|---|---|---|---|\n')
    for i, r in tab.iterrows():
        f.write(f"| {i} | {r.seen:.3f} | {r.unseen:.3f} | {r.parameter:.3f} | {r.structural:.3f} | {r['mean']:.3f} | {r.worst:.3f} | {r.hybridB_vs} |\n")
pd.set_option('display.width', 250); print(tab.to_string())
