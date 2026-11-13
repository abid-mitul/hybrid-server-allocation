"""Revision item 2: compare the proposed hybrid with the "model + c_max fallback"
baseline on all 69 scenarios (revised results with measured utilisation).

Paired two-sided Wilcoxon tests on the 20 common evaluation seeds; within each
scenario the Holm correction now covers eight comparisons (the seven of
Table 8 plus the fallback baseline).  Output: results/analysis_measured/
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_results_measured import revised, grp, FX, OUT, RES
from stats import _holm

NETS = ['stable', 'degraded', 'noisy']
FB = 'model_fallback_c6'; REF = 'hybrid_adaptive'
COMP = ['fixed_c6', 'fixed_priority', 'ga', 'queueing_model', 'adaptive_model', 'ppo',
        'ppo_ensemble', FB]

df = pd.concat([revised()] + [pd.read_csv(os.path.join(RES, 'revision', 'fallback', f'eval_fallback_{n}.csv'))
                              for n in NETS], ignore_index=True)
df['group'] = df.scenario.map(grp)
pe = df.groupby(['scenario', 'policy', 'eval_seed']).total_reward.mean().reset_index()
w = pe.groupby(['scenario', 'policy']).total_reward.mean().unstack()
bf = w[FX].max(axis=1)
show = ['fixed_c6', 'adaptive_model', 'ppo_ensemble', REF, FB]
sf = w[show].rsub(bf, axis=0); sf['group'] = sf.index.map(grp)

rows = []
for sc in w.index:
    a = pe[(pe.scenario == sc) & (pe.policy == REF)].set_index('eval_seed').total_reward
    ps = []
    for comp in COMP:
        b = pe[(pe.scenario == sc) & (pe.policy == comp)].set_index('eval_seed').total_reward
        d = (a - b).dropna()
        p = 1.0 if np.allclose(d, 0) else st.wilcoxon(d).pvalue
        ps.append(p); rows.append(dict(scenario=sc, comp=comp, diff=d.mean(), p=p))
    adj = _holm(np.array(ps))
    for i in range(len(COMP)): rows[-len(COMP) + i]['p_holm'] = adj[i]
T = pd.DataFrame(rows); T['group'] = T.scenario.map(grp)
T['res'] = np.where(T.p_holm < .05, np.where(T['diff'] > 0, '+', '-'), '=')

sec = df[df.policy.isin([REF, FB])].groupby(['scenario', 'policy'])[['ppo_fraction', 'mean_c', 'n_switches']].mean().unstack()

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print("=== Mean shortfall by group\n",
      pd.concat([sf.groupby('group')[show].mean().T, sf[show].mean().rename('ALL mean'),
                 sf[show].max().rename('ALL worst')], axis=1).round(3))
x = T[T.comp == FB]
print("\n=== Hybrid vs fallback (better/same/worse):",
      {g: f"{(xx.res == '+').sum()}/{(xx.res == '=').sum()}/{(xx.res == '-').sum()}"
       for g, xx in list(x.groupby('group')) + [('all', x)]})
s = sf[sf.group == 'structural'][show].copy()
s['test'] = x.set_index('scenario').res
s['hyb_learned'] = sec[('ppo_fraction', REF)]; s['fb_c6_share'] = sec[('ppo_fraction', FB)]
s['hyb_c'] = sec[('mean_c', REF)]; s['fb_c'] = sec[('mean_c', FB)]
s['hyb_sw'] = sec[('n_switches', REF)]; s['fb_sw'] = sec[('n_switches', FB)]
print("\n=== Structural scenarios\n", s.sort_index().round(3).to_string())
nonstruct = x[x.group != 'structural']
print("\nNon-structural scenarios where the two differ significantly:",
      nonstruct[nonstruct.res != '='][['scenario', 'diff', 'res']].round(3).to_string(index=False))
sf.to_csv(OUT + 'shortfall_with_fallback.csv'); T.to_csv(OUT + 'tests_with_fallback.csv', index=False)
s.to_csv(OUT + 'structural_hybrid_vs_fallback.csv')
