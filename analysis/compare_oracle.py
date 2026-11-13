"""Revision item 3: compare the controllers with the oracle adaptive baseline
(evaluate_oracle.py), on the revised results with measured utilisation.

Reports, per scenario group,
  * the oracle's shortfall relative to the best fixed allocation (negative:
    adapting with perfect information beats every fixed allocation);
  * every controller's gap to the oracle, i.e. reward of the oracle minus reward
    of the controller (the true regret where the model's structure holds);
  * paired Wilcoxon tests of the proposed hybrid against the oracle.
Output: results/analysis_measured/oracle_*.csv
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_results_measured import revised, grp, FX, OUT, RES

NETS = ['stable', 'degraded', 'noisy']
OR, REF, FB = 'oracle_model', 'hybrid_adaptive', 'model_fallback_c6'
show = ['fixed_c6', 'queueing_model', 'adaptive_model', 'ppo', 'ppo_ensemble', REF, FB]
ORDER = ['seen', 'unseen', 'parameter', 'structural']

df = pd.concat([revised()]
               + [pd.read_csv(os.path.join(RES, 'revision', 'fallback', f'eval_fallback_{n}.csv')) for n in NETS]
               + [pd.read_csv(os.path.join(RES, 'revision', 'oracle', f'eval_oracle_{n}.csv')) for n in NETS],
               ignore_index=True)
pe = df.groupby(['scenario', 'policy', 'eval_seed']).total_reward.mean().reset_index()
w = pe.groupby(['scenario', 'policy']).total_reward.mean().unstack()
bf = w[FX].max(axis=1)
g = pd.Series(w.index.map(grp), index=w.index)

sf_or = bf - w[OR]                                     # oracle vs best fixed
gap = w[show].rsub(w[OR], axis=0)                      # oracle minus controller
gap['group'] = g

rows = []
for sc in w.index:
    a = pe[(pe.scenario == sc) & (pe.policy == REF)].set_index('eval_seed').total_reward
    b = pe[(pe.scenario == sc) & (pe.policy == OR)].set_index('eval_seed').total_reward
    d = (a - b).dropna()
    p = 1.0 if np.allclose(d, 0) else st.wilcoxon(d).pvalue
    rows.append(dict(scenario=sc, group=g[sc], diff=d.mean(), p=p))
T = pd.DataFrame(rows)
T['res'] = np.where(T.p < .05, np.where(T['diff'] > 0, '+', '-'), '=')

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print("=== Oracle shortfall vs best fixed allocation (negative = better than every fixed c)")
print(sf_or.groupby(g).agg(['mean', 'min', 'max']).loc[ORDER].round(3))
print("\n=== Gap to the oracle (oracle reward minus controller reward), group means")
tab = gap.groupby('group')[show].mean().loc[ORDER].T
tab['in-model (39)'] = gap[gap.group != 'structural'][show].mean()
tab['all (69)'] = gap[show].mean()
print(tab.round(3))
rel = (gap[show].div(w[OR], axis=0) * 100)
print("\nHybrid gap to oracle, %% of oracle reward: in-model %.2f%%, all %.2f%%"
      % (rel[gap.group != 'structural'][REF].mean(), rel[REF].mean()))
print("\n=== Hybrid vs oracle, paired Wilcoxon p<0.05 (better/same/worse):",
      {k: f"{(x.res == '+').sum()}/{(x.res == '=').sum()}/{(x.res == '-').sum()}"
       for k, x in list(T.groupby('group')) + [('all', T)]})
big = gap[gap.group != 'structural'][[REF, 'adaptive_model']].sort_values(REF, ascending=False).head(8)
print("\nLargest in-model gaps of the hybrid to the oracle:\n", big.round(3))
st_ = pd.DataFrame({'oracle_vs_bestfixed': sf_or, 'hybrid_vs_oracle': gap[REF],
                    'fallback_vs_oracle': gap[FB]})[g == 'structural']
print("\n=== Structural scenarios\n", st_.round(3).to_string())
pd.DataFrame({'oracle_shortfall': sf_or}).join(gap).to_csv(OUT + 'oracle_gap.csv')
T.to_csv(OUT + 'oracle_tests.csv', index=False)
