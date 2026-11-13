"""Revision item 4: regime-agnostic controllers (regime_agnostic.py) against
their regime-aware counterparts, on all 69 scenarios with measured utilisation.

Pairs (agnostic -> aware):
  ra_model               -> adaptive_model      (self-calibrating model)
  ra_fallback_c6         -> model_fallback_c6   (model + test, c = 6 on alarm)
  ra_hybrid_ppo_stable   -> hybrid_adaptive     (model + test, PPO ensemble on alarm;
                                                 agnostic: stable-trained agents only)
Output: results/analysis_measured/regime_agnostic_*.csv
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_results_measured import revised, grp, FX, OUT, RES

NETS = ['stable', 'degraded', 'noisy']
ORDER = ['seen', 'unseen', 'parameter', 'structural']
PAIRS = [('ra_model', 'adaptive_model'), ('ra_fallback_c6', 'model_fallback_c6'),
         ('ra_hybrid_ppo_stable', 'hybrid_adaptive')]

ra = pd.concat([pd.read_csv(os.path.join(RES, 'revision', 'regime_agnostic', f'eval_ra_{n}.csv'))
                for n in NETS], ignore_index=True)
df = pd.concat([revised(), ra]
               + [pd.read_csv(os.path.join(RES, 'revision', 'fallback', f'eval_fallback_{n}.csv'))
                  for n in NETS], ignore_index=True)
pe = df.groupby(['scenario', 'policy', 'eval_seed']).total_reward.mean().reset_index()
w = pe.groupby(['scenario', 'policy']).total_reward.mean().unstack()
bf = w[FX].max(axis=1)
cols = [x for p in PAIRS for x in p] + ['ppo_ensemble', 'fixed_c6']
sf = w[cols].rsub(bf, axis=0); sf['group'] = sf.index.map(grp)

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
tab = pd.concat([sf.groupby('group')[cols].mean().loc[ORDER].T,
                 sf[cols].mean().rename('mean'), sf[cols].max().rename('worst')], axis=1)
print("=== Shortfall vs best fixed allocation\n", tab.round(3))

rows = []
for sc in w.index:
    for ag, aw in PAIRS:
        a = pe[(pe.scenario == sc) & (pe.policy == ag)].set_index('eval_seed').total_reward
        b = pe[(pe.scenario == sc) & (pe.policy == aw)].set_index('eval_seed').total_reward
        d = (a - b).dropna()
        p = 1.0 if np.allclose(d, 0) else st.wilcoxon(d).pvalue
        rows.append(dict(scenario=sc, group=grp(sc), pair=ag, diff=d.mean(), p=p))
T = pd.DataFrame(rows)
T['res'] = np.where(T.p < .05, np.where(T['diff'] > 0, '+', '-'), '=')
print("\n=== Agnostic vs aware, paired Wilcoxon p<0.05 (agnostic better/same/worse)")
for ag, _ in PAIRS:
    x = T[T.pair == ag]
    print(f"  {ag:22s}", {k: f"{(y.res == '+').sum()}/{(y.res == '=').sum()}/{(y.res == '-').sum()}"
                          for k, y in [(g, x[x.group == g]) for g in ORDER] + [('all', x)]})

acc = ra.assign(group=ra.scenario.map(grp)).groupby(['group', 'policy']).regime_correct.mean().unstack()
print("\n=== Share of decisions with the correct regime estimate\n", acc.loc[ORDER].round(3))
fb = ra[ra.policy != 'ra_model'].assign(group=ra.scenario.map(grp)).groupby(['group', 'policy']).ppo_fraction.mean().unstack()
print("\n=== Share of decisions in fallback mode\n", fb.loc[ORDER].round(3))
worst = T[T.res == '-'].sort_values('diff').head(10)
print("\n=== Largest significant losses from not knowing the regime\n", worst[['scenario', 'pair', 'diff']].round(3).to_string(index=False))
tab.to_csv(OUT + 'regime_agnostic_shortfall.csv'); T.to_csv(OUT + 'regime_agnostic_tests.csv', index=False)
sf.to_csv(OUT + 'regime_agnostic_by_scenario.csv')
