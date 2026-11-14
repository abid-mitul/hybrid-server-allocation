"""Revision item 5: CUSUM vs persistence test (cusum_test.py), regime-aware,
69 scenarios, measured utilisation, common seeds.

Pairs (CUSUM -> persistence):
  cusum_fallback_c6 -> model_fallback_c6   (c = 6 on alarm)
  cusum_hybrid_ppo  -> hybrid_adaptive     (PPO ensemble on alarm)
Also: detection delay (window of the first alarm) and the share of episodes
with any alarm, from the logged alarm times (persistence_probe re-runs the
original test with c = 6 and must equal model_fallback_c6).
Output: results/analysis_measured/cusum_*.csv
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_results_measured import revised, grp, FX, OUT, RES

NETS = ['stable', 'degraded', 'noisy']
ORDER = ['seen', 'unseen', 'parameter', 'structural']
PAIRS = [('cusum_fallback_c6', 'model_fallback_c6'), ('cusum_hybrid_ppo', 'hybrid_adaptive')]

cu = pd.concat([pd.read_csv(os.path.join(RES, 'revision', 'cusum', f'eval_cusum_{n}.csv'))
                for n in NETS], ignore_index=True)
fb = pd.concat([pd.read_csv(os.path.join(RES, 'revision', 'fallback', f'eval_fallback_{n}.csv'))
                for n in NETS], ignore_index=True)
k = ['scenario', 'eval_seed']
chk = cu[cu.policy == 'persistence_probe'].merge(fb[k + ['total_reward']], on=k)
print("persistence_probe reproduces model_fallback_c6: max |diff| =",
      float((chk.total_reward_x - chk.total_reward_y).abs().max()))

df = pd.concat([revised(), fb, cu[cu.policy != 'persistence_probe']], ignore_index=True)
pe = df.groupby(['scenario', 'policy', 'eval_seed']).total_reward.mean().reset_index()
w = pe.groupby(['scenario', 'policy']).total_reward.mean().unstack()
bf = w[FX].max(axis=1)
cols = [x for p in PAIRS for x in p] + ['fixed_c6']
sf = w[cols].rsub(bf, axis=0); sf['group'] = sf.index.map(grp)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
tab = pd.concat([sf.groupby('group')[cols].mean().loc[ORDER].T,
                 sf[cols].mean().rename('mean'), sf[cols].max().rename('worst')], axis=1)
print("\n=== Shortfall vs best fixed allocation\n", tab.round(3))

rows = []
for sc in w.index:
    for a_, b_ in PAIRS:
        a = pe[(pe.scenario == sc) & (pe.policy == a_)].set_index('eval_seed').total_reward
        b = pe[(pe.scenario == sc) & (pe.policy == b_)].set_index('eval_seed').total_reward
        d = (a - b).dropna()
        p = 1.0 if np.allclose(d, 0) else st.wilcoxon(d).pvalue
        rows.append(dict(scenario=sc, group=grp(sc), pair=a_, diff=d.mean(), p=p))
T = pd.DataFrame(rows); T['res'] = np.where(T.p < .05, np.where(T['diff'] > 0, '+', '-'), '=')
print("\n=== CUSUM vs persistence, paired Wilcoxon p<0.05 (CUSUM better/same/worse)")
for a_, _ in PAIRS:
    x = T[T.pair == a_]
    print(f"  {a_:18s}", {g: f"{(y.res == '+').sum()}/{(y.res == '=').sum()}/{(y.res == '-').sum()}"
                          for g, y in [(g, x[x.group == g]) for g in ORDER] + [('all', x)]})

al = cu[cu.policy.isin(['cusum_fallback_c6', 'persistence_probe'])].copy()
al['group'] = al.scenario.map(grp); al['alarmed'] = al.first_alarm >= 0
det = al.groupby(['group', 'policy']).agg(
    episodes_with_alarm=('alarmed', 'mean'),
    first_alarm_window=('first_alarm', lambda s: s[s >= 0].mean()),
    fallback_share=('ppo_fraction', 'mean'), handovers=('handovers', 'mean')).unstack()
print("\n=== Detection (c = 6 fallback): share of episodes with an alarm, mean window of first alarm\n",
      det.loc[ORDER].round(2))
s = al[al.group == 'structural'].groupby(['scenario', 'policy']).agg(
    fa=('first_alarm', lambda s: s[s >= 0].mean()), share=('ppo_fraction', 'mean')).unstack()
s['sf_cusum'] = sf['cusum_fallback_c6']; s['sf_persist'] = sf['model_fallback_c6']
s['test'] = T[T.pair == 'cusum_fallback_c6'].set_index('scenario').res
print("\n=== Structural scenarios (c = 6 fallback)\n", s.round(2).to_string())
tab.to_csv(OUT + 'cusum_shortfall.csv'); T.to_csv(OUT + 'cusum_tests.csv', index=False)
det.to_csv(OUT + 'cusum_detection.csv'); s.to_csv(OUT + 'cusum_structural.csv')
