"""Consolidate all evaluation results (69 scenarios): shortfalls, paired tests,
secondary measures and the server-cost study. Output: results/analysis/
"""
import numpy as np, pandas as pd
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scipy import stats as st
from stats import _holm
RES = os.path.join(ROOT, 'results'); H = os.path.join(RES, 'controllers') + '/'
N = ['stable', 'degraded', 'noisy']; OUT = os.path.join(RES, 'analysis') + '/'
os.makedirs(OUT, exist_ok=True)
parts = [pd.read_csv(os.path.join(RES, 'main', 'evaluation_results.csv'))]
parts += [pd.read_csv(os.path.join(RES, 'parameter_mismatch', f'eval_extra_{n}.csv')) for n in N]
parts += [pd.read_csv(H + f'eval_{k}_{n}.csv') for k in ['hybrid', 'variants', 'hybadapt', 'structural', 'structural2'] for n in N]
df = pd.concat(parts, ignore_index=True)
# drop duplicated policy runs: structural file re-runs everything for structural scenarios only (no overlap)
def grp(sc):
    if sc.startswith(('highvar', 'bursty', 'mixedspeed', 'batch', 'svcburst')): return 'structural'
    if sc.startswith(('slowservers', 'fastservers', 'congestion', 'failures', 'combined')): return 'parameter'
    if sc.startswith(('unseen', 'drop')): return 'unseen'
    return 'seen'
df['group'] = df.scenario.map(grp)
df['type'] = df.scenario.str.rsplit('_', n=1).str[0]
df['net'] = df.scenario.str.rsplit('_', n=1).str[1]
assert df.groupby(['scenario', 'policy', 'train_seed', 'eval_seed']).size().max() == 1
df.to_csv(OUT + 'all_episodes.csv', index=False)
pe = df.groupby(['scenario', 'policy', 'eval_seed']).total_reward.mean().reset_index()
w = pe.groupby(['scenario', 'policy']).total_reward.mean().unstack()
FX = ['fixed_c3', 'fixed_c4', 'fixed_c5', 'fixed_c6']
bf = w[FX].max(axis=1); bc = w[FX].idxmax(axis=1).str[-1].astype(int)
POL = ['fixed_c6', 'fixed_priority', 'ga', 'queueing_model', 'adaptive_model', 'ppo', 'ppo_ensemble',
       'hybrid', 'hybrid_ensemble', 'hybrid_adaptive']
sf = w[POL].rsub(bf, axis=0)
meta = df.drop_duplicates('scenario').set_index('scenario')[['group', 'type', 'net']]
sf = sf.join(meta); sf['best_fixed'] = bf; sf['best_c'] = bc
sf.to_csv(OUT + 'shortfall.csv')
# relative shortfall (% of best fixed reward)
rel = sf[POL].div(bf, axis=0) * 100; rel.join(meta).to_csv(OUT + 'shortfall_rel.csv')
# paired tests: proposed hybrid vs every other policy, Holm within each scenario
REF = 'hybrid_adaptive'; COMP = ['fixed_c6', 'fixed_priority', 'ga', 'queueing_model', 'adaptive_model', 'ppo', 'ppo_ensemble']
rows = []
for sc in w.index:
    a = pe[(pe.scenario == sc) & (pe.policy == REF)].set_index('eval_seed').total_reward
    ps = []
    for comp in COMP:
        b = pe[(pe.scenario == sc) & (pe.policy == comp)].set_index('eval_seed').total_reward
        d = (a - b).dropna()
        p = 1.0 if np.allclose(d, 0) else st.wilcoxon(d).pvalue
        h = st.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        ps.append(p); rows.append(dict(scenario=sc, comp=comp, diff=d.mean(), ci=h, p=p))
    adj = _holm(np.array(ps))
    for i in range(len(COMP)): rows[-len(COMP) + i]['p_holm'] = adj[i]
T = pd.DataFrame(rows).merge(meta, left_on='scenario', right_index=True)
T['res'] = np.where(T.p_holm < .05, np.where(T['diff'] > 0, '+', '-'), '=')
T.to_csv(OUT + 'tests.csv', index=False)
# CI half-width of mean reward per scenario/policy (for text)
ci = pe.groupby(['scenario', 'policy']).total_reward.agg(lambda x: st.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
ci.to_csv(OUT + 'ci_halfwidth.csv')
# secondary metrics
sec = df.groupby(['scenario', 'policy'])[['mean_c', 'n_switches', 'sla_compliance', 'p_b', 'mean_sojourn_ms', 'ppo_fraction', 'handovers']].mean()
sec.to_csv(OUT + 'secondary.csv')
# PPO seed variation
sd = df[df.policy.isin(['ppo', 'hybrid'])].groupby(['scenario', 'policy', 'train_seed']).total_reward.mean().reset_index()
sd['sf'] = sd.scenario.map(bf) - sd.total_reward
sd.to_csv(OUT + 'seed_shortfall.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(sf.groupby('group')[POL].mean().round(3).T)
print('ALL mean', sf[POL].mean().round(3).to_dict()); print('worst', sf[POL].max().round(3).to_dict())
for comp in COMP:
    x = T[T.comp == comp]
    print(comp, {g: f"{(xx.res=='+').sum()}/{(xx.res=='=').sum()}/{(xx.res=='-').sum()}" for g, xx in [('all', x)] + list(x.groupby('group'))})
print(len(sf), 'scenarios')

# server-cost study (w_C = 0.10, stable network, 8 scenarios)
a = pd.read_csv(os.path.join(RES, 'server_cost_x2', 'evaluation_results.csv'))
b = pd.read_csv(os.path.join(RES, 'server_cost_x2', 'eval_wc_variants.csv'))
pw = pd.concat([a, b]).groupby(['scenario', 'policy', 'eval_seed']).total_reward.mean().reset_index()
ww = pw.groupby(['scenario', 'policy']).total_reward.mean().unstack()
ww[POL].rsub(ww[FX].max(axis=1), axis=0).to_csv(OUT + 'shortfall_wc010.csv')
