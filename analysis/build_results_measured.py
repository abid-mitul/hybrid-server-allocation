"""Revision item 1: rebuild the 69-scenario shortfall table and the paired tests
with the non-structural scenarios re-run under measured utilisation
(results/revision/measured_util/, copied from data/measured/ after
rerun_measured_util.py), and compare
them with the published results.  Output: results/analysis_measured/
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats as st
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from stats import _holm

RES = os.path.join(ROOT, 'results'); H = os.path.join(RES, 'controllers') + '/'
NETS = ['stable', 'degraded', 'noisy']
OUT = os.path.join(RES, 'analysis_measured') + '/'
os.makedirs(OUT, exist_ok=True)
FX = ['fixed_c3', 'fixed_c4', 'fixed_c5', 'fixed_c6']
POL = ['fixed_c6', 'fixed_priority', 'ga', 'queueing_model', 'adaptive_model', 'ppo',
       'ppo_ensemble', 'hybrid', 'hybrid_ensemble', 'hybrid_adaptive']
REF = 'hybrid_adaptive'
COMP = ['fixed_c6', 'fixed_priority', 'ga', 'queueing_model', 'adaptive_model', 'ppo', 'ppo_ensemble']


def grp(sc):
    if sc.startswith(('highvar', 'bursty', 'mixedspeed', 'batch', 'svcburst')): return 'structural'
    if sc.startswith(('slowservers', 'fastservers', 'congestion', 'failures', 'combined')): return 'parameter'
    if sc.startswith(('unseen', 'drop')): return 'unseen'
    return 'seen'


def published():
    parts = [pd.read_csv(os.path.join(RES, 'main', 'evaluation_results.csv'))]
    parts += [pd.read_csv(os.path.join(RES, 'parameter_mismatch', f'eval_extra_{n}.csv')) for n in NETS]
    parts += [pd.read_csv(H + f'eval_{k}_{n}.csv')
              for k in ['hybrid', 'variants', 'hybadapt', 'structural', 'structural2'] for n in NETS]
    return pd.concat(parts, ignore_index=True)


def revised():
    old = published()
    old['group'] = old.scenario.map(grp)
    struct = old[old.group == 'structural']
    new = pd.concat([pd.read_csv(os.path.join(RES, 'revision', 'measured_util', f'eval_{n}.csv'))
                     for n in NETS], ignore_index=True)
    return pd.concat([struct, new], ignore_index=True)


def analyse(df):
    df = df.copy()
    df['group'] = df.scenario.map(grp)
    df = df[df.policy.isin(FX + POL)]
    assert df.groupby(['scenario', 'policy', 'train_seed', 'eval_seed']).size().max() == 1
    pe = df.groupby(['scenario', 'policy', 'eval_seed']).total_reward.mean().reset_index()
    w = pe.groupby(['scenario', 'policy']).total_reward.mean().unstack()
    bf = w[FX].max(axis=1)
    sf = w[POL].rsub(bf, axis=0)
    sf['group'] = sf.index.map(grp); sf['best_fixed'] = bf
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
    sec = df.groupby(['scenario', 'policy'])[['ppo_fraction', 'handovers', 'mean_c']].mean()
    return sf, T, sec


def counts(T):
    out = {}
    for comp in COMP:
        x = T[T.comp == comp]
        out[comp] = {g: f"{(xx.res == '+').sum()}/{(xx.res == '=').sum()}/{(xx.res == '-').sum()}"
                     for g, xx in list(x.groupby('group')) + [('all', x)]}
    return pd.DataFrame(out).T[['seen', 'unseen', 'parameter', 'structural', 'all']]


if __name__ == "__main__":
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    sf_o, T_o, _ = analyse(published())
    sf_n, T_n, sec_n = analyse(revised())
    sf_n.to_csv(OUT + 'shortfall.csv'); T_n.to_csv(OUT + 'tests.csv', index=False)
    sec_n.to_csv(OUT + 'secondary.csv')
    show = ['fixed_c6', 'fixed_priority', 'ga', 'queueing_model', 'adaptive_model', 'ppo',
            'ppo_ensemble', 'hybrid_adaptive']
    gm = lambda sf: pd.concat([sf.groupby('group')[show].mean().T,
                               sf[show].mean().rename('ALL mean'), sf[show].max().rename('ALL worst')], axis=1)
    print("=== Group mean shortfall: PUBLISHED (oracle utilisation)\n", gm(sf_o).round(3))
    print("\n=== Group mean shortfall: REVISED (measured utilisation)\n", gm(sf_n).round(3))
    d = (sf_n[show] - sf_o[show]).join(sf_n[['group']])
    print("\n=== Largest per-scenario changes (revised - published)")
    long = d.melt(id_vars='group', ignore_index=False).reset_index()
    long['abs'] = long.value.abs()
    print(long.sort_values('abs', ascending=False).head(15).round(4).to_string(index=False))
    print("\n=== Paired tests, hybrid vs each (better/same/worse): PUBLISHED\n", counts(T_o))
    print("\n=== Paired tests: REVISED\n", counts(T_n))
    par = sf_n[sf_n.group == 'parameter'][show].round(3)
    print("\n=== Revised parameter-mismatch scenarios\n", par)
    par.to_csv(OUT + 'parameter_mismatch_shortfall.csv')
    print("\nwithin 0.05 of best fixed:",
          {p: int((sf_n[p] <= 0.05).sum()) for p in ['adaptive_model', 'hybrid_adaptive']})
