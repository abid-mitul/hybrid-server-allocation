"""Figures of the revised paper (Hybrid B), in the style of make_figures.py.
Run after the revision experiments; the figures of the model itself
(fig3_queue_metrics, fig4_static_optimum) and the learning curves are unchanged.

Output (figures/):
  fig_shortfall_groups_v2.pdf   mean shortfall by scenario group, revised controllers
  fig_hybridB_trace.pdf         one bursty-traffic episode: CUSUM, servers, compliance
  fig_sensitivity_v2.pdf        switching thresholds, original + extended parameters
  fig_cusum_detection.pdf       first-alarm window, persistence test vs CUSUM
  fig_server_cost.pdf           mean and worst shortfall at normal and doubled cost
"""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from build_results_measured import revised, grp

OUT = os.path.join(ROOT, 'figures') + '/'; RES = os.path.join(ROOT, 'results')
RV = os.path.join(RES, 'revision'); V = os.path.join(RES, 'analysis') + '/'
NETS = ['stable', 'degraded', 'noisy']; CATS = ['seen', 'unseen', 'parameter', 'structural']
CLAB = ['Seen', 'Unseen', 'Parameter\nmismatch', 'Structural\nmismatch']
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.linewidth': 0.6, 'axes.edgecolor': '#555',
                     'xtick.color': '#555', 'ytick.color': '#555', 'axes.labelcolor': '#222',
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
GRID = dict(axis='y', color='#e6e6e3', lw=0.5)
PC = {'autoscaler_hpa': '#1baf7a', 'adaptive_model': '#a3401a', 'ppo_ensemble': '#2a78d6',
      'cusum_fallback_c6': '#eda100', 'hybrid3_B': '#222222'}
PL = {'autoscaler_hpa': 'Autoscaler', 'adaptive_model': 'Self-calibrating model',
      'ppo_ensemble': 'PPO ensemble', 'cusum_fallback_c6': 'Model + CUSUM + $c_{\\max}$',
      'hybrid3_B': 'CHAMPS'}
rd = lambda sub, pat: pd.concat([pd.read_csv(os.path.join(RV, sub, pat.format(n))) for n in NETS], ignore_index=True)


def shortfall(df, pols):
    pe = df.groupby(['scenario', 'policy', 'eval_seed']).total_reward.mean().reset_index()
    w = pe.groupby(['scenario', 'policy']).total_reward.mean().unstack()
    sf = w[pols].rsub(w[['fixed_c3', 'fixed_c4', 'fixed_c5', 'fixed_c6']].max(axis=1), axis=0)
    sf['group'] = sf.index.map(grp)
    return sf


# ---------------------------------------------------------------- data, normal cost
rv = revised()
cu = rd('cusum', 'eval_cusum_{}.csv')
h3 = rd('hybrid3', 'eval_h3_{}.csv')
base = pd.concat([rv[rv.policy.str.startswith('fixed') | rv.policy.isin(['adaptive_model', 'ppo_ensemble'])],
                  rd('autoscaler', 'eval_hpa_{}.csv'), cu[cu.policy == 'cusum_fallback_c6'],
                  h3[h3.policy == 'hybrid3_B']], ignore_index=True)
pols = list(PC)
sf = shortfall(base, pols)

# ---- Fig 1: mean shortfall by group
m = sf.groupby('group')[pols].mean().loc[CATS]
fig, ax = plt.subplots(figsize=(3.45, 2.35)); W = 0.15
for j, p in enumerate(pols):
    xs = np.arange(len(CATS)) + (j - 2) * (W + 0.008)
    ax.bar(xs, m[p], width=W, color=PC[p], label=PL[p], zorder=2, edgecolor='white', linewidth=0.3)
ax.set_xticks(np.arange(len(CATS))); ax.set_xticklabels(CLAB); ax.axhline(0, color='#555', lw=0.6)
ax.set_ylabel('Mean shortfall'); ax.grid(**GRID); ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0), handlelength=1.1, columnspacing=0.8)
fig.savefig(OUT + 'fig_shortfall_groups_v2.pdf', bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- Fig 2: Hybrid B trace
from fast_env import make_fast_env, env_options, T_SLA, C_MIN, C_MAX, ACTION_TO_C
from fast_evaluate import PPOPolicy, env_kwargs_for
from variants import EnsemblePolicy, AdaptiveModelPolicy
from hybrid3 import Hybrid3Policy, ALL

cfg = json.load(open(os.path.join(ROOT, 'experiment_config.json'))); tr, ev = cfg['train'], cfg['eval']
opt = env_options(tr); kw = dict(w_c=opt['w_c'], w_switch=opt['w_switch'], deg_cong=opt['degraded_congestion'])
NET, SC, SEED = 'stable', 'bursty280_stable', ev['seed_base']


class TraceHybrid3(Hybrid3Policy):
    def predict(self, obs, deterministic=True):
        a_hpa, _ = self.hpa.predict(obs)
        r = self.ratio(obs)
        self.S = max(0.0, self.S + np.log(max(r, 1e-9)) - self.k_ref)
        self.run_lo = 0 if r > self.threshold else self.run_lo + 1
        if not self.use_ppo and self.S > self.h:
            self.use_ppo = True; self.n_handover += 1
        elif self.use_ppo and self.run_lo >= self.n_off:
            self.use_ppo, self.S = False, 0.0
        a_model, _ = self.model.predict(obs)
        a_ppo = int(self.ppo.predict(obs)[0])
        a = max(a_ppo, int(a_hpa), int(a_model)) if self.use_ppo else int(a_model)
        self.log.append(dict(ratio=r, S=self.S, fallback=self.use_ppo, c=ACTION_TO_C[a],
                             c_model=ACTION_TO_C[int(a_model)], c_ppo=ACTION_TO_C[a_ppo],
                             c_hpa=ACTION_TO_C[int(a_hpa)]))
        if self.use_ppo:
            self.ppo_steps += 1
        return a, None


thr = json.load(open(os.path.join(ROOT, 'cusum_thresholds.json')))[NET]
tg = json.load(open(os.path.join(ROOT, 'autoscaler_targets.json')))[NET]['target']
ens = EnsemblePolicy([PPOPolicy(os.path.join(ROOT, 'models', NET, f'ppo_seed_{s}.zip'), 'ppo', s) for s in tr['seeds']])
hb = TraceHybrid3(NET, ens, threshold=thr['tau'], h=thr['h'], hpa_target=tg, use_model_floor=True, name='hybrid3_B', **kw)
am = AdaptiveModelPolicy(NET, **kw)
k = env_kwargs_for(ALL(NET)[SC], tr, ev['episode_duration']); k['measured_util'] = True


def trace(pol):
    env = make_fast_env(**k); obs, _ = env.reset(seed=SEED); pol.reset(); pol.log = []
    out, done = [], False
    while not done:
        a, _ = pol.predict(obs)
        obs, rwd, done, _, info = env.step(a)
        out.append(dict(c=info['c'], sla=info['sla_compliance']))
    return pd.DataFrame(out), pd.DataFrame(pol.log) if pol.log else None


ta, _ = trace(am); th, lg = trace(hb)
st = np.arange(1, len(th) + 1)
fig, axs = plt.subplots(3, 1, figsize=(4.6, 4.2), sharex=True, gridspec_kw={'hspace': 0.22})
ax = axs[0]; ax.plot(st, lg.ratio, color='#222', lw=1.2, marker='o', ms=2)
ax.axhline(thr['tau'], color='#d03b3b', lw=0.8, ls='--')
ax.text(30.4, thr['tau'], 'threshold $\\psi^{*}$', color='#d03b3b', fontsize=8, va='bottom', ha='right')
for s, fb in zip(st, lg.fallback):
    if fb:
        for a in axs: a.axvspan(s - 0.5, s + 0.5, color='#2a78d6', alpha=0.10, lw=0)
ax.set_ylabel('Measured/\npredicted'); ax.grid(**GRID)
ax = axs[1]
ax.step(st, ta.c, where='mid', color=PC['adaptive_model'], lw=1.2, label=PL['adaptive_model'])
ax.step(st, th.c + 0.06, where='mid', color='#222', lw=1.2, label=PL['hybrid3_B'])
ax.set_ylabel('Servers $c$'); ax.set_yticks([3, 4, 5, 6]); ax.set_ylim(2.7, 6.4); ax.grid(**GRID)
ax.legend(frameon=False, loc='lower left', ncol=2, handlelength=1.4)
ax = axs[2]
ax.plot(st, ta.sla, color=PC['adaptive_model'], lw=1.1, marker='o', ms=2)
ax.plot(st, th.sla, color='#222', lw=1.1, marker='o', ms=2)
ax.set_ylabel('Compliance'); ax.set_xlabel('Decision $k$ (60 s windows)'); ax.grid(**GRID); ax.set_xlim(0.5, 30.5)
for i, ax in enumerate(axs): ax.text(0.005, 1.02, '(%s)' % 'abc'[i], transform=ax.transAxes, fontsize=8, va='bottom')
fig.savefig(OUT + 'fig_hybridB_trace.pdf', bbox_inches='tight'); plt.close(fig)
lg.assign(c_applied=th.c, sla=th.sla).to_csv(os.path.join(RES, 'analysis_measured', 'trace_hybridB_bursty280_stable.csv'), index=False)

# ---------------------------------------------------------------- Fig 3: sensitivity (two regimes)
rows = pd.DataFrame(json.load(open(V + 'sensitivity_extended.json')))
lab = {'mu': '$\\mu$', 'fail': '$q_{01},q_{12}$', 'b10': '$q_{10}$', 'b21': '$q_{21}$', 'lam1': '$\\lambda_1$',
       'eta': '$\\kappa$', 'beta': '$\\beta$', 'd0': '$\\delta_0$', 'T': '$T_{\\mathrm{SLA}}$', 'K': '$K$',
       'wc': '$w_C$', 'xi': '$\\xi$'}
CO = ['#2a78d6', '#eb6834', '#1baf7a']; KEYS = ['3->4', '4->5', '5->6']


def panel(ax, mode, order):
    r = rows[rows['mode'] == mode]; nom = r[r.param == 'nominal'].iloc[0].th
    yl = []
    for i, (p, s) in enumerate(order):
        rr = r[(r.param == p) & np.isclose(r.scale.astype(float), s)].iloc[0]
        for j, key in enumerate(KEYS):
            v = rr.th.get(key, np.nan); ax.plot([nom[key], v], [i, i], color='#bbb', lw=0.6, zorder=1)
            ax.scatter(v, i, s=16, color=CO[j], zorder=2, label=['$3\\to4$', '$4\\to5$', '$5\\to6$'][j] if i == 0 else None)
        if p == 'xi':
            yl.append(f"$\\xi={2.5 * s:g}$")
        elif p == 'beta':
            yl.append(f"$\\beta={1.4 * s:.1f}$")
        else:
            yl.append(f"{lab[p]} $\\times${s:g}".replace('0.666667', '2/3').replace('1.33333', '4/3'))
    for key, c in zip(KEYS, CO): ax.axvline(nom[key], color=c, lw=0.6, ls=':')
    ax.set_yticks(range(len(order))); ax.set_yticklabels(yl, fontsize=8); ax.invert_yaxis()
    ax.grid(axis='x', color='#e6e6e3', lw=0.5)


A = [('mu', 0.7), ('mu', 0.85), ('mu', 1.15), ('mu', 1.3), ('fail', 0.5), ('fail', 1.5), ('fail', 2.0),
     ('b10', 0.5), ('b10', 2.0), ('b21', 0.5), ('b21', 2.0), ('lam1', 0.5), ('lam1', 1.5),
     ('eta', 0.5), ('eta', 2.0), ('beta', 1.2 / 1.4), ('beta', 1.6 / 1.4), ('d0', 0.6), ('d0', 1.6),
     ('T', 2 / 3), ('T', 4 / 3), ('K', 0.6), ('K', 2.0), ('wc', 0.5), ('wc', 2.0)]
B = [('xi', 2.1 / 2.5), ('xi', 3.0 / 2.5), ('xi', 4.0 / 2.5)]
fig = plt.figure(figsize=(6.4, 4.6))
gs = fig.add_gridspec(len(A), 2, width_ratios=[1, 1], wspace=0.42)
ax0 = fig.add_subplot(gs[:, 0]); ax1 = fig.add_subplot(gs[0:len(B) + 2, 1])
panel(ax0, 'stable', A); panel(ax1, 'degraded', B)
ax0.set_xlabel('Population size $N$ at which\nthe static optimum switches')
ax1.set_xlabel('Population size $N$ at which\nthe static optimum switches')
ax0.text(0.0, 1.01, '(a) stable network', transform=ax0.transAxes, fontsize=9, va='bottom')
ax1.text(0.0, 1.03, '(b) degraded network, Pareto shape', transform=ax1.transAxes, fontsize=9, va='bottom')
h, l = ax0.get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc='upper left', bbox_to_anchor=(0.6, 0.55), fontsize=8.5, handletextpad=0.3,
           title='Switch of the static optimum', title_fontsize=8.5)
fig.savefig(OUT + 'fig_sensitivity_v2.pdf', bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- Fig 4: CUSUM detection speed
al = cu[cu.policy.isin(['persistence_probe', 'cusum_fallback_c6'])].copy()
al = al[al.scenario.map(grp) == 'structural']
al['kind'] = al.scenario.str.extract(r'^([a-z]+)')[0]
names = {'highvar': 'High\nvariance', 'bursty': 'Bursty\ntraffic', 'mixedspeed': 'Mixed\nspeeds',
         'batch': 'Batch\narrivals', 'svcburst': 'Bursty\nservice'}
fa = al[al.first_alarm >= 0].groupby(['kind', 'policy']).first_alarm.mean().unstack().reindex(list(names))
fig, ax = plt.subplots(figsize=(3.45, 2.1)); W = 0.34
for j, (p, c, l) in enumerate([('persistence_probe', '#bbbbbb', 'Persistence test ($m_{\\mathrm{on}}=2$)'),
                               ('cusum_fallback_c6', '#222222', 'CUSUM')]):
    ax.bar(np.arange(len(fa)) + (j - 0.5) * (W + 0.02), fa[p] + 1, width=W, color=c, label=l, zorder=2)
ax.set_xticks(range(len(fa))); ax.set_xticklabels([names[x] for x in fa.index])
ax.set_ylabel('Window of first alarm'); ax.set_ylim(0, 5); ax.grid(**GRID); ax.set_axisbelow(True)
ax.legend(frameon=False, loc='upper right')
fig.savefig(OUT + 'fig_cusum_detection.pdf', bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- Fig 5: server cost
P5 = ['autoscaler_hpa', 'cusum_fallback_c6', 'cusum_floor_ppo', 'hybrid3_B']
L5 = {'autoscaler_hpa': 'Autoscaler', 'cusum_fallback_c6': '$c_{\\max}$ fallback',
      'cusum_floor_ppo': 'PPO, model floor', 'hybrid3_B': 'CHAMPS'}
fl = rd('safety_floor', 'eval_floor_{}.csv')
n1 = shortfall(pd.concat([base, fl], ignore_index=True), P5)
x2s = pd.read_csv(os.path.join(RV, 'server_cost_x2', 'eval_stable.csv'))
x2s = pd.concat([x2s[x2s.policy.str.startswith('fixed') | (x2s.policy == 'cusum_fallback_c6')],
                 pd.read_csv(os.path.join(RV, 'autoscaler', 'eval_hpa_wc010.csv')),
                 pd.read_csv(os.path.join(RV, 'safety_floor', 'eval_floor_wc010.csv')),
                 pd.read_csv(os.path.join(RV, 'hybrid3', 'eval_h3_wc010.csv'))], ignore_index=True)
x2o = pd.concat([pd.read_csv(os.path.join(RV, 'final_checks', f'eval_x2_{n}.csv')) for n in ('degraded', 'noisy')])
n2 = shortfall(pd.concat([x2s, x2o], ignore_index=True), P5)
fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.0), sharey=False, gridspec_kw={'wspace': 0.35})
for ax, stat, t in zip(axs, ['mean', 'max'], ['(a) mean shortfall', '(b) worst case']):
    v = pd.DataFrame({'Normal': getattr(n1[P5], stat)(), 'Doubled': getattr(n2[P5], stat)()})
    for j, p in enumerate(P5):
        ax.bar(np.arange(2) + (j - 1.5) * 0.2, v.loc[p], width=0.18, color=['#1baf7a', '#eda100', '#8fb8ea', '#222'][j],
               label=L5[p], zorder=2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(v.columns); ax.grid(**GRID); ax.set_axisbelow(True)
    ax.text(0.0, 1.02, t, transform=ax.transAxes, fontsize=7, va='bottom'); ax.set_xlabel('Server cost')
axs[0].legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(1.1, 1.12), handlelength=1.1, columnspacing=0.8)
fig.savefig(OUT + 'fig_server_cost.pdf', bbox_inches='tight'); plt.close(fig)
print('ok')
