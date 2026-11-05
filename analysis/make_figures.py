"""Figures of the paper. Run build_results.py and sensitivity_sweep.py first.
Output: figures/ (PDF) in the repository root."""
import os, sys, json, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = os.path.join(ROOT, 'figures') + '/'; os.makedirs(OUT, exist_ok=True)
RES = os.path.join(ROOT, 'results')
V = os.path.join(RES, 'analysis') + '/'
from model_policy import mmck_probs, delay_cdf, expected_step_reward
from parameters import SERVER, LATENCY
from fast_env import T_SLA
tr=json.load(open(os.path.join(ROOT,'experiment_config.json')))['train']
wc,dc=tr['w_c'],tr['degraded_congestion']; mu,K=SERVER['mu'],SERVER['K']
rate=4.0905
COL={3:'#2a78d6',4:'#eb6834',5:'#1baf7a',6:'#eda100'}; MK={3:'o',4:'s',5:'^',6:'D'}; LS={3:'-',4:'--',5:'-.',6:':'}
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'mathtext.fontset':'stix',
 'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
 'axes.linewidth':0.6,'axes.edgecolor':'#555','xtick.color':'#555','ytick.color':'#555','axes.labelcolor':'#222',
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def sla(lam,c,mode='stable'):
    p=mmck_probs(lam,mu,c,K); acc=p[:K]/(1-p[K]); k=np.arange(K)
    n=np.maximum(k-c,0); ew=np.where(k>=c,(k-c+1)/(c*mu),0.0)
    md=LATENCY['delta_0']+LATENCY['eta']*(n/c)**LATENCY['beta']
    return float(np.sum(acc*delay_cdf(T_SLA-ew-1/mu,md,mode,dc))), p[K]
N=np.arange(60,371,2)
fig,ax=plt.subplots(2,1,figsize=(3.45,3.9),sharex=True,gridspec_kw={'hspace':0.12})
for c in (3,4,5,6):
    v=np.array([sla(n*rate,c) for n in N])
    for a,y in ((ax[0],v[:,1]),(ax[1],v[:,0])):
        a.plot(N,y,color=COL[c],ls=LS[c],lw=1.3,marker=MK[c],markevery=12,ms=3.5,mec='white',mew=0.5,label=f'$c={c}$')
ax[0].set_ylabel('Blocking probability $P_B$'); ax[1].set_ylabel('SLA compliance $\\Pr(T\\leq T_{\\mathrm{SLA}})$')
ax[1].set_xlabel('Number of machines $N$'); ax[1].set_ylim(0,1.02)
for a,lab in zip(ax,('(a)','(b)')):
    a.grid(axis='y',color='#e6e6e3',lw=0.5); a.text(0.01,0.95,lab,transform=a.transAxes,va='top',fontsize=8)
ax[0].legend(loc='upper left',bbox_to_anchor=(0.08,1.0),frameon=False,ncol=2,handlelength=2.4)
fig.savefig(OUT+'fig3_queue_metrics.pdf',bbox_inches='tight')

# Fig 4: expected reward per decision, stable network, with switching thresholds
fig,ax=plt.subplots(figsize=(3.45,2.5))
R=np.array([[expected_step_reward(n*rate,c,'stable',wc,dc) for c in (3,4,5,6)] for n in N])
for j,c in enumerate((3,4,5,6)):
    ax.plot(N,R[:,j],color=COL[c],ls=LS[c],lw=1.3,marker=MK[c],markevery=12,ms=3.5,mec='white',mew=0.5,label=f'$c={c}$')
ax.plot(N,R.max(1),color='#222',lw=0.8,alpha=0.55,zorder=0,label='Static optimum')
thr=[128.1,193.4,262.6]
for t in thr: ax.axvline(t,color='#999',lw=0.6,ls=(0,(2,2)),zorder=0)
lo,hi=-0.09,0.36; ax.set_ylim(lo,hi)
for x0,x1,c in ((60,thr[0],3),(thr[0],thr[1],4),(thr[1],thr[2],5),(thr[2],370,6)):
    ax.text((x0+x1)/2,hi-0.012,f'$c^*={c}$',ha='center',va='top',fontsize=7,color='#333')
ax.set_xlabel('Number of machines $N$'); ax.set_ylabel('Expected reward per decision $R(c,\\Lambda)$')
ax.grid(axis='y',color='#e6e6e3',lw=0.5); ax.set_xlim(60,370)
ax.legend(loc='lower center',bbox_to_anchor=(0.5,1.0),frameon=False,ncol=3,handlelength=2.4,columnspacing=1.0)
fig.savefig(OUT+'fig4_static_optimum.pdf',bbox_inches='tight')


# ---------- Fig: learning curves (aggregated over agents, blocks of 1e4 decisions)
rows=[]
for f in glob.glob(os.path.join(RES,'main','learning_curves','*_curve.csv')):
    net,_,seed,_=os.path.basename(f).split('_'); d=pd.read_csv(f); d['net']=net; d['seed']=int(seed); rows.append(d)
c=pd.concat(rows); c['bin']=(c.timestep//10000+1)*10000
b=c.groupby(['net','seed','bin']).episode_return.mean().reset_index()
a=b.groupby(['net','bin']).episode_return.agg(['mean','min','max']).reset_index(); a=a[a.bin<=200000]
NC={'stable':'#2a78d6','degraded':'#eb6834','noisy':'#1baf7a'}; NLS={'stable':'-','degraded':'--','noisy':'-.'}
GRID=dict(axis='y',color='#e6e6e3',lw=0.5)
s=pd.read_csv(os.path.join(RES,'main','summary_ci.csv')); w=s.pivot(index='scenario',columns='policy',values='mean')
ref={n:np.mean([w.loc[f'{x}_{n}',['fixed_c3','fixed_c4','fixed_c5','fixed_c6']].max() for x in ['low','medium','high']]) for n in NC}
fig,ax=plt.subplots(figsize=(3.45,2.3))
for n in ['stable','degraded','noisy']:
    x=a[a.net==n]; k=x.bin/1000
    ax.fill_between(k,x['min'],x['max'],color=NC[n],alpha=0.15,lw=0)
    ax.plot(k,x['mean'],color=NC[n],ls=NLS[n],lw=1.4,label=n.capitalize())
    ax.axhline(ref[n],color=NC[n],lw=0.7,ls=(0,(1,2)))
ax.set_xlabel('Training decisions ($\\times10^3$)'); ax.set_ylabel('Mean episode return'); ax.grid(**GRID); ax.set_xlim(10,200)
ax.legend(frameon=False,loc='lower right',handlelength=2.6)
fig.savefig(OUT+'fig_learning_curves.pdf',bbox_inches='tight')

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'mathtext.fontset':'stix',
 'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.linewidth':0.6,
 'axes.edgecolor':'#555','xtick.color':'#555','ytick.color':'#555','axes.labelcolor':'#222','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
GRID=dict(axis='y',color='#e6e6e3',lw=0.5)
PC={'ga':'#1baf7a','queueing_model':'#eb6834','adaptive_model':'#a3401a','ppo':'#8fb8ea','ppo_ensemble':'#2a78d6','hybrid_adaptive':'#222222'}
PL={'ga':'GA-tuned rule','queueing_model':'Queueing model','adaptive_model':'Self-calibrating model','ppo':'PPO (single)','ppo_ensemble':'PPO ensemble','hybrid_adaptive':'Proposed hybrid'}
# ---- Fig A: mean shortfall by group
sf=pd.read_csv(V+'shortfall.csv',index_col=0)
pols=list(PC); cats=['seen','unseen','parameter','structural']; labs=['Seen','Unseen','Parameter\nmismatch','Structural\nmismatch']
m=sf.groupby('group')[pols].mean().loc[cats]
fig,ax=plt.subplots(figsize=(3.45,2.35)); W=0.13
for j,p in enumerate(pols):
    xs=np.arange(len(cats))+(j-2.5)*(W+0.008)
    ax.bar(xs,m[p],width=W,color=PC[p],label=PL[p],zorder=2,edgecolor='white',linewidth=0.3)
ax.set_xticks(np.arange(len(cats))); ax.set_xticklabels(labs); ax.axhline(0,color='#555',lw=0.6)
ax.set_ylabel('Mean shortfall'); ax.grid(**GRID); ax.set_axisbelow(True)
ax.legend(frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(0.5,1.0),handlelength=1.1,columnspacing=0.8)
fig.savefig(OUT+'fig_shortfall_groups.pdf',bbox_inches='tight')
# ---- Fig B: bursty trace
d=pd.read_csv(os.path.join(RES,'controllers','trace_hybrid.csv')); x=d[d.scenario=='bursty280_stable']
h=x[x.policy=='hybrid_adaptive']; a=x[x.policy=='adaptive_model']
fig,axs=plt.subplots(3,1,figsize=(3.45,3.6),sharex=True,gridspec_kw={'hspace':0.18,'height_ratios':[1,1,1]})
ax=axs[0]; ax.plot(h.step,h.ratio,color='#222',lw=1.2,marker='o',ms=2)
ax.axhline(h.threshold.iloc[0],color='#d03b3b',lw=0.8,ls='--'); ax.text(30.4,h.threshold.iloc[0],'threshold',color='#d03b3b',fontsize=6.5,va='bottom',ha='right')
for s,md in zip(h.step,h['mode']):
    if md=='ppo': ax.axvspan(s-0.5,s+0.5,color='#2a78d6',alpha=0.12,lw=0)
ax.set_ylabel('Measured/\npredicted'); ax.grid(**GRID)
ax=axs[1]; ax.step(a.step,a.c,where='mid',color=PC['adaptive_model'],lw=1.2,label=PL['adaptive_model'])
ax.step(h.step,h.c+0.06,where='mid',color='#222',lw=1.2,label=PL['hybrid_adaptive'])
ax.set_ylabel('Servers $c$'); ax.set_yticks([3,4,5,6]); ax.set_ylim(2.7,6.4); ax.grid(**GRID)
ax.legend(frameon=False,loc='lower left',ncol=2,handlelength=1.4)
ax=axs[2]; ax.plot(a.step,a.sla,color=PC['adaptive_model'],lw=1.1,marker='o',ms=2); ax.plot(h.step,h.sla,color='#222',lw=1.1,marker='o',ms=2)
ax.set_ylabel('Compliance'); ax.set_xlabel('Decision $k$ (60 s windows)'); ax.grid(**GRID); ax.set_xlim(0.5,30.5)
for i,ax in enumerate(axs): ax.text(0.005,1.02,'(%s)'%'abc'[i],transform=ax.transAxes,fontsize=8,va='bottom')
fig.savefig(OUT+'fig_hybrid_trace.pdf',bbox_inches='tight')
# ---- Fig C: sensitivity of thresholds (stable network) to selected parameters
rows=json.load(open(V+'sensitivity.json'))
r=pd.DataFrame(rows); r=r[r['mode']=='stable']
nom=r[r.param=='nominal'].iloc[0].th
fig,ax=plt.subplots(figsize=(3.45,2.2))
order=[('mu',0.7),('mu',0.85),('mu',1.15),('mu',1.3),('fail',0.5),('fail',1.5),('fail',2.0),('eta',0.5),('eta',2.0),('beta',1.2/1.4),('beta',1.6/1.4),('d0',0.6),('d0',1.6),('T',2/3),('T',4/3),('K',0.6),('K',2.0),('wc',0.5),('wc',2.0)]
lab={'mu':'$\\mu$','fail':'$q_{01},q_{12}$','eta':'$\\kappa$','beta':'$\\gamma$','d0':'$\\delta_0$','T':'$T_{\\mathrm{SLA}}$','K':'$K$','wc':'$w_C$'}
ys=[];yl=[]
for i,(p,s) in enumerate(order):
    rr=r[(r.param==p)&(np.isclose(r.scale,s))].iloc[0]
    for j,key in enumerate(['3->4','4->5','5->6']):
        v=rr.th.get(key,np.nan); ax.plot([nom[key],v],[i,i],color='#bbb',lw=0.6,zorder=1)
        ax.scatter(v,i,s=9,color=['#2a78d6','#eb6834','#1baf7a'][j],zorder=2,label=['$3\\to4$','$4\\to5$','$5\\to6$'][j] if i==0 else None)
    sl={'beta':{1.2/1.4:'$\\gamma=1.2$',1.6/1.4:'$\\gamma=1.6$'}}.get(p,{}).get(s)
    ys.append(i); yl.append(sl if sl else f"{lab[p]} $\\times${s:g}".replace('0.666667','2/3').replace('1.33333','4/3'))
for key,c in zip(['3->4','4->5','5->6'],['#2a78d6','#eb6834','#1baf7a']): ax.axvline(nom[key],color=c,lw=0.6,ls=':')
ax.set_yticks(ys); ax.set_yticklabels(yl,fontsize=6); ax.invert_yaxis(); ax.set_xlabel('Population size $N$ at which the static optimum switches')
ax.legend(frameon=False,loc='lower right',fontsize=6.5,handletextpad=0.2); ax.grid(axis='x',color='#e6e6e3',lw=0.5)
fig.set_size_inches(3.45,3.3)
fig.savefig(OUT+'fig_sensitivity.pdf',bbox_inches='tight')
print('ok')
