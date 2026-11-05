"""LaTeX table rows of the paper from results/analysis. Output: results/analysis/tables/"""
import pandas as pd, numpy as np
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
V = os.path.join(ROOT, 'results', 'analysis') + '/'
T_OUT = os.path.join(V, 'tables'); os.makedirs(T_OUT, exist_ok=True)
sf=pd.read_csv(V+'shortfall.csv',index_col=0); T=pd.read_csv(V+'tests.csv')
P=['fixed_c6','fixed_priority','ga','queueing_model','adaptive_model','ppo','ppo_ensemble','hybrid_adaptive']
f=lambda x: ('$-$%.3f'%abs(x)) if x<=-0.0005 else ('0.000' if abs(x)<0.0005 else '%.3f'%x)
types=[('seen',[('low','Low ($N=80$)'),('medium','Medium ($N=200$)'),('high','High ($N=350$)'),('burst','Burst ($200\\to350$)')]),
 ('unseen',[('unseen120','Unseen ($N=120$)'),('unseen280','Unseen ($N=280$)'),('drop','Drop ($350\\to200$)')]),
 ('parameter',[('slowservers','Slow servers ($0.7\\mu$)'),('slowservers85','Slow servers ($0.85\\mu$)'),('fastservers','Fast servers ($1.3\\mu$)'),('congestion','Congestion ($2\\kappa$)'),('failures','Failures ($2q_{01},2q_{12}$)'),('combined','Combined ($0.7\\mu$, $2\\kappa$)')]),
 ('structural',[('highvar200','High variance, $N=200$'),('highvar280','High variance, $N=280$'),('bursty200','Bursty traffic, $N=200$'),('bursty280','Bursty traffic, $N=280$'),('mixedspeed200','Mixed speeds, $N=200$'),('mixedspeed280','Mixed speeds, $N=280$'),('batch200','Batched traffic, $N=200$'),('batch280','Batched traffic, $N=280$'),('svcburst200','Bursty service, $N=200$'),('svcburst280','Bursty service, $N=280$')])]
gname={'seen':'Seen','unseen':'Unseen','parameter':'Parameter mismatch','structural':'Structural mismatch'}
L=[]
for g,rows in types:
    L.append('\\multicolumn{9}{@{}l}{\\textit{%s}}\\\\'%gname[g])
    for t,lab in rows:
        x=sf[sf.type==t][P].mean()
        best=x.min()
        cells=[('\\textbf{%s}'%f(v)) if abs(v-best)<0.0005 else f(v) for v in x]
        L.append(lab+' & '+' & '.join(cells)+'\\\\')
    m=sf[sf.group==g][P].mean(); L.append('\\quad Group mean & '+' & '.join(f(v) for v in m)+'\\\\[2pt]')
L.append('\\hline')
L.append('All 69 scenarios, mean & '+' & '.join(f(v) for v in sf[P].mean())+'\\\\')
L.append('All 69 scenarios, worst & '+' & '.join(f(v) for v in sf[P].max())+'\\\\')
open(os.path.join(T_OUT,'table_types.tex'),'w').write('\n'.join(L)+'\n')
# significance
C=[('fixed_c6','Fixed $c=6$'),('fixed_priority','Threshold rule'),('ga','GA-tuned rule'),('queueing_model','Queueing model'),('adaptive_model','Self-calibrating model'),('ppo','PPO (single agent)'),('ppo_ensemble','PPO ensemble')]
R=[]
for c,lab in C:
    x=T[T.comp==c]; cells=[]
    for g in ['seen','unseen','parameter','structural',None]:
        xx=x if g is None else x[x.group==g]
        cells.append('%d/%d/%d'%((xx.res=='+').sum(),(xx.res=='=').sum(),(xx.res=='-').sum()))
    R.append(lab+' & '+' & '.join(cells)+'\\\\')
open(os.path.join(T_OUT,'table_sig2.tex'),'w').write('\n'.join(R)+'\n')
# ablation
A=[('hybrid','Fixed-parameter model + single agent'),('hybrid_ensemble','Fixed-parameter model + ensemble'),('hybrid_adaptive','Self-calibrating model + ensemble (proposed)')]
B=[]
for p,lab in A:
    m=sf.groupby('group')[p].mean()
    B.append(lab+' & '+' & '.join(f(m[g]) for g in ['seen','unseen','parameter','structural'])+' & '+f(sf[p].mean())+' & '+f(sf[p].max())+'\\\\')
open(os.path.join(T_OUT,'table_ablation.tex'),'w').write('\n'.join(B)+'\n')
# reward weights
w1=pd.read_csv(V+'shortfall_wc010.csv',index_col=0); w0=sf[(sf.net=='stable')&(sf.index.isin(w1.index))]
Q=[('ga','GA-tuned rule'),('queueing_model','Queueing model'),('adaptive_model','Self-calibrating model'),('ppo','PPO (single agent)'),('ppo_ensemble','PPO ensemble'),('hybrid_adaptive','Proposed hybrid')]
W=[]
for p,lab in Q:
    W.append(lab+' & '+f(w0[p].mean())+' & '+f(w0[p].max())+' & '+f(w1[p].mean())+' & '+f(w1[p].max())+'\\\\')
open(os.path.join(T_OUT,'table_wc.tex'),'w').write('\n'.join(W)+'\n')
print(open(os.path.join(T_OUT,'table_types.tex')).read()); print(open(os.path.join(T_OUT,'table_sig2.tex')).read()); print(open(os.path.join(T_OUT,'table_ablation.tex')).read()); print(open(os.path.join(T_OUT,'table_wc.tex')).read())
