"""Records one episode of the hybrid controller and the self-calibrating model
(trace figure of the paper). Output: data/trace_hybrid.csv"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import json, pandas as pd, numpy as np
from pathlib import Path
from fast_env import make_fast_env, env_options, T_SLA, C_MIN
from fast_evaluate import scenarios_for, extra_scenarios_for, structural_scenarios_for, env_kwargs_for, PPOPolicy
from hybrid_policy import load_thresholds
from variants import EnsemblePolicy, AdaptiveModelPolicy, HybridAdaptivePolicy
cfg=json.load(open('experiment_config.json')); ev,tr=cfg['eval'],cfg['train']; opt=env_options(tr); thr=load_thresholds()
kw=dict(w_c=opt['w_c'],w_switch=opt['w_switch'],deg_cong=opt['degraded_congestion'])
rows=[]
for net,sc in [('degraded','slowservers_degraded'),('stable','bursty280_stable')]:
    allsc={**scenarios_for(net),**extra_scenarios_for(net),**structural_scenarios_for(net)}
    ppos=[PPOPolicy(Path('models')/net/f'ppo_seed_{s}.zip','ppo',s) for s in tr['seeds']]
    pols={'hybrid_adaptive':HybridAdaptivePolicy(net,EnsemblePolicy(ppos),threshold=thr['adaptive_'+net],**kw),
          'adaptive_model':AdaptiveModelPolicy(net,**kw)}
    env=make_fast_env(**env_kwargs_for(allsc[sc],tr,ev['episode_duration']))
    for name,p in pols.items():
        obs,_=env.reset(seed=1000); p.reset(); done=False; k=0
        while not done:
            ratio=p.ratio(obs) if name=='hybrid_adaptive' else np.nan
            a,_=p.predict(obs)
            mode=('ppo' if p.use_ppo else 'model') if name=='hybrid_adaptive' else 'model'
            obs,r,done,_,info=env.step(a); k+=1
            rows.append(dict(scenario=sc,policy=name,step=k,c=info['c'],ratio=ratio,mode=mode,sla=info['sla_compliance'],reward=r,threshold=thr['adaptive_'+net]))
pd.DataFrame(rows).to_csv('data/trace_hybrid.csv',index=False); print('ok')
