"""Alarm statistics of the hybrid controller under nominal conditions
(fresh seeds 7000-7019), used for Proposition 3 of the paper."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import json, numpy as np
from fast_env import make_fast_env, env_options
from hybrid_policy import load_thresholds
from variants import HybridAdaptivePolicy
cfg=json.load(open('experiment_config.json')); tr=cfg['train']; opt=env_options(tr); thr=load_thresholds()
for net in tr['networks']:
    h=HybridAdaptivePolicy(net,None,opt['w_c'],opt['w_switch'],opt['degraded_congestion'],threshold=1e9)
    A=[]
    for N in (80,200,350):
        env=make_fast_env(n_machines=N,network_mode=net,episode_duration=1800,step_duration=60,measured_util=True,**opt)
        for s in range(7000,7020):   # fresh seeds, neither calibration nor evaluation
            obs,_=env.reset(seed=s); h.reset(); done=False; row=[]
            while not done:
                row.append(h.ratio(obs)>thr['adaptive_'+net]); a,_=h.predict(obs); obs,_,done,_,_=env.step(a)
            A.append(row)
    A=np.array(A); a1=A.mean(); a1_late=A[:,3:].mean(); first=A[:,:3].mean()
    pair=(A[:,1:]&A[:,:-1]).mean(); pair_late=(A[:,4:]&A[:,3:-1]).mean()
    cond=(A[:,1:]&A[:,:-1]).sum()/max(A[:,:-1].sum(),1)
    print(net,'alarm rate',round(a1,4),'first3',round(first,4),'after3',round(a1_late,4),'pair',round(pair,5),'pair after3',round(pair_late,5),'P(alarm|prev alarm)',round(cond,3))
