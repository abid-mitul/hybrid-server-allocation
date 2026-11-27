"""v1.3 task C3: evaluation on the held-out structural departures.

Scenarios: heldout_env.py (frozen in commit 4474e5a, 2026-11-25, before any
controller was run on them): H1 correlated service, H2 server failures,
H3 periodic polling, at N = 200 and 280, in each network regime.
Evaluation seeds 30000-30019 (new).  Every controller is used exactly as in
the main study, with no change after this run.

Controllers: fixed c = 3..6 (reference), threshold rule, GA-tuned rule,
autoscaler, queueing model, self-calibrating model, single PPO agents (5),
PPO ensemble, CUSUM + c_max, CUSUM + floors without agents, CHAMPS (with
binding log) and the oracle.  The oracle reads the true arrival rate and the
nominal parameters; it is not told about server failures (H2).

    python c3_heldout.py --network stable        # repeat for degraded, noisy
Writes data/v13/c3_<network>.csv
"""
import argparse, json, time
from pathlib import Path
import pandas as pd
from fast_env import env_options, C_MAX, ACTION_TO_C
from fast_evaluate import run_episode, FixedCPolicy, PPOPolicy, fit_ga
from baselines import FixedPriorityPolicy
from model_policy import QueueingModelPolicy
from variants import EnsemblePolicy, AdaptiveModelPolicy
from autoscaler import HPAPolicy
from cusum_test import CusumHybridPolicy
from evaluate_oracle import OracleModelPolicy
from heldout_env import heldout_scenarios, make_heldout_env, HELDOUT_SEED_BASE
from v13_ppo_role import LoggedHybrid3

EPISODES = 20


def main(net):
    cfg = json.load(open("experiment_config.json")); tr = cfg["train"]; opt = env_options(tr)
    kw = dict(w_c=opt["w_c"], w_switch=opt["w_switch"], deg_cong=opt["degraded_congestion"])
    t = json.load(open("cusum_thresholds.json"))[net]
    tgt = json.load(open("autoscaler_targets.json"))[net]["target"]
    ppos = [PPOPolicy(Path("models") / net / f"ppo_seed_{s}.zip", "ppo", s) for s in tr["seeds"]]
    ens = EnsemblePolicy(ppos)
    t0 = time.time(); ga = fit_ga(net, cfg)
    print(f"  GA fitted in {time.time() - t0:.0f}s, thresholds {[round(x, 3) for x in ga.thresholds]}",
          flush=True)
    hkw = dict(threshold=t["tau"], h=t["h"], hpa_target=tgt, **kw)
    pols = [FixedCPolicy(c) for c in ACTION_TO_C] + [
        FixedPriorityPolicy(), ga, HPAPolicy(tgt),
        QueueingModelPolicy(net, **kw), AdaptiveModelPolicy(net, **kw)] + ppos + [
        ens,
        CusumHybridPolicy(net, FixedCPolicy(C_MAX), threshold=t["tau"], h=t["h"],
                          name="cusum_fallback_c6", **kw),
        LoggedHybrid3(net, ens, name="no_agents", use_agents=False, **hkw),
        LoggedHybrid3(net, ens, name="champs", use_agents=True, **hkw)]
    rows = []
    for sname, sc in heldout_scenarios(net).items():
        env = make_heldout_env(sc["heldout"], sc["n_machines"], net, opt,
                               step_duration=tr["step_duration"])
        oracle = OracleModelPolicy(env, net, **kw)
        t0 = time.time()
        for pol in pols + [oracle]:
            for ep in range(EPISODES):
                seed = HELDOUT_SEED_BASE + ep
                res, _ = run_episode(env, pol, seed)
                extra = {}
                if hasattr(pol, "ppo_steps"):
                    extra = dict(ppo_fraction=pol.ppo_steps / env.n_steps_per_episode,
                                 handovers=pol.n_handover)
                if hasattr(pol, "log"):
                    extra.update(pol.log)
                rows.append(dict(scenario=sname, group="heldout", heldout=sc["heldout"],
                                 n_machines=sc["n_machines"], network=net, policy=pol.name,
                                 train_seed=getattr(pol, "train_seed", ""), episode=ep,
                                 eval_seed=seed, n_failures=env.n_failures, **extra, **res))
        print(f"    {sname}: {time.time() - t0:.0f}s", flush=True)
    out = Path("data") / "v13"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"c3_{net}.csv", index=False)
    print("done", net, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--network", required=True)
    main(ap.parse_args().network)
