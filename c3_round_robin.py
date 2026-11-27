"""v1.3 task C3, addendum: Round-Robin on the held-out departures.

Round-Robin was evaluated in the main study but not in the first held-out
run.  It is added here on the same frozen scenarios (heldout_env.py, commit
4474e5a) and the same seeds 30000-30019, so that the held-out table covers
every baseline named in the thesis proposal.  No other controller is changed.

    python c3_round_robin.py --network stable     # repeat for degraded, noisy
Writes data/v13/c3_rr_<network>.csv (same columns as data/v13/c3_<network>.csv)
"""
import argparse, json
from pathlib import Path
import pandas as pd
from fast_env import env_options
from fast_evaluate import run_episode
from baselines import RoundRobinPolicy
from heldout_env import heldout_scenarios, make_heldout_env, HELDOUT_SEED_BASE

EPISODES = 20


def main(net):
    cfg = json.load(open("experiment_config.json")); tr = cfg["train"]; opt = env_options(tr)
    pol, rows = RoundRobinPolicy(), []
    for sname, sc in heldout_scenarios(net).items():
        env = make_heldout_env(sc["heldout"], sc["n_machines"], net, opt, step_duration=tr["step_duration"])
        for ep in range(EPISODES):
            seed = HELDOUT_SEED_BASE + ep
            res, _ = run_episode(env, pol, seed)
            rows.append(dict(scenario=sname, group="heldout", heldout=sc["heldout"],
                             n_machines=sc["n_machines"], network=net, policy=pol.name, train_seed="",
                             episode=ep, eval_seed=seed, n_failures=env.n_failures, **res))
        print(f"    {sname} done", flush=True)
    out = Path("data") / "v13"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"c3_rr_{net}.csv", index=False)
    print("done", net, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--network", required=True)
    main(ap.parse_args().network)
