"""
Robustness experiments.

1) Further parameter-mismatch tests (no training; uses the PPO models in models/):

       python run_extra.py --mismatch

   Evaluates every policy on 5 parameter-mismatch scenarios per network
   (extra_scenarios_for in fast_evaluate.py) and writes
   data/eval_extra_<network>.csv. A network that is already done is skipped.

2) Reward-weight sensitivity (training, stable network only):

       python run_extra.py --setup-weights

   Creates a sibling folder ../DRL_wc010 with the same code and a configuration
   in which the server cost w_C is doubled (0.05 -> 0.10). Then run there:

       python run_experiments.py --hours 4
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

CODE_FILES = ["fast_env.py", "fast_evaluate.py", "model_policy.py", "baselines.py",
              "parameters.py", "stats.py", "run_experiments.py", "requirements.txt"]


def run_mismatch(cfg):
    from fast_evaluate import evaluate_network
    missing = [n for n in cfg["train"]["networks"] for s in cfg["train"]["seeds"]
               if not (ROOT / "models" / n / f"ppo_seed_{s}.zip").exists()]
    if missing:
        print("ERROR: trained models are missing in models/. Run this command in the "
              "folder where run_experiments.py trained the agents.")
        sys.exit(1)
    t_all = time.time()
    for net in cfg["train"]["networks"]:
        out = ROOT / "data" / f"eval_extra_{net}.csv"
        if out.exists():
            print(f"[{net}] already done ({out.name}), skipped")
            continue
        print(f"[{net}] extra mismatch scenarios", flush=True)
        t = time.time()
        evaluate_network(net, cfg, ROOT, extra=True)
        print(f"  done in {(time.time() - t) / 60:.0f} min", flush=True)
    print(f"\nAll extra mismatch tests finished ({(time.time() - t_all) / 60:.0f} min).")
    print("Results: data/eval_extra_stable.csv, data/eval_extra_degraded.csv, "
          "data/eval_extra_noisy.csv")


def setup_weights(cfg):
    dest = ROOT.parent / "DRL_wc010"
    if dest.exists():
        print(f"{dest} already exists; nothing changed. To continue the run:\n"
              f"    cd {dest}\n    python run_experiments.py --hours 4")
        return
    dest.mkdir()
    for f in CODE_FILES:
        shutil.copy2(ROOT / f, dest / f)
    new = json.loads(json.dumps(cfg))
    new["name"] = "journal_wc010"
    new["train"]["networks"] = ["stable"]
    new["train"]["w_c"] = 0.10
    with open(dest / "experiment_config.json", "w") as f:
        json.dump(new, f, indent=2)
    print(f"Created {dest}\n(server cost w_C = 0.10, stable network, 5 seeds)\n\n"
          f"Now run:\n    cd {dest}\n    python run_experiments.py --hours 4")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mismatch", action="store_true")
    ap.add_argument("--setup-weights", action="store_true")
    ap.add_argument("--config", default="experiment_config.json")
    a = ap.parse_args()
    with open(ROOT / a.config) as f:
        cfg = json.load(f)
    if a.mismatch:
        run_mismatch(cfg)
    if a.setup_weights:
        setup_weights(cfg)
    if not (a.mismatch or a.setup_weights):
        ap.print_help()


if __name__ == "__main__":
    main()
