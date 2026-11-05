"""
One command for the whole experiment plan, in time-limited sessions.

    python run_experiments.py --hours 4

Each time you run it, it continues exactly where the previous session stopped:
  1. trains PPO for every (network, seed) in experiment_config.json
  2. evaluates all policies on the 12 scenarios (per network)
  3. writes the merged results to data/

It stops cleanly before the time limit, saving the model and progress.
A checkpoint is also saved every few minutes, so a power cut or PC shutdown
loses at most those few minutes. Just run the same command again next time.

Other commands:
    python run_experiments.py --status         # show what is done / remaining
    python run_experiments.py --hours 4 --only-train
    python run_experiments.py --hours 4 --only-eval

Outputs:
    models/<network>/ppo_seed_<seed>.zip      final trained models
    logs/<network>_seed_<seed>_curve.csv      learning curve (one row per episode)
    data/eval_<network>.csv, data/traces_<network>.csv
    data/evaluation_results.csv               all evaluation episodes
    data/summary_by_scenario_policy.csv       means and standard deviations
    progress/status.json                      what has been done
"""
import argparse
import csv
import functools
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

STATUS = ROOT / "progress" / "status.json"
CKPT = ROOT / "progress" / "checkpoints"


# ------------------------------------------------------------------ helpers
def load_json(p, default):
    try:
        with open(p) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, p)  # atomic: never leaves a half-written status file


def fmt(sec):
    sec = int(sec)
    return f"{sec // 3600}h {sec % 3600 // 60:02d}m"


def build_jobs(cfg):
    tr = cfg["train"]
    jobs = []
    for net in tr["networks"]:
        for s in tr["seeds"]:
            jobs.append(("train", net, s))
    for net in tr["networks"]:
        jobs.append(("eval", net, None))
    return jobs


def job_id(kind, net, seed):
    return f"{kind}_{net}" + (f"_seed_{seed}" if seed is not None else "")


def n_envs_from(cfg):
    n = cfg.get("n_envs", "auto")
    if n == "auto":
        n = max(1, min(4, os.cpu_count() or 1))
    return int(n)


def _make_env(net, tr, seed):
    from fast_env import make_fast_env, env_options
    return make_fast_env(n_machines=tr["n_machines"], network_mode=net,
                         episode_duration=tr["episode_duration"],
                         step_duration=tr["step_duration"], seed=seed,
                         **env_options(tr))


# ------------------------------------------------------------------ training
def train_job(net, seed, cfg, deadline, status):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    torch.set_num_threads(1)  # small network: one thread is fastest on CPU

    tr, pp = cfg["train"], cfg["ppo"]
    jid = job_id("train", net, seed)
    st = status.setdefault(jid, {"timesteps": 0, "done": False, "wall_seconds": 0.0,
                                 "sessions": 0})
    target = int(tr["timesteps"])
    n_envs = n_envs_from(cfg)
    st["sessions"] += 1
    # different env seed per session so a resumed run does not replay episodes
    env_seed = seed * 1000 + st["sessions"] * 17
    fns = [functools.partial(_make_env, net, tr, env_seed + i) for i in range(n_envs)]
    venv = (SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns))
    venv = VecMonitor(venv)
    venv.seed(env_seed)

    CKPT.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT / f"{jid}.zip"
    if ckpt.exists():
        model = PPO.load(ckpt, env=venv, device="cpu")
        print(f"  resuming from {model.num_timesteps:,} / {target:,} timesteps", flush=True)
    else:
        n_steps = max(1, pp["n_steps_total"] // n_envs)
        model = PPO("MlpPolicy", venv, learning_rate=pp["learning_rate"],
                    n_steps=n_steps, batch_size=pp["batch_size"],
                    n_epochs=pp["n_epochs"], gamma=pp["gamma"],
                    gae_lambda=pp["gae_lambda"], clip_range=pp["clip_range"],
                    ent_coef=pp["ent_coef"],
                    policy_kwargs=dict(net_arch=list(pp["net_arch"])),
                    seed=seed, device="cpu", verbose=0)
        print(f"  new run, {n_envs} parallel env(s), target {target:,} timesteps", flush=True)

    curve_path = ROOT / "logs" / f"{net}_seed_{seed}_curve.csv"
    curve_path.parent.mkdir(exist_ok=True)
    new_file = not curve_path.exists()
    curve_f = open(curve_path, "a", newline="")
    writer = csv.writer(curve_f)
    if new_file:
        writer.writerow(["timestep", "episode_return", "episode_length",
                         "mean_c", "wall_seconds_total"])

    ckpt_every = 60 * float(cfg.get("checkpoint_minutes", 10))
    t_start = time.time()
    wall_before = st["wall_seconds"]

    class Cb(BaseCallback):
        def __init__(self):
            super().__init__()
            self.last_ckpt = time.time()
            self.last_print = time.time()
            self.cs = [[] for _ in range(n_envs)]
            self.out_of_time = False
            self.recent = []

        def _on_step(self):
            infos = self.locals.get("infos", [])
            for i, info in enumerate(infos):
                if "c" in info:
                    self.cs[i].append(info["c"])
                if "episode" in info:
                    ep = info["episode"]
                    mc = float(np.mean(self.cs[i])) if self.cs[i] else float("nan")
                    self.cs[i] = []
                    self.recent.append(ep["r"])
                    writer.writerow([self.num_timesteps, round(float(ep["r"]), 6),
                                     int(ep["l"]), round(mc, 3),
                                     round(wall_before + time.time() - t_start, 1)])
            now = time.time()
            if now - self.last_ckpt > ckpt_every:
                self.model.save(ckpt)
                st["timesteps"] = int(self.num_timesteps)
                st["wall_seconds"] = wall_before + now - t_start
                save_json(STATUS, status)
                curve_f.flush()
                self.last_ckpt = now
            if now - self.last_print > 60:
                r = np.mean(self.recent[-200:]) if self.recent else float("nan")
                print(f"    {self.num_timesteps:>9,} / {target:,} steps   "
                      f"mean return (last 200 ep) {r:.4f}   "
                      f"time left {fmt(deadline - now)}", flush=True)
                self.last_print = now
            if now >= deadline:
                self.out_of_time = True
                return False
            return True

    cb = Cb()
    remaining = target - model.num_timesteps
    if remaining > 0:
        model.learn(total_timesteps=remaining, callback=cb,
                    reset_num_timesteps=False, progress_bar=False)
    curve_f.close()
    venv.close()

    st["timesteps"] = int(model.num_timesteps)
    st["wall_seconds"] = wall_before + time.time() - t_start
    if model.num_timesteps >= target and not cb.out_of_time:
        out = ROOT / "models" / net
        out.mkdir(parents=True, exist_ok=True)
        model.save(out / f"ppo_seed_{seed}.zip")
        st["done"] = True
        if ckpt.exists():
            ckpt.unlink()
        print(f"  finished: models/{net}/ppo_seed_{seed}.zip "
              f"(total training time {fmt(st['wall_seconds'])})", flush=True)
    else:
        model.save(ckpt)
        print(f"  paused at {model.num_timesteps:,} / {target:,} steps "
              f"(checkpoint saved)", flush=True)
    save_json(STATUS, status)


# ------------------------------------------------------------------ main
def print_status(cfg, status):
    jobs = build_jobs(cfg)
    print("\nExperiment progress")
    print("-" * 60)
    for kind, net, seed in jobs:
        jid = job_id(kind, net, seed)
        st = status.get(jid, {})
        if st.get("done"):
            mark = "done"
        elif kind == "train" and st.get("timesteps"):
            mark = f"{100 * st['timesteps'] / cfg['train']['timesteps']:.0f}%"
        else:
            mark = "-"
        extra = f"  ({fmt(st['wall_seconds'])})" if st.get("wall_seconds") else ""
        print(f"  {jid:32s} {mark}{extra}")
    left = [j for j in jobs if not status.get(job_id(*j), {}).get("done")]
    print("-" * 60)
    print(f"  {len(jobs) - len(left)} of {len(jobs)} jobs done")
    return left


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=4.0,
                    help="stop cleanly after this many hours (default 4)")
    ap.add_argument("--config", default="experiment_config.json")
    ap.add_argument("--status", action="store_true", help="show progress and exit")
    ap.add_argument("--only-train", action="store_true")
    ap.add_argument("--only-eval", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    status = load_json(STATUS, {})
    # protect against silently mixing two different configurations
    saved_cfg = status.get("_config")
    if saved_cfg and saved_cfg != cfg:
        print("ERROR: experiment_config.json changed since these runs started.\n"
              "Either restore the old config, or start a fresh experiment by\n"
              "renaming the folders progress/, models/, logs/ and data/.")
        sys.exit(1)
    status["_config"] = cfg

    left = print_status(cfg, status)
    if args.status or not left:
        if not left:
            print("\nAll jobs are complete. Results are in data/.")
        return

    t0 = time.time()
    margin = 5 * 60  # keep 5 minutes for saving and evaluation wrap-up
    deadline = t0 + args.hours * 3600 - margin
    print(f"\nSession limit {args.hours:g} h -> will stop by "
          f"{time.strftime('%H:%M', time.localtime(deadline + margin))}\n")

    from fast_evaluate import evaluate_network, summarize
    for kind, net, seed in left:
        if time.time() >= deadline:
            break
        jid = job_id(kind, net, seed)
        if kind == "train" and args.only_eval:
            continue
        if kind == "eval":
            if args.only_train:
                continue
            pending = [s for s in cfg["train"]["seeds"]
                       if not status.get(job_id("train", net, s), {}).get("done")]
            if net != "stable":
                pending += [s for s in cfg["train"]["seeds"]
                            if not status.get(job_id("train", "stable", s), {}).get("done")]
            if pending:
                print(f"[{jid}] waiting for training to finish, skipped for now")
                continue
            if deadline - time.time() < 20 * 60:
                print(f"[{jid}] less than 20 min left, evaluation postponed")
                break
            print(f"[{jid}] evaluating all policies on the {net} scenarios", flush=True)
            t = time.time()
            evaluate_network(net, cfg, ROOT)
            summarize(ROOT)
            status[jid] = {"done": True, "wall_seconds": time.time() - t}
            save_json(STATUS, status)
            print(f"  done in {fmt(time.time() - t)}; data/eval_{net}.csv written", flush=True)
            continue
        print(f"[{jid}]", flush=True)
        train_job(net, seed, cfg, deadline, status)

    print(f"\nSession finished after {fmt(time.time() - t0)}.")
    left = print_status(cfg, status)
    if left:
        print("\nRun the same command again next time to continue.")
    else:
        print("\nAll jobs are complete. Results are in data/.")


if __name__ == "__main__":
    main()
