"""Live demonstration of the CHAMPS study (runs in about a minute).

Step 1  Check the simulator against the exact M/M/c/K formulas
        (c = 4 servers, N = 200 and 350 machines, stable network, 10 simulated minutes each).
Step 2  Run one episode with bursty traffic (N = 280, stable network, 30 decisions)
        with the self-calibrating model alone and with CHAMPS, print what each
        does, and draw the comparison.

    python demo.py            # prints the results and saves demo_trace.png
    python demo.py --show     # also opens the figure in a window
"""
import argparse, json, time
import numpy as np
import matplotlib
from parameters import SERVER
from validate_fast_env import mmck, run_fast
from fast_env import make_fast_env, env_options, ACTION_TO_C
from fast_evaluate import PPOPolicy, env_kwargs_for
from variants import EnsemblePolicy, AdaptiveModelPolicy
from hybrid3 import Hybrid3Policy, ALL

NET, SCENARIO = "stable", "bursty280_stable"


def line(t=""):
    print("\n" + t + "\n" + "-" * len(t)) if t else print()


def step1():
    line("Step 1: simulator against the exact queueing formulas (c = 4 servers)")
    t0 = time.time()
    print("  load                    arrivals/s   blocking: simulated  exact    wait in queue: simulated  exact")
    for n, label in ((200, "normal, N=200"), (350, "overload, N=350")):
        r = run_fast(n, 4, NET, 600.0, seed=1)
        pb, wq = mmck(r["lam"], SERVER["mu"], 4, SERVER["K"])
        print(f"  {label:22s}  {r['lam']:9.1f}   {100 * r['P_B']:15.2f}%  {100 * pb:6.2f}%"
              f"   {r['W_q_ms']:17.2f} ms  {1000 * wq:6.2f} ms")
    print(f"  ({time.time() - t0:.0f} s, including compiling the simulator)")


class Champs(Hybrid3Policy):
    """CHAMPS, recording for every decision whether the learned mode was active."""
    def reset(self):
        super().reset(); self.modes = []

    def predict(self, obs, deterministic=True):
        a, s = super().predict(obs, deterministic)
        self.modes.append(self.use_ppo)
        return a, s


def episode(pol, kw, seed):
    env = make_fast_env(**kw); obs, _ = env.reset(seed=seed); pol.reset()
    c, sla, total, done = [], [], 0.0, False
    while not done:
        a, _ = pol.predict(obs)
        obs, r, done, _, info = env.step(a)
        c.append(info["c"]); sla.append(info["sla_compliance"]); total += r
    return np.array(c), np.array(sla), total


def step2(show):
    line("Step 2: one episode with bursty traffic (N=280, stable network)")
    cfg = json.load(open("experiment_config.json")); tr, ev = cfg["train"], cfg["eval"]
    opt = env_options(tr)
    kw = dict(w_c=opt["w_c"], w_switch=opt["w_switch"], deg_cong=opt["degraded_congestion"])
    thr = json.load(open("cusum_thresholds.json"))[NET]
    tgt = json.load(open("autoscaler_targets.json"))[NET]["target"]
    ens = EnsemblePolicy([PPOPolicy(f"models/{NET}/ppo_seed_{s}.zip", "ppo", s) for s in tr["seeds"]])
    champs = Champs(NET, ens, threshold=thr["tau"], h=thr["h"], hpa_target=tgt,
                    use_model_floor=True, name="champs", **kw)
    model = AdaptiveModelPolicy(NET, **kw)
    k = env_kwargs_for(ALL(NET)[SCENARIO], tr, ev["episode_duration"]); k["measured_util"] = True
    seed = ev["seed_base"]
    cm, sm, rm = episode(model, k, seed)
    cc, scc, rc = episode(champs, k, seed)
    learned = np.array(champs.modes)
    print("  window:              " + " ".join(f"{i:2d}" for i in range(1, 31)))
    print("  model alone, servers " + " ".join(f"{x:2d}" for x in cm))
    print("  CHAMPS, servers      " + " ".join(f"{x:2d}" for x in cc))
    print("  CHAMPS mode          " + " ".join(" L" if m else " M" for m in learned)
          + "   (M = model decides, L = learned mode after a CUSUM alarm)")
    line()
    print(f"                        mean servers  mean compliance  lowest compliance  episode reward")
    print(f"  model alone           {cm.mean():12.2f}  {sm.mean():15.2f}  {sm.min():17.2f}  {rm:14.2f}")
    print(f"  CHAMPS                {cc.mean():12.2f}  {scc.mean():15.2f}  {scc.min():17.2f}  {rc:14.2f}")

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = np.arange(1, 31)
    fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    for i in np.where(learned)[0]:
        for a in ax:
            a.axvspan(w[i] - 0.5, w[i] + 0.5, color="#1f3a5f", alpha=0.08, lw=0)
    ax[0].step(w, cm, where="mid", color="#a3401a", label="Self-calibrating model alone")
    ax[0].step(w, cc + 0.05, where="mid", color="#222222", label="CHAMPS")
    ax[0].set_ylabel("Servers"); ax[0].set_yticks([3, 4, 5, 6]); ax[0].legend(frameon=False, loc="lower left")
    ax[1].plot(w, sm, "o-", color="#a3401a", ms=3); ax[1].plot(w, scc, "o-", color="#222222", ms=3)
    ax[1].set_ylabel("Compliance"); ax[1].set_xlabel("Decision window (60 s each); shading = learned mode")
    fig.suptitle("Bursty traffic, N = 280: the model alone cuts servers, CHAMPS holds them")
    fig.tight_layout(); fig.savefig("demo_trace.png", dpi=150)
    print("\n  Figure saved: demo_trace.png")
    if show:
        plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    t0 = time.time(); step1(); step2(a.show)
    print(f"\nDone in {time.time() - t0:.0f} s.")
