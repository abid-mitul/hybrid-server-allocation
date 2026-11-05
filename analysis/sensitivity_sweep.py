"""Sensitivity of the static optimal allocation to the model parameters.

Recomputes the switching thresholds of the static optimum for 63 parameter
variations and the loss of the nominal model policy on the perturbed system.
Output: results/analysis/sensitivity.json
"""
import math, json
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from model_policy import mmck_probs
BASE = dict(mu=333.0, K=50, d0=0.050, eta=0.807, beta=1.4, xi=2.5, pmin=0.050,
            T=0.300, wX=0.5, wD=0.3, wB=0.2, wc=0.05, lam0=5.0, lam1=2.0,
            a01=0.1, a12=0.2, b10=0.5, b21=0.333)
CS = [3, 4, 5, 6]; LMAX = 350 * 5.0

def pi_star(p):
    Q = np.array([[-p["a01"], p["a01"], 0], [p["b10"], -(p["b10"] + p["a12"]), p["a12"]],
                  [0, p["b21"], -p["b21"]]])
    A = np.vstack([Q.T, np.ones(3)]); b = np.r_[0, 0, 0, 1]
    return np.linalg.lstsq(A, b, rcond=None)[0]

def lam_bar(p):
    pi = pi_star(p); return p["lam0"] * pi[0] + p["lam1"] * pi[1]

def cdf(t, md, mode, p):
    pos = np.maximum(t, 0.0)
    fe = np.where(t > 0, 1 - np.exp(-pos / md), 0.0)
    xm = p["pmin"] * md / p["d0"]
    fp = np.where(t > xm, 1 - (xm / np.maximum(pos, 1e-12)) ** p["xi"], 0.0)
    return fe if mode == "stable" else fp if mode == "degraded" else 0.5 * fe + 0.5 * fp

def R(lam, c, mode, p):
    K, mu = p["K"], p["mu"]
    q = mmck_probs(lam, mu, c, K); pb = q[K]
    acc = q[:K] / max(1 - pb, 1e-12); k = np.arange(K)
    nw = np.maximum(k - c, 0); ew = np.where(k >= c, (k - c + 1) / (c * mu), 0.0)
    md = p["d0"] + p["eta"] * (nw / c) ** p["beta"]
    ed = float(np.sum(acc * cdf(p["T"] - ew - 1 / mu, md, mode, p)))
    return (p["wX"] * lam * (1 - pb) / LMAX - p["wD"] * (1 - ed) - p["wB"] * pb
            - p["wc"] * (c - 3) / 3)

def thresholds(mode, p, Ngrid=np.arange(20, 501, 1)):
    lb = lam_bar(p)
    best = [CS[int(np.argmax([R(N * lb, c, mode, p) for c in CS]))] for N in Ngrid]
    th = {}
    for i in range(1, len(Ngrid)):
        if best[i] != best[i - 1]:
            th[f"{best[i-1]}->{best[i]}"] = int(Ngrid[i])
    return th, best

def regret(mode, p_true, Ns=(80, 120, 200, 280, 350)):
    """Loss per decision from using the nominal-model optimum on the true system."""
    lbt, lbn = lam_bar(p_true), lam_bar(BASE)
    out = []
    for N in Ns:
        lam_t = N * lbt
        c_nom = CS[int(np.argmax([R(lam_t, c, mode, BASE) for c in CS]))]  # nominal model, observed load
        r = [R(lam_t, c, mode, p_true) for c in CS]
        out.append(max(r) - r[CS.index(c_nom)])
    return out

VARS = [("mu", [0.7, 0.85, 1.15, 1.3]), ("eta", [0.5, 0.7, 1.3, 2.0]),
        ("beta", [1.2 / 1.4, 1.6 / 1.4]), ("d0", [0.6, 1.6]), ("T", [2 / 3, 4 / 3]),
        ("K", [0.6, 2.0]), ("fail", [0.5, 1.5, 2.0]), ("wc", [0.5, 2.0])]
if __name__ == "__main__":
    OUT = os.path.join(ROOT, "results", "analysis"); os.makedirs(OUT, exist_ok=True)
    rows = []
    for mode in ["stable", "degraded", "noisy"]:
        th, _ = thresholds(mode, BASE)
        rows.append(dict(mode=mode, param="nominal", scale=1.0, lam_bar=lam_bar(BASE), th=th,
                         regret=[0] * 5))
        for name, scales in VARS:
            for s in scales:
                p = dict(BASE)
                if name == "fail":
                    p["a01"] *= s; p["a12"] *= s
                elif name == "K":
                    p["K"] = int(round(BASE["K"] * s))
                else:
                    p[name] = BASE[name] * s
                th, _ = thresholds(mode, p)
                rg = regret(mode, p) if name not in ("wc",) else [None] * 5
                rows.append(dict(mode=mode, param=name, scale=s, lam_bar=lam_bar(p), th=th,
                                 regret=rg))
                print(mode, name, round(s, 3), th, [None if x is None else round(x, 4) for x in rg], flush=True)
    json.dump(rows, open(os.path.join(OUT, "sensitivity.json"), "w"), indent=1, default=float)
