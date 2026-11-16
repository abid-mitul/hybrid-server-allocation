"""Extended sensitivity analysis (revision): adds the parameters that the
original sweep (sensitivity_sweep.py, 63 variations) did not vary:
  b10   restoration rate degraded -> operational    x0.5, x2
  b21   repair-initiation rate failed -> degraded   x0.5, x2
  xi    Pareto shape of the heavy-tailed delay      2.1, 3.0, 4.0  (nominal 2.5)
  lam1  task rate in the degraded state (lambda_1/lambda_0 = 0.4)  x0.5, x1.5
For each variation and network regime: the switching thresholds of the static
optimum and the loss per decision of the nominal-model policy on the perturbed
system, exactly as in the original sweep.
Output: results/analysis/sensitivity_extended.json (original + new rows)
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sensitivity_sweep import BASE, thresholds, regret, lam_bar

NEW = [("b10", [0.5, 2.0]), ("b21", [0.5, 2.0]), ("xi", [2.1 / 2.5, 3.0 / 2.5, 4.0 / 2.5]),
       ("lam1", [0.5, 1.5])]

if __name__ == "__main__":
    OUT = os.path.join(ROOT, "results", "analysis")
    rows = json.load(open(os.path.join(OUT, "sensitivity.json")))
    for mode in ["stable", "degraded", "noisy"]:
        for name, scales in NEW:
            for s in scales:
                p = dict(BASE); p[name] = BASE[name] * s
                th, _ = thresholds(mode, p)
                rg = regret(mode, p)
                rows.append(dict(mode=mode, param=name, scale=s, lam_bar=lam_bar(p), th=th, regret=rg))
                print(mode, name, round(p[name], 3), th, [round(x, 4) for x in rg], flush=True)
    json.dump(rows, open(os.path.join(OUT, "sensitivity_extended.json"), "w"), indent=1, default=float)
    new = [r for r in rows if r["param"] in dict(NEW)]
    worst = {n: max(max(r["regret"]) for r in new if r["param"] == n) for n, _ in NEW}
    old = {n: max(max(x for x in r["regret"] if x is not None) for r in rows
                  if r["param"] == n and r["param"] not in dict(NEW) and r["regret"][0] is not None)
           for n in {r["param"] for r in rows if r["param"] not in ("nominal", "wc") and r["param"] not in dict(NEW)}}
    print("worst loss per decision, new parameters:", {k: round(v, 4) for k, v in worst.items()})
    print("worst loss per decision, original parameters:", {k: round(v, 4) for k, v in old.items()})
