# Hybrid Server Allocation for Industrial Multi-Server Systems

Code, trained models and results for the paper

> **Integrating Stochastic Optimization and Deep Reinforcement Learning for Resource Allocation in Industrial Multi-Server Systems**
> Md Abid Hassan Mitul and Most. Nasrin Akhter, Department of Mathematics, Dhaka University of Engineering & Technology (DUET).

The project studies how many servers an industrial installation (for example, a SCADA server layer) should keep active while hundreds of machines send it data. The machines degrade, fail and are repaired, production lines start and stop, and the network delay grows with server congestion. The repository contains:

- a stochastic model of the system: a continuous-time Markov chain of machine conditions, an M/M/c/K server pool and a congestion-coupled network delay;
- a fast, validated event-driven simulator of this model;
- allocation controllers: fixed and threshold rules, a GA-tuned rule, a queueing-model policy, a **self-calibrating** queueing policy that estimates the service rate online, PPO agents and an ensemble of PPO agents;
- the proposed **hybrid controller**, which uses the calibrated queueing model by default and hands the decision to the PPO ensemble when the measured sojourn time persistently exceeds the model's prediction;
- an evaluation over 69 scenarios (seen and unseen loads, parameter mismatch and structural mismatch) with paired statistical tests.

## Main result

Mean shortfall relative to the best fixed allocation of each scenario (lower is better; 69 scenarios, 20 paired episodes each):

| Controller | Mean | Worst case |
|---|---|---|
| **Proposed hybrid** | **0.059** | 0.789 |
| PPO ensemble | 0.085 | 0.664 |
| PPO (single agent) | 0.153 | 1.036 |
| Fixed 6 servers | 0.177 | 1.381 |
| Self-calibrating model | 0.215 | 1.932 |
| Queueing model (fixed parameters) | 0.249 | 2.074 |
| GA-tuned rule | 0.456 | 2.210 |
| Threshold rule | 0.513 | 2.769 |

A calibrated queueing model is sufficient whenever the structure of the model holds; learning is decisive when work arrives or is processed in bursts. Full tables are produced by `analysis/make_tables.py`.

## Repository structure

```
.
├── parameters.py               model parameters (engineering estimates)
├── fast_env.py                 event-driven simulator (Gymnasium environment, Numba)
├── factory_env.py              independent SimPy reference implementation
├── validate_fast_env.py        validation against SimPy and the M/M/c/K closed form
├── baselines.py                round robin, threshold rule, GA-tuned rule
├── model_policy.py             queueing-model policy and model predictions
├── variants.py                 PPO ensemble, self-calibrating policy, proposed hybrid
├── hybrid_policy.py            hybrid with the fixed-parameter model (ablation)
├── calibrate_hybrid.py         alarm thresholds, calibrated on nominal conditions only
├── hybrid_thresholds.json      calibrated thresholds
├── run_experiments.py          PPO training and evaluation (resumable, time-limited)
├── fast_evaluate.py            scenarios and evaluation of all policies
├── run_extra.py                parameter-mismatch tests and server-cost study
├── evaluate_hybrid.py          hybrid with fixed-parameter model on all scenarios
├── evaluate_variants.py        ensemble, self-calibrating model, hybrids
├── evaluate_structural.py      structural-mismatch scenarios (two sets)
├── stats.py                    confidence intervals, Wilcoxon tests, Holm correction
├── experiment_config.json      main configuration
├── experiment_config_server_cost_x2.json   configuration with doubled server cost
├── models/                     trained PPO agents (3 network regimes x 5 seeds)
├── models_server_cost_x2/      PPO agents trained with doubled server cost
├── results/                    all evaluation data used in the paper
│   ├── main/                   seen, unseen and slow-server scenarios; learning curves
│   ├── parameter_mismatch/     further parameter-mismatch scenarios
│   ├── controllers/            ensemble, self-calibrating and hybrid controllers; structural scenarios
│   ├── server_cost_x2/         doubled server-cost study
│   └── analysis/               consolidated shortfalls, tests and LaTeX tables
├── analysis/                   scripts that produce the tables and figures of the paper
└── figures/                    figures of the paper
```

## Installation

Python 3.9 or newer.

```bash
pip install -r requirements.txt
```

## Reproducing the results

All results in `results/` can be regenerated with the commands below. Every evaluation uses the same random seeds (common random numbers), and the simulator gives identical results on Windows and Linux.

**1. Validate the simulator** (about 2 minutes):

```bash
python validate_fast_env.py --quick
```

**2. Train the PPO agents and evaluate the main scenarios.** The runner stops cleanly before the time limit and continues where it stopped when started again:

```bash
python run_experiments.py --hours 4
python run_experiments.py --status
```

The trained agents used in the paper are already in `models/`; running this command in a fresh copy retrains them and overwrites that folder. One agent trains in about 14-17 minutes on a laptop with an Intel Core i5-1335U processor; the full plan (15 agents and the evaluation) takes about 4-5 hours.

**3. Robustness experiments:**

```bash
python run_extra.py --mismatch          # parameter-mismatch scenarios
python run_extra.py --setup-weights     # doubled server cost (then run_experiments.py in ../DRL_wc010)
```

**4. Proposed hybrid and the other controllers:**

```bash
python calibrate_hybrid.py              # thresholds of the fixed-parameter hybrid
python calibrate_hybrid.py --adaptive   # thresholds of the proposed hybrid
python evaluate_hybrid.py --network stable          # repeat for degraded and noisy
python evaluate_variants.py --network stable
python evaluate_structural.py --network stable
python evaluate_structural.py --network stable --set2
```

New results are written to `data/`; the published ones are in `results/`.

**5. Tables and figures of the paper** (about 15 seconds):

```bash
python analysis/sensitivity_sweep.py
python analysis/build_results.py
python analysis/make_tables.py
python analysis/make_figures.py
```

## Model parameters

The parameters in `parameters.py` describe a representative SCADA server layer of a textile plant. They are engineering estimates, not measurements from a particular plant; reward weights and costs are design choices. The paper examines the sensitivity of the results to these values (63 parameter variations) and evaluates the controllers when the true parameters and the model structure differ from those assumed.

## Citation

If you use this code, please cite the paper (citation details will be added after publication) or the repository via `CITATION.cff`.

## License

MIT License. See `LICENSE`.

## Contact

**Md Abid Hassan Mitul**

- Email: mitul.mdabidhassan@gmail.com
- Website: https://abid-mitul.github.io
- ORCID: https://orcid.org/0000-0003-4268-9873
- LinkedIn: https://www.linkedin.com/in/mitul-abid
