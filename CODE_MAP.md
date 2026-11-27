# Code map: where each part of the paper lives

Section numbers refer to the CHAMPS manuscript (version 1.3). Start with
`python demo.py` for a one-minute live run of the main ideas.

## Section 3: the stochastic model of the plant

| File | What it does | Paper |
|---|---|---|
| `parameters.py` | All case-study parameters: machine rates, task rates, service rate, buffer, delay, SLA limit, reward weights | 3, 6.2 (Table 2) |
| `fast_env.py` | The main plant simulator: machines move between operational, degraded and failed; requests queue at `c` servers with buffer `K`; delay grows with congestion. Event-driven, compiled with Numba. Also the Gymnasium environment the PPO agents learn in | 3.1 to 3.5, 5.1, 6.1 |
| `factory_env.py` | Independent second simulator written in SimPy, used only to check `fast_env.py` | 6.1 |
| `validate_fast_env.py` | Checks `fast_env.py` against `factory_env.py` and the exact M/M/c/K formulas (`mmck`) | 6.1 |

## Section 4: queueing analysis and the static optimum

| File | What it does | Paper |
|---|---|---|
| `model_policy.py` | M/M/c/K formulas (blocking, waiting, compliance) and the expected reward of each allocation; the queueing-model policy (`QueueingModelPolicy`) | 4.1, 4.2 |
| `analysis/sensitivity_sweep.py`, `analysis/sensitivity_extended.py` | Varies the 12 parameters (90 variations) and shows that only the service rate moves the optimum | 4.3 |

## Section 5: the CHAMPS controller

| File | What it does | Paper |
|---|---|---|
| `variants.py` | `AdaptiveModelPolicy`: the self-calibrating model that estimates the service rate online; `EnsemblePolicy`: median of the 5 PPO agents | 5.2, 5.4 |
| `cusum_test.py` | The CUSUM validity test on the ratio of measured to predicted response time (`CusumHybridPolicy`) | 5.3 |
| `calibrate_hybrid.py`, `cusum_thresholds.json` | Calibration of the test thresholds under normal conditions only | 5.3 |
| `autoscaler.py`, `autoscaler_targets.json` | The Kubernetes-type autoscaler (`HPAPolicy`), used as a baseline and as a floor | 5.5, 6.4 |
| `hybrid3.py` | **CHAMPS itself** (`Hybrid3Policy`): model by default, PPO ensemble after an alarm, never below the autoscaler or model floor | 5.6 |
| `regime_agnostic.py` | The variant that does not know the network regime | 5.6, 7.6 |
| `run_experiments.py` | Trains the 15 PPO agents (5 seeds x 3 networks, 200,000 steps each) and runs the main evaluation | 5.4, Appendix C |

## Section 6: experimental design

| File | What it does | Paper |
|---|---|---|
| `fast_evaluate.py` | Scenario definitions, `run_episode`, the GA baseline (`fit_ga`) and the evaluation loop | 6.3, 6.4 |
| `baselines.py` | Round-Robin and the threshold (fixed-priority) rule | 6.4 |
| `evaluate_oracle.py` | The perfect-information model policy (oracle) | 6.4 |
| `experiment_config.json` | Seeds, training length, episode length, evaluation settings | 6.4 |
| `heldout_env.py` | The three held-out departures (correlated service, server failures, periodic polling), frozen in commit 4474e5a | 6.3 |
| `stats.py` | Paired Wilcoxon tests with Holm correction | 6.4 |

## Section 7: results

| File | What it does | Paper |
|---|---|---|
| `rerun_measured_util.py`, `evaluate_structural.py`, `evaluate_fallback.py`, `safety_floor.py` | Main 69-scenario runs and ablations | 7.1 to 7.4 |
| `v13_ppo_role.py` | CHAMPS with and without the agents, and which component sets each decision | 7.4 |
| `server_cost_study.py`, `final_checks.py` | Doubled server cost | 7.5 |
| `regime_switch.py` | Network change during operation | 7.6 |
| `c3_heldout.py`, `c3_round_robin.py` | Held-out evaluation | 7.7 |
| `c4_false_alarm.py` | False-alarm rate of the CUSUM test | 5.3 |
| `analysis/v13_tables.py` | Results tables of Section 7 and Appendix B | 7, App. B |
| `analysis/v13_heldout_summary.py` | Held-out table | 7.7 |
| `analysis/v13_metrics_table.py` | Latency, blocking, SLA, utilisation table | App. B |
| `analysis/make_figures_v2.py` | Figures, including the bursty-traffic trace | 4, 7 |

## Folders

| Folder | Contents |
|---|---|
| `models/` | The 15 trained PPO agents (`models/<network>/ppo_seed_<s>.zip`) |
| `models_server_cost_x2/` | Agents retrained for the doubled server cost |
| `data/` | Raw per-episode results of every run |
| `results/` | Result tables and summaries used in the paper |
| `figures/` | Figures of the paper |
