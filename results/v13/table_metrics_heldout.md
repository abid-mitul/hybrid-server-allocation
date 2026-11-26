| Controller | throughput | latency_ms | utilisation_pct | blocking_pct | sla_violation_pct | servers | reward |
|---|---|---|---|---|---|---|---|
| Fixed c_max = 6 | 986.9 | 131.2 | 49.4 | 0.001 | 3.72 | 6.00 | 6.615 |
| Round-Robin | 957.7 | 5570.4 | 64.9 | 2.448 | 30.59 | 4.43 | 3.952 |
| Threshold rule | 986.6 | 360.7 | 58.9 | 0.043 | 11.43 | 5.02 | 6.393 |
| GA-tuned rule | 986.8 | 268.1 | 56.1 | 0.031 | 8.41 | 5.25 | 6.415 |
| Autoscaler | 986.5 | 164.9 | 54.4 | 0.004 | 5.74 | 5.42 | 6.722 |
| Self-calibrating model | 986.6 | 154.3 | 53.8 | 0.003 | 5.28 | 5.47 | 6.738 |
| PPO (single agent) | 986.9 | 162.3 | 53.0 | 0.004 | 5.34 | 5.59 | 6.671 |
| PPO ensemble | 986.8 | 145.2 | 52.1 | 0.002 | 4.64 | 5.68 | 6.686 |
| CUSUM + floors, no agents | 986.6 | 152.8 | 53.8 | 0.003 | 5.23 | 5.47 | 6.739 |
| CHAMPS | 986.6 | 147.2 | 53.2 | 0.002 | 4.95 | 5.53 | 6.723 |
| Perfect-information model policy | 986.7 | 154.8 | 53.9 | 0.003 | 5.29 | 5.47 | 6.738 |
