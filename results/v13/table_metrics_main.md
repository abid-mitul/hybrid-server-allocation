| Controller | throughput | latency_ms | utilisation_pct | blocking_pct | sla_violation_pct | servers | reward |
|---|---|---|---|---|---|---|---|
| Fixed c_max = 6 | 947.4 | 276.3 | 47.4 | 0.002 | 8.28 | 6.00 | 5.894 |
| Round-Robin | 904.4 | 6479.5 | 61.3 | 3.328 | 33.82 | 4.43 | 3.315 |
| Threshold rule | 943.9 | 862.0 | 56.0 | 0.172 | 16.59 | 5.01 | 5.552 |
| GA-tuned rule | 943.0 | 957.0 | 54.5 | 0.325 | 14.16 | 5.11 | 5.610 |
| Autoscaler | 946.5 | 360.6 | 53.0 | 0.024 | 10.63 | 5.32 | 6.007 |
| Self-calibrating model | 945.1 | 595.7 | 53.6 | 0.121 | 12.28 | 5.23 | 5.862 |
| PPO (single agent) | 946.6 | 408.3 | 52.3 | 0.035 | 11.13 | 5.39 | 5.918 |
| PPO ensemble | 947.0 | 325.5 | 51.6 | 0.008 | 10.04 | 5.46 | 5.986 |
| CUSUM + floors, no agents | 946.2 | 346.6 | 52.7 | 0.014 | 10.46 | 5.33 | 6.018 |
| CHAMPS | 946.7 | 321.6 | 52.1 | 0.011 | 9.77 | 5.40 | 6.041 |
| Perfect-information model policy | 946.9 | 323.4 | 53.4 | 0.002 | 10.82 | 5.26 | 6.029 |
