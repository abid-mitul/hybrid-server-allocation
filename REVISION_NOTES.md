# Revision notes: Draft v1.1 → v1.2

Branch: `revision/measured-utilisation`. The numbering follows `03_action_plan.md` from the internal review.

## Item 1. Utilisation oracle removed: done

**What changed.** In the 39 non-structural scenarios (seen, unseen and parameter mismatch), the simulator used to compute utilisation as |J|/(τ c µ_true). That made the self-calibration (22) return µ exactly. All 39 scenarios have been re-run with utilisation computed from the recorded processing times, which the 30 structural scenarios already did. The doubled server-cost study (Table 10) has been re-run the same way.

Training is unchanged: the PPO agents, the GA rule and the alarm thresholds. The thresholds were already calibrated with measured utilisation.

**Reproducibility check.** With the old setting, this code reproduces the published episode rewards exactly (difference 0), in both the main study and the doubled server-cost study. Every change below is therefore caused by the utilisation alone.

**Result.** The headline holds and improves slightly:

| Controller | Published mean (worst) | Revised mean (worst) |
|---|---|---|
| Proposed hybrid | 0.059 (0.789) | **0.053** (0.789) |
| PPO ensemble | 0.085 (0.664) | 0.084 (0.664) |
| PPO (single agent) | 0.153 (1.036) | 0.153 (1.036) |
| Fixed c = 6 | 0.177 (1.381) | 0.177 (1.381) |
| Self-calibrating model | 0.215 (1.932) | 0.209 (1.932) |
| Queueing model | 0.249 (2.074) | 0.249 (2.074) |
| GA-tuned rule | 0.456 (2.210) | 0.461 (2.210) |
| Threshold rule | 0.513 (2.769) | 0.519 (2.769) |

- In every parameter-mismatch row of Table 7, the self-calibrating model and the hybrid are unchanged to three decimals (group means 0.001 and 0.011). The oracle did not inflate these results; with about 20,000–60,000 completed tasks per window, the measured utilisation is accurate to about 1%.
- The changes come from transients at high load. In the overloaded first window, the oracle value read 0.746, because delivered throughput lags behind processing. The measured value reads 0.997. The calibrated model therefore sees the overload and selects six servers immediately, instead of five. The calibrated rate briefly swings (249, then 408) and is back within about 1% of 333 by the third window.
  - High load: 0.074 → 0.009.
  - Drop 350 → 200: −0.060 → −0.129.
  - Seen group for the hybrid: 0.020 → 0.003.
- The hybrid is within 0.05 of the best fixed allocation in 52 of 69 scenarios (was 49); the self-calibrating model in 53 (was 50).
- The hybrid's mean shortfall is 1.0% of the best fixed reward (was 1.1%).
- The threshold rule and the GA rule are slightly worse (they were tuned on the expected utilisation, which is noise-free). The fixed allocations and the fixed-parameter queueing model do not read utilisation and are unchanged.

**Revised Table 8** (hybrid significantly better / no significant difference / significantly worse):

| Hybrid vs. | Seen (12) | Unseen (9) | Param. (18) | Struct. (30) | All (69) |
|---|---|---|---|---|---|
| Fixed c = 6 | 6/6/0 | 6/3/0 | 10/7/1 | 5/14/11 | 27/30/12 |
| Threshold rule | 6/6/0 | 9/0/0 | 6/11/1 | 20/8/2 | 41/25/3 |
| GA-tuned rule | 1/11/0 | 6/3/0 | 9/8/1 | 24/5/1 | 40/27/2 |
| Queueing model | 3/9/0 | 2/7/0 | 9/8/1 | 11/17/2 | 25/41/3 |
| Self-calibrating model | 0/12/0 | 0/9/0 | 0/16/2 | 15/14/1 | 15/51/3 |
| PPO (single agent) | 4/8/0 | 5/4/0 | 17/1/0 | 9/19/2 | 35/32/2 |
| PPO ensemble | 1/11/0 | 4/5/0 | 11/7/0 | 2/24/4 | 18/47/4 |

The hybrid is no longer significantly worse than any controller on the seen or unseen loads.

**Revised Table 9** (seen / unseen / param. / struct. / mean / worst): only the proposed row changes, to 0.003 / −0.042 / 0.011 / 0.128 / 0.053 / 0.789. The two fixed-parameter hybrids are unchanged.

**Revised Table 10** (stable network, 8 scenarios; mean and worst):

| Controller | w_C = 0.05 | w_C = 0.10 |
|---|---|---|
| GA-tuned rule | 0.172, 0.550 | 0.399, 1.546 |
| Queueing model | 0.078, 0.593 | 0.196, 2.224 |
| Self-calibrating model | −0.015, 0.019 | −0.098, 0.007 |
| PPO (single agent) | 0.047, 0.339 | 0.130, 1.226 |
| PPO ensemble | 0.026, 0.361 | 0.118, 1.559 |
| Proposed hybrid | −0.014, 0.019 | −0.098, 0.007 |

The full revised Table 7 by scenario type is in `results/analysis_measured/table7_by_type.csv`.

**Text changes for item 1.**
1. Section 6.1, simulator paragraph. Replace "In these configurations the utilisation Û_k is computed from the recorded processing times, as a server log would provide it; otherwise it is computed as |J_k|/(τ c_k µ) with the true service rate, its expected value." with: "In every scenario the utilisation Û_k is computed from the processing times recorded in the window, as a server log would provide it."
2. Section 5.3, after (22), add: "Because throughput is counted when responses are delivered and processing time when tasks complete, the estimate departs from the true rate for one or two windows after a sudden overload, and then settles within about one percent of it."
3. Section 7.2: update all overall numbers from the first table above.
4. Section 7.3: the 0.074 loss at high load now belongs to the fixed-parameter policy only. Add that the self-calibrating model avoids most of it (0.009): the measured utilisation of the overloaded first window is close to one, so the calibrated rate falls and the policy moves to six servers at once.
5. Section 8.4 (Limitations): delete the sentence "In the parametric scenarios the simulator computes the utilisation from the true service rate…". Replace it with a sentence saying that utilisation is measured from simulated processing times, and that a real log may add noise or timestamp misalignment beyond what is modelled.

## Item 2. "Model + c_max fallback" baseline: done, framing decision needed

**Baseline.** The proposed hybrid with its learned mode replaced by six fixed servers. It uses the same self-calibrating model, the same mismatch test and threshold, and m_on = 2, m_off = 3. It was run on all 69 scenarios with measured utilisation and the same seeds (`evaluate_fallback.py`).

| Controller | Seen | Unseen | Param. | Struct. | Mean | Worst |
|---|---|---|---|---|---|---|
| Proposed hybrid (PPO ensemble on alarm) | 0.003 | −0.042 | 0.011 | 0.128 | 0.053 | 0.789 |
| Model + c = 6 on alarm | 0.003 | −0.042 | 0.010 | **0.112** | **0.047** | **0.629** |
| Fixed c = 6 | 0.417 | 0.311 | 0.139 | 0.064 | 0.177 | 1.381 |

Paired tests, hybrid vs. fallback, Holm-adjusted over eight comparisons per scenario: **3 better / 63 no difference / 3 worse.**
- **The hybrid is better in three scenarios:**
  - high-variance service at N = 200 on the stable and noisy networks, where PPO keeps about 5 servers instead of 5.1–5.4;
  - bursty service at N = 200 on the stable network, where the fallback oscillates: at six servers the alarm clears, the model drops servers, and the alarm fires again (14 switches per episode against 8.6).
- **The hybrid is worse in three scenarios:** bursty traffic at N = 200 on the degraded and noisy networks (0.517 against 0.141; 0.653 against 0.328), and the combined mismatch on the stable network.
- In most structural scenarios the learned mode is active in about 90% of the decisions and chooses about 5.95 servers on average. In practice it largely reproduces "six servers".

**What this means.** The reviewer's concern is confirmed. The robustness under structural mismatch comes from the validity test and the conservative fallback, not from learning. Given the alarm, PPO adds value only in a few scenarios and loses it in about as many. Both controllers still trail fixed c = 6 in the structural group (0.064), because the first windows before the alarm fires are lost.

**Options for the paper (the author's decision):**
- **(A) Reframe around the validity test.** The contribution becomes "a calibrated queueing model plus a statistical validity test"; the PPO ensemble and c = 6 are two fallback choices, compared honestly. This is the most defensible option. The title and the DRL emphasis would need softening, and the question "when does learning add value?" would be answered with "rarely, once the model is checked".
- **(B) Keep PPO as the fallback and show where it pays off.** Report the fallback baseline, argue for PPO where it saves servers or avoids oscillation (high variance, bursty service), and add a server-cost-weighted comparison. The overall numbers favour the fallback, so reviewers may still push back.
- **(C) Improve the learned mode.** For example, train the agents with structural disturbances or with the alarm state in the observation. The structural group would then no longer be a clean out-of-sample test, so new held-out structural cases would be needed. This is the most work.

## Item 3. Oracle adaptive baseline: done

**Baseline.** `evaluate_oracle.py` applies the model rule (21), with the same horizon and switching cost, using perfect information instead of estimates. At each decision it reads from the simulator:
- the true arrival rate of the next window, including a population change scheduled for that window;
- the true processing rate of the active servers, including mixed speeds;
- the true congestion coefficient.

It is the best the queueing analysis can do when nothing is estimated. Where the model's structure holds (39 scenarios), it is the upper reference. Under structural mismatch it is *not* an upper bound. The run covers all 69 scenarios with the common seeds; the analysis is in `analysis/compare_oracle.py`.

**Results.**

| | Seen | Unseen | Param. | Struct. |
|---|---|---|---|---|
| Oracle vs. best fixed (negative = oracle better) | −0.037 | −0.052 | −0.001 | +0.127 |
| Gap of the hybrid to the oracle | 0.041 | 0.009 | 0.012 | 0.000 |
| Gap of the self-calibrating model to the oracle | 0.041 | 0.009 | 0.002 | 0.365 |
| Gap of the PPO ensemble to the oracle | 0.061 | 0.056 | 0.135 | −0.024 |

- **The best fixed allocation is essentially optimal except when the population changes.** In the 33 in-model scenarios with a constant population, the oracle improves on it by only 0.001, and the hybrid is within **0.008** of the oracle.
- **The gap is concentrated in the six population-change scenarios** (burst 200 → 350 and drop 350 → 200). There the oracle beats the best fixed allocation by 0.151, because it knows of the change before the window in which it happens. The hybrid trails the oracle by 0.089 there: it can only react one window later, so this is a gap in information, not in modelling.
- **Over all 39 in-model scenarios,** the hybrid's gap to the oracle is 0.020, which is 0.29% of the oracle's reward. Paired tests (hybrid better / no difference / worse than the oracle) give 0/30/9 in-model, mainly in the burst scenarios.
- **One false alarm stands out.** Under doubled congestion on the degraded network, the hybrid trails the oracle by 0.140, while the self-calibrating model alone trails it by 0.000. The calibration corrects µ but not κ, so the mismatch test fires and hands control to the ensemble unnecessarily.
- **Under structural mismatch, perfect parameters do not help.** The oracle loses up to 0.654 to the best fixed allocation (batched traffic, bursty traffic and bursty service at N = 200). The hybrid and the fallback beat the oracle in those scenarios. This shows directly that the structure of the model, not its parameters, limits the analysis there, which supports the role of the validity test.

**Text consequences.**
- Abstract (item 8): replace "about one percent of the achievable reward" with a statement against both references. For example: within 0.3% of a perfect-information model-based oracle wherever the model's structure holds, and 1.0% below the best fixed allocation on average over all 69 scenarios.
- Results: add the oracle as a reference column. State that "best fixed" understates what adaptation can achieve only when the population changes.
- Discussion: mention the false alarm under doubled congestion, a case for calibrating κ as well as µ.

## Framing decision

The author chose **option A**. The contribution is the self-calibrating queueing model plus the statistical validity test. The fallback on alarm is a design choice: six servers or the PPO ensemble, and the two are compared openly.

## Item 4. Regime knowledge removed: done

**Method** (`regime_agnostic.py`). The regime is estimated online, with no new measurements.
- For each settled window (same c as the window before, no blocking, sojourn below T_SLA), the controller computes the log-ratio of measured to predicted mean sojourn under each regime. It picks the regime whose running mean log-ratio is closest to zero.
- Under nominal conditions the regimes are 0.21–0.49 apart in this log-ratio, with a window-to-window spread of 0.02–0.10.
- The mismatch test compares each window with the regime estimated *before* that window.
- One alarm threshold (1.4064) is calibrated on nominal runs pooled over the three regimes and the training loads. It lies within the range of the three regime-specific thresholds, 1.21–1.65.
- The learned fallback uses only the agents trained on the stable network.
- Nothing in the controller depends on the true regime.

**Results** (69 scenarios, measured utilisation, common seeds):

| Controller | Seen | Unseen | Param. | Struct. | Mean | Worst |
|---|---|---|---|---|---|---|
| Self-calibrating model, regime known | 0.003 | −0.043 | 0.001 | 0.492 | 0.209 | 1.932 |
| Self-calibrating model, regime estimated | 0.000 | −0.044 | −0.001 | 0.441 | 0.186 | 1.482 |
| Model + test + c = 6, regime known | 0.003 | −0.042 | 0.010 | 0.112 | **0.047** | **0.629** |
| Model + test + c = 6, regime estimated | 0.000 | −0.044 | 0.003 | 0.128 | 0.051 | 0.703 |
| Model + test + PPO, regime known (v1.1 proposed) | 0.003 | −0.042 | 0.011 | 0.128 | 0.053 | 0.789 |
| Model + test + PPO (stable agents), regime estimated | 0.000 | −0.044 | −0.000 | 0.145 | 0.058 | 0.866 |

Paired tests, regime-agnostic vs. regime-aware (better / same / worse): model 6/62/1; c = 6 fallback 6/59/4; PPO fallback 4/62/3.

- **Within the model** (seen, unseen and parameter mismatch), knowing the regime brings nothing. The agnostic versions are equal or slightly better: the regime is identified correctly in about 92% of decisions on the seen and unseen loads, and the misses are in the first two windows and at N = 350, where every regime calls for six servers.
- **One side benefit:** under doubled congestion, the regime estimate absorbs the extra delay and the false alarm noted in item 3 almost disappears. On the degraded network the shortfall falls from 0.140 to 0.037.
- **Under structural mismatch, there is some masking.** Bursty traffic or bursty service at N = 200 on the stable and noisy networks is read as a heavier regime: the regime is correct in only 0–10% of those decisions, and the alarm fires less often (fallback share 0.43–0.61, against about 0.9 with the regime known). Structural shortfall rises from 0.112 to 0.128 for the c = 6 fallback, and the worst case from 0.629 to 0.703.

**Recommended text** (Section 5 and Limitations). Present the regime-aware controller as the configuration for a site whose network has been characterised once at commissioning; the regime is a property of the installation that changes rarely. Report the online estimator as the fully agnostic variant, which costs 0.004 in mean shortfall. State the masking mechanism plainly: an online estimate can partly absorb a structural departure. Point out that the regime should therefore be estimated during commissioning, or only during windows in which the model is known to hold.

## Remaining items

Under option A, the validity test is the core contribution, so the order of the remaining should-do items changes:
- **Item 5 (CUSUM or another sequential change-detection test) becomes essential.** It is the natural competitor to the persistence counter, and it may also reduce the delayed or missed alarms that cause both the structural losses and the masking above.
- **Item 6 (supervised classifier vs. DRL) matters less.** PPO is now one fallback option, not the core.
- **Items 8–18 (text) and 19–22 (submission package)** are unchanged.
