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

## Remaining items

Items 3–22 are still open. Item 3 (the oracle adaptive baseline c*(Λ_true)) is the next experiment.
