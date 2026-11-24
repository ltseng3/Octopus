# E1 multiobjective figures: joint risk, marginal accuracy, energy and latency

**Updated for OCTOPUS-Sim 0.6.0 frame execution.** This document describes the reproducible **figures**, not a new AI model or a field measurement. Each frame now runs one physical DAG and projects its recorded outputs into multiple ROI decision events; the original source outcomes are unchanged.

## Run everything

From the repository root:

```bash
python -m pip install -e '.[plot,test]'
python scripts/run_e1_multiobjective.py --output samples/multiobjective_e1
python -m pytest -q
```

The generator uses one V2V4Real-derived trace (the bundled, verified sequence `0007`), plus the existing three-agent **synthetic** world. The synthetic experiment is enlarged from **80** to **3,000 independent decision events** by default to avoid interpreting a zero error count from a tiny sample as a zero underlying error probability. Reproduction takes no CUDA, OPV2V installation, or LiDAR input. It requires `matplotlib` and its NumPy dependency.

## What the four main figures show

| Output | Dataset | Left panel | Right panel |
| --- | --- | --- | --- |
| `01_v2v4real_0007_error_vs_energy.png` | Real recorded vision predictions, 2 agents | All-agents-wrong error rate vs incremental energy | Average agent error vs incremental energy |
| `02_three_agent_synthetic_error_vs_energy.png` | Synthetic, 3 agents | All-three-wrong error vs incremental energy | Average agent error vs incremental energy |
| `03_v2v4real_0007_error_vs_latency.png` | V2V4Real predictions, 2 agents | All-agents-wrong error vs mean delivered latency | Average agent error vs mean delivered latency |
| `04_three_agent_synthetic_error_vs_latency.png` | Synthetic, 3 agents | All-three-wrong error vs mean delivered latency | Average agent error vs mean delivered latency |

Supplemental outputs:

- `05_energy_profile_sensitivity_bands.png`: shaded **deterministic sensitivity ranges** across nine hypothetical energy configurations, with the nominal profile as a line. The plots are **separate per workload**, not a shared scale comparison of different numbers of agents.
- `06_latency_mean_and_p95.png`: modeled mean and P95 per-agent delivery latency as fusion selection rises.
- `multiobjective_policy_metrics.csv`: joint and marginal rates, both error intervals, normalized and aggregate joules, assumed energy range, mean/P95 latency, misses, complete-event rate, jobs and transmitted bytes for all policies.
- `energy_profile_sensitivity_grid.csv`: every one of the nine power-profile perturbations for each of the ten workload/policy combinations.
- `v2v4real_block_bootstrap_sensitivity.csv`: sensitivity of the within-sequence error intervals to using 6-, 12-, or 24-frame temporal bootstrap blocks.
- `PROVENANCE.json`: source and statistical limitations, experiment settings and calibration status.

## Definitions and fair comparisons

The five edge policies are local-only (`κ=0`) and probabilistic cooperative late fusion at `κ=0.25,0.50,0.75,1.00`. When fusion is selected, **all local detectors still run**, results are uploaded, fusion executes once at the edge, and the common result is delivered to all participating agents. These configurations use the correct P0 inference DAG; sharing does **not** save local detection work in this example.

For every common binary ROI decision event:

- **Collective joint error:** 1 if *all agents* received wrong decisions, otherwise 0. These plots require `complete_event_rate=100%`; missing decisions would require separate operational-failure treatment.
- **Average marginal error:** for the event, the number of wrong agent decisions divided by the number of agents; average over decision events. Within-agent and cross-agent correlation is preserved by resampling events/frames rather than individual agent predictions.
- **Incremental energy:** sum of actual modeled compute work and endpoint radio TX/RX energy. It **excludes** the common idle baseline. Per-event energy is total incremental joules divided by the number of common decision events. Work draining after the common horizon remains charged (see `ENERGY_E0_E1.md`). The reported total joules and idle baseline are available in the CSV.
- **Delivered latency:** per-agent end-to-end elapsed simulation time from decision-event release to result delivery; plots show the mean over delivered agent results. P95 and missed deadlines are separately recorded. These are simulated timing values, not measured latency on V2V4Real vehicles.

In V2V4Real, the diagnostic uses **two regions per recorded frame** arriving at the same simulated time. Under v0.6 frame execution they **share one frame-level detector output per vehicle**; the two ROI queries do not generate extra inference jobs or fictitious intra-frame queues. Real queueing can occur between *distinct* frames when their physical workflows overlap. The frame release interval, network speed, inference execution times, and edge worker count remain explicitly assumed. No claim is made about actual vehicle runtime.

## Error intervals: different evidence warrants different resampling

**Three-agent synthetic:** repeated event outcomes are independently generated by the synthetic backend (despite modeled correlation among agents *within* an event). We use 3,000 independent event clusters and a deterministic 3,000-replicate nonparametric percentile bootstrap. Intervals are Monte Carlo estimates for the configured synthetic distribution.

**V2V4Real:** 442 binary decisions are **not** 442 independent scenes. They arise from **221 consecutive frames of a single driving sequence**, with two regions per frame. We resample *whole recorded frames* using a circular moving-block bootstrap of **12 consecutive frames** (3,000 replicates, fixed seed). This retains the two region outcomes within each frame and some temporal dependence between adjacent frames. We also provide 6- and 24-frame sensitivity checks. These intervals are exploratory estimates **conditional on this sequence and the sampled counterfactual policy**. They **do not** support a claim about variation across driving scenes or datasets. The separate existing nine-sequence reliability study in `empirical/v2v4real` remains the broader dataset-level evidence.

Intervals are pointwise, not simultaneous across all plotted policies or an assertion of causal effect. The same resampling design is used across policies for comparability.

## Energy sensitivity bands: not statistical confidence intervals

The default energy profiles are hypothetical. To disclose uncertainty in these **assumptions**, the generator varies:

- Radio TX/RX joules per bit by factors `0.25`, `1`, and `4`.
- Edge fusion incremental watts by factors `0.5`, `1`, and `2`.

For each already-executed plan, the energy is exactly reweighted as:

\[
E(r,f) = E_{\mathrm{compute,other}}+fE_{\mathrm{fusion}}+rE_{\mathrm{radio}}.
\]

This is valid because the E1 profiles have **linear, additive compute power** and the **per-bit radio model**. We tested the equation against actual simulator reruns at the parameter-grid endpoints. It does not change the execution plan, latency, model prediction or packet-loss outcome. The horizontal bars on the energy tradeoff panels and the shaded `05` plot show minimum and maximum values across these **nine user-selected profiles**. These bands **must not** be labeled 95% CIs or interpreted as measured hardware variability.

## Principal numerical observations

**Bundled V2V4Real sequence `0007` (442 decisions, one recorded scene):**

| Shared late fusion | Both wrong | Average marginal error | Incremental energy (J / decision) | Mean delivered latency (ms) | P95 delivered (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0% | 0.905% | 22.624% | 0.0800 | 20.000 | 20.000 |
| 25% | 0.905% | 17.421% | 0.0898 | 23.844 | 36.339 |
| 50% | 1.357% | 12.557% | 0.1002 | 27.911 | 36.339 |
| 75% | 2.262% | 8.145% | 0.1106 | 31.977 | 36.339 |
| 100% | 3.620% | 3.620% | 0.1217 | 36.339 | 36.339 |

**Correction from v0.5.1:** the physical workflow is executed once per
synchronized frame (221 frames) and reused across the two region decisions.
Older v0.5.1 results charged the detector and fusion twice per frame, making
resource and latency estimates too high. We retain the old outputs only in
`samples/legacy_v051/`. The middle three joint-risk values can change because
v0.6 chooses the fusion mode once per frame, not independently for each ROI.

**Three-agent synthetic benchmark (3,000 independent events):**

| Shared late fusion | All three wrong | Average marginal error | Incremental energy (J / decision) | Mean delivered latency (ms) |
| --- | ---: | ---: | ---: | ---: |
| 0% | 1.333% | 9.933% | 0.2400 | 20.000 |
| 25% | 2.667% | 9.044% | 0.2664 | 24.672 |
| 50% | 3.767% | 8.044% | 0.2923 | 29.238 |
| 75% | 4.467% | 6.800% | 0.3186 | 33.892 |
| 100% | 5.367% | 5.367% | 0.3448 | 38.514 |

The `0%` synthetic estimate differs from the previous 80-event figure (which happened to observe zero all-three-wrong events). This is an **intentional larger independent synthetic run**, not a change to the original fixed 80-event fixture or to the inference engine.

In both examples, **joint risk increases while average marginal error decreases** as fusion selection increases. At the same time, **assumed incremental energy and latency increase**. Hence local-only is *not* globally dominant: although it has lower joint risk and resource consumption in these configurations, its individual-agent accuracy is worse. A meaningful edge-allocation policy must consider all objectives.

## Important limits for publication

1. The energy and timing values are **assumed**, not hardware measurements or calibrated V2V4Real ground truth. The figures are engineering/benchmark demonstrations, not measured savings or costs in a deployed system.
2. The vision predictions are genuine publicly released V2V4Real-derived detections, but the cooperative reuse policies are **counterfactual**.
3. V2V4Real here is **one bundled sequence**, not the full nine-sequence event-level resource replay. Do not attach the earlier nine-sequence scene-cluster bootstrap interpretation to this plot.
4. Changing inference allocation choices changes multiple aspects of the full edge workflow, including the cooperative fusion method. The study benchmarks complete system choices; it does not assert that numerical differences were caused exclusively by reuse.
5. There are no missing deliveries in these plotted configurations. The `scripts/run_frame_cache_study.py` experiment demonstrates high-load deadline misses, which must be reported separately from delivered-only latency.
6. Error does **not depend on latency** in this version of the simulator except indirectly via deadline misses. The latency–error plot shows which simulated configurations lie at which points, **not** that waiting longer causes inaccurate neural-network predictions.
7. The five points are one seeded sequence of policy selections. Statistical robustness to source-scene variation and different random policy schedules deserves a separate experimental study.

## Future extensions

- Rerun all nine V2V4Real scenes with their full recorded candidate traces and report scene-clustered uncertainty for resource/risk analyses.
- Add measured model runtime, payload-size, and TX/RX power profiles; parameter uncertainty should become measurement-backed rather than arbitrary envelopes.
- Add multi-objective resource- and dependence-aware edge policy baselines that selectively verify shared results.
- When deadlines become binding, report `operational_failure` and timely accuracy in addition to accuracy conditional on delivery.
