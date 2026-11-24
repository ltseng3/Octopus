# F0 parameter study for a CVC manuscript

**Status:** reproducible exploratory research figure package; not peer-reviewed, hardware-calibrated, or a real-world fusion quality claim. **Recommended main-text figures:** 01, 02, and 03. Figures 04 and 05 are useful supplements or limitation panels.

## Research question

Can a reliability-aware edge policy reduce some forms of collective error by recognizing shared evidence, without necessarily reducing *all-agent failure*, resource consumption, or the need for trustworthy lineage/error-probability metadata?

We deliberately distinguish three things that are often conflated:

- **R_m** = Pr(at least *m* agents make the wrong binary decision for the same ROI). R_5 is all-five failure; R_3 is a majority-level failure. R_m is a **diagnostic simultaneous-error probability**, not an operational safety guarantee.
- **Prediction-level correction** = number of final agent binary labels different from each agent's original local label. This is **not** the same as a shared result ID being delivered or an accepted fusion action.
- **Executed work** = physical frame-level local inference, uploads, edge-side F0 decision, optional frame downlinks. A rejected F0 decision still incurs already-executed computation and uplink costs.

## Controlled synthetic construction

We generate **2,000 distinct physical frames** with **two ROI labels each**, totaling **4,000 common binary decisions**, and five agents `a,b,c,d,e`. Truth is Bernoulli(0.5). The true **per-agent** marginal wrong probabilities are:

| Agent | a | b | c | d | e |
|---|---:|---:|---:|---:|---:|
| True wrong probability | 0.055 | 0.080 | 0.180 | 0.180 | 0.180 |

Agents **c,d,e form a potentially correlated clique**: with probability `rho`, their error indicators share a single Bernoulli(0.18) draw; otherwise, they use independent Bernoulli(0.18) draws. Agents a,b remain independent of one another and the clique's error process. `rho` is the **mixture weight**, not a measured Pearson correlation. The sweep uses `rho = 0, .25, .50, .75, 1`.

The generator draws all truths and underlying uniform random variables **once**, then varies `rho` while preserving the same potential source draws. Every competing policy sees **exactly the same** truth and local prediction candidates for a fixed `rho`. Different ROI labels from the same frame remain distinct decision events but reuse one physical execution DAG and result payload. The ROI-level errors themselves are independent across distinct ROIs by construction in this test; frame-level grouping is still preserved for inference accounting and resampling.

Seed `5701`; NumPy `Generator` (PCG64); source JSONL and its SHA-256 values are bundled. The trace contains `edge:*` placeholders copied from local predictions because the unchanged legacy parser requires those keys for local-only execution; **no separate edge-model candidate or cooperative pretrained fusion candidate is evaluated on this synthetic trace.** The F0 comparisons use only explicit `local:*` predictions.

### Methods compared

1. **Local-only:** no edge attempt; all agents keep their own labels.
2. **Naive majority:** all five agent votes counted, regardless of lineage; a tie would preserve local labels (not possible for five complete binary inputs).
3. **Naive error-weighted vote:** weights `log((1-q_i)/q_i)` for each agent with no dependence correction or selective fallback. This is a heuristic when errors are dependent.
4. **Lineage-aware F0:** grouping `(a)`, `(b)`, `(c,d,e)`, nominally correct source error estimates, minimum two distinct supporting evidence groups, and minimum surrogate confidence .90.
5. **Incorrect-lineage F0:** the same q estimates, but falsely declares c,d,e to be separate groups.
6. **Optimistic-clique F0:** declares the correct clique but falsely reports error `.04` instead of true `.18` for each clique member.
7. **Masked-clique F0:** true error estimates, correct lineage; all three clique members are excluded because their q exceeds the configured `.15` mask threshold. **This configuration did not improve upon correct-lineage F0 for this particular benchmark.**

We separately sweep the F0 surrogate-confidence acceptance threshold over `{.55,.70,.80,.90,.94,.97,.985,.995,.999}` with true q values and correct lineage. **The surrogate-confidence score is not a calibrated posterior**; the thresholds are sensitivity parameters, not empirical safety constraints.

### Execution model and checks

All **binary F0 decisions** in the synthetic grid are produced by the **actual integrated `SelectiveFuser.decide()`** (or the actual `unweighted_majority()` / `weighted_vote()` baseline), operating over the generated candidate observations. We use an efficient deterministic offline **decision replay** for the large grid rather than rerunning the discrete-event scheduler for every configuration. **Energy and physical network costs are analytic projections** under the pinned additive E1 per-bit model, with fixed 20 ms local detection, 1 ms F0 edge decision, 5 ms propagation, 250 ms frame arrivals, and 180 ms deadlines. Energy is **hypothetical**. No contention, packet loss or missed deadline occurs in this fixture.

**Integration validation:** the first **180 full frames / 360 ROI decisions** were independently executed by the **actual `GraphExecutor`** for eight configurations (including local-only, naive majority, naive weighted, correct/incorrect lineage, optimistic q, masking, strict fallback). **Every final ROI prediction, physical byte count, model job count, and joule breakdown matched** the analytic replay within tight numerical tolerance. See `online_executor_consistency.csv`. This does **not** mean all 31 large-grid configurations were rerun through the event-driven runtime.

For each generated policy result, the study reports marginal error, R_1...R_5, fusion coverage, actual changed labels, and modeled incremental joules per physical frame. **Costs for the local-only baseline were also checked using the actual executor**. Idle power is excluded from these comparisons; it is a policy-independent baseline over the configured common horizon.

### Statistical intervals

The study uses **2,500 fixed-seed, paired percentile bootstrap resamples** of **physical frames**, preserving both ROIs together in each resampled cluster. Shaded Figure 02 intervals are **pointwise 95% synthetic sampling intervals**, not family-wise simultaneous confidence bands over the many policies and rho settings. The same resample indices are used across policies to estimate paired changes. A difference in observed rates is not a guarantee under source-model misspecification or a claim about unseen real driving scenes.

At rho=.75, lineage-aware F0 minus local-only:

- **R_3 difference:** -12.050 percentage points, 95% paired frame-bootstrap interval approximately [-13.075, -11.050] pp.
- **R_5 difference:** +0.225 percentage points, interval approximately [+0.100, +0.400] pp.

These are conditional on the designed synthetic generator and are not calibrated empirical operating risks. Any eventual CVC claim needs evaluation under alternative error processes, unseen seeds, and real multi-sequence prediction traces.

## Figure interpretations and paper-ready caption drafts

### Figure 01 — The policy ranking depends on what counts as a collective failure

Files: `01_risk_definition_reversal.{pdf,svg,png}`.

**Suggested caption:** *Simultaneous error under different failure thresholds in a five-agent synthetic ensemble with a correlated three-agent clique (rho=.75; 4,000 binary ROI decisions). Left: R_m = Pr(at least m agents wrong), revealing that reductions in majority-level failure R_3 need not reduce all-agent failure R_5. Right: marginal error versus R_5 for the same executed decision policies. Error probabilities are controlled synthetic parameters, not calibrated model confidence.*

**Interpretation:** Local-only yields R_3=14.525% but R_5=0.050%, while lineage-aware F0 yields R_3=2.475% and R_5=0.275%. Naive majority turns majority-level clique failures into all-five failures. Naive reliability weighting improves average marginal error but still couples every agent to its wrong votes on accepted cases. **Do not write that F0 dominates local on both collective-risk metrics. It does not.**

### Figure 02 — Dependence metadata can matter more than voting confidence

Files: `02_lineage_quality_vs_correlation.{pdf,svg,png}`.

**Suggested caption:** *Effect of correlated evidence on selective edge fusion in a controlled five-agent synthetic experiment. A three-agent clique uses a shared wrong-indicator draw with mixture weight rho. F0 with correctly specified lineage discounts the clique as one group; the incorrect-lineage variant counts its votes separately. Shaded bands are pointwise 95% physical-frame bootstrap intervals, not measured real-world variability. The two panels expose different rankings for majority-level and complete-team failure.*

**Interpretation:** At rho=.75, treating correlated evidence as independent increases R_3 from 2.475% to 14.525%, with all other source predictions held fixed. The cost of a mislabeled provenance group can exceed that of changing the fusion confidence threshold.

### Figure 03 — Selective fallback is not automatically an energy-saving policy

Files: `03_selectivity_vs_sunk_energy.{pdf,svg,png}`.

**Suggested caption:** *Post-inference selectivity and incremental energy under fixed hypothetical E1 device/radio profiles. Left: accepting fewer ROI fusion decisions need not proportionately reduce frame-result downlinks, because one accepted ROI triggers one shared frame payload for every agent. Right: additional energy above local-only in mJ per physical frame; local detections, uploads and edge fusion assessment are paid before the acceptance gate, while only result downlinks can be avoided. No hardware measurements are claimed.*

**Interpretation:** F0's fixed **additional** edge+uplink cost is **22.9264 mJ/frame** above local-only. At maximum acceptance the additional downlink can add **0.6656 mJ/frame**. Even at a threshold `.999` with **zero accepted ROI decisions**, total incremental energy is **0.4229264 J/frame**, versus **0.4000 J/frame** local-only. For this particular profile, avoiding *every* result downlink can save only 0.6656 mJ/frame, about 2.8% of the maximum F0 incremental overhead above local-only. This is an architectural consequence of deciding to fall back **after** expensive evidence acquisition, not a universal law of edge inference.

### Figure 04 — Masking alone may not repair the remaining evidence

Files: `04_masking_and_miscalibration.{pdf,svg,png}`.

**Suggested caption:** *Five-agent synthetic error at rho=.75 under correct versus incorrectly specified source error probabilities and dependence groups. Over-optimistic reliability estimates for the three-agent clique negate lineage-aware F0's majority-risk advantage. Masking the clique is not universally better: in this fixture it produces the same R_3 and R_5 as correctly grouped F0, illustrating that suppression of evidence is distinct from guaranteed recovery.*

**Interpretation:** Static reported q values are **known by construction** in the correctly specified case and deliberately wrong in other cases. Neither a calibrated risk estimator nor on-device trust measurement is being claimed. This figure is most appropriate for a model-robustness appendix.

### Figure 05 — Real predictions: share coverage can be nonzero without any label correction

Files: `05_v2v4real_agreement_trap.{pdf,svg,png}`.

**Suggested caption:** *Single-sequence V2V4Real-derived prediction replay (scene 0007; 221 physical frames, 442 diagnostic ROI decisions). The conservative two-source F0 rule accepts some agreeing evidence, resulting in 56.56% fused-result usage and 500 per-agent shared-result replacements, but **zero actual changed binary labels** relative to local-only inference. Modeled incremental energy nevertheless rises from 0.160 to 0.180 J/frame. Recorded cooperative late fusion is a **different pretrained inference pipeline**, not an F0-controlled same-model intervention.*

**Interpretation:** This is a useful *negative control*, not proof that learned fusion cannot improve real-world systems. Existing sequence 0007 ROI definitions are diagnostic, not official V2V4Real detection metrics. We do not extrapolate the 221 frames to independent scenes.

## How to reproduce

```bash
python -m pip install -e '.[test,plot]'
python scripts/run_f0_cvc_parameter_study.py \
  --frames 2000 --rois 2 --bootstrap 2500 --online-check-frames 180 \
  --output samples/f0_cvc_parameter_study
python -m pytest -q tests/test_f0_cvc_parameter_study.py
```

All figure PDFs are vector output with editable SVG alternatives and 300 dpi PNG previews. CSVs and generated trace JSONL files are included. No original OCTOPUS-Sim simulator/F0 implementation files were modified for this study; only additive study files and the original v0.7.1 release are packaged.

## What should **not** be claimed yet

- No demonstrated improvement on the actual V2V4Real two-agent replay; F0's strict two-support rule changes **zero labels** there.
- No power measurement, real network trace, hardware calibration or verified absolute joule prediction.
- No guarantee that the surrogate confidence is calibrated, or that the q/lineage metadata can be estimated faithfully in deployments.
- No causal statement that the recorded pretrained cooperative-fusion model is the same as F0's weighted vote; they use different pipelines.
- No official V2V4Real mAP or end-to-end driving safety metric.
- No generalization to multiple real driving sequences from the single bundled `0007` replay.
- No experimental family-wise multiple-comparison correction, and no more than one synthetic seed in the primary figure grid. Treat parameter tuning as exploratory; preregister and validate future conclusions on independent seeds/environments.

**Suggested presentation strategy:** one combined main-text figure using panels 01(a) and 02, one energy/coverage figure from 03, and use 05 as a compact limitations/control example. Keep the full 04 sensitivity grid in supplementary material. This places the dependence-aware risk tradeoff ahead of incremental implementation detail.
