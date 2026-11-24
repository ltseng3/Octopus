# Methods — three-agent F0 reliability, calibration, cost and deadline tradeoffs

## Research questions

1. With three binary agents, when can a minority expert correct a correlated
   mistaken majority, and what happens to the all-agent-failure probability?
2. How does the answer depend on the **difference between actual and assumed**
   expert error probability?
3. Does selecting fewer fused ROI results substantially reduce energy once
   local inference and uploads have already happened?
4. What happens when the edge fusion verdict arrives *after* the decision
   deadline but the local predictions are already available?

## Evaluation dataset — entirely synthetic

For `N=2,000` physically distinct frames and `K=2` binary ROI decisions/frame,
we generate balanced independent truth labels `Y∈{0,1}`. Agent A has an
independent Bernoulli wrong-answer process with probability `q_A=0.015`.
B and C have nominal error rates `q_B=0.26`, `q_C=0.32`, with a common
binary environmental shock of probability `s`:

\[
E_A \sim \mathrm{Bernoulli}(q_A),\qquad
Z \sim \mathrm{Bernoulli}(s),\qquad
E_i=Z\lor U_i, \quad
U_i \sim \mathrm{Bernoulli}\left(\frac{q_i-s}{1-s}\right),\;i\in\{B,C\}.
\]

The shock levels are `s={0,0.06,0.12,0.18,0.24}`. This construction holds
B/C's *marginal* wrong-answer probabilities fixed while changing their
cross-agent dependence. A is independent of the shared shock. All scenarios
reuse a seeded bank of latent uniforms so comparisons are paired. Each agent
has a different **physical inference result ID**; B/C dependence is a shared
*environmental fault*, not inferred reuse of the same physical result.

True population source rates are **known by construction**. The F0 trust input
is an assumed error-rate table, never per-decision correctness. This is an
idealized calibration baseline and should not be described as a learned or
validated real-world risk estimator. The model is binary and symmetric; it
omits real object geometry, confidence distributions and temporally correlated
sensor errors. Both ROIs in one frame share compute and network work but have
independently simulated truth/error variables. Frames are sampled independently.

## Tested policy definitions

- **Local:** leave all three local predictions unchanged and do not perform F0.
- **Majority broadcast:** always issue the per-agent majority binary label.
- **Strict F0 (unchanged):** `SelectiveFuser(FusionConfig(
  min_independent_groups=2, min_supporting_groups=2,
  min_score_confidence=0.90, max_conflict_fraction=0.4,
  max_source_error=0.49, dependence_discount=1))`.
- **Experimental one-expert override:** call strict F0 first; if and only if
  its *sole* rejection reason is `insufficient_independent_support_for_winner`,
  the winner has one distinct supporting dependence group, that group belongs
  to predeclared expert A with assumed `q_A ≤ 0.05`, and no input is masked
  or otherwise violates the base guards, allow sharing the weighted winner.
  This is a diagnostic modification in the study script, **not a change to
  core F0's default decision logic**. Under the selected q_A=0.015 example,
  it effectively broadcasts A's opinion on every ROI event.

The algorithm receives only recorded binary candidate outputs, provenance,
production timestamps and the configured assumed q table. `truth` is used
only after `decide()` returns to compute evaluation metrics. Tests replace
all truth labels while preserving the source predictions and verify unchanged
fusion actions and per-agent output predictions.

The weighted score is log-odds-like; its sigmoid is **not a calibrated
posterior** under B/C shared environmental shocks. Neither F0 nor the
experimental expert extension has a valid collective-risk guarantee.

## Risks and paired statistical analysis

For a common ROI event, define:

\[
R_2 = \Pr\left(\sum_{i=A,B,C} {\bf 1}[\hat y_i\ne Y]\ge2\right), \quad
R_3 = \Pr(\hat y_A\ne Y,\hat y_B\ne Y,\hat y_C\ne Y).
\]

Marginal error is the mean fraction of agents wrong per ROI decision. Share
coverage is the fraction of ROI decisions on which the fuser accepts a shared
label; it is not the fraction of distinct physical frames requiring downlink.

With binary labels, strict F0's two-support gate can only accept an existing
majority; thus `R2` is invariant relative to local-only when everyone receives
the result on time. A wrong local majority remains wrong; sharing it can
increase `R3`. The experimental one-expert override can break that invariant
only by allowing an agent with a minority position to override the other two,
which exposes **all agents** to that single source's mistakes.

The paired bootstrap resamples complete frames (both ROI decisions together),
2,000 replicates, and reports pointwise percentile intervals for **differences
versus local** at `s=0.18`. A separate six-seed audit checks the sign of the
result using independent synthetic draws. Neither supports distribution shift
or real-world deployment confidence intervals.

## Figure 1: shock sensitivity and conflicting risk objectives

Main paired realization at s=0.18, 4,000 ROI decisions:

| Policy | R2 | R3 | Marginal error | ROI acceptance |
|:--|--:|--:|--:|--:|
| Local | 19.800% | 0.200% | 20.025% | 0% |
| Majority | 19.800% | 19.800% | 19.800% | 100% |
| Strict F0 | 19.800% | 0.400% | 13.658% | 79.625% |
| Expert override (experimental) | 1.375% | 1.375% | 1.375% | 100% |

The expert override reduces R2 by **18.425 percentage points** relative to
local, paired 95% frame-bootstrap interval **[−19.651, −17.150] pp**, while
increasing R3 by **+1.175 pp**, interval **[+0.875, +1.525] pp**. These are
simultaneous objectives in tension, not a proof of universal superiority.
Six independent synthetic seeds produce expert-over-local R2 changes in
**[−20.25, −17.45] pp** and R3 changes in **[+1.00, +1.425] pp**.

## Figure 2: calibration failure map

Hold B/C's source errors and 18% common shock fixed, while changing A's **true**
wrong-answer probability over `0.015…0.45` using the same latent uniforms.
Independently vary A's **assumed** wrong probability from `0.003…0.10`.
The two heatmaps show the experimental override's change in R2 and R3 vs
local, in absolute percentage points. Positive values (red) mean **greater
failure risk**; negative values (blue) mean less.

For instance, at assumed `q_A=0.015`, true `q_A=0.015` yields a **−18.425 pp**
R2 reduction and **+1.175 pp** R3 increase. If true q_A has drifted to
`0.45` without updating the assumed 0.015, R2 instead increases **+16.75 pp**
and R3 increases **+36.80 pp**. A sharp transition across assumed q_A near
0.02 is a **deterministic confidence-threshold artifact** of this fixed binary
fusion model; do not claim that q_A=0.02 is a universal phase boundary.

## Figure 3: ROI acceptance vs physical downlinks and energy

For 600 physical frames, vary ROI cardinality K={1,2,4} and strict F0's
confidence threshold in `{.80,.87,.90,.92,.94,.975,.985,.995,.999}`.
Use a common trace bank; K=1/2 are nested prefixes of K=4 for paired comparison.

- **Acceptance:** fraction of individual ROI decisions accepted by F0.
- **Frame downlink probability:** fraction of **frames with ≥1 accepted ROI**, for
  which the integrated executor sends one F0 result per agent (independent of
  the number of ROI questions that result answers).
- **Energy:** incremental compute and TX/RX energy from the **actual simulated
  DAG**. Local detector, three uploads and one edge F0 stage are charged for
  every frame; downlink is skipped only if no ROI in that frame accepts fusion.

At zero acceptance, F0 still uses **0.25736 J/frame**, versus **0.24000
J/frame** for local-only (a 7.2% cost premium for no changed predictions).
Across the entire sweep, conditional downlinks add at most approximately
**0.40 mJ/frame** to this already-sunk F0 workload—around 0.16% of F0's
incremental modeled energy. Modeling zero per-ROI projection cost makes
energy per scored ROI fall about fourfold as K increases from 1 to 4;
this is a **modeling assumption**, not a claim of physical free perception.

## Figure 4: deadline cliff — accuracy benefit disappears, energy remains

Use 600 frames, 2 ROI decisions/frame and fixed F0 threshold 0.90. Sweep
absolute decision deadlines of 21 to 45 ms after frame release, with fixed
service-time profiles and no packet loss. Local predictions arrive at 20 ms;
F0's downlink arrives at about 31.5 ms in the uncontended configuration.

At 21 ms the local decision is timely but F0's accepted update arrives too
late: all three strategies have **20.31% marginal error** on the paired trace.
Yet F0 still consumes **0.25774–0.25776 J/frame**, vs **0.24000 J/frame** local.
At 32 ms, F0 updates are timely, reducing strict F0 error to **14.11%**, and
experimental expert-override error to **1.33%** in this toy.

The sharp jump comes from deterministic fixed model and network durations;
it is **not** a hardware latency ceiling. A real workload with jitter and
contention would broaden the transition. The executor charges energy for
uploads, edge work, and attempted downlinks even when updates are late.

## Reproducing every result

```bash
python -m pip install -e '.[plot,test]'
python scripts/run_three_agent_tradeoff_frontier.py \
  --frames 2000 --rois 2 --seed 77109 --online-frames 600 \
  --bootstrap 2000 --output samples/f0_three_agent_extended
python -m pytest -q tests/test_f0_three_agent_extended.py
```

The script saves all CSV data, four PDF/PNG/SVG figures, synthetic JSONL traces
and a `PROVENANCE.json` manifest. The online checker compares each selected
policy's final output predictions with its offline algorithm for every ROI,
checks 3 physical local jobs/frame and one edge F0 job/frame (if enabled), and
asserts zero missing/deadline-missed decisions in the idealized work sweep.

## Nonclaims / limitations

- **Not measured three-agent V2V4Real**; the public two-vehicle sequence 0007
  cannot supply a genuine third camera. Synthetic source error probabilities
  are not fitted to held-out data.
- Experimental expert override is deliberately unsafe under calibration drift.
  Its favorable risk reduction is a consequence of the chosen very strong A;
  no claimed robustness under unknown or unmodeled dependencies.
- Predicted F0 confidence is a heuristic under correlated errors, not a
  calibrated posterior or certified bound on any collective risk.
- Actual executor timing/energy come from **hypothetical model profiles**.
  No hardware measurement, wireless radio calibration or physical collision
  risk is established. Potential energy after deadline remains charged.
- ROIs are synthetic binary questions; we assume they can reuse one physical
  detection output at zero extra ROI-computation cost.
- Full source code and benchmark numbers are reproducible, but hosted GitHub
  CI and third-party asset redistribution checks remain external steps.
