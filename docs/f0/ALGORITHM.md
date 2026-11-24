# F0 algorithm specification

## Inputs and information flow

For one **ROI decision event** inside a physical frame, every agent offers a binary prediction \(\hat y_i\in\{0,1\}\), an estimated conditional error probability \(q_i\), a physical `result_id`, a declared **dependence group**, a production timestamp and optional public quality proxy. The true label \(Y\) is **not** in `FusionObservation`, `ReliabilityEstimate`, or `FusionConfig`. It is in `LabeledCase` for offline evaluation only.

`result_id` identifies the **physical frame-level** inference artifact. The same ID can serve different ROIs, **with different binary predictions**, but must give one consistent prediction within any one ROI. `dependence_group` identifies a conservative cluster of potentially correlated evidence. Different result IDs can share a group. If you cannot justify independent lineages, group conservatively. Merely having different vehicles does not prove independence.

## Reliability-weighted vote

Under calibrated symmetric binary-error rates, conditional independence **between retained result groups**, and a valid prior \(\pi=\Pr(Y=1)\), the posterior log odds is

\[
S=\log\frac{\pi}{1-\pi}+\sum_{g} (2\hat y_g-1)\log\frac{1-q_g}{q_g}.
\]

F0 groups contributions before accumulating weights. If one group contains several consistent reports, it uses the **largest** estimated error among the eligible reports, rather than multiplying confidence from duplicate consumers. Contradictory reports within the same dependence group cause the whole group to be masked.

A user-controlled `dependence_discount` \(d\in[0,1]\) multiplies the summed group evidence. When \(d\ne1\) or the independence/calibration assumptions do not hold, \(\sigma(S)\) is merely a **ranking/confidence surrogate**, not a calibrated posterior probability. F0 never labels \(1-\sigma(|S|)\) as a verified joint-risk bound.

Numerical implementation uses `log1p(-q)-log(q)` to avoid overflow when \(q\) is extremely small, and a sign-stable sigmoid to avoid `exp` overflow.

## Mask and accept/fallback rules

For each ROI:

1. Validate unique agent IDs, physical-result consistency, timestamps and probability inputs.
2. Mask every declared dependence group whose evidence contradicts itself, before examining any contributor's error rate; this avoids cherry-picking one group member.
3. Mask a source with estimated \(q_i\ge q_{\max}\) (default 0.25), or staleness greater than 120 ms.
4. For each retained dependence group, calculate one conservative reliability-weighted vote.
5. Require at least two distinct groups, at least two groups supporting the winning class, sufficient surrogate confidence (default 0.90), and conflict fraction no greater than 0.40.
6. If an **external risk-bound guard** is configured, require those bounds to be available and satisfied. A comparison of the form \(R_{\mathrm{shared}}\le R_{\mathrm{local}}+\delta\) is only certified by \(\mathrm{UB}(R_{\mathrm{shared}})\le\mathrm{LB}(R_{\mathrm{local}})+\delta\), **not** by comparing upper bounds to upper bounds.
7. If all tests pass, emit `SHARE_FUSED` or `MASK_AND_FUSE`. Otherwise emit `KEEP_LOCAL`, preserving each agent's original decision exactly.

The per-agent source trust is an **estimated propensity**, not an observed correctness flag; masking is not proof that the sensor was faulty. With only two disagreeing sources and no independent corroboration, the default policy keeps local opinions.

## Why dependence groups matter

Suppose agents A and B both forward the **same inference result**, and C supplies an independent contradictory result. A naive majority treats the duplicated A/B answer as two votes and overrides C. F0 counts the shared A/B result once; with no independent support for either label, it falls back. Even different results may be highly correlated if they use shared model weights or overlap sensor coverage; dependency groups are conservative metadata, not inferred by the algorithm.

## Why selective fallback is different from voting

Broadcast fusion makes all agents wrong together whenever the shared answer is wrong. `KEEP_LOCAL` preserves the original collection of decisions, so one mistaken agent need not produce an all-agent failure. It does not guarantee correct majority behavior, and a common-shock situation can still make every local result wrong.

F0 does not model the edge's physical computation or delivery time; its decision is an **offline post-inference step**. F1 must represent when the estimates and frame-level inference outputs become visible, conditional transfers, and the energy/latency already spent on rejected fusion attempts.

## Baselines / metrics

- `local_only`: no shared decision; each agent retains original label.
- `majority`: naive per-agent majority; ties preserve local decisions.
- `weighted_naive`: unmasked per-agent reliability-weighted vote; ties preserve local decisions; **deliberately counts duplicate results multiple times**.
- `selective_f0`: provenance-group-aware masking, weighted vote and selective fallback.

Report average marginal agent error, all-agent simultaneous-error frequency, shared-fusion coverage, error rate **conditional on accepted fusion**, and fallback/mask rates. Error conditional on acceptance can be misleading if reported without coverage. Do not describe a low error among accepted samples as a whole-population risk improvement.

## Explicit non-goals

No neural-network fusion, box-level geometric fusion/NMS, per-object association, real probability calibration, co-simulation, conditional resource scheduler integration, online risk learning, cross-frame caching, cache invalidation, or stochastic safety guarantees. These are potential F1/F2 research tasks.
