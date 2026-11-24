# Three-agent selective fusion: implementation, experiment and limitations

**Study status:** controlled synthetic predictions, not a third V2V4Real vehicle or a demonstrated real-world fusion improvement. Reproduced with the actual OCTOPUS-Sim v0.7.1 physical execution DAG and `octopus_fusion.fusion.SelectiveFuser`.

## Why three agents change the problem

With two binary agents, a disagreement means one vote per label. F0's default `min_supporting_groups=2` must reject fusion; agreements are shared without changing either agent's answer. Therefore the two-agent V2V4Real replay shows no corrected binary labels. With **three genuinely distinct** inference lineages, F0 can use two agreeing sources to correct the third.

A third **agent is not necessarily a third independent opinion**. Two agents consuming the same physical result belong to one `dependence_group`, and their agreement is not double-counted. The source code enforces consistent `result_id` → `dependence_group` mappings. A group with conflicting labels is masked entirely.

## Exact F0 pipeline (`octopus_fusion/fusion.py`)

For one binary ROI event, F0 sees, for each agent, a binary prediction `y_i ∈ {0,1}`, a physical `result_id`, a `dependence_group`, a production time and an estimated wrong-answer probability `q_i ∈ (0,1)`. It receives no ground-truth labels, error indicators or future predictions.

1. **Build dependency groups.** Reports from the same lineage count as **one** group. If two reports claiming one lineage contradict each other, mask the entire group (do not choose a convenient label).
2. **Mask unreliable/stale reports.** Default thresholds are `q_i >= 0.25` (also `q_i >= 0.5`), or information older than `120 ms`. Masking removes an input from the edge's *vote*, not from its owning agent's local decision.
3. **Conservative group evidence.** Each unmasked group contributes one binary opinion and uses its **largest** member-error estimate `q_g=max(q_i)`.
4. **Compute a weighted log-likelihood-style score:**

   \[
   S=\log\frac{\pi}{1-\pi}+\alpha\sum_{g\in\text{usable}}(2y_g-1)\log\frac{1-q_g}{q_g},\qquad C=\sigma(|S|).
   \]

   Here `π` is the configured positive-label prior, and `α ∈ [0,1]` is a dependence discount (default 1). The score's sigmoid **is not calibrated** unless the error rates, prior and conditional-independence assumptions have been established.
5. **Apply accept/reject guards.** The default checks are ≥2 usable independent groups, ≥2 groups supporting the winning label, conflict fraction ≤0.40, surrogate confidence ≥0.90; optional *external* joint-risk upper-bound guards can also be supplied. A masked group can coexist with accepted fusion (`MASK_AND_FUSE`).
6. **Return an action.** If all guards pass, `SHARE_FUSED` / `MASK_AND_FUSE` gives the derived binary vote to all agents that receive the update before deadline. Otherwise `KEEP_LOCAL` preserves each agent's **own** original binary prediction and its physical inference identity.

A low-confidence or rejected edge result does **not** undo detector execution, upload, or edge F0-decision computation. A masked agent still receives the accepted shared result if it arrives; masking affects evidence contribution, **not** delivery eligibility.

### Concrete example

| Agent | Binary report | Assumed wrong probability | Vote status |
|---|---|---:|---|
| A | 1 (hazard) | 0.06 | trusted |
| B | 1 (hazard) | 0.12 | trusted |
| C | 0 (clear) | 0.30 | masked by the 0.25 threshold |

A and B are distinct lineages. Their log-odds evidence is about `log(.94/.06)+log(.88/.12)=4.745`; its sigmoid exceeds 0.99. Two independent groups support the same label. F0 can **share hazard (1)** to all three agents, repairing C if C's local decision was wrong.

If A and B are the two wrong agents, sharing instead creates an additional third wrong answer. F0 does not know which case holds at decision time.

### A structural limitation of the conservative three-agent rule

Define `K` as the number of wrong **local** decisions for a binary ROI with three distinct sources. An accepted F0 output requires ≥2 supporters and therefore cannot overturn the **local majority**. A fallback preserves all local labels. Thus:

| Local wrong count K | If share majority | If fallback | Indicator `K >= 2` |
|---:|---:|---:|---|
| 0 | 0 wrong | 0 wrong | unchanged |
| 1 | 0 wrong | 1 wrong | unchanged |
| 2 | 3 wrong | 2 wrong | unchanged |
| 3 | 3 wrong | 3 wrong | unchanged |

Hence, in this *three-agent, binary, one-majority-answer-or-fallback* design:

\[
\boxed{R_{\ge2}^{\text{F0}}=R_{\ge2}^{\text{local}},\qquad R_{3}^{\text{F0}}\ge R_{3}^{\text{local}}.}
\]

This is an algorithmic limitation, not a numerical accident. F0 **can reduce average marginal error** and correct individual minority faults, but cannot repair a wrong majority; it can only preserve or increase all-three simultaneous failure. This result assumes timely complete delivery and distinct local binary opinions; packet loss and asynchronous updates may create different partial-delivery behaviors.

To truly reduce `R_>=2`, a more capable algorithm must sometimes reject an incorrect majority, e.g. using a verified minority source, temporally independent evidence, a new sensor/modality, or a trustworthy external cooperative candidate—not merely repeat majority-preserving voting.

## Controlled three-agent experiment

Run:

```bash
python -m pip install -e '.[test,plot]'
python scripts/run_f0_three_agent_diagnostic.py \
  --frames 2000 --rois 2 --seed 60421 --bootstrap 2500 \
  --output samples/f0_three_agent
python -m pytest -q tests/test_f0_three_agent_diagnostic.py
```

Dataset: **2,000 physical frames**, **4,000 binary ROI decisions**, and **three agents**. Each frame creates one detector execution per agent and one optional F0 decision execution; both ROI projections share physical inference artifacts. The simulator has no packet loss or missed deadlines in this experiment. All detector runtimes, radio rates and energy profiles are **hypothetical**.

The controlled source marginal wrong-answer probabilities are `q_A=0.06`, `q_B=0.12`, `q_C=0.30`. Errors are not always independent: a latent common-mode shock can make all three agents wrong. We test shock probabilities `s=0` and `s=0.04`. In the absence of a common shock, per-agent error Bernoulli probabilities are adjusted to `(q_i-s)/(1-s)` so marginal rates remain approximately constant as `s` varies. These true population rates are **known by construction**, not estimated from unseen deployment data.

Methods: local-only, always-sharing unweighted majority, always-sharing naive reliability weighting, selective F0 with three inputs, selective F0 masking C (error threshold 0.25), and F0 wrongly claiming C has `q_C=0.08` rather than its true 0.30. All fusion baselines run the same physical F0 decision graph, while local-only omits uploading and edge F0 work.

### Main results (`s=0.04`)

| Policy | Mean individual wrong | R≥2 (majority wrong) | R3 (all wrong) | Corrected local labels | Newly wrong labels | Model energy (J/frame) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Local only | 15.93% | 6.78% | 4.08% | 0 | 0 | 0.2400 |
| Always share majority | 6.78% | 6.78% | 6.78% | 1,207 | 108 | 0.2578 |
| Always share naive weighted | 6.78% | 6.78% | 6.78% | 1,207 | 108 | 0.2578 |
| F0 (all sources allowed) | 8.19% | 6.78% | 4.12% | 931 | 2 | 0.2578 |
| F0 (C masked as unreliable) | 8.19% | 6.78% | 4.12% | 931 | 2 | 0.2578 |
| F0 (C mistakenly overtrusted) | 6.43% | 6.78% | 4.55% | 1,159 | 19 | 0.2578 |

**Important negative findings:** masking C and allowing C as a low-weight contributor produce **identical final decisions in this parameter setting**. Although the audit action differs (`MASK_AND_FUSE` vs `SHARE_FUSED`), both variants accept/reject the same ROI decisions because the remaining F0 confidence and two-support guards already exclude C's potentially misleading votes. We must **not** claim that masking alone produced the gain. Here the benefit comes from **conservative selective acceptance / fallback**, not a mask-specific causal effect.

Both naive baselines make the same binary decisions in this parameter setting; this is not proof they are generally equivalent.

The overtrusted-C configuration has better individual accuracy but a higher extreme shared-failure rate: incorrect reliability estimates can encourage dangerous additional overrides.

At zero common-shock probability, local `R3` is 0.30%, selective F0 `R3` is 0.78%, and naive majority `R3` is 5.85%. With a 4% shared fault mode, local `R3` rises to 4.08%, selective F0 to 4.12% and naive majority to 6.78%. This illustrates that the underlying shared fault process can dominate the risk floor even with guarded sharing.

### Uncertainty

One seeded synthetic trace is used for all policies at each shock setting (paired potential outcomes). The accompanying `paired_frame_bootstrap.csv` contains 2,500-replicate paired percentile intervals that resample **whole frames**, preserving two ROI decisions per frame. For `s=0.04`, selective F0 masking C versus local-only:

- Marginal-error improvement: **−7.74 percentage points**, 95% within-generator bootstrap interval **[−8.18, −7.33] pp**.
- R3 increase: **+0.05 percentage points**, interval **[0.00, +0.125] pp**.
- R≥2 change: **0** by algorithmic construction, not merely lack of statistical power.

These intervals describe variability under this **particular synthetic generator**, not uncertainty across measured real scenes or different hardware. The modeled J/frame values should not be presented as real device energy measurements.

### Relationship to real V2V4Real

The bundled V2V4Real validation sequence `0007` has only **two real captured vehicles**; this study does not invent a third recording or claim measured real three-agent perception. Its independently trained cooperative-fusion candidate has richer information than F0's binary votes. The three-agent synthetic run is an engineering/proof-of-concept follow-up, not an empirical proof that F0 repairs real V2V4Real mistakes.

## Figure files and independent test cases

- `01_three_agent_error_tradeoff.pdf/png`: marginal accuracy vs `R3`, alongside the structural invariance of `R≥2`.
- `02_three_agent_corrections_energy.pdf/png`: corrected vs newly wrong agent decisions and incremental modeled energy.
- `03_common_shock_sensitivity.pdf/png`: how common error changes the all-agent risk of each policy.
- `three_agent_policy_comparison.csv`, `paired_frame_bootstrap.csv`, `PROVENANCE.json`: source numbers, uncertainty intervals, and provenance.
- `tests/test_f0_three_agent_diagnostic.py`: direct all-8-pattern mathematical invariant, no-leakage ground-truth flips, exact physical frame job counts, deterministic simulation, mask/fallback guards and reproducibility.

For the CVC paper, frame **the result as fault containment versus fault recovery**, not a claim that conservative fusion reduces every notion of collective failure. Genuine majority-fault recovery requires new evidence and a different decision mechanism.
