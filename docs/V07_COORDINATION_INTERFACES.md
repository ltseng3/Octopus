# V0.7: Policy-visible coordination interfaces

**Status:** V0.7 research interfaces, not an optimized scheduling policy or hardware-calibrated predictor.

## Goals and boundaries

OCTOPUS-Sim separates (1) *what the edge knows*, (2) *what it estimates*,
(3) *which complete inference workflow it selects*, and (4) *what actually
happens*. An estimator computes predictions over **candidate execution DAGs**;
only the chosen DAG is submitted to the scheduler. The hidden `WorldEvent.truth`,
true error draws, future loss draws, and inference outcomes from unexecuted
candidates are never arguments to coordinator plugins.

```
World (private truth, potential predictions)  --->  execution + scoring
                                                   ^
explicit public features --+                         |
committed queue snapshots -+--> CoordinatorObservation --> candidate provider
edge-resident result log --+                         |
                                                    v
                             immutable CandidatePlan + compiled DAG
                                        /          \
                           RiskEstimator          ResourceCostEstimator
                                        \          /
                              candidate assessments
                                       |
                           selector -> one candidate ID
```

The existing `BasicPolicy` path is retained. Turn on `coordination.enabled` to
produce candidate assessments while still using existing policies, or use
`candidate_fixed` to demonstrate the new selection contract. **When enabled,
even a legacy BasicPolicy must select a physical allocation present in the
validated candidate inventory; a legacy policy cannot bypass trace provenance
restrictions.** We deliberately
**do not** add an energy-optimizing or learned policy in V0.7 (planned for P1).

## Installation and first examples

From the repository root, Python 3.10+:

```bash
python -m pip install -e '.[test]'
python -m octopus_sim validate examples/coordination/three_agent_contextual_local.json
python -m octopus_sim run examples/coordination/three_agent_contextual_local.json \
    --no-events --output outputs/v07_local
python -m octopus_sim run examples/coordination/three_agent_contextual_fusion.json \
    --no-events --output outputs/v07_fusion
python -m octopus_sim run examples/coordination/v2v4real_candidate_inventory.json \
    --no-events --output outputs/v07_v2v4real
python scripts/run_v07_coordination.py
python -m pytest -q tests/test_v07_coordination.py
```

The new `candidates.jsonl` contains one record **per candidate per physical
frame**, not per ROI. Each record includes a selected flag, estimated joint
risk, hypothetical resource costs, and explicitly visible features. Existing
`plans.jsonl`, `inferences.jsonl`, `scores.jsonl`, and energy reports still
capture the actual selected plan only.

For V2V4Real sequence `0007`, the replay contains 221 physical frames and 442
scored decisions. Recorded vision outcomes come from the DMSTrack-derived trace;
**risk estimates and power/latency costs are configured assumptions**.

## Public context (`CoordinatorObservation`)

All fields are immutable dataclasses or tuples of immutable objects; there is
no pointer to the simulation `World`, evaluation data, energy ledger or live
`FIFOResource` objects.

| Field | What it means | Availability assumption |
|---|---|---|
| `batch_id`, `frame_id` | Physical execution identity | Known at decision release |
| `agent_ids` | Participating agents | Known |
| `time_us`, `deadline_us` | Absolute simulation clock and decision deadline | Known |
| `scene_features` | **Allowlisted** ex-ante numeric proxies only | Trace author certifies ex-ante availability |
| `available_models` | Model IDs, devices, durations, output sizes | Public configured profiles |
| `resource_snapshot.edge` | Reservations already committed to edge worker queue | Coordinator owns edge scheduler |
| `resource_snapshot.network` | Reservations to simplified shared link | Coordinator controls/knows modeled link queue |
| `resource_snapshot.local` | `None` by default; local queues if `reported_all` | Fresh local telemetry assumed if enabled, no telemetry transmission overhead modeled |
| `resource_snapshot.in_flight_workflows` | Previously released DAGs still running | Known from coordinator allocations |
| `completed_inferences` | **Metadata only** for up to 32 completed edge-resident results | Available only after compute at edge completes |
| `policy_draw` | Independent control variate for randomized baseline | Does not encode ground truth |

Agent-local model executions are **not** automatically inserted into the edge's
completed-inference catalog. Uploading their results/telemetry would require
an explicit communication model; current V0.7 catalog omits them.

### Scene features and truth separation

In a synthetic scenario:

```json
"coordination": {
  "enabled": true,
  "resource_visibility": "edge_only",
  "visible_features": ["visibility_proxy"],
  "synthetic_features": {
    "visibility_proxy": {"kind": "uniform", "low": 0.0, "high": 1.0}
  },
  "risk_estimator": {"kind": "contextual", "context_feature": "visibility_proxy",
                     "context_slope": 0.07}
}
```

These features are generated by a **distinct deterministic random namespace**.
They are deliberately **not** derived from the synthetic world's true error
or latent-factor draws. Therefore, contextual risk estimates are not inherently
predictive of actual errors; assessing estimator validity is a later study.

For a recorded trace, optional input `public_features` is a numeric map on
**each ROI record**. Only names listed in `coordination.visible_features` reach
the policy. All ROIs belonging to one physical frame must carry identical
allowlisted values. Missing, nonnumeric or nonfinite values are rejected.
An input trace may contain other private fields, but they are not copied into
the coordinator context. The dataset author must still ensure that an
allowlisted feature is genuinely available **before** the decision: renaming a
ground-truth variable to `visibility_proxy` is still information leakage.

## Candidate execution plans

`StandardCandidateProvider.generate(observation)` returns a bounded catalog:

- `local_only`: distinct local results, one per agent.
- `independent_edge`: one independent edge result per agent (distinct input source).
- `shared_edge:<agent>`: one edge result sourced from that agent, delivered to the team; all sources are enumerated to avoid silently favoring the first agent.
- `partial_share`: one shared group and one singleton (only if at least three agents). **Not an exhaustive partition search.**
- `cooperative_fusion`: all participating agents detect locally, upload their results, one edge fusion executes, and its output is distributed.

Each candidate compiles through the **same** `compile_execution_graph` and
`ExecutionGraph.validate` used for physical execution. Proposals are pure
construction; they do not reserve workers, execute neural models, generate
prediction outcomes, or charge joules. `StandardCandidateProvider` can be
replaced by another object implementing `CandidateProvider`.

**Trace provenance gate:** when `coordination.enabled` is true and the backend
is a recorded trace, `coordination.candidate_ids` is **mandatory**. No code
infers that a recorded tensor is an independent edge prediction merely because
its key begins with `edge:`. In the bundled V2V4Real trace, `edge:ego` is also
the recorded cooperative-fusion output. V0.7 explicitly rejects non-fusion
candidates that would consume that aliased key under a generic model cost.
The bundled provenance-valid inventory contains only `local_only` and
`cooperative_fusion`. Other distinct edge results require evidence of their
source/model semantics in the trace.

## Replaceable synthetic risk estimator

`CollectiveRiskEstimator.estimate(observation, candidate, *, failure_threshold)`
returns `RiskEstimate`: marginal agent error rates, all-agent error probability,
and probability of at least `m` agent errors. Three modes are provided:

- `constant`: fixed configured local, edge and fusion marginal error rates.
- `contextual`: linearly adjusts the **assumed** error rates with an allowlisted
  numerical observable; rates are clamped to [0,1].
- `misspecified`: applies a configurable additive bias to assumptions, including
  deliberate severe misspecification for robustness tests.

The default synthetic risk assumptions use local=0.10, edge=0.06 and
fusion=0.06, matching the original default synthetic marginal rates only by
initial numerical choice. They are **not dynamically linked** to `world.*`
parameters. Changing true-world rates does not silently change the edge's
risk assumptions. The `contextual` and `misspecified` variants can intentionally
produce incorrect risk estimates.

The synthetic estimator treats each distinct inference result as one Bernoulli
error shared by its consumers. For `k` result groups with respective error
probabilities `p_g`, it uses a **valid global common-shock mixture** (not
arbitrary pairwise correlations):

```
J_all = (1 - rho) * product_g(p_g) + rho * min_g(p_g)
```

For at least `m` errors, the independent branch uses a count-distribution DP
weighted by group size; the common-shock branch integrates a shared uniform
latent threshold. `rho` is an **estimator assumption**, not an oracle draw from
the hidden world. A real learned estimator can implement the same protocol later.

## Resource cost prediction (`ResourceCostEstimator`)

`AnalyticalResourceCostEstimator` copies the resource and energy profiles at
initialization. It runs a virtual deterministic DAG schedule over **copies** of
currently reserved queue-availability times. It reports:

- per-device and per-model aggregate busy microseconds;
- exact configured payload bytes by transfer role;
- mean and worst ideal delivery latency (ms), nominal deadline feasibility and slack;
- estimated incremental compute/radio joules by device, if energy is enabled;
- probability that all required network transfers succeed under an **independent-loss assumption**;
- machine-readable assumptions.

The predictor **never** invokes `World.predict`, consumes random error draws,
updates the energy ledger, or reserves actual resources. Computation is in
microseconds; J = W × seconds; bytes are not bits. Baseline idle energy is not
part of the predicted incremental workflow joules.

### Important limit: currently committed reservations ≠ full future workload

A running DAG can have a local detection still in progress whose *future*
upload/fusion stages have not yet reserved the network/edge worker. A snapshot
cannot know those future reservations without replaying all in-flight DAGs.
The V0.7 cost estimator **only forecasts currently committed work**; its
latency is optimistic under sustained overlapping workloads. When local queue
visibility is `edge_only`, unknown agent queues are additionally treated as
available at the decision time. We expose `in_flight_workflows` and the
assumption flags so these approximations are never hidden.

The current cost predictor assumes successful transmissions for latency; when
packet losses occur the real DAG can stop early and use less energy. Estimated
incremental energy therefore describes the **complete successful path**, not
an unconditional expected cost in a lossy network. This distinction is
explicit in the result fields and documentation.

For deterministic no-loss, non-overlapping examples, predicted model energy,
bytes, and mean/worst latency match the executor. A queue-stress example
intentionally demonstrates latency mismatch under overlapping workflows.

## Policy API, extensibility and co-simulation

```python
class CandidateProvider(Protocol):
    def generate(self, observation: CoordinatorObservation) -> tuple[CandidatePlan, ...]: ...

class CollectiveRiskEstimator(Protocol):
    def estimate(self, observation, candidate, *, failure_threshold: int) -> RiskEstimate: ...

class ResourceCostEstimator(Protocol):
    def estimate(self, observation, candidate, graph) -> ResourceEstimate: ...

class CoordinationSelector(Protocol):
    def select(self, observation, candidates: tuple[CandidateAssessment, ...]) -> str: ...
```

Inject providers through `Simulator(config, candidate_provider=...,
risk_estimator=..., resource_estimator=..., coordinator_selector=...)` or
`run_simulation(config, **coordination_plugins)`; the observation and
assessments are immutable. The selector returns an **existing candidate ID**;
unknown IDs are rejected rather than silently executing arbitrary work.

In V0.7, `candidate_fixed` is an intentional non-optimizing baseline for
contract validation. P1 will implement budget-constrained selectors, not risk
estimation inside the policy object.

A future measured-trace predictor can implement `ResourceCostEstimator` while
the lightweight default stays usable. A co-simulator may also implement a
**separate resource-execution adapter**, since a detailed external simulator
can affect actual reservations, delays, losses and elapsed simulated time; it
must not be shoehorned into a simple cost-lookup API.

## Validation and known limitations

Contract tests verify immutable observations, absence of private outcomes,
frame-consistent features, old-versus-new execution invariance, 2/3/5-agent
semantics, exact ideal cost on deterministic fixtures, queue-visibility
assumptions, prior-edge-result availability, strict recorded candidate
eligibility, downstream failure handling and legacy compatibility.

Limitations intentionally deferred: empirical estimator fitting, adaptive
sequential verification after results arrive, calibration/uncertainty in
hardware profiles, exact in-flight DAG cost forecasting, real radio telemetry
costs, co-simulation, cross-frame staleness and cache eviction, and a
budgeted/learned allocation optimizer. Policy plugins are **trusted Python
code**, not a security sandbox; the interface prevents accidental oracle
access but cannot stop a malicious plugin importing global modules or
reading raw input files.
