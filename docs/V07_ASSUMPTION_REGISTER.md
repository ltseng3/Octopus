# V0.7 audit: discovered, corrected and outstanding assumptions

This is a **research methods artifact**, not a claim of production correctness.

| # | Assumption identified in the V0.6 → V0.7 audit | Action in V0.7 | Residual risk / next phase |
|---|---|---|---|
| A01 | Recorded `edge:ego` may mean cooperative fusion, not generic edge inference | Mandatory trace candidate allowlist; reject fusion-key aliases for non-fusion candidate DAGs | Trace authors must still verify provenance of every allowed candidate; no automatic model-identity oracle |
| A02 | Every detector completion in simulator memory is visible to central edge | Expose **only completed edge-resident result metadata**, with bounded history | Explicit agent→edge result notification, telemetry latency and costs deferred |
| A03 | Central edge always knows agents' local queue depths | Default `edge_only` hides local queue commitments; `reported_all` is explicit opt-in | Reported queue freshness and signaling overhead not modeled |
| A04 | Current queue reservations describe all work soon to execute | Predictor labels and exposes its omission of future stages in already-in-flight DAGs | Need in-flight DAG forecasting for stronger P1 deadline guarantees |
| A05 | Independent predictions are always physically available in a recorded trace | Candidate inventory must be explicit and source-model consistent | Dataset provenance may require additional offline checks |
| A06 | Each ROI may be a separate physical inference | V0.6 frame-batch contract retained; V0.7 candidates evaluated **per frame** | Cross-frame cache reuse/staleness not yet modeled |
| A07 | Sharing the same binary output means every agent has the same error | Synthetic risk estimator couples consumers **within one decision event**; separate ROIs retain their individual labels | Other fusion/verification policies may have non-identical downstream decisions |
| A08 | Context features automatically predict hidden errors | Synthetic public proxies use a distinct RNG namespace; never built from true error draws | Estimator misspecification and calibration are future research questions |
| A09 | Per-model compute time predicts total latency exactly | Predict queued compute/transfer service on copied commitments; validate no-load exactness | Overlapping workflows and unknown local queues cause optimistic errors |
| A10 | Predicted energy is measured hardware energy | Preserve labeled hypothetical profiles, predict incremental successful-path joules only | Hardware calibration deferred; do not report measured savings |
| A11 | Packet loss affects only decision accuracy, not physical energy | Actual executor charges attempted transmissions; predictor states successful-workflow path and separate delivery probability | Need expected-work-under-failures model if P1 optimizes lossy scenarios |
| A12 | All-agent error is sufficient to capture systemic risk | Estimator interface supports any 1 ≤ m ≤ n with a valid common-shock joint law | Contextual and empirical risk estimators deferred |
| A13 | Full reuse always uses the first agent's inference | Enumerate `shared_edge:<agent>` for every possible source | P1 must implement fair selection; source-specific quality assumptions may be necessary |
| A14 | Edge budgets must combine energy, latency and risk into a single scalar | Expose separate cost/risk estimates; do not implement scalarized optimizer in V0.7 | P1 needs explicit constraints, infeasible outcomes, operational-risk metric |
| A15 | A provider could estimate candidates by actually invoking the runtime | Candidate assessment compiles validated DAGs and performs pure virtual calculations; only selected DAG runs | Third-party Python plugins remain trusted, not sandboxed |
| A16 | Every result score is available to the policy at frame release | `CoordinatorObservation` contains metadata about already-completed edge results, never the ground truth, future candidate outputs, or realized errors | Future learned estimators require explicit labeled-feedback timing |
| A17 | Frame identity can be inferred from identical timestamps | Require explicit trace frame IDs and consistent arrival slots/features for frame execution | Trace authors remain responsible for globally unique scene/frame identifiers |
| A18 | All agent count results imply genuine independence | Common-shock estimator defines a valid distribution over **result groups** rather than assigning unconstrained pairwise correlations | Higher fidelity physical error dependencies remain unmodeled |
| A19 | New trace feature validation cannot affect legacy datasets | When coordination is disabled, extra `public_features` are ignored just like old unknown trace fields | Future schema changes need opt-in semantics and regression fixtures |
| A20 | Legacy policies can safely bypass the new inventory | Coordination mode requires **all** policies, including legacy policies, to match a validated candidate allocation | Recorded traces still need author-provided evidence for candidate semantics |
| A21 | Python plugin DTOs are a security isolation boundary | Immutable typed records and no runtime/World references reduce *accidental* leakage | Plugins are trusted Python, **not** sandboxed untrusted user code |

## Provenance categories for research outputs

* **Observed prediction outcomes:** source-verified replayed labels (V2V4Real sequence 0007 only in the resource replay) or simulated synthetic error draws. Never mix them under one 'empirical' label.
* **Estimated joint risk:** configured synthetic estimator; no fitting or calibration is claimed.
* **Predicted resource costs:** L0 virtual DAG forecast over observable reservations. Not exact with future work from overlapping DAGs.
* **Executed resource costs:** simulated FIFO execution with configured compute, network and hypothetical energy profiles.
* **Hardware-measured cost:** **not provided** by this release.

## Release gates

V0.7 must retain all V0.6 regression tests; newly enabled observation features
must not alter legacy outcomes; only the selected candidate runs; frame-level
physical work is not duplicated; estimators cannot mutate queues; unknown
trace inference provenance is rejected; device/local state is not silently
exposed; and the compiled package must pass validation from a clean extraction.
