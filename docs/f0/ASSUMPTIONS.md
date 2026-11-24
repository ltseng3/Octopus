# F0 assumption register — explicit and testable

| ID | Assumption | V0.1 treatment | Required future check |
| --- | --- | --- | --- |
| F0-A01 | Agent wrong probabilities \(q_i\) are available before fusion | External static/synthetic estimator; no ground truth passed into fuser | Fit and calibrate on held-out data, check distribution shift |
| F0-A02 | Error rates apply to the current ROI, not just lifetime averages | Each ROI carries explicit `ReliabilityEstimate` | Verify empirical bin-wise calibration and uncertainty |
| F0-A03 | Binary label error is symmetric for positive and negative labels | Log-likelihood vote uses symmetric error rates | Calibrate separate sensitivity/specificity for asymmetric class costs |
| F0-A04 | Separate evidence groups are conditionally independent | Optional discount, **no independence guarantee** | Measure overlap, model sharing and dependence on held-out scenes |
| F0-A05 | Duplicate result consumers provide no extra independent information | One vote per declared dependence group | Validate provenance graph from original engine |
| F0-A06 | Different models or devices imply different lineage | **Not assumed**; caller supplies grouping | Audit cross-agent correlation and common weights |
| F0-A07 | Every source provides a current binary decision | Strict inputs; incomplete inputs rejected | F1 availability/deadline-aware partial decisions |
| F0-A08 | Stale/high-q evidence should be masked | Configurable thresholds | Stress-test quality proxy errors and adversarial masking |
| F0-A09 | Disagreement identifies a faulty source | **Rejected**; within-lineage contradictions mask group, cross-group conflict triggers fallback | Use independent verification to identify faults |
| F0-A10 | Confidence from discounted votes is calibrated | **Rejected**; called `score_confidence` surrogate | Calibration tests, proper scoring rules and credible intervals |
| F0-A11 | Marginal error probabilities alone bound collective failure | **Rejected**; external risk bounds are optional and separate | Joint-error model with independently validated bounds |
| F0-A12 | Sharing after local computation saves energy | **Rejected**; F0 only reasons about decision outputs | F1 conditional DAG charges all executed/rejected work |
| F0-A13 | Rejected fusion needs a new edge result broadcast | **Rejected**; each agent retains local result | Implement correct local retention / notification protocol in F1 |
| F0-A14 | Multiple ROIs require duplicated frame inference | **Rejected**; result ID can recur across ROIs with different labels | Verify v0.6/v0.7 physical DAG projection semantics in source |
| F0-A15 | A source with low estimated q is never wrong | **Rejected**; common-shock toy can make all sources wrong | Study correlated failures and failure-recovery limits |
| F0-A16 | Two disagreeing agents can recover correct truth from error rates alone | **Not assumed**; need independent corroboration | External verification or confidence calibrated independently |
| F0-A17 | Offline files make future outputs automatically visible | **Rejected**; offline replay assumes completed available outputs | F1 model upload/availability timestamps, queues and deadlines |
| F0-A18 | Toy success transfers directly to V2V4Real | **Rejected**; no real trace or measured energy bundled | Benchmark with licensed real traces and held-out scenes |
| F0-A19 | Same physical result must have same binary label across all ROIs | **Rejected**; per-ROI projection is allowed to differ | Preserve `predictions_by_event_id` mapping and event IDs |
| F0-A20 | Comparing two upper risk bounds proves shared risk is lower | **Rejected**; relative guard needs upper(shared) vs **lower(local)** | Derive or calibrate bounds from held-out data |
| F0-A21 | External plugins are isolated from arbitrary Python code | **Not claimed**; immutable DTOs reduce accidental leakage | Sandbox separately for untrusted plugins |
| F0-A22 | F0 reproduces OCTOPUS-Sim V0.7 source | **Not claimed**; earlier release archive unavailable here | Merge/test against real v0.7 source before declaring integration |
| F0-A23 | Consecutive ROI outcomes are independent samples | **Rejected** in threshold sweep; frame-cluster bootstrap used | Scene-cluster bootstrap for real-world studies |

**Critical deployment distinction:** `KEEP_LOCAL` is a *decision policy* in F0. A real system must ensure local results are retained on devices and the edge's rejection instruction arrives before the deadline; those actions have bandwidth, time and energy costs. F0 makes no safety or energy guarantee.
