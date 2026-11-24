# F0 integrated into OCTOPUS-Sim v0.7.1

**Research implementation, not deployment validation.** This release integrates
OCTOPUS-Fusion F0 v0.1.0 into the **actual uploaded V0.7 execution engine**.
The complete inference DAG, shared FIFO queues, energy ledger, deadline checks,
result provenance, and ROI-level scoring are exercised; no trained neural
fusion model is executed.

## 1. What changed in the runtime

Choose `policy.name="selective_fusion_f0"`. The graph compiler generates the
following **one physical execution workflow per recorded frame**:

```
agent A: local detector ----> local result available immediately -----+ (fallback retained)
                         \                                            |
                          detection upload ----+                      |
agent B: local detector ----> local result    | edge                 |
                         \                     +--> F0 decision compute
                          detection upload ----+        |
                                                       ROI0: SHARE_FUSED
                                                       ROI1: KEEP_LOCAL
                                                       ROI2: MASK_AND_FUSE
                                                        |
                                         If any ROI accepted: one frame result
                                         transfer to each receiving agent
                                                        |
                                         on-time accepted ROI only: override
                                         previously available local result
```

1. All detectors run **once per agent per physical frame**, not per ROI.
2. All local results are immediately available to their respective owners.
3. Detection uploads run once per agent per frame and incur communication
   energy even when F0 later rejects fusion.
4. Only after **every required upload arrives** does one CPU-modeled edge
   `f0_decider` compute stage run. Its configurable time/power is a cost proxy
   for a cheap ROI-level binary-vote implementation.
5. The F0 engine uses each ROI's previously produced local binary predictions.
   It sees explicit estimated agent error rates, ages, result IDs and declared
   dependence groups, **never ground truth** or unexecuted edge candidates.
6. One result artifact can carry accepted projections for *several* ROIs. If
   any ROI accepts fusion, one result transfer **per receiving agent per frame**
   is charged. If every ROI falls back, **no downlink is scheduled**.
7. An accepted ROI replaces an agent's local decision **only if that agent's
   downlink actually arrives on time**. Failed or late downlinks leave local
   results intact. Rejected ROIs always keep their own local predictions.
8. Late work still consumes CPU, radio and energy ledger entries; a fallback
   does not refund uploads or edge F0 computation.

This is **local-first optional correction** semantics. Agents may act on
local results before receiving an optional edge update. A system requiring one
strictly finalized decision must model a separate commit/notification protocol
and its communication/deadline costs. This release does not silently assume
instantaneous edge-to-vehicle fallback commands.

### Distinct inference identities

The result ID `frame:<frame_id>:result:local:<agent>` is the physical local
inference artifact. The edge's derived artifact has a distinct
`frame:<frame_id>:result:f0_decide` ID, with all participating local IDs in
`parent_result_ids`. A given frame's accepted ROI predictions share that
physical artifact but may have different binary values. A rejected ROI retains
**different local result IDs for different agents**.

`inferences.jsonl` includes the edge F0 record even when all ROIs reject the
fused candidate: computation has still occurred. Only accepted projections are
stored in its `predictions_by_event_id` map. `fusion_decisions.jsonl` records
**every** ROI action, mask/fallback reasons, input artifact IDs and the agents
to whom a fused update was actually delivered. `scores.jsonl` evaluates final
per-agent results, their result IDs and end-to-end latency, including packet
loss and deadlines.

## 2. API contracts and estimator selection

The original F0 module was included unchanged as the importable
`octopus_fusion` Python package, including its original tests. Under
`selective_fusion_f0`, `Simulator` creates a `FusionObservation` *after* upload
and F0 edge processing. It passes the observation to a replaceable F0
`ErrorProbabilityEstimator` and the original `SelectiveFuser`.

```python
from octopus_sim.config import ExperimentConfig
from octopus_sim.simulator import run_simulation
from octopus_fusion.estimators import QualityProxyEstimator

config = ExperimentConfig.load('examples/f0/v2v4real_f0.json')
# Optional injected estimator (requires explicitly configured public quality proxies):
# result = run_simulation(config, f0_error_estimator=QualityProxyEstimator())
result = run_simulation(config)
print(result.summary['f0_shared_roi_coverage'])
```

`fusion_f0.error_by_agent` is a **required external estimate**, one strictly
positive and less-than-one probability per participating agent. There is no
implicit lookup of the simulator's `world.local_error`. Optional
`dependence_group_by_agent` names potentially coupled sources, and
`quality_proxy_by_agent` can supply declared ex-ante quality indicators for
an injected `QualityProxyEstimator`. Estimator inputs are immutable, and its
outputs are not calibrated unless the author independently establishes that.

Only **trustworthy source-prediction and provenance fields** are passed to
F0. The input is a binary per-ROI label, not raw LiDAR, 3D boxes or neural
logits. The adjusted log-odds sigmoid is an **uncalibrated surrogate score**,
not an error-probability guarantee. Declared error estimates, prior, source
age threshold, evidence dependence and support thresholds all matter.

V0.7's pre-inference `coordination.enabled` candidate estimator is deliberately
incompatible with this F0 policy for now, because F0 is a **post-inference
conditional** controller. The existing V0.7 pipeline remains available for
all older policies; P1 can later model sequential decisions and their candidate
costs without pretending this conditional graph is a static DAG.

### Conditional graph validation

The compiler's `selective_fusion_f0` mode requires one ordinary local
`deliver` and one `conditional_deliver` per agent. Both depend on the correct
upstream compute result. The conditional stage cannot be used by other graphs.
This extension does not alter the old graph topology or behavior of the
legacy `local_only`, independent, shared-edge and cooperative-late-fusion modes.

## 3. Reproducible replay results

**V2V4Real sequence 0007, 221 frames, 442 common decision ROIs, 2 agents.**
The local and cooperative prediction candidates are recorded V2V4Real-derived
outputs. The F0 decisions are computed from **only the local binary outputs**.
All model durations, endpoint energy coefficients and static F0 error rates
are **hypothetical**. The three strategies do **not** run the same trained
perception pipeline; this is a systems strategy comparison.

| Strategy | Both-agent wrong | Average marginal wrong | Modeled incremental energy | Mean simulated delivered latency |
| --- | ---: | ---: | ---: | ---: |
| Local-only | 4/442 (0.905%) | 22.624% | 35.36 J | 20.000 ms |
| Recorded cooperative late-fusion output | 16/442 (3.620%) | 3.620% | 53.79 J | 36.339 ms |
| F0 selective binary fusion, threshold 0.90 | 4/442 (0.905%) | 22.624% | 39.87 J | 26.413 ms |

F0 accepted 250/442 ROI decisions and retained independent local outputs for
192/442. **It did not correct any decisions in this two-agent experiment.**
With F0's conservative two-independent-support rule, shared predictions
require the two sources to agree. Disagreement retains local outcomes, so the
ROI predictions stay equal to local-only. This is expected and **negative**
empirical evidence for the hypothesis that this simple rule already provides
fault recovery with only two available sources. A truly independent third
model/sensor or an independently validated error-bound mechanism may be needed.

The F0 run performs 442 local detector jobs and 221 lightweight edge F0 jobs;
its 1 ms per-frame F0 compute model is **not equivalent** to the recorded
6 ms cooperative late-fusion compute model. Differences between those energy
figures cannot be attributed to selective acceptance alone.

**Synthetic three-agent experiment, 3,000 independent decision events:**

| Strategy | All-three wrong | Mean marginal wrong | Modeled incremental energy |
| --- | ---: | ---: | ---: |
| Local-only | 1.333% | 9.933% | 720.00 J |
| Cooperative fusion | 5.367% | 5.367% | 1034.27 J |
| F0 acceptance threshold 0.70 | 3.767% | 3.767% | 773.27 J |
| F0 acceptance threshold 0.90 | 3.133% | 6.267% | 773.16 J |
| F0 acceptance threshold 0.95 | 1.333% | 9.933% | 772.99 J |

The 3-agent toy shows a reliability/accuracy threshold tradeoff, but it is
**not evidence of calibrated error probabilities or generalization**. In
particular, even when the F0 threshold suppresses all effective sharing,
compute and upload energy remain incurred; changing a threshold cannot undo
those already scheduled stages.

Run:

```bash
python -m pip install -e '.[test,plot]'
python -m octopus_sim validate examples/f0/v2v4real_f0.json
python -m octopus_sim run examples/f0/v2v4real_f0.json \
  --no-events --output outputs/f0_v2v4real
python scripts/run_f0_integrated_study.py --output samples/f0_integrated
python -m pytest -q tests/test_f0_execution_integration.py tests/f0_original
```

`examples/f0/three_agent_f0.json` demonstrates synthetic multi-agent outcomes;
`examples/f0/v2v4real_f0.json` demonstrates real-source prediction replay. The
script regenerates CSV comparison data, explicit provenance and two PNG plots.

## 4. Failure and accounting semantics

| Event | Local fallback available? | F0 compute charged? | Downlink charged? |
| --- | --- | --- | --- |
| All uploads arrive; at least one ROI accepted | Yes, unless local detector late | Yes | Yes, one per agent per frame |
| All uploads arrive; all ROIs rejected | Yes | Yes | No |
| One required upload is lost | Yes | No (F0 not executed) | No |
| Fusion accepted but downlink lost | Yes; retained | Yes | Yes, attempted transmission |
| Fusion accepted but downlink late | Yes, if local result on time | Yes | Yes, including post-window work |
| Local detection misses deadline | No timely local result | Charged if executed | Charged if attempted |

A failed upstream upload affects the whole frame's edge-decision attempt, not
only one ROI. Output records distinguish accepted policy decisions from
**actually applied** per-agent overrides. For high-load experiments, report
incomplete events and deadline misses, not just error among delivered results.

## 5. Next steps and unresolved assumptions

1. The F0 error estimator currently uses external static/quality-proxy
   estimates. Calibrate on training scenes and validate on held-out **scenes**
   before claiming risk control.
2. Add third-source independent verification for disagreeing two-agent frames.
   Returning local is graceful degradation, not fault recovery by itself.
3. Support a real result-finalization protocol for applications unable to
   accept local-first streaming corrections. Charge notification costs.
4. Extend P1's resource cost predictor to represent **conditional** workflows
   and the probability of F0 outcomes; do not assume all conditional downlinks
   always happen.
5. Consider realistic per-ROI projection costs, frame-level result sizes,
   multiple wireless channels, and non-stationary error probabilities.
6. Expand beyond the single recorded V2V4Real sequence with held-out scene
   groups and calibrated model-specific reliability.
7. Deployable fault recovery also needs logits, tracks, objects or geometric
   evidence; F0 is explicitly a post-perception binary-decision baseline.

See [assumption audit](F0_ASSUMPTION_AUDIT.md),
[reproduction guide](F0_QUICKSTART.md) and
[the original F0 algorithm description](f0/ALGORITHM.md).
