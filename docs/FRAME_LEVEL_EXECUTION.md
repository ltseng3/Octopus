# Frame-level execution and cache semantics (OCTOPUS-Sim v0.6)

## Why this change is necessary

OCTOPUS-Sim scores **decision events**, such as binary vehicle presence in each
region of interest (ROI). A camera/LiDAR detector, however, normally produces
an **entire frame-level detection output**. Multiple ROIs can query that same
output without another model execution.

The older v0.5.1 V2V4Real resource experiment mapped each ROI to an independent
execution graph. It therefore charged two detector/fusion invocations for two
regions of the same sensor frame. That artificially inflated compute energy,
network traffic and (because of per-agent FIFO queueing) delivery latency.
The recorded model **error outcomes** remain valid; the old resource accounting
did not represent the physical frame-level workflow.

This release distinguishes **three identities**:

1. **Physical observation epoch**: `frame_id`, unique across scenes and frames.
2. **Physical inference result**: one stable `result_id` per producing compute
   stage in that frame's graph; an actual inference result may contain many
   detected boxes and answer multiple decision queries.
3. **Decision event**: `event_id`, a specific ROI with its own truth and
   per-candidate binary output. It can consume a physical inference artifact
   without reexecuting the model.

A coincident `arrival_slot` is **not** enough for caching: two unrelated frames
can arrive simultaneously. The frame ID must be explicit and checked.

## Opt-in contract

Use the default `decision` mode for old scenarios, or opt in to `frame` mode:

```json
{
  "world": {
    "backend": "trace",
    "trace_file": "trace.jsonl",
    "execution_granularity": "frame",
    "fusion_candidate_key": "edge:ego"
  }
}
```

For each ROI, trace JSONL records provide an explicit shared `frame_id`, a
nonnegative integer `arrival_slot`, a unique `event_id`, binary `truth`, and
the requisite recorded `candidates`:

```json
{"event_id":"scene001:frame010:roi_a","frame_id":"scene001:frame010","arrival_slot":10,"truth":1,"candidates":{"local:ego":1,"local:partner":0,"edge:ego":1,"edge:partner":1}}
{"event_id":"scene001:frame010:roi_b","frame_id":"scene001:frame010","arrival_slot":10,"truth":0,"candidates":{"local:ego":0,"local:partner":0,"edge:ego":0,"edge:partner":0}}
```

Two ROI decisions can have **different predictions**, even when consuming the
same physical result. The cached item is not the boolean `1` or `0`. It is the
frame-level inference artifact, represented by a `predictions_by_event_id`
mapping during binary replay.

Invalid or missing frame IDs, missing slots, or reuse of one `frame_id` at
multiple arrival slots cause hard failures. Frame execution requires the strict
trace backend: synthetic independent events remain decision-level by default.

### Policy and execution semantics

- For each frame, the coordinator sees the frame identity, agent list,
  configured assumptions and deadline, **not ground truth**. It chooses one
  allocation plan for the whole frame.
- One immutable validated P0 execution DAG runs once per frame. It may contain
  several local detections, per-device uploads, one edge fusion, and per-agent
  result deliveries. This graph is **physical work**, and is not multiplied by
  the number of ROI evaluations.
- All ROI decisions in the frame are released together and individually
  scored against their own ground-truth labels, with the same deadline.
- One per-frame result transfer to a receiving agent covers **the entire
  configured result payload**. Projection onto multiple ROIs on that device is
  currently modeled as negligible extra work. This is an explicit modeling
  assumption, not an empirically measured optimization.
- Packet loss is sampled and charged **per physical transfer**, so a lost
  frame result can affect all ROI consumers. Each attempted transfer consumes
  modeled radio energy. Queueing and failed upstream stages propagate through
  the graph normally.
- Reuse never crosses a frame boundary. Even two frames with the same release
  timestamp retain distinct result IDs, execution graphs, and resource costs.
  There is no cache TTL, cross-frame temporal reuse or staleness model yet.
- A randomized policy samples **once per frame**, not once per ROI. This is
  the correct allocation unit for a frame-level model result. Interior mixed
  policy results may differ between old and new runs even with the same seed;
  the policy randomization unit is intentionally different.
- The graph-template cache stores immutable **stage topology only**; it never
  stores model outputs or ground truth between physical frames.

### Result and score schema

For a frame-level inference record:

- `result_id`: `frame:<frame_id>:result:<stage_id>`.
- `frame_id`, `decision_event_ids` identify all ROI decisions it supports.
- `predictions_by_event_id`: the ROI-specific recorded binary projection.
- `prediction`: `null` for multi-ROI artifacts, intentionally, because no
  single scalar represents the complete frame output. Singleton legacy runs
  retain the scalar prediction.
- `source_agents`, `parent_result_ids`, `observation_ids`: preserve the
  multi-source inference DAG ancestry.
- `consumers_delivered`: physical receiving agents (at most once per result).
- `decision_deliveries`: explicit `(event_id, agent)` projections after
  delivery. A single physical result may appear in several `scores.jsonl`
  events and many agents' result IDs.

Each scored decision also includes `frame_id` and retains the original truth,
per-agent predictions, error indicators, misses, and delivery latencies.

In `summary.json`, use:

- `execution_granularity`: `frame` or `decision`.
- `physical_execution_batches`: number of actual P0 execution DAG instances.
- `frame_projection_cache_hits`: the number of additional decision-level
  projections served by physically produced frame inference artifacts,
  calculated as `(regions_per_frame - 1)` for each successful compute result.
  It is a **projection reuse count**, not saved wall-clock milliseconds.
- `frame_projection_cache_hits_by_model`: model-level equivalent.
- `decision_events_sharing_physical_frame`: number of decision events belonging
  to a multi-ROI frame.
- Existing local/edge job counts, CPU busy times, bytes, queue times, and
  energy are still physical resource totals; `episodes` remains the number of
  decision events, NOT frames.

## V2V4Real source-verified example

`examples/v2v4real_real_seq0007/trace.jsonl` provides 221 synchronized source
frames and 442 newly defined decision regions. Separate local predictions and
cooperative late-fusion predictions are recorded model outputs. Execution
choices, processing times, payload sizes and power profiles are simulated.

With hypothetical E1 profiles, *after correcting physical execution counts*:

| System | Local detector runs | Edge fusion runs | Bytes transmitted | Mean delivered latency | Incremental modeled energy | All-agent errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Local only | 442 | 0 | 0 | 20.000 ms | 35.360 J | 4 / 442 |
| Cooperative fusion | 442 | 221 | 1,867,008 | 36.339 ms | 53.793 J | 16 / 442 |

The older per-ROI numbers (884 detectors, 442 fusions, and 3,734,016 bytes for
fusion; 30.000 ms local latency; 46.339 ms full-fusion latency) are archived
in `samples/legacy_v051/` and should **not** be used as physically faithful
frame-level costs. Their error endpoints were unaffected by this particular
accounting error.

### Reproduce

```bash
python -m octopus_sim validate examples/energy/v2v4real_0007_energy.json
python -m octopus_sim compare examples/energy/v2v4real_0007_energy.json \
  --policies local_only,cooperative_late_fusion,mixed_local_fusion \
  --reuse-grid 0,0.25,0.5,0.75,1 --no-events \
  --output outputs/v06_v2v
python scripts/run_energy_e1.py --output samples/energy_e1
python scripts/run_e1_multiobjective.py --output samples/multiobjective_e1
python scripts/run_frame_cache_study.py --output samples/frame_cache_v06
python -m pytest -q
```

To reproduce v0.5.1's exact **per-ROI physical accounting assumption** using
the current code, switch `execution_granularity` to `decision` or run
`examples/energy/v2v4real_0007_legacy_per_roi.json`. This comparison is for
software regression only, not an alternative valid physical workflow.

## Load and latency sanity checks

Under the supplied 500 ms frame interval, all operations complete long before
the deadline and have fixed service durations, so the measured per-agent
latency has only a few discrete outcomes. **There is no latency cap.** When
frames arrive more frequently than workers can complete their work, FIFO
queues grow. The load-sweep figure
`samples/frame_cache_v06/01_latency_under_load_frame_vs_roi.png` plots P95
**delivered** latency alongside the fraction of missing/deadline-missed agent
results. Do not interpret delivered-only P95 as safe under overload: it
excludes deliveries that missed their deadlines.

Because the counterfactual scheduler can drop outcomes after deadlines, under
heavy load the *conditional delivered-only error rate* changes due to
selection, even if the underlying recorded prediction arrays do not. Always
report both raw decision error and operational failure/miss metrics.

## What v0.6 intentionally does not model

- Incremental compute/communication cost of *individual ROI projections*;
  current binary ROI lookup is treated as free after a frame result arrives.
- ROI-specific **adaptive** inference decisions within a frame. The current
  coordinator selects a shared allocation for all ROIs in the same frame.
- Reusing or refreshing old results **across** distinct physical frames, cache
  invalidation policies, TTL, age-of-information, or adaptive model execution.
- Dataset-native box filtering, NMS or trained models inside the replay
  scheduler. The source trace contains precomputed binary projections.
- Real hardware runtimes, wireless radio power, or collision-risk semantics.

These are deliberate extension points, not hidden approximations. Future
ROI-specific policies can compile a union of per-ROI execution requirements
while retaining the same physical `frame_id`/`result_id` model.

## Regression and engineering acceptance criteria

`tests/test_frame_execution_cache.py` exercises multiple ROI counts, two/
three/five agents, all major graph policies, exact job/byte counts, source
identities, frame-scoped outcomes, loss propagation, deadlines, deterministic
results and configured energy totals. The legacy 125 tests still run, with
expected values updated only where a canonical scenario intentionally opts in
to frame mode. Additional `tests/test_frame_cache_study.py` verifies the
non-capped queueing-load experiment. Every generated archive must pass the
full test suite from a clean extracted directory.
