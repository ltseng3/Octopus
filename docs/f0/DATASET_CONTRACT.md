# Data contract and provenance

F0 accepts **one JSON object per ROI decision event** (`octopus-fusion replay`). A minimal example is in [`examples/two_agent_fixture.jsonl`](../examples/two_agent_fixture.jsonl).

```json
{
  "event_id": "scene/frame000/roi0",
  "frame_id": "scene/frame000",
  "as_of_ms": 20.0,
  "prior_positive": 0.5,
  "truth": 1,
  "evidence": [
    {
      "agent_id": "vehicle_a",
      "prediction": 1,
      "result_id": "scene/frame000/local_result_a",
      "dependence_group": "local_origin_a",
      "produced_at_ms": 18.0,
      "assumed_error_probability": 0.08,
      "quality_proxy": 0.90
    },
    {
      "agent_id": "vehicle_b",
      "prediction": 0,
      "result_id": "scene/frame000/local_result_b",
      "dependence_group": "local_origin_b",
      "produced_at_ms": 19.0,
      "assumed_error_probability": 0.21,
      "quality_proxy": 0.65
    }
  ]
}
```

- `truth` is required for **offline evaluation only**. The loader constructs a separate `LabeledCase` and never passes truth to `SelectiveFuser.decide`.
- `assumed_error_probability` must be a finite probability strictly between 0 and 1. It must be estimated from information available before the decision, not reconstructed from the same event's correctness label.
- `result_id` is a unique physical inference artifact identity. A result can be reused in several ROI records but may return **different binary ROI labels**. Within one ROI, the same result ID must not contradict itself.
- `dependence_group` should conservatively join reports that are not known to be independent, including exact result copies and shared origins. Multiple groups are not automatically statistically independent.
- `produced_at_ms` must precede the edge's `as_of_ms`. F0 does not itself model source→edge transfer time; the exporter must choose a meaningful observation availability time.
- `event_id` must be unique; `frame_id` identifies the frame; all agents in one F0 ROI observation are distinct.
- Optional `quality_proxy` is a numeric ex-ante feature, not a correctness bit. The externally supplied assumed q remains authoritative in `replay` mode.

## Using V2V4Real, OPV2V or other data

A true collaborative-perception evaluation needs recorded **per-agent predictions** for the *same common decision event*. If an imported scene only supplies a fused result, it cannot be treated as an independent per-agent detector result. The V0.7 offline adapter rejects incomplete local artifacts rather than silently inventing separate outputs. **No real V2V4Real/OPV2V dataset, detector checkpoint, score file or generated derivative is included in this F0 release.**

Before public release of derived data, confirm the original dataset, model exports and checkpoints permit redistribution; include version, source URLs, checksums, split boundaries and frame/ROI-definition code. A dataset's licensing conditions are separate from this code's MIT license.
