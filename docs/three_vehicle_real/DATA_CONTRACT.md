# CooperScene-to-OCTOPUS three-vehicle binary ROI export contract

The importer does not inspect `.pcd`, `.yaml`, `.pth` or point-cloud tensors.
It accepts only a **matched binary ROI export** from three independently
executed physical vehicle detector runs. Upstream detection/label conversion
and global coordinate alignment are the dataset operator's responsibility.

## Roles and provenance

CooperScene identifies physical vehicles `"1"`, `"2"`, and `"3"` and
roadside infrastructure `"0"`. The importer requires `vehicle_ids` exactly
`["1","2","3"]` and three distinct `physical_detector_runs`. A cooperative
model output (`V+2V`) is a **derived dependent result**, not an additional
vehicle. Ground truth comes from a separate aligned scoring stream.

The provided manifest must contain:

```json
{
  "dataset": "CooperScene",
  "split": "test",
  "vehicle_ids": ["1", "2", "3"],
  "roi_protocol": "predeclared_world_xy_grid_v1",
  "coordinate_frame": "cooperscene_global_xy_m",
  "prediction_origin": "real_detector_export",
  "physical_detector_runs": {
    "1": "unique_vehicle1_run_id",
    "2": "unique_vehicle2_run_id",
    "3": "unique_vehicle3_run_id"
  },
  "assumed_source_error": {"1": 0.06, "2": 0.12, "3": 0.30},
  "error_estimator_status": "unfitted_assumption",
  "interval_ms": 100.0,
  "deadline_ms": 250.0,
  "checkpoint_sha256": {
    "1": "<64 hex digits>", "2": "<64 hex digits>", "3": "<64 hex digits>"
  },
  "per_vehicle_prediction_exports": {
    "1": {"path": "vehicle1.jsonl", "sha256": "<64 hex digits>"},
    "2": {"path": "vehicle2.jsonl", "sha256": "<64 hex digits>"},
    "3": {"path": "vehicle3.jsonl", "sha256": "<64 hex digits>"}
  },
  "ground_truth_export": {
    "path": "ground_truth.jsonl", "sha256": "<64 hex digits>"
  }
}
```

**Warning:** numbers above are illustrative q assumptions, not calibrated
CooperScene error rates; replace before a scientific experiment. The converter
requires the checkpoint, three per-vehicle export, and GT hashes for
`real_detector_export`, and re-hashes the actual files. It also requires their
predictions to match the merged input row-by-row. A metadata assertion cannot
replace genuine upstream inference provenance.

The external vehicle file contains one row per matched ROI:

```json
{"take_id":"7","frame_token":"481260","roi_id":"grid_00","vehicle_id":"1","prediction":1}
```

The GT file contains:

```json
{"take_id":"7","frame_token":"481260","roi_id":"grid_00","truth":1}
```

The *merged* input file contains all three predictions and truth on the same
common event, with a fixed global ROI selection:

```json
{"split":"test","take_id":"7","frame_token":"481260",
 "roi_id":"grid_00", "truth":1,
 "predictions":{"1":1,"2":1,"3":0},
 "roi":{"frame":"cooperscene_global_xy_m",
        "center_xy_m":[15.0,20.0],"radius_m":7.5}}
```

All fields `truth` and `prediction` must be true JSON integers `0` or `1`,
not booleans, probabilistic logits, or string labels. All three vehicles
must have a prediction for **every** frame/ROI. Frame ID includes dataset,
split, take and timestamp to prevent cross-take cache contamination.
The converter does not support partially observed frames and does not merge
arbitrarily asynchronous measurement streams.

The converter writes `trace.jsonl`, `scenario.json`, and `PROVENANCE.json` for
OCTOPUS-Sim. `trace.jsonl` intentionally contains only `local:1`, `local:2`,
`local:3` candidates. **No fake `edge:*` predictions are created.**

## Error estimates / split leakage

`assumed_source_error` must be supplied externally; the converter **never
computes it from the test trace's truth**. If rates were fitted on a distinct
training subset, set `error_estimator_status` to
`fit_on_disjoint_training_split` and `error_fit_split` to `train`.
That is a researcher's assertion, not independent verification of training.

## Physical model assumptions

- Three different physical vehicles produce three local per-frame inference
  jobs. Shared model weights across different vehicles are allowed; the
  physical inference artifacts remain distinct.
- Three physically separate inferencers may nevertheless have correlated
  errors from weather, viewpoint or common training data. The current F0
  lineage grouping does not estimate that statistical dependence.
- The conversion supports multiple ROIs per physical frame and leaves them
  as free projections of each physical result, consistent with V0.7 frame
  semantics. This approximation should be revisited if projection is costly.
- The F0 selector receives only the predictions and **declared q estimates**,
  never test truth. Truth is consumed by scoring after the decision.
- Cost and deadline results use configured modeled durations and hypothetical
  energy profiles, NOT CooperScene measured device power or direct C-V2X
  packet-trace replay.

## Dataset licensing

CooperScene dataset files: **CC BY-NC-SA 4.0** according to its upstream
[license notice](https://github.com/UCR-CISL/CooperScene#license). CooperScene
code: MIT. Use the dataset under its license; do not merge upstream files into
an unrelated MIT source archive without reviewing the license and attribution.
