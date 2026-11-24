# Real OPV2V / V2V4Real prediction traces — V0.2 tutorial

## Current status (read first)

**Implemented and tested:** a strict converter for multi-agent 3D detection predictions, world-coordinate alignment, common binary decision-event definitions, provenance/source checksums, a standalone capture script for the official V2V4Real/OpenCOOD model environment, full OCTOPUS-Sim replay, and paired scene-clustered analysis.

**Not completed here:** downloading the OPV2V/V2V4Real dataset, obtaining model checkpoints, or running the upstream deep learning inference pipeline on real driving frames. The official source data are large; this execution environment cannot access their download hosts. `fixtures/constructed_v2v/` contains constructed numeric boxes for testing only. **No result produced from those boxes is an empirical V2V4Real/OPV2V finding.** The source capture script is grounded in the upstream code's published interfaces but was not exercised against the actual dataset and checkpoints. It may need environment-specific adjustments.

## Source datasets

- [V2V4Real official repository](https://github.com/ucla-mobility/V2V4Real): real-world synchronized two-vehicle dataset, 3D annotations and camera/LiDAR observations.
- [OpenCOOD official documentation](https://opencood.readthedocs.io/en/latest/md_files/data_intro.html): OPV2V is **simulated**, with scenario / CAV / timestamp layout. Do not label OPV2V observations as real-world sensor recordings.
- [V2V4Real paper](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_V2V4Real_A_Real-World_Large-Scale_Dataset_for_Vehicle-to-Vehicle_Cooperative_Perception_CVPR_2023_paper.html).

The official V2V4Real `inference.py --save_npy` writes `0000_pred.npy` and `0000_gt.npy` for the **ego viewpoint**. Such files **do not** by themselves supply paired predictions for every independently observing vehicle. Blindly treating different dataloader indices as different agents would invalidate a joint-error benchmark. The accompanying `scripts/capture_v2v4real.py` iterates cars **within the same scene/frame**, invokes models separately on each car's features and post-processes each prediction in the shared reference-ego coordinate frame.

## Common decision event (fully specified)

For a synchronized scene `s`, frame `t`, and ROI `r`, all agents answer exactly the same question:

> Is the **center of any ground-truth vehicle 3D box** inside the circular region centered at `(x_r,y_r)` in shared world coordinates, with radius `R` meters?

Ground truth is computed from the reference-frame GT 3D boxes after transforming to world coordinates. For each candidate, the corresponding model's 3D detection boxes are transformed to **the same** world frame. Candidate prediction is 1 if any predicted box center lies within the same region, and 0 otherwise. This is a *binary occupancy/presence proxy*, **not** full 3D object-detection AP, a collision-risk measure, or an assertion that an agent saw all objects.

All ROIs from the same source frame share an `arrival_slot` in the replay trace, so they arrive together at the simulated edge; the interval between frame slots is a configurable simulated workload period, not necessarily the dataset’s true capture interval.

For each frame, the ROI base anchor is the mean XY position of participating vehicle LiDARs. By default ROI offsets are `(-12,0)`, `(0,0)`, `(12,0)` meters and radius is `5` meters. Only ROIs whose **entire circular extent** is within a configured nominal sensor range of **every** agent are retained (default `45` meters). A nominal range restriction does not guarantee actual line of sight or reliable labels. This method is explicit, fixed before examining predictions, and reproducible. It can be replaced with fixed world-space intersections/hazard polygons in later versions.

Every event ID has the form `scene@frame:roi_XX`. All agents share the same `truth` and event ID. `provenance.jsonl` records the scene, frame, ROI position, and prediction sources; `import_report.json` records source SHA-256 digests and class balance.

## How to capture genuine predictions

1. Download the licensed official **V2V4Real** test set (OPV2V folder layout) following its website. Use **OPV2V** for an easier synthetic-dataset alternative. **No source dataset is redistributed here.**
2. Obtain two compatible trained 3D-detection checkpoints and test their performance in the official V2V4Real environment. For the first study, two PointPillar/late-fusion-compatible models with identical preprocessing and postprocessing settings but different weights or architectures supported by that postprocessor are simplest. Using different checkpoints does **not** establish statistical independence—joint errors must be measured.
3. In that existing Python environment, with `opencood` importable, run the capture utility. It uses the official late-fusion *per-vehicle* postprocessor and disabled artificial localization/async noise so that OCTOPUS can vary those effects itself. **The capture step may require GPU/CUDA or a compatible NMS build; only subsequent replay is guaranteed CPU-first.**

```bash
# Run this script from an environment where V2V4Real's opencood is importable.
python scripts/capture_v2v4real.py \
  --data-dir /data/v2v4real/test \
  --local-model-dir /checkpoints/local \
  --edge-model-dir /checkpoints/edge \
  --output /data/octopus_captured \
  --max-frames 500 --stride 5 --device auto
```

The script outputs `frames.jsonl` referencing ground-truth and two sets of per-vehicle `.npy` arrays. For every included frame it uses the official scenario database's actual scenario and timestamp keys rather than a bare dataloader index. Each `local:` and `edge:` candidate is a **recorded** separate model invocation using that source vehicle's observation, but the two may still have strongly dependent errors. The script is provided as a source-ready extraction path, not a claim of successful execution against real-world files in this session.

### Format of the input frame manifest

One JSON object per synchronized frame:

```json
{
  "dataset": "V2V4Real",
  "scene_id": "SCENE_FROM_SOURCE_FOLDER",
  "frame_id": "00042",
  "reference_lidar_pose": [0, 0, 0, 0, 0, 0],
  "ground_truth": {"path": "boxes/frame_gt.npy", "coordinates": "reference_lidar"},
  "agents": [
    {"agent_id": "1", "lidar_pose": [0, 0, 0, 0, 0, 0],
     "local_prediction": {"path": "boxes/a_local.npy", "coordinates": "reference_lidar"},
     "edge_prediction": {"path": "boxes/a_edge.npy", "coordinates": "reference_lidar"}},
    {"agent_id": "2", "lidar_pose": [5, 0, 0, 0, 0, 0],
     "local_prediction": {"path": "boxes/b_local.npy", "coordinates": "reference_lidar"},
     "edge_prediction": {"path": "boxes/b_edge.npy", "coordinates": "reference_lidar"}}
  ]
}
```

Coordinates can be `world`, `reference_lidar`, or `agent_lidar` (the latter is for per-agent predictions). The supported 3D box representation is `[N,8,3]` corners (as exported by OpenCOOD), `[N,7]` center/size/yaw, or `[N,3]` centers; `[N,4,2]` is also recognized. `.npy` loads with `allow_pickle=False`; empty predictions should be numeric arrays of shape `(0,8,3)`. This JSON example describes a schema, **not an actual V2V4Real sample**.

## Convert and run entirely on CPU

Once genuine prediction arrays have been exported, no video or neural inference is necessary for the benchmark replay:

```bash
# NumPy needed for .npy imports, but not for JSON-only test fixtures
python -m pip install -e '.[vision_import]'

python -m octopus_sim import-opv2v /data/octopus_captured/frames.jsonl \
  --output outputs/v2v4real_decisions \
  --radius 5 --common-range 45 --offsets='-12,0;0,0;12,0'

python -m octopus_sim validate outputs/v2v4real_decisions/scenario.json
python -m octopus_sim compare outputs/v2v4real_decisions/scenario.json \
  --policies local_only,always_independent,always_share,partial_share \
  --no-events --output outputs/v2v4real_comparison

python -m octopus_sim analyze-empirical \
  outputs/v2v4real_comparison \
  outputs/v2v4real_decisions/provenance.jsonl \
  --baseline always_independent --iterations 2000 \
  --output outputs/v2v4real_empirical_analysis.json
```

**Interpret resource numbers cautiously.** The generated scenario's link, compute, and deadline settings are illustrative defaults, not calibrated to the recorded hardware or V2V4Real network. Configure measured workloads and communication traces before reporting absolute latency/energy claims. The dataset frame interval and the simulator's event-arrival interval may not coincide: the default is a *workload replay schedule*, not reconstructed physical traffic time.

## Quick end-to-end constructed fixture (runs immediately)

```bash
python -m octopus_sim import-opv2v \
  fixtures/constructed_v2v/frames.jsonl \
  --output outputs/constructed_import
python -m octopus_sim compare outputs/constructed_import/scenario.json \
  --policies always_independent,always_share,local_only \
  --no-events --output outputs/constructed_compare
python -m octopus_sim analyze-empirical outputs/constructed_compare \
  outputs/constructed_import/provenance.jsonl \
  --baseline always_independent --iterations 1000 \
  --output outputs/constructed_analysis.json
```

This fixture contains **two hand-constructed frames from one fictional scene**, not a model, recorded sensor data, or an empirical measurement. The empirical analyzer will correctly flag that scene-clustered confidence intervals are not estimable with a single scene.

## Strictness and threats to validity

- Event alignment is by **(source scene, actual frame, vehicle ID)**. Do not mix timepoints or ego reference frames. The importer requires a stable participating agent set within the imported slice.
- We never fabricate absent predictions. Both local and edge model outputs are mandatory per source agent. Same-file aliases are rejected by default; even distinct checkpoints are not assumed statistically independent.
- GT uses the upstream *shared ego-aligned* 3D annotations. It is not separately inferred from each vehicle's partial observation.
- Multiple ROI tasks within a frame and neighboring frames are dependent. **Use scene-clustered paired bootstraps**, not event-wise binomial confidence intervals, when publishing empirical estimates. The `analyze-empirical` command suppresses CIs if fewer than two scenes are available.
- The archive includes no official raw data or model weights; original dataset and model licenses apply.
- The capture script's compatibility with specific upstream commits is **untested here**. Verify it in a real V2V4Real environment before making reproducibility claims.
- Presence within a circular ROI is a simplified proxy. This may exaggerate/understate practical inference error and does not measure downstream collision or autonomous-driving safety.
- A cached inference across agents should be applicable to the same defined ROI. The V1 simulator can replay decisions as result reuse but does not model additional contextual incompatibilities or stale-world motion beyond its existing network deadlines.
- A useful future upgrade is adding region-level visibility and class filters and providing fully audited evidence availability per sensor.

## What to publish in the eventual paper

Show the number of independent scenes, frames, and events; prevalence and ROI generation; checkpoint identifiers and model configurations; predictions' source-vehicle identities; marginal and joint error rates; scene-clustered paired confidence intervals; reused versus independently generated result IDs; and the compute/bandwidth assumptions. Provide error examples with the original scene and frame IDs so another researcher can inspect the source dataset.

## Validation snapshot

The offline test suite includes a mocked-upstream capture-contract test that creates per-vehicle `.npy` predictions via the same API shape and verifies the conversion and replay steps. **It does not execute actual V2V4Real/OpenCOOD code or train models.** The unit and integration tests of v0.2 passed **53 tests** at packaging time.
