# CooperScene mini: CPU validation and GPU execution protocol

**Status of this copy: MINI SENSOR ARCHIVE NOT DOWNLOADED OR INSPECTED.** The 349 existing + new tests verify software; they do not establish empirical detector results. The current branch uses `COOPERSCENE_MINI_RELEASE_SHA256.json` for integrity, while preserving earlier release manifests as historical references.

The official release offers a **2.9 GB mini.zip** with a claimed **180 synchronized physical frames** (120 train, 30 validate, 30 test). The upstream directory structure uses `<split>/<take>/<agent>/<timestamp>.{pcd,yaml}`. Physical vehicles are `1`, `2`, `3`; roadside infrastructure is `0`. CooperScene data is **CC BY-NC-SA 4.0**, separate from MIT source code. [Official project](https://cisl.ucr.edu/CooperScene/) · [Official mini archive](https://data.ucr.edu/datasets/cooperscene/mini.zip) · [Official code](https://github.com/UCR-CISL/CooperScene).

## 1. Obtain the mini release

From an Internet-connected computer (the current hosted execution sandbox cannot access the binary archive):

```bash
python -m pip install -e '.[test,yaml]'
python scripts/check_cooperscene_mini.py download \
  --archive /path/to/mini.zip
```

If a browser download is more reliable, open the official mini URL above. Keep the original downloaded ZIP and checksum. **Do not place raw data in the public OCTOPUS source repository.**

Then unpack with path traversal checks:

```bash
python scripts/check_cooperscene_mini.py extract \
  --archive /path/to/mini.zip \
  --directory /path/to/mini_extracted
```

The utility refuses to silently replace an existing archive or extract into a nonempty directory. It limits total extracted size to 24 GiB.

## 2. Scan actual 3-vehicle sensor/label alignment (CPU only)

```bash
python scripts/check_cooperscene_mini.py scan \
  --data-root /path/to/mini_extracted \
  --output outputs/cooperscene_mini_scan \
  --strict-mini
```

The tool auto-discovers the data root up to two nested directories below `--data-root`. `--strict-mini` expects 120/30/30 synchronized physical frames (per official README). A non-strict scan is supported for partial local copies, but requires all three vehicles and the RSU for each included frame.

Outputs:

- `mini_scan.json` — split/take counts, LiDAR/annotation inventory, warnings, upstream license, and *metadata-only* status.
- `frame_index.csv` — one row per synchronized physical frame; four platform IDs, per-platform annotation counts and PCD point counts.

Checks: three independently identified vehicle directories; RSU is not counted as a vehicle; identical timestamp sets for vehicles and RSU; matching `.pcd/.bin` and `.yaml`; valid PCD header fields; safely parsed numeric pose and annotation dictionary; no cross-frame inference creation. It **does not** verify GT consistency after fusion, transform 3D boxes to world coordinates, estimate model accuracy, or run inference.

If mini train/test windows share take IDs, the scanner warns about *temporal adjacency*. The mini has only 30 validation and 30 test frames: it should be treated as a pipeline smoke test, not adequate statistical evidence for CVC claims. Avoid tuning q or thresholds using those same test labels.

## 3. Run official upstream converters and a first model (CUDA machine)

After the CPU scan passes, on a GPU workstation or cluster:

```bash
git clone https://github.com/UCR-CISL/CooperScene.git
cd CooperScene
git rev-parse HEAD > /path/to/upstream_cooperscene_commit.txt

# Pin an upstream commit in your reproducibility manifest.
# Official Docker environment: PyTorch 2.1.1, CUDA 12.1, mmengine/mmdet3d.
docker build -t cooperscene -f docker/Dockerfile .
docker run --gpus all -it --rm \
  -v "$(pwd)":/workspace/CooperScene \
  -v /path/to/mini_extracted/cooperscene:/workspace/CooperScene/data/cooperscene \
  cooperscene bash

# Inside the container; fix mount path if mini ZIP has another root name.
python tools/dataset_converters/data_converter.py \
  --data-root data/cooperscene --out-dir data/cooperscene --convert-pcd
python tools/download.py bevfusion_single_lidar
python tools/test.py \
  assets/bevfusion_single_lidar/bevfusion_single_lidar.py \
  assets/bevfusion_single_lidar/bevfusion_single_lidar.pth
```

**Caveat:** asset layout is governed by the precise pinned upstream revision and must be checked after `tools/download.py`; do not assume the paths above work on every revision. The official README states that `tools/download.py` creates matching `assets/<model>/{<model>.py,<model>.pth}` files. The test command above validates upstream evaluation *only*; it does not create the individual ROI JSONL exports that F0 requires.

## 4. Required detector-result export for OCTOPUS-F0

Preserve **three physically separate** prediction streams, containing 3D predicted boxes, confidence values, source vehicle ID, and dataset `split/take/timestamp`. The model checkpoint can be identical across vehicles; the physical executions and outputs must be separate. Do not pass an aggregate `V+2V` cooperative prediction as one of the three local observations.

The existing strict OCTOPUS import contract is in [`DATA_CONTRACT.md`](DATA_CONTRACT.md). It expects *predeclared fixed world-XY ROI centers*, independent vehicle binary projections, shared truth for scoring, SHA-256 hashes, and three checkpoint hashes. There is currently **no verified automatic box-to-world-grid converter or detector-export hook** bundled in this branch. This work requires validation on actual mini data and detector output conventions before real F0 risk plots can be produced.

Next step after real detector exports: generate matched local predictions and ground truth using a frozen ROI grid **defined without test labels**, prepare with `scripts/run_three_vehicle_real_study.py prepare`, evaluate with `... evaluate`, and report `R_2`, `R_3`, marginal error, corrections/harm and modeled energy. For the publication study move to full independent takes, train-only error calibration and sequence-level uncertainty.

## Research provenance to report

* The mini's three sensor streams and labels are **real dataset data** only once downloaded and scanned.
* Single-agent BEVFusion predictions become **real model predictions** only when the pinned detector runs successfully on them.
* OCTOPUS-F0 `R_2`/`R_3` become **measured replay results** only after validated per-vehicle prediction exports exist.
* Energy/latency in OCTOPUS are **modeled**, even with empirical predictions, until independently calibrated with hardware or network measurements.
* This source release is **software tested**, not an empirical three-vehicle F0 claim.
