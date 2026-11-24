# CooperScene mini integration entrypoint

This folder contains no CooperScene raw data or fabricated per-vehicle predictions.

Start at [`docs/three_vehicle_real/MINI_DATASET_SMOKE.md`](../../docs/three_vehicle_real/MINI_DATASET_SMOKE.md).

The upstream mini data archive is ~2.9 GB and its dataset license is CC BY-NC-SA 4.0. Use `scripts/check_cooperscene_mini.py` to download (only where outbound networking is enabled), safely extract, scan the synchronized `0` (RSU) + `1,2,3` (vehicle) raw frame inventory, and produce `mini_scan.json` and `frame_index.csv`.

`tests/test_cooperscene_mini_validator.py` uses synthetic *file-layout* fixtures, never a fake measurement dataset. No real F0 joint-risk results are bundled.
