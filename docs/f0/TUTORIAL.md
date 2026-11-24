# F0 hands-on tutorial

The complete tutorial runs on a laptop with **CPU only**; no perception models or original dataset are required.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,plot]'
pytest -q
```

On Windows, activate `.venv\Scripts\activate` instead. `pytest` is a dev dependency; runtime fusion itself uses the standard library.

## 2. Run a three-agent masking example

```bash
python examples/demo_fusion.py
```

Agent A predicts hazard (estimated error 5%), B predicts clear (35%), and C predicts hazard (10%). The default threshold masks B; A and C are independent enough under the **declared** groups to support a shared hazard result. Try `FusionConfig(require_risk_bounds=True)`: no external risk estimate is available, so the algorithm returns `KEEP_LOCAL`.

The key object:

```python
from octopus_fusion import (
    AgentEvidence, FusionObservation, ReliabilityEstimate,
    SelectiveFuser,
)

obs = FusionObservation('f1/roi1','f1',100.0,(
    AgentEvidence('a',1,'f1/rA','lineage_a',95.0),
    AgentEvidence('b',0,'f1/rB','lineage_b',95.0),
    AgentEvidence('c',1,'f1/rC','lineage_c',95.0),
))
q = ReliabilityEstimate((('a',.05),('b',.35),('c',.10)))
decision = SelectiveFuser().decide(obs,q)
print(decision.action, dict(decision.per_agent_decisions))
```

The `result_id` is a frame result. You may reuse `f1/rA` on another ROI, but must construct a *separate ROI observation* with its own binary predictions. Never cache the binary ROI result across ROIs.

## 3. Try a two-agent conflict

```bash
octopus-fusion replay examples/two_agent_fixture.jsonl --out results/two_agent
cat results/two_agent/comparison.csv
```

The first ROI contains disagreement between two agents and triggers `KEEP_LOCAL`; the second ROI is unanimous and may accept fusion. The same physical result IDs are valid for both ROIs, but their projections differ. Evaluation truth appears only in the labeled replay record; the fuser never receives it.

## 4. Reproduce the larger synthetic demonstration

```bash
octopus-fusion toy --frames 1800 --agents 3 --rois 2 --seed 4701 \
  --out results/my_toy
cat results/my_toy/comparison.csv
```

Each benchmark policy receives exactly the same generated cases. The simulation includes occasional low-quality agents, duplicated result lineages and correlated hidden error shocks. The quality-proxy-to-error mapping is an **assumption**, not a fitted model.

Expected metrics (approximate): selective F0 shares 90.8% of decisions, with 7.58% average agent error and 4.03% all-agent error. Local-only's joint error is lower (3.64%); naive always-sharing has lower marginal error but higher joint error (6.47–6.78%). Changing the toy settings can reverse these rankings.

## 5. Sweep thresholds and make a figure

```bash
python scripts/threshold_sweep.py --frames 1500 --bootstrap 300 \
  --out results/my_sweep
```

Outputs: `threshold_sweep.csv`, `threshold_tradeoff.png`, `PROVENANCE.json`. The confidence intervals bootstrap **whole frames**, retaining both ROIs in each frame together. The bars/lines are descriptive of a synthetic configuration, not an empirical detector study.

## 6. Import your own binary decision events

Use the explicit JSONL contract in [DATASET_CONTRACT.md](DATASET_CONTRACT.md). Provide external `assumed_error_probability` estimates for each agent and event. Do **not** derive them from the ground-truth label on the same evaluated event. Use held-out calibration data or a named synthetic assumption.

Then run:

```bash
octopus-fusion replay /path/to/your_events.jsonl --out results/your_run
```

`selective_decisions.jsonl` contains only decisions and mask reasons, not ground truth. `comparison.csv` contains aggregate evaluation metrics, and `manifest.json` records which kind of evidence was used.

## 7. Attempt an offline V0.7 adapter (optional)

See [V07_INTEGRATION.md](V07_INTEGRATION.md). This adapter was tested against fixtures matching the previously documented V0.7 export schema, not the unavailable original V0.7 archive. Its outputs cannot replace an integrated F1 experiment.

## 8. Push to GitHub

Follow [GITHUB_SETUP.md](GITHUB_SETUP.md) to create an empty repository and push. The ZIP contains ready-made CI, project metadata, a LICENSE and release hashes.
