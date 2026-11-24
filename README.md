# OCTOPUS-Sim 0.7.1

**Inference diversity in collaborative edge vision.** OCTOPUS studies how separate perception outputs, cooperative fusion, and shared inference change the distribution of errors across a group of observers. It is a CPU-first discrete-event research benchmark, not a 3D detector or an autonomous-driving simulator.

Different viewpoints offer complementary observations; separate predictions may also preserve redundancy. Fusing evidence and making every observer adopt one prediction are distinct decisions. Neither independence nor consistency is universally preferable: evaluate misses, false alarms, collective error thresholds, and timely delivery for the application.

## Start here

- [Research guide and roadmap](docs/RESEARCH_GUIDE.md): rationale, three research questions, evidence, next experiments and acceptance criteria.
- [Reproduce the research figures](figure_reproduction/README.md): exact scripts, input tables, figure map and upstream regeneration commands.
- [Execution and F0 design](docs/F0_ONLINE_INTEGRATION.md): result identities, post-inference masking, conditional delivery and fallback.
- [Frame-level execution](docs/FRAME_LEVEL_EXECUTION.md), [energy model](docs/ENERGY_E0_E1.md), [empirical task](docs/EMPIRICAL_V2V4REAL_RESULTS.md).

## Quick start (Python 3.10+)

```bash
git clone https://github.com/ltseng3/Octopus.git
cd Octopus
python -m venv .venv
source .venv/bin/activate     # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[test,yaml,empirical_plot,empirical_download]'
python examples/reuse_toy.py
python -m octopus_sim validate examples/tutorial/toy_f0.json
python -m octopus_sim run examples/tutorial/toy_local.json --output outputs/toy_local
python -m octopus_sim run examples/tutorial/toy_f0.json --output outputs/toy_f0
python -m octopus_sim run examples/tutorial/toy_strict.json --output outputs/toy_strict
```

The exact reuse toy has five receivers and three independent inference jobs. With 10% error per artifact, a 2+2+1 partition has 2.8% majority-failure risk, versus 10% for 3+1+1; both have 0.1% all-five failure risk. The simulator example uses 80 synthetic decisions and three agents with seed 123. Expected standard F0 output: 240 local jobs, 80 edge decision jobs, 72 accepted decisions, 5% average error, R2=2.5%, R3=1.25%, and no missed deadlines. These are small deterministic smoke tests, not generalizable empirical results.

Inspect `summary.json`, `scores.jsonl`, `inferences.jsonl`, `operations.jsonl` and `fusion_decisions.jsonl` in each run. Change one input at a time and retain the seed for paired comparisons.

## Source versus data

This public source mirror contains the simulator and F0 runtime, study scripts, test source, configurations, and constructed/synthetic fixtures from the tested v0.7.1 release. It intentionally does **not** redistribute real detector-derived trace/provenance files, upstream sensor data, archived results or the manuscript. The separately supplied complete student archive retains historical data and checksums. The public mirror does not satisfy the historical full-release inventories; use `python scripts/verify_source_snapshot.py` for this tree.

The synthetic smoke tests below need no real data. Tests and scripts that use recorded traces or archived results require the corresponding data from the student archive or independently regenerated source files; do not manufacture missing predictions.

```bash
python scripts/verify_source_snapshot.py
python -m pytest -q tests/test_engine_and_resources.py tests/test_reliability.py tests/f0_original
```

The three-vehicle CooperScene loader is readiness tooling only; V2V4Real has two physical vehicles and genuine three-vehicle inference has not been executed. F0 is a transparent binary fusion baseline, not trained geometric box fusion. Configured error estimates are not calibrated probabilities. Energy and service times are hypothetical unless explicitly calibrated.
