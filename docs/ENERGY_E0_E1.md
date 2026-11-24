# OCTOPUS-Sim E0/E1 — Event-driven energy accounting

**Release:** 0.5.0. **Status:** reproducible simulation prototype; no energy profile in this repository has been calibrated on the V2V4Real vehicles, their radios, or their edge infrastructure. The V2V4Real inference predictions are from actual public model exports; energy comes solely from *illustrative configured values*. Do not characterize the resulting joules as measured power consumption or physical vehicle safety effects.

## Quick start (CPU only)

From repository root, install optional plotting/testing dependencies:

```bash
python -m pip install -e '.[plot,test]'
python -m octopus_sim validate examples/energy/three_agent_per_bit.json
python -m octopus_sim run examples/energy/three_agent_per_bit.json --no-events --output outputs/e1_three
python -m octopus_sim run examples/energy/three_agent_radio_state.json --no-events --output outputs/e1_radio_state
python -m octopus_sim run examples/energy/v2v4real_0007_energy.json --no-events --output outputs/e1_v2v4real
python scripts/run_energy_e1.py --output samples/energy_e1
python -m pytest -q
```

`summary.json` includes `energy_by_device_j`, `energy_compute_by_model_j`, `energy_radio_by_transfer_role_j`, system totals, the common measurement window and `energy_post_window_incremental_j`. `scenario_resolved.json` and `manifest.json` preserve the energy configuration and provenance of the replay trace.

## Physical boundary and units

Energy is charged to the **two-or-more agent devices and one logical edge server**. The shared link is not an extra power-consuming device: the endpoints pay transmission and reception costs. This excludes unmodeled routers, base stations, cooling infrastructure, sensor acquisition and battery voltage dynamics. No DVFS, thermal behavior, fixed radio wake-up/tail cost or MAC retransmissions are simulated. `packet_loss` is a single modeled transmission attempt, not retransmission logic.

Time is represented internally in **integer microseconds**. Power is in watts, energy in joules, payload in bytes, link speed in Mbps. Conversions always use `1 second = 1_000_000 microseconds` and `8 bits = 1 byte`.

For device `d`, over the **common comparison window** `[0,H]`:

```
E_baseline(d) = P_baseline(d) * H
E_compute(d)  = SUM_over_executed_jobs  (P_compute_incremental(model,d) * actual_service_duration)
                OR fixed incremental joules per executed model invocation
E_radio(d)    = SUM_over_transfers (bits * energy_per_bit)  [per_bit mode]
                OR SUM (incremental_TX_or_RX_power * actual_link_serialization_time) [state mode]
E_total(d)    = E_baseline(d) + E_compute(d) + E_radio(d)
```

* **Baseline:** charged once per physical device over `H`, including radio idle power if any. The edge is assumed powered on even under local-only policies; this is intentional, not an edge-power-management experiment.
* **Compute:** charged only when the queued job actually starts running; *never* for compute queue waiting. A model-level `model_incremental_w` overrides the device default. Alternatively, `model_energy_j` provides a fixed *incremental* joule value per completed operation. Specify one or the other for each model, not both.
* **Communication:** charged to *both endpoints*. Upload TX is on the agent and RX on the edge; result TX is on the edge and RX on each receiving agent. Physical-link serialization is separated from propagation delay; no active radio power during propagation is assumed. Even lost packets consume modeled transmission energy.
* **Overlapping operations:** compute and radio incremental powers are additive; device baseline is **not multiplied by the worker count**. This linear-power approximation is configurable, not a claim about accelerator concurrency. An external nonlinear power model can later replace the ledger's incremental mapping.
* **Uncompleted or late work:** the simulation continues draining the queue after deadlines. Incremental energy used after `H` is explicitly retained in `energy_post_window_incremental_j`. There is no free energy simply because the answer arrived too late.

### Two energy totals — both have precise semantics

`energy_window_total_j = E_baseline(H) + E_incremental_during_[0,H]`

`energy_total_j = E_baseline(H) + E_incremental_during_[0,H] + E_incremental_after_H`

`energy_total_j` is the primary **fixed-baseline plus all-work** budget. It **does not include post-window idle power** (which is generally policy-dependent because different policies may drain differently). The post-window incremental portion is separately disclosed, allowing authors to construct a different system-level horizon if needed. Do not mix `energy_window_total_j` with `energy_total_j` in a comparison without saying so.

The default common horizon is **last event release + configured decision deadline**, identical across policies with the same scenario and trace. Set `energy.measurement_window_ms` explicitly to override this; it must be strictly positive. This prevents a fast policy from paying less baseline energy just because it finishes before the deadline.

### Radio modes

| Mode | Model | Useful for | Limitation |
| --- | --- | --- | --- |
| `per_bit` | `8 * bytes * tx_j_per_bit` and `8 * bytes * rx_j_per_bit` | Cheap, portable link accounting | Does not vary with throughput for fixed payload |
| `state` | incremental TX/RX watts × link serialization time | Evaluate link-speed/power dependence | Requires `network_channels: 1` in v0.5, a single shared-link/half-duplex proxy |

The two modes are **mutually exclusive per device**; mixing energy/bit and active-radio-watts for one device is rejected. Device A can use per-bit while device B uses state if the channel constraint is satisfied. Radio wake-up, tail, sleep and retransmissions are future fidelity levels; none is silently estimated.

## Configuration

This is a representative fragment copied from the three-agent example (all values **hypothetical**):

```json
{
  "energy": {
    "enabled": true,
    "calibration": "hypothetical",
    "calibration_source": "Illustrative E1 profile only; NOT hardware measurements",
    "measurement_window_ms": null,
    "device_profiles": {
      "agent": {
        "baseline_w": 1.6,
        "compute_incremental_w": 4.0,
        "radio": {
          "mode": "per_bit",
          "tx_j_per_bit": 1.2e-7,
          "rx_j_per_bit": 8e-8,
          "tx_incremental_w": 0,
          "rx_incremental_w": 0
        }
      },
      "edge": {
        "baseline_w": 17,
        "compute_incremental_w": 9,
        "radio": {
          "mode": "per_bit",
          "tx_j_per_bit": 5e-8,
          "rx_j_per_bit": 5e-8,
          "tx_incremental_w": 0,
          "rx_incremental_w": 0
        }
      }
    },
    "agent_overrides": {},
    "model_incremental_w": {"local_model": 4.0, "late_fusion": 12.0},
    "model_energy_j": {}
  }
}
```

The `agent_overrides` mapping assigns full device profiles by agent ID, e.g. `"ego": {"baseline_w": ..., "compute_incremental_w": ..., "radio": {...}}`. Model overrides refer to actual IDs in `resources.model_profiles` or the two built-in model IDs. Unknown agents/models, invalid units, negative/NaN values, incompatible radio modes, and missing explicit device profiles are rejected.

For a **calibrated** profile, use `calibration: "measured"` and a detailed `calibration_source`. This is user-declared metadata: OCTOPUS-Sim verifies that the source is present but cannot independently validate the underlying laboratory measurements.

## E1 experiments and results

Generate CSVs and vector/PNG figures with `scripts/run_energy_e1.py`. No GPU or original V2V4Real LiDAR data are needed.

| Scenario | Allocation | Joint wrong | Incremental energy | Total energy (fixed baseline + work) |
| --- | --- | ---: | ---: | ---: |
| 3-agent synthetic, 80 common events | Separate local | 0 / 80 | 19.200 J | 885.750 J |
| 3-agent synthetic, 80 common events | Cooperative fusion | 3 / 80 | 27.580 J | 894.130 J |
| V2V4Real source-verified sequence 0007, 442 binary decisions | Separate local | 4 / 442 | 35.360 J | 2265.440 J |
| Same V2V4Real replay | Cooperative fusion | 16 / 442 | 53.793 J | 2283.873 J |

V0.6 runs **one detector/fusion DAG per recorded frame**, not once per ROI.
Previous per-ROI E1 energy tables are archived in `samples/legacy_v051/`.

These results are **not performance claims about V2V4Real hardware**. Local and fusion model energy parameters are hypothetical, and the policies are counterfactual. The 3-agent baseline happens to have no *all-three-wrong* events among 80 simulations, which is not an estimate that the actual risk is zero. A statistically rigorous paper should use many independent scenes/episodes, confidence intervals and calibrated power measurements.

Outputs:

* `samples/energy_e1/policy_energy_comparison.csv`: per-policy accuracy, resource and energy accounting.
* `samples/energy_e1/energy_sensitivity.csv`: radio power × fusion-power perturbations.
* `samples/energy_e1/01_risk_vs_incremental_energy.{png,pdf}`: risk–energy plot, separate curves for two distinct tasks.
* `samples/energy_e1/02_{scenario}_energy_breakdown.{png,pdf}`: compute vs communication energy.
* `samples/energy_e1/03_v2v4real_energy_sensitivity.{png,pdf}`: illustrative parameter uncertainty.
* `samples/energy_e1/PROVENANCE.json`: calibration and source conditions.

Note the resource conclusion in these particular scenarios: cooperative late fusion requires all agents to run their local detectors, then uploads, an edge fusion job, and result delivery. It is therefore **more expensive** than local-only under this accounting. The benchmark must not presume inference sharing saves resources. Policies that share an edge-computed result without repeating local work may show savings; that is a different execution DAG and must be evaluated on its own terms.

## API and maintainability

1. The existing `FIFOResource.request(duration_us, on_complete)` remains valid; `on_start` is an optional keyword callback. It fires when a worker actually begins service, not upon job submission.
2. `GraphExecutor` observes compute and transfer *start and finish* and emits accounting tokens. Its prediction and provenance logic is unchanged.
3. `EnergyLedger` owns the energy terms, a fixed measurement horizon, and per-device and per-model reports. It does not access ground truth or alter policies.
4. Energy is **off by default**, so old scenarios run without new required configuration, scheduler energy callbacks, or changes to inference outcomes.
5. New energy-report fields are emitted only when `energy.enabled` is true. Existing output fields stay stable.

The ledger can be replaced with more detailed power laws without changing world events, inference identities, or the policy API. A **resource-aware `CoordinatorView`** is a future P1 feature; current policies do not observe true power consumption or use energy to choose allocations.

## Test plan and acceptance criteria

`python -m pytest -q` runs the previous 86-test suite and new E0/E1 tests. Targeted tests cover: exact compute joules, per-agent accounting, two radio modes, queueing, loss, concurrency, post-deadline work, fixed horizons, five-agent fusion, both compute profile styles, config invalidity, determinism, and unchanged reliability with energy enabled.

Before external public release, conduct clean CI runs for supported Python versions; independently review energy math and upstream dataset licenses; measure and document actual hardware/software workloads; and verify prediction replays against their source checksums.

## Design precedents

- EdgeSimPy linear/squared server power functions (official source): https://github.com/EdgeSimPy/EdgeSimPy
- LEAF energy simulation and `PowerMeter`: https://github.com/dos-group/leaf
- PureEdgeSim `EnergyModelComputingNode` and network energy: https://github.com/CharafeddineMechalikh/PureEdgeSim/blob/master/docs/models.md
- ns-3 energy framework: https://www.nsnam.org/docs/models/html/energy.html

The implementation is original, small, and self-contained; it does not vendor third-party simulator code.
