# F0 integration assumption audit (OCTOPUS-Sim v0.7.1)

The following assumptions were revisited against the **uploaded V0.7 source**
during implementation, not reconstructed from a previous summary.

| # | Discovered / revisited assumption | Correction or test | Residual limitation |
| --- | --- | --- | --- |
| 01 | F0 standalone's offline adapter had no actual scheduler integration. | F0 computes through `GraphExecutor`, uses `FIFOResource` and `EnergyLedger`, and emits simulated results. | Upstream `octopus_fusion` offline CLI still labels itself unverified for standalone usage; online integration is separately verified. |
| 02 | A source frame can contain multiple decision ROIs. | One local detector, one upload and one edge F0 job per physical frame; ROI-specific outcome projections. | ROI projection cost currently zero. |
| 03 | Fallback might silently erase spent energy. | Every local compute and attempted upload is always charged, regardless of later decision; F0 compute charged even when all ROIs reject. | More detailed radio wake/tail states not modeled. |
| 04 | The agent already has its local result. | Local-first delivery occurs at detector completion; F0 may replace it after a successful and timely downlink. | If application needs finalization before acting, a control protocol is necessary. |
| 05 | Every ROI rejected does not mean every frame transfer is zero. | If **any** ROI is accepted, one frame result goes to each agent; otherwise no downlink. | Constant per-frame payload assumption, independent of accepted ROI count. |
| 06 | Downlink packet loss should not delete correct local evidence. | Deadline/loss-aware conditional delivery leaves local result IDs unchanged. | No MAC retransmissions. |
| 07 | The underlying V0.7 trace parser requires irrelevant edge outputs for all policies. | Only F0 policy is permitted to use local-only trace candidates. Legacy trace validation unchanged. | Source traces still must provide a full set of local candidate labels. |
| 08 | A shared result ID could be copied across different ROI labels. | F0 result has per-ROI predictions and one parent chain; fallback uses each agent's own local result ID. | No raw 3D perception data/geometry. |
| 09 | F0 could infer true per-agent error rates from the simulator. | Strict external `fusion_f0.error_by_agent` mapping; no access to `world.local_error`, truths or latent shocks. | Estimates are uncalibrated unless separately validated. |
| 10 | F0 could see private source quality for free. | Optional `quality_proxy_by_agent` is an explicit static configured proxy; injection requires opt-in. | Acquisition/telemetry costs are not modeled. |
| 11 | Dependence groups always correspond to different agents. | Explicit group IDs, conflicting members masked; F0's original lineage-aware guard is preserved. | Group accuracy depends on author-supplied lineage information. |
| 12 | Collective-error risk can be bounded from the sigmoid score. | Source F0's surrogate score is **not** a calibrated probability. External risk bounds are disabled until a validated estimator exists. | No claimed formal joint-risk guarantees. |
| 13 | Calling F0 after uploads is costless. | Dedicated edge compute model `f0_decider` with configured duration/power; no duplicate ROI compute jobs. | Cost is hypothetical, not benchmarked hardware. |
| 14 | A retracted accepted fusion must return every agent to the same local result. | Each agent retains **its own** local result, with its own result ID, even after failure to receive an accepted fusion. | Mixed per-agent outcomes under partial downlink loss are possible and reported. |
| 15 | V0.7 pre-inference candidate estimates automatically understand post-inference branches. | Explicit incompatibility of `coordination.enabled` with F0 policy; runtime uses a conditional graph. | Future P1 estimator must support branch probabilities. |
| 16 | Two agents plus a weighted vote always yield fault correction. | 221-frame real replay shows F0's two-support rule does **not** change local-only predictions. | Need additional independent evidence or a different verified decision mechanism. |
| 17 | A lower aggregate energy under F0 proves inference reuse savings. | Separate local, cooperative and F0 executed DAGs and profile IDs are recorded. | Not a controlled same-model energy experiment; F0 is a cheap post-processing method. |
| 18 | Label leakage can be ruled out by output inspection alone. | Flip recorded ground truth while preserving candidates; F0 actions, result values and costs remain unchanged. | Custom third-party Python plugins are trusted, not sandboxed. |
| 19 | Source F0 offline and online paths may disagree because of frame projection. | Verified 442 event-level F0 actions and decisions agree between actual V0.7 local-only exports and online F0 execution. | Assumes same timing-independent static estimators and nonbinding F0 age cutoff. |
| 20 | Deadline misses can be interpreted as normal prediction errors. | Existing scoring keeps missing results distinct; F0 decision audits separate intended share from on-time applied overrides. | Delayed finalization beyond deadline is excluded from scored result. |

A reproducible release includes original upstream F0 tests and new integrated
regression tests for 2/3/5 agents, packet loss, energy, latency, masking,
partial acceptance, frame-level caching, error-label independence, and trace
provenance. No hardware calibration, trained fusion network or multi-scene
V2V4Real generalization is claimed.
