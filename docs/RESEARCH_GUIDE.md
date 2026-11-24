# Inference diversity: research and onboarding guide

**OCTOPUS-Sim 0.7.1 | For the MobiHoc poster authors and new contributors**

## 1. Where we are heading

Our MobiHoc poster established a useful fact: sharing one inference result changes which observers fail together, even at fixed marginal error. We now want to understand what this means for collaborative computer vision. Two vehicles may see different sides of an occlusion; a correct observation from one can recover a miss by the other (complementarity). Yet a common incorrect prediction can displace a correct local prediction, eliminating a source of alternative evidence (redundancy). These effects are not opposites: we can fuse observations without forcing every agent to adopt the fused result.

The project is vision-centered. We work with a common binary region task to understand the distribution of detection errors, simple adoption policies and the cost of acquiring and delivering evidence. We are not developing a new learned detector, or claiming that joint error is a universal safety metric. Separate inference IDs establish that executions are distinct, not that their errors are statistically independent.

### Three research questions

1. **RQ1 — How does collaborative perception change the distribution of errors across agents?** Look at marginal error, collective risk, class-specific errors, and the spatial query design.
2. **RQ2 — Can selective fusion retain the accuracy benefits of collaboration while limiting shared failures?** Compare simple Boolean policies, a recorded cooperative candidate, and conservative F0. Identify both recoveries and introduced errors; use synthetic dependence to explain why methods differ.
3. **RQ3 — When is retaining inference diversity worth its additional cost?** Examine the compute, transmission, and deadline consequences of preserving or exchanging inference results, without claiming actual hardware energy measurements.

The application determines the objective. If any correct observer can trigger an alert, simultaneous missed detections matter; if every vehicle must make a correct decision, individual and other collective risks matter too. False alarms and inconsistent interpretations have different consequences. Report individual error and R_m = P(K >= m), where K is the number of wrong agents; report joint misses conditional on positive truth and joint false alarms conditional on negative truth. Treat missing/late decisions separately via operational-failure metrics. Do not describe every jointly wrong label as an accident.

## 2. Architecture and analytical mechanism

Local detectors produce frame-level outputs, carrying physical result IDs. In an edge-centric pipeline, detections/features and metadata are uploaded, aligned, and fused; a separate adoption gate decides whether to distribute a correction. Each vehicle retains an available local result. A missed deadline or dropped update does not erase a local result. In F0 the actual implemented fusion is binary ROI-level voting; geometric alignment and learned 3D fusion belong to the upstream perception models, not the simulator.

A physical frame can contain multiple ROI decisions, so one detector invocation may serve several queries. Charge computation and uploads once per physical frame in frame mode. An accepted ROI may trigger a frame-result download; fallback saves no work already performed. The evidence ledger records parent IDs, group declarations, and actual deliveries. Ground truth is private to the evaluator. The simulator runs synthetic predictions or replays recorded candidates; it does not run LiDAR neural inference.

For n receivers adopting k distinct results, let the group sizes be s_1,...,s_k, with sum n. With independent artifact errors of common probability q, the probability-generating function is the product over j of [(1-q) + q z^(s_j)]. Summing coefficients from m through n gives R_m. Identical marginal accuracy and identical job count do not fix the intermediate risk thresholds. Five receivers, three jobs, q=0.10: groups 2+2+1 give R3=2.8%, groups 3+1+1 give R3=10%; both give R5=0.1%. This is an exact conditional model, not an empirical independence assertion. The toy script enumerates all outcomes and matches all 20 points in the analytical plot.

## 3. What the experiments actually establish

**RQ1: recorded two-vehicle outcomes.** V2V4Real provides two sensor-equipped physical vehicles. We use publicly released DMSTrack detection files pinned to `d3b9949499c8e68ea33060873bd1cb95b6d4d323`. The converter creates one GT-centered positive and one eligible offset negative query per synchronized frame (Car, score >=0.30, 7.5 m radius in common KITTI X–Z plane). Ground truth defines the offline task and scores it, not the policy. Nine scenes: 1,993 frames, 3,986 binary queries; mean local error 14.44%, shared late-fusion error 4.87%, both wrong 2.63% versus 4.87%. The different detector pipelines mean this is not an isolated reuse intervention. The balanced task is not natural prevalence, AP/mAP, or a safety measurement. The fixed grid has 23,916 queries; local errors 4.26% and 7.18%, fusion 4.51%, both wrong 1.22% vs 4.51%. Report sensitivity rather than selecting only favorable evidence.

**RQ2: class-specific analysis on 0007.** This is the only complete real event-level trace bundled. Across 221 positive and 221 negative queries, local detectors disagree 192 times (182 positive, 10 negative); 183 of 221 frames contain a disagreement. The cooperative candidate is positive on every disagreement. Joint misses are four for local, shared late fusion and OR. Joint false alarms are 0, 12 and 10 respectively. The disagreement gate exactly equals OR on this trace, and mean error is 3.17% for both vs 22.62% local and 3.62% always cooperative. This is why a simple baseline and class-conditioned analysis are mandatory. Do not generalize this equivalence to other sequences.

**RQ2: controlled multi-agent experiments.** F0 masks stale or high-error sources, groups declared dependent evidence, weights representatives by log((1-q)/q), and accepts only if its surrogate score and support/conflict guards pass. Otherwise it retains separate local results. The score is not a calibrated posterior. With two binary agents and two independent supporters required, F0 cannot change disagreeing labels. With three it can correct a minority but cannot overturn a wrong majority: R2 is unchanged under complete timely delivery and R3 cannot decrease. In the five-agent experiment (common-shock mixture weight 0.75), grouped F0 gives R3=2.48%, R5=0.28%; local gives 14.53%, 0.05%; weighted broadcast gives 2.48%, 2.48%; F0 wrongly assuming independence gives 14.53%, 2.00%. The three-agent expert override can overturn a wrong majority by trusting A, but may spread A's error. At the favorable 18% B/C shock, it essentially broadcasts A. Always-A is a necessary baseline. True and assumed reliability must remain distinct; no event truth may influence policy.

**RQ3: physical accounting.** The E1 resource sweep uses the real 0007 predictions but modeled device/network profiles. The synthetic deadline sweep shows useful corrections disappearing when updates are late; costs remain. At a 21 ms deadline, local labels are available but the fused updates arrive too late under the chosen deterministic profile. This is not a universal hardware threshold. Measure actual compute, radio, payload and deadline behavior before making operational claims.

## 4. Getting started: a runnable tutorial

Use Python 3.10 or newer. On a machine with internet access, clone the public source or extract the complete student archive. Open a terminal in the root containing `pyproject.toml`. On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,yaml,empirical_plot,empirical_download]'
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. The core simulator and analytical toy use only the standard library. Plotting uses NumPy/Matplotlib; no GPU required for replay. Run the exact toy and read its CSV:

```bash
python examples/reuse_toy.py
python examples/reuse_toy.py --q 0.2 --output outputs/reuse_q20.csv
```

Now run the **actual simulator** on three synthetic agents. Three configurations share seed 123, 80 decisions, 0.10 local error and 0.15 common-shock correlation. Only adoption policy or F0 confidence threshold changes. These use decision-granularity synthetic execution (one query per event).

```bash
python -m octopus_sim validate examples/tutorial/toy_f0.json
python -m octopus_sim run examples/tutorial/toy_local.json --output outputs/toy_local
python -m octopus_sim run examples/tutorial/toy_f0.json --output outputs/toy_f0
python -m octopus_sim run examples/tutorial/toy_strict.json --output outputs/toy_strict
```

Expected deterministic smoke results (80 decisions; small sample, not a performance claim):

| Policy | Mean error | R2 | R3 | Incremental J |
|---|---:|---:|---:|---:|
| Local | 10.42% | 2.50% | 0.00% | 19.2000 |
| F0, confidence 0.90 | 5.00% | 2.50% | 1.25% | 20.6172 |
| F0, confidence 0.95 | 10.42% | 2.50% | 0.00% | 20.6112 |

All runs execute 240 local jobs and have zero missed deadlines. Both F0 runs execute 80 edge assessments, accepting 72 and 57 ROIs respectively. Stricter acceptance does not refund work; a zero count does not prove zero risk. Use `summary.json` for metrics, `scores.jsonl` for final agent outcomes and misses, `inferences.jsonl` for physical result IDs, `fusion_decisions.jsonl` for F0 actions, and `operations.jsonl` for actual executed/attempted stages. Compare selected versus realized corrections.

Next vary one quantity at a time: set `min_score_confidence` to 0.99, increase `local_correlation`, or shorten `deadline_ms` while keeping seed fixed. Record the config, number of complete events, mean error, R2/R3, operational failures, energy and bytes. Explain changes using the ledgers before running larger sweeps. For the poster baseline use `examples/two_agents_poster.json` and `scripts/validate_poster.py`. For real 0007 replay (complete student archive required), run:

```bash
python -m octopus_sim run examples/f0/v2v4real_f0.json --output outputs/f0_0007
```

## 5. Reproduce the figures and data

There are two distinct operations: redraw the exact research figures from frozen inputs, or rerun the experiment that generated those inputs. The student bundle includes `figure_reproduction/` with all input tables, traces, and the four plotting/analysis scripts. The public source mirror contains the scripts only: copy the `sources/` directory from the separately shared student archive, not invented rows. See `figure_reproduction/README.md` for every figure's filename, input, and generating script, plus the full simulator commands. From the root:

```bash
python figure_reproduction/regenerate_figures.py
```

This reruns the 0007 class analysis and draws twelve vector PDFs plus the extra disagreement/frame diagnostic. It does not execute detectors or rerun the entire synthetic/scheduler grid. Recomputed numerical tables should match archived inputs; PDF byte hashes may differ with rendering metadata or versions. The full reproduction instructions distinguish those layers and retain seed, frame count, bootstrap and source paths.

## 6. Roadmap and acceptance criteria

**P0 — Evidence and simple baselines.** This is the current research priority, not a new simulator architecture. Regenerate all nine aligned event traces from pinned DMSTrack inputs; compare aggregate counts to the archived CSV; report class-specific joint FN/FP, mean errors, complete frames and paired transitions by sequence. Generalize the 0007 OR/AND/gate analysis without imposing that scene's equality assertion. Include CoBEVT separately when available, always-A when privileging a source, and F0. Calibrate any threshold only on other scenes. Deliver a per-sequence CSV, denominators, source manifest, and uncertainty using sequence or temporal blocks.

**P0 — Spatial interpretation.** Audit X–Z coordinate conventions, score >=0.30, 7.5 m center radius, GT-informed positive and negative offset selection. Recreate the fixed grid and sweep a small predefined radius/score grid (e.g. 2.5/5/7.5 m and .2/.3/.5). Make actual top-down detection plots (rescued miss, shared false alarm, shared miss) with common coordinates, box IDs and query radius. If multiple cars occupy one region, explain what the binary query measures. Do not imply this is 3D IoU/AP or a natural hazard prevalence estimate.

**P1 — Controlled mechanism and robustness.** Reproduce the analytical reuse partitions, three-agent majority and expert cases, and five-agent grouped/misgrouped dependence study. Compare equal and heterogeneous reliability, independent results with common environmental failures, and exact copies. Use paired seeds, true versus assumed q, frame-level uncertainty, and no oracle at decision time. Deliver sweep CSVs, source manifests and tests of invariants. If a simple baseline is as good as F0, explain why instead of hiding it.

**P1 — Physical execution.** Reproduce E1 and deadline sweeps with frame reuse, distinct job counts and source IDs; report incremental energy separately from idle baseline, late work, link loss, bytes and complete events. Keep costs hypothetical until measured. An offline OR or disagreement rule has no measured latency or energy unless the actual DAG for it is implemented. Deliver one clear accuracy/collective-risk/cost plot and a sensitivity table.

**P2 — Later extensions.** Real three-vehicle predictions (CooperScene or Mixed Signals), held-out reliability estimation using visual evidence, measured hardware profiles, and budget-aware acquisition are possible follow-ups. The CooperScene mini validator exists but the archive and actual three-vehicle detector runs have not been obtained in our execution environment. Do not represent a roadside sensor or a fused output as an extra vehicle. No new large dataset or learned controller is needed to answer the present RQs.

### Team deliverables and review

Suggested division: one student leads the empirical geometry and class-specific analysis; the other leads the synthetic/analytical and execution validation. Both review simple baselines, interpretations and reproducibility. Keep experiment seeds, configs, CSVs, logs and provenance together in named output folders. Before sharing third-party derived data or checkpoint weights, check their licenses. Public source is separate from the complete student archive. Prefer falsifiable evidence over attractive plots: if the nine-scene class analysis or provenance audit fails, revise the scientific claim. Our goal is to understand when collaborative visual evidence corrects errors, when shared adoption concentrates errors, and what independent evidence is worth.
