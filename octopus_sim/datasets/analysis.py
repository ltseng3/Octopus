"""Scene-clustered paired empirical policy analysis.

Do not interpret per-event Wilson intervals as independent-observation CIs
when many ROIs and frames come from the same physical driving scene.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import random
from collections import defaultdict


class EmpiricalAnalysisError(ValueError):
    pass


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise EmpiricalAnalysisError(f'Missing file: {path}')
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def _quantile(values: list[float], p: float) -> float:
    values = sorted(values)
    ix = p * (len(values)-1)
    lo, hi = int(ix), min(len(values)-1, int(ix)+1)
    return values[lo] + (ix-lo) * (values[hi]-values[lo])


def _cluster_mean(groups: dict, sampled_scenes: list[str]) -> float:
    count = sum(len(groups[s]) for s in sampled_scenes)
    return sum(sum(groups[s]) for s in sampled_scenes)/count


def analyze_empirical(comparison_dir: str | Path, provenance_path: str | Path, *,
                      baseline: str | None = None, iterations: int = 2000, seed: int = 42,
                      output_file: str | Path | None = None) -> dict:
    if not isinstance(iterations, int) or not 100 <= iterations <= 100000:
        raise EmpiricalAnalysisError('iterations must be 100..100000')
    comparison = Path(comparison_dir)
    provenance_path = Path(provenance_path)
    provenance = _read_jsonl(provenance_path)
    eid_to_scene: dict[str,str] = {}
    for r in provenance:
        eid, scene = r['event_id'], r['scene_id']
        if eid in eid_to_scene:
            raise EmpiricalAnalysisError(f'duplicate provenance event {eid}')
        eid_to_scene[eid] = scene
    if not eid_to_scene:
        raise EmpiricalAnalysisError('no provenance events')
    policy_dirs = sorted(p for p in comparison.iterdir()
                         if p.is_dir() and (p/'scores.jsonl').is_file())
    if len(policy_dirs) < 2:
        raise EmpiricalAnalysisError('need >=2 policy result directories')
    policies = {}
    event_truth = {}
    for directory in policy_dirs:
        scores = _read_jsonl(directory/'scores.jsonl')
        record = {}
        for s in scores:
            event_id = s['event_id']
            if event_id in record:
                raise EmpiricalAnalysisError(f'duplicate scored event {event_id} in {directory}')
            if event_id not in eid_to_scene:
                raise EmpiricalAnalysisError(f'missing provenance {event_id}')
            if event_id in event_truth and event_truth[event_id] != s['ground_truth']:
                raise EmpiricalAnalysisError(f'ground truth mismatch for {event_id}')
            event_truth[event_id] = s['ground_truth']
            record[event_id] = s
        if set(record) != set(eid_to_scene):
            raise EmpiricalAnalysisError(f'nonmatching event IDs for {directory.name}')
        policies[directory.name] = record
    if baseline is None:
        baseline = policy_dirs[0].name
    if baseline not in policies:
        raise EmpiricalAnalysisError(f'baseline {baseline} not found')
    scenes = sorted(set(eid_to_scene.values()))
    cluster_counts = {scene: sum(s==scene for s in eid_to_scene.values()) for scene in scenes}
    out = {'provenance_sha256': hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
           'baseline': baseline, 'policies': {}, 'paired_differences': {},
           'scene_count': len(scenes), 'event_count': len(eid_to_scene),
           'events_per_scene': cluster_counts,
           'uncertainty_method': 'paired scene-cluster bootstrap, resampling whole scenes with replacement',
           'bootstrap_iterations': iterations, 'seed': seed}
    rng = random.Random(seed)
    # Common resampled scenes for every policy and every paired contrast.
    draws = [[rng.choice(scenes) for _ in scenes] for _ in range(iterations)] if len(scenes) >= 2 else []
    group = {}
    for name, scored in policies.items():
        group[name] = defaultdict(list)
        for eid, scene in eid_to_scene.items():
            group[name][scene].append(int(scored[eid]['all_wrong']))
        estimate = _cluster_mean(group[name], scenes)
        bootstrap = [_cluster_mean(group[name], d) for d in draws]
        out['policies'][name] = {
            'joint_all_error_rate': estimate,
            'scene_clustered_95_ci': [_quantile(bootstrap,0.025),_quantile(bootstrap,0.975)] if bootstrap else None,
            'completed_fraction': sum(bool(s['complete']) for s in scored.values())/len(scored),
            'joint_operational_fail_rate': sum(bool(s['all_operational_fail']) for s in scored.values())/len(scored),
        }
    for name in policies:
        if name == baseline:
            continue
        diffs = {s:[a-b for a,b in zip(group[name][s],group[baseline][s])]
                 for s in scenes}
        # Preserve matching event ordering: each group comes from the same provenance sequence.
        point = _cluster_mean(diffs, scenes)
        bootstrap = [_cluster_mean(diffs,d) for d in draws]
        out['paired_differences'][name] = {
            'vs': baseline, 'difference_joint_all_error_rate': point,
            'scene_clustered_95_ci': [_quantile(bootstrap,.025),_quantile(bootstrap,.975)] if bootstrap else None,
        }
    if len(scenes) < 2:
        out['warning'] = 'Only one independent scene: no meaningful scene-clustered confidence intervals.'
    if output_file is not None:
        dest = Path(output_file)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2, sort_keys=True)+'\n',encoding='utf-8')
    return out
