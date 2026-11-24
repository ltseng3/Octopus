"""Outcome metrics and interpretable uncertainty intervals."""
from __future__ import annotations
from math import sqrt
from statistics import mean
from typing import Any


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials == 0:
        return [0.0, 1.0]
    p = successes / trials
    denominator = 1 + z*z / trials
    center = (p + z*z/(2*trials)) / denominator
    margin = z * sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / denominator
    return [0.0 if successes == 0 else max(0.0, center-margin),
            1.0 if successes == trials else min(1.0, center+margin)]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    pos = p * (len(sorted_values)-1)
    lo = int(pos)
    hi = min(lo+1, len(sorted_values)-1)
    return sorted_values[lo] + (sorted_values[hi]-sorted_values[lo])*(pos-lo)


def summarize(scores: list[dict[str, Any]], resources: dict,
              plans: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(scores)
    if n == 0:
        raise ValueError('No completed decision events')
    agent_count = len(scores[0]['wrong'])
    joint = sum(s['all_wrong'] for s in scores)
    any_wrong = sum(s['any_wrong'] for s in scores)
    complete = sum(s['complete'] for s in scores)
    operational_joint = sum(s['all_operational_fail'] for s in scores)
    misses = sum(sum(s['missing'].values()) for s in scores)
    errors = sum(sum(v is True for v in s['wrong'].values()) for s in scores)
    latencies = [v for s in scores for v in s['latency_ms'].values() if v is not None]
    shared_group_events = sum(p.get('execution_mode') == 'cooperative_late_fusion' or
                              any(len(g['consumers']) > 1 for g in p['groups']) for p in plans)
    tail_errors = {str(m): sum(s['error_count'] >= m for s in scores) / n
                   for m in range(1, agent_count + 1)}
    tail_failures = {str(m): sum(s['operational_failure_count'] >= m for s in scores) / n
                     for m in range(1, agent_count + 1)}
    per_agent = {a: {
        'error_rate_all_events': sum(s['wrong'][a] is True for s in scores)/n,
        'deadline_miss_rate': sum(s['missing'][a] for s in scores)/n,
    } for a in scores[0]['wrong']}
    result = {
        'episodes': n, 'agents': agent_count,
        'tail_error_rates_by_m': tail_errors,
        'tail_operational_failure_rates_by_m': tail_failures,
        'per_agent_rates': per_agent,
        'joint_all_error_rate': joint/n,
        'joint_all_error_wilson_95': wilson(joint, n),
        'joint_all_error_count': joint,
        'any_error_rate': any_wrong/n,
        'marginal_error_rate_all_events': errors/(n*agent_count),
        'complete_event_rate': complete/n,
        'agent_deadline_miss_rate': misses/(n*agent_count),
        'all_operational_failure_rate': operational_joint/n,
        'mean_delivered_latency_ms': mean(latencies) if latencies else None,
        'p95_delivered_latency_ms': percentile(latencies, .95),
        'events_with_shared_inference_rate': shared_group_events/n,
        'predicted_infeasible_event_rate': sum(p['predicted_infeasible'] for p in plans)/n,
    }
    result.update(resources)
    return result


def paired_joint_risk_difference(a: list[dict], b: list[dict]) -> dict[str, float]:
    if len(a) != len(b) or [s['event_id'] for s in a] != [s['event_id'] for s in b]:
        raise ValueError('Policies must be evaluated over the same event IDs')
    n = len(a)
    differences = [int(x['all_wrong']) - int(y['all_wrong']) for x, y in zip(a,b)]
    delta = sum(differences) / n
    variance = sum((x - delta)**2 for x in differences) / (n-1) if n > 1 else 0.0
    margin = 1.96 * sqrt(variance/n)
    return {'difference_a_minus_b': delta, 'approx_95_ci': [delta-margin,delta+margin],
            'paired_events': n}
