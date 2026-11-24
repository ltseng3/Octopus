"""Strict, explicit JSONL reader for offline F0 studies.

Truth is separated into an evaluation-only LabeledCase. The fuser receives
only FusionObservation and ReliabilityEstimate. No hidden ground truth is
inferred from observed predictions.
"""
from __future__ import annotations
from pathlib import Path
import json
from .contracts import AgentEvidence, FusionObservation, ReliabilityEstimate
from .evaluate import LabeledCase


def load_labeled_cases(path: str | Path) -> tuple[tuple[LabeledCase, ReliabilityEstimate], ...]:
    rows=[]
    identities=set()
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        if not line.strip():
            continue
        try:
            data=json.loads(line)
            evidence=[]
            q=[]
            for row in data['evidence']:
                item=AgentEvidence(
                    agent_id=row['agent_id'], prediction=row['prediction'],
                    result_id=row['result_id'], dependence_group=row['dependence_group'],
                    produced_at_ms=row['produced_at_ms'],
                    quality_proxy=row.get('quality_proxy'),
                )
                evidence.append(item)
                q.append((item.agent_id,row['assumed_error_probability']))
            obs=FusionObservation(
                event_id=data['event_id'],frame_id=data['frame_id'],
                as_of_ms=data['as_of_ms'],evidence=tuple(evidence),
                prior_positive=data.get('prior_positive',.5),
            )
            if obs.event_id in identities:
                raise ValueError('Repeated event_id')
            identities.add(obs.event_id)
            estimate=ReliabilityEstimate(tuple(q), estimator_id='externally_supplied_assumption')
            rows.append((LabeledCase(obs,data['truth']),estimate))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Invalid F0 JSONL row {number}: {exc}') from exc
    if not rows:
        raise ValueError('No labeled cases found')
    return tuple(rows)


def write_labeled_cases(path: str | Path,
                        paired: tuple[tuple[LabeledCase, ReliabilityEstimate], ...]) -> None:
    """For toy/data preparation only; avoid publishing licensed source outcomes."""
    with Path(path).open('w',encoding='utf-8') as out:
        for case,reliability in paired:
            estimated=dict(reliability.error_by_agent)
            o=case.observation
            obj={
                'event_id':o.event_id,'frame_id':o.frame_id,
                'as_of_ms':o.as_of_ms,'prior_positive':o.prior_positive,
                'truth':case.truth,
                'evidence':[{**{
                    'agent_id':e.agent_id,'prediction':e.prediction,
                    'result_id':e.result_id,'dependence_group':e.dependence_group,
                    'produced_at_ms':e.produced_at_ms,
                    'quality_proxy':e.quality_proxy,
                    'assumed_error_probability':estimated[e.agent_id],
                }} for e in o.evidence],
            }
            out.write(json.dumps(obj,sort_keys=True)+'\n')
