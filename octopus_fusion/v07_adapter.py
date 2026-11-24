"""Offline **candidate** adapter for documented OCTOPUS-Sim V0.7 output layout.

The V0.7 source archive is not available in this environment, so this reader is
validated with representative JSONL fixtures only; integration with the
original V0.7 code is UNVERIFIED. It fails closed on missing local artifacts.

The adapter only uses scores for event identity and evaluation truth. ALL
policy-facing predictions come from completed local inference artifacts.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from collections.abc import Mapping
from .contracts import AgentEvidence, FusionObservation, ReliabilityEstimate
from .evaluate import LabeledCase


def _read_jsonl(path: str | Path) -> list[dict]:
    entries=[]
    for i,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        if line.strip():
            try:
                item=json.loads(line)
                if not isinstance(item,dict):
                    raise ValueError('JSON object required')
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{path} line {i}: {exc}') from exc
            entries.append(item)
    return entries


def load_v07_local_outputs(inferences: str | Path, scores: str | Path,
                           assumed_error_by_agent: Mapping[str, float],
                           *, decision_delay_ms: float = 0.0) -> tuple[tuple[LabeledCase, ReliabilityEstimate], ...]:
    """Assemble a per-ROI observation from recorded **local** detector artifacts.

    No co-simulation or actual uplink is modeled; completion times are local
    production times and `decision_delay_ms` is an assumed offline availability
    delay applied to the latest detector. Separate source lineages are assumed
    distinct unless explicit common `observation_ids` show an identical origin.
    """
    from .contracts import finite
    finite('decision_delay_ms',decision_delay_ms)
    if decision_delay_ms < 0:
        raise ValueError('decision_delay_ms cannot be negative')
    artifact_by_event: dict[str, dict[str, dict]] = defaultdict(dict)
    for rec in _read_jsonl(inferences):
        if rec.get('location') != 'local':
            continue
        agent=rec.get('source_agent')
        if not isinstance(agent,str) or not agent:
            raise ValueError('local inference record missing source_agent')
        preds=rec.get('predictions_by_event_id')
        if preds is None and rec.get('event_id') is not None:
            preds={rec['event_id']:rec['prediction']}
        if not isinstance(preds,dict) or not preds:
            raise ValueError('local inference record missing per-event prediction map')
        for event, y in preds.items():
            if agent in artifact_by_event[event]:
                raise ValueError(f'duplicate local inference for {agent} on {event}')
            artifact_by_event[event][agent]={**rec,'_prediction_for_event':y}
    cases=[]
    seen=set()
    for score in _read_jsonl(scores):
        event=score['event_id']
        if event in seen:
            raise ValueError('duplicate event score')
        seen.add(event)
        frame=score.get('frame_id')
        if not isinstance(frame,str) or not frame:
            raise ValueError('frame_id missing; cannot validate physical observation identity')
        if event not in artifact_by_event:
            raise ValueError(f'no executed local inference available for {event}')
        agents=tuple(sorted(artifact_by_event[event]))
        if set(agents)!=set(assumed_error_by_agent):
            raise ValueError(f'executed local results do not cover assumed agents for {event}')
        latest=max(float(r['created_at_ms']) for r in artifact_by_event[event].values())
        evidence=[]
        for agent in agents:
            result=artifact_by_event[event][agent]
            if result.get('frame_id') not in (None,frame):
                raise ValueError('inference frame ID does not match evaluation score frame ID')
            # Evidence lineage follows real result identity. If two agent-local
            # computations share an exact observation lineage ID, group them.
            obsids=tuple(sorted(result.get('observation_ids') or ()))
            origin=('origin:'+'|'.join(obsids)) if obsids else 'result:'+result['result_id']
            evidence.append(AgentEvidence(
                agent_id=agent, prediction=result['_prediction_for_event'],
                result_id=result['result_id'], dependence_group=origin,
                produced_at_ms=float(result['created_at_ms']),
            ))
        obs=FusionObservation(event_id=event,frame_id=frame,
                              as_of_ms=latest+decision_delay_ms,
                              evidence=tuple(evidence))
        estimated=ReliabilityEstimate(tuple((a,assumed_error_by_agent[a]) for a in agents),
                                       estimator_id='user_supplied_unvalidated_static_rates')
        cases.append((LabeledCase(obs, score['ground_truth']),estimated))
    if not cases:
        raise ValueError('no scored cases')
    return tuple(cases)
