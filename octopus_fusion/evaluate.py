"""Evaluation is separate from fusion to prevent oracle leakage.

Only this module accepts true binary labels. Its output is descriptive metrics,
not a theorem about operational risk or physical safety.
"""
from __future__ import annotations
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from .contracts import FusionObservation, FusionDecision


@dataclass(frozen=True)
class LabeledCase:
    observation: FusionObservation
    truth: int

    def __post_init__(self) -> None:
        if type(self.truth) is not int or self.truth not in (0, 1):
            raise ValueError('truth must be binary')


@dataclass(frozen=True)
class Evaluation:
    cases: int
    average_agent_error: float
    all_agent_error_rate: float
    shared_coverage: float
    accepted_fusion_error_rate: float | None
    fallback_rate: float
    mask_and_fuse_rate: float


def evaluate(cases: Iterable[LabeledCase], predictions: Mapping[str, FusionDecision]) -> Evaluation:
    errors, joint, shared, wrong_accepted, masked_shared, total = 0.0, 0, 0, 0, 0, 0
    seen = set()
    for case in cases:
        obs, truth = case.observation, case.truth
        if obs.event_id in seen:
            raise ValueError('Duplicate event ID in evaluation')
        seen.add(obs.event_id)
        if obs.event_id not in predictions:
            raise ValueError(f'Missing fusion decision for {obs.event_id}')
        result = predictions[obs.event_id]
        if result.event_id != obs.event_id or result.frame_id != obs.frame_id:
            raise ValueError('Evaluation case/result identity mismatch')
        agent_ids = {e.agent_id for e in obs.evidence}
        answers = dict(result.per_agent_decisions)
        if len(answers) != len(result.per_agent_decisions) or set(answers) != agent_ids:
            raise ValueError('Decisions do not match agent set')
        if not result.shared and answers != {e.agent_id: e.prediction for e in obs.evidence}:
            raise ValueError('KEEP_LOCAL must preserve each local answer')
        wrong = sum(y != truth for y in answers.values())
        errors += wrong / len(answers)
        joint += (wrong == len(answers))
        total += 1
        if result.shared:
            if set(answers.values()) != {result.fused_prediction}:
                raise ValueError('Shared decision must actually share the fused label')
            shared += 1
            wrong_accepted += (result.fused_prediction != truth)
            masked_shared += (result.action.value == 'MASK_AND_FUSE')
    if set(predictions) != seen or total == 0:
        raise ValueError('Must evaluate a nonempty, exactly matched case set')
    return Evaluation(total, errors / total, joint / total, shared / total,
                      wrong_accepted/shared if shared else None,
                      1 - shared/total, masked_shared/total)
