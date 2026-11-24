"""Immutable, explicitly limited F0 fusion inputs and decisions.

No ground truth, future outcomes, energy accounting, or simulator World object
appears in the fusion input contracts. Evidence is scoped to one decision ROI,
while result_id identifies a *physical frame-level inference result*.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math


def finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')


def probability(name: str, value: float, *, strict: bool = False) -> None:
    finite(name, value)
    if (not 0 <= value <= 1) or (strict and value in (0, 1)):
        raise ValueError(f'{name} must be a probability'+(' strictly between 0 and 1' if strict else ''))


class Action(str, Enum):
    SHARE_FUSED = 'SHARE_FUSED'
    MASK_AND_FUSE = 'MASK_AND_FUSE'
    KEEP_LOCAL = 'KEEP_LOCAL'


class MaskReason(str, Enum):
    TOO_UNRELIABLE = 'too_unreliable'
    STALE = 'stale'
    DEPENDENCE_CONFLICT = 'dependence_conflict'


@dataclass(frozen=True)
class AgentEvidence:
    agent_id: str
    prediction: int  # must be 0/1 (not a probability)
    result_id: str   # same physical result may project to different ROI labels
    dependence_group: str  # independent evidence lineage, NOT the agent count
    produced_at_ms: float
    quality_proxy: float | None = None  # ex-ante public observable; not correctness

    def __post_init__(self) -> None:
        for k in ('agent_id', 'result_id', 'dependence_group'):
            if not isinstance(getattr(self, k), str) or not getattr(self, k).strip():
                raise ValueError(f'{k} must be a nonempty string')
        if type(self.prediction) is not int or self.prediction not in (0, 1):
            raise ValueError('prediction must be an integer 0 or 1')
        finite('produced_at_ms', self.produced_at_ms)
        if self.produced_at_ms < 0:
            raise ValueError('produced_at_ms must be nonnegative')
        if self.quality_proxy is not None:
            probability('quality_proxy', self.quality_proxy)


@dataclass(frozen=True)
class FusionObservation:
    event_id: str
    frame_id: str
    as_of_ms: float
    evidence: tuple[AgentEvidence, ...]
    prior_positive: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError('event_id must be nonempty')
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise ValueError('frame_id must be nonempty')
        finite('as_of_ms', self.as_of_ms)
        if self.as_of_ms < 0:
            raise ValueError('as_of_ms must be nonnegative')
        probability('prior_positive', self.prior_positive, strict=True)
        if len(self.evidence) < 2 or len({e.agent_id for e in self.evidence}) != len(self.evidence):
            raise ValueError('Require >=2 evidence entries with unique agent identities')
        # One physical result cannot simultaneously claim two distinct lineages.
        lineage: dict[str, str] = {}
        predicted: dict[str, int] = {}
        for e in self.evidence:
            if e.produced_at_ms > self.as_of_ms:
                raise ValueError('Evidence cannot originate in the future')
            if e.result_id in lineage and lineage[e.result_id] != e.dependence_group:
                raise ValueError('Same physical result cannot claim independent dependence groups')
            if e.result_id in predicted and predicted[e.result_id] != e.prediction:
                raise ValueError('Same physical result cannot contradict itself within one ROI')
            lineage[e.result_id] = e.dependence_group
            predicted[e.result_id] = e.prediction


@dataclass(frozen=True)
class ReliabilityEstimate:
    """Error probabilities indexed by agent; estimates, never hidden correctness."""
    error_by_agent: tuple[tuple[str, float], ...]
    estimator_id: str = 'external'

    def __post_init__(self) -> None:
        names = [name for name, _ in self.error_by_agent]
        if not names or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
            raise ValueError('error_by_agent requires unique nonempty agent IDs')
        for name, q in self.error_by_agent:
            probability(f'error_probability[{name}]', q, strict=True)
        if not isinstance(self.estimator_id, str) or not self.estimator_id:
            raise ValueError('estimator_id must be nonempty')


@dataclass(frozen=True)
class RiskBounds:
    """Optional EXTERNAL risk-estimator upper bounds, not inferred from vote confidence."""
    shared_error_upper: float | None = None
    local_joint_error_lower: float | None = None
    estimator_id: str = 'unspecified'

    def __post_init__(self) -> None:
        for name in ('shared_error_upper', 'local_joint_error_lower'):
            value = getattr(self, name)
            if value is not None:
                probability(name, value)


@dataclass(frozen=True)
class FusionConfig:
    max_source_error: float = 0.25
    max_age_ms: float = 120.0
    dependence_discount: float = 1.0
    min_independent_groups: int = 2
    min_supporting_groups: int = 2
    max_conflict_fraction: float = 0.40
    min_score_confidence: float = 0.90
    require_risk_bounds: bool = False
    max_shared_error_upper: float | None = None
    max_risk_increase_over_local: float | None = None

    def __post_init__(self) -> None:
        probability('max_source_error', self.max_source_error, strict=True)
        if self.max_source_error > .5:
            raise ValueError('max_source_error must not exceed 0.5')
        finite('max_age_ms', self.max_age_ms)
        if not 0 <= self.max_age_ms:
            raise ValueError('max_age_ms must be nonnegative')
        probability('dependence_discount', self.dependence_discount)
        if type(self.min_independent_groups) is not int or self.min_independent_groups < 2:
            raise ValueError('min_independent_groups must be >=2')
        if type(self.min_supporting_groups) is not int or self.min_supporting_groups < 2:
            raise ValueError('min_supporting_groups must be >=2')
        probability('max_conflict_fraction', self.max_conflict_fraction)
        probability('min_score_confidence', self.min_score_confidence, strict=True)
        for k in ('max_shared_error_upper', 'max_risk_increase_over_local'):
            v = getattr(self, k)
            if v is not None:
                probability(k, v)


@dataclass(frozen=True)
class FusionDecision:
    event_id: str
    frame_id: str
    action: Action
    per_agent_decisions: tuple[tuple[str, int], ...]
    fused_prediction: int | None
    surrogate_score: float
    score_confidence: float
    used_groups: tuple[str, ...]
    supporting_groups: tuple[str, ...]
    masked_agents: tuple[tuple[str, str], ...]
    fallback_reasons: tuple[str, ...]
    estimator_id: str

    def __post_init__(self) -> None:
        names=[agent for agent,_ in self.per_agent_decisions]
        if not names or any(not isinstance(n,str) or not n for n in names) or len(set(names))!=len(names):
            raise ValueError('per_agent_decisions requires unique nonempty agents')
        if any(type(y) is not int or y not in (0,1) for _,y in self.per_agent_decisions):
            raise ValueError('per_agent_decisions contains a nonbinary decision')
        finite('surrogate_score', self.surrogate_score)
        probability('score_confidence', self.score_confidence)
        if self.score_confidence < .5:
            raise ValueError('score_confidence must be >= 0.5')
        if self.action == Action.KEEP_LOCAL:
            if self.fused_prediction is not None:
                raise ValueError('Local fallback cannot carry a fused prediction')
        elif self.action in (Action.SHARE_FUSED,Action.MASK_AND_FUSE):
            if type(self.fused_prediction) is not int or self.fused_prediction not in (0,1):
                raise ValueError('Shared action requires binary fused prediction')
            if {y for _,y in self.per_agent_decisions}!={self.fused_prediction}:
                raise ValueError('Shared action must deliver the same fused label to all agents')
            if self.fallback_reasons:
                raise ValueError('Accepted fusion cannot carry fallback reasons')
            if self.action == Action.MASK_AND_FUSE and not self.masked_agents:
                raise ValueError('MASK_AND_FUSE needs an explicitly masked contribution')
            if self.action == Action.SHARE_FUSED and self.masked_agents:
                raise ValueError('SHARE_FUSED cannot hide a mask')
        else:
            raise ValueError('Unknown action')
        if len(set(self.used_groups))!=len(self.used_groups) or len(set(self.supporting_groups))!=len(self.supporting_groups):
            raise ValueError('Repeated group identity')
        if set(self.supporting_groups)-set(self.used_groups):
            raise ValueError('Supporting group was not used')
        masked=[a for a,_ in self.masked_agents]
        if len(set(masked))!=len(masked) or set(masked)-set(names):
            raise ValueError('Invalid masked agent identity')
        if not self.estimator_id:
            raise ValueError('Estimator ID required')

    @property
    def shared(self) -> bool:
        return self.action in (Action.SHARE_FUSED, Action.MASK_AND_FUSE)
