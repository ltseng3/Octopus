"""Deterministic F0 weighting, lineage-aware masking, and selective local fallback.

The log-likelihood sum is Bayes-optimal ONLY with calibrated symmetric binary
errors, correct prior, and conditionally independent *result groups*. The
configurable discount and score sigmoid are heuristics otherwise. They are
never labeled measured posteriors or verified collective-risk probabilities.
"""
from __future__ import annotations

from collections import defaultdict
import math

from .contracts import (
    Action, FusionConfig, FusionDecision, FusionObservation,
    ReliabilityEstimate, RiskBounds, MaskReason,
)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _weight(q: float) -> float:
    return math.log1p(-q) - math.log(q)


def _priors(obs: FusionObservation) -> float:
    return math.log(obs.prior_positive) - math.log1p(-obs.prior_positive)


def _q_by_agent(obs: FusionObservation, estimates: ReliabilityEstimate) -> dict[str, float]:
    q = dict(estimates.error_by_agent)
    if set(q) != {e.agent_id for e in obs.evidence}:
        raise ValueError('Reliability estimates must name exactly the observed agents')
    return q


def unweighted_majority(observation: FusionObservation) -> int | None:
    """Unprotected per-agent majority baseline; ties return None.

    Deliberately counts agents rather than unique evidence groups to expose
    the danger of naive voting over multiple consumers of one shared result.
    """
    count = sum(e.prediction for e in observation.evidence)
    if count * 2 == len(observation.evidence):
        return None
    return int(count * 2 > len(observation.evidence))


def weighted_vote(observation: FusionObservation,
                  reliability: ReliabilityEstimate) -> int | None:
    """Naive reliability-weighted per-agent vote baseline, no masking or reuse guard."""
    errors = _q_by_agent(observation, reliability)
    score = _priors(observation) + sum(
        (2 * evidence.prediction - 1) * _weight(errors[evidence.agent_id])
        for evidence in observation.evidence
    )
    return 1 if score > 1e-12 else 0 if score < -1e-12 else None


class SelectiveFuser:
    """F0 produces a per-ROI verdict only; there is NO resource-scheduling side effect.

    - Low-reliability/stale contributors are masked.
    - Reports derived from the same lineage form **one** independent group.
    - Contradictory reports in one dependence group mask that entire group.
    - Same-label group consensus uses the most conservative (largest) q among
      eligible group members; a duplicate cannot amplify the evidence weight.
    - When evidence/risk checks fail, every agent retains its OWN local label.
    """

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()

    def decide(self, observation: FusionObservation,
               reliability: ReliabilityEstimate,
               *, risk_bounds: RiskBounds | None = None) -> FusionDecision:
        cfg = self.config
        errors = _q_by_agent(observation, reliability)
        agents = tuple(sorted(observation.evidence, key=lambda x: x.agent_id))
        local = tuple((e.agent_id, e.prediction) for e in agents)
        masked: dict[str, str] = {}
        groups: dict[str, list] = defaultdict(list)
        all_group_labels: dict[str, set[int]] = defaultdict(set)
        for item in agents:
            all_group_labels[item.dependence_group].add(item.prediction)
        conflicted_groups = {g for g, labels in all_group_labels.items() if len(labels)>1}
        for e in agents:
            q = errors[e.agent_id]
            if e.dependence_group in conflicted_groups:
                masked[e.agent_id] = MaskReason.DEPENDENCE_CONFLICT.value
            elif q >= cfg.max_source_error or q >= .5:
                masked[e.agent_id] = MaskReason.TOO_UNRELIABLE.value
            elif observation.as_of_ms - e.produced_at_ms > cfg.max_age_ms:
                masked[e.agent_id] = MaskReason.STALE.value
            else:
                groups[e.dependence_group].append(e)

        usable = []  # (group_id, group_label, conservative_error)
        for gid, members in sorted(groups.items()):
            if len({e.prediction for e in members}) != 1:
                # Even if the physical result_id differs, the declared lineage
                # is not independent. Picking a convenient label is unsafe.
                for e in members:
                    masked[e.agent_id] = MaskReason.DEPENDENCE_CONFLICT.value
                continue
            usable.append((gid, members[0].prediction,
                           max(errors[e.agent_id] for e in members)))

        logit = _priors(observation) + cfg.dependence_discount * sum(
            (2 * label - 1) * _weight(q) for _, label, q in usable
        )
        pos = _sigmoid(logit)
        confidence = max(pos, 1 - pos)
        label = int(logit > 0)
        reasons = []
        if len(usable) < cfg.min_independent_groups:
            reasons.append('insufficient_independent_groups')
        if abs(logit) <= 1e-12:
            reasons.append('ambiguous_vote')
        support = tuple(g for g, y, _ in usable if y == label)
        if len(support) < cfg.min_supporting_groups:
            reasons.append('insufficient_independent_support_for_winner')
        # Conflict ratio counts independent group votes, not agent votes.
        opposing = len(usable) - len(support)
        conflict_fraction = min(len(support), opposing) / max(1, len(usable))
        if conflict_fraction > cfg.max_conflict_fraction:
            reasons.append('excessive_conflict')
        if confidence < cfg.min_score_confidence:
            reasons.append('low_surrogate_confidence')
        if cfg.require_risk_bounds or cfg.max_shared_error_upper is not None or cfg.max_risk_increase_over_local is not None:
            if risk_bounds is None or risk_bounds.shared_error_upper is None:
                reasons.append('external_risk_bound_unavailable')
            else:
                if (cfg.max_shared_error_upper is not None and
                        risk_bounds.shared_error_upper > cfg.max_shared_error_upper):
                    reasons.append('shared_risk_budget_exceeded')
                if cfg.max_risk_increase_over_local is not None:
                    if risk_bounds.local_joint_error_lower is None:
                        reasons.append('local_risk_bound_unavailable')
                    elif (risk_bounds.shared_error_upper >
                          risk_bounds.local_joint_error_lower + cfg.max_risk_increase_over_local):
                        reasons.append('risk_increase_budget_exceeded')
        if reasons:
            return FusionDecision(
                observation.event_id, observation.frame_id, Action.KEEP_LOCAL,
                local, None, logit, confidence,
                tuple(g for g, _, _ in usable), support,
                tuple(sorted(masked.items())), tuple(reasons), reliability.estimator_id,
            )
        action = Action.MASK_AND_FUSE if masked else Action.SHARE_FUSED
        return FusionDecision(
            observation.event_id, observation.frame_id, action,
            tuple((a, label) for a, _ in local), label, logit, confidence,
            tuple(g for g, _, _ in usable), support,
            tuple(sorted(masked.items())), (), reliability.estimator_id,
        )
