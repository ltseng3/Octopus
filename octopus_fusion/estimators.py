"""Replaceable, *non-oracle* toy estimators. Outputs are assumptions.

Real use needs per-scene/per-agent measured calibration with a held-out split;
this module makes NO claim that quality proxies are calibrated error rates.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Protocol
from .contracts import FusionObservation, ReliabilityEstimate, probability, finite


class ErrorProbabilityEstimator(Protocol):
    def estimate(self, observation: FusionObservation) -> ReliabilityEstimate: ...


@dataclass(frozen=True)
class StaticErrorEstimator:
    """Explicitly configured per-agent assumed error probabilities."""
    by_agent: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        ReliabilityEstimate(self.by_agent, estimator_id='static')

    def estimate(self, observation: FusionObservation) -> ReliabilityEstimate:
        return ReliabilityEstimate(self.by_agent, estimator_id='static')


@dataclass(frozen=True)
class QualityProxyEstimator:
    """Monotone synthetic proxy mapping; NOT a learned or calibrated model.

    q = min(max_error, max(min_error, base_error + quality_slope*(1-quality))).
    The proxy is ex-ante quality metadata supplied to the edge, NOT correctness.
    """
    base_error: float = 0.025
    quality_slope: float = 0.32
    min_error: float = 0.005
    max_error: float = 0.49

    def __post_init__(self) -> None:
        for name in ('base_error', 'min_error', 'max_error'):
            probability(name, getattr(self, name), strict=True)
        finite('quality_slope', self.quality_slope)
        if self.quality_slope < 0 or not self.min_error <= self.max_error:
            raise ValueError('invalid proxy-estimator bounds or slope')

    def estimate(self, observation: FusionObservation) -> ReliabilityEstimate:
        result = []
        for item in observation.evidence:
            if item.quality_proxy is None:
                raise ValueError('quality_proxy is required; do not infer it from truth')
            q = min(self.max_error, max(self.min_error,
                                         self.base_error + self.quality_slope * (1-item.quality_proxy)))
            result.append((item.agent_id, q))
        return ReliabilityEstimate(tuple(result), estimator_id='uncalibrated_quality_proxy')
