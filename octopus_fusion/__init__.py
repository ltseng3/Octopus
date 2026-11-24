"""OCTOPUS F0: offline, decision-only selective fusion research baseline.

Does not schedule inference, charge physical energy, or access hidden labels.
"""
from .contracts import (
    AgentEvidence, FusionObservation, ReliabilityEstimate, RiskBounds,
    FusionConfig, FusionDecision, Action, MaskReason,
)
from .fusion import SelectiveFuser, unweighted_majority, weighted_vote
from .estimators import QualityProxyEstimator, StaticErrorEstimator

__version__ = "0.1.0"
__all__ = [
    'AgentEvidence', 'FusionObservation', 'ReliabilityEstimate', 'RiskBounds',
    'FusionConfig', 'FusionDecision', 'Action', 'MaskReason', 'SelectiveFuser',
    'unweighted_majority', 'weighted_vote', 'QualityProxyEstimator',
    'StaticErrorEstimator',
]
