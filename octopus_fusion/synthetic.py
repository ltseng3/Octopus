"""Reproducible offline toy data. Ground truth lives ONLY in LabeledCase.

This generator is an *assumption-driven toy*, NOT a sensor simulator or a
reproduction of V2V4Real. Two ROI decisions can share one frame-level result ID
while having different binary predictions; F0 never caches a binary ROI label
across events. Strong common shocks intentionally defeat simple voting.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

from .contracts import AgentEvidence, FusionObservation
from .evaluate import LabeledCase


@dataclass(frozen=True)
class ToySettings:
    frames: int = 1800
    rois_per_frame: int = 2
    agents: int = 3
    seed: int = 4701
    shared_origin_fraction: float = 0.15
    common_shock_probability: float = 0.035
    degraded_probability: float = 0.27

    def __post_init__(self) -> None:
        if min(self.frames, self.rois_per_frame, self.agents) <= 0 or self.agents < 2:
            raise ValueError('Require >=1 frame, >=1 ROI and >=2 agents')
        for f in ('shared_origin_fraction','common_shock_probability','degraded_probability'):
            v=getattr(self,f)
            if not 0 <= v <= 1:
                raise ValueError(f'{f} must be in [0,1]')


def generate(settings: ToySettings = ToySettings()) -> tuple[LabeledCase, ...]:
    rng=random.Random(settings.seed)
    cases=[]
    for f in range(settings.frames):
        # Public quality proxies are generated separately from the true label;
        # sensor failure is drawn later and not returned as an F0 input.
        proxies=[rng.uniform(.73,.99) for _ in range(settings.agents)]
        if rng.random() < settings.degraded_probability:
            target = rng.randrange(settings.agents)
            proxies[target]=rng.uniform(.02,.52)
        # Two agents sometimes consume the same physical result. Their public
        # metadata and outcome must then be identical within a given ROI.
        linked=(rng.random() < settings.shared_origin_fraction)
        if linked:
            proxies[-1]=proxies[0]
        group=[f'agent:{i}' for i in range(settings.agents)]
        if linked:
            group[-1]=group[0]
        time_ms=float(f*150+20)
        for roi in range(settings.rois_per_frame):
            truth=int(rng.random()<.44)
            shock=rng.random()<settings.common_shock_probability
            outcomes={}
            ev=[]
            for i, gid in enumerate(group):
                if gid not in outcomes:
                    # True error probabilities are a toy *data-generation*
                    # process. The estimator sees proxies, never error draws.
                    q_true=.018 + .37*(1-proxies[i])
                    outcomes[gid]=int(shock or rng.random()<q_true)
                err=outcomes[gid]
                ev.append(AgentEvidence(
                    agent_id=f'agent_{i}', prediction=truth^err,
                    result_id=f'frame_{f}/result_{gid}', dependence_group=gid,
                    produced_at_ms=time_ms-4,
                    quality_proxy=proxies[i],
                ))
            obs=FusionObservation(
                event_id=f'frame_{f}/roi_{roi}', frame_id=f'frame_{f}',
                as_of_ms=time_ms, evidence=tuple(ev), prior_positive=.44,
            )
            cases.append(LabeledCase(obs,truth))
    return tuple(cases)
