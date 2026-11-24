"""Edge coordination policies. Never receive ground truth or latent error draws."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CoordinatorView:
    event_id: str
    agents: tuple[str, ...]
    deadline_ms: float
    policy_draw: float  # scenario-independent policy control variate, not a world RNG seed
    # Estimates/public assumptions, NOT oracle draws or ground truth.
    estimated_edge_error: float
    estimated_separate_correlation: float


@dataclass(frozen=True)
class Group:
    consumers: tuple[str, ...]
    source_agent: str
    location: str  # local or edge


@dataclass(frozen=True)
class Plan:
    groups: tuple[Group, ...]
    policy_name: str
    estimated_full_reuse_probability: float | None = None
    predicted_infeasible: bool = False
    execution_mode: str = 'groups'


def validate_plan(plan: Plan, agents: tuple[str, ...]) -> None:
    if plan.execution_mode in {'cooperative_late_fusion', 'selective_fusion_f0'}:
        if plan.groups:
            raise ValueError('Fusion plan may not contain legacy groups')
        return
    if plan.execution_mode != 'groups':
        raise ValueError('Unknown execution mode')
    covered: list[str] = []
    for group in plan.groups:
        if not group.consumers:
            raise ValueError('Empty inference group')
        if group.location not in {'local', 'edge'}:
            raise ValueError('Invalid group execution location')
        if group.source_agent not in group.consumers:
            raise ValueError('Inference source must consume its own result')
        if group.location == 'local' and len(group.consumers) != 1:
            raise ValueError('Local inference groups must contain one consumer')
        covered.extend(group.consumers)
    if len(covered) != len(agents) or set(covered) != set(agents) or len(set(covered)) != len(covered):
        raise ValueError('Policy must assign every agent exactly once')


class EdgePolicy(Protocol):
    def plan(self, view: CoordinatorView) -> Plan: ...


class BasicPolicy:
    def __init__(self, name: str, reuse_probability: float = 0.5,
                 joint_failure_budget: float | None = None,
                 max_edge_executions: int = 2):
        self.name = name
        self.reuse_probability = reuse_probability
        self.joint_failure_budget = joint_failure_budget
        self.max_edge_executions = max_edge_executions

    def plan(self, view: CoordinatorView) -> Plan:
        agents = view.agents
        if self.name == 'selective_fusion_f0':
            return Plan((), self.name, execution_mode='selective_fusion_f0')
        if self.name == 'cooperative_late_fusion':
            return Plan((), self.name, execution_mode='cooperative_late_fusion')
        if self.name == 'mixed_local_fusion':
            if view.policy_draw < self.reuse_probability:
                return Plan((), self.name, self.reuse_probability, execution_mode='cooperative_late_fusion')
            return Plan(tuple(Group((a,), a, 'local') for a in agents),
                        self.name, self.reuse_probability)
        if self.name == 'local_only':
            groups = tuple(Group((a,), a, 'local') for a in agents)
            return Plan(groups, self.name)
        if self.name == 'always_independent':
            return Plan(tuple(Group((a,), a, 'edge') for a in agents), self.name)
        if self.name == 'always_share':
            return Plan((Group(agents, agents[0], 'edge'),), self.name)
        if self.name == 'partial_share':
            k = min(self.max_edge_executions, len(agents))
            # One joint group and k-1 singletons; a controlled partial-reuse baseline.
            first = agents[:len(agents) - k + 1]
            groups = [Group(first, first[0], 'edge')]
            groups.extend(Group((a,), a, 'edge') for a in agents[len(first):])
            return Plan(tuple(groups), self.name)
        if self.name == 'mixed_local_shared':
            if view.policy_draw < self.reuse_probability:
                groups = (Group(agents, agents[0], 'edge'),)
            else:
                groups = tuple(Group((a,), a, 'local') for a in agents)
            return Plan(groups, self.name, self.reuse_probability)
        if self.name == 'randomized_reuse':
            p = self.reuse_probability
            return self._mixture(view, p)
        if self.name == 'risk_budget':
            assert self.joint_failure_budget is not None
            f = view.estimated_edge_error
            rho = view.estimated_separate_correlation
            n = len(agents)
            # Common-shock model for the ALL-agents-wrong risk.
            q = rho * f + (1 - rho) * f ** n
            span = f - q
            p = 1.0 if span <= 1e-12 and q <= self.joint_failure_budget else (
                0.0 if span <= 1e-12 else
                max(0.0, min(1.0, (self.joint_failure_budget - q) / span))
            )
            plan = self._mixture(view, p)
            return Plan(plan.groups, self.name, p, q > self.joint_failure_budget + 1e-12)
        raise ValueError(f'Unknown policy: {self.name}')

    def _mixture(self, view: CoordinatorView, p: float) -> Plan:
        if view.policy_draw < p:
            groups = (Group(view.agents, view.agents[0], 'edge'),)
        else:
            groups = tuple(Group((a,), a, 'edge') for a in view.agents)
        return Plan(groups, self.name, p)


def make_policy(config: dict) -> BasicPolicy:
    p = config['policy']
    return BasicPolicy(
        name=p['name'], reuse_probability=float(p.get('reuse_probability', 0.5)),
        joint_failure_budget=p.get('joint_failure_budget'),
        max_edge_executions=p.get('max_edge_executions', 2),
    )
