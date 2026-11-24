"""V0.7 policy-facing, non-oracular coordination contracts.

Observations contain only explicit public signals and snapshots of *committed*
resource reservations. Candidate evaluation is side-effect-free and optimistic
about future dependencies of already in-flight DAGs. None of these interfaces
receives a WorldEvent, a prediction backend, the energy ledger, or ground truth.

The interfaces deliberately do not optimize policies: optimization is P1.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from math import isfinite
import heapq
import math
from typing import Protocol

from .engine import to_us
from .execution import ExecutionGraph, compile_execution_graph, effective_profiles
from .policies import Group, Plan


@dataclass(frozen=True)
class QueueSnapshot:
    """Absolute worker availability times, including previously reserved jobs."""
    next_free_us: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.next_free_us or any(type(v) is not int or v < 0 for v in self.next_free_us):
            raise ValueError('QueueSnapshot requires nonnegative integer availability times')


@dataclass(frozen=True)
class ResourceSnapshot:
    now_us: int
    edge: QueueSnapshot
    local: tuple[tuple[str, QueueSnapshot | None], ...]
    network: QueueSnapshot
    # Scheduling an earlier DAG does not immediately reserve its *future* stages.
    # A one-shot snapshot thus cannot guarantee exact prediction under contention.
    in_flight_workflows: int = 0

    def __post_init__(self) -> None:
        if self.now_us < 0 or self.in_flight_workflows < 0:
            raise ValueError('Negative scheduler time or workflow count')
        if len(set(name for name, _ in self.local)) != len(self.local):
            raise ValueError('Duplicate local resource identity')


@dataclass(frozen=True)
class AvailableModel:
    model_id: str
    device_class: str
    compute_us: int
    output_bytes: int


@dataclass(frozen=True)
class PublicInference:
    result_id: str
    model_id: str
    source_agents: tuple[str, ...]
    created_at_us: int


@dataclass(frozen=True)
class CoordinatorObservation:
    """Immutable, explicitly allowlisted data available at decision time.

    A mapping is not used: tuples prevent hidden mutable sub-objects from
    reaching policy plugins. The simulator constructs these from public records.
    """
    batch_id: str
    frame_id: str | None
    agent_ids: tuple[str, ...]
    deadline_us: int                 # absolute deadline
    time_us: int                     # decision time
    scene_features: tuple[tuple[str, float], ...]
    completed_inferences: tuple[PublicInference, ...]
    resource_snapshot: ResourceSnapshot
    available_models: tuple[AvailableModel, ...]
    fusion_enabled: bool
    policy_draw: float

    def feature(self, name: str) -> float | None:
        return next((v for key, v in self.scene_features if key == name), None)


@dataclass(frozen=True)
class CandidatePlan:
    candidate_id: str
    plan: Plan


class CandidateProvider(Protocol):
    def generate(self, observation: CoordinatorObservation) -> tuple[CandidatePlan, ...]: ...


class StandardCandidateProvider:
    """Bounded, deterministic candidates; no combinatorial partition search."""

    def __init__(self, allowed_ids: tuple[str, ...] = ()) -> None:
        self.allowed_ids = allowed_ids

    def generate(self, observation: CoordinatorObservation) -> tuple[CandidatePlan, ...]:
        agents = observation.agent_ids
        generated = [
            CandidatePlan('local_only', Plan(tuple(Group((a,), a, 'local') for a in agents), 'local_only')),
            CandidatePlan('independent_edge', Plan(tuple(Group((a,), a, 'edge') for a in agents), 'always_independent')),
        ]
        generated.extend(
            CandidatePlan(f'shared_edge:{a}', Plan((Group(agents, a, 'edge'),), 'always_share'))
            for a in agents
        )
        if len(agents) >= 3:
            generated.append(CandidatePlan(
                'partial_share', Plan((Group(agents[:-1], agents[0], 'edge'),
                                      Group((agents[-1],), agents[-1], 'edge')), 'partial_share')))
        if observation.fusion_enabled:
            generated.append(CandidatePlan('cooperative_fusion', Plan((), 'cooperative_late_fusion',
                                                                      execution_mode='cooperative_late_fusion')))
        if self.allowed_ids:
            eligible = {c.candidate_id: c for c in generated}
            unknown = set(self.allowed_ids) - set(eligible)
            if unknown:
                raise ValueError(f'Unavailable coordinator candidates: {sorted(unknown)}')
            return tuple(eligible[cid] for cid in self.allowed_ids)
        return tuple(generated)


@dataclass(frozen=True)
class RiskEstimate:
    failure_threshold: int
    at_least_m_error_probability: float
    all_agent_error_probability: float
    marginal_by_agent: tuple[tuple[str, float], ...]
    estimator_id: str
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        for p in [self.at_least_m_error_probability, self.all_agent_error_probability,
                  *(v for _, v in self.marginal_by_agent)]:
            if not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError('Invalid risk probability')
        if not 1 <= self.failure_threshold <= len(self.marginal_by_agent):
            raise ValueError('Invalid failure threshold')
        if self.at_least_m_error_probability + 1e-10 < self.all_agent_error_probability:
            raise ValueError('Threshold risk cannot be lower than all-agent risk')
        names = [name for name, _ in self.marginal_by_agent]
        if len(set(names)) != len(names) or not all(names):
            raise ValueError('Marginal risk agent identities must be unique')
        if self.all_agent_error_probability > min(v for _,v in self.marginal_by_agent) + 1e-10:
            raise ValueError('All-agent error cannot exceed the smallest marginal error')


class CollectiveRiskEstimator(Protocol):
    def estimate(self, observation: CoordinatorObservation, candidate: CandidatePlan,
                 *, failure_threshold: int) -> RiskEstimate: ...


class SyntheticRiskEstimator:
    """Configurable common-shock model over *distinct inference-result groups*.

    A group is perfectly coupled because it distributes ONE binary result to
    all consumers. Distinct groups can share a global latent shock with weight
    rho; the estimator does not access the world's actual latent shocks.
    """

    def __init__(self, config: Mapping[str, object]) -> None:
        self.params = dict(config)

    def estimate(self, observation: CoordinatorObservation, candidate: CandidatePlan,
                 *, failure_threshold: int) -> RiskEstimate:
        agents = observation.agent_ids
        if not 1 <= failure_threshold <= len(agents):
            raise ValueError('Failure threshold outside team size')
        groups = ([tuple(agents)] if candidate.plan.execution_mode == 'cooperative_late_fusion'
                  else [group.consumers for group in candidate.plan.groups])
        kind = self.params['kind']
        def probability(location: str) -> float:
            key = ('fusion_error' if location == 'fusion' else
                   'local_error' if location == 'local' else 'edge_error')
            p = float(self.params[key])
            if kind == 'contextual':
                visible = observation.feature(str(self.params['context_feature']))
                if visible is None:
                    raise ValueError('Required contextual feature is not visible to the edge')
                p -= float(self.params['context_slope']) * (visible - 0.5)
            if kind == 'misspecified':
                p += float(self.params['bias'])
            return max(0.0, min(1.0, p))
        if candidate.plan.execution_mode == 'cooperative_late_fusion':
            probabilities = [probability('fusion')]
        else:
            probabilities = [probability(group.location) for group in candidate.plan.groups]
        if sum(map(len, groups)) != len(agents):
            raise ValueError('Candidate does not partition the team')
        # Independent-group distribution: each result may cause multiple errors.
        dp = [1.0] + [0.0] * len(agents)
        for members, p in zip(groups, probabilities):
            nxt = [0.0] * (len(agents) + 1)
            for errors, mass in enumerate(dp):
                if mass:
                    nxt[errors] += mass * (1 - p)
                    if errors + len(members) <= len(agents):
                        nxt[errors + len(members)] += mass * p
            dp = nxt
        independent = sum(dp[failure_threshold:])
        # Comonotonic common-shock branch: U ~ Uniform[0,1], result wrong iff U<p.
        bounds = sorted({0.0, 1.0, *probabilities})
        comonotonic = sum(
            (high - low) for low, high in zip(bounds, bounds[1:])
            if sum(len(g) for g, p in zip(groups, probabilities) if p > (low + high) / 2)
            >= failure_threshold
        )
        rho = float(self.params['correlation'])
        overall = (1-rho) * independent + rho * comonotonic
        all_joint = (1-rho) * math.prod(probabilities) + rho * min(probabilities)
        marginal = {agent: p for group, p in zip(groups, probabilities) for agent in group}
        return RiskEstimate(
            failure_threshold, overall, all_joint,
            tuple((agent, marginal[agent]) for agent in agents),
            'synthetic_' + str(kind),
            ('model_assumptions_not_ground_truth', 'common_shock_between_distinct_results',
             'identical_error_within_shared_result'))


@dataclass(frozen=True)
class ResourceEstimate:
    # Incremental consumption only; fixed-horizon baseline belongs to evaluation.
    compute_us_by_device: tuple[tuple[str, int], ...]
    compute_us_by_model: tuple[tuple[str, int], ...]
    transfer_bytes: int
    bytes_by_role: tuple[tuple[str, int], ...]
    expected_network_delivery_probability: float
    completion_latency_ms: float
    mean_delivery_latency_ms: float
    ideal_deadline_feasible: bool
    ideal_deadline_slack_ms: float
    incremental_energy_j: float | None
    incremental_energy_by_device_j: tuple[tuple[str, float], ...] | None
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.expected_network_delivery_probability <= 1:
            raise ValueError('Invalid delivery probability')
        if self.transfer_bytes < 0 or sum(v for _,v in self.bytes_by_role) != self.transfer_bytes:
            raise ValueError('Inconsistent network byte estimate')
        if any(type(v) is not int or v < 0 for _,v in (*self.compute_us_by_device,
                                                       *self.compute_us_by_model)):
            raise ValueError('Invalid compute estimate')
        if sum(v for _,v in self.compute_us_by_device) != sum(v for _,v in self.compute_us_by_model):
            raise ValueError('Inconsistent predicted compute work')
        if (not isfinite(self.completion_latency_ms) or self.completion_latency_ms < 0 or
            not isfinite(self.mean_delivery_latency_ms) or self.mean_delivery_latency_ms < 0 or
            self.mean_delivery_latency_ms > self.completion_latency_ms + 1e-8 or
            not isfinite(self.ideal_deadline_slack_ms) or
            self.ideal_deadline_feasible != (self.ideal_deadline_slack_ms >= -1e-8)):
            raise ValueError('Invalid latency or deadline estimate')
        if self.incremental_energy_j is None:
            if self.incremental_energy_by_device_j is not None:
                raise ValueError('Energy detail without energy estimate')
        else:
            if (not isfinite(self.incremental_energy_j) or self.incremental_energy_j < 0 or
                self.incremental_energy_by_device_j is None or
                any(not isfinite(v) or v < 0 for _,v in self.incremental_energy_by_device_j) or
                not math.isclose(sum(v for _,v in self.incremental_energy_by_device_j),
                                 self.incremental_energy_j, rel_tol=1e-10,abs_tol=1e-12)):
                raise ValueError('Inconsistent energy estimate')


class ResourceCostEstimator(Protocol):
    def estimate(self, observation: CoordinatorObservation, candidate: CandidatePlan,
                 graph: ExecutionGraph) -> ResourceEstimate: ...


class AnalyticalResourceCostEstimator:
    """L0 deterministic DAG scheduler over a COPY of committed reservations.

    No queue mutation, random draws, packet-loss outcomes, or energy ledger use.
    The forecast ignores stages not yet reserved by already-running DAGs and
    therefore is NOT an exact oracle under overlapping workflows.
    """

    def __init__(self, resources: Mapping, energy: Mapping):
        # Defensive deep copies. A user-supplied provider cannot mutate the
        # config while the simulator is executing.
        from copy import deepcopy
        self._resources = deepcopy(dict(resources))
        self._energy = deepcopy(dict(energy))
        self._profiles = effective_profiles(self._resources)

    def estimate(self, observation: CoordinatorObservation, candidate: CandidatePlan,
                 graph: ExecutionGraph) -> ResourceEstimate:
        graph.validate()
        now = observation.time_us
        snapshots = observation.resource_snapshot
        if snapshots.now_us != now or graph.agents != observation.agent_ids:
            raise ValueError('Snapshot time/agents incompatible with candidate graph')
        queues = {'edge': list(snapshots.edge.next_free_us),
                  **{name: list(q.next_free_us) if q is not None else [now] for name, q in snapshots.local},
                  'network': list(snapshots.network.next_free_us)}
        count = 0
        heap = []
        lookup = {s.stage_id: s for s in graph.stages}
        dependents: dict[str, list[str]] = {s.stage_id: [] for s in graph.stages}
        unresolved = {s.stage_id: len(s.dependencies) for s in graph.stages}
        for stage in graph.stages:
            for parent in stage.dependencies:
                dependents[parent].append(stage.stage_id)
        active_us = {}
        network_bytes_by_role: dict[str, int] = {}
        compute_by_device: dict[str, int] = {}
        compute_by_model: dict[str, int] = {}
        delivery = {}
        incremental_energy_by_device: dict[str, float] = {}
        compute_energy = 0.0
        radio_energy = 0.0
        energy_enabled = bool(self._energy['enabled'])
        def enqueue(at: int, label: str, sid: str) -> None:
            nonlocal count
            count += 1
            heapq.heappush(heap, (at, count, label, sid))
        def reserve(resource: str, requested: int, duration: int) -> tuple[int, int]:
            workers = queues[resource]
            ix = min(range(len(workers)), key=lambda i: (workers[i], i))
            start = max(requested, workers[ix])
            finish = start + duration
            workers[ix] = finish
            return start, finish
        def schedule(stage, at: int) -> None:
            nonlocal compute_energy, radio_energy
            if stage.kind == 'deliver':
                delivery[stage.consumer] = at
                enqueue(at, 'done', stage.stage_id)
            elif stage.kind == 'compute':
                duration = to_us(self._profiles[stage.model_id]['compute_ms'])
                start, finish = reserve(stage.device, at, duration)
                active_us[stage.stage_id] = (start, finish)
                compute_by_device[stage.device] = compute_by_device.get(stage.device, 0) + duration
                compute_by_model[stage.model_id] = compute_by_model.get(stage.model_id, 0) + duration
                if energy_enabled:
                    overrides = self._energy['model_energy_j']
                    watt = self._energy['model_incremental_w']
                    profile = self._device_profile(stage.device)
                    joules = (overrides[stage.model_id] if stage.model_id in overrides else
                              duration / 1_000_000 * watt.get(stage.model_id, profile['compute_incremental_w']))
                    compute_energy += joules
                    incremental_energy_by_device[stage.device] = incremental_energy_by_device.get(stage.device, 0) + joules
                enqueue(finish, 'done', stage.stage_id)
            else:
                size = stage.bytes
                duration = math.ceil(size * 8 / self._resources['network_bandwidth_mbps'])
                start, sent = reserve('network', at, duration)
                network_bytes_by_role[stage.transfer_role] = network_bytes_by_role.get(stage.transfer_role, 0) + size
                if energy_enabled:
                    for dev, action in ((stage.sender, 'tx'), (stage.recipient, 'rx')):
                        radio = self._device_profile(dev)['radio']
                        joules = (8 * size * radio[f'{action}_j_per_bit'] if radio['mode'] == 'per_bit'
                                  else duration / 1_000_000 * radio[f'{action}_incremental_w'])
                        radio_energy += joules
                        incremental_energy_by_device[dev] = incremental_energy_by_device.get(dev, 0) + joules
                enqueue(sent + to_us(self._resources['network_latency_ms']), 'done', stage.stage_id)
        for stage in graph.stages:
            if not stage.dependencies:
                schedule(stage, now)
        while heap:
            at, _, _, sid = heapq.heappop(heap)
            for child in dependents[sid]:
                unresolved[child] -= 1
                if unresolved[child] == 0:
                    schedule(lookup[child], at)
        if len(delivery) != len(observation.agent_ids):
            raise RuntimeError('Candidate graph did not deliver to every agent')
        times = [max(0.0, (at - now) / 1000) for at in delivery.values()]
        transfers = sum(1 for s in graph.stages if s.kind == 'transfer')
        return ResourceEstimate(
            tuple(sorted(compute_by_device.items())), tuple(sorted(compute_by_model.items())),
            sum(network_bytes_by_role.values()), tuple(sorted(network_bytes_by_role.items())),
            (1 - float(self._resources['packet_loss'])) ** transfers,
            max(times), sum(times) / len(times),
            now + to_us(max(times)) <= observation.deadline_us,
            (observation.deadline_us - now) / 1000 - max(times),
            (compute_energy + radio_energy) if energy_enabled else None,
            tuple(sorted(incremental_energy_by_device.items())) if energy_enabled else None,
            ('successful_network_transfers_assumed_for_latency',
             'packet_delivery_probability_assumes_independent_message_losses',
             'committed_queue_reservations_only',
             'no_unreserved_future_work_from_in_flight_graphs',
             'unknown_local_queues_treated_as_free_at_decision_time',
             'incremental_energy_excludes_fixed_idle_baseline',
             'fixed_service_durations', 'no_effect_of_packet_loss_on_service_work'))

    def _device_profile(self, device: str) -> Mapping:
        e = self._energy
        return (e['device_profiles']['edge'] if device == 'edge' else
                e['agent_overrides'].get(device, e['device_profiles']['agent']))


@dataclass(frozen=True)
class CandidateAssessment:
    candidate_id: str
    plan: Plan
    risk: RiskEstimate
    cost: ResourceEstimate


class CoordinationSelector(Protocol):
    def select(self, observation: CoordinatorObservation,
               candidates: tuple[CandidateAssessment, ...]) -> str: ...


def build_observation(*, batch_id: str, frame_id: str | None, agents: tuple[str, ...],
                      now_us: int, deadline_us: int, features: Mapping[str, float],
                      completed: tuple[PublicInference, ...], snapshot: ResourceSnapshot,
                      resources: Mapping, fusion_enabled: bool, policy_draw: float) -> CoordinatorObservation:
    """Construct a defensive, immutable policy-facing DTO, not an internal World object."""
    models = effective_profiles(dict(resources))
    catalog = tuple(AvailableModel(name, profile['device'], to_us(profile['compute_ms']),
                                   profile['output_bytes']) for name, profile in sorted(models.items()))
    if any(not math.isfinite(v) for v in features.values()):
        raise ValueError('Observable features must be finite')
    return CoordinatorObservation(batch_id, frame_id, agents, deadline_us, now_us,
                                  tuple(sorted(features.items())), completed, snapshot,
                                  catalog, fusion_enabled, policy_draw)
