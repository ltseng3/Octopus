"""Event-driven, edge-coordinated collaborative inference simulator.

The coordinator sees public event metadata and configured estimates only.
Truth remains in WorldEvent and is referenced by backend/evaluator only.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from typing import Any
from .config import ExperimentConfig
from .engine import EventLoop, FIFOResource, to_us
from .policies import CoordinatorView, make_policy, validate_plan
from .rng import uniform
from .world import World, WorldEvent, ExecutionBatch
from .metrics import summarize
from .execution import compile_execution_graph, ExecutionGraph
from .execution_runtime import GraphExecutor
from .energy import EnergyLedger
from collections import defaultdict, deque
from .coordination import (
    QueueSnapshot, ResourceSnapshot, PublicInference, build_observation,
    StandardCandidateProvider, SyntheticRiskEstimator, AnalyticalResourceCostEstimator,
    CandidateAssessment, CandidateProvider, CollectiveRiskEstimator, ResourceCostEstimator,
    RiskEstimate, ResourceEstimate,
    CoordinationSelector,
)


@dataclass
class SimulationResult:
    config: dict[str, Any]
    summary: dict[str, Any]
    scores: list[dict[str, Any]]
    inferences: list[dict[str, Any]]
    events: list[dict[str, Any]]
    policy_plans: list[dict[str, Any]]
    operations: list[dict[str, Any]] | None = None
    candidate_estimates: list[dict[str, Any]] | None = None
    fusion_decisions: list[dict[str, Any]] | None = None


class Simulator:
    def __init__(self, cfg: ExperimentConfig, detailed_events: bool = True, *,
                 candidate_provider: CandidateProvider | None = None,
                 risk_estimator: CollectiveRiskEstimator | None = None,
                 resource_estimator: ResourceCostEstimator | None = None,
                 coordinator_selector: CoordinationSelector | None = None,
                 f0_error_estimator=None, f0_fuser=None):
        self.cfg = cfg
        self.c = cfg.data
        self.s = self.c['scenario']
        self.w = self.c['world']
        self.r = self.c['resources']
        self.loop = EventLoop()
        self.policy = make_policy(self.c)
        self.coord = self.c['coordination']
        self.candidate_estimates: list[dict[str, Any]] = []
        self.fusion_decisions: list[dict[str, Any]] = []
        if self.c['policy']['name'] != 'selective_fusion_f0' and (f0_error_estimator is not None or f0_fuser is not None):
            raise ValueError('F0 estimator/fuser plugins require selective_fusion_f0 policy')
        self.f0_config = self.c['fusion_f0']
        self.f0_fuser = None
        self.f0_error_estimator = None
        if self.c['policy']['name'] == 'selective_fusion_f0':
            from octopus_fusion.contracts import FusionConfig
            from octopus_fusion.estimators import StaticErrorEstimator
            from octopus_fusion.fusion import SelectiveFuser
            self.f0_fuser = SelectiveFuser(FusionConfig(**{
                k:v for k,v in self.f0_config.items() if k not in
                {'error_by_agent','dependence_group_by_agent','quality_proxy_by_agent','prior_positive'}}))
            self.f0_error_estimator = StaticErrorEstimator(tuple(sorted(
                self.f0_config['error_by_agent'].items())))
            if f0_error_estimator is not None:
                self.f0_error_estimator = f0_error_estimator
            if f0_fuser is not None:
                self.f0_fuser = f0_fuser
        self._active_batches: set[str] = set()
        self._edge_public_history: deque[PublicInference] = deque(maxlen=32)
        if any(x is not None for x in (candidate_provider, risk_estimator, resource_estimator,
                                        coordinator_selector)) and not self.coord['enabled']:
            raise ValueError('External coordination plugins require coordination.enabled')
        self.candidate_provider = (candidate_provider or
                                   StandardCandidateProvider(tuple(self.coord['candidate_ids'])))
        self.risk_estimator = (risk_estimator or
                               SyntheticRiskEstimator(self.coord['risk_estimator']))
        self.resource_estimator = (resource_estimator or
                                   AnalyticalResourceCostEstimator(self.r, self.c['energy']))
        self.coordinator_selector = coordinator_selector
        self.world = World(self.c, cfg.resolve_trace() if self.w['backend'] == 'trace' else None)
        self.edge_resource = FIFOResource(self.loop, self.r['edge_workers'])
        self.local_resources = {agent: FIFOResource(self.loop, 1) for agent in self.s['agents']}
        self.link = FIFOResource(self.loop, self.r['network_channels'])
        self.scores: list[dict[str, Any]] = []
        self.inferences: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.policy_plans: list[dict[str, Any]] = []
        self.detailed_events = detailed_events
        self.bytes_transmitted = 0
        self.messages_attempted = 0
        self.messages_lost = 0
        self._pending: dict[str, dict[str, Any]] = {}
        self.operations: list[dict[str, Any]] = []
        self.compute_count_by_model: dict[str, int] = defaultdict(int)
        self.compute_busy_by_model: dict[str, float] = defaultdict(float)
        self.bytes_by_transfer_role: dict[str, int] = defaultdict(int)
        self._graph_cache: dict[tuple, ExecutionGraph] = {}
        self.frame_projection_cache_hits = 0
        self.frame_projection_cache_hits_by_model: dict[str, int] = defaultdict(int)
        self.energy: EnergyLedger | None = None
        if self.c['energy']['enabled']:
            interval_us = to_us(self.s['arrival_interval_ms'])
            # A fixed, policy-independent horizon: last event release + deadline.
            last_release_us = max((e.arrival_slot if e.arrival_slot is not None else e.index)
                                  for e in self.world.events) * interval_us
            nominal_horizon_us = last_release_us + to_us(self.s['deadline_ms'])
            configured = self.c['energy']['measurement_window_ms']
            horizon_us = to_us(configured) if configured is not None else nominal_horizon_us
            self.energy = EnergyLedger(self.c['energy'], tuple(self.s['agents']),
                                       horizon_us, self.r['network_channels'])

    def _log(self, kind: str, event_id: str, **data: Any) -> None:
        if self.detailed_events:
            self.events.append({
                'time_ms': self.loop.now_us / 1000, 'event_id': event_id,
                'type': kind, **data,
            })

    def _resource_snapshot(self) -> ResourceSnapshot:
        return ResourceSnapshot(
            now_us=self.loop.now_us,
            edge=QueueSnapshot(tuple(self.edge_resource.available_at_us)),
            local=tuple((a, QueueSnapshot(tuple(self.local_resources[a].available_at_us))
                         if self.coord['resource_visibility'] == 'reported_all' else None)
                        for a in self.s['agents']),
            network=QueueSnapshot(tuple(self.link.available_at_us)),
            in_flight_workflows=len(self._active_batches),
        )

    def _public_features(self, batch: ExecutionBatch) -> dict[str, float]:
        if self.w['backend'] == 'trace':
            return dict(batch.events[0].public_features)
        result = {}
        for name in self.coord['visible_features']:
            spec = self.coord['synthetic_features'][name]
            if spec['kind'] == 'constant':
                result[name] = float(spec['value'])
            else:
                u = uniform(self.s['seed'], batch.batch_id, 'policy_public_feature', name)
                result[name] = float(spec['low'] + u * (spec['high'] - spec['low']))
        return result

    def _assess_candidates(self, batch: ExecutionBatch, deadline_us: int, policy_draw: float):
        profiles = self.r.get('model_profiles', {})
        binds = self.r['model_bindings']
        fusion_ready = (binds['local'] in profiles and binds['fusion'] in profiles and
                        (self.w['backend'] == 'synthetic' or bool(self.w.get('fusion_candidate_key'))))
        # Past completed inference metadata only. No stored predictions, current
        # ground truth, error outcomes, or future event references are shared.
        history = tuple(self._edge_public_history)
        observation = build_observation(
            batch_id=batch.batch_id, frame_id=batch.frame_id,
            agents=batch.events[0].agents, now_us=self.loop.now_us,
            deadline_us=deadline_us, features=self._public_features(batch),
            completed=history, snapshot=self._resource_snapshot(),
            resources=self.r, fusion_enabled=fusion_ready, policy_draw=policy_draw)
        assessments = []
        records = []
        generated = self.candidate_provider.generate(observation)
        ids = [c.candidate_id for c in generated]
        if len(ids) != len(set(ids)):
            raise ValueError('Candidate inventory contains duplicate IDs')
        if self.w['backend'] == 'trace' and set(ids) - set(self.coord['candidate_ids']):
            raise ValueError('Trace candidate provider exceeded explicit provenance allowlist')
        for candidate in generated:
            validate_plan(candidate.plan, observation.agent_ids)
            graph = compile_execution_graph(candidate.plan, observation.agent_ids,
                                            self.r, self.w.get('fusion_candidate_key'))
            # Critical trace-provenance guard: edge:ego may be a *fusion* result
            # alias in the V2V4Real trace. Replaying it as generic edge work is
            # inconsistent even though the binary value exists in the file.
            fusion_key = self.w.get('fusion_candidate_key')
            if (self.w['backend'] == 'trace' and fusion_key and
                    candidate.plan.execution_mode != 'cooperative_late_fusion' and
                    any(s.kind == 'compute' and
                        (s.candidate_key or f'{s.candidate_location}:{s.source_agent}') == fusion_key
                        for s in graph.stages)):
                raise ValueError(f'Candidate {candidate.candidate_id!r} aliases recorded '
                                 f'cooperative output {fusion_key!r}; use provenance-valid candidates')
            risk = self.risk_estimator.estimate(observation, candidate,
                                                failure_threshold=len(observation.agent_ids))
            cost = self.resource_estimator.estimate(observation, candidate, graph)
            if not isinstance(risk, RiskEstimate) or not isinstance(cost, ResourceEstimate):
                raise TypeError('Estimators must return validated RiskEstimate and ResourceEstimate records')
            if risk.failure_threshold != len(observation.agent_ids) or tuple(name for name,_ in risk.marginal_by_agent) != observation.agent_ids:
                raise ValueError('Risk estimate does not cover the requested agent team')
            assessments.append(CandidateAssessment(candidate.candidate_id, candidate.plan, risk, cost))
            records.append({
                'batch_id': batch.batch_id, 'frame_id': batch.frame_id,
                'time_ms': self.loop.now_us / 1000,
                'candidate_id': candidate.candidate_id,
                'risk': asdict(risk), 'cost': asdict(cost),
                'public_scene_features': dict(observation.scene_features),
                'snapshot_in_flight_workflows': observation.resource_snapshot.in_flight_workflows,
                'selected': False,
            })
        if not assessments:
            raise ValueError('No feasible candidate plans in the allowlisted inventory')
        # Commit together: an invalid candidate must not leave a partial catalog.
        self.candidate_estimates.extend(records)
        return observation, tuple(assessments)

    def _release(self, event: WorldEvent) -> None:
        """Keep the legacy singleton entry point for external simulation adapters."""
        self._release_batch(ExecutionBatch(event.event_id, (event,),
                                           event.arrival_slot if event.arrival_slot is not None else event.index,
                                           None))

    def _release_batch(self, batch: ExecutionBatch) -> None:
        event = batch.events[0]
        policy_draw = uniform(self.s['seed'], batch.batch_id, 'policy', 'share-mixture')
        deadline_us = self.loop.now_us + to_us(self.s['deadline_ms'])
        selected_candidate = None
        if self.coord['enabled']:
            observation, assessments = self._assess_candidates(batch, deadline_us, policy_draw)
            if self.coordinator_selector is not None:
                selected_candidate = self.coordinator_selector.select(observation, assessments)
            elif self.c['policy']['name'] == 'candidate_fixed':
                selected_candidate = self.c['policy']['candidate_id']
            if selected_candidate is not None:
                choices = {a.candidate_id: a for a in assessments}
                if selected_candidate not in choices:
                    raise ValueError(f'Policy selected unavailable candidate {selected_candidate!r}')
                plan = replace(choices[selected_candidate].plan,
                               policy_name=self.c['policy']['name'])
                for record in self.candidate_estimates[-len(assessments):]:
                    record['selected'] = record['candidate_id'] == selected_candidate
        if selected_candidate is None:
            view = CoordinatorView(
                event_id=batch.batch_id, agents=event.agents,
                deadline_ms=self.s['deadline_ms'],
                policy_draw=policy_draw,
                # Explicit policy assumptions, NEVER read true world parameters here.
                estimated_edge_error=float(self.c['policy'].get('assumed_edge_error', 0.06)),
                estimated_separate_correlation=float(self.c['policy'].get('assumed_edge_correlation', 0.0)),
            )
            plan = self.policy.plan(view)
        validate_plan(plan, event.agents)
        if self.coord['enabled'] and selected_candidate is None:
            # Existing BasicPolicy may run in observation mode, but it must
            # still select a provenance-eligible candidate from the catalog.
            # Compare physical allocations, not policy names/probabilities.
            matching = [a.candidate_id for a in assessments if
                        a.plan.execution_mode == plan.execution_mode and a.plan.groups == plan.groups]
            if not matching:
                raise ValueError('Selected legacy allocation is not in the explicit coordinator candidate inventory')
            selected_candidate = matching[0]
            for record in self.candidate_estimates[-len(assessments):]:
                record['selected'] = record['candidate_id'] == selected_candidate
        released_ms = self.loop.now_us / 1000
        for decision in batch.events:
            self._pending[decision.event_id] = {
                'released_ms': released_ms, 'deadline_us': deadline_us, 'delivered': {},
            }
        # Only physical DAG topology is cached across frames, never outcome values.
        # Per-frame result identities and predictions live inside its executor.
        shape = (plan.execution_mode, event.agents,
                 tuple((g.source_agent, g.location, g.consumers) for g in plan.groups))
        if shape not in self._graph_cache:
            self._graph_cache[shape] = compile_execution_graph(
                plan, event.agents, self.r, self.w.get('fusion_candidate_key'))
        graph = self._graph_cache[shape]
        for decision in batch.events:
            self.policy_plans.append({
                'execution_mode': plan.execution_mode,
                'stage_count': len(graph.stages),
                'event_id': decision.event_id, 'frame_id': batch.frame_id,
                'physical_execution_id': batch.batch_id,
                'policy': plan.policy_name,
                'groups': [
                    {'location': g.location, 'source_agent': g.source_agent,
                     'consumers': list(g.consumers)} for g in plan.groups
                ],
                'estimated_full_reuse_probability': plan.estimated_full_reuse_probability,
                'predicted_infeasible': plan.predicted_infeasible,
                **({'selected_candidate_id': selected_candidate} if self.coord['enabled'] else {}),
            })
            self._log('event_released', decision.event_id, agents=list(decision.agents),
                      frame_id=batch.frame_id, physical_execution_id=batch.batch_id)
            # Score every ROI independently, against its own ground truth.
            self.loop.at(deadline_us + 1, lambda e=decision: self._score(e))
        self._active_batches.add(batch.batch_id)
        GraphExecutor(self, batch, graph).start()

    def _score(self, event: WorldEvent) -> None:
        pending = self._pending[event.event_id]  # retained for late-delivery audit
        delivered = pending['delivered']
        wrong = {}
        missing = {}
        decisions = {}
        latency = {}
        result_ids = {}
        for agent in event.agents:
            entry = delivered.get(agent)
            missing[agent] = entry is None
            wrong[agent] = None if entry is None else bool(entry['prediction'] != event.truth)
            decisions[agent] = None if entry is None else entry['prediction']
            latency[agent] = None if entry is None else entry['latency_ms']
            result_ids[agent] = None if entry is None else entry['result_id']
        self.scores.append({
            'event_id': event.event_id, 'frame_id': event.frame_id, 'event_index': event.index,
            'released_ms': pending['released_ms'], 'ground_truth': event.truth,
            'predictions': decisions, 'wrong': wrong, 'missing': missing,
            'latency_ms': latency, 'result_ids': result_ids,
            'error_count': sum(v is True for v in wrong.values()),
            'operational_failure_count': sum(v is None or v is True for v in wrong.values()),
            'all_wrong': all(v is True for v in wrong.values()),
            'any_wrong': any(v is True for v in wrong.values()),
            'all_operational_fail': all(v is None or v is True for v in wrong.values()),
            'complete': all(not v for v in missing.values()),
        })
        self._log('event_scored', event.event_id,
                  all_wrong=self.scores[-1]['all_wrong'], complete=self.scores[-1]['complete'])

    def run(self) -> SimulationResult:
        interval_us = to_us(self.s['arrival_interval_ms'])
        for batch in self.world.batches:
            self.loop.at(batch.arrival_slot * interval_us,
                         lambda b=batch: self._release_batch(b))
        self.loop.run()
        # Deterministic report ordering independent of scheduling interleavings.
        self.scores.sort(key=lambda e: e['event_index'])
        if self.fusion_decisions:
            event_order = {e.event_id:e.index for e in self.world.events}
            self.fusion_decisions.sort(key=lambda d: event_order[d['event_id']])
        resources = {
            'edge_inference_executions': self.edge_resource.jobs,
            'local_inference_executions': sum(r.jobs for r in self.local_resources.values()),
            'edge_cpu_busy_ms': self.edge_resource.busy_us / 1000,
            'local_cpu_busy_ms': sum(r.busy_us for r in self.local_resources.values()) / 1000,
            'edge_queue_wait_ms': self.edge_resource.queue_wait_us / 1000,
            'local_queue_wait_ms': sum(r.queue_wait_us for r in self.local_resources.values()) / 1000,
            'transmitted_bytes': self.bytes_transmitted,
            'messages_attempted': self.messages_attempted,
            'messages_lost': self.messages_lost,
            'link_busy_ms': self.link.busy_us / 1000,
            'link_queue_wait_ms': self.link.queue_wait_us / 1000,
            'peak_pending_events': self.loop.peak_pending_events,
            'simulated_duration_ms': self.loop.now_us / 1000,
            'compute_executions_by_model': dict(sorted(self.compute_count_by_model.items())),
            'compute_busy_ms_by_model': dict(sorted(self.compute_busy_by_model.items())),
            'transmitted_bytes_by_role': dict(sorted(self.bytes_by_transfer_role.items())),
            'execution_granularity': self.w['execution_granularity'],
            'physical_execution_batches': len(self.world.batches),
            'frame_projection_cache_hits': self.frame_projection_cache_hits,
            'frame_projection_cache_hits_by_model': dict(sorted(self.frame_projection_cache_hits_by_model.items())),
            'decision_events_sharing_physical_frame': sum(len(b.events) for b in self.world.batches if len(b.events) > 1),
        }
        if self.energy is not None:
            energy_report = self.energy.finalize()
            successful = sum(s['complete'] and not s['any_wrong'] for s in self.scores)
            energy_report['energy_successful_team_decisions'] = successful
            energy_report['energy_incremental_j_per_successful_team_decision'] = (
                energy_report['energy_workload_incremental_j'] / successful if successful else None)
            energy_report['energy_total_j_per_successful_team_decision'] = (
                energy_report['energy_total_j'] / successful if successful else None)
            resources.update(energy_report)
        summary = summarize(self.scores, resources, self.policy_plans)
        summary['scenario'] = self.s['name']
        summary['policy'] = self.c['policy']['name']
        summary['world_backend'] = self.w['backend']
        summary['seed'] = self.s['seed']
        if self.c['policy']['name'] == 'selective_fusion_f0':
            from collections import Counter
            counts = Counter(d['action'] for d in self.fusion_decisions)
            summary['f0_fusion_decisions'] = len(self.fusion_decisions)
            summary['f0_shared_roi_decisions'] = sum(
                counts[x] for x in ('SHARE_FUSED','MASK_AND_FUSE'))
            summary['f0_masked_fusion_decisions'] = counts['MASK_AND_FUSE']
            summary['f0_keep_local_roi_decisions'] = counts['KEEP_LOCAL']
            summary['f0_shared_roi_coverage'] = (summary['f0_shared_roi_decisions'] / len(self.fusion_decisions)
                                                  if self.fusion_decisions else 0)
            summary['f0_realized_agent_overrides'] = sum(
                len(d.get('applied_agents', [])) for d in self.fusion_decisions)
            summary['f0_fallback_reason_counts'] = dict(sorted(Counter(
                reason for d in self.fusion_decisions for reason in d['fallback_reasons']).items()))
        if self.coord['enabled']:
            summary['coordination_candidate_evaluations'] = len(self.candidate_estimates)
            summary['coordination_model'] = 'v0.7_policy_visible_synthetic_estimates'
        return SimulationResult(self.c, summary, self.scores,
                                self.inferences, self.events, self.policy_plans, self.operations,
                                self.candidate_estimates if self.coord['enabled'] else None,
                                self.fusion_decisions if self.c['policy']['name'] == 'selective_fusion_f0' else None)


def run_simulation(config: ExperimentConfig, detailed_events: bool = True, **coordination_plugins) -> SimulationResult:
    """Run with optional injected V0.7 providers; default behavior is unchanged."""
    return Simulator(config, detailed_events=detailed_events, **coordination_plugins).run()
