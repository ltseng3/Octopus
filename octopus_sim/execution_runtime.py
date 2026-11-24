"""Run a validated inference DAG using OCTOPUS-Sim's discrete-event resources.

No perception algorithm is reimplemented: prediction values come from synthetic
potential outcomes or strict external traces. This executor models *work* and
*dependencies*, including upstream packet failures, not neural-network fusion.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import math
from typing import TYPE_CHECKING

from .engine import to_us
from .execution import ExecutionGraph, Stage, effective_profiles
from .rng import uniform
from .world import ExecutionBatch, WorldEvent

if TYPE_CHECKING:
    from .simulator import Simulator


class GraphExecutor:
    def __init__(self, sim: 'Simulator', event: WorldEvent | ExecutionBatch, graph: ExecutionGraph):
        graph.validate()
        self.sim = sim
        self.batch = (event if isinstance(event, ExecutionBatch) else
                      ExecutionBatch(event.event_id, (event,),
                                     event.arrival_slot if event.arrival_slot is not None else event.index,
                                     None))
        self.events = self.batch.events
        self.event = self.events[0]  # Legacy scheduler interfaces; NOT the whole frame output.
        self.graph = graph
        self.by_id = {s.stage_id: s for s in graph.stages}
        self.dependents: dict[str, list[str]] = defaultdict(list)
        self.unresolved = {s.stage_id: len(s.dependencies) for s in graph.stages}
        self.status: dict[str, str] = {}
        self.ready_results: dict[str, dict] = {}
        self.started_at_us: dict[str, int] = {}
        self.consumers = defaultdict(list)
        for s in graph.stages:
            for dep in s.dependencies:
                self.dependents[dep].append(s.stage_id)
            if s.kind == 'deliver':
                self.consumers[s.result_from].append(s.consumer)
        self.profiles = effective_profiles(self.sim.r)
        self.f0_by_event: dict[str, dict] = {}

    def _energy_token(self, stage: Stage) -> str:
        return f'{self.batch.batch_id}:{stage.stage_id}'

    def start(self) -> None:
        for stage in self.graph.stages:
            if not stage.dependencies:
                self._start_stage(stage)

    def _log(self, kind: str, **fields) -> None:
        self.sim._log(kind, self.batch.batch_id, **fields)

    def _start_stage(self, stage: Stage) -> None:
        if stage.stage_id in self.status:
            raise RuntimeError(f'Duplicate stage scheduling {stage.stage_id}')
        if (self.graph.workflow == 'selective_fusion_f0' and
                stage.kind == 'transfer' and stage.stage_id.startswith('f0_result_down:') and
                not any(d['action'] != 'KEEP_LOCAL' for d in self.f0_by_event.values())):
            # A fallback after actual edge computation incurs NO result downlink;
            # local inference and upload energy have already been charged.
            self._skip(stage, reason='no_fused_roi_accepted')
            return
        self.status[stage.stage_id] = 'running'
        self.started_at_us[stage.stage_id] = self.sim.loop.now_us
        if stage.kind == 'compute':
            profile = self.profiles[stage.model_id]
            worker = (self.sim.edge_resource if stage.device == 'edge'
                      else self.sim.local_resources[stage.device])
            self.sim.compute_count_by_model[stage.model_id] += 1
            self.sim.compute_busy_by_model[stage.model_id] += to_us(profile['compute_ms']) / 1000
            self._log('compute_requested', stage_id=stage.stage_id,
                      model_id=stage.model_id, device=stage.device,
                      parent_stages=list(stage.parent_compute_stages))
            if self.sim.energy is None:
                worker.request(to_us(profile['compute_ms']), lambda s=stage: self._complete_compute(s))
            else:
                token = self._energy_token(stage)
                def started() -> None:
                    self.sim.energy.begin_compute(token, stage.device, stage.model_id, self.sim.loop.now_us)
                def completed() -> None:
                    self.sim.energy.end_compute(token, self.sim.loop.now_us)
                    self._complete_compute(stage)
                worker.request(to_us(profile['compute_ms']), completed, on_start=started)
        elif stage.kind == 'transfer':
            self.sim.messages_attempted += 1
            self.sim.bytes_transmitted += stage.bytes
            self.sim.bytes_by_transfer_role[stage.transfer_role] += stage.bytes
            duration = math.ceil(stage.bytes * 8 / self.sim.r['network_bandwidth_mbps'])
            lost = (uniform(self.sim.s['seed'], self.batch.batch_id, 'network',
                            stage.stage_id, stage.sender, stage.recipient)
                    < self.sim.r['packet_loss'])
            self._log('message_sent', phase=stage.stage_id, sender=stage.sender,
                      recipient=stage.recipient, bytes=stage.bytes,
                      transfer_role=stage.transfer_role)
            def sent() -> None:
                def arrived() -> None:
                    if lost:
                        self.sim.messages_lost += 1
                        self._log('packet_lost', phase=stage.stage_id, sender=stage.sender,
                                  recipient=stage.recipient, bytes=stage.bytes)
                        self._finish(stage, False, 'packet_lost')
                    else:
                        self._log('message_arrived', phase=stage.stage_id,
                                  sender=stage.sender, recipient=stage.recipient,
                                  bytes=stage.bytes)
                        self._finish(stage, True, 'completed')
                self.sim.loop.after(to_us(self.sim.r['network_latency_ms']), arrived)
            if self.sim.energy is None:
                self.sim.link.request(duration, sent)
            else:
                token = self._energy_token(stage)
                def started_transfer() -> None:
                    self.sim.energy.begin_transfer(token, stage.sender, stage.recipient,
                                                   stage.bytes, stage.transfer_role, self.sim.loop.now_us)
                def completed_transfer() -> None:
                    self.sim.energy.end_transfer(token, self.sim.loop.now_us)
                    sent()
                self.sim.link.request(duration, completed_transfer, on_start=started_transfer)
        elif stage.kind == 'conditional_deliver':
            self._deliver_f0(stage)
        else:
            # One physical result transfer to this agent serves *all* decision
            # regions from the same frame. The ROI projection itself costs no
            # additional inference or network transfer in this benchmark.
            result = self.ready_results[stage.result_from]
            delivered_any = False
            for event in self.events:
                pending = self.sim._pending[event.event_id]
                delivered = pending['delivered']
                on_time = self.sim.loop.now_us <= pending['deadline_us']
                if on_time and stage.consumer not in delivered:
                    delivered[stage.consumer] = {
                        'prediction': result['predictions_by_event_id'][event.event_id],
                        'result_id': result['result_id'],
                        'latency_ms': self.sim.loop.now_us / 1000 - pending['released_ms'],
                    }
                    delivered_any = True
                    result['decision_deliveries'].append({'event_id': event.event_id,
                                                          'agent': stage.consumer})
                self.sim._log('result_delivered' if on_time else 'result_arrived_late',
                              event.event_id, result_id=result['result_id'], agent=stage.consumer)
            if delivered_any:
                result['consumers_delivered'].append(stage.consumer)
            self._finish(stage, True, 'completed')

    def _deliver_f0(self, stage: Stage) -> None:
        record = self.ready_results['f0_decide']
        recipient = stage.consumer
        for event in self.events:
            decision = self.f0_by_event[event.event_id]
            if decision['action'] == 'KEEP_LOCAL':
                continue
            pending = self.sim._pending[event.event_id]
            if self.sim.loop.now_us > pending['deadline_us']:
                self._log('f0_result_arrived_late', stage_id=stage.stage_id,
                          roi_event_id=event.event_id, agent=recipient)
                continue
            pending['delivered'][recipient] = {
                'prediction': record['predictions_by_event_id'][event.event_id],
                'result_id': record['result_id'],
                'latency_ms': self.sim.loop.now_us / 1000 - pending['released_ms'],
            }
            record['decision_deliveries'].append({'event_id':event.event_id,'agent':recipient})
            decision['applied_agents'].append(recipient)
            self._log('f0_result_delivered', stage_id=stage.stage_id,
                      roi_event_id=event.event_id, agent=recipient)
        if any(d['agent'] == recipient for d in record['decision_deliveries']):
            record['consumers_delivered'].append(recipient)
        self._finish(stage, True, 'completed')

    def _complete_f0_compute(self, stage: Stage) -> None:
        from octopus_fusion.contracts import AgentEvidence, FusionObservation
        from octopus_fusion.contracts import Action
        frame_id = self.batch.frame_id or 'event:' + self.batch.batch_id
        result_id = f'{self.batch.batch_id}:result:f0_decide'
        parent_ids = [self.ready_results[f'local:{a}']['result_id'] for a in self.graph.agents]
        accepted = {}
        self.f0_by_event = {}
        for event in self.events:
            evidence = tuple(AgentEvidence(
                agent_id=a,
                prediction=self.ready_results[f'local:{a}']['predictions_by_event_id'][event.event_id],
                result_id=self.ready_results[f'local:{a}']['result_id'],
                dependence_group=self.sim.f0_config['dependence_group_by_agent'].get(a, a),
                produced_at_ms=self.ready_results[f'local:{a}']['created_at_ms'],
                quality_proxy=self.sim.f0_config['quality_proxy_by_agent'].get(a),
            ) for a in self.graph.agents)
            obs = FusionObservation(event.event_id, frame_id,
                                    self.sim.loop.now_us / 1000, evidence,
                                    self.sim.f0_config['prior_positive'])
            risk = self.sim.f0_error_estimator.estimate(obs)
            outcome = self.sim.f0_fuser.decide(obs, risk)
            audit = asdict(outcome)
            audit['action'] = outcome.action.value
            audit['as_of_ms'] = self.sim.loop.now_us / 1000
            audit['input_result_ids'] = parent_ids.copy()
            audit['result_id'] = result_id if outcome.shared else None
            audit['applied_agents'] = []
            self.f0_by_event[event.event_id] = audit
            self.sim.fusion_decisions.append(audit)
            if outcome.shared:
                accepted[event.event_id] = outcome.fused_prediction
        record = {
            'result_id':result_id,'event_id': self.event.event_id if len(self.events) == 1 else None,
            'frame_id':self.batch.frame_id, 'decision_event_ids':[e.event_id for e in self.events],
            'predictions_by_event_id':accepted, 'prediction':None,
            'location':'edge','device':'edge','model_id':stage.model_id,
            'source_agent':None,'source_agents':list(self.graph.agents),
            'candidate_source_agent':None,'candidate_key':'derived:f0:selective_vote',
            'observation_ids':[f'{self.batch.batch_id}:observation:{a}' for a in self.graph.agents],
            'parent_result_ids':parent_ids,'producer_stage_id':stage.stage_id,
            'consumers_requested':list(self.graph.agents), 'consumers_delivered':[],
            'decision_deliveries':[], 'output_bytes':self.profiles[stage.model_id]['output_bytes'],
            'compute_ms':to_us(self.profiles[stage.model_id]['compute_ms'])/1000,
            'created_at_ms':self.sim.loop.now_us/1000,
            'f0_evaluated_event_ids':[e.event_id for e in self.events],
        }
        if len(self.events)>1:
            hits=len(self.events)-1
            self.sim.frame_projection_cache_hits+=hits
            self.sim.frame_projection_cache_hits_by_model[stage.model_id]+=hits
        self.ready_results[stage.stage_id]=record
        self.sim.inferences.append(record)
        from .coordination import PublicInference
        self.sim._edge_public_history.append(PublicInference(
            result_id=result_id,model_id=stage.model_id,
            source_agents=tuple(self.graph.agents),created_at_us=self.sim.loop.now_us))
        self._log('f0_decisions_ready', accepted=len(accepted), total=len(self.events),
                  result_id=result_id, stage_id=stage.stage_id)
        self._finish(stage,True,'completed')

    def _complete_compute(self, stage: Stage) -> None:
        if stage.candidate_location == 'f0':
            self._complete_f0_compute(stage)
            return
        # A frame is a physical inference unit. Its recorded frame-level box
        # output is represented by one projection per evaluated ROI. Do not
        # cache a single binary ROI answer and reuse it for different regions.
        result_id = f'{self.batch.batch_id}:result:{stage.stage_id}'
        predictions = {
            event.event_id: self.sim.world.predict(event, stage.source_agent,
                                                    stage.candidate_location,
                                                    candidate_key=stage.candidate_key)
            for event in self.events
        }
        parents = [self.ready_results[p]['result_id'] for p in stage.parent_compute_stages]
        profile = self.profiles[stage.model_id]
        record = {
            'result_id': result_id,
            'event_id': self.event.event_id if len(self.events) == 1 else None,
            'frame_id': self.batch.frame_id,
            'decision_event_ids': list(predictions),
            'predictions_by_event_id': predictions,
            'location': 'edge' if stage.device == 'edge' else 'local',
            'device': stage.device,
            'model_id': stage.model_id,
            'source_agent': (None if stage.parent_compute_stages else stage.source_agent),
            'source_agents': list(stage.observation_agents),
            'candidate_source_agent': stage.source_agent,
            'candidate_key': stage.candidate_key or f'{stage.candidate_location}:{stage.source_agent}',
            'observation_ids': [f'{self.batch.batch_id}:observation:{a}'
                                for a in stage.observation_agents],
            'parent_result_ids': parents,
            'producer_stage_id': stage.stage_id,
            'consumers_requested': list(self.consumers.get(stage.stage_id, [])),
            'consumers_delivered': [],
            'decision_deliveries': [],
            # Legacy singleton result remains scalar. A frame-level *physical*
            # result has no single binary value; it supplies several ROI views.
            'prediction': next(iter(predictions.values())) if len(predictions) == 1 else None,
            'output_bytes': profile['output_bytes'],
            'compute_ms': to_us(profile['compute_ms']) / 1000,
            'created_at_ms': self.sim.loop.now_us / 1000,
        }
        if len(self.events) > 1:
            self.sim.frame_projection_cache_hits += len(self.events) - 1
            self.sim.frame_projection_cache_hits_by_model[stage.model_id] += len(self.events) - 1
        self.ready_results[stage.stage_id] = record
        self.sim.inferences.append(record)
        if stage.device == 'edge':
            # Central coordinator knows an edge-local result once its compute
            # completes; agent-local results are NOT automatically visible.
            from .coordination import PublicInference
            self.sim._edge_public_history.append(PublicInference(
                result_id=result_id, model_id=stage.model_id,
                source_agents=tuple(stage.observation_agents),
                created_at_us=self.sim.loop.now_us))
        self._log('inference_ready', result_id=result_id, source=stage.source_agent,
                  model_id=stage.model_id, parent_result_ids=parents,
                  projected_decision_events=list(predictions))
        self._finish(stage, True, 'completed')

    def _finish(self, stage: Stage, success: bool, final_status: str) -> None:
        if self.status.get(stage.stage_id) != 'running':
            raise RuntimeError(f'Unexpected stage completion {stage.stage_id}')
        self.status[stage.stage_id] = 'succeeded' if success else 'failed'
        if self.sim.detailed_events:
            self.sim.operations.append({
            'event_id': self.event.event_id if len(self.events) == 1 else None,
            'frame_id': self.batch.frame_id, 'decision_event_ids': [e.event_id for e in self.events],
            'stage_id': stage.stage_id,
            'kind': stage.kind, 'status': final_status,
            'submitted_at_ms': self.started_at_us[stage.stage_id] / 1000,
            'finished_at_ms': self.sim.loop.now_us / 1000,
            'model_id': stage.model_id, 'device': stage.device,
            'bytes': stage.bytes if stage.kind == 'transfer' else 0,
            'transfer_role': stage.transfer_role,
            })
        self._notify(stage.stage_id)

    def _skip(self, stage: Stage, *, reason: str = 'dependency_failed') -> None:
        if stage.stage_id in self.status:
            return
        self.status[stage.stage_id] = 'skipped'
        if stage.stage_id == 'f0_decide':
            # Upload or compute dependency failed: no edge evidence exists.
            # Local outputs were delivered independently if still on time.
            for event in self.events:
                self.sim.fusion_decisions.append({
                    'event_id':event.event_id, 'frame_id':self.batch.frame_id or 'event:'+self.batch.batch_id,
                    'action':'KEEP_LOCAL','fused_prediction':None,
                    'fallback_reasons':['unavailable_edge_evidence'],
                    'masked_agents':[], 'used_groups':[], 'supporting_groups':[],
                    'per_agent_decisions':[(a,self.ready_results[f'local:{a}']['predictions_by_event_id'][event.event_id])
                                           for a in self.graph.agents if f'local:{a}' in self.ready_results],
                    'input_result_ids':[self.ready_results[f'local:{a}']['result_id']
                                        for a in self.graph.agents if f'local:{a}' in self.ready_results],
                    'result_id':None, 'applied_agents':[],
                    'as_of_ms':self.sim.loop.now_us/1000,
                    'estimator_id':'unavailable',
                })
        if self.sim.detailed_events:
            self.sim.operations.append({
            'event_id': self.event.event_id if len(self.events) == 1 else None,
            'frame_id': self.batch.frame_id, 'decision_event_ids': [e.event_id for e in self.events],
            'stage_id': stage.stage_id,
            'kind': stage.kind, 'status': reason,
            'submitted_at_ms': None,
            'finished_at_ms': self.sim.loop.now_us / 1000,
            'model_id': stage.model_id, 'device': stage.device,
            'bytes': 0, 'transfer_role': stage.transfer_role,
        })
        self._log('stage_skipped', stage_id=stage.stage_id, reason=reason)
        self._notify(stage.stage_id)

    def _notify(self, finished_id: str) -> None:
        for next_id in self.dependents[finished_id]:
            self.unresolved[next_id] -= 1
            if self.unresolved[next_id] < 0:
                raise RuntimeError('Invalid DAG dependency accounting')
            if self.unresolved[next_id] == 0:
                following = self.by_id[next_id]
                if all(self.status[d] == 'succeeded' for d in following.dependencies):
                    self._start_stage(following)
                else:
                    self._skip(following)
        if len(self.status) == len(self.graph.stages) and all(
                v in {'succeeded', 'failed', 'skipped'} for v in self.status.values()):
            self.sim._active_batches.discard(self.batch.batch_id)
