"""Immutable, validated execution DAGs for edge-coordinated inference.

This module owns *intent*, not scheduling or statistical inference. A policy
chooses an inference allocation. The compiler turns that allocation into
explicit compute/transfer/delivery operations. An executor interprets those
operations against a clock, resources and a prediction backend.

An operation is uniquely identified within its physical execution batch
(a decision event in legacy mode or an explicitly identified frame).
A compute operation creates exactly one inference artifact; transfers do not
create new inference artifacts. Consequently, a fan-out delivery preserves result identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from .policies import Plan


StageKind = Literal['compute', 'transfer', 'deliver', 'conditional_deliver']


@dataclass(frozen=True)
class Stage:
    stage_id: str
    kind: StageKind
    dependencies: tuple[str, ...] = ()
    # Compute semantics
    model_id: str | None = None
    device: str | None = None  # 'edge' or an agent ID
    source_agent: str | None = None
    candidate_location: str | None = None  # current backend: 'local' or 'edge'
    candidate_key: str | None = None  # explicit trace key for multi-source fusion
    observation_agents: tuple[str, ...] = ()
    parent_compute_stages: tuple[str, ...] = ()
    # Transfer semantics
    sender: str | None = None
    recipient: str | None = None
    bytes: int = 0
    transfer_role: str | None = None  # observation, detection, result
    # Delivery semantics: both direct and network-delivery are explicit stages
    result_from: str | None = None
    consumer: str | None = None


@dataclass(frozen=True)
class ExecutionGraph:
    stages: tuple[Stage, ...]
    agents: tuple[str, ...]
    workflow: str

    def validate(self) -> None:
        if not self.stages:
            raise ValueError('Execution graph may not be empty')
        lookup = {s.stage_id: s for s in self.stages}
        if len(lookup) != len(self.stages):
            raise ValueError('Duplicate stage ID')
        if not self.agents or len(set(self.agents)) != len(self.agents):
            raise ValueError('Graph agents must be unique and nonempty')
        delivered = []
        conditional_delivered = []
        for stage in self.stages:
            if not stage.stage_id or stage.kind not in {'compute', 'transfer', 'deliver', 'conditional_deliver'}:
                raise ValueError('Invalid stage ID or kind')
            if len(set(stage.dependencies)) != len(stage.dependencies):
                raise ValueError(f'Duplicate dependencies in {stage.stage_id}')
            if any(dep not in lookup for dep in stage.dependencies):
                raise ValueError(f'Unknown dependency in {stage.stage_id}')
            if stage.kind == 'compute':
                if not stage.model_id or not stage.device or not stage.source_agent:
                    raise ValueError('Compute stage requires model, device, source')
                if stage.candidate_location not in {'local', 'edge', 'f0'}:
                    raise ValueError('Unknown candidate backend')
                if stage.candidate_location == 'f0' and self.workflow != 'selective_fusion_f0':
                    raise ValueError('F0 compute only valid in F0 graph')
                if stage.source_agent not in self.agents:
                    raise ValueError('Unknown inference source agent')
                if stage.device not in ('edge', *self.agents):
                    raise ValueError('Unknown compute device')
                if not stage.observation_agents or any(a not in self.agents for a in stage.observation_agents):
                    raise ValueError('Invalid observation ancestry')
                if any(p not in stage.dependencies and not self._ancestor(p, stage.stage_id, lookup)
                       for p in stage.parent_compute_stages):
                    raise ValueError('Parent inference must be upstream of child')
                if any(lookup.get(p, stage).kind != 'compute' for p in stage.parent_compute_stages):
                    raise ValueError('Parent inference must refer to a compute stage')
            elif stage.kind == 'transfer':
                if stage.sender not in ('edge', *self.agents) or stage.recipient not in ('edge', *self.agents):
                    raise ValueError('Transfer endpoints must be known')
                if stage.sender == stage.recipient:
                    raise ValueError('A transfer must cross devices')
                if type(stage.bytes) is not int or stage.bytes < 0:
                    raise ValueError('Transfer bytes must be a nonnegative integer')
                if stage.transfer_role not in {'observation', 'detection', 'result'}:
                    raise ValueError('Unknown transfer role')
            elif stage.kind in {'deliver', 'conditional_deliver'}:
                if stage.consumer not in self.agents:
                    raise ValueError('Delivery to unknown agent')
                if stage.result_from not in lookup or lookup[stage.result_from].kind != 'compute':
                    raise ValueError('Delivery must reference a compute result')
                if not self._ancestor(stage.result_from, stage.stage_id, lookup):
                    raise ValueError('Delivery must depend on its inference result')
                if stage.kind == 'conditional_deliver':
                    if self.workflow != 'selective_fusion_f0':
                        raise ValueError('Conditional delivery only valid in F0 graph')
                    if stage.result_from != 'f0_decide' or stage.consumer in conditional_delivered:
                        raise ValueError('F0 conditional delivery must uniquely consume F0 result')
                    conditional_delivered.append(stage.consumer)
                else:
                    delivered.append(stage.consumer)
        if sorted(delivered) != sorted(self.agents):
            raise ValueError('Exactly one delivery per participating agent required')
        if self.workflow == 'selective_fusion_f0' and sorted(conditional_delivered) != sorted(self.agents):
            raise ValueError('F0 requires exactly one conditional delivery per agent')
        visiting: set[str] = set()
        visited: set[str] = set()
        def visit(stage_id: str) -> None:
            if stage_id in visiting:
                raise ValueError('Cycle in execution graph')
            if stage_id in visited:
                return
            visiting.add(stage_id)
            for dep in lookup[stage_id].dependencies:
                visit(dep)
            visiting.remove(stage_id)
            visited.add(stage_id)
        for stage in self.stages:
            visit(stage.stage_id)

    @staticmethod
    def _ancestor(ancestor: str, child: str, lookup: Mapping[str, Stage]) -> bool:
        if child not in lookup or ancestor not in lookup or ancestor == child:
            return False
        stack = list(lookup[child].dependencies)
        seen = set()
        while stack:
            name = stack.pop()
            if name == ancestor:
                return True
            if name not in seen:
                seen.add(name)
                stack.extend(lookup[name].dependencies)
        return False


def effective_profiles(resources: dict) -> dict[str, dict]:
    """Resolve explicit cost profiles, preserving v0.3 settings as defaults.

    Values are reference cost units, not measured hardware performance.
    """
    profiles = {
        'local_model': {'device': 'local', 'compute_ms': resources['local_inference_ms'],
                        'output_bytes': resources['result_bytes']},
        'edge_model': {'device': 'edge', 'compute_ms': resources['edge_inference_ms'],
                       'output_bytes': resources['result_bytes']},
    }
    profiles.update(resources.get('model_profiles', {}))
    return profiles


def compile_execution_graph(plan: Plan, agents: tuple[str, ...], resources: dict,
                            fusion_candidate_key: str | None = None) -> ExecutionGraph:
    """Compile policy allocation into an executable DAG, with no hidden work.

    The cooperative late-fusion workflow performs per-agent local detection,
    uploads *each* detection, performs a single fusion compute operation, then
    distributes the same inference artifact to all agents. The edge model's
    prediction is a replay candidate; the simulator does not run neural fusion.
    """
    profiles = effective_profiles(resources)
    bindings = resources.get('model_bindings', {'local': 'local_model', 'edge': 'edge_model',
                                               'fusion': 'late_fusion'})
    stages: list[Stage] = []

    def compute(stage_id: str, deps: tuple[str, ...], model: str, device: str,
                source: str, location: str, observation_agents: tuple[str, ...],
                parents: tuple[str, ...] = (), candidate_key: str | None = None) -> None:
        if model not in profiles:
            raise ValueError(f'Missing model profile {model!r}')
        expected = profiles[model]['device']
        if (device == 'edge') != (expected == 'edge'):
            raise ValueError(f'Model {model!r} cannot execute on {device!r}')
        stages.append(Stage(stage_id, 'compute', deps, model_id=model, device=device,
                            source_agent=source, candidate_location=location,
                            candidate_key=candidate_key,
                            observation_agents=observation_agents,
                            parent_compute_stages=parents))

    def transfer(stage_id: str, deps: tuple[str, ...], sender: str, recipient: str,
                 size: int, role: str) -> None:
        stages.append(Stage(stage_id, 'transfer', deps, sender=sender, recipient=recipient,
                            bytes=size, transfer_role=role))

    def deliver(stage_id: str, deps: tuple[str, ...], result_from: str, consumer: str, *, conditional: bool = False) -> None:
        stages.append(Stage(stage_id, 'conditional_deliver' if conditional else 'deliver',
                            deps, result_from=result_from, consumer=consumer))

    if plan.execution_mode == 'selective_fusion_f0':
        f0 = resources.get('f0_decision_model_id', 'f0_decider')
        if bindings['local'] not in resources.get('model_profiles', {}):
            raise ValueError('F0 requires explicit local detector model profile')
        if f0 not in resources.get('model_profiles', {}):
            raise ValueError('F0 requires an explicit edge decision-model profile')
        uploads, parents = [], []
        for agent in agents:
            cid = f'local:{agent}'
            uid = f'detection_upload:{agent}'
            compute(cid, (), bindings['local'], agent, agent, 'local', (agent,))
            # Optimistic local-first service. F0 can later override with on-time
            # shared evidence; rejecting fusion does not undo the local decision.
            deliver(f'local_deliver:{agent}', (cid,), cid, agent)
            transfer(uid, (cid,), agent, 'edge', profiles[bindings['local']]['output_bytes'], 'detection')
            parents.append(cid)
            uploads.append(uid)
        compute('f0_decide', tuple(uploads), f0, 'edge', agents[0], 'f0',
                agents, tuple(parents))
        for agent in agents:
            down = f'f0_result_down:{agent}'
            transfer(down, ('f0_decide',), 'edge', agent,
                     profiles[f0]['output_bytes'], 'result')
            deliver(f'f0_conditional_deliver:{agent}', (down,), 'f0_decide', agent,
                    conditional=True)
    elif plan.execution_mode == 'cooperative_late_fusion':
        if bindings['fusion'] not in resources.get('model_profiles', {}):
            raise ValueError('cooperative_late_fusion requires a fusion model profile')
        if bindings['local'] not in resources.get('model_profiles', {}):
            raise ValueError('cooperative_late_fusion requires a local model profile')
        inputs = []
        parents = []
        for agent in agents:
            cid = f'local:{agent}'
            uid = f'detection_upload:{agent}'
            compute(cid, (), bindings['local'], agent, agent, 'local', (agent,))
            transfer(uid, (cid,), agent, 'edge', profiles[bindings['local']]['output_bytes'], 'detection')
            parents.append(cid)
            inputs.append(uid)
        fused = 'late_fusion'
        compute(fused, tuple(inputs), bindings['fusion'], 'edge', agents[0],
                'edge', agents, tuple(parents), candidate_key=fusion_candidate_key)
        for agent in agents:
            down = f'result_down:{agent}'
            transfer(down, (fused,), 'edge', agent, profiles[bindings['fusion']]['output_bytes'], 'result')
            deliver(f'deliver:{agent}', (down,), fused, agent)
    elif plan.execution_mode == 'groups':
        for i, group in enumerate(plan.groups):
            cid = f'group:{i}:compute'
            if group.location == 'edge':
                up = f'group:{i}:observation_upload'
                transfer(up, (), group.source_agent, 'edge', resources['observation_bytes'], 'observation')
                compute(cid, (up,), bindings['edge'], 'edge', group.source_agent, 'edge', (group.source_agent,))
                for consumer in group.consumers:
                    down = f'group:{i}:down:{consumer}'
                    transfer(down, (cid,), 'edge', consumer, profiles[bindings['edge']]['output_bytes'], 'result')
                    deliver(f'group:{i}:deliver:{consumer}', (down,), cid, consumer)
            else:
                compute(cid, (), bindings['local'], group.source_agent, group.source_agent,
                        'local', (group.source_agent,))
                deliver(f'group:{i}:deliver:{group.source_agent}', (cid,), cid, group.source_agent)
    else:
        raise ValueError(f'Unsupported execution mode {plan.execution_mode}')
    graph = ExecutionGraph(tuple(stages), agents, plan.execution_mode)
    graph.validate()
    return graph
