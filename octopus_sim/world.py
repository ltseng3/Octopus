"""Synthetic and replay worlds. World truth is private to backend/evaluator."""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
from pathlib import Path
from .rng import uniform


class TraceError(ValueError):
    pass


@dataclass(frozen=True)
class WorldEvent:
    event_id: str
    index: int
    truth: int
    agents: tuple[str, ...]
    candidates: dict[str, int] | None = None
    arrival_slot: int | None = None  # event release slot; not sufficient for frame identity
    frame_id: str | None = None  # explicitly shared physical observation/inference epoch
    public_features: tuple[tuple[str, float], ...] = ()  # explicit ex-ante observable proxies only


@dataclass(frozen=True)
class ExecutionBatch:
    """One physical workflow and one policy selection, possibly serving multiple ROIs."""
    batch_id: str
    events: tuple[WorldEvent, ...]
    arrival_slot: int
    frame_id: str | None


class World:
    def __init__(self, config: dict, trace_path: Path | None = None):
        self.config = config
        self.seed = config['scenario']['seed']
        self.agents = tuple(config['scenario']['agents'])
        self.world_config = config['world']
        if self.world_config['backend'] == 'trace':
            if trace_path is None:
                raise TraceError('Missing trace path')
            self.events = self._read_trace(trace_path)
        else:
            self.events = tuple(WorldEvent(
                event_id=f'event_{i:06d}', index=i,
                truth=int(uniform(self.seed, i, 'truth') < 0.5), agents=self.agents,
            ) for i in range(config['scenario']['episodes']))
        # Preflight frame consistency before scheduling any execution.
        self.batches = self.execution_batches()

    def _read_trace(self, path: Path) -> tuple[WorldEvent, ...]:
        records = []
        seen = set()
        for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                eid, truth, candidates = obj['event_id'], obj['truth'], obj['candidates']
                if not isinstance(eid, str) or not eid or eid in seen:
                    raise TraceError(f'Invalid or duplicate event ID {eid!r}')
                if type(truth) is not int or truth not in (0, 1):
                    raise TraceError('Ground truth must be 0/1')
                if not isinstance(candidates, dict):
                    raise TraceError('candidates must be a map')
                required = [f'local:{a}' for a in self.agents]
                if self.config['policy']['name'] not in {'selective_fusion_f0','local_only'}:
                    required.extend(f'edge:{a}' for a in self.agents)
                if self.config['policy']['name'] != 'selective_fusion_f0' and self.world_config.get('fusion_candidate_key'):
                    required.append(self.world_config['fusion_candidate_key'])
                missing = [key for key in required if key not in candidates]
                if missing:
                    raise TraceError(f'Missing candidates in {eid}: {missing}')
                if any(type(v) is not int or v not in (0, 1) for v in candidates.values()):
                    raise TraceError('Candidate predictions must be 0 or 1')
                slot = obj.get('arrival_slot')
                if slot is not None and (type(slot) is not int or slot < 0):
                    raise TraceError('arrival_slot must be a nonnegative integer')
                frame = obj.get('frame_id')
                if frame is not None and (not isinstance(frame, str) or not frame.strip()):
                    raise TraceError('frame_id must be a nonempty string')
                if self.world_config['execution_granularity'] == 'frame':
                    if frame is None or slot is None:
                        raise TraceError('frame execution requires explicit frame_id and arrival_slot on every event')
                # Preserve old trace compatibility when the public-observation
                # interface is disabled. Previously arbitrary extra trace
                # fields were ignored; do not validate them unexpectedly.
                public = ()
                if self.config['coordination']['enabled']:
                    raw_features = obj.get('public_features', {})
                    if not isinstance(raw_features, dict):
                        raise TraceError('public_features must be a mapping')
                    import math
                    for name, value in raw_features.items():
                        if (not isinstance(name,str) or not name or isinstance(value,bool) or
                                not isinstance(value,(int,float)) or not math.isfinite(value)):
                            raise TraceError('public_features require nonempty names and finite numeric values')
                    visible = set(self.config['coordination']['visible_features'])
                    if not visible.issubset(raw_features):
                        raise TraceError(f'Missing allowlisted public features: {sorted(visible-set(raw_features))}')
                    # Only allowlisted names are visible. Other trace fields
                    # are neither retained nor passed to the coordinator.
                    public = tuple(sorted((name, float(value)) for name,value in raw_features.items()
                                          if name in visible))
                records.append(WorldEvent(eid, len(records), truth, self.agents, candidates, slot, frame, public))
                seen.add(eid)
            except (KeyError, ValueError, TypeError) as exc:
                raise TraceError(f'{path}:{line_no}: {exc}') from exc
        if not records:
            raise TraceError('Trace has no events')
        return tuple(records)

    def execution_batches(self) -> tuple[ExecutionBatch, ...]:
        """Plan grouping is explicit: coincident timestamps alone do NOT enable caching.

        A frame has one immutable observation epoch, one allocation decision and
        one physical DAG. Different region labels are projections of frame outputs.
        Legacy decision-granularity events still compile independently.
        """
        if self.world_config['execution_granularity'] == 'decision':
            return tuple(ExecutionBatch(e.event_id, (e,),
                                        e.arrival_slot if e.arrival_slot is not None else e.index,
                                        None) for e in self.events)
        grouped: dict[str, list[WorldEvent]] = {}
        slots: dict[str, int] = {}
        feature_sets: dict[str, tuple[tuple[str, float], ...]] = {}
        for event in self.events:
            assert event.frame_id is not None and event.arrival_slot is not None
            if event.frame_id in slots and slots[event.frame_id] != event.arrival_slot:
                raise TraceError(f'Frame {event.frame_id!r} spans different arrival slots; cannot share physical work')
            if event.frame_id in feature_sets and feature_sets[event.frame_id] != event.public_features:
                raise TraceError(f'Frame {event.frame_id!r} has inconsistent public features across ROIs')
            feature_sets[event.frame_id] = event.public_features
            slots[event.frame_id] = event.arrival_slot
            grouped.setdefault(event.frame_id, []).append(event)
        return tuple(ExecutionBatch(f'frame:{frame}', tuple(events), slots[frame], frame)
                     for frame, events in grouped.items())

    def predict(self, event: WorldEvent, source_agent: str, location: str,
                candidate_key: str | None = None) -> int:
        if event.candidates is not None:
            return event.candidates[candidate_key or f'{location}:{source_agent}']
        w = self.world_config
        error = float(w['local_error'] if location == 'local' else w['edge_error'])
        rho = float(w['local_correlation'] if location == 'local' else w['edge_correlation'])
        # Latent common shock with probability rho. Separately executed predictions
        # are correlated even without reusing the same inference result.
        shared_mode = uniform(self.seed, event.event_id, location, 'common_mode') < rho
        if shared_mode:
            wrong = uniform(self.seed, event.event_id, location, 'common_error') < error
        else:
            wrong = uniform(self.seed, event.event_id, location, 'individual_error', source_agent) < error
        return event.truth ^ int(wrong)
