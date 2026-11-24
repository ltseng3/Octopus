"""Validated, minimal user configuration. No heavyweight schema dependency."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import json
import math
from typing import Any


class ConfigurationError(ValueError):
    pass


DEFAULTS: dict[str, Any] = {
    'schema_version': 1,
    'scenario': {
        'name': 'demo', 'seed': 42, 'episodes': 1000,
        'agents': ['vehicle_a', 'vehicle_b'],
        'arrival_interval_ms': 250.0, 'deadline_ms': 200.0,
    },
    'world': {
        'backend': 'synthetic', 'execution_granularity': 'decision', 'local_error': 0.10, 'edge_error': 0.06,
        'local_correlation': 0.0, 'edge_correlation': 0.0,
    },
    'resources': {
        'edge_workers': 1, 'edge_inference_ms': 30.0,
        'local_inference_ms': 15.0, 'network_channels': 1,
        'network_latency_ms': 5.0, 'network_bandwidth_mbps': 100.0,
        'observation_bytes': 1024, 'result_bytes': 128,
        'packet_loss': 0.0, 'model_profiles': {},
        'f0_decision_model_id': 'f0_decider',
        'model_bindings': {'local': 'local_model', 'edge': 'edge_model', 'fusion': 'late_fusion'},
    },
    'policy': {'name': 'always_share', 'reuse_probability': 0.5},
    'fusion_f0': {
        # No probabilities or dependence are inferred from ground truth.
        'error_by_agent': {}, 'dependence_group_by_agent': {}, 'quality_proxy_by_agent': {},
        'prior_positive': 0.5,
        'max_source_error': 0.25, 'max_age_ms': 120.0,
        'dependence_discount': 1.0, 'min_independent_groups': 2,
        'min_supporting_groups': 2, 'max_conflict_fraction': 0.4,
        'min_score_confidence': 0.9,
        'require_risk_bounds': False, 'max_shared_error_upper': None,
        'max_risk_increase_over_local': None,
    },
    # Optional V0.7 policy-visible estimates: no change to default execution.
    'coordination': {
        'enabled': False,
        'visible_features': [],
        'resource_visibility': 'edge_only',  # or reported_all (zero-cost fresh telemetry assumption)
        'synthetic_features': {},
        'candidate_ids': [],
        'risk_estimator': {
            'kind': 'constant', 'local_error': 0.10, 'edge_error': 0.06,
            'fusion_error': 0.06, 'correlation': 0.0,
            'context_feature': None, 'context_slope': 0.0, 'bias': 0.0,
        },
    },
    'energy': {
        'enabled': False, 'calibration': 'hypothetical', 'calibration_source': None,
        'measurement_window_ms': None,
        'device_profiles': {'agent': {}, 'edge': {}},
        'agent_overrides': {}, 'model_incremental_w': {}, 'model_energy_j': {},
    },
}


def _merge(base: dict, addition: dict, scope: str = '') -> dict:
    if not isinstance(addition, dict):
        raise ConfigurationError(f'{scope or "root"} must be a mapping')
    result = deepcopy(base)
    allowed_extras = {
        'world': {'trace_file', 'fusion_candidate_key'},
        'policy': {'joint_failure_budget', 'assumed_edge_error',
                   'assumed_edge_correlation', 'max_edge_executions', 'candidate_id'},
    }
    for key, value in addition.items():
        if scope in {'agent_overrides', 'model_incremental_w', 'model_energy_j', 'synthetic_features', 'error_by_agent', 'dependence_group_by_agent', 'quality_proxy_by_agent'}:
            if not isinstance(key, str) or not key:
                raise ConfigurationError(f'{scope} keys must be nonempty strings')
            result[key] = deepcopy(value)
            continue
        if scope in {'agent', 'edge'} and key not in result:
            # The full profile is validated below; no hidden default power values.
            result[key] = deepcopy(value)
            continue
        if scope == 'model_profiles':
            if not isinstance(key, str) or not key or not isinstance(value, dict):
                raise ConfigurationError('model_profiles requires named model mappings')
            result[key] = deepcopy(value)
            continue
        if key not in result and key not in allowed_extras.get(scope, set()):
            raise ConfigurationError(f'Unknown configuration field: {scope + "." if scope else ""}{key}')
        if key in result and isinstance(result[key], dict):
            if not isinstance(value, dict):
                raise ConfigurationError(f'{key} must be a mapping')
            result[key] = _merge(result[key], value, key)
        else:
            result[key] = value
    return result


def _prob(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f'{label} must be a number between 0 and 1')
    value = float(value)
    if not 0 <= value <= 1:
        raise ConfigurationError(f'{label} must be between 0 and 1')
    return value


def _nonnegative(value: object, label: str, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f'{label} must be a number')
    invalid = (not math.isfinite(value)) or (value <= 0 if strict else value < 0)
    if invalid:
        raise ConfigurationError(f'{label} must be {"positive" if strict else "nonnegative"}')
    return float(value)


def _integer(value: object, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f'{label} must be an integer >= {minimum}')
    return value


@dataclass(frozen=True)
class ExperimentConfig:
    data: dict[str, Any]
    source_path: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], source_path: Path | None = None) -> 'ExperimentConfig':
        if not isinstance(data, dict):
            raise ConfigurationError('Root configuration must be a mapping')
        result = _merge(DEFAULTS, data)
        if result['schema_version'] != 1:
            raise ConfigurationError('Only schema_version: 1 is supported')
        s, w, r, p = (result[k] for k in ('scenario', 'world', 'resources', 'policy'))
        f0 = result['fusion_f0']
        energy = result['energy']
        coordination = result['coordination']
        _integer(s['seed'], 'scenario.seed', 0)
        _integer(s['episodes'], 'scenario.episodes')
        agents = s['agents']
        if not isinstance(agents, list) or len(agents) < 2 or not all(isinstance(x, str) and x for x in agents):
            raise ConfigurationError('scenario.agents must have >=2 nonempty string IDs')
        if len(set(agents)) != len(agents):
            raise ConfigurationError('scenario.agents must be unique')
        if not isinstance(s['name'], str) or not s['name']:
            raise ConfigurationError('scenario.name must be nonempty')
        _nonnegative(s['arrival_interval_ms'], 'scenario.arrival_interval_ms')
        _nonnegative(s['deadline_ms'], 'scenario.deadline_ms', strict=True)
        if w['backend'] not in {'synthetic', 'trace'}:
            raise ConfigurationError('world.backend must be synthetic or trace')
        if w['execution_granularity'] not in {'decision', 'frame'}:
            raise ConfigurationError('world.execution_granularity must be decision or frame')
        if w['execution_granularity'] == 'frame' and w['backend'] != 'trace':
            raise ConfigurationError('frame execution requires a recorded trace with explicit frame_id')
        for k in ('local_error', 'edge_error', 'local_correlation', 'edge_correlation'):
            _prob(w[k], 'world.' + k)
        if 'fusion_candidate_key' in w and (not isinstance(w['fusion_candidate_key'], str)
                                           or not w['fusion_candidate_key']):
            raise ConfigurationError('world.fusion_candidate_key must be a nonempty string')
        if w['backend'] == 'trace':
            if not isinstance(w.get('trace_file'), str) or not w['trace_file']:
                raise ConfigurationError('world.trace_file is mandatory for trace backend')
        # Coordinator observations are opt-in and include only explicitly
        # allowlisted public signals, never hidden ground truth/error draws.
        if type(coordination['enabled']) is not bool:
            raise ConfigurationError('coordination.enabled must be boolean')
        if coordination['resource_visibility'] not in {'edge_only', 'reported_all'}:
            raise ConfigurationError('coordination.resource_visibility must be edge_only or reported_all')
        visible = coordination['visible_features']
        if not isinstance(visible, list) or any(not isinstance(x, str) or not x for x in visible):
            raise ConfigurationError('coordination.visible_features must contain nonempty names')
        if len(visible) != len(set(visible)):
            raise ConfigurationError('Duplicate visible feature names')
        forbidden = {'truth', 'ground_truth', 'error', 'wrong', 'latent_error', 'oracle'}
        if any(x.lower() in forbidden for x in visible):
            raise ConfigurationError('Ground truth/error variable cannot be policy-visible feature')
        sf = coordination['synthetic_features']
        if not isinstance(sf, dict):
            raise ConfigurationError('coordination.synthetic_features must be mapping')
        if sf and w['backend'] != 'synthetic':
            raise ConfigurationError('Synthetic feature generators require synthetic backend')
        if set(sf) - set(visible):
            raise ConfigurationError('Synthetic feature must be explicitly visible')
        for name, spec in sf.items():
            if not isinstance(spec, dict) or spec.get('kind') not in {'constant', 'uniform'}:
                raise ConfigurationError(f'Invalid synthetic feature generator: {name}')
            expected = ({'kind','value'} if spec['kind'] == 'constant' else {'kind','low','high'})
            if set(spec) != expected:
                raise ConfigurationError(f'Synthetic feature {name} requires {sorted(expected)}')
            for key in expected - {'kind'}:
                if isinstance(spec[key], bool) or not isinstance(spec[key], (int, float)) or not math.isfinite(spec[key]):
                    raise ConfigurationError(f'Synthetic feature {name}.{key} must be finite')
            if spec['kind'] == 'uniform' and spec['low'] > spec['high']:
                raise ConfigurationError('Synthetic feature uniform low must not exceed high')
        ids = coordination['candidate_ids']
        if not isinstance(ids, list) or any(not isinstance(x,str) or not x for x in ids) or len(ids) != len(set(ids)):
            raise ConfigurationError('coordination.candidate_ids must be unique nonempty names')
        if coordination['enabled'] and w['backend'] == 'trace' and not ids:
            raise ConfigurationError('Trace coordination requires explicit candidate_ids (prediction provenance is not inferred)')
        re = coordination['risk_estimator']
        if re['kind'] not in {'constant','contextual','misspecified'}:
            raise ConfigurationError('Unsupported synthetic risk estimator kind')
        for k in ('local_error','edge_error','fusion_error','correlation'):
            _prob(re[k], 'coordination.risk_estimator.'+k)
        for k in ('context_slope', 'bias'):
            if isinstance(re[k],bool) or not isinstance(re[k],(int,float)) or not math.isfinite(re[k]):
                raise ConfigurationError('Invalid synthetic estimator parameter '+k)
        if re['kind'] == 'contextual' and (not isinstance(re['context_feature'],str)
                or re['context_feature'] not in visible):
            raise ConfigurationError('Contextual estimator requires allowlisted context_feature')
        if re['kind'] != 'contextual' and re['context_feature'] is not None:
            raise ConfigurationError('Only contextual estimator may set context_feature')
        if coordination['enabled'] and w['backend'] == 'synthetic' and set(visible) != set(sf):
            raise ConfigurationError('Every visible synthetic feature requires an explicit generator')
        if p['name'] == 'candidate_fixed':
            if not coordination['enabled']:
                raise ConfigurationError('candidate_fixed requires coordination.enabled')
            if not isinstance(p.get('candidate_id'), str) or not p['candidate_id']:
                raise ConfigurationError('candidate_fixed requires policy.candidate_id')
            if ids and p['candidate_id'] not in ids:
                raise ConfigurationError('Selected candidate must be in candidate_ids')
        if p['name'] == 'selective_fusion_f0':
            from octopus_fusion.contracts import FusionConfig
            if coordination['enabled']:
                raise ConfigurationError('F0 is a post-inference controller; V0.7 pre-inference candidate estimation must be disabled')
            for mid in (r['model_bindings']['local'], r['f0_decision_model_id']):
                if mid not in r['model_profiles']:
                    raise ConfigurationError(f'F0 requires explicit model profile for {mid}')
            if r['model_profiles'][r['f0_decision_model_id']]['device'] != 'edge':
                raise ConfigurationError('F0 decision compute must run on edge')
            if r['model_profiles'][r['model_bindings']['local']]['device'] != 'local':
                raise ConfigurationError('F0 detector compute must run locally')
            if set(f0['error_by_agent']) != set(agents):
                raise ConfigurationError('fusion_f0.error_by_agent must explicitly cover every agent')
            for agent, q in f0['error_by_agent'].items():
                _prob(q, f'fusion_f0.error_by_agent.{agent}')
                if q in (0, 1):
                    raise ConfigurationError('F0 error estimates must be strictly between 0 and 1')
            if not isinstance(f0['dependence_group_by_agent'], dict) or (set(f0['dependence_group_by_agent']) - set(agents)):
                raise ConfigurationError('F0 dependence groups may name only known agents')
            if any(not isinstance(v,str) or not v.strip() for v in f0['dependence_group_by_agent'].values()):
                raise ConfigurationError('F0 dependence group IDs must be nonempty strings')
            quality = f0['quality_proxy_by_agent']
            if not isinstance(quality, dict) or (quality and set(quality) != set(agents)):
                raise ConfigurationError('F0 quality_proxy_by_agent must be empty or cover every agent')
            for a, v in quality.items():
                _prob(v, f'fusion_f0.quality_proxy_by_agent.{a}')
            _prob(f0['prior_positive'], 'fusion_f0.prior_positive')
            if f0['prior_positive'] in (0, 1):
                raise ConfigurationError('F0 prior_positive must be strictly between 0 and 1')
            try:
                FusionConfig(**{k:v for k,v in f0.items() if k not in
                                {'error_by_agent','dependence_group_by_agent','quality_proxy_by_agent','prior_positive'}})
            except (ValueError, TypeError) as exc:
                raise ConfigurationError(f'Invalid fusion_f0 settings: {exc}') from exc
            if f0['require_risk_bounds'] or f0['max_shared_error_upper'] is not None or f0['max_risk_increase_over_local'] is not None:
                raise ConfigurationError('External risk bounds require a validated F0 estimator; static F1 integration cannot claim them')
            if w['backend'] == 'trace' and w['execution_granularity'] != 'frame':
                raise ConfigurationError('F0 trace replay requires physical frame execution')
        if type(energy['enabled']) is not bool:
            raise ConfigurationError('energy.enabled must be a boolean')
        if energy['calibration'] not in {'hypothetical', 'measured', 'mixed'}:
            raise ConfigurationError('energy.calibration must be hypothetical, measured, or mixed')
        if energy['calibration_source'] is not None and (not isinstance(energy['calibration_source'], str) or not energy['calibration_source'].strip()):
            raise ConfigurationError('energy.calibration_source must be a nonempty string or null')
        if energy['calibration'] != 'hypothetical' and (not isinstance(energy['calibration_source'], str) or
                                                        not energy['calibration_source'].strip()):
            raise ConfigurationError('Measured/mixed energy calibration requires calibration_source')
        if energy['measurement_window_ms'] is not None:
            _nonnegative(energy['measurement_window_ms'], 'energy.measurement_window_ms', strict=True)
        def validate_radio(radio: object, label: str) -> None:
            keys = {'mode','tx_j_per_bit','rx_j_per_bit','tx_incremental_w','rx_incremental_w'}
            if not isinstance(radio, dict) or set(radio) != keys:
                raise ConfigurationError(f'{label} requires exactly {sorted(keys)}')
            if radio['mode'] not in {'per_bit', 'state'}:
                raise ConfigurationError(f'{label}.mode must be per_bit or state')
            for key in keys - {'mode'}:
                _nonnegative(radio[key], f'{label}.{key}')
            forbidden = ({'tx_incremental_w','rx_incremental_w'} if radio['mode'] == 'per_bit'
                         else {'tx_j_per_bit','rx_j_per_bit'})
            if any(radio[k] != 0 for k in forbidden):
                raise ConfigurationError(f'{label} cannot mix per-bit and state radio energy')
        def validate_device(profile: object, label: str) -> None:
            if not isinstance(profile, dict) or set(profile) != {'baseline_w','compute_incremental_w','radio'}:
                raise ConfigurationError(f'{label} requires baseline_w, compute_incremental_w, radio')
            for key in ('baseline_w', 'compute_incremental_w'):
                _nonnegative(profile[key], f'{label}.{key}')
            validate_radio(profile['radio'], f'{label}.radio')
        if not isinstance(energy['model_energy_j'], dict):
            raise ConfigurationError('energy.model_energy_j must be a mapping')
        if set(energy['model_energy_j']) & set(energy['model_incremental_w']):
            raise ConfigurationError('Specify model_energy_j or model_incremental_w for each model, not both')
        for key, val in energy['model_energy_j'].items():
            _nonnegative(val, f'energy.model_energy_j.{key}')
        if not isinstance(energy['model_incremental_w'], dict):
            raise ConfigurationError('energy.model_incremental_w must be a mapping')
        if not isinstance(energy['agent_overrides'], dict):
            raise ConfigurationError('energy.agent_overrides must be a mapping')
        for key, val in energy['model_incremental_w'].items():
            _nonnegative(val, f'energy.model_incremental_w.{key}')
        for key, profile in energy['agent_overrides'].items():
            if key not in agents:
                raise ConfigurationError(f'Unknown energy agent override {key}')
            validate_device(profile, f'energy.agent_overrides.{key}')
        if energy['enabled']:
            if 'edge' in agents:
                raise ConfigurationError('edge is reserved as a device identifier when energy is enabled')
            raw_energy = data.get('energy', {})
            if (not isinstance(raw_energy, dict) or
                not isinstance(raw_energy.get('device_profiles'), dict) or
                set(raw_energy['device_profiles']) != {'agent', 'edge'}):
                raise ConfigurationError('Energy requires explicit agent and edge device profiles')
            for category in ('agent','edge'):
                validate_device(energy['device_profiles'][category], f'energy.device_profiles.{category}')
            if r['network_channels'] != 1 and any(
                prof['radio']['mode'] == 'state' for prof in
                [*energy['device_profiles'].values(), *energy['agent_overrides'].values()]):
                raise ConfigurationError('Radio state mode currently requires one shared network channel')
        for k in ('edge_workers', 'network_channels'):
            _integer(r[k], 'resources.' + k)
        for k in ('observation_bytes', 'result_bytes'):
            _integer(r[k], 'resources.' + k, minimum=0)
        for k in ('edge_inference_ms', 'local_inference_ms', 'network_latency_ms'):
            _nonnegative(r[k], 'resources.' + k)
        _nonnegative(r['network_bandwidth_mbps'], 'resources.network_bandwidth_mbps', strict=True)
        _prob(r['packet_loss'], 'resources.packet_loss')
        if not isinstance(r['f0_decision_model_id'], str) or not r['f0_decision_model_id']:
            raise ConfigurationError('resources.f0_decision_model_id must be a model ID')
        profiles = r['model_profiles']
        if not isinstance(profiles, dict):
            raise ConfigurationError('resources.model_profiles must be a mapping')
        bindings = r['model_bindings']
        if not isinstance(bindings, dict) or set(bindings) != {'local', 'edge', 'fusion'}:
            raise ConfigurationError('resources.model_bindings must specify local, edge, fusion')
        if any(not isinstance(v, str) or not v for v in bindings.values()):
            raise ConfigurationError('resources.model_bindings values must be model names')
        for model, profile in profiles.items():
            if not isinstance(model, str) or not model or not isinstance(profile, dict):
                raise ConfigurationError('Invalid model profile name or entry')
            if set(profile) != {'device', 'compute_ms', 'output_bytes'}:
                raise ConfigurationError('model profile requires exactly device, compute_ms, output_bytes')
            if profile['device'] not in {'local', 'edge'}:
                raise ConfigurationError('model profile device must be local or edge')
            _nonnegative(profile['compute_ms'], f'model_profiles.{model}.compute_ms')
            _integer(profile['output_bytes'], f'model_profiles.{model}.output_bytes', minimum=0)
        for mid in set(energy['model_incremental_w']) | set(energy['model_energy_j']):
            if mid not in profiles and mid not in {'local_model', 'edge_model'}:
                raise ConfigurationError(f'Unknown model in energy.model_incremental_w: {mid}')
        for role, device in [('local', 'local'), ('edge', 'edge'), ('fusion', 'edge')]:
            mid = bindings[role]
            if mid in profiles and profiles[mid]['device'] != device:
                raise ConfigurationError(f'{role} binding {mid} must use {device} device')
            if (mid not in profiles and mid not in {'local_model', 'edge_model'} and
                    (role != 'fusion' or p['name'] in {'cooperative_late_fusion', 'mixed_local_fusion'})):
                raise ConfigurationError(f'No model profile for {role} binding {mid}')
        allowed = {'selective_fusion_f0', 'cooperative_late_fusion', 'mixed_local_fusion', 'local_only', 'always_share', 'always_independent', 'randomized_reuse', 'mixed_local_shared',
                   'risk_budget', 'partial_share', 'candidate_fixed'}
        if p['name'] not in allowed:
            raise ConfigurationError(f'Unknown policy {p["name"]!r}; choices: {sorted(allowed)}')
        _prob(p['reuse_probability'], 'policy.reuse_probability')
        if p['name'] in {'cooperative_late_fusion', 'mixed_local_fusion'}:
            for role in ('local', 'fusion'):
                mid = bindings[role]
                if mid not in profiles:
                    raise ConfigurationError(f'{p["name"]} requires resources.model_profiles.{mid}')
            if w['backend'] == 'trace' and not w.get('fusion_candidate_key'):
                raise ConfigurationError('Trace fusion requires explicit world.fusion_candidate_key')
        if p['name'] == 'risk_budget':
            if 'joint_failure_budget' not in p:
                raise ConfigurationError('risk_budget requires policy.joint_failure_budget')
            _prob(p['joint_failure_budget'], 'policy.joint_failure_budget')
            if 'assumed_edge_error' in p:
                _prob(p['assumed_edge_error'], 'policy.assumed_edge_error')
            if 'assumed_edge_correlation' in p:
                _prob(p['assumed_edge_correlation'], 'policy.assumed_edge_correlation')
        if p['name'] == 'partial_share':
            k = _integer(p.get('max_edge_executions', 2), 'policy.max_edge_executions')
            if k > len(agents):
                raise ConfigurationError('max_edge_executions exceeds agents')
        return cls(result, source_path)

    @classmethod
    def load(cls, path: str | Path) -> 'ExperimentConfig':
        path = Path(path).expanduser().resolve()
        text = path.read_text(encoding='utf-8')
        if path.suffix.lower() in {'.yaml', '.yml'}:
            try:
                import yaml
            except ImportError as exc:
                raise ConfigurationError('YAML requires PyYAML; use the JSON example otherwise') from exc
            raw = yaml.safe_load(text)
        elif path.suffix.lower() == '.json':
            raw = json.loads(text)
        else:
            raise ConfigurationError('Scenario path must end in .json, .yaml or .yml')
        return cls.from_dict(raw, path)

    def resolve_trace(self) -> Path:
        loc = Path(self.data['world']['trace_file'])
        if not loc.is_absolute() and self.source_path is not None:
            loc = self.source_path.parent / loc
        return loc.resolve()

    def with_policy(self, name: str, **kwargs: Any) -> 'ExperimentConfig':
        config = deepcopy(self.data)
        config['policy'] = {'name': name, 'reuse_probability': 0.5, **kwargs}
        return ExperimentConfig.from_dict(config, self.source_path)
