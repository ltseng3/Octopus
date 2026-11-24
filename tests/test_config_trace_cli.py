import copy
import json
import subprocess
import sys
import pytest
from octopus_sim.config import ConfigurationError, ExperimentConfig
from octopus_sim.simulator import run_simulation
from octopus_sim.world import TraceError, World
from octopus_sim.policies import Group, Plan, validate_plan
from octopus_sim.metrics import paired_joint_risk_difference, wilson


def test_duplicate_agents_and_bad_rates_rejected():
    with pytest.raises(ConfigurationError):
        ExperimentConfig.from_dict({'scenario':{'agents':['a','a']}})
    with pytest.raises(ConfigurationError):
        ExperimentConfig.from_dict({'world':{'edge_correlation':1.1}})
    with pytest.raises(ConfigurationError):
        ExperimentConfig.from_dict({'world':{'edge_error':-0.1}})
    with pytest.raises(ConfigurationError):
        ExperimentConfig.from_dict({'scenario':{'episodes':0}})
    with pytest.raises(ConfigurationError):
        ExperimentConfig.from_dict({'world':{'unrecognized_setting':1}})


def test_invalid_policy_assignments_rejected():
    with pytest.raises(ValueError):
        validate_plan(Plan((Group(('a','b'),'a','local'),),'bad'), ('a','b'))
    with pytest.raises(ValueError):
        validate_plan(Plan((Group(('a','a'),'a','edge'),),'bad'), ('a','b'))


def test_trace_replay_has_known_outcomes(root):
    cfg = ExperimentConfig.load(root / 'examples/toy_trace.yaml')
    assert len(World(cfg.data, cfg.resolve_trace()).events) == 24
    share = run_simulation(cfg.with_policy('always_share'), detailed_events=False)
    separate = run_simulation(cfg.with_policy('always_independent'), detailed_events=False)
    partial = run_simulation(cfg, detailed_events=False)
    # a's edge inference is incorrect on {2,7,11,21} (four events).
    assert share.summary['joint_all_error_count'] == 4
    # All agents are wrong on event 7 only.
    assert separate.summary['joint_all_error_count'] == 1
    assert partial.summary['joint_all_error_count'] == 1
    assert share.summary['complete_event_rate'] == 1.0


def test_trace_missing_candidates_fails_fast(tmp_path):
    trace = tmp_path / 'bad.jsonl'
    trace.write_text(json.dumps({'event_id':'e1','truth':1,
        'candidates': {'local:a':0}})+'\n')
    cfg = ExperimentConfig.from_dict({
        'scenario':{'agents':['a','b']},
        'world':{'backend':'trace','trace_file':str(trace)},
    })
    with pytest.raises(TraceError):
        World(cfg.data, cfg.resolve_trace())


def test_wilson_and_paired_intervals():
    assert wilson(0,100)[0] == 0
    assert wilson(100,100)[1] == 1
    s1 = [{'event_id':'a','all_wrong':True},{'event_id':'b','all_wrong':False}]
    s2 = [{'event_id':'a','all_wrong':False},{'event_id':'b','all_wrong':False}]
    diff = paired_joint_risk_difference(s1,s2)
    assert diff['difference_a_minus_b'] == .5
    with pytest.raises(ValueError):
        paired_joint_risk_difference(s1, list(reversed(s2)))


def test_cli_creates_all_artifacts(tmp_path, root):
    out = tmp_path / 'results'
    cmd = [sys.executable,'-m','octopus_sim','run', str(root / 'examples/toy_trace.yaml'),
           '--output',str(out)]
    child = subprocess.run(cmd, capture_output=True, text=True, cwd=root, check=True)
    assert 'J_all=' in child.stdout
    for name in ('summary.json','scenario_resolved.json','scores.jsonl',
                 'inferences.jsonl','events.jsonl','plans.jsonl','manifest.json'):
        assert (out/name).exists()
    assert json.loads((out/'manifest.json').read_text())['trace_sha256']


def test_cli_comparison_paired_outputs(tmp_path, root):
    out=tmp_path/'compare'
    cmd=[sys.executable,'-m','octopus_sim','compare',str(root/'examples/toy_trace.yaml'),
         '--policies','always_share,always_independent', '--no-events','--output',str(out)]
    subprocess.run(cmd, cwd=root, check=True, capture_output=True, text=True)
    assert (out/'comparison.csv').exists()
    contrasts=json.loads((out/'paired_comparisons_vs_first.json').read_text())
    assert contrasts['baseline']=='always_share'
    assert contrasts['contrasts']['always_independent']['difference_a_minus_b']==-3/24
