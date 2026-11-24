"""Further cross-cutting invariants and multi-agent analytical cases."""
from copy import deepcopy
import pytest
from octopus_sim.config import ExperimentConfig
from octopus_sim.simulator import run_simulation
from octopus_sim.engine import EventLoop


def test_zero_error_and_certain_error_are_exact(build):
    for policy in ('local_only', 'always_independent', 'always_share', 'partial_share'):
        args = {'max_edge_executions': 2} if policy == 'partial_share' else {}
        z = run_simulation(build(policy, agents=('a','b','c'), episodes=30,
                 edge_error=0, local_error=0, policy_args=args), detailed_events=False)
        o = run_simulation(build(policy, agents=('a','b','c'), episodes=30,
                 edge_error=1, local_error=1, policy_args=args), detailed_events=False)
        assert z.summary['joint_all_error_rate'] == 0
        assert o.summary['joint_all_error_rate'] == 1


def test_three_agent_shared_result_and_independent_results_agree_on_marginals(build):
    common = dict(agents=('a','b','c'), episodes=10000, edge_error=.08,
                  edge_correlation=0)
    shared = run_simulation(build('always_share', **common), detailed_events=False)
    separate = run_simulation(build('always_independent', **common), detailed_events=False)
    assert abs(shared.summary['joint_all_error_rate']-.08) < .012
    assert abs(separate.summary['joint_all_error_rate']-.08**3) < .002
    assert abs(shared.summary['marginal_error_rate_all_events'] - .08) < .012
    assert abs(separate.summary['marginal_error_rate_all_events'] - .08) < .012


def test_score_result_provenance_integrity(build):
    result = run_simulation(build('partial_share', agents=('a','b','c','d'), episodes=15,
        policy_args={'max_edge_executions': 2}))
    records = {rec['result_id']:rec for rec in result.inferences}
    for score in result.scores:
        for agent, rid in score['result_ids'].items():
            if rid is not None:
                assert rid in records
                record=records[rid]
                assert record['event_id'] == score['event_id']
                assert agent in record['consumers_delivered']
                assert record['prediction'] == score['predictions'][agent]


def test_no_scheduler_time_travel():
    loop = EventLoop()
    loop.at(100, lambda: None)
    loop.run()
    with pytest.raises(ValueError):
        loop.at(99, lambda: None)


def test_trace_policy_does_not_mutate_scenario(build):
    cfg = build('randomized_reuse', episodes=20, policy_args={'reuse_probability': .33})
    before = deepcopy(cfg.data)
    run_simulation(cfg, detailed_events=False)
    assert cfg.data == before
