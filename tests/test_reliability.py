import pytest
from octopus_sim.simulator import run_simulation
from octopus_sim.policies import BasicPolicy, CoordinatorView


@pytest.mark.parametrize('policy,k,expected', [
    ('local_only', None, .01),
    ('always_independent', None, .0036),
    ('always_share', None, .06),
    ('randomized_reuse', .5, .0036 + .5*(.06-.0036)),
])
def test_mobihoc_model_monte_carlo(build, policy, k, expected):
    args = {'reuse_probability': k} if k is not None else {}
    cfg = build(policy, episodes=20000, policy_args=args)
    result = run_simulation(cfg, detailed_events=False)
    j = result.summary['joint_all_error_rate']
    # Generous but statistically meaningful tolerance for stochastic Monte Carlo.
    assert abs(j - expected) < .005, (policy, j, expected)
    assert result.summary['complete_event_rate'] == 1.0
    if policy in ('always_share','randomized_reuse','always_independent'):
        assert abs(result.summary['marginal_error_rate_all_events']-.06) < .008


def test_common_cause_correlation_separate_outputs(build):
    f, rho = .09, .4
    n = 3
    result = run_simulation(build('always_independent', agents=('a','b','c'),
                            episodes=22000, edge_error=f, edge_correlation=rho),
                            detailed_events=False)
    expected = rho*f + (1-rho)*f**n
    assert abs(result.summary['joint_all_error_rate'] - expected) < .008


def test_correlated_separate_results_no_shared_provenance(build):
    result = run_simulation(build('always_independent', agents=('a','b','c'), episodes=5,
                                  edge_correlation=1))
    for score in result.scores:
        assert len(set(score['result_ids'].values())) == 3
        assert len({i['source_agent'] for i in result.inferences
                    if i['event_id']==score['event_id']}) == 3
        # With fully common latent error, all predictions coincide even though
        # all inference executions and observation IDs remain distinct.
        assert len(set(score['predictions'].values())) == 1


def test_risk_budget_analytical_threshold(build):
    cfg = build('risk_budget', episodes=6000, policy_args={
        'joint_failure_budget': .01, 'assumed_edge_error': .06,
        'assumed_edge_correlation': 0})
    result = run_simulation(cfg, detailed_events=False)
    threshold = (.01 - .06**2)/(.06-.06**2)
    assert abs(result.summary['events_with_shared_inference_rate']-threshold) < .025
    assert result.summary['joint_all_error_rate'] <= .018
    assert result.summary['predicted_infeasible_event_rate'] == 0


def test_infeasible_budget_is_reported(build):
    cfg = build('risk_budget', episodes=10, policy_args={
        'joint_failure_budget': 0.001,
        'assumed_edge_error': .06, 'assumed_edge_correlation': 0})
    result = run_simulation(cfg)
    assert result.summary['predicted_infeasible_event_rate'] == 1.0
    assert result.summary['events_with_shared_inference_rate'] == 0


def test_policy_view_does_not_contain_world_truth():
    view = CoordinatorView('event_0', ('a','b'), 100, .5, .06, 0.0)
    assert 'truth' not in vars(view)
    assert 'edge_correlation' not in vars(view)
    assert 'ground_truth' not in vars(view)
    assert 'seed' not in vars(view)
    assert len(BasicPolicy('always_share').plan(view).groups) == 1


def test_policy_reuse_keeps_source_agent_outcome_fixed(build):
    common = dict(agents=('a','b'), episodes=300, edge_correlation=.3)
    shared = run_simulation(build('always_share', **common), detailed_events=False)
    separate = run_simulation(build('always_independent', **common), detailed_events=False)
    for a,b in zip(shared.scores, separate.scores):
        assert a['predictions']['a'] == b['predictions']['a']


def test_multiagent_tail_risk_monotone_and_extremes(build):
    r=run_simulation(build('partial_share', agents=('a','b','c','d'), episodes=300,
              policy_args={'max_edge_executions': 2}), detailed_events=False)
    tail=r.summary['tail_error_rates_by_m']
    assert set(tail)=={'1','2','3','4'}
    assert all(tail[str(m)] >= tail[str(m+1)] for m in range(1,4))
    assert tail['1']==r.summary['any_error_rate']
    assert tail['4']==r.summary['joint_all_error_rate']
    assert len(r.summary['per_agent_rates'])==4
