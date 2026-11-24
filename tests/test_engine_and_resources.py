from octopus_sim.engine import EventLoop, FIFOResource, to_us
from octopus_sim.simulator import run_simulation


def test_event_loop_order_and_resource_fifo():
    loop = EventLoop()
    worker = FIFOResource(loop, 1)
    sequence=[]
    loop.at(0, lambda: (worker.request(100, lambda: sequence.append((loop.now_us,'A'))),
                        worker.request(50, lambda: sequence.append((loop.now_us,'B')))))
    loop.run()
    assert sequence == [(100,'A'), (150,'B')]
    assert worker.queue_wait_us == 100
    assert worker.busy_us == 150


def test_shared_inference_one_execution_three_deliveries(build):
    r = run_simulation(build('always_share', agents=('a','b','c'), episodes=2))
    assert r.summary['edge_inference_executions'] == 2
    assert r.summary['messages_attempted'] == 2*4
    assert r.summary['transmitted_bytes'] == 2*(1024+3*128)
    for e in r.scores:
        assert len(set(e['result_ids'].values())) == 1
    for i in r.inferences:
        assert len(i['consumers_requested']) == 3
        assert len(i['consumers_delivered']) == 3


def test_independent_edge_executes_per_agent(build):
    r = run_simulation(build('always_independent', agents=('a','b','c'), episodes=2))
    assert r.summary['edge_inference_executions'] == 6
    assert r.summary['transmitted_bytes'] == 2*3*(1024+128)
    for s in r.scores:
        assert len(set(s['result_ids'].values())) == 3


def test_local_requires_no_network(build):
    r=run_simulation(build('local_only', agents=('a','b','c'), episodes=3))
    assert r.summary['edge_inference_executions'] == 0
    assert r.summary['local_inference_executions'] == 9
    assert r.summary['transmitted_bytes'] == 0
    assert r.summary['complete_event_rate'] == 1


def test_no_deadline_contamination_and_late_arrivals(build):
    from octopus_sim.config import ExperimentConfig
    cfg = build('always_share', episodes=3, network_ms=50)
    data=cfg.data
    data['scenario']['deadline_ms'] = 2
    cfg = ExperimentConfig.from_dict(data)
    r = run_simulation(cfg)
    assert r.summary['complete_event_rate'] == 0
    assert r.summary['joint_all_error_rate'] == 0
    assert r.summary['all_operational_failure_rate'] == 1
    assert any(e['type'] == 'result_arrived_late' for e in r.events)
    assert all(s['result_ids']['a'] is None for s in r.scores)


def test_packet_loss_and_local_fallback(build):
    edge = run_simulation(build('always_share', episodes=3, packet_loss=1))
    assert edge.summary['agent_deadline_miss_rate'] == 1
    assert edge.summary['messages_lost'] == 3
    local = run_simulation(build('local_only', episodes=3, packet_loss=1))
    assert local.summary['agent_deadline_miss_rate'] == 0


def test_queueing_changes_latency(build):
    cfg=build('always_independent', episodes=3,
              agents=('a','b','c'), edge_workers=1, edge_ms=30)
    result=run_simulation(cfg)
    assert result.summary['edge_inference_executions'] == 9
    assert result.summary['edge_queue_wait_ms'] > 0
    assert result.summary['mean_delivered_latency_ms'] > 30


def test_reproducibility_complete_results(build):
    cfg=build('randomized_reuse', episodes=30, edge_correlation=.2,
              policy_args={'reuse_probability': .43})
    a = run_simulation(cfg)
    b = run_simulation(cfg)
    assert a.summary == b.summary
    assert a.scores == b.scores
    assert a.inferences == b.inferences
    assert a.events == b.events
    assert a.policy_plans == b.policy_plans


def test_partial_share_three_agents(build):
    r = run_simulation(build('partial_share', agents=('a','b','c'),
                             episodes=4, policy_args={'max_edge_executions': 2}))
    assert r.summary['edge_inference_executions'] == 8
    for s in r.scores:
        assert s['result_ids']['a'] == s['result_ids']['b']
        assert s['result_ids']['c'] != s['result_ids']['a']
