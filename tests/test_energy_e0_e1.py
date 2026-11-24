"""Energy E0 invariants: actual service, full resource DAG, reproducibility."""
from __future__ import annotations
from copy import deepcopy
import math

import pytest

from octopus_sim.config import ConfigurationError, ExperimentConfig
from octopus_sim.engine import EventLoop, FIFOResource
from octopus_sim.energy import EnergyLedger
from octopus_sim.simulator import run_simulation


def radio(mode='per_bit', tx=0.000001, rx=0.0000005, tx_w=0, rx_w=0):
    return dict(mode=mode, tx_j_per_bit=tx if mode=='per_bit' else 0,
                rx_j_per_bit=rx if mode=='per_bit' else 0,
                tx_incremental_w=tx_w if mode=='state' else 0,
                rx_incremental_w=rx_w if mode=='state' else 0)


def energy_config(*, mode='per_bit', horizon=None, overrides=None):
    return dict(enabled=True, calibration='hypothetical', calibration_source='illustrative only',
                measurement_window_ms=horizon,
                device_profiles={
                    'agent': dict(baseline_w=2, compute_incremental_w=4,
                                  radio=radio(mode, tx_w=3, rx_w=1)),
                    'edge': dict(baseline_w=10, compute_incremental_w=8,
                                  radio=radio(mode, tx_w=2, rx_w=1)),
                }, agent_overrides=overrides or {}, model_incremental_w={})


def scenario(policy='local_only', *, agents=('a','b'), episodes=1,
             arrival_ms=1000, deadline_ms=300, bandwidth=100, channels=1,
             edge_workers=1, energy=None, packet_loss=0):
    d=dict(scenario=dict(name='energy_test',seed=8,episodes=episodes,agents=list(agents),
                         arrival_interval_ms=arrival_ms,deadline_ms=deadline_ms),
           policy=dict(name=policy),
           resources=dict(local_inference_ms=20,edge_inference_ms=30,
                          observation_bytes=800,result_bytes=100,
                          edge_workers=edge_workers,network_channels=channels,
                          network_latency_ms=2,network_bandwidth_mbps=bandwidth,
                          packet_loss=packet_loss))
    if policy in {'cooperative_late_fusion', 'mixed_local_fusion'}:
        d['resources']['model_profiles']={'local_model':{'device':'local','compute_ms':20,'output_bytes':800},
                                          'late_fusion':{'device':'edge','compute_ms':10,'output_bytes':100}}
    if energy is not None: d['energy']=energy
    return ExperimentConfig.from_dict(d)


def test_idle_devices_and_baseline_once_per_physical_device():
    e=energy_config()
    config=scenario(energy=e,episodes=1,deadline_ms=100)
    run=run_simulation(config)
    # 100 ms means 0.1 seconds for each of two agents and edge.
    assert run.summary['energy_baseline_j']==pytest.approx((2+2+10)*.1)
    assert run.summary['energy_compute_incremental_j']==pytest.approx(2*4*.02)
    assert run.summary['energy_radio_incremental_j']==0
    assert run.summary['energy_total_j']==pytest.approx(1.4+.16)
    assert run.summary['energy_vehicle_total_j']==pytest.approx(2*(2*.1+4*.02))


def test_disabled_energy_preserves_old_results():
    original=scenario('always_share',episodes=8)
    with_energy_disabled=deepcopy(original.data)
    with_energy_disabled['energy']['enabled']=False
    a=run_simulation(original)
    b=run_simulation(ExperimentConfig.from_dict(with_energy_disabled))
    assert a.summary == b.summary
    assert a.events == b.events
    assert a.operations == b.operations
    assert not any(k.startswith('energy_') for k in a.summary)


def test_actual_queue_service_not_request_wait():
    c=scenario('always_share',episodes=1,edge_workers=1,energy=energy_config())
    s=run_simulation(c).summary
    assert s['edge_inference_executions']==1
    assert s['energy_compute_by_model_j']['edge_model']==pytest.approx(.03*8)
    # Separate queueing across episodes extends wall-clock time, not total joules per job.
    d=deepcopy(c.data);d['scenario']['episodes']=4;d['scenario']['arrival_interval_ms']=0
    d['scenario']['deadline_ms']=900
    out=run_simulation(ExperimentConfig.from_dict(d)).summary
    assert out['edge_queue_wait_ms']>0
    assert out['energy_compute_by_model_j']['edge_model']==pytest.approx(4*.03*8)
    assert out['energy_measurement_window_ms']==900


def test_callback_starts_on_actual_worker_availability():
    loop=EventLoop();res=FIFOResource(loop,1);events=[]
    def go():
        res.request(100, lambda: events.append(('finish1',loop.now_us)),
                    on_start=lambda:events.append(('start1',loop.now_us)))
        res.request(40, lambda:events.append(('finish2',loop.now_us)),
                    on_start=lambda:events.append(('start2',loop.now_us)))
    loop.at(0,go);loop.run()
    assert events==[('start1',0),('finish1',100),('start2',100),('finish2',140)]


def test_full_fusion_three_agents_exact_joules_and_bytes():
    d=scenario('cooperative_late_fusion',agents=('a','b','c'),energy=energy_config())
    # A fusion DAG has 3 local detectors, 3 uploads, 1 fused edge result,
    # 3 downlinks. All model / payload timings declared explicitly.
    raw=deepcopy(d.data)
    raw['resources']['model_profiles']={
        'local_model': {'device':'local','compute_ms':20,'output_bytes':1024},
        'late_fusion': {'device':'edge','compute_ms':10,'output_bytes':128},
    }
    raw['energy']['model_incremental_w']={'late_fusion':12}
    d=ExperimentConfig.from_dict(raw)
    r=run_simulation(d);s=r.summary
    assert s['local_inference_executions']==3
    assert s['edge_inference_executions']==1
    assert s['transmitted_bytes_by_role']=={'detection':3072,'result':384}
    assert s['energy_compute_by_model_j']['local_model']==pytest.approx(3*.02*4)
    assert s['energy_compute_by_model_j']['late_fusion']==pytest.approx(.01*12)
    expect_bits=8*(3072+384)
    assert s['energy_radio_tx_incremental_j']==pytest.approx(expect_bits*.000001)
    assert s['energy_radio_rx_incremental_j']==pytest.approx(expect_bits*.0000005)
    assert s['energy_workload_incremental_j']==pytest.approx(.36+expect_bits*.0000015)
    assert s['energy_total_j']==pytest.approx(s['energy_baseline_j']+s['energy_workload_incremental_j'])
    assert len({v['result_id'] for v in r.inferences if v['model_id']=='late_fusion'})==1


def test_energy_allocation_by_device_matches_total():
    r=run_simulation(scenario('always_share',agents=('a','b','c'),episodes=5,
                              energy=energy_config()))
    s=r.summary;dev=s['energy_by_device_j']
    assert s['energy_total_j']==pytest.approx(sum(x['total_j'] for x in dev.values()))
    assert s['energy_vehicle_total_j']==pytest.approx(sum(x['total_j'] for k,x in dev.items() if k!='edge'))
    assert s['energy_edge_total_j']==pytest.approx(dev['edge']['total_j'])
    assert s['energy_compute_incremental_j']==pytest.approx(sum(s['energy_compute_by_model_j'].values()))
    assert s['energy_radio_incremental_j']==pytest.approx(sum(s['energy_radio_by_transfer_role_j'].values()))


def test_packet_loss_still_consumes_airtime_energy():
    full=run_simulation(scenario('always_share',energy=energy_config(),packet_loss=0)).summary
    lost=run_simulation(scenario('always_share',energy=energy_config(),packet_loss=1)).summary
    assert lost['messages_lost']==1
    assert lost['edge_inference_executions']==0
    assert lost['energy_radio_incremental_j']>0
    assert lost['energy_radio_incremental_j']<full['energy_radio_incremental_j']
    assert lost['energy_compute_by_model_j'].get('edge_model',0)==0


def test_radio_state_mode_integration_and_rate_sensitivity():
    e=energy_config(mode='state')
    r=run_simulation(scenario('always_share',energy=e,bandwidth=8)).summary
    # First payload 800 bytes, return 2x100; 8 Mbps => 800us + 2x100us.
    assert r['link_busy_ms']==pytest.approx(1.0)
    # Agent A TX 800us @3 W, edge RX 800us @1W,
    # edge TX 200us @2W, two agents each RX 100us @1W.
    assert r['energy_radio_incremental_j']==pytest.approx(.0008*4+.0002*2+.0002*1)
    faster=run_simulation(scenario('always_share',energy=e,bandwidth=16)).summary
    assert faster['energy_radio_incremental_j']==pytest.approx(r['energy_radio_incremental_j']/2)


def test_radio_perbit_energy_independent_of_bandwidth():
    e=energy_config()
    a=run_simulation(scenario('always_share',energy=e,bandwidth=4)).summary
    b=run_simulation(scenario('always_share',energy=e,bandwidth=100)).summary
    assert a['energy_radio_incremental_j']==pytest.approx(b['energy_radio_incremental_j'])
    assert a['link_busy_ms']>b['link_busy_ms']


def test_radio_state_multiple_channels_rejected():
    with pytest.raises(ConfigurationError, match='one shared network channel'):
        scenario(energy=energy_config(mode='state'),channels=2)


def test_horizon_clips_and_preserves_post_window_jobs():
    e=energy_config(horizon=5)
    r=run_simulation(scenario('local_only',energy=e,deadline_ms=100))
    s=r.summary
    assert s['energy_measurement_window_ms']==5
    assert s['energy_baseline_j']==pytest.approx((2+2+10)*.005)
    assert s['energy_compute_incremental_j']==pytest.approx(2*.02*4)
    assert s['energy_post_window_incremental_j']==pytest.approx(2*.015*4)
    assert s['energy_window_total_j']==pytest.approx(s['energy_baseline_j']+2*.005*4)
    assert s['energy_total_j']==pytest.approx(s['energy_window_total_j']+s['energy_post_window_incremental_j'])


def test_radio_horizon_split_on_link_serialization():
    e=energy_config(horizon=1)
    r=run_simulation(scenario('always_share',energy=e,bandwidth=4,deadline_ms=500)).summary
    # One 800-byte upload takes 1.6 ms; only first 1ms is in horizon.
    assert r['energy_radio_incremental_j']>0
    assert r['energy_post_window_incremental_j']>0
    assert r['energy_total_j']==pytest.approx(r['energy_window_total_j']+r['energy_post_window_incremental_j'])


def test_concurrent_workers_do_not_double_count_idle_baseline():
    e=energy_config()
    a=run_simulation(scenario('always_independent',episodes=1,edge_workers=1,energy=e)).summary
    b=run_simulation(scenario('always_independent',episodes=1,edge_workers=2,energy=e)).summary
    assert a['energy_compute_incremental_j']==pytest.approx(b['energy_compute_incremental_j'])
    assert a['energy_baseline_j']==pytest.approx(b['energy_baseline_j'])
    assert a['energy_total_j']==pytest.approx(b['energy_total_j'])
    assert a['energy_edge_total_j']==pytest.approx(b['energy_edge_total_j'])


def test_model_power_override_and_agent_specific_power():
    e=energy_config(overrides={'a':dict(baseline_w=5,compute_incremental_w=8,radio=radio())})
    e['model_incremental_w']={'local_model':3}
    s=run_simulation(scenario('local_only',energy=e)).summary
    assert s['energy_compute_by_model_j']['local_model']==pytest.approx(2*.02*3)
    assert s['energy_baseline_j']==pytest.approx((5+2+10)*.3)


def test_reproducible_summaries_and_no_change_to_errors():
    with_energy=scenario('randomized_reuse',episodes=20,energy=energy_config())
    no_energy=deepcopy(with_energy.data);no_energy['energy']['enabled']=False
    a=run_simulation(with_energy);b=run_simulation(with_energy)
    c=run_simulation(ExperimentConfig.from_dict(no_energy))
    assert a.summary==b.summary
    assert a.scores==b.scores==c.scores
    for key in c.summary:
        assert a.summary[key]==c.summary[key]


def test_energy_five_agent_fusion_provenance():
    d=deepcopy(scenario('local_only',agents=tuple(f'a{i}' for i in range(5)),
                       energy=energy_config()).data)
    d['policy']['name']='cooperative_late_fusion'
    d['resources']['model_profiles']={'local_model':{'device':'local','compute_ms':10,'output_bytes':200},
                                      'late_fusion':{'device':'edge','compute_ms':4,'output_bytes':50}}
    s=run_simulation(ExperimentConfig.from_dict(d)).summary
    assert s['local_inference_executions']==5
    assert s['edge_inference_executions']==1
    assert s['messages_attempted']==10
    assert s['energy_radio_incremental_j']==pytest.approx(8*1250*(.000001+.0000005))


@pytest.mark.parametrize('mutate',[
    lambda e: e.update({'calibration':'measured','calibration_source':None}),
    lambda e: e.update({'enabled':'yes'}),
    lambda e: e.update({'measurement_window_ms':-1}),
    lambda e: e['device_profiles']['agent'].update({'baseline_w':-1}),
    lambda e: e['device_profiles']['agent']['radio'].update({'mode':'unknown'}),
    lambda e: e['device_profiles']['agent']['radio'].update({'tx_incremental_w':2}),
    lambda e: e['agent_overrides'].update({'nonexistent':e['device_profiles']['agent']}),
    lambda e: e['model_incremental_w'].update({'bad_model':10}),
    lambda e: e['model_incremental_w'].update({'local_model':float('nan')}),
])
def test_bad_profiles_rejected(mutate):
    e=energy_config();mutate(e)
    with pytest.raises(ConfigurationError): scenario(energy=e)


def test_missing_explicit_profiles_rejected():
    with pytest.raises(ConfigurationError,match='explicit'):
        scenario(energy={'enabled':True})


def test_energy_enabled_without_events_never_silently_uses_default():
    # missing profiles even when requested by API, not just the CLI
    with pytest.raises(ConfigurationError):
        ExperimentConfig.from_dict({'energy':{'enabled':True}})


def test_ledgers_refuse_mismatched_events_and_double_finalization():
    ledger=EnergyLedger(energy_config(),('a','b'),1_000_000,1)
    with pytest.raises(RuntimeError,match='Missing compute'):
        ledger.end_compute('missing',0)
    ledger.begin_compute('x','a','local_model',10)
    with pytest.raises(RuntimeError,match='open jobs'):
        ledger.finalize()
    ledger.end_compute('x',20)
    ledger.finalize()
    with pytest.raises(RuntimeError): ledger.finalize()


def test_fixed_joules_operation_profile_and_exclusive_selection():
    e=energy_config()
    e['model_energy_j']={'local_model':0.375}
    s=run_simulation(scenario('local_only',energy=e)).summary
    assert s['energy_compute_incremental_j']==pytest.approx(.75)
    assert s['energy_compute_by_model_j']['local_model']==pytest.approx(.75)
    e['model_incremental_w']={'local_model':1.0}
    with pytest.raises(ConfigurationError,match='not both'):
        scenario('local_only',energy=e)


def test_zero_duration_operation_fixed_energy_charged_once():
    e=energy_config()
    e['model_energy_j']={'local_model':0.01}
    c=scenario('local_only',energy=e)
    d=deepcopy(c.data);d['resources']['local_inference_ms']=0
    s=run_simulation(ExperimentConfig.from_dict(d)).summary
    assert s['energy_compute_incremental_j']==pytest.approx(.02)


def test_measured_label_requires_declared_source_but_does_not_claim_verification():
    e=energy_config()
    e['calibration']='measured';e['calibration_source']='example lab protocol 2026: no verification'
    s=run_simulation(scenario('local_only',energy=e)).summary
    assert s['energy_calibration']=='measured'
    assert s['energy_calibration_source']==e['calibration_source']


def test_state_and_per_bit_modes_can_mix_by_device_when_link_is_serialized():
    e=energy_config()
    e['agent_overrides']['a']=dict(baseline_w=2,compute_incremental_w=4,
                                  radio=radio('state',tx_w=1.25,rx_w=.75))
    s=run_simulation(scenario('always_share',energy=e,bandwidth=8)).summary
    assert s['energy_radio_incremental_j']>0
    assert s['energy_by_device_j']['a']['radio_tx_j']==pytest.approx(.0008*1.25)


def test_energy_replay_v2v4real_full_fusion_source_verified(root):
    conf=ExperimentConfig.load(root/'examples/energy/v2v4real_0007_energy.json')
    a=run_simulation(conf.with_policy('local_only'),detailed_events=False)
    b=run_simulation(conf.with_policy('cooperative_late_fusion'),detailed_events=False)
    assert a.summary['joint_all_error_count']==4
    assert b.summary['joint_all_error_count']==16
    assert a.summary['energy_workload_incremental_j']==pytest.approx(35.36)
    assert b.summary['energy_workload_incremental_j']==pytest.approx(53.79302656)
    assert b.summary['transmitted_bytes']==1_867_008
    assert a.summary['energy_baseline_j']==pytest.approx(b.summary['energy_baseline_j'])
