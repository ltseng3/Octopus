"""Frame execution is a physical-work identity, not a binary ROI prediction cache."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path

import pytest

from octopus_sim.config import ExperimentConfig, ConfigurationError
from octopus_sim.simulator import run_simulation
from octopus_sim.world import World, TraceError
from tests.test_energy_e0_e1 import energy_config  # internal test fixture, no runtime dependency


def _trace(tmp_path: Path, *, nframes=2, nrois=3, n_agents=3,
           different_slots=False, include_frames=True):
    agents=[f'a{i}' for i in range(n_agents)]
    records=[]
    for frame in range(nframes):
        for roi in range(nrois):
            rec={
                'event_id':f'frame{frame}:roi{roi}',
                'arrival_slot':(frame+roi if different_slots else frame),
                'truth': roi % 2,
                'candidates': {
                    **{f'local:{a}':int((roi+i+frame)%2) for i,a in enumerate(agents)},
                    **{f'edge:{a}':int((roi+i+frame)%2) for i,a in enumerate(agents)},
                    'cooperative:fused': int((roi+frame)%2),
                }
            }
            if include_frames: rec['frame_id']=f'frame{frame}'
            records.append(rec)
    trace=tmp_path/'trace.jsonl'
    trace.write_text(''.join(json.dumps(r)+'\n' for r in records))
    return trace,agents,records


def _config(trace,agents,*,scope='frame',policy='local_only',energy=False,loss=0,
            deadline=2000,channels=1,workers=1,interval=5000):
    return ExperimentConfig.from_dict({
        'scenario':{'agents':agents,'episodes':1,'seed':3,'name':'frame_caching',
                    'arrival_interval_ms':interval,'deadline_ms':deadline},
        'world':{'backend':'trace','trace_file':str(trace),
                 'execution_granularity':scope,'fusion_candidate_key':'cooperative:fused'},
        'resources':{
            'edge_workers':workers,'network_channels':channels,'network_bandwidth_mbps':16,
            'network_latency_ms':2,'packet_loss':loss,
            'model_profiles':{
                'local_model':{'device':'local','compute_ms':12,'output_bytes':512},
                'late_fusion':{'device':'edge','compute_ms':7,'output_bytes':96}},
        },
        'policy':{'name':policy,'reuse_probability':.5},
        **({'energy':energy_config()} if energy else {}),
    })


def _score_signature(r):
    return [(s['event_id'],s['ground_truth'],s['predictions'],s['wrong'],s['all_wrong'])
            for s in r.scores]


@pytest.mark.parametrize('agents', [2,3,5])
@pytest.mark.parametrize('rois', [2,3,4])
@pytest.mark.parametrize('policy', ['local_only','always_independent','always_share','cooperative_late_fusion'])
def test_physical_work_one_per_frame_and_decision_outcomes_preserved(tmp_path,agents,rois,policy):
    trace,names,_=_trace(tmp_path,nframes=2,nrois=rois,n_agents=agents)
    physical=run_simulation(_config(trace,names,policy=policy),False)
    legacy=run_simulation(_config(trace,names,scope='decision',policy=policy),False)
    assert _score_signature(physical)==_score_signature(legacy)
    assert physical.summary['physical_execution_batches']==2
    assert physical.summary['episodes']==2*rois
    if policy in ('local_only','cooperative_late_fusion'):
        assert physical.summary['local_inference_executions']==2*agents
        assert legacy.summary['local_inference_executions']==2*agents*rois
    if policy=='cooperative_late_fusion':
        assert physical.summary['edge_inference_executions']==2
        assert legacy.summary['edge_inference_executions']==2*rois
        assert physical.summary['transmitted_bytes']==2*(agents*512+agents*96)
        assert legacy.summary['transmitted_bytes']==rois*physical.summary['transmitted_bytes']
        assert physical.summary['frame_projection_cache_hits']==2*(agents+1)*(rois-1)
    elif policy=='local_only':
        assert physical.summary['frame_projection_cache_hits']==2*agents*(rois-1)
    elif policy=='always_independent':
        assert physical.summary['edge_inference_executions']==2*agents
    elif policy=='always_share':
        assert physical.summary['edge_inference_executions']==2


def test_one_frame_result_contains_distinct_roi_answers_and_same_physical_id(tmp_path):
    trace,names,_=_trace(tmp_path,nframes=2,nrois=3,n_agents=2)
    r=run_simulation(_config(trace,names,policy='cooperative_late_fusion'),True)
    fused=[x for x in r.inferences if x['model_id']=='late_fusion']
    assert len(fused)==2
    f0=fused[0]
    assert f0['frame_id']=='frame0'
    assert f0['prediction'] is None  # physical frame output is NOT one ROI bit
    assert f0['predictions_by_event_id']=={'frame0:roi0':0,'frame0:roi1':1,'frame0:roi2':0}
    assert set(f0['source_agents'])==set(names)
    assert len(f0['parent_result_ids'])==2
    assert len(f0['decision_deliveries'])==len(names)*3
    for score in r.scores[:3]:
        assert set(score['result_ids'].values())=={f0['result_id']}
        assert score['predictions']==dict.fromkeys(names,f0['predictions_by_event_id'][score['event_id']])
    assert len([op for op in r.operations if op['kind']=='compute'])==6
    assert all(len(op['decision_event_ids'])==3 for op in r.operations)
    assert all(v['observation_ids'][0].startswith('frame:frame') for v in r.inferences)


def test_distinct_frames_same_release_slot_do_not_share(tmp_path):
    trace,names,rows=_trace(tmp_path,nframes=2,nrois=2,n_agents=2)
    for r in rows: r['arrival_slot']=0
    trace.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    out=run_simulation(_config(trace,names,policy='local_only'),False)
    assert out.summary['physical_execution_batches']==2
    assert out.summary['local_inference_executions']==4
    assert out.summary['local_queue_wait_ms']>0
    assert len({r['result_id'] for r in out.inferences})==4


def test_frame_metadata_never_implicitly_enables_caching_in_legacy_scope(tmp_path):
    trace,names,rows=_trace(tmp_path,nframes=1,nrois=3,n_agents=2)
    a=run_simulation(_config(trace,names,scope='decision',policy='local_only'),False)
    assert a.summary['local_inference_executions']==6
    assert a.summary['physical_execution_batches']==3
    for r in rows: r.pop('frame_id')
    trace.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    b=run_simulation(_config(trace,names,scope='decision',policy='local_only'),False)
    assert _score_signature(a)==_score_signature(b)
    assert b.summary['frame_projection_cache_hits']==0


@pytest.mark.parametrize('bad_frame,bad_slot,match', [
    (True,False,'frame_id'),(False,True,'arrival_slot'),
])
def test_missing_explicit_frame_metadata_rejected(tmp_path,bad_frame,bad_slot,match):
    trace,names,records=_trace(tmp_path,nframes=1,nrois=2)
    if bad_frame: records[1].pop('frame_id')
    if bad_slot: records[1].pop('arrival_slot')
    trace.write_text(''.join(json.dumps(r)+'\n' for r in records))
    with pytest.raises(TraceError,match=match):
        World(_config(trace,names).data,trace)


def test_bad_frame_identity_and_mismatched_arrivals_rejected(tmp_path):
    trace,names,records=_trace(tmp_path,nframes=1,nrois=2)
    records[0]['frame_id']='   '
    trace.write_text(''.join(json.dumps(r)+'\n' for r in records))
    with pytest.raises(TraceError,match='frame_id'):
        World(_config(trace,names).data,trace)
    trace,names,_=_trace(tmp_path,nframes=1,nrois=2,different_slots=True)
    with pytest.raises(TraceError,match='different arrival slots'):
        World(_config(trace,names).data,trace)


def test_frame_scope_synthetic_rejected(tmp_path):
    trace,names,_=_trace(tmp_path,nframes=1,nrois=2)
    raw=deepcopy(_config(trace,names).data)
    raw['world']['backend']='synthetic'
    with pytest.raises(ConfigurationError,match='frame execution requires'):
        ExperimentConfig.from_dict(raw)


def test_packet_loss_is_one_physical_attempt_and_energy_charged(tmp_path):
    trace,names,_=_trace(tmp_path,nframes=1,nrois=3,n_agents=3)
    cfg=_config(trace,names,policy='cooperative_late_fusion',energy=True,loss=1)
    r=run_simulation(cfg,True)
    assert r.summary['messages_attempted']==3
    assert r.summary['messages_lost']==3
    assert r.summary['edge_inference_executions']==0
    assert r.summary['energy_radio_incremental_j']>0
    assert all(not s['complete'] for s in r.scores)
    assert all(all(x is None for x in s['result_ids'].values()) for s in r.scores)
    assert len([x for x in r.inferences if x['model_id']=='local_model'])==3


def test_same_frame_energy_saved_without_changing_prediction_content(tmp_path):
    trace,names,_=_trace(tmp_path,nframes=2,nrois=3,n_agents=3)
    physical=run_simulation(_config(trace,names,policy='cooperative_late_fusion',energy=True),False)
    legacy=run_simulation(_config(trace,names,scope='decision',policy='cooperative_late_fusion',energy=True),False)
    assert physical.summary['energy_workload_incremental_j']==pytest.approx(
        legacy.summary['energy_workload_incremental_j']/3)
    assert physical.summary['energy_baseline_j']==legacy.summary['energy_baseline_j']
    assert physical.summary['transmitted_bytes']==legacy.summary['transmitted_bytes']/3
    assert all(s['complete'] for s in physical.scores)


def test_deadline_failures_propagate_to_all_roi_and_post_window_energy(tmp_path):
    trace,names,_=_trace(tmp_path,nframes=1,nrois=3,n_agents=2)
    c=_config(trace,names,policy='cooperative_late_fusion',energy=True,deadline=5)
    r=run_simulation(c,True)
    assert r.summary['joint_all_error_count']==0
    assert r.summary['complete_event_rate']==0
    assert r.summary['agent_deadline_miss_rate']==1
    assert r.summary['energy_post_window_incremental_j']>0
    assert len([x for x in r.inferences if x['model_id']=='late_fusion'])==1
    assert not any(x['consumers_delivered'] for x in r.inferences)


def test_frame_batch_projection_is_reproducible(tmp_path):
    trace,names,_=_trace(tmp_path,nframes=3,nrois=5,n_agents=3)
    c=_config(trace,names,policy='mixed_local_fusion',energy=True)
    a=run_simulation(c,True);b=run_simulation(c,True)
    assert a.summary==b.summary
    assert a.scores==b.scores
    assert a.inferences==b.inferences
    assert a.operations==b.operations
    assert a.policy_plans==b.policy_plans
    assert {p['physical_execution_id'] for p in a.policy_plans}=={'frame:frame0','frame:frame1','frame:frame2'}
    for frame in range(3):
        plans=[p for p in a.policy_plans if p['frame_id']==f'frame{frame}']
        assert len(plans)==5
        assert len({p['execution_mode'] for p in plans})==1  # policy acts once per frame
