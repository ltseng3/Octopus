"""Real-data fidelity: refuse nonexistent V2V4Real third vehicle and fake CooperScene results."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from octopus_sim.config import ExperimentConfig
from octopus_sim.datasets.three_vehicle_real import (
    ThreeVehicleDataError, read_protocol, convert_to_trace,
)
from octopus_sim.simulator import run_simulation
from scripts.run_three_vehicle_real_study import perform_evaluation
from scripts.prepare_dmstrack_v2v4real import source_paths


def fixture(tmp: Path):
    manifest={
        'dataset':'CooperScene', 'split':'test', 'vehicle_ids':['1','2','3'],
        'roi_protocol':'predeclared_world_xy_grid_v1',
        'coordinate_frame':'cooperscene_global_xy_m',
        'prediction_origin':'synthetic_fixture',
        'physical_detector_runs':{'1':'source_run_v1','2':'source_run_v2','3':'source_run_v3'},
        'assumed_source_error':{'1':.06,'2':.12,'3':.30},
        'error_estimator_status':'unfitted_assumption',
        'interval_ms':150.0,
    }
    rows=[]
    values=[
        (1,{'1':1,'2':1,'3':0}),
        (0,{'1':1,'2':0,'3':0}),
        (1,{'1':0,'2':1,'3':1}),
        (0,{'1':0,'2':0,'3':0}),
    ]
    for i,(truth,preds) in enumerate(values):
        rows.append({'split':'test','take_id':'take_a','frame_token':str(i//2),
                     'roi_id':f'grid_{i%2}','truth':truth,'predictions':preds,
                     'roi':{'frame':'cooperscene_global_xy_m',
                            'center_xy_m':[10.0+10*(i%2),20.0],'radius_m':7.5}})
    mp=tmp/'manifest.json'; mp.write_text(json.dumps(manifest))
    src=tmp/'predictions.jsonl'; src.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return mp,src,manifest,rows


def test_v2v4real_official_source_has_only_two_physical_vehicle_detections():
    keys=source_paths('0007')
    assert {name for name in keys if name in ('ego','agent1')}=={'ego','agent1'}
    assert 'agent2' not in keys
    assert 'cobevt' in keys and 'late_fusion' in keys # algorithms, not physical vehicles


def test_real_file_requires_real_three_vehicle_provenance(tmp_path):
    manifest,src,_,_=fixture(tmp_path)
    with pytest.raises(ThreeVehicleDataError, match='Synthetic fixture rejected'):
        convert_to_trace(src,manifest,tmp_path/'out')
    with pytest.raises(ThreeVehicleDataError,match='Synthetic fixture rejected'):
        read_protocol(manifest)
    with pytest.raises(ThreeVehicleDataError,match='real data export'):
        modified=json.loads(manifest.read_text());modified['prediction_origin']='real_detector_export'
        manifest.write_text(json.dumps(modified))
        read_protocol(manifest,allow_fixture=True)


def test_no_v2v4real_three_vehicle_alias_or_infrastructure_substitution(tmp_path):
    mp,src,m,_=fixture(tmp_path)
    for violation,pattern in [
        ({'dataset':'V2V4Real'},'only has two vehicles'),
        ({'vehicle_ids':['0','1','2']},'physical vehicles'),
        ({'vehicle_ids':['1','1','3']},'physical vehicles'),
        ({'physical_detector_runs':{'1':'same','2':'same','3':'other'}},'distinct physically'),
        ({'roi_protocol':'ground_truth_anchored'},'predeclared'),
        ({'coordinate_frame':'KITTI_camera_xz'},'global XY'),
        ({'assumed_source_error':{'1':.06,'2':.12,'3':0}},'rates strictly'),
        ({'error_estimator_status':'fit_on_disjoint_training_split'},'disjoint train'),
    ]:
        d=deepcopy(m);d.update(violation);mp.write_text(json.dumps(d))
        with pytest.raises(ThreeVehicleDataError,match=pattern):
            convert_to_trace(src,mp,tmp_path/'out',allow_fixture=True)


def test_prepare_and_real_simulator_execution_with_three_physical_agents(tmp_path):
    manifest,src,_,_=fixture(tmp_path)
    folder=tmp_path/'converted'
    record=convert_to_trace(src,manifest,folder,allow_fixture=True)
    assert record['physical_frames']==2 and record['decision_events']==4
    cfg=ExperimentConfig.load(folder/'scenario.json')
    assert cfg.data['scenario']['agents']==['1','2','3']
    assert set(cfg.data['fusion_f0']['error_by_agent'])=={'1','2','3'}
    assert cfg.data['world']['execution_granularity']=='frame'
    assert all('edge:' not in x for x in json.loads((folder/'trace.jsonl').read_text().splitlines()[0])['candidates'])
    r=run_simulation(cfg,detailed_events=False)
    assert r.summary['physical_execution_batches']==2
    assert r.summary['local_inference_executions']==6
    assert r.summary['edge_inference_executions']==2
    assert len(r.scores)==4
    assert len(r.fusion_decisions)==4
    # Multiple ROI questions do not cause additional physically identical detection jobs.
    assert r.summary['local_inference_executions'] != 3*len(r.scores)
    # A local-only policy accepts just the actual local predictions.
    d=deepcopy(cfg.data);d['policy']['name']='local_only'
    local=run_simulation(ExperimentConfig.from_dict(d,folder/'scenario.json'),detailed_events=False)
    assert local.summary['physical_execution_batches']==2
    assert local.summary['local_inference_executions']==6
    assert local.summary['edge_inference_executions']==0


def test_run_study_explicit_fixture_gate_and_risk_consistency(tmp_path):
    mp,src,_,_=fixture(tmp_path)
    folder=tmp_path/'converted';convert_to_trace(src,mp,folder,allow_fixture=True)
    with pytest.raises(ThreeVehicleDataError,match='Synthetic input fixture'):
        perform_evaluation(folder/'scenario.json',tmp_path/'results')
    rows=perform_evaluation(folder/'scenario.json',tmp_path/'results',allow_fixture=True,
                            thresholds=(.9,.99))
    assert [r['method'] for r in rows]==['local_only','majority','f0_0.900','f0_0.990']
    assert all(r['physical_local_jobs']==6 for r in rows)
    assert all(r['physical_edge_jobs']==2 for r in rows[1:])
    assert all(r['dataset']=='SYNTHETIC_FIXTURE_NOT_REAL' for r in rows)
    assert all(r['r_ge_2']>=r['r_ge_3'] for r in rows)
    assert (tmp_path/'results'/'comparison.csv').exists()


def test_no_truth_leak_into_f0_actions_or_resource_cost(tmp_path):
    mp,src,_,rows=fixture(tmp_path)
    folder=tmp_path/'a';convert_to_trace(src,mp,folder,allow_fixture=True)
    r1=run_simulation(ExperimentConfig.load(folder/'scenario.json'),detailed_events=False)
    # Change only hidden truth, while the candidate predictions and q stay fixed.
    for row in rows:row['truth']^=1
    src.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    folder2=tmp_path/'b';convert_to_trace(src,mp,folder2,allow_fixture=True)
    r2=run_simulation(ExperimentConfig.load(folder2/'scenario.json'),detailed_events=False)
    decisions=lambda r:[(d['action'],d['per_agent_decisions'],d['fused_prediction'],
                         d['surrogate_score']) for d in r.fusion_decisions]
    assert decisions(r1)==decisions(r2)
    assert r1.summary['energy_workload_incremental_j']==pytest.approx(r2.summary['energy_workload_incremental_j'])
    assert r1.summary['mean_delivered_latency_ms']==pytest.approx(r2.summary['mean_delivered_latency_ms'])


def test_missing_agent_binary_value_and_nonfinite_roi_rejected(tmp_path):
    mp,src,_,rows=fixture(tmp_path)
    for change,pattern in [
        ({'predictions':{'1':1,'2':0}},'three per-vehicle predictions'),
        ({'predictions':{'1':1,'2':0,'3':True}},'integer 0 or 1'),
        ({'predictions':{'1':1,'2':0,'3':1,'0':0}},'three per-vehicle predictions'),
        ({'roi':{'frame':'cooperscene_global_xy_m','center_xy_m':[float('nan'),0.0],'radius_m':7.5}},'finite center'),
        ({'roi':{'frame':'wrong','center_xy_m':[0.0,0.0],'radius_m':7.5}},'coordinate frame'),
    ]:
        rr=deepcopy(rows);rr[0].update(change);src.write_text(''.join(json.dumps(r)+'\n' for r in rr))
        with pytest.raises(ThreeVehicleDataError,match=pattern):
            convert_to_trace(src,mp,tmp_path/'out',allow_fixture=True)


def test_duplicate_roi_and_cross_take_frame_identity(tmp_path):
    mp,src,_,rows=fixture(tmp_path)
    rr=deepcopy(rows);rr.append(rr[0]);src.write_text(''.join(json.dumps(r)+'\n' for r in rr))
    with pytest.raises(ThreeVehicleDataError,match='Duplicate'):
        convert_to_trace(src,mp,tmp_path/'out',allow_fixture=True)
    rr=deepcopy(rows)
    rr[-1]['take_id']='take_b';rr[-2]['take_id']='take_b'
    src.write_text(''.join(json.dumps(r)+'\n' for r in rr))
    out=tmp_path/'clean';rec=convert_to_trace(src,mp,out,allow_fixture=True)
    assert rec['physical_frames']==2
    names={json.loads(line)['frame_id'] for line in (out/'trace.jsonl').read_text().splitlines()}
    assert names=={'CooperScene:test:take_a:0','CooperScene:test:take_b:1'}


def test_no_3_vehicle_measurements_are_claimed_from_published_map(tmp_path):
    from octopus_sim.datasets.three_vehicle_real import SOURCE_URL
    assert 'CooperScene' in SOURCE_URL
    # F0's ROI joint error is NOT derivable from six-model aggregate mAP.
    assert not (tmp_path/'r2_measured_from_map.csv').exists()


def test_crosschecks_all_three_external_export_artifacts_and_ground_truth(tmp_path):
    import hashlib
    mp,src,m,rows=fixture(tmp_path)
    m['prediction_origin']='real_detector_export'  # test only: source is still artificial
    m['checkpoint_sha256']={a:(str(int(a))*64) for a in ('1','2','3')}
    m['per_vehicle_prediction_exports']={}
    for a in ('1','2','3'):
        p=tmp_path/f'agent_{a}.jsonl'
        payload=''.join(json.dumps({k:r[k] for k in ('take_id','frame_token','roi_id')} | {
            'vehicle_id':a,'prediction':r['predictions'][a]})+'\n' for r in rows)
        p.write_text(payload)
        m['per_vehicle_prediction_exports'][a]={'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    gt=tmp_path/'ground_truth.jsonl'
    gt.write_text(''.join(json.dumps({k:r[k] for k in ('take_id','frame_token','roi_id')}|{
         'truth':r['truth']})+'\n' for r in rows))
    m['ground_truth_export']={'path':gt.name,'sha256':hashlib.sha256(gt.read_bytes()).hexdigest()}
    mp.write_text(json.dumps(m))
    p=convert_to_trace(src,mp,tmp_path/'real_structural_fixture')
    assert set(p['verifiable_source_artifacts']['per_vehicle_prediction_export_sha256'])=={'1','2','3'}
    assert p['prediction_origin']=='real_detector_export'
    # A malicious author could still fabricate all four mutually consistent
    # files; integrity verification is necessary but NOT proof of sensor origin.
    # This test checks the engineering contract, NOT empirical validity.
    gt.write_text(gt.read_text()+'\n')
    with pytest.raises(ThreeVehicleDataError,match='SHA-256 mismatch'):
        convert_to_trace(src,mp,tmp_path/'another')
    m['ground_truth_export']['sha256']=hashlib.sha256(gt.read_bytes()).hexdigest()
    mp.write_text(json.dumps(m))
    altered=deepcopy(rows);altered[0]['predictions']['1']^=1
    src.write_text(''.join(json.dumps(r)+'\n' for r in altered))
    with pytest.raises(ThreeVehicleDataError,match='contradicts physical vehicle 1'):
        convert_to_trace(src,mp,tmp_path/'another')


def test_published_cooperscene_three_vehicle_numbers_match_pinned_official_table():
    from scripts.plot_cooperscene_published import records
    models, rows=records()
    assert len(models)==6
    assert rows['V2VNet',3]['map_05_cv2x']==.23
    assert rows['V2VNet',3]['map_05_unlimited']==.74
    assert rows['CoSDH',3]['map_05_cv2x']==.61
    assert rows['CoSDH',3]['sharing_mb']==.1190
    assert rows['ERMVP',3]['map_05_cv2x']==.57
    assert rows['V2VNet',3]['map_05_cv2x'] < rows['V2VNet',2]['map_05_cv2x']
    for m in models:
        assert rows[m,3]['map_05_unlimited'] > rows[m,2]['map_05_unlimited']
    import csv
    csv_path=Path(__file__).resolve().parents[1]/'samples'/'three_vehicle_real_published'/'cooperscene_published_v2v_v2v2.csv'
    with csv_path.open() as f: headers=csv.DictReader(f).fieldnames
    assert not any(name in ('r2','r3','joint_error_rate','f0_corrections') for name in headers)
    prov=json.loads((csv_path.parent/'UPSTREAM_BENCHMARK_PROVENANCE.json').read_text())
    assert prov['source_readme_git_blob_sha']=='2de04bda334c5baa2bead8744d90ee0417579e7c'
