"""Tests of pinned public V2V4Real-derived diagnostic replay and importer."""
import csv
import importlib.util
from pathlib import Path
import sys
from collections import defaultdict

import pytest
from octopus_sim.config import ExperimentConfig
from octopus_sim.simulator import run_simulation
from octopus_sim.policies import BasicPolicy, CoordinatorView

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=ROOT/'scripts'
sys.path.insert(0,str(SCRIPTS))
from prepare_dmstrack_v2v4real import (parse_boxes,present,choose_regions,analyze_scene,
                                        source_paths,SEQ_FRAMES,COMMIT)
from unpack_verified_v2v4real import unpack


def test_public_git_revision_and_path_contract():
    assert len(COMMIT)==40 and len(SEQ_FRAMES)==9 and sum(SEQ_FRAMES)==1993
    paths=source_paths('0000')
    assert set(paths)=={'gt','ego','agent1','cobevt','late_fusion'}
    assert paths['ego']!=paths['agent1']
    assert all(p.endswith('0000.txt') for p in paths.values())


def test_real_source_kitti_schema_parsing():
    # Numeric examples from public DMSTrack/AB3DMOT sequence 0000, frame 0.
    gt='0 1 Car 0 0 0 0 0 0 0 1.6980522 2.0622315 3.9799347 -20.500578 -0.978255 -0.10612263 0.026179945'
    ego='0,2,0,0,0,0,0.552287,1.7276435,1.9572803,4.29604,-20.691498,-1.1837013,-0.09900059,0.0483056,0.000000'
    other='0,2,0,0,0,0,0.619345,1.663162,1.8595805,4.197101,-20.90078,0.31291088,1.1157186,-0.07246299,0.000000'
    gold=parse_boxes(gt,ground_truth=True)
    assert gold[0][0]==pytest.approx((-20.500578,-0.10612263))
    g=gold[0][0]
    for row in [ego,other]:
        observed=parse_boxes(row,ground_truth=False)
        assert present(observed[0],*g)
    assert not parse_boxes(ego,ground_truth=False,score_threshold=.99)
    assert not present(gold[0],500,500)


def test_region_selection_does_not_use_predictions():
    gt={0:[(-20.,0.),(30.,10.)],1:[(15.,25.)]}
    assert choose_regions(gt,0)==choose_regions(gt,0)
    for frame in [0,1]:
        regions=choose_regions(gt,frame)
        assert [x[0] for x in regions]==['positive','negative']
        assert present(gt[frame],*regions[0][1])
        assert not present(gt[frame],*regions[1][1])
    assert len(choose_regions(gt,0,task='fixed_grid'))==12


def test_pinned_scene_validation_rejects_alignment_errors():
    maps={k:{0:[(0.,0.)]} for k in ('gt','ego','agent1','cobevt','late_fusion')}
    with pytest.raises(ValueError, match='frame count'):
        analyze_scene('0000',3,maps)
    row,events=analyze_scene('0000',1,maps)
    assert len(events)==2
    assert row['balanced_events']==2
    assert row['separate_joint_errors']==0
    assert row['late_fusion_errors']==0
    assert row['balanced_positives']==1


def test_policy_mixture_endpoints_two_agents():
    v=CoordinatorView('e',('ego','agent1'),400,0.5,0.05,0)
    first=BasicPolicy('mixed_local_shared',reuse_probability=0.0).plan(v)
    assert len(first.groups)==2 and all(g.location=='local' for g in first.groups)
    last=BasicPolicy('mixed_local_shared',reuse_probability=1.0).plan(v)
    assert len(last.groups)==1 and last.groups[0].location=='edge'
    assert last.groups[0].consumers==('ego','agent1')


def test_packed_genuine_source_sequence_and_end_to_end_replay(tmp_path):
    base=ROOT/'empirical/v2v4real'
    n=unpack(base/'packed_outcomes.json',base/'sequence_counts.csv',tmp_path)
    assert n==442
    expanded=[__import__('json').loads(line) for line in (tmp_path/'trace.jsonl').read_text().splitlines()]
    assert len({r['frame_id'] for r in expanded})==221
    assert expanded[0]['frame_id']==expanded[1]['frame_id']
    cfg=ExperimentConfig.load(tmp_path/'scenario.json')
    independent=run_simulation(cfg.with_policy('local_only'),detailed_events=False)
    shared=run_simulation(cfg.with_policy('always_share'),detailed_events=False)
    assert independent.summary['joint_all_error_rate']==pytest.approx(4/442)
    assert shared.summary['joint_all_error_rate']==pytest.approx(16/442)
    assert independent.summary['complete_event_rate']==1.0
    assert shared.summary['complete_event_rate']==1.0
    assert independent.summary['local_inference_executions']==442
    assert shared.summary['edge_inference_executions']==221
    # Source identity is genuinely reused, not duplicated.
    assert all(len(set(v for v in row['result_ids'].values() if v is not None))==1 for row in shared.scores)


def test_aggregate_source_counts_and_reversal():
    p=ROOT/'empirical/v2v4real/sequence_counts.csv'
    with p.open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==9
    assert sum(int(r['frames']) for r in rows)==1993
    assert sum(int(r['balanced_events']) for r in rows)==3986
    assert sum(int(r['separate_joint_errors']) for r in rows)==105
    assert sum(int(r['late_fusion_errors']) for r in rows)==194
    assert sum(int(r['late_fusion_errors'])>int(r['separate_joint_errors']) for r in rows)==8
