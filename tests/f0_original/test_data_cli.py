import json
from dataclasses import replace
from pathlib import Path
import math
import pytest

from octopus_fusion import (
    AgentEvidence, FusionObservation, ReliabilityEstimate, FusionConfig,
    SelectiveFuser, QualityProxyEstimator, StaticErrorEstimator,
)
from octopus_fusion.synthetic import ToySettings, generate
from octopus_fusion.evaluate import LabeledCase, evaluate
from octopus_fusion.benchmark import compare, _baseline
from octopus_fusion.io import load_labeled_cases, write_labeled_cases
from octopus_fusion.v07_adapter import load_v07_local_outputs
from octopus_fusion.cli import main


def toy_cases():
    cases=generate(ToySettings(frames=15,rois_per_frame=2,agents=3,seed=221))
    e=QualityProxyEstimator()
    return tuple((c,e.estimate(c.observation)) for c in cases)


def test_toy_reproducible_frames_and_roi_projections():
    settings=ToySettings(frames=12,rois_per_frame=3,seed=19,shared_origin_fraction=1)
    a=generate(settings)
    b=generate(settings)
    assert a==b
    assert len(a)==36
    assert len({c.observation.frame_id for c in a})==12
    assert a[0].observation.evidence[0].result_id==a[1].observation.evidence[0].result_id
    assert a[0].observation.evidence[-1].result_id==a[0].observation.evidence[0].result_id
    assert {c.truth for c in a} <= {0,1}
    assert len({c.observation.event_id for c in a})==36


def test_toy_five_agents_and_deterministic_evaluation():
    settings=ToySettings(frames=50,agents=5,rois_per_frame=2,seed=99)
    a=generate(settings)
    est=QualityProxyEstimator()
    paired=tuple((c,est.estimate(c.observation)) for c in a)
    out=compare(paired)
    assert [row['policy'] for row in out]==['local_only','majority','weighted_naive','selective_f0']
    assert all(row['events']==100 for row in out)
    assert all(0<=row['joint_all_wrong_rate']<=1 for row in out)
    assert out==compare(paired)


def test_evaluation_truth_flip_does_not_change_any_f0_decision():
    paired=toy_cases()
    f=SelectiveFuser()
    decisions_before={c.observation.event_id:f.decide(c.observation,e) for c,e in paired}
    modified=tuple((replace(c,truth=1-c.truth),e) for c,e in paired)
    decisions_after={c.observation.event_id:f.decide(c.observation,e) for c,e in modified}
    assert decisions_before==decisions_after
    assert evaluate((c for c,_ in paired),decisions_before).cases==30
    assert evaluate((c for c,_ in modified),decisions_after).cases==30


def test_evaluate_checks_exact_identity_and_keeps_fallback_local():
    paired=toy_cases()
    f=SelectiveFuser()
    d={c.observation.event_id:f.decide(c.observation,r) for c,r in paired}
    cases=tuple(c for c,_ in paired)
    result=evaluate(cases,d)
    assert result.cases==30
    assert result.fallback_rate==pytest.approx(1-result.shared_coverage)
    with pytest.raises(ValueError,match='exactly'):
        evaluate(cases,{**d,'extra':next(iter(d.values()))})
    with pytest.raises(ValueError,match='Missing'):
        evaluate(cases,{k:v for k,v in list(d.items())[:-1]})
    with pytest.raises(ValueError,match='Duplicate'):
        evaluate((*cases,cases[0]),d)


def test_quality_estimator_requires_public_proxy():
    a=generate(ToySettings(frames=1,rois_per_frame=1))[0].observation
    assert len(QualityProxyEstimator().estimate(a).error_by_agent)==3
    b=replace(a,evidence=tuple(replace(e,quality_proxy=None) for e in a.evidence))
    with pytest.raises(ValueError,match='quality_proxy'):
        QualityProxyEstimator().estimate(b)


def test_io_roundtrip_and_invalid_duplicate_event(tmp_path):
    paired=toy_cases()
    path=tmp_path/'toy.jsonl'
    write_labeled_cases(path,paired)
    actual=load_labeled_cases(path)
    assert tuple(c for c,_ in actual)==tuple(c for c,_ in paired)
    assert [r.error_by_agent for _,r in actual]==[r.error_by_agent for _,r in paired]
    path.write_text(path.read_text()+path.read_text().splitlines()[0]+'\n')
    with pytest.raises(ValueError,match='Repeated'):
        load_labeled_cases(path)


def test_io_strict_provenance_rejects_malformed_input(tmp_path):
    p=tmp_path/'bad.jsonl'
    p.write_text(json.dumps({'event_id':'e','frame_id':'f','truth':1,'evidence':[]})+'\n')
    with pytest.raises(ValueError):load_labeled_cases(p)
    p.write_text('{invalid json')
    with pytest.raises(ValueError):load_labeled_cases(p)


def test_cli_toy_outputs_no_truth_in_decisions(tmp_path):
    out=tmp_path/'outputs'
    main(['toy','--frames','15','--rois','3','--agents','5','--out',str(out),'--write-toy-trace'])
    decisions=[json.loads(line) for line in (out/'selective_decisions.jsonl').read_text().splitlines()]
    assert len(decisions)==45
    assert all('truth' not in x for x in decisions)
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['energy_latency']=='not simulated by F0; F1 integration required'
    import csv
    assert len(list(csv.DictReader((out/'comparison.csv').open())))==4
    main(['replay',str(out/'synthetic_labeled_trace.jsonl'),'--out',str(tmp_path/'replay')])
    assert (tmp_path/'replay'/'comparison.csv').read_bytes()==(out/'comparison.csv').read_bytes()


def test_static_assumptions_not_hidden_outcomes():
    e=StaticErrorEstimator((('A',.08),('B',.22)))
    a=FusionObservation('e','f',20,(
        AgentEvidence('A',1,'rA','A',10),AgentEvidence('B',0,'rB','B',10)))
    b=replace(a, evidence=(replace(a.evidence[0],prediction=0),replace(a.evidence[1],prediction=1)))
    assert e.estimate(a)==e.estimate(b)


def write_v07_fixture(tmp_path, *, contradict=False):
    inference=[]
    score=[]
    for frame in (0,1):
        for agent in ('a','b','c'):
            p0=(frame+len(agent))%2
            pred={'frame%d/roi0'%frame:p0,'frame%d/roi1'%frame:1-p0}
            inference.append({
                'result_id':f'frame:{frame}:result:{agent}','frame_id':f'frame{frame}',
                'location':'local','source_agent':agent,'model_id':'local_model',
                'observation_ids':[f'frame:{frame}:obs:{agent}'],
                'created_at_ms':frame*100+20,'predictions_by_event_id':pred,
            })
        for roi in (0,1):
            score.append({'event_id':f'frame{frame}/roi{roi}','frame_id':f'frame{frame}',
                          'ground_truth':int(roi==1)})
    inf=tmp_path/'inferences.jsonl'
    sc=tmp_path/'scores.jsonl'
    inf.write_text(''.join(json.dumps(x)+'\n' for x in inference))
    sc.write_text(''.join(json.dumps(x)+'\n' for x in score))
    return inf,sc


def test_v07_adapter_uses_artifacts_not_score_predictions(tmp_path):
    inf,scores=write_v07_fixture(tmp_path)
    q={'a':.05,'b':.13,'c':.09}
    paired=load_v07_local_outputs(inf,scores,q)
    assert len(paired)==4
    e0=paired[0][0].observation.evidence[0]
    e1=paired[1][0].observation.evidence[0]
    assert e0.result_id==e1.result_id
    assert e0.prediction!=e1.prediction
    original=[x[0].observation for x in paired]
    changes=[]
    for row in scores.read_text().splitlines():
        rec=json.loads(row)
        rec['ground_truth']=1-rec['ground_truth']
        rec['predictions']={'a':999,'b':999,'c':999}  # ignored: not an oracle input
        changes.append(rec)
    scores.write_text(''.join(json.dumps(x)+'\n' for x in changes))
    again=load_v07_local_outputs(inf,scores,q)
    assert [x[0].observation for x in again]==original
    assert all(a[0].truth!=b[0].truth for a,b in zip(paired,again))


def test_v07_adapter_missing_model_outputs_rejected(tmp_path):
    inf,scores=write_v07_fixture(tmp_path)
    entries=[json.loads(x) for x in inf.read_text().splitlines()]
    inf.write_text(''.join(json.dumps(x)+'\n' for x in entries[:-1]))
    with pytest.raises(ValueError,match='do not cover assumed agents'):
        load_v07_local_outputs(inf,scores,{'a':.1,'b':.2,'c':.3})


def test_v07_adapter_frame_identity_conflict_rejected(tmp_path):
    inf,scores=write_v07_fixture(tmp_path)
    entries=[json.loads(x) for x in inf.read_text().splitlines()]
    entries[0]['frame_id']='wrongframe'
    inf.write_text(''.join(json.dumps(x)+'\n' for x in entries))
    with pytest.raises(ValueError,match='frame ID'):
        load_v07_local_outputs(inf,scores,{'a':.1,'b':.2,'c':.3})


def test_v07_cli_adapter(tmp_path):
    inf,score=write_v07_fixture(tmp_path)
    q=tmp_path/'rates.json'
    q.write_text(json.dumps({'a':.05,'b':.2,'c':.1}))
    dest=tmp_path/'out'
    main(['v07','--inferences',str(inf),'--scores',str(score),
          '--error-rates',str(q),'--out',str(dest)])
    assert len((dest/'selective_decisions.jsonl').read_text().splitlines())==4
    assert json.loads((dest/'manifest.json').read_text())['kind']=='v07_offline_adapter_unverified'


def test_toy_setting_validation_and_rng_seed():
    for kwargs in ({'agents':1},{'frames':0},{'shared_origin_fraction':1.1}):
        with pytest.raises(ValueError):ToySettings(**kwargs)
    assert generate(ToySettings(frames=12,seed=1)) != generate(ToySettings(frames=12,seed=2))
