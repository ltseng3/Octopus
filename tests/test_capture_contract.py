"""Capture smoke test using FAKE upstream interfaces, NOT actual pretrained models."""
from __future__ import annotations
import json
import runpy
import sys
from pathlib import Path
import types

import pytest


def test_capture_export_contract_with_fake_upstream(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    np = pytest.importorskip('numpy')

    # Construct lightweight doubles for the API shapes documented by
    # V2V4Real: yaml_utils, train_utils, build_dataset, LateFusionDataset.
    def install(name):
        mod = types.ModuleType(name)
        mod.__path__ = []
        monkeypatch.setitem(sys.modules,name,mod)
        if '.' in name:
            parent, part = name.rsplit('.',1)
            setattr(sys.modules[parent],part,mod)
        return mod

    for name in ('opencood','opencood.hypes_yaml','opencood.hypes_yaml.yaml_utils',
                 'opencood.tools','opencood.tools.train_utils',
                 'opencood.data_utils','opencood.data_utils.datasets'):
        install(name)

    def load_yaml(_unused, opt):
        return {'preprocess':{'method':'mock'}, 'postprocess':{'method':'mock'},
                'fusion':{'core_method':'LateFusionDataset'},'checkpoint':opt.model_dir}

    sys.modules['opencood.hypes_yaml.yaml_utils'].load_yaml=load_yaml

    class FakeModel(torch.nn.Module):
        def __init__(self, ckpt):
            super().__init__();self.ckpt=ckpt
        def forward(self, content):
            # Return independent source-specific mocked scores. The postprocessor
            # interprets them below as detection centers; this is NOT a trained model.
            source=float(content['source'])
            if self.ckpt == 'edge':
                val=2.0 if source==1.0 else 14.0
            else:
                val=2.0 if source==1.0 else 10.0
            return torch.tensor([val,0.0,0.0])

    def create_model(hyp):
        kind=Path(hyp['checkpoint']).name
        return FakeModel(kind)

    def load_saved_model(dirname,model):
        return 1,model

    def to_device(data,_device):return data
    tr=sys.modules['opencood.tools.train_utils']
    tr.create_model=create_model;tr.load_saved_model=load_saved_model;tr.to_device=to_device

    class FakePostProcessor:
        def generate_gt_bbx(self,_batch):
            return torch.tensor([[[2.,0,0]]*8, [[14.,0,0]]*8])
        def post_process(self,_batch,predictions):
            value=next(iter(predictions.values()))
            return value.reshape(1,1,3).repeat(1,8,1), torch.tensor([.7])

    class LateFusionDataset:
        def __init__(self):
            self.post_processor=FakePostProcessor()
            self.len_record=[2]
            self.scenario_folders=[str(tmp_path/'real_scene_mock')]
            self.scenario_database={0:{
                '101':{'00000':{},'00001':{},'ego':True},
                '202':{'00000':{},'00001':{},'ego':False},
            }}
        def __len__(self):return 2
        def return_timestamp_key(self,_db,i):return f'{i:05}'
        def retrieve_base_data(self,i):
            return {
                '101':{'ego':True, 'params':{'lidar_pose':[0,0,0,0,0,0]}},
                '202':{'ego':False,'params':{'lidar_pose':[4,0,0,0,0,0]}},
            }
        def __getitem__(self,i):return {'frame':i}
        def collate_batch_test(self,items):
            return {'ego':{'source':torch.tensor(1)},'202':{'source':torch.tensor(2)}}

    sys.modules['opencood.data_utils.datasets'].build_dataset=lambda *a,**kw:LateFusionDataset()
    data=tmp_path/'data';data.mkdir()
    local=tmp_path/'local';local.mkdir()
    edge=tmp_path/'edge';edge.mkdir()
    out=tmp_path/'captured'
    script=Path(__file__).resolve().parents[1]/'scripts'/'capture_v2v4real.py'
    monkeypatch.setattr(sys,'argv',['capture_v2v4real.py',
        '--local-model-dir',str(local),'--edge-model-dir',str(edge),
        '--data-dir',str(data),'--output',str(out),'--max-frames','2',
        '--stride','1','--device','cpu'])
    runpy.run_path(str(script),run_name='__main__')
    lines=[json.loads(x) for x in (out/'frames.jsonl').read_text().splitlines()]
    assert [x['frame_id'] for x in lines]==['00000','00001']
    assert [a['agent_id'] for a in lines[0]['agents']]==['101','202']
    assert lines[0]['reference_lidar_pose']==[0.0]*6
    from octopus_sim.datasets.opv2v import import_box_manifest
    rep=import_box_manifest(out/'frames.jsonl',tmp_path/'converted')
    assert rep.valid_events==6 and rep.positive_events==4
    events=[json.loads(x) for x in rep.trace_path.read_text().splitlines()]
    assert events[1]['candidates']['edge:101']==1
    assert events[1]['candidates']['edge:202']==0
