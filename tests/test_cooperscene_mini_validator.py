"""Only CPU synthetic test fixtures. These are not real CooperScene measurements."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from zipfile import ZipFile
import pytest

from octopus_sim.datasets.cooperscene_mini import (
    MiniDataError, scan_mini,locate_root,write_reports,OFFICIAL_MINI_URL,
)
from scripts.check_cooperscene_mini import safe_extract,download


def make_mini(tmp_path:Path,*,splits=(('train',2),('validate',1),('test',1)),same_take=False):
    root=tmp_path/'cooperscene'
    for split,n in splits:
        take='same_take' if same_take else f'{split}_take'
        for a in '0123':
            directory=root/split/take/a
            directory.mkdir(parents=True)
            for frame in range(n):
                token=f'{frame:06}'
                (directory/f'{token}.pcd').write_bytes((
                    'VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n'
                    'TYPE F F F F\nCOUNT 1 1 1 1\n'
                    'WIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA ascii\n1 2 3 0.3\n').encode())
                (directory/f'{token}.yaml').write_text(
                    'lidar_pose: [1, 2, 3, 0, 0, 0]\n'
                    'vehicles:\n  vehicle_1:\n    location: [0, 0, 0]\n'
                    '    extent: [2, 1, 0.5]\n    angle: [0, 0, 0]\n'
                )
    return root


def test_scanner_returns_exact_physical_agent_id_and_frame_inventory(tmp_path):
    root=make_mini(tmp_path)
    result=scan_mini(root)
    assert result['status']=='actual_raw_dataset_layout_checked_not_detector_inference'
    assert result['total_frames']==4
    assert [result['splits'][s]['synchronized_frames'] for s in ('train','validate','test')]==[2,1,1]
    assert all(r['agents']==['0','1','2','3'] and r['vehicle_ids']==['1','2','3']
               for s in result['splits'].values() for r in s['frames'])
    assert all(r['reported_annotation_counts']==[1,1,1,1]
               for s in result['splits'].values() for r in s['frames'])
    assert all(r['pcd_points']=={'0':1,'1':1,'2':1,'3':1}
               for s in result['splits'].values() for r in s['frames'])
    assert OFFICIAL_MINI_URL.endswith('/mini.zip')
    report=tmp_path/'out'
    write_reports(result,report)
    assert json.loads((report/'mini_scan.json').read_text())['total_frames']==4
    with (report/'frame_index.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==4
    assert all(row['agent_ids']=='0,1,2,3' for row in rows)


def test_autodetect_nested_zip_root(tmp_path):
    root=make_mini(tmp_path/'wrapper')
    assert locate_root(tmp_path/'wrapper')==root.resolve()
    assert scan_mini(tmp_path/'wrapper')['total_frames']==4


def test_missing_vehicle_timestamp_rejected(tmp_path):
    root=make_mini(tmp_path)
    (root/'train'/'train_take'/'3'/'000001.yaml').unlink()
    (root/'train'/'train_take'/'3'/'000001.pcd').unlink()
    with pytest.raises(MiniDataError,match='missing synchronized vehicle'):
        scan_mini(root)


def test_missing_rsu_timestamp_rejected(tmp_path):
    root=make_mini(tmp_path)
    (root/'test'/'test_take'/'0'/'000000.yaml').unlink()
    (root/'test'/'test_take'/'0'/'000000.pcd').unlink()
    with pytest.raises(MiniDataError,match='RSU/vehicle timestamp mismatch|no timestamped annotations'):
        scan_mini(root)


def test_missing_sensor_and_orphan_points_rejected(tmp_path):
    root=make_mini(tmp_path)
    p=root/'train'/'train_take'/'1'/'000001.pcd'
    p.unlink()
    with pytest.raises(MiniDataError,match='YAML without point cloud'):
        scan_mini(root)
    p.write_bytes(b'not-a-real-pcd')
    (root/'train'/'train_take'/'1'/'orphan.pcd').write_bytes(b'nope')
    with pytest.raises(MiniDataError,match='unannotated point clouds'):
        scan_mini(root)


def test_invalid_yaml_pose_and_annotation_object_rejected(tmp_path):
    root=make_mini(tmp_path)
    y=root/'test'/'test_take'/'1'/'000000.yaml'
    y.write_text('vehicles: {}\n')
    with pytest.raises(MiniDataError,match='lidar_pose'):
        scan_mini(root)
    y.write_text('lidar_pose: [1,2,3,0,0,0]\nvehicles: []\n')
    with pytest.raises(MiniDataError,match='vehicles annotations'):
        scan_mini(root)
    # Strictly no executable PyYAML tags from user-provided sensor metadata.
    y.write_text('!!python/object/apply:os.system ["echo unsafe"]\n')
    with pytest.raises(MiniDataError,match='safely parse'):
        scan_mini(root)


def test_invalid_pcd_header_rejected(tmp_path):
    root=make_mini(tmp_path)
    (root/'test'/'test_take'/'2'/'000000.pcd').write_bytes(b'hello')
    with pytest.raises(MiniDataError,match='Invalid PCD header'):
        scan_mini(root)
    # Users can explicitly skip expensive header validation; metadata checks remain.
    assert scan_mini(root,check_pcd_headers=False)['total_frames']==4


def test_mini_count_is_a_strict_opt_in_contract(tmp_path):
    root=make_mini(tmp_path)
    assert scan_mini(root,strict_mini=False)['total_frames']==4
    with pytest.raises(MiniDataError,match='expected 120 frames'):
        scan_mini(root,strict_mini=True)


def test_shared_take_ids_flag_possible_temporal_leakage(tmp_path):
    root=make_mini(tmp_path,same_take=True)
    warnings=scan_mini(root)['warnings']
    assert any('temporally adjacent' in w and 'train' in w and 'test' in w for w in warnings)


def test_zip_extract_rejects_traversal(tmp_path):
    z=tmp_path/'malicious.zip'
    with ZipFile(z,'w') as archive:archive.writestr('../escape.txt','no')
    with pytest.raises(MiniDataError,match='unsafe ZIP'):
        safe_extract(z,tmp_path/'out')
    assert not (tmp_path/'escape.txt').exists()


def test_zip_extract_valid_and_protects_nonempty_destination(tmp_path):
    z=tmp_path/'fixture.zip'
    with ZipFile(z,'w') as archive:archive.writestr('root/train/ok.txt','test')
    out=tmp_path/'out'
    result=safe_extract(z,out)
    assert result['members']==1
    assert (out/'root'/'train'/'ok.txt').read_text()=='test'
    with pytest.raises(MiniDataError,match='nonempty'):
        safe_extract(z,out)


def test_zip_limits_prevent_decompression_bombs(tmp_path):
    z=tmp_path/'large.zip'
    with ZipFile(z,'w') as archive:archive.writestr('a.txt','a'*1000)
    with pytest.raises(MiniDataError,match='expands beyond'):
        safe_extract(z,tmp_path/'out',limit_bytes=100)


def test_download_is_opt_in_and_does_not_replace_existing_file(tmp_path):
    file=tmp_path/'mini.zip'
    file.write_text('existing dataset')
    with pytest.raises(MiniDataError,match='already exists'):
        download('https://invalid.example/mini.zip',file)
    assert file.read_text()=='existing dataset'
