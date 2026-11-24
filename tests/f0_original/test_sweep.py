import csv
import json
from pathlib import Path
from scripts.threshold_sweep import main,sweep


def test_sweep_reproducibility_frame_cluster_intervals():
    rows=sweep(frames=40,bootstrap=40)
    assert rows==sweep(frames=40,bootstrap=40)
    assert len(rows)==15
    assert all(x['frame_clusters']==40 and x['decision_events']==80 for x in rows)
    for x in rows:
        for stem in ('joint','marginal','coverage'):
            key=('shared_coverage' if stem=='coverage' else stem+'_error')
            assert 0<=x[stem+'_ci_lo']<=x[key]<=x[stem+'_ci_hi']<=1


def test_sweep_cli_outputs_csv_without_plot_requirement(tmp_path):
    main(['--frames','30','--bootstrap','40','--no-plot','--out',str(tmp_path)])
    assert len(list(csv.DictReader((tmp_path/'threshold_sweep.csv').open())))==15
    assert not (tmp_path/'threshold_tradeoff.png').exists()
    p=json.loads((tmp_path/'PROVENANCE.json').read_text())
    assert p['bootstrap_unit']=='whole frame, two ROIs per frame'
