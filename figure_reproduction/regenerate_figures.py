"""Regenerate the twelve research PDFs from frozen inputs, not new detector runs."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FIGURES = (
    'fig_intro_complementarity', 'fig_edge_architecture', 'fig_partition_analytical',
    'fig1_real_accuracy_joint_risk', 'fig2_class_specific_risk',
    'fig_prevalence_sensitivity', 'fig3_three_agent_risk_tradeoff',
    'fig_five_agent_lineage', 'fig_real_energy_tradeoff', 'fig6_deadline_cliff',
    'fig4_expert_calibration', 'fig5_sunk_cost_and_roi_reuse',
)
SCRIPTS = ('analyze_seq0007.py', 'make_figures.py', 'make_class_figure.py', 'make_added_figures.py')


def main():
    manifest = ROOT / 'FROZEN_INPUT_SHA256.json'
    if not (ROOT / 'sources' / 'v2v4real_seq0007_trace.jsonl').is_file():
        raise SystemExit('Missing frozen input data. Copy sources/ and FROZEN_INPUT_SHA256.json from the complete student bundle.')
    if manifest.is_file():
        for name, sha in json.loads(manifest.read_text()).items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
                raise SystemExit(f'Frozen input checksum mismatch: {name}')
        print('Frozen source checksums verified.')
    else:
        print('Warning: no frozen input manifest; verify input provenance before interpretation.')
    (ROOT / 'figures').mkdir(exist_ok=True)
    for script in SCRIPTS:
        subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)
    for fig in FIGURES:
        path = ROOT / 'figures' / (fig + '.pdf')
        if not path.is_file() or not path.read_bytes().startswith(b'%PDF-'):
            raise SystemExit(f'Missing PDF: {path}')
    print(f'Regenerated all {len(FIGURES)} research PDFs in {ROOT / "figures"}')


if __name__ == '__main__':
    main()
