"""Verify checksums for THIS CooperScene mini-readiness source branch.

Earlier manifests are historical provenance records, not current release hashes.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    inventory=json.loads((ROOT/'COOPERSCENE_MINI_RELEASE_SHA256.json').read_text())
    assert inventory['status']=='source_only_no_mini_dataset_downloaded'
    for name,expected in sorted(inventory['files'].items()):
        p=ROOT/name
        if not p.is_file():raise RuntimeError(f'Missing release file: {name}')
        current=hashlib.sha256(p.read_bytes()).hexdigest()
        if current!=expected:raise RuntimeError(f'SHA256 mismatch {name}')
    print(f"CooperScene mini-source release integrity verified: {len(inventory['files'])} files")

if __name__=='__main__':main()
