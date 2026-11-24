"""Validate the SHA256 inventory for the new three-agent study addendum."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'F0_THREE_AGENT_EXTENDED_SHA256.json'

def verify():
    records=json.loads(MANIFEST.read_text(encoding='utf-8'))['files']
    assert records, 'No files in extended study inventory'
    for rel,sha in sorted(records.items()):
        file=ROOT/rel
        if not file.is_file(): raise RuntimeError(f'Missing {rel}')
        actual=hashlib.sha256(file.read_bytes()).hexdigest()
        if actual!=sha:raise RuntimeError(f'Checksum mismatch: {rel}')
    print(f'PASS: {len(records)} new study source/figure/data assets SHA-256 verified.')

if __name__=='__main__':verify()
