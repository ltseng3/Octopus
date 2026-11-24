"""Verify the shipped CVC study supplement and its unchanged v0.7 base manifest."""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify() -> int:
    manifest=json.loads((ROOT/'CVC_STUDY_SHA256.json').read_text(encoding='utf-8'))
    base=ROOT/'SOURCE_SHA256.json'
    if sha256(base.read_bytes()).hexdigest() != manifest['parent_source_manifest_sha256']:
        raise ValueError('Original v0.7 source inventory changed')
    for name, expected in manifest['files'].items():
        path=ROOT/name
        if not path.is_file():raise ValueError(f'Missing study file: {name}')
        if sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Hash mismatch for {name}')
    return len(manifest['files'])


if __name__=='__main__':
    print(f'Verified {verify()} CVC study files and unchanged parent inventory')
