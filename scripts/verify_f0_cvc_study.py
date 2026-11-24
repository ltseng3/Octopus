#!/usr/bin/env python3
"""Verify *new* F0 CVC study files, independent of the parent release inventory."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=json.loads((ROOT/'F0_CVC_STUDY_SHA256.json').read_text(encoding='utf-8'))
    entries=manifest['files']
    for relative, expected in entries.items():
        path=ROOT/relative
        if not path.is_file():raise RuntimeError(f'Missing study file: {relative}')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:raise RuntimeError(f'Changed study file: {relative}')
    assert len(entries)>=30,'Study inventory unexpectedly small'
    print(f'Verified {len(entries)} F0 CVC study files (SHA-256).')


if __name__=='__main__':main()
