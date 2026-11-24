"""Verify the source-file SHA-256 inventory shipped with a release.

This checks file contents, not third-party licensing or upstream provenance.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    manifest = ROOT / 'THREE_VEHICLE_REAL_SHA256.json'
    if not manifest.exists():
        manifest = ROOT / 'INTEGRATION_SHA256.json'
    if not manifest.exists():
        manifest = ROOT / 'SOURCE_SHA256.json'  # historical v0.7-only archive
    inventory=json.loads(manifest.read_text(encoding='utf-8'))['files']
    for name, expected in sorted(inventory.items()):
        p=ROOT/name
        if not p.is_file():
            raise RuntimeError(f'Missing release file {name}')
        actual=hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'SHA-256 mismatch: {name}')
    print(f'Verified {len(inventory)} release files (SHA-256).')

if __name__=='__main__':
    main()
