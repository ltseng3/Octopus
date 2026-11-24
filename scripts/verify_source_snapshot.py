"""Verify this public source snapshot, not the historical full-data inventories."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def main():
    inventory = json.loads((ROOT / 'PUBLIC_SOURCE_SHA256.json').read_text())['files']
    for name, sha in inventory.items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise SystemExit(f'Missing or changed source: {name}')
    print(f'Public source snapshot verified: {len(inventory)} files')

if __name__ == '__main__':
    main()
