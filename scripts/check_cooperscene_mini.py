"""Download or inspect CooperScene mini without requiring NVIDIA/CUDA.

Download uses standard urllib, runs on machines with ordinary network access,
and intentionally requires an explicit user action. Raw data is never bundled
inside the OCTOPUS source archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from urllib.request import urlopen, Request
from zipfile import ZipFile, BadZipFile

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from octopus_sim.datasets.cooperscene_mini import (OFFICIAL_MINI_URL,MiniDataError,
                                                  scan_mini,write_reports)


def download(url: str, target: Path) -> dict:
    target=Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        raise MiniDataError(f'Archive already exists. Inspect rather than silently replace: {target}')
    part=target.with_name(target.name+'.partial')
    if part.exists():
        raise MiniDataError(f'Partial download exists; remove or relocate before retrying: {part}')
    sha=hashlib.sha256();count=0
    try:
        with (urlopen(Request(url,headers={'User-Agent':'OCTOPUS-Sim-Mini-Validator/1.0'}),timeout=45) as inp,
              part.open('wb') as out):
            kind=inp.headers.get('Content-Type','')
            if 'html' in kind.lower():
                raise MiniDataError('Server returned an HTML page, not a mini archive')
            while True:
                chunk=inp.read(2**20)
                if not chunk:break
                out.write(chunk);sha.update(chunk);count+=len(chunk)
        if not count or not part.open('rb').read(4).startswith(b'PK\x03\x04'):
            raise MiniDataError('Downloaded file is not ZIP data')
        with ZipFile(part) as z:
            bad=z.testzip()
            if bad:raise MiniDataError('Corrupt ZIP member: '+str(bad))
        part.replace(target)
        return {'url':url,'file':str(target),'sha256':sha.hexdigest(),'bytes':count}
    finally:
        if part.exists():part.unlink()


def safe_extract(archive: Path, dest: Path, *, limit_bytes: int=24*2**30) -> dict:
    archive=Path(archive).resolve(); dest=Path(dest).resolve()
    if not archive.is_file():
        raise MiniDataError(f'Mini archive not found: {archive}')
    with ZipFile(archive) as z:
        infos=z.infolist()
        if sum(i.file_size for i in infos)>limit_bytes:
            raise MiniDataError('Mini archive expands beyond configured limit')
        if len(infos)>300000:
            raise MiniDataError('Too many ZIP members')
        for info in infos:
            path=Path(info.filename)
            if (path.is_absolute() or any(part=='..' for part in path.parts)
                or '\\' in info.filename or (info.external_attr>>16)&0o170000 == 0o120000):
                raise MiniDataError('Potentially unsafe ZIP entry: '+info.filename)
            actual=(dest/path).resolve()
            if not actual.is_relative_to(dest):
                raise MiniDataError('ZIP path escapes extraction directory: '+info.filename)
        if dest.exists() and any(dest.iterdir()):
            raise MiniDataError('Refusing to extract into nonempty directory')
        dest.mkdir(parents=True,exist_ok=True)
        z.extractall(dest)
    return {'zip_file':str(archive),'destination':str(dest),'members':len(infos)}


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='action',required=True)
    get=sub.add_parser('download');get.add_argument('--archive',type=Path,required=True)
    ext=sub.add_parser('extract');ext.add_argument('--archive',type=Path,required=True)
    ext.add_argument('--directory',type=Path,required=True)
    chk=sub.add_parser('scan');chk.add_argument('--data-root',type=Path,required=True)
    chk.add_argument('--output',type=Path,required=True)
    chk.add_argument('--strict-mini',action='store_true')
    chk.add_argument('--no-yaml',action='store_true');chk.add_argument('--no-pcd-header',action='store_true')
    args=ap.parse_args()
    try:
        if args.action=='download': result=download(OFFICIAL_MINI_URL,args.archive)
        elif args.action=='extract':result=safe_extract(args.archive,args.directory)
        else:
            result=scan_mini(args.data_root,strict_mini=args.strict_mini,
                             parse_yaml=not args.no_yaml,
                             check_pcd_headers=not args.no_pcd_header)
            write_reports(result,args.output)
            result={'report':str(args.output/'mini_scan.json'),
                    'frame_index':str(args.output/'frame_index.csv'),
                    'total_frames':result['total_frames'],
                    'splits':{s:v['synchronized_frames'] for s,v in result['splits'].items()},
                    'warnings':result['warnings']}
        print(json.dumps(result,indent=2))
    except (MiniDataError,OSError,BadZipFile) as exc:
        ap.exit(2,f'ERROR: {exc}\n')

if __name__=='__main__':main()
