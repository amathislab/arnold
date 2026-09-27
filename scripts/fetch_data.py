#!/usr/bin/env python3
"""Install pinned Arnold artifacts using only Python's standard library."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]


def verify_archive(path, artifact):
    """Verify size and the archive checksum before any extraction."""
    if path.stat().st_size != artifact['bytes']:
        raise ValueError('Archive size mismatch: ' + str(path))
    checksum = artifact['checksum']
    digest = hashlib.new(checksum['algorithm'])
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != checksum['value']:
        raise ValueError('Archive checksum mismatch: ' + str(path))


def fetch_archive(artifact, cache):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / artifact['filename']
    if path.exists():
        return path
    print('Downloading ' + artifact['filename'], flush=True)
    handle, temporary = tempfile.mkstemp(prefix=artifact['filename'] + '.', suffix='.part', dir=str(cache))
    try:
        request = urllib.request.Request(artifact['url'], headers={'User-Agent': 'Arnold-reproduction/1'})
        with os.fdopen(handle, 'wb') as output, urllib.request.urlopen(request, timeout=60) as response:
            downloaded = 0
            next_update = 64 * 1024 * 1024
            for block in iter(lambda: response.read(1024 * 1024), b''):
                output.write(block)
                downloaded += len(block)
                if downloaded >= next_update:
                    print('  {:.0f}/{:.0f} MB'.format(downloaded / 1e6, artifact['bytes'] / 1e6), flush=True)
                    next_update += 64 * 1024 * 1024
        verify_archive(Path(temporary), artifact)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def checked_destination(root, relative):
    path = root / relative
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        raise ValueError('Destination escapes installation root: ' + str(path))
    if path.is_symlink():
        raise ValueError('Destination is a symlink: ' + str(path))
    return path


def same_file(a, b):
    if a.stat().st_size != b.stat().st_size:
        return False
    with a.open('rb') as left, b.open('rb') as right:
        while True:
            block = left.read(1024 * 1024)
            if block != right.read(1024 * 1024):
                return False
            if not block:
                return True


def install_archive(path, artifact, root):
    verify_archive(path, artifact)
    data_dir = checked_destination(root, Path('data'))
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = checked_destination(root, Path(artifact['destination']))
    with tempfile.TemporaryDirectory(prefix='.artifact-staging-', dir=str(data_dir)) as temporary:
        staging = Path(temporary)
        count = 0
        with tarfile.open(str(path), mode='r:gz') as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                if name.is_absolute() or '..' in name.parts:
                    raise ValueError('Unsafe archive path: ' + member.name)
                if any(p.startswith('._') or p in ('.DS_Store', '__MACOSX') for p in name.parts):
                    continue
                if not name.parts or name.parts[0] != artifact['archive_root']:
                    raise ValueError('Unexpected archive root: ' + member.name)
                relative = Path(*name.parts[1:])
                if member.isdir():
                    (staging / relative).mkdir(parents=True, exist_ok=True)
                elif member.isfile() and relative.parts:
                    target = staging / relative
                    if target.exists():
                        raise ValueError('Duplicate archive file: ' + member.name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, target.open('wb') as output:
                        shutil.copyfileobj(source, output)
                    count += 1
                else:
                    raise ValueError('Unsupported archive entry: ' + member.name)
        if count == 0:
            raise ValueError('Archive contains no data files: ' + str(path))
        staged_files = sorted(p for p in staging.rglob('*') if p.is_file())
        # Preflight all conflicts before changing installed files. Matching data is reusable.
        for source in staged_files:
            target = checked_destination(root, Path(artifact['destination']) / source.relative_to(staging))
            if target.exists() and (not target.is_file() or not same_file(source, target)):
                raise FileExistsError('Existing file differs; leaving it untouched: ' + str(target))
        destination.mkdir(parents=True, exist_ok=True)
        for source in staged_files:
            target = destination / source.relative_to(staging)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.replace(source, target)
    print('Verified and installed {} ({} files) -> {}'.format(artifact['filename'], count, destination), flush=True)


def install_bundled_references(root):
    source_dir = REPO_ROOT / 'data/final_benchmarks/expert_policies'
    provenance_file = source_dir / 'provenance.json'
    provenance = json.loads(provenance_file.read_text())
    files = [provenance_file] + [
        REPO_ROOT / ref['file'] for ref in provenance['references'].values()
    ]
    pairs = [(source, checked_destination(root, source.relative_to(REPO_ROOT))) for source in files]
    for source, target in pairs:
        if target.exists() and (not target.is_file() or not same_file(source, target)):
            raise FileExistsError('Existing expert reference differs; leaving it untouched: ' + str(target))
    for source, target in pairs:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


def main():
    manifest = json.loads((REPO_ROOT / 'data/reproduction/artifacts.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=sorted(manifest['profiles']), default='benchmarks')
    parser.add_argument('--root', type=Path, default=REPO_ROOT, help='Installation root (default: repository root).')
    parser.add_argument('--cache-dir', type=Path, help='Reuse/download archives here; default: <root>/data/cache/archives/21807280.')
    parser.add_argument('--list', action='store_true', help='List the selected archives and destinations without downloading.')
    parser.add_argument('--verify-only', action='store_true', help='Verify cached archive checksums.')
    args = parser.parse_args()
    root = args.root.resolve()
    cache = args.cache_dir if args.cache_dir is not None else root / 'data/cache/archives' / manifest['release']['record']
    selected = manifest['profiles'][args.profile]
    if args.list:
        print('Pinned release: ' + manifest['release']['doi'])
        for ident in selected:
            artifact = manifest['artifacts'][ident]
            print('{} ({} bytes) -> {}'.format(artifact['filename'], artifact['bytes'], artifact['destination']))
        return
    if not args.verify_only:
        install_bundled_references(root)
    for ident in selected:
        artifact = manifest['artifacts'][ident]
        path = cache / artifact['filename'] if args.verify_only else fetch_archive(artifact, cache)
        if args.verify_only:
            verify_archive(path, artifact)
            print('Verified ' + str(path), flush=True)
        else:
            install_archive(path, artifact, root)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, tarfile.TarError, urllib.error.URLError) as error:
        sys.exit('Artifact installation failed: ' + str(error))
