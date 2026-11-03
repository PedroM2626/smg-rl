"""Pack, verify and unpack the published Dolphin savestate bundles.

Savestates are never committed to Git. They are distributed as GitHub release
assets: uncompressed tar archives whose first member, ``manifest.json``, lists
every repository-relative destination with its size and SHA-256. Unpacking
verifies each file before it is moved into place and never overwrites a
different existing file unless ``--force`` is given.
"""

import argparse
import hashlib
import io
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'manifest.json'
SCHEMA = 1
HUMAN_BANK = '.local/exploration-study-20261029/human-bank'
HUMAN_WAYPOINTS = (1, 2, 4, 8, 14, 18, 23, 29, 34, 38, 40)
BUNDLES = {
    'smg-rl-states-core': {
        'scope': 'Mission starts and near-star detector fixtures for both games (data/).',
        'files': [f'data/{game}.s0{slot}' for game in ('RMGE01', 'SB4E01') for slot in range(1, 5)],
    },
    'smg-rl-states-smg1-human-bank': {
        'scope': 'Validated human-assisted Sweet Sweet anchors for the backward curriculum.',
        'files': [f'{HUMAN_BANK}/bank.json']
                 + [f'{HUMAN_BANK}/human-waypoint-{w:02d}.state' for w in HUMAN_WAYPOINTS],
    },
    'smg-rl-states-regional-curriculum': {
        'scope': 'Earlier assisted regional-curriculum starts for SMG1 and SMG2.',
        'files': ['.local/regional-curriculum/smg1/bank.json',
                  '.local/regional-curriculum/smg1/sweet-sweet-first-chocolate.state',
                  '.local/regional-curriculum/smg1/sweet-sweet-first-pink.state',
                  '.local/regional-curriculum/smg1/sweet-sweet-right-approach.state',
                  '.local/regional-curriculum/smg2/bank.json',
                  '.local/regional-curriculum/smg2/double-time-first-bridge.state',
                  '.local/regional-curriculum/smg2/double-time-platform-approach.state',
                  '.local/regional-curriculum/smg2/spinning-first-bridge.state',
                  '.local/regional-curriculum/smg2/spinning-obstacle-approach.state',
                  '.local/regional-curriculum/smg2/spinning-obstacle-jump-approach.state',
                  '.local/regional-curriculum/smg2/spinning-platform-approach.state'],
    },
}
NOTICE = ('Savestates are Dolphin RAM snapshots of commercial games and require your own '
          'legally obtained game image. They are not learned mission completions.')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_destination(name):
    """Reject absolute paths, drive letters and parent traversal in archive members."""
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or '..' in path.parts or ':' in name or '\\' in name
            or path.parts[0] not in ('data', '.local')):
        raise ValueError(f'Unsafe bundle destination: {name!r}')
    return path


def pack(name, root, out):
    spec = BUNDLES[name]
    entries = []
    for relative in spec['files']:
        source = Path(root) / safe_destination(relative)
        if not source.is_file():
            raise FileNotFoundError(f'Missing bundle file: {relative}')
        entries.append({'path': relative, 'size': source.stat().st_size, 'sha256': sha256(source)})
    manifest = {'schema': SCHEMA, 'bundle': name, 'scope': spec['scope'], 'notice': NOTICE,
                'files': entries}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(manifest, indent=1).encode()
    with tarfile.open(out, 'w', format=tarfile.PAX_FORMAT) as archive:
        info = tarfile.TarInfo(MANIFEST)
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
        for entry in entries:
            archive.add(Path(root) / entry['path'], arcname=entry['path'], recursive=False)
    return manifest


def read_manifest(archive):
    first = archive.next()
    if first is None or first.name != MANIFEST:
        raise ValueError('Bundle does not start with its manifest.')
    manifest = json.loads(archive.extractfile(first).read())
    if manifest.get('schema') != SCHEMA or not isinstance(manifest.get('files'), list):
        raise ValueError('Unsupported bundle manifest.')
    for entry in manifest['files']:
        safe_destination(entry['path'])
    return manifest


def unpack(bundle, root, force=False, check_only=False):
    """Verify every member against the manifest; install only after all files pass."""
    root = Path(root)
    with tarfile.open(bundle, 'r:') as archive, tempfile.TemporaryDirectory(dir=root) as staging:
        manifest = read_manifest(archive)
        expected = {entry['path']: entry for entry in manifest['files']}
        seen = set()
        for member in archive.getmembers()[1:]:
            if member.name not in expected or member.name in seen or not member.isfile():
                raise ValueError(f'Unexpected bundle member: {member.name!r}')
            seen.add(member.name)
            target = Path(staging) / safe_destination(member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open('wb') as stream:
                shutil.copyfileobj(source, stream, 1 << 20)
            entry = expected[member.name]
            if target.stat().st_size != entry['size'] or sha256(target) != entry['sha256']:
                raise ValueError(f'Checksum mismatch: {member.name}')
        if seen != set(expected):
            raise ValueError(f'Bundle is missing files: {sorted(set(expected) - seen)}')
        if check_only:
            return manifest, []
        conflicts = [path for path in expected if (root / path).exists()
                     and sha256(root / path) != expected[path]['sha256']]
        if conflicts and not force:
            raise FileExistsError(f'Different local files exist (use --force): {conflicts}')
        installed = []
        for path in expected:
            destination = root / path
            if destination.exists() and path not in conflicts:
                continue  # Identical file already present.
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(Path(staging) / path), destination)
            installed.append(path)
    return manifest, installed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)
    packer = commands.add_parser('pack', help='Create release bundles from local savestates.')
    packer.add_argument('--bundle', choices=sorted(BUNDLES), action='append')
    packer.add_argument('--out', default='dist/states')
    for command in ('verify', 'unpack'):
        sub = commands.add_parser(command)
        sub.add_argument('bundles', nargs='+')
        if command == 'unpack':
            sub.add_argument('--force', action='store_true',
                             help='Replace local files whose checksum differs.')
    args = parser.parse_args(argv)
    if args.command == 'pack':
        for name in args.bundle or sorted(BUNDLES):
            out = Path(args.out) / f'{name}.tar'
            manifest = pack(name, ROOT, out)
            (out.parent / f'{name}.sha256').write_text(f'{sha256(out)}  {out.name}\n')
            print(f'{out}: {len(manifest["files"])} files')
        return 0
    for bundle in args.bundles:
        manifest, installed = unpack(bundle, ROOT, force=getattr(args, 'force', False),
                                     check_only=args.command == 'verify')
        verb = 'verified' if args.command == 'verify' else f'installed {len(installed)} of'
        print(f'{bundle}: {verb} {len(manifest["files"])} files ({manifest["bundle"]})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
