"""Read-only RAM scan of a verified human replay for values that behave like level progress.

capture: replay a verified recording and save MEM1 snapshots every STRIDE samples (no memory is ever written).
analyse: rank aligned 32-bit words by how closely their increments follow the route progress of the recording.

Usage:
  python scripts/scan_progress_ram.py capture SOURCE CONFIG VARIATION OUTPUT [--stride 12]
  python scripts/scan_progress_ram.py analyse OUTPUT
A hit is only a candidate: correlation never validates a native progress field.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import psutil

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.human_recorder import FRAMES_PER_SAMPLE, exact_observation
from smg_rl.native import NativeDolphin

MEM1 = (0x80000000, 0x1800000)


def capture(source, config_path, variation, output, stride, maximum_seconds=1700):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('Use a new output directory.')
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('Existing emulators must finish before the scan.')
    manifest = json.loads((source / 'manifest.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        if Path(name).name != name or digest(source / name) != expected:
            raise ValueError('Human source integrity failed.')
    config = load_config(config_path)
    mission = next(v for v in config['variations'] if v['name'] == variation)
    game_id = config['game']['id']
    state_file = Path(config['paths']['assets']) / f'{game_id}.s{mission["start_slot"]:02d}'
    if digest(state_file) != manifest['source_state_sha256']:
        raise ValueError('Canonical source savestate changed.')
    with (source / 'trajectory.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    points = np.array([[float(r[k]) for k in ('x', 'y', 'z')] for r in rows])
    inputs = [json.loads(line) for line in (source / 'inputs.jsonl').read_text().splitlines()]
    images = np.memmap(source / 'observations.gray8', dtype=np.uint8, mode='r', shape=(len(rows), 84, 84))
    output.mkdir(parents=True)
    config['paths']['user'] = str((output / 'profile' / 'template').resolve())
    config['parallel'].update(frame_skip=FRAMES_PER_SAMPLE, native_capture='full', workers=1)
    arc = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    taken, backend = [], None
    deadline = time.monotonic() + maximum_seconds
    try:
        backend = NativeDolphin(config)
        backend.load_slot(mission['start_slot'])
        for index in range(len(rows)):
            if time.monotonic() >= deadline:
                raise TimeoutError('Scan time budget reached.')
            if index:
                backend.advance(tuple(inputs[index - 1]['controls']))
            t = backend.memory.sample()
            if np.linalg.norm(np.array(t.position) - points[index]) > .1:
                raise ValueError(f'Replay position diverged at {index}.')
            if not np.array_equal(exact_observation(backend), images[index]):
                raise ValueError(f'Replay observation diverged at {index}.')
            if index % stride == 0 or index == len(rows) - 1:
                data = backend.memory.reader.read_bytes(*MEM1)
                name = f'mem1-{len(taken):04d}.bin'
                (output / name).write_bytes(data)
                taken.append({'file': name, 'sample': index, 'arc': float(arc[index]),
                              'arc_fraction': float(arc[index] / arc[-1]), 'position': list(t.position),
                              'health': t.health, 'grounded': t.grounded, 'scene_frame': t.scene_frame})
            if t.dead or t.complete:
                break
    finally:
        if backend is not None:
            backend.close()
        atomic_json(output / 'capture.json', {'source': str(source), 'config': str(config_path),
                                              'variation': variation, 'stride': stride, 'snapshots': taken,
                                              'region': {'address': MEM1[0], 'size': MEM1[1]}})
    return len(taken)


def pearson(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    denom = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    with np.errstate(invalid='ignore', divide='ignore'):
        return (a * b).sum(0) / denom


def analyse(output, chunk_words=262144, level=0.97, delta_level=0.6, monotone=0.95, alias=0.98):
    output = Path(output)
    meta = json.loads((output / 'capture.json').read_text())
    shots = meta['snapshots']
    progress = np.array([s['arc_fraction'] for s in shots])
    coordinates = np.array([s['position'] for s in shots])
    memmaps = [np.memmap(output / s['file'], dtype='>u4', mode='r') for s in shots]
    words = len(memmaps[0])
    hits = []
    for start in range(0, words, chunk_words):
        block = np.stack([np.asarray(m[start:start + chunk_words]) for m in memmaps])
        for kind in ('f4', 'u4', 'i4'):
            if kind == 'f4':
                values = block.view('>f4').astype(np.float64)
            elif kind == 'u4':
                values = block.astype(np.float64)
            else:
                values = block.astype('>u4').view('>i4').astype(np.float64)
            valid = np.isfinite(values).all(0) & (np.abs(values) < 1e9).all(0) & (np.ptp(values, axis=0) > 0)
            values = np.where(valid[None], values, 0.0)
            r = pearson(values, progress[:, None])
            keep = valid & (np.abs(np.nan_to_num(r)) >= level)
            if not keep.any():
                continue
            index = np.flatnonzero(keep)
            sub = values[:, index]
            dv, dp = np.diff(sub, axis=0), np.diff(progress)[:, None]
            dr = np.nan_to_num(pearson(dv, np.broadcast_to(dp, dv.shape).copy()))
            sign = np.sign(r[index])
            moving = np.abs(dv) > 0
            consistent = ((np.sign(dv) == sign[None]) & moving).sum(0) / np.maximum(moving.sum(0), 1)
            mirror = np.max(np.abs(np.stack([np.nan_to_num(pearson(sub, coordinates[:, a, None])) for a in range(3)])), 0)
            for j, column in enumerate(index):
                if abs(dr[j]) >= delta_level and consistent[j] >= monotone and mirror[j] < alias:
                    hits.append({'address': f'0x{MEM1[0] + 4 * (start + int(column)):08X}', 'dtype': kind,
                                 'level_corr': float(r[column]), 'delta_corr': float(dr[j]),
                                 'monotone_share': float(consistent[j]), 'max_coordinate_corr': float(mirror[j]),
                                 'first': float(sub[0, j]), 'last': float(sub[-1, j])})
    hits.sort(key=lambda h: -abs(h['delta_corr']))
    atomic_json(output / 'candidates.json', {'snapshots': len(shots), 'words': words, 'candidates': hits[:300],
                                              'total_hits': len(hits),
                                              'scope': 'Unvalidated scalar candidates from one replay.'})
    return len(hits)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    cap = commands.add_parser('capture')
    cap.add_argument('source', type=Path)
    cap.add_argument('config', type=Path)
    cap.add_argument('variation')
    cap.add_argument('output', type=Path)
    cap.add_argument('--stride', type=int, default=12)
    ana = commands.add_parser('analyse')
    ana.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'capture':
        print(json.dumps({'snapshots': capture(args.source, args.config, args.variation, args.output, args.stride)}))
    else:
        print(json.dumps({'hits': analyse(args.output)}))
