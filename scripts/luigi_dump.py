"""Dump MEM1 and MEM2 of a loaded owner savestate (read-only) after an optional number of idle decisions.

Usage: python scripts/luigi_dump.py GAME_ID STATE OUT.npz [--idle N] [--press NAMES --count K]
"""
import argparse
from pathlib import Path

import numpy as np

from smg_rl.config import load_config
from smg_rl.native import NativeDolphin

CONFIGS = {'SB4E01': 'config/motor8-smg2.yaml', 'RMGE01': 'config/motor8-smg1.yaml'}
REGIONS = {'mem1': (0x80000000, 0x1800000), 'mem2': (0x90000000, 0x4000000)}


def open_backend(game_id, scratch, capture='cpu'):
    config = load_config(CONFIGS[game_id])
    config['parallel'].update(native_capture=capture, workers=1)
    config['paths']['user'] = str(Path(scratch).resolve() / 'template')
    return NativeDolphin(config)


def load(backend, state):
    try:
        backend.load_state(state)
    except RuntimeError as error:  # the bridge rejects a frame size at load time; the load itself still happens
        if 'error 2' not in str(error):
            raise
        backend.paused = True
        backend.advance(())


def dump(backend):
    result = {}
    for name, region in REGIONS.items():
        try:
            result[name] = np.frombuffer(backend.memory.reader.read_bytes(*region), dtype='>u4').copy()
        except Exception:  # MEM2 is not always located by the reader; MEM1 alone is still dumped
            continue
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_id')
    parser.add_argument('state', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--idle', type=int, default=2)
    args = parser.parse_args()
    backend = open_backend(args.game_id, args.output.parent / ('scratch-' + args.output.stem))
    try:
        load(backend, args.state)
        for _ in range(args.idle):
            backend.advance(())
        np.savez(args.output, **dump(backend))
    finally:
        backend.close()


if __name__ == '__main__':
    main()
