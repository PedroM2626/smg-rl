"""Play a loaded owner savestate with a simple forward-biased random policy and log chosen RAM words.

Read-only for the game: only inputs are sent. Frames are saved so a person can read the HUD counters.

Usage: python scripts/luigi_explore.py GAME_ID STATE OUTDIR ADDR[,ADDR...] [--decisions N] [--frame-every K] [--seed S]
"""
import argparse
import json
import os
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from luigi_dump import load, open_backend  # noqa: E402
from luigi_probe import raw_state  # noqa: E402

POLICY = ((),) if os.environ.get('LUIGI_IDLE') else (('w',), ('w',), ('w', 'jump'), ('w', 'a'), ('w', 'd'), ('w', 'jump'), ('a',), ('d',), ('jump',), ('w', 'spin'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_id')
    parser.add_argument('state', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('addresses')
    parser.add_argument('--decisions', type=int, default=150)
    parser.add_argument('--frame-every', type=int, default=15)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--stage')
    args = parser.parse_args()
    addresses = [int(a, 16) for a in args.addresses.split(',')]
    args.output.mkdir(parents=True, exist_ok=True)
    backend = open_backend(args.game_id, args.output / 'scratch', capture='full')
    rng = np.random.default_rng(args.seed)
    log = []
    try:
        load(backend, args.state)
        if args.stage:
            backend.memory.expected_stage = args.stage
        for decision in range(args.decisions + 1):
            words = [struct.unpack('>I', backend.memory.reader.read_bytes(a, 4))[0] for a in addresses]
            try:
                state = raw_state(backend.memory.reader, args.game_id)
            except Exception:  # transitions can leave the pointer chain invalid for a moment
                state = {'position': None, 'scene_nerve': None}
            try:
                t = backend.memory.sample()
                state['telemetry'] = {'complete': t.complete, 'dead': t.dead, 'scene_nerve': hex(t.scene_nerve),
                                      'health': t.health, 'grounded': t.grounded, 'scene_frame': t.scene_frame}
            except Exception as error:
                state['telemetry'] = {'error': str(error)[:60]}
            log.append({'decision': decision, 'words': words, 'position': state['position'], 'scene_nerve': state['scene_nerve'], 'telemetry': state['telemetry']})
            if decision % args.frame_every == 0:
                cv2.imwrite(str(args.output / f'frame-{decision:04d}.png'),
                            cv2.cvtColor(backend.bridge.image(), cv2.COLOR_RGBA2BGR))
            backend.advance(POLICY[int(rng.integers(len(POLICY)))])
    finally:
        backend.close()
        (args.output / 'log.json').write_text(json.dumps({'addresses': [hex(a) for a in addresses], 'log': log}))
    changed = [(hex(a), sorted({row['words'][i] for row in log})[:12]) for i, a in enumerate(addresses)]
    print(json.dumps(changed))


if __name__ == '__main__':
    main()
