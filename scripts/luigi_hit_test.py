"""Look for a live damage event in the Galaxy 2 Mario Squared mission and check the energy value against the HUD.

Only inputs are sent; no game memory is written and the owner savestate is only read. A forward-biased random policy
plays several seeds from the owner start state. Every change of the energy value (or death) saves the frame after it,
so a person can read the HUD life meter at that moment.

Usage: python scripts/luigi_hit_test.py OUTDIR [--seeds 8] [--decisions 900]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from luigi_dump import load  # noqa: E402

from smg_rl.checkpoints import digest  # noqa: E402
from smg_rl.config import load_config  # noqa: E402
from smg_rl.native import NativeDolphin  # noqa: E402

STATE = Path('D:/luigi-coins/SB4E01.s02')
POLICY = (('w',),) * 6 + (('w', 'jump'),) * 2 + (('w', 'a'), ('w', 'd'))  # mostly forward, so it stays on the platforms


def frame(backend, path):
    cv2.imwrite(str(path), cv2.cvtColor(backend.bridge.image(), cv2.COLOR_RGBA2BGR))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--decisions', type=int, default=900)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    before = digest(STATE)
    config = load_config('config/luigi-mario-squared-altbtr-20261110.yaml')
    config['parallel'].update(workers=1, native_capture='full')
    config['paths']['user'] = str((args.output / 'profile' / 'template').resolve())
    backend = NativeDolphin(config)
    report = {'state_sha256': before, 'runs': [], 'events': []}
    try:
        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 + seed)
            load(backend, STATE)
            t = backend.memory.sample()
            run = {'seed': seed, 'decisions': 0, 'coins': t.purple_coins, 'health_trace': [t.health], 'end': 'budget'}
            for decision in range(1, args.decisions + 1):
                previous = t
                backend.advance(POLICY[int(rng.integers(len(POLICY)))])
                try:
                    t = backend.memory.sample()
                except Exception as error:  # transitions can briefly break the pointer chain
                    run['end'] = f'sample error: {str(error)[:80]}'
                    break
                run['decisions'], run['coins'] = decision, max(run['coins'], t.purple_coins or 0)
                if t.health != previous.health or t.dead:
                    name = f'seed{seed}-decision{decision:04d}'
                    frame(backend, args.output / f'{name}-after.png')
                    report['events'].append({'seed': seed, 'decision': decision, 'health_before': previous.health,
                                             'health_after': t.health, 'dead': t.dead, 'coins': t.purple_coins,
                                             'position_before': previous.position, 'position_after': t.position,
                                             'frame': f'{name}-after.png'})
                    run['health_trace'].append(t.health)
                if t.dead:
                    run['end'] = 'dead'
                    break
                if t.complete:
                    run['end'] = 'star'
                    break
            report['runs'].append(run)
            (args.output / 'report.json').write_text(json.dumps(report, indent=1, default=str))
            print(seed, run['decisions'], run['end'], run['health_trace'], flush=True)
    finally:
        backend.close()
        report['state_sha256_after'] = digest(STATE)
        (args.output / 'report.json').write_text(json.dumps(report, indent=1, default=str))
    print('events', [(e['seed'], e['decision'], e['health_before'], e['health_after']) for e in report['events']])
    print('owner state unchanged', report['state_sha256_after'] == before)


if __name__ == '__main__':
    main()
