"""Check the Luigi's Purple Coins RAM signals on the owner savestates, read-only, and write the evidence.

Usage: python scripts/verify_luigi_signals.py OUTPUT.json
Each owner state is loaded from its original path (the backend only copies it into its own disposable profile),
sampled through GameMemory with the configured signals, and the star-ready state is stepped with a jump until
the existing completion check fires. The state files are hashed before and after.
"""
import json
import struct
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from luigi_dump import load, open_backend  # noqa: E402

from smg_rl.checkpoints import digest  # noqa: E402

STAR_POLICY = (('jump',), ('w', 'jump'), ('w',))


def word(backend, address):
    return struct.unpack('>I', backend.memory.reader.read_bytes(address, 4))[0]


def sample(backend, mirrors):
    t = backend.memory.sample()
    return {'purple_coins': t.purple_coins, 'health': t.health, 'max_health': t.max_health, 'complete': t.complete,
            'dead': t.dead, 'grounded': t.grounded, 'mirrors': [word(backend, a) for a in mirrors],
            'scene_nerve': hex(t.scene_nerve)}


def check(name, mission, scratch):
    backend = open_backend(mission['game_id'], scratch / name)
    mirrors = mission['signals']['purple_coins'].get('mirrors', [])
    result = {'mission': mission['mission'], 'stage': mission['stage'], 'states': {}}
    try:
        for label, path in mission['states'].items():
            before = digest(path)
            load(backend, path)
            backend.memory.expected_stage = mission['stage']
            backend.memory.signals = mission['signals']
            for _ in range(2):
                backend.advance(())
            entry = {'path': path, 'sha256': before, 'loaded': sample(backend, mirrors)}
            if label == 'star_ready':
                for decision in range(1, 13):
                    backend.advance(STAR_POLICY[decision % 3])
                    t = backend.memory.sample()
                    if t.complete:
                        entry['star'] = {'decision': decision, 'scene_nerve': hex(t.scene_nerve),
                                         'purple_coins': t.purple_coins, 'dead': t.dead}
                        break
                else:
                    entry['star'] = None
            entry['sha256_after'] = digest(path)
            result['states'][label] = entry
    finally:
        backend.close()
    return result


def main():
    config = yaml.safe_load(Path('config/luigi-coins-signals.yaml').read_text())
    scratch = Path('.local/luigi-coins/verify')
    report = {name: check(name, mission, scratch) for name, mission in config.items()}
    Path(sys.argv[1]).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
