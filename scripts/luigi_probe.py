"""Read-only probe of an owner savestate: stage name, position, a full-resolution frame and optional memory dumps.

The savestate is only read (the backend copies it into its own disposable profile). No game memory is written.

Usage: python scripts/luigi_probe.py GAME_ID STATE OUTDIR [--idle N] [--dump]
"""
import argparse
import json
import struct
from pathlib import Path

import cv2
import numpy as np

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.native import NativeDolphin
from smg_rl.profiles import PROFILES

CONFIGS = {'SB4E01': 'config/motor8-smg2.yaml', 'RMGE01': 'config/motor8-smg1.yaml'}


def raw_state(reader, game_id):
    """The same pointer chain as GameMemory.sample(), without its stage or scenario restrictions."""
    p = PROFILES[game_id]

    def u32(address):
        return struct.unpack('>I', reader.read_bytes(address, 4))[0]

    root = u32(p.root)
    controller = u32(root + 0x24)
    scene = u32(controller + p.scene_offset)
    stage = reader.read_bytes(controller + p.stage_offset, 32).split(b'\0')[0].decode('ascii', 'replace')
    scenario = u32(controller + p.scenario_offset)
    holder = u32(scene + 0x10)
    mario_holder = u32(holder + p.mario_holder_offset)
    actor = u32(mario_holder + p.actor_offset)
    position = struct.unpack('>fff', reader.read_bytes(actor + p.position_offset, 12))
    return {'stage': stage, 'scenario': scenario, 'position': position, 'actor': hex(actor),
            'scene_nerve': hex(u32(p.complete[0])), 'controller': hex(controller)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_id')
    parser.add_argument('state', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--idle', type=int, default=0)
    parser.add_argument('--capture', default='full')
    args = parser.parse_args()
    before = digest(args.state)
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(CONFIGS[args.game_id])
    config['parallel'].update(native_capture=args.capture, workers=1)
    config['paths']['user'] = str((args.output / 'profile' / 'template').resolve())
    backend = NativeDolphin(config)
    report = {'state': str(args.state), 'sha256': before, 'frames': []}
    try:
        try:
            backend.load_state(args.state)
        except RuntimeError as error:
            # Some owner states publish a frame the bridge rejects at load time; the load itself may still have
            # happened, which the raw read below and an idle step verify.
            report['load_error'] = str(error)[:200]
            backend.paused = True
            backend.advance(())
        report['loaded'] = raw_state(backend.memory.reader, args.game_id)
        cv2.imwrite(str(args.output / 'frame-0.png'), cv2.cvtColor(backend.bridge.image(), cv2.COLOR_RGBA2BGR))
        for step in range(args.idle):
            backend.advance(())
        if args.idle:
            report['after_idle'] = raw_state(backend.memory.reader, args.game_id)
            cv2.imwrite(str(args.output / 'frame-idle.png'),
                        cv2.cvtColor(backend.bridge.image(), cv2.COLOR_RGBA2BGR))
    finally:
        backend.close()
    report['sha256_after'] = digest(args.state)
    (args.output / 'probe.json').write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != 'frames'}, default=str))
    assert np is not None


if __name__ == '__main__':
    main()
