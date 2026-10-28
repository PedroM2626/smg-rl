"""Record and verify a saved scripted search path from the canonical start."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.env import GalaxyEnv
from smg_rl.native import NativeDolphin
from smg_rl.parallel import runtime_directory
from smg_rl.plotting import plot_trajectory
from smg_rl.policy_video import VideoBackend
from smg_rl.profiles import PROFILES
from smg_rl.train import new_run

os.chdir(Path(__file__).resolve().parents[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    source = json.loads(Path(args.summary).read_text(encoding='utf-8'))
    entry = next(item for item in source['archive'] if item['sha256'] == args.sha256)
    if digest(entry['file']) != args.sha256:
        raise ValueError('The search snapshot changed after collection.')
    config = load_config('config/smg1-six-two-hour.yaml')
    config['model_free_run'] = 'evaluation'
    config['replay']['enabled'] = False
    config['paths']['runs'] = str(Path(config['root']) / 'runs/sweet-scripted-replay-20261028')
    run = new_run(config, 'replay')
    config['paths']['user'] = str(runtime_directory(config, run) / 'template')
    native = dict(config)
    native['parallel'] = dict(config['parallel'], frame_skip=2, native_capture='full')
    backend = NativeDolphin(native)
    video = VideoBackend(backend, run / 'scripted-replay.mp4', 'SCRIPTED FULL-START SEARCH REPLAY', decision_frames=12)
    result = {'state': 'running', 'run': str(run), 'source_summary': args.summary, 'source_state_sha256': args.sha256,
              'scope': 'Scripted physical full-start replay, not trained policy completion.', 'expected': entry,
              'physical_decisions': 0}
    atomic_json(run / 'result.json', result)
    print(json.dumps({'run': str(run)}), flush=True)
    env = None
    try:
        env = GalaxyEnv(config, video, run / 'worker-00', memory=backend.memory, intrinsic=False, variation=0)
        observation, initial = env.reset(seed=11, options={'variation': 0, 'full_start': True})
        if env.start_kind != 'full_start':
            raise ValueError('Replay did not use the canonical initial state.')
        result['initial'] = initial
        info = {}
        for action in entry['full_start_actions']:
            video.policy_image = observation['image'][-1].copy()
            video.policy_action_source = 'scripted_fixture'
            observation, _, terminal, truncated, info = env.step(action)
            result['physical_decisions'] += 1
            if terminal or truncated:
                break
        t = env.telemetry
        star = bool(t.complete and not t.dead and t.scene_nerve in PROFILES['RMGE01'].complete)
        result.update(state='completed', final=t.to_dict(), full_start_ram_star=star,
                      reached_exact_saved_state=bool(not t.dead and not t.complete
                          and t.scene_frame == entry['scene_frame'] and t.health == entry.get('health', 3)
                          and np.linalg.norm(np.subtract(t.position, entry['position'])) <= .01),
                      termination='star' if star else 'death' if t.dead else 'cut')
    except BaseException as exc:
        result.update(state='failed', error=repr(exc))
        raise
    finally:
        (env or video).close()
        result['frames_written'] = video.frames_written
        trace = run / 'worker-00' / 'sweet_sweet' / 'episode-000001.csv'
        if trace.exists():
            result['trajectory'] = str(trace)
            result['plot'] = str(plot_trajectory(trace, run / 'trajectory.png'))
        atomic_json(run / 'result.json', result)
    print(json.dumps({key: result[key] for key in ('state', 'run', 'physical_decisions',
        'full_start_ram_star', 'reached_exact_saved_state', 'termination')}), flush=True)


if __name__ == '__main__':
    main()
