"""Bounded frozen-policy timing sensitivity; no checkpoint conversion or learning."""

import argparse
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import psutil
import torch

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.env import GalaxyEnv
from smg_rl.native import NativeDolphin
from smg_rl.parallel import runtime_directory
from smg_rl.policy_video import VideoBackend
from smg_rl.replay import replay_signature
from smg_rl.train import new_run
from smg_rl.world_model import ImaginedAgent

CONDITIONS = ('legacy12', 'held12_inertia', 'held12_carry', 'held4_carry')
SEEDS = (11, 23, 47)
MAXIMUM_FRAMES = 1200


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game', choices=['smg1', 'smg2'], required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    base = load_config(f'config/regional-{args.game}-12.yaml')
    checkpoint = Path(args.checkpoint)
    metadata = json.loads(checkpoint.with_suffix('.json').read_text())
    expected = replay_signature(base)
    for key in ('game_id', 'scenario_names', 'action_names', 'image_size', 'crop'):
        if metadata['signature'].get(key) != expected.get(key):
            raise ValueError(f'Inference interfaces differ: {key}.')
    agent = ImaginedAgent.load(checkpoint, metadata['signature'], metadata['settings'], 'cuda')
    original = (agent.real_decisions, agent.model_updates, agent.actor_updates, agent.imagined_transitions)
    base['paths']['runs'] = 'runs/regional-intervention'
    run = new_run(base, f'control-{args.game}')
    report = {'state': 'running', 'game': args.game, 'run': str(run), 'attempts': [],
              'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
              'source_counts': dict(zip(('real', 'model', 'actor', 'imagined'), original)),
              'seeds': list(SEEDS), 'conditions': list(CONDITIONS), 'maximum_frames_per_attempt': MAXIMUM_FRAMES,
              'scope': 'Explicit inference-only adaptation. Frozen policy, canonical starts, no learning or replay.',
              'limitation': 'Four-frame recurrent dynamics are unvalidated. This is not a learning-rate comparison.',
              'wall_time_note': 'Seed11 includes video encoding and full RGBA capture. Compare throughput only within a capture mode.'}
    atomic_json(run / 'result.json', report)
    atomic_json(run / 'controller.json', {'pid': os.getpid(), 'created_time': psutil.Process().create_time(),
                                         'maximum_frames_per_attempt': MAXIMUM_FRAMES,
                                         'attempts': len(CONDITIONS) * len(SEEDS) * len(base['variations'])})
    print(json.dumps({'run': str(run), 'pid': os.getpid()}), flush=True)
    try:
        for condition in CONDITIONS:
            config = deepcopy(base)
            frames = 4 if condition == 'held4_carry' else 12
            config['parallel']['frame_skip'] = frames
            config['actions'] = ({'profile': 'motor8-v1'} if condition == 'legacy12' else
                {'profile': 'motor8-timed-v2', 'decision_frames': frames, 'pulse_frames': frames,
                 'jump_direction': 'carry' if 'carry' in condition else 'inertia'})
            config['regional_reward']['max_episode_frames'] = MAXIMUM_FRAMES
            config['world_model']['gamma'] = config['regional_reward']['gamma_per_frame'] ** frames
            config['replay']['enabled'] = False
            config['paths']['user'] = str(runtime_directory(config, run / condition) / 'template')
            folder = run / condition
            folder.mkdir()
            atomic_json(folder / 'experiment-config.json', config)
            for seed in SEEDS:
                # Identical capture path for paired conditions. Only seed11 adds video overlays.
                native_config = deepcopy(config)
                native_config['parallel'].update(native_capture='full' if seed == 11 else 'cpu',
                                                  frame_skip=2 if seed == 11 else frames)
                backend = NativeDolphin(native_config)
                video = None
                if seed == 11:
                    video = VideoBackend(backend, folder / 'seed11.mp4',
                        f"FROZEN POLICY TIMING TEST | {args.game} | {condition} | actor updates {agent.actor_updates}", frames)
                env = None
                try:
                    env = GalaxyEnv(config, video or backend, folder / f'seed-{seed}', memory=backend.memory, intrinsic=False)
                    for variation in range(len(config['variations'])):
                        torch.manual_seed(seed)
                        observation, _ = env.reset(seed=seed, options={'variation': variation, 'full_start': True})
                        latent, previous, first = None, None, np.ones(1, dtype=bool)
                        if video:
                            video.episode, video.step = variation + 1, 0
                        start = time.perf_counter()
                        infos = []
                        for _ in range(MAXIMUM_FRAMES // frames):
                            batch = {key: value[None] for key, value in observation.items()}
                            actions, latent, previous = agent.act(batch, latent, previous, first, epsilon=.05)
                            if video:
                                video.policy_image = observation['image'][-1].copy()
                                with torch.no_grad():
                                    scene = torch.as_tensor(batch['scenario'], device=agent.device)
                                    probabilities = agent.actor(agent.world.feature(latent, scene)).softmax(-1)[0]
                                    video.policy_probabilities = (.95 * probabilities + .05 / 8).cpu().numpy()
                            observation, _, terminated, truncated, info = env.step(int(actions[0]))
                            first[:] = False
                            infos.append(info)
                            if terminated or truncated:
                                break
                        elapsed = time.perf_counter() - start
                        report['attempts'].append({'condition': condition, 'seed': seed,
                            'variation': config['variations'][variation]['name'], 'frames': env.emulated_frames,
                            'decisions': env.steps, 'wall_seconds': elapsed,
                            'emulated_frames_per_second': env.emulated_frames / elapsed,
                            'confirmed_opening_fraction': env.route.best,
                            'confirmed_distance': env.route.best * env.route.total_length,
                            'damage': sum(i['health_lost'] for i in infos), 'dead': env.telemetry.dead,
                            'ram_star': env.telemetry.complete, 'start_kind': env.start_kind,
                            'outcome': env.rows[-1]['outcome'] if env.rows else
                                ('success' if env.telemetry.complete else 'death' if env.telemetry.dead else 'timeout'),
                            'video': str(video.destination) if video else None})
                        atomic_json(run / 'result.json', report)
                        print(json.dumps(report['attempts'][-1]), flush=True)
                finally:
                    if env:
                        env.close()
                    else:
                        (video or backend).close()
        assert original == (agent.real_decisions, agent.model_updates, agent.actor_updates, agent.imagined_transitions)
        assert digest(checkpoint) == report['checkpoint_sha256']
        report['state'] = 'completed'
    except BaseException as error:
        report.update(state='failed', error=repr(error))
        raise
    finally:
        atomic_json(run / 'result.json', report)


if __name__ == '__main__':
    main()
