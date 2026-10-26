"""Finite canonical-start model-free policy evaluation with exact Q/input overlays."""

import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import psutil
import torch

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.dqn import DistributionalAgent
from smg_rl.env import GalaxyEnv
from smg_rl.native import NativeDolphin
from smg_rl.parallel import runtime_directory
from smg_rl.policy_video import VideoBackend
from smg_rl.replay import replay_signature
from smg_rl.train import new_run


def record_dqn(config_path, checkpoint, variation, seeds=(11, 23, 47), maximum_steps=300):
    if not seeds or maximum_steps < 1:
        raise ValueError('Finite evaluation seeds and a positive decision budget are required.')
    config = load_config(config_path)
    if variation not in range(len(config['variations'])):
        raise ValueError('Evaluation variation is outside the configured missions.')
    config['model_free_run'] = 'evaluation'
    agent = DistributionalAgent.load_policy(checkpoint, replay_signature(config), config['training']['device'])
    torch.set_num_threads(config['training']['torch_threads'])
    run = new_run(config, 'eval-dqn-video')
    atomic_json(run / 'controller.json', {'pid': os.getpid(), 'created_time': psutil.Process().create_time(),
        'role': 'frozen_model_free_evaluation', 'checkpoint': str(checkpoint),
        'variation': variation, 'seeds': list(seeds), 'maximum_decisions_per_seed': maximum_steps})
    print({'evaluation_run': str(run), 'variation': variation}, flush=True)
    config['paths']['user'] = str(runtime_directory(config, run) / 'template')
    native_config = deepcopy(config)
    native_config['parallel'].update(frame_skip=2, native_capture='full')
    backend = NativeDolphin(native_config)
    name = config['variations'][variation]['name']
    video = VideoBackend(backend, run / 'policy.mp4',
        f"{agent.settings['algorithm'].upper()} EVALUATION | {name} | real {agent.real_decisions} | Q updates {agent.updates}",
        decision_frames=config['parallel']['frame_skip'])
    env = None
    record = {'state': 'running', 'checkpoint': str(checkpoint),
        'agent_sha256': digest(Path(checkpoint) / 'agent.pt'), 'variation': name, 'episodes': [],
        'optimizer_updates': agent.updates, 'algorithm': agent.settings['algorithm'],
        'physical_training_decisions_added': 0, 'imagined_transitions': 0,
        'evaluation_epsilon': agent.settings.get('evaluation_epsilon', 0),
        'scope': 'Canonical-start frozen policy evaluation, noisy parameters disabled; small explicit epsilon. Not training-success accounting.',
        'overlay': 'Exact policy grayscale frame, raw expected Q returns (not probabilities), selected action and applied controls.',
        'video': str((run / 'policy.mp4').resolve())}
    atomic_json(run / 'recording.json', record)
    try:
        env = GalaxyEnv(config, video, run / 'worker-00', memory=backend.memory, intrinsic=False, variation=variation)
        for episode, seed in enumerate(seeds, 1):
            agent.rng = np.random.default_rng(seed)
            torch.manual_seed(seed)
            observation, _ = env.reset(seed=seed, options={'full_start': True})
            video.episode, video.step = episode, 0
            completed, info = False, {}
            for step in range(maximum_steps):
                actions, metrics = agent.act({k: v[None] for k, v in observation.items()}, evaluate=True, return_values=True)
                video.policy_image = observation['image'][-1].copy()
                video.policy_q_values = metrics['q_values'][0]
                observation, _, terminated, truncated, info = env.step(int(actions[0]))
                if terminated or truncated:
                    completed = True
                    break
            record['episodes'].append({'seed': seed, 'decisions': step + 1,
                'outcome': 'star' if info['complete'] else 'death' if info['dead'] else 'timeout' if completed else 'evaluation_budget_cut',
                'full_start_ram_star': bool(info['complete'] and not info['dead']),
                'opening_fraction': info['route_prefix_progress'],
                'maximum_geometric_projection_percent': 100 * info['best_progress'],
                'telemetry': env.telemetry.to_dict()})
            atomic_json(run / 'recording.json', record)
    except BaseException as exc:
        record.update(state='failed', error=repr(exc))
        raise
    finally:
        try:
            (env or video).close()
        finally:
            record['frames_written'] = video.frames_written
            atomic_json(run / 'recording.json', record)
    record.update(state='completed', frames_written=video.frames_written)
    assert digest(Path(checkpoint) / 'agent.pt') == record['agent_sha256']
    atomic_json(run / 'recording.json', record)
    return run
