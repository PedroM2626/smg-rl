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


def checkpoint_signature(config):
    """The identity a checkpoint must carry to be evaluated under `config`.

    Normally it is the config's own replay signature. A config with a `zero_shot` block is an evaluation-only transfer
    test: the checkpoint belongs to the source config, and the network inputs and outputs (pixels, crop, frame skip,
    action set and execution, scenario count) must be identical so the frozen weights apply unchanged.
    """
    own = replay_signature(config)
    source = config.get('zero_shot', {}).get('source_config')
    if not source:
        return own
    if config.get('training', {}).get('enabled'):
        raise ValueError('A zero-shot evaluation config must have training disabled.')
    trained = replay_signature(load_config(source))
    for key in ('image_size', 'crop', 'frame_skip', 'action_names', 'action_execution'):
        if trained.get(key) != own.get(key):
            raise ValueError(f'Zero-shot transfer needs the same {key} as the source config.')
    if len(trained['scenario_names']) != len(own['scenario_names']):
        raise ValueError('Zero-shot transfer needs the same number of scenario inputs as the source config.')
    return trained


def record_dqn(config_path, checkpoint, variation, seeds=(11, 23, 47), maximum_steps=300, stochastic=False,
               epsilon_override=None, assisted_start=False, anchor=None):
    if not seeds or maximum_steps < 1:
        raise ValueError('Finite evaluation seeds and a positive decision budget are required.')
    if epsilon_override is not None and (not stochastic or not np.isfinite(epsilon_override)
                                         or not 0 <= epsilon_override <= 1):
        raise ValueError('An explicit epsilon requires stochastic evaluation and must be between zero and one.')
    config = load_config(config_path)
    if variation not in range(len(config['variations'])):
        raise ValueError('Evaluation variation is outside the configured missions.')
    if anchor is not None and not assisted_start:
        raise ValueError('A fixed evaluation anchor requires an assisted start.')
    if assisted_start and not config.get('curriculum', {}).get('anchor_diagnostic'):
        raise ValueError('Assisted-start evaluation requires an anchor-only diagnostic curriculum.')
    config['model_free_run'] = 'evaluation'
    agent = DistributionalAgent.load_policy(checkpoint, checkpoint_signature(config), config['training']['device'])
    epsilon = agent.epsilon(evaluate=not stochastic) if epsilon_override is None else epsilon_override
    mode = 'noisynet_epsilon' if stochastic else 'epsilon_greedy' if epsilon else 'greedy'
    torch.set_num_threads(config['training']['torch_threads'])
    run = new_run(config, 'eval-dqn-video')
    atomic_json(run / 'controller.json', {'pid': os.getpid(), 'created_time': psutil.Process().create_time(),
        'role': 'frozen_model_free_evaluation', 'checkpoint': str(checkpoint),
        'variation': variation, 'seeds': list(seeds), 'maximum_decisions_per_seed': maximum_steps,
        'evaluation_mode': mode, 'evaluation_epsilon': epsilon, 'assisted_start': assisted_start,
        'anchor': anchor})
    print({'evaluation_run': str(run), 'variation': variation}, flush=True)
    config['paths']['user'] = str(runtime_directory(config, run) / 'template')
    native_config = deepcopy(config)
    native_config['parallel'].update(frame_skip=2, native_capture='full')
    backend = NativeDolphin(native_config)
    name = config['variations'][variation]['name']
    ancestry = config.get('study', {})
    counter = (f"new real {agent.real_decisions} | source real {ancestry['source_real_decisions']}"
               if ancestry.get('source_real_decisions') is not None else f"real {agent.real_decisions}")
    video = VideoBackend(backend, run / 'policy.mp4',
        f"{agent.settings['algorithm'].upper()} {mode.upper()} EVAL | {name} | eps {epsilon:.3f} | {counter}",
        decision_frames=config['parallel']['frame_skip'])
    env = None
    record = {'state': 'running', 'checkpoint': str(checkpoint),
        'agent_sha256': digest(Path(checkpoint) / 'agent.pt'), 'variation': name, 'episodes': [],
        'optimizer_updates': agent.updates, 'algorithm': agent.settings['algorithm'],
        'initialization_ancestry': ancestry,
        'physical_training_decisions_added': 0, 'imagined_transitions': 0,
        'evaluation_epsilon': epsilon, 'evaluation_mode': mode, 'parameter_noise_enabled': stochastic,
        'quantiles': 'random' if stochastic else 'fixed_midpoints',
        'assisted_start': assisted_start, 'anchor': anchor,
        'scope': ('Assisted curriculum-anchor start' if assisted_start else 'Canonical-start')
                 + ' frozen weights; explicit parameter-noise/epsilon mode. No fitting, training or training-success accounting.',
        'overlay': 'Exact stacked policy grayscale frames (four, oldest first), raw expected Q returns (not probabilities), selected action and applied controls.',
        'video': str((run / 'policy.mp4').resolve())}
    atomic_json(run / 'recording.json', record)
    try:
        env = GalaxyEnv(config, video, run / 'worker-00', memory=backend.memory, intrinsic=False, variation=variation)
        if anchor is not None:
            # Evaluate one named verified anchor regardless of the learner's current frontier.
            fixed = next(e for e in env.curriculum_bank.entries if e['file'] == anchor)
            env.curriculum_bank.choose = lambda rng, scenario: fixed
        for episode, seed in enumerate(seeds, 1):
            agent.rng = np.random.default_rng(seed)
            torch.manual_seed(seed)
            observation, reset_info = env.reset(seed=seed, options={'full_start': not assisted_start})
            video.episode, video.step = episode, 0
            completed, info, random_decisions = False, {}, 0
            for step in range(maximum_steps):
                actions, metrics = agent.act({k: v[None] for k, v in observation.items()},
                                            evaluate=not stochastic, return_values=True,
                                            epsilon_override=epsilon_override)
                random_action = bool(metrics['epsilon_random_mask'][0])
                random_decisions += int(random_action)
                video.policy_image = observation['image'][-1].copy()
                video.policy_stack = observation['image'].copy()
                video.policy_q_values = metrics['q_values'][0]
                video.policy_action_source = 'epsilon_random' if random_action else 'noisynet_argmax' if stochastic else 'greedy_argmax'
                video.policy_epsilon = epsilon
                observation, _, terminated, truncated, info = env.step(int(actions[0]))
                if terminated or truncated:
                    completed = True
                    break
            record['episodes'].append({'seed': seed, 'decisions': step + 1,
                'outcome': 'star' if info['complete'] else 'death' if info['dead'] else 'timeout' if completed else 'evaluation_budget_cut',
                'start_kind': reset_info['start_kind'], 'anchor_id': reset_info['anchor_id'],
                'full_start_ram_star': bool(info['complete'] and not info['dead']) and not assisted_start,
                'assisted_ram_star': bool(info['complete'] and not info['dead']) and assisted_start,
                'epsilon_random_decisions': random_decisions,
                'opening_fraction': info['route_prefix_progress'],
                'purple_coins_best': info.get('purple_coins_best'),
                'zero_shot_source': config.get('zero_shot', {}).get('source_config'),
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
