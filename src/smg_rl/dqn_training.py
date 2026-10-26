"""Finite real-pixel Rainbow/IQN collection, optimization and TensorBoard logging."""

import json
import os
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.utils.tensorboard import SummaryWriter

from smg_rl.actions import action_set
from smg_rl.diagnostics import StageProfiler, atomic_json
from smg_rl.dqn import DistributionalAgent
from smg_rl.dqn_replay import PrioritizedFrames
from smg_rl.parallel import WorkerFactory, runtime_directory
from smg_rl.profiles import PROFILES, game_id, require_training_enabled
from smg_rl.replay import replay_signature
from smg_rl.train import new_run
from smg_rl.vector import MeasuredSubprocVecEnv


def prepare_dqn(config, resume=None):
    require_training_enabled(config)
    if 'world_model' in config or config.get('novelty', {}).get('enabled', True):
        raise ValueError('Model-free profiles must omit world_model and explicitly disable visual curiosity.')
    s = config['dqn']
    integers = ('capacity', 'n_step', 'batch_size', 'learning_starts', 'train_every',
                'target_update_interval', 'quantiles', 'beta_steps', 'frame_stack')
    if any(type(s[k]) is not int or s[k] < 1 for k in integers):
        raise ValueError('DQN budgets/dimensions must be positive integers.')
    values = [s[k] for k in ('gamma', 'alpha', 'beta_start', 'epsilon', 'learning_rate', 'max_grad_norm', 'priority_epsilon')]
    if (not np.isfinite(values).all() or not 0 < s['gamma'] <= 1 or not 0 <= s['alpha'] <= 1
            or not 0 <= s['beta_start'] <= 1 or not 0 <= s['epsilon'] <= 1
            or min(s['learning_rate'], s['max_grad_norm'], s['priority_epsilon']) <= 0
            or s['capacity'] < s['learning_starts'] or s['frame_stack'] != config['environment']['frame_stack']
            or len(action_set(config)) != 8 or config['parallel']['frame_skip'] != 12):
        raise ValueError('Invalid model-free replay/learning settings.')
    parameters = {'capacity': s['capacity'], 'image_size': config['environment']['image_size'],
        'stack': s['frame_stack'], 'workers': config['parallel']['workers'],
        'n_step': s['n_step'], 'gamma': s['gamma'], 'alpha': s['alpha']}
    estimated = (2 * s['capacity'] + 2 * parameters['workers'] * (s['n_step'] + s['frame_stack']) + 16) * parameters['image_size'] ** 2
    if estimated > s['max_replay_bytes'] or estimated > .4 * psutil.virtual_memory().available:
        raise MemoryError('Configured frame replay exceeds its memory budget/current RAM headroom.')
    torch.set_num_threads(config['training']['torch_threads'])
    signature = replay_signature(config)
    if resume:
        return DistributionalAgent.load(resume, signature, s, parameters, config['training']['device'])
    return (DistributionalAgent(signature, s, config['training']['device'], config['training']['seed']),
            PrioritizedFrames(**parameters, seed=config['training']['seed']))


def train_dqn(config, steps, resume=None):
    config = deepcopy(config)
    workers = config['parallel']['workers']
    if steps < 1 or steps % workers:
        raise ValueError('Use a finite real-decision budget divisible by the worker count.')
    agent, replay = prepare_dqn(config, resume)
    config['model_free_run'] = 'online'
    run = new_run(config, 'dqn')
    atomic_json(run / 'controller.json', {'pid': os.getpid(), 'created_time': psutil.Process().create_time(),
        'workers': workers, 'algorithm': agent.settings['algorithm'], 'resume': str(resume) if resume else None,
        'requested_additional': steps, 'source_count': agent.real_decisions, 'target_count': agent.real_decisions + steps})
    print(json.dumps({'run': str(run)}), flush=True)
    config['paths']['user'] = str(runtime_directory(config, run) / 'template')
    started, initial = time.monotonic(), agent.real_decisions
    start_updates = agent.updates
    state, error, vector = 'running', None, None
    metrics, infos, actions_metrics = {}, (), {}
    counts = np.zeros(len(action_set(config)), np.int64)
    profiler = StageProfiler()
    logdir = Path(config['paths']['tensorboard']) / agent.settings['algorithm'] / game_id(config) / run.name
    writer = SummaryWriter(str(logdir), flush_secs=10)
    ledger = (run / 'collector-steps.jsonl').open('x', encoding='utf-8')
    writer.add_text('configuration', json.dumps(config, indent=2))
    writer.add_text('accounting', 'Only real emulator decisions; no world model, synthetic rollouts or imagined transitions.')
    full_stars, assisted_stars, deaths, damage, episode_counts = 0, 0, 0, 0, 0

    def publish():
        elapsed = time.monotonic() - started
        record = {'state': state, 'algorithm': agent.settings['algorithm'],
            'timesteps': agent.real_decisions, 'real_decisions': agent.real_decisions,
            'requested_additional': steps, 'source_count': initial, 'target_count': initial + steps,
            'optimizer_updates': agent.updates, 'target_updates': agent.target_updates,
            'imagined_transitions': 0, 'sampled_transitions': agent.updates * agent.settings['batch_size'],
            'replay_transitions': replay.size, 'replay_emitted': replay.emitted,
            'replay_bytes': replay.bytes_allocated, 'elapsed_seconds': elapsed,
            'decisions_per_second': (agent.real_decisions - initial) / max(elapsed, 1e-6),
            'emulated_frames_per_second': 12 * (agent.real_decisions - initial) / max(elapsed, 1e-6),
            'full_start_stars_this_run': full_stars, 'assisted_stars_this_run': assisted_stars,
            'episodes_this_run': episode_counts, 'deaths_this_run': deaths, 'nonfatal_hp_lost_this_run': damage,
            'tensorboard': str(logdir), 'updated_at': datetime.now(timezone.utc).isoformat(),
            'losses': metrics, 'timings': profiler.snapshot(), 'error': error,
            'start_kinds': [i.get('start_kind') for i in infos],
            'start_relative_route_progress': [i.get('start_relative_route_progress') for i in infos],
            'note': 'Opening route/projection values are not native mission percentage. No final-policy success claim from collection maxima.'}
        atomic_json(run / 'status.json', record)
        for tag, value in {**metrics, **actions_metrics}.items():
            writer.add_scalar(f'learning/{tag}', value, agent.real_decisions)
        for key in ('optimizer_updates', 'replay_transitions', 'replay_bytes', 'decisions_per_second',
                    'emulated_frames_per_second', 'full_start_stars_this_run', 'assisted_stars_this_run', 'deaths_this_run'):
            writer.add_scalar(f'training/{key}', record[key], agent.real_decisions)
        writer.add_scalar('resources/available_ram_gib', psutil.virtual_memory().available / 2**30, agent.real_decisions)
        for index, name in enumerate(agent.signature['action_names']):
            writer.add_scalar(f'actions/{name}', counts[index] / max(1, counts.sum()), agent.real_decisions)
        for name, item in profiler.snapshot().items():
            if isinstance(item, dict) and 'mean_ms' in item:
                writer.add_scalar(f'timing/{name}_mean_ms', item['mean_ms'], agent.real_decisions)

    try:
        shape = (config['environment']['frame_stack'], config['environment']['image_size'], config['environment']['image_size'])
        vector = MeasuredSubprocVecEnv([WorkerFactory(config, i, str(run)) for i in range(workers)],
                                      shape, shared=False, receive_order='ordered', timeout=config['parallel']['worker_timeout'])
        vector.seed(config['training']['seed'] + agent.real_decisions)
        observation = vector.reset()
        current = [replay.add_frame(image[-1]) for image in observation['image']]
        assisted = [i.get('start_kind') == 'curriculum' for i in vector.reset_infos]
        tick = 0
        publish()
        while agent.real_decisions < initial + steps and not (run / 'STOP').exists():
            policy_updates = agent.updates
            with profiler.measure('policy_inference'):
                actions, actions_metrics = agent.act(observation)
            counts += np.bincount(actions, minlength=len(counts))
            with profiler.measure('emulator_collection'):
                next_observation, rewards, done, infos = vector.step(actions)
            agent.real_decisions += workers
            ledger.write(json.dumps({'real_before_action': agent.real_decisions - workers,
                'policy_optimizer_updates': policy_updates, 'actions': actions.tolist(),
                'worker_episodes': [i['worker_episode'] for i in infos],
                'episode_steps': [i['episode_step'] for i in infos]}) + '\n')
            for index in range(workers):
                info = infos[index]
                terminal = bool(done[index] and not info.get('TimeLimit.truncated', False))
                successor = info['terminal_observation'] if done[index] else {
                    k: v[index] for k, v in next_observation.items()}
                next_frame = replay.add_frame(successor['image'][-1], current[index])
                replay.append(index, current[index], int(actions[index]), float(rewards[index]), next_frame,
                              int(observation['scenario'][index].argmax()), terminal, bool(done[index]), assisted[index])
                current[index] = (replay.add_frame(next_observation['image'][index, -1]) if done[index] else next_frame)
                damage += int(info.get('health_lost', 0)) if not info.get('dead') and not info.get('complete') else 0
                if done[index]:
                    name, kind = info['variation'], info['start_kind']
                    star = info.get('complete') and not info.get('dead') and info['scene_nerve'] in PROFILES[game_id(config)].complete
                    full_stars += int(bool(star) and kind == 'full_start')
                    assisted_stars += int(bool(star) and kind != 'full_start')
                    deaths += int(info.get('dead', False))
                    episode_counts += 1
                    for key, value in {'return': info['episode']['r'], 'decisions': info['episode']['l'],
                        'ram_star': bool(star), 'opening_advance': info.get('start_relative_route_progress', 0)}.items():
                        writer.add_scalar(f'episodes/{name}/{kind}/{key}', value, agent.real_decisions)
                    assisted[index] = vector.reset_infos[index].get('start_kind') == 'curriculum'
            observation = next_observation
            # Real-decision interval, independent of vector width.
            due = agent.real_decisions // agent.settings['train_every'] - (agent.real_decisions - workers) // agent.settings['train_every']
            if replay.size >= agent.settings['learning_starts']:
                for _ in range(due):
                    with profiler.measure('q_update'):
                        metrics = agent.update(replay)
            tick += 1
            if tick % 64 == 0:
                assert all(np.array_equal(replay.observation(current[i]), observation['image'][i]) for i in range(workers))
                writer.add_image('observations/actual_four_frame_stack', np.concatenate(observation['image'][0], axis=1),
                                 agent.real_decisions, dataformats='HW')
                publish()
                ledger.flush()
        state = 'stopped' if (run / 'STOP').exists() else 'completed'
    except KeyboardInterrupt:
        state = 'stopped'
    except BaseException as exc:
        state, error = 'failed', repr(exc)
        (run / 'error.txt').write_text(error, encoding='utf-8')
        raise
    finally:
        try:
            if vector is not None:
                vector.close()
            replay.flush_cuts()
            with profiler.measure('checkpoint'):
                agent.save(run / 'checkpoint-final', replay)
            assert replay.emitted == agent.real_decisions
            atomic_json(run / 'block-summary.json', {'source_real': initial, 'final_real': agent.real_decisions,
                'new_real': agent.real_decisions - initial, 'new_optimizer_updates': agent.updates - start_updates,
                'state': state, 'checkpoint': str(run / 'checkpoint-final'), 'imagined_transitions': 0})
            publish()
        finally:
            writer.flush()
            writer.close()
            ledger.close()
    return run
