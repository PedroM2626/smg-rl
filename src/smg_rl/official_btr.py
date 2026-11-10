"""Drive the vendored official BTR agent (third_party/btr, MIT) with this project's environment workers.

The agent, its prioritised replay and its networks are the unchanged files of VIPTankz/BTR; this module only
supplies the collection loop of that repository's `main.py` on top of `MeasuredSubprocVecEnv`: choose actions for
the batch, start the step, call `learn`, finish the step, store every transition. Defaults are the official
`--vector 0` desktop settings (four environments, batch 16, learning rate 5e-5, replay ratio 1, spectral norm,
Munchausen, IQN, noisy layers, hard target copy every 500 updates, rewards clipped to [-1, 1]); the scale-dependent
settings of the official 50M-frame runs (the 200,000-transition warm-up, the 2M-step epsilon schedule and the
1M-transition memory) are configurable and are scaled for the budgets used here.
"""

import json
import os
import sys
import time
import types
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch

from smg_rl.actions import action_set
from smg_rl.diagnostics import atomic_json
from smg_rl.lean_worker import LeanWorkerFactory
from smg_rl.parallel import runtime_directory
from smg_rl.profiles import PROFILES, game_id, require_training_enabled
from smg_rl.resource_disk import disk_safe, disk_snapshot, require_disk_headroom
from smg_rl.resource_memory import memory_safe, memory_snapshot, require_memory_headroom
from smg_rl.train import new_run
from smg_rl.vector import MeasuredSubprocVecEnv

VENDOR = Path(__file__).resolve().parents[2] / 'third_party' / 'btr'

# Official defaults of `main.py --vector 0`; `official_btr` in the configuration overrides any of them.
OFFICIAL = {'batch_size': 16, 'rr': 1, 'lr': 5e-5, 'taus': 8, 'model_size': 2, 'maxpool_size': 6, 'linear_size': 512,
            'ncos': 64, 'noisy': True, 'spectral': True, 'munch': True, 'munch_alpha': 0.9, 'iqn': True,
            'double': False, 'dueling': True, 'impala': True, 'discount': 0.997, 'adamw': False, 'per': True,
            'per_alpha': 0.2, 'target_replace': 500, 'ema': False, 'n': 3, 'grad_clip': 10, 'eps_steps': 2_000_000,
            'eps_disable': True, 'max_mem_size': 1_048_576, 'min_sampling_size': 200_000, 'reward_clip': True}


def load_official_agent():
    """Import `Agent` from the vendored directory; `Agent.py` imports torchvision only in commented code."""
    try:
        import torchvision.transforms.functional  # noqa: F401
    except ImportError:
        root = types.ModuleType('torchvision')
        transforms = types.ModuleType('torchvision.transforms')
        functional = types.ModuleType('torchvision.transforms.functional')
        root.transforms, transforms.functional = transforms, functional
        sys.modules.update({'torchvision': root, 'torchvision.transforms': transforms,
                            'torchvision.transforms.functional': functional})
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    import Agent as official
    return official.Agent


def official_settings(config, workers):
    settings = {**OFFICIAL, **config.get('official_btr', {})}
    unknown = set(config.get('official_btr', {})) - set(OFFICIAL)
    if unknown:
        raise ValueError(f'Unknown official BTR settings: {sorted(unknown)}')
    if abs(settings['discount'] - config['dqn']['gamma']) > 1e-12 or config['dqn']['frame_stack'] != 4:
        raise ValueError('The official agent discount must equal the configured learner discount (four-frame stack).')
    if (type(settings['batch_size']) is not int or settings['batch_size'] < 1 or settings['min_sampling_size'] < 1
            or settings['max_mem_size'] < settings['min_sampling_size'] or workers < 1):
        raise ValueError('Invalid official BTR batch, warm-up or memory size.')
    return settings


def train_official_btr(config, steps, max_seconds=None, vector_factory=None):
    config = deepcopy(config)
    require_training_enabled(config)
    workers = config['parallel']['workers']
    settings = official_settings(config, workers)
    physical_reserve = config.get('resource_guard', {}).get('physical_reserve_bytes', 512 * 2**20)
    memory_safe(memory_snapshot(), physical_reserve_bytes=physical_reserve)
    if steps < 1 or steps % workers:
        raise ValueError('Use a finite real-decision budget divisible by the worker count.')
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError('The optional wall-clock budget must be positive and finite.')
    require_disk_headroom(config)
    Agent = load_official_agent()
    device = torch.device(config['training']['device'] if torch.cuda.is_available() or
                          config['training']['device'] == 'cpu' else 'cpu')
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    stack, size = config['environment']['frame_stack'], config['environment']['image_size']
    run = new_run(config, 'official-btr')
    atomic_json(run / 'controller.json', {'pid': os.getpid(), 'created_time': psutil.Process().create_time(),
                'workers': workers, 'algorithm': 'official-btr', 'requested_additional': steps,
                'max_seconds': max_seconds, 'settings': settings})
    print(json.dumps({'run': str(run)}), flush=True)
    config['paths']['user'] = str(runtime_directory(config, run) / 'template')
    agent = Agent(n_actions=len(action_set(config)), input_dims=[stack, size, size], device=device, num_envs=workers,
                  agent_name='official-btr', total_frames=steps, testing=False, batch_size=settings['batch_size'],
                  rr=settings['rr'], lr=settings['lr'], maxpool_size=settings['maxpool_size'], ema=settings['ema'],
                  target_replace=settings['target_replace'], noisy=settings['noisy'], spectral=settings['spectral'],
                  munch=settings['munch'], iqn=settings['iqn'], double=settings['double'], dueling=settings['dueling'],
                  impala=settings['impala'], discount=settings['discount'], adamw=settings['adamw'],
                  per=settings['per'], taus=settings['taus'], model_size=settings['model_size'],
                  linear_size=settings['linear_size'], ncos=settings['ncos'], maxpool=True, replay_period=workers,
                  framestack=stack, arch='impala', per_alpha=settings['per_alpha'], eps_steps=settings['eps_steps'],
                  eps_disable=settings['eps_disable'], max_mem_size=settings['max_mem_size'], n=settings['n'],
                  munch_alpha=settings['munch_alpha'], grad_clip=settings['grad_clip'], imagex=size, imagey=size)
    agent.min_sampling_size = settings['min_sampling_size']  # 200,000 in the official 50M-frame runs
    started, decisions, episodes, stars, deaths = time.perf_counter(), 0, 0, 0, 0
    attempts = {}
    state, error, stop_reason, vector = 'running', None, None, None
    attempts_ledger = (run / 'attempts.jsonl').open('x', encoding='utf-8')
    counts = np.zeros(len(action_set(config)), np.int64)

    def publish():
        elapsed = time.perf_counter() - started
        atomic_json(run / 'status.json', {
            'state': state, 'algorithm': 'official-btr', 'real_decisions': decisions, 'requested_additional': steps,
            'optimizer_updates': agent.grad_steps, 'elapsed_seconds': elapsed, 'max_seconds': max_seconds,
            'stop_reason': stop_reason, 'epsilon': agent.epsilon.eps, 'min_sampling_size': agent.min_sampling_size,
            'decisions_per_second': decisions / max(elapsed, 1e-6), 'full_start_stars_this_run': stars,
            'episodes_this_run': episodes, 'deaths_this_run': deaths, 'attempts': attempts,
            'completed_attempts': sum(v['completed'] for v in attempts.values()),
            'action_share': (counts / max(1, counts.sum())).round(4).tolist(),
            'updated_at': datetime.now(timezone.utc).isoformat(), 'error': error, 'memory': memory_snapshot(),
            'note': 'Official BTR agent code; opening route values are not native mission percentage.'})

    try:
        require_memory_headroom()
        vector = (vector_factory or default_vector)(config, run, workers, (stack, size, size))
        vector.seed(config['training']['seed'])
        observation = vector.reset()
        publish()
        tick = 0
        while decisions < steps and not (run / 'STOP').exists():
            if max_seconds is not None and time.perf_counter() - started >= max_seconds:
                stop_reason = 'wall_clock_budget'
                break
            actions = np.asarray(agent.choose_action(observation['image']).cpu().numpy(), dtype=np.int64)
            counts += np.bincount(actions, minlength=len(counts))
            vector.step_async(actions)
            agent.learn()  # As in the official loop: learn while the environments step.
            next_observation, rewards, done, infos = vector.step_wait()
            decisions += workers
            for index in range(workers):
                info = infos[index]
                truncated = bool(done[index] and info.get('TimeLimit.truncated', False))
                terminated = bool(done[index] and not truncated)
                successor = info['terminal_observation']['image'] if done[index] else next_observation['image'][index]
                reward = float(np.clip(rewards[index], -1., 1.)) if settings['reward_clip'] else float(rewards[index])
                agent.store_transition(observation['image'][index], int(actions[index]), reward, successor,
                                       terminated, truncated, stream=index)
                if done[index]:
                    kind = info['start_kind']
                    star = bool(info.get('complete') and not info.get('dead')
                                and info['scene_nerve'] in PROFILES[game_id(config)].complete)
                    stars += int(star and kind == 'full_start')
                    deaths += int(bool(info.get('dead', False)))
                    episodes += 1
                    key = f"{info['variation']}/{kind}"
                    record = attempts.setdefault(key, {'completed': 0, 'ram_stars': 0, 'deaths': 0, 'timeouts': 0})
                    record['completed'] += 1
                    record['ram_stars'] += int(star)
                    record['deaths'] += int(bool(info.get('dead')))
                    record['timeouts'] += int(not info.get('dead') and not star)
                    attempts_ledger.write(json.dumps({
                        'mission': info['variation'], 'start_kind': kind, 'attempt': record['completed'],
                        'worker': index, 'worker_episode': info['worker_episode'], 'real_decisions': decisions,
                        'optimizer_updates': agent.grad_steps, 'ram_star': star, 'goal_reached': bool(info.get('goal_reached')), 'dead': bool(info.get('dead')),
                        'opening_advance': float(info.get('start_relative_route_progress') or 0.),
                        'return': float(info['episode']['r']), 'decisions': int(info['episode']['l'])}) + '\n')
            observation = next_observation
            tick += 1
            if tick % 64 == 0:
                publish()
                attempts_ledger.flush()
                if not memory_safe(memory_snapshot(), physical_reserve_bytes=physical_reserve):
                    stop_reason = 'memory_pressure'
                    break
                if not disk_safe(disk_snapshot(config)):
                    stop_reason = 'disk_pressure'
                    break
        state = 'stopped' if (run / 'STOP').exists() or stop_reason in ('memory_pressure', 'disk_pressure') else 'completed'
        stop_reason = stop_reason or ('owner_stop' if state == 'stopped' else 'decision_budget')
    except KeyboardInterrupt:
        state, stop_reason = 'stopped', 'interrupted'
    except BaseException as exc:
        state, error = 'failed', repr(exc)
        (run / 'error.txt').write_text(error, encoding='utf-8')
        raise
    finally:
        try:
            if vector is not None:
                vector.close()
            folder = run / 'checkpoint-final'
            folder.mkdir(exist_ok=True)
            torch.save({'net': agent.net.state_dict(), 'target': agent.tgt_net.state_dict(),
                        'optimizer': agent.optimizer.state_dict(), 'env_steps': agent.env_steps,
                        'grad_steps': agent.grad_steps, 'settings': settings}, folder / 'official-btr.pt')
            atomic_json(run / 'block-summary.json', {'final_real': decisions, 'state': state,
                        'stop_reason': stop_reason, 'optimizer_updates': agent.grad_steps,
                        'checkpoint': str(folder), 'note': 'Network weights only; the replay is not saved, so a run cannot resume.'})
            publish()
        finally:
            attempts_ledger.close()
    return {'run': str(run), 'state': state, 'real_decisions': decisions, 'stop_reason': stop_reason}


def default_vector(config, run, workers, shape):
    return MeasuredSubprocVecEnv([LeanWorkerFactory(config, i, str(run)) for i in range(workers)], shape,
                                 shared=False, receive_order='ordered', timeout=config['parallel']['worker_timeout'],
                                 lean=True)
