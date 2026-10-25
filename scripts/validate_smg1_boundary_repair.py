"""Finite fresh full-start prediction audit; never records training replay."""

import argparse
import json
import os
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
from smg_rl.prediction_audit import validation_batches
from smg_rl.replay import SequenceReplay, replay_signature
from smg_rl.train import new_run
from smg_rl.world_ablation import FixedValidation, evaluate
from smg_rl.world_model import ImaginedAgent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--seeds', nargs=4, type=int, default=[101, 137, 173, 211])
    args = parser.parse_args()
    if len(set(args.seeds)) != 4:
        raise ValueError('Use four distinct predeclared evaluation seeds.')
    config = load_config('config/motor8-smg1.yaml')
    checkpoint = Path(args.checkpoint)
    settings = json.loads(checkpoint.with_suffix('.json').read_text())['settings']
    agent = ImaginedAgent.load(checkpoint, replay_signature(config), settings, config['training']['device'])
    counts = {k: getattr(agent, k) for k in
              ('real_decisions', 'model_updates', 'actor_updates', 'imagined_transitions')}
    torch.set_num_threads(1)
    config = deepcopy(config)
    config['world_model'] = settings
    config['replay']['enabled'] = False
    config['frontier'] = {'enabled': False}
    config['environment']['max_steps'] = 256
    config['paths']['runs'] = 'runs/m8-smg1-boundary'
    run = new_run(config, 'fresh-prediction-validation')
    atomic_json(run / 'controller.json', {'pid': os.getpid(), 'created_time': psutil.Process().create_time(),
                                         'maximum_physical_decisions': 1024, 'training_allowed': False})
    config['paths']['user'] = str(runtime_directory(config, run) / 'template')
    report = {'state': 'collecting', 'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
              'source_counts': counts, 'seeds': args.seeds, 'maximum_decisions_per_seed': 256,
              'episodes': [], 'physical_validation_decisions': 0,
              'scope': 'Fresh full-start untrained-prior trajectories, no optimizer or training replay. '
              'These episodes were not used for fitting or candidate selection.'}
    atomic_json(run / 'validation.json', report)
    print(json.dumps({'run': str(run), 'maximum_physical_decisions': 1024}), flush=True)
    backend = NativeDolphin(config)
    env = GalaxyEnv(config, backend, run / 'worker-00', memory=backend.memory, intrinsic=False)
    traces = []
    try:
        for episode, seed in enumerate(report['seeds']):
            torch.manual_seed(seed)
            observation, _ = env.reset(seed=seed)
            images = [observation['image'][-1].copy()]
            actions, rewards, terminals, truncations = [], [], [], []
            latent, previous, first = None, None, np.ones(1, dtype=bool)
            for _ in range(256):
                batch = {k: v[None] for k, v in observation.items()}
                action, latent, previous = agent.act(batch, latent, previous, first, epsilon=.15)
                observation, reward, terminated, truncated, info = env.step(int(action[0]))
                images.append(observation['image'][-1].copy())
                actions.append(int(action[0]))
                rewards.append(reward)
                terminals.append(terminated)
                truncations.append(truncated)
                first[:] = False
                if terminated or truncated:
                    break
            trace = {'images': np.stack(images), 'actions': np.asarray(actions, np.int64),
                     'rewards': np.asarray(rewards, np.float32), 'terminated': np.asarray(terminals),
                     'truncated': np.asarray(truncations),
                     'is_first': np.asarray([True] + [False] * len(actions)),
                     'scenario': np.ones(1, np.float32)}
            traces.append(trace)
            # Deliberately not a ReplayRecorder archive: incompatible keys/purpose
            # prevent this audit data from entering ordinary training replay.
            np.savez_compressed(run / f'validation-only-{episode}.npz',
                                purpose=np.asarray('fresh_validation_never_fit'),
                                **{('observation_frames' if k == 'images' else k): v for k, v in trace.items()})
            report['physical_validation_decisions'] += len(actions)
            report['episodes'].append({'seed': seed, 'decisions': len(actions),
                                       'outcome': 'death' if info['dead'] else 'star' if info['complete'] else 'timeout'})
            atomic_json(run / 'validation.json', report)
            print(json.dumps(report['episodes'][-1]), flush=True)
    finally:
        env.close()
    batches = validation_batches(traces, settings)
    fixed = FixedValidation(agent.signature, batches)
    replay = SequenceReplay(config['paths']['replay'], agent.signature)
    summary = replay.refresh()
    summary['validation_episodes'] = len(traces)
    summary['validation_scenarios'] = {'sweet_sweet': report['physical_validation_decisions']}
    report['gate_coverage'] = summary
    report['coverage_note'] = 'Replay minimum refers to the retained4096 transitions; validation coverage refers only to these four fresh episodes. No audit data added to training replay.'
    report['audits'] = [evaluate(agent, fixed, settings, seed, summary) for seed in (11, 23, 47)]
    report['gate_passes'] = sum(a['gate']['ready'] for a in report['audits'])
    report['state'] = 'completed'
    assert counts == {k: getattr(agent, k) for k in counts}
    assert digest(checkpoint) == report['checkpoint_sha256']
    atomic_json(run / 'validation.json', report)
    print(json.dumps({'gate_passes': report['gate_passes'], 'physical_validation_decisions': report['physical_validation_decisions'],
                      'metrics': [a['metrics']['sweet_sweet'] for a in report['audits']]}), flush=True)


if __name__ == '__main__':
    main()
