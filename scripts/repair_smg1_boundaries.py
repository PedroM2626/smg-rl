"""Finite paired boundary-loss repair on retained real replay, without actors."""

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
from smg_rl.exploration_trial import exclusive_state
from smg_rl.replay import SequenceReplay, replay_signature
from smg_rl.train import new_run
from smg_rl.world_ablation import evaluate, freeze_validation
from smg_rl.world_model import ImaginedAgent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['control', 'balanced', 'heads', 'heads_strong',
                                        'decoder', 'decoder_rollout', 'prior'], required=True)
    parser.add_argument('--updates', type=int, default=256)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    run_repair(args, Path('.local/smg1-boundary-repair-state.json'))


@exclusive_state(1)
def run_repair(args, state_path):
    """Serialize allocation claims so two invocations cannot spend one budget."""
    if args.updates not in (64, 128, 256, 512):
        raise ValueError('Use an explicit finite reviewed repair allocation.')
    state = json.loads(state_path.read_text())
    allocation = state['allocations'][args.arm]
    if allocation['state'] != 'pending' or allocation['maximum_updates'] != args.updates:
        raise ValueError('This exact repair allocation is absent, active or already completed.')
    source = Path(args.source)
    source_hash = allocation.get('source_sha256', state['source_sha256'])
    if digest(source) != source_hash:
        raise ValueError('Repair source changed.')
    config = load_config('config/motor8-smg1.yaml')
    metadata = json.loads(source.with_suffix('.json').read_text())
    torch.set_num_threads(1)
    agent = ImaginedAgent.load(source, replay_signature(config), metadata['settings'],
                               config['training']['device'])
    source_counts = {k: getattr(agent, k) for k in
                     ('real_decisions', 'model_updates', 'actor_updates', 'imagined_transitions')}
    settings = deepcopy(metadata['settings'])
    if args.arm in ('balanced', 'heads', 'heads_strong'):
        settings['balance_terminal_losses'] = True
    if args.arm.startswith('heads'):
        weight = 40.0 if args.arm == 'heads_strong' else 20.0
        settings.update(terminal_reward_weight=weight, terminal_weight=weight)
        for parameter in list(agent.world.reward.parameters()) + list(agent.world.continuation.parameters()):
            agent.model_optimizer.state.pop(parameter, None)
    frozen = {k: v.clone() for k, v in agent.world.state_dict().items()
              if not k.startswith(('reward.', 'continuation.'))} if args.arm.startswith('heads') else {}
    if args.arm.startswith('decoder'):
        if args.arm == 'decoder_rollout':
            settings['open_loop_training_weight'] = 2.0
        frozen = {k: v.clone() for k, v in agent.world.state_dict().items()
                  if not k.startswith(('decoder.', 'decode_input.'))}
        for group in agent.model_optimizer.param_groups:
            group['lr'] = 1e-4
    if args.arm == 'prior':
        settings['open_loop_training_weight'] = 2.0
        frozen = {k: v.clone() for k, v in agent.world.state_dict().items()
                  if not k.startswith('prior_net.')}
    agent.settings = agent.world.settings = settings
    config['world_model'] = settings
    config['paths']['runs'] = 'runs/m8-smg1-boundary'
    config['world_model_run'] = 'offline-boundary-repair-no-actor'
    run = new_run(config, f'boundary-{args.arm}')
    allocation.update(state='running', run=str(run), source_counts=source_counts,
                      command=f'{__file__} --arm {args.arm} --updates {args.updates} --source {source}')
    controller = {'pid': os.getpid(), 'created_time': psutil.Process().create_time(),
                  'workers': 0, 'maximum_model_updates': args.updates, 'actor_updates_allowed': 0}
    allocation['controller'] = controller
    assert atomic_json(run / 'controller.json', controller)
    assert atomic_json(state_path, state)
    replay = SequenceReplay(config['paths']['replay'], agent.signature, 20261025)
    summary = replay.refresh()
    if summary['rejected'] or summary['transitions'] != 4096:
        raise ValueError('Repair requires the unchanged complete compatible bootstrap.')
    fixed, episodes = freeze_validation(replay, settings, 8, True)
    report = {'state': 'running', 'arm': args.arm, 'source': str(source),
              'source_sha256': digest(source), 'source_counts': source_counts,
              'settings': settings, 'replay': summary, 'audit_episodes': episodes,
              'seeds': [11, 23, 47], 'maximum_updates': args.updates, 'stages': [],
              'scope': 'Only training episodes fitted. Reused held-out prediction audit, '
              'not unseen testing or independent training replications. No actor, '
              'imagined transitions, emulator or new real decisions.'}
    report['baseline'] = [evaluate(agent, fixed, settings, seed, summary)
                          for seed in report['seeds']]
    try:
        for step in range(args.updates):
            torch.manual_seed(41000 + step)
            batch = replay.sample(settings['batch_size'], settings['sequence_length'],
                                  settings['burn_in'], terminal_fraction=settings['training_terminal_fraction'])
            if args.arm.startswith('heads'):
                metrics = agent.update_prediction_heads(batch)
            elif args.arm.startswith('decoder'):
                metrics = agent.update_visual_decoder(batch)
            elif args.arm == 'prior':
                metrics = agent.update_prior_dynamics(batch)
            else:
                metrics, _, _ = agent.update_model(batch)
            if not np.isfinite(list(metrics.values())).all():
                raise FloatingPointError('Nonfinite repair update.')
            interval = 32 if args.arm.startswith('heads') else 64 if args.arm.startswith('decoder') or args.arm == 'prior' else 128
            if (step + 1) % interval == 0:
                checkpoint = run / f'world-{step + 1}.pt'
                agent.replay_rng_state = replay.rng.bit_generator.state
                if args.arm.startswith('decoder'):
                    for group in agent.model_optimizer.param_groups:
                        group['lr'] = settings['model_lr']
                agent.save(checkpoint)
                if args.arm.startswith('decoder'):
                    for group in agent.model_optimizer.param_groups:
                        group['lr'] = 1e-4
                audits = [evaluate(agent, fixed, settings, seed, summary) for seed in report['seeds']]
                stage = {'additional_model_updates': step + 1, 'checkpoint': str(checkpoint),
                         'checkpoint_sha256': digest(checkpoint), 'audits': audits,
                         'gate_passes': sum(a['gate']['ready'] for a in audits)}
                report['stages'].append(stage)
                assert atomic_json(run / 'repair.json', report)
                print(json.dumps({'arm': args.arm, 'updates': step + 1,
                                  'gate_passes': stage['gate_passes'],
                                  'metrics': [a['metrics']['sweet_sweet'] for a in audits]}), flush=True)
        for key in ('real_decisions', 'actor_updates', 'imagined_transitions'):
            assert getattr(agent, key) == source_counts[key]
        assert digest(source) == source_hash
        assert all(torch.equal(value, agent.world.state_dict()[key]) for key, value in frozen.items())
        report['state'] = 'completed'
        allocation.update(state='completed_review_pending', additional_model_updates=args.updates,
                          checkpoint=report['stages'][-1]['checkpoint'],
                          gate_passes=report['stages'][-1]['gate_passes'])
    except BaseException as error:
        report['state'] = 'failed'
        allocation.update(state='failed', error=repr(error))
        raise
    finally:
        assert atomic_json(run / 'repair.json', report)
        current = json.loads(state_path.read_text())
        current['allocations'][args.arm] = allocation
        assert atomic_json(state_path, current)


if __name__ == '__main__':
    main()
