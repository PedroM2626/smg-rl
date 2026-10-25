"""Reviewed finite exploration factorial; one explicitly selected job per invocation."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import uuid
from copy import deepcopy
from functools import wraps
from pathlib import Path

import numpy as np
import psutil
import torch

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.profiles import require_training_enabled
from smg_rl.replay import SequenceReplay, replay_signature
from smg_rl.train import new_run
from smg_rl.world_ablation import evaluate, freeze_validation
from smg_rl.world_model import ImaginedAgent
from smg_rl.world_training import train_world_model

ARMS = {'control': (0.01, 0.003), 'no_intrinsic': (0.0, 0.003),
        'low_entropy': (0.01, 0.0003), 'no_intrinsic_low_entropy': (0.0, 0.0003)}
COUNTS = ('real_decisions', 'model_updates', 'actor_updates', 'imagined_transitions')


def exclusive_state(index):
    """An OS lock prevents concurrent dispatch/review from racing JSON updates."""
    def decorate(function):
        @wraps(function)
        def guarded(*args, **kwargs):
            path = Path(args[index])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.with_suffix('.lock').open('a+b') as lock:
                try:
                    lock.seek(0)
                    if os.name == 'nt':
                        import msvcrt
                        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as error:
                    raise RuntimeError('Another process owns this experiment state.') from error
                return function(*args, **kwargs)
        return guarded
    return decorate


def _counts(agent):
    return {key:getattr(agent, key) for key in COUNTS}


def review_run(run):
    run = Path(run)
    status = json.loads((run / 'status.json').read_text())
    rows, outcomes = [], []
    for path in sorted(run.glob('worker-*/*/episode-*.csv')):
        with path.open(newline='', encoding='utf-8') as stream:
            trace = list(csv.DictReader(stream))
        if trace:
            outcomes.append(trace[-1]['outcome'])
            rows.extend(r for r in trace if r['action'] != 'reset')
    route_values = [float(r['route_prefix_progress']) for r in rows
                    if r.get('route_prefix_progress') not in (None,'')]
    return {'run':str(run), 'state':status['state'], 'counters':{k:status[k] for k in COUNTS},
            'gate':status.get('diagnostics', {}).get('imagination_gate', {}),
            'replay_rejected':status.get('diagnostics', {}).get('replay', {}).get('rejected', []),
            'deaths':outcomes.count('death'), 'episodes':len(outcomes),
            'successes':sum(r['complete'] == 'True' for r in rows),
            'health_lost':sum(int(r.get('health_lost') or 0) for r in rows),
            'route_measurement_available':bool(route_values),
            'route_prefix_exited':any(r.get('route_prefix_exited') == 'True' for r in rows) if route_values else None,
            'maximum_route_prefix_progress':max(route_values, default=None),
            'maximum_geometric_projection':max((float(r['best_progress']) for r in rows), default=0),
            'special_action_fraction':sum(any(k in r['action'].split('+') for k in ('jump','spin','crouch'))
                                          for r in rows) / max(1, len(rows)),
            'decisions_per_second':status['decisions_per_second']}


@exclusive_state(2)
def initialize(config, checkpoint, state_path, critic_updates=64, actor_updates=128):
    """Freeze the world, adapt critics equally, then fit four paired actor arms."""
    require_training_enabled(config)
    state_path, checkpoint = Path(state_path), Path(checkpoint)
    if state_path.exists():
        raise FileExistsError('Retain the existing experiment state; never initialize twice.')
    if not 1 <= critic_updates <= 256 or not 1 <= actor_updates <= 1024:
        raise ValueError('Use finite critic and actor budgets.')
    torch.set_num_threads(1)
    metadata = json.loads(checkpoint.with_suffix('.json').read_text())
    signature, settings = replay_signature(config), metadata['settings']
    if signature['game_id'] != 'RMGE01' or signature['scenario_names'] != ['sweet_sweet']:
        raise ValueError('This initial factorial protocol is specific to SMG1 Sweet Sweet Galaxy.')
    replay = SequenceReplay(config['paths']['replay'], signature)
    summary = replay.refresh()
    if summary['rejected']:
        raise ValueError('Rejected source replay prevents a paired trial.')
    fixed, episodes = freeze_validation(replay, settings, 8, True)
    root = new_run(config, 'exploration-factorial')
    state = {'schema':1, 'state':'offline_fitting', 'root':str(root), 'source':str(checkpoint),
             'source_sha256':digest(checkpoint), 'source_replay':config['paths']['replay'],
             'protocol':'docs/exploration-interventions.md', 'seeds':[11,23,47],
             'critic_updates_per_case':critic_updates, 'actor_updates_per_case':actor_updates,
             'initial_health_budget':256, 'first_block_total_budget':2000,
             'extension_total_budget':10000, 'audit_reuses_validation':True,
             'audit_episode_ids':episodes, 'audits':[], 'cases':[], 'reviews':[]}
    atomic_json(state_path, state)
    agent = ImaginedAgent.load(checkpoint, signature, settings, config['training']['device'])
    for seed in state['seeds']:
        state['audits'].append({'seed':seed, **evaluate(agent, fixed, settings, seed, summary)})
    del agent
    if not all(row['gate']['ready'] for row in state['audits']):
        state['state'] = 'offline_gate_failed'
        atomic_json(state_path, state)
        return state
    for seed in state['seeds']:
        for name, (curiosity, entropy) in ARMS.items():
            case_root = root / f'{name}-seed-{seed}'
            case_root.mkdir()
            agent = ImaginedAgent.load(checkpoint, signature, settings, config['training']['device'])
            original = _counts(agent)
            frozen = {k:v.clone() for k,v in agent.world.state_dict().items()}
            actor = {k:v.clone() for k,v in agent.actor.state_dict().items()}
            variant = {**settings, 'disagreement_scale':curiosity, 'entropy':entropy}
            agent.settings = agent.world.settings = variant
            replay.rng = np.random.default_rng(seed)
            for step in range(critic_updates + actor_updates):
                require_training_enabled(config)
                if (root / 'STOP').exists():
                    raise InterruptedError('Experiment stopped at the requested boundary.')
                torch.manual_seed(seed * 100000 + step)
                batch = replay.sample(variant['batch_size'], variant['sequence_length'], variant['burn_in'])
                with torch.no_grad():
                    _, _, posterior, scenarios = agent.world.loss(agent.tensors(batch), auxiliary=False)
                agent.update_imagination(posterior, scenarios, train_actor=step >= critic_updates)
                if step == critic_updates - 1:
                    assert all(torch.equal(v, actor[k]) for k,v in agent.actor.state_dict().items())
            assert agent.real_decisions == original['real_decisions'] and agent.model_updates == original['model_updates']
            assert all(torch.equal(v, frozen[k]) for k,v in agent.world.state_dict().items())
            destination = case_root / 'adapted.pt'
            agent.save(destination)
            case_config = deepcopy(config)
            case_config['world_model'] = variant
            case_config['training']['seed'] = seed
            trial_id = hashlib.sha256(str(root).encode()).hexdigest()[:8]
            code = {'control':'c','no_intrinsic':'n','low_entropy':'e','no_intrinsic_low_entropy':'ne'}[name] + str(seed)
            case_config['paths']['runs'] = str(Path(config['root']) / 'runs' / f'explore-{trial_id}' / code)
            case_config['paths']['replay'] = str(Path(config['root']) / 'runs' / f'explore-{trial_id}-replay' / code)
            case_config['paths']['route'] = str(Path(config['root']) / 'config/routes/smg1-initial-exit.json')
            case_config['frontier'] = {'enabled':False}
            case_config['parallel']['workers'] = 1
            case = {'name':name, 'seed':seed, 'state':'fit_ready', 'checkpoint':str(destination),
                    'config':case_config, 'source_counters':original, 'fitted_counters':_counts(agent),
                    'new_online_decisions':0, 'runs':[], 'reviews':[]}
            state['cases'].append(case)
            atomic_json(state_path, state)
            print(json.dumps({'fitted':name, 'seed':seed, 'actor_updates':actor_updates,
                              'critic_only_updates':critic_updates}), flush=True)
            del agent
    assert digest(checkpoint) == metadata['sha256']
    state['state'] = 'awaiting_health_review'
    atomic_json(state_path, state)
    return state


def _snapshot(source, destination):
    """Flat atomic snapshots avoid Windows path limits and incomplete replays."""
    if destination.exists():
        marker = destination / 'snapshot-complete.json'
        if not marker.exists():
            raise ValueError('Incomplete replay snapshot; retain it and select a fresh short destination.')
        for item in json.loads(marker.read_text())['files']:
            if Path(item['file']).name != item['file'] or not (destination/item['file']).is_file():
                raise ValueError('A completed replay snapshot has missing or invalid files.')
        return
    pending = destination.parent.with_name(destination.parent.name + '.snapshots') / uuid.uuid4().hex[:8]
    pending.mkdir(parents=True)
    entries = []
    for source_file in sorted(source.rglob('*.npz')):
        relative = str(source_file.relative_to(source))
        target = pending / (hashlib.sha256(relative.encode()).hexdigest()[:24] + '.npz')
        try:
            os.link(source_file, target)
        except OSError:
            shutil.copy2(source_file, target)
        entries.append({'source_relative_path':relative, 'file':target.name, 'sha256':digest(target)})
    atomic_json(pending / 'snapshot-complete.json', {'source':str(source), 'files':entries})
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending.replace(destination)


@exclusive_state(0)
def online(state_path, name, seed, target):
    """Run one 256/2000/10000 cumulative block, never its successor."""
    state_path = Path(state_path)
    state = json.loads(state_path.read_text())
    if state['state'] != 'awaiting_health_review':
        raise ValueError('The complete offline comparison must finish before online dispatch.')
    case = next(c for c in state['cases'] if c['name'] == name and c['seed'] == seed)
    targets = {0:256, 256:2000, 2000:10000}
    if targets.get(case['new_online_decisions']) != target:
        raise ValueError('Run/review the previous cumulative block first.')
    if case['state'] not in ('fit_ready', 'reviewed'):
        raise ValueError('This case needs review, is running or failed. Never duplicate it.')
    if target > 256 and case['state'] != 'reviewed':
        raise ValueError('A documented stage review is required.')
    for other in state['cases']:
        if other.get('state') == 'running':
            raise RuntimeError('Review the running case first; online jobs are serialized.')
    config = case['config']
    require_training_enabled(config)
    _snapshot(Path(state['source_replay']), Path(config['paths']['replay']) / 'source')
    case.update(state='running', controller_pid=os.getpid(), controller_created=psutil.Process().create_time())
    atomic_json(state_path, state)
    try:
        run = train_world_model(config, target - case['new_online_decisions'], case['checkpoint'])
        review = review_run(run)
        case['runs'].append(str(run))
        case['checkpoint'] = str(Path(run) / 'world-final.pt')
        case['new_online_decisions'] = target
        case['latest_evidence'] = review
        case['state'] = 'awaiting_review'
        atomic_json(state_path, state)
        print(json.dumps(review), flush=True)
        return review
    except BaseException as error:
        case.update(state='failed', error=repr(error))
        atomic_json(state_path, state)
        raise


@exclusive_state(0)
def review(state_path, name, seed, decision, reason):
    state_path = Path(state_path)
    state = json.loads(state_path.read_text())
    if decision not in ('continue','stop'):
        raise ValueError('Choose an explicit continue or stop review decision.')
    case = next(c for c in state['cases'] if c['name'] == name and c['seed'] == seed)
    if case['state'] != 'awaiting_review' or len(reason.strip()) < 20:
        raise ValueError('Review completed evidence and provide a meaningful reason.')
    evidence = review_run(case['runs'][-1])
    if decision == 'continue' and (evidence['state'] != 'completed' or evidence['replay_rejected']):
        raise ValueError('Fix failed collection/replay before continuing.')
    case['reviews'].append({'decisions':case['new_online_decisions'], 'decision':decision,
                            'reason':reason, 'evidence':evidence})
    case['state'] = 'reviewed' if decision == 'continue' else 'stopped'
    atomic_json(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['initialize','online','review'])
    parser.add_argument('--state', default='.local/exploration-trial-state.json')
    parser.add_argument('--config', default='config/staged-smg1-control.yaml')
    parser.add_argument('--checkpoint', default='runs/staged-smg1-exploration/control/world-model-20261025-004551-267957/world-final.pt')
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seed', type=int, choices=[11,23,47])
    parser.add_argument('--target', type=int, choices=[256,2000,10000])
    parser.add_argument('--decision', choices=['continue','stop'])
    parser.add_argument('--reason')
    args = parser.parse_args()
    if args.command == 'initialize':
        initialize(load_config(args.config), args.checkpoint, args.state)
    elif args.command == 'online':
        online(args.state, args.arm, args.seed, args.target)
    else:
        review(args.state, args.arm, args.seed, args.decision, args.reason or '')


if __name__ == '__main__':
    main()
