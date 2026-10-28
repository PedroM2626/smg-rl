"""Bounded, explicitly paired genetic evaluation for route-free experiments."""

import hashlib
import json
import math
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.utils.tensorboard import SummaryWriter

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.dqn import DistributionalAgent
from smg_rl.genetic import (
    GeneticEvaluator,
    GeneticHead,
    checked_json,
    genome_digest,
    next_population,
)
from smg_rl.lean_worker import LeanWorkerFactory
from smg_rl.parallel import runtime_directory
from smg_rl.profiles import require_training_enabled
from smg_rl.replay import replay_signature
from smg_rl.resource_disk import require_disk_headroom
from smg_rl.resource_memory import require_memory_headroom
from smg_rl.vector import MeasuredSubprocVecEnv

CONFIG = 'config/genetic-goal-distance-smg1.yaml'
SOURCE = Path('runs/model-free/smg1-six-two-hour/dqn-20261026-204307-660910/checkpoint-final')
HEAD = Path('runs/model-free/genetic-goal-distance-smg1/genetic-20261027-191407-420841/champion.pt')


def network_digest(network):
    result = hashlib.sha256()
    for name, value in sorted(network.state_dict().items()):
        result.update(name.encode('utf-8') + b'\0')
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def verified_stage_review(path, stage, run=None):
    if path is None:
        raise ValueError('A completed, independently reviewed prior stage is required.')
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    parent = Path(report['run'])
    if (not report.get('audit_passed') or report.get('stage') != stage
            or report.get('state') != 'completed'
            or run is not None and parent.resolve() != Path(run).resolve()
            or digest(parent / 'summary.json') != report['summary_sha256']
            or digest(parent / 'trials.jsonl') != report['trials_sha256']):
        raise ValueError('Prior-stage review is invalid, blocked or stale.')
    return report


def draw_seed(generation, draw, namespace=11):
    """Common action-randomness seeds; never a claim of independent Wii states."""
    return int(np.random.SeedSequence([namespace, generation, draw]).generate_state(1)[0])


def aggregate_draws(records, population_size, draws):
    """Only a complete group of distinct physical draws may enter selection."""
    groups = {i: [] for i in range(population_size)}
    for record, _ in records:
        if not record['fitness_eligible'] or record['individual'] not in groups:
            raise ValueError('Partial or unknown physical draw cannot enter selection.')
        groups[record['individual']].append(record)
    scores = []
    for group in groups.values():
        if len(group) != draws or len({r['draw'] for r in group}) != draws:
            raise ValueError('Each candidate requires the exact declared independent action draws.')
        if len({r['genome_sha256'] for r in group}) != 1:
            raise ValueError('Draws from different genomes cannot be aggregated.')
        scores.append(float(np.mean([r['fitness_world_units'] for r in group])))
    return np.asarray(scores, dtype=np.float64)


def calibration_decision(records, base, sigmas=(.02, .05, .10)):
    """Predeclared viability/diversity rule, using behavior prefix rather than parameter L2."""
    groups = {}
    for sigma in sigmas:
        rows = [r for r in records if r['sigma'] == sigma]
        if len(rows) != 16 or any(not r['fitness_eligible'] for r in rows):
            raise ValueError('Calibration requires 16 complete valid trials per mutation scale.')
        viable = sum(bool(r['ram_star']) or (r['decisions'] > 60 and r['fitness_world_units'] > 0) for r in rows)
        different = sum(r['action_prefix60_sha256'] != base['action_prefix60_sha256'] for r in rows)
        groups[str(sigma)] = {'trials': len(rows), 'viable': viable, 'behavior_changed': different,
                              'mean_fitness': float(np.mean([r['fitness_world_units'] for r in rows])),
                              'stars': sum(bool(r['ram_star']) for r in rows)}
    qualifying = [sigma for sigma in sigmas if groups[str(sigma)]['viable'] >= 8
                  and groups[str(sigma)]['behavior_changed'] >= 4]
    return {'groups': groups, 'selected_sigma': max(qualifying) if qualifying else None,
            'rule': 'Largest sigma with >=8/16 viable and >=4/16 distinct first-60 action sequence; a star is viable.'}


def setup(stage, seconds, max_decisions):
    if not math.isfinite(seconds) or not 60 <= seconds <= 10800 or max_decisions not in (400, 1200):
        raise ValueError('Stage requires an explicit finite time and declared decision horizon.')
    config = load_config(CONFIG)
    require_training_enabled(config)
    require_disk_headroom(config)
    require_memory_headroom()
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('A Dolphin instance is active; refuse overlapping collectors.')
    source_hashes = {name: digest(SOURCE / name) for name in ('agent.pt', 'replay.npz', 'manifest.json')}
    parent = json.loads((HEAD.parent / 'summary.json').read_text(encoding='utf-8'))
    if (parent['state'] != 'completed' or parent['error'] or not parent['source_unchanged']
            or parent['fitness_mode'] != 'goal-distance-xyz-v1'
            or parent['source_hashes'] != source_hashes or digest(HEAD) != parent['champion_sha256']):
        raise ValueError('Parent champion/source audit failed.')
    torch.set_num_threads(config['training'].get('torch_threads', 1))
    agent = DistributionalAgent.load_policy(SOURCE, replay_signature(config), config['training']['device'])
    policy = GeneticHead(agent)
    saved = torch.load(HEAD, map_location='cpu', weights_only=True)
    if not all(torch.equal(v.cpu(), saved['source_online'][k]) for k, v in policy.network.state_dict().items()):
        raise ValueError('Frozen visual source mismatch.')
    policy.source = saved['genome'].numpy().copy()
    if policy.source.shape != (4104,) or not np.isfinite(policy.source).all():
        raise ValueError('Initial genetic head is invalid.')
    config = deepcopy(config)
    config['curriculum']['enabled'] = False
    config['regional_reward']['enabled'] = False
    config['route_reward']['enabled'] = False
    config['paths'].pop('route', None)
    config['environment'].update(max_steps=max_decisions, idle_limit=max_decisions)
    config['parallel']['plot_every'] = 1000000
    config['paths']['runs'] = str(Path(config['root']) / 'runs/model-free/genetic-robust-smg1')
    config['paths']['user'] = str(Path(config['root']) / '.local/genetic-robust-smg1/template')
    run = Path(config['paths']['runs']) / (stage + '-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    run.mkdir(parents=True, exist_ok=False)
    config['paths']['user'] = str(runtime_directory(config, run) / 'template')
    checked_json(run / 'configuration.json', config)
    checked_json(run / 'controller.json', {'pid': psutil.Process().pid, 'created_time': psutil.Process().create_time(),
                                           'stage': stage, 'seconds': seconds, 'source': str(SOURCE),
                                           'head': str(HEAD), 'head_sha256': digest(HEAD)})
    print(json.dumps({'run': str(run), 'stage': stage}), flush=True)
    status = {'state': 'running', 'stage': stage, 'fitness_mode': 'goal-distance-xyz-v1',
              'real_decisions': 0, 'completed_trials': 0, 'partial_trials': 0, 'full_start_stars': 0,
              'deaths': 0, 'nonfatal_hp_lost': 0, 'action_counts': [0] * 8, 'gradient_updates': 0,
              'source_hashes': source_hashes, 'initial_head_sha256': digest(HEAD),
              'initial_genome_sha256': genome_digest(policy.source),
              'frozen_network_before': network_digest(policy.network)}
    checked_json(run / 'status.json', status)
    return config, policy, run, status


def run_calibration(seconds=1800):
    """Evaluate one baseline and 48 mutations; finish early once all are reviewed."""
    started, deadline = time.monotonic(), time.time() + seconds
    config, policy, run, status = setup('calibration', seconds, 400)
    rng = np.random.default_rng(20261027)
    noise = rng.normal(0, 1, (16, len(policy.source))).astype(np.float32)
    candidates = [{'individual': 0, 'seed': 11, 'genome': policy.source.copy(), 'sigma': 0.0}]
    for sigma in (.02, .05, .10):
        for i in range(16):
            candidates.append({'individual': len(candidates), 'seed': 11, 'sigma': sigma,
                               'genome': np.clip(policy.source + sigma * noise[i], -5, 5).astype(np.float32)})
    with (run / 'population-initial.npz').open('xb') as stream:
        np.savez_compressed(stream, genomes=np.stack([c['genome'] for c in candidates]))
    checked_json(run / 'protocol.json', {'stage': 'calibration', 'candidates': 49, 'per_sigma': 16,
        'sigmas': [.02, .05, .10], 'noise_seed': 20261027, 'max_decisions': 400,
        'rule': 'Largest sigma with >=8 viable and >=4 distinct first-60 action signatures; stars are viable.',
        'fixed_game_state': True, 'seconds_ceiling': seconds})
    writer = SummaryWriter(str(Path(config['paths']['tensorboard']) / 'robust-calibration' / run.name))
    vector = evaluator = None
    try:
        shape = (config['environment']['frame_stack'], config['environment']['image_size'], config['environment']['image_size'])
        vector = MeasuredSubprocVecEnv([LeanWorkerFactory(config, i, str(run)) for i in range(6)], shape,
                                       shared=False, receive_order='ordered', timeout=config['parallel']['worker_timeout'], lean=True)
        evaluator = GeneticEvaluator(config, run, policy, vector, started + seconds, status, writer, deadline)
        trials = evaluator.evaluate(candidates, 'calibration', 0, started + seconds)
        if len(trials) != len(candidates):
            status.update(state='incomplete', reason='Time budget expired before every candidate completed.')
        else:
            rows = [record for record, _ in trials]
            base = next(r for r in rows if r['sigma'] == 0)
            decision = calibration_decision(rows, base)
            status.update(state='completed' if decision['selected_sigma'] is not None else 'blocked',
                          decision=decision, baseline=base)
    except BaseException as exc:
        status.update(state='failed', error=repr(exc))
        raise
    finally:
        if vector is not None:
            vector.close()
        if evaluator is not None:
            evaluator.ledger.close()
        writer.flush()
        writer.close()
        status['elapsed_seconds'] = time.monotonic() - started
        status['source_unchanged'] = all(digest(SOURCE / n) == h for n, h in status['source_hashes'].items())
        status['initial_head_unchanged'] = digest(HEAD) == status['initial_head_sha256']
        status['frozen_network_after'] = network_digest(policy.network)
        if (not status['source_unchanged'] or not status['initial_head_unchanged']
                or status['frozen_network_before'] != status['frozen_network_after']):
            status.update(state='failed', error='Protected source/head changed.')
        checked_json(run / 'status.json', status)
        checked_json(run / 'summary.json', status)
    return run


def run_arm(arm, sigma, seconds=10800, population_seed=20261027, calibration_review=None,
            prior_arm_review=None):
    """Run one bounded, fresh arm. Partial generations never reproduce."""
    if arm not in ('B', 'C') or sigma not in (.02, .05, .10):
        raise ValueError('Use a declared arm and calibrated sigma.')
    review = verified_stage_review(calibration_review, 'calibration')
    if review['decision']['selected_sigma'] != sigma:
        raise ValueError('Requested mutation scale did not pass the preregistered calibration.')
    if arm == 'C':
        previous = verified_stage_review(prior_arm_review, 'arm-b')
        previous_summary = json.loads((Path(previous['run']) / 'summary.json').read_text(encoding='utf-8'))
        if (previous_summary['protocol']['mutation_std'] != sigma
                or previous_summary['protocol']['population_seed'] != population_seed
                or previous_summary['generations_completed'] < 1):
            raise ValueError('Control arm must match the reviewed treatment arm parameters.')
    started, deadline = time.monotonic(), time.time() + seconds
    config, policy, run, status = setup('arm-' + arm.lower(), seconds, 1200)
    rng = np.random.default_rng(population_seed)
    population = np.stack([np.clip(policy.source + rng.normal(0, sigma, len(policy.source)), -5, 5)
                           .astype(np.float32) for _ in range(48)])
    population[0] = policy.source.copy()
    mode, draws = ('epsilon', 3) if arm == 'B' else ('argmax', 1)
    protocol = {'arm': arm, 'population': 48, 'elites': 8, 'tournament': 3, 'mutation_std': sigma,
                'population_seed': population_seed, 'selection_mode': mode,
                'evaluation_epsilon': .05 if arm == 'B' else 0.0, 'selection_draws': draws,
                'argmax_diagnostic': True, 'max_decisions': 1200, 'seconds_ceiling': seconds,
                'fitness': 'mean of completed real alive-grounded XYZ proximity draws',
                'pairing': 'same action seeds per draw/generation across all candidates; same canonical Wii state',
                'fresh_population': True, 'reused_fitness': False}
    protocol['calibration_review'] = str(calibration_review)
    protocol['calibration_review_sha256'] = digest(calibration_review)
    if prior_arm_review:
        protocol['prior_arm_review'] = str(prior_arm_review)
        protocol['prior_arm_review_sha256'] = digest(prior_arm_review)
    checked_json(run / 'protocol.json', protocol)
    status.update(protocol=protocol, generations_completed=0, last_complete_generation=None)
    checked_json(run / 'status.json', status)
    writer = SummaryWriter(str(Path(config['paths']['tensorboard']) / f'robust-{arm.lower()}' / run.name))
    vector = evaluator = None
    try:
        shape = (config['environment']['frame_stack'], config['environment']['image_size'], config['environment']['image_size'])
        vector = MeasuredSubprocVecEnv([LeanWorkerFactory(config, i, str(run)) for i in range(6)], shape,
                                       shared=False, receive_order='ordered', timeout=config['parallel']['worker_timeout'], lean=True)
        evaluator = GeneticEvaluator(config, run, policy, vector, started + seconds, status, writer, deadline)
        for generation in range(1000):
            if evaluator.stopped():
                break
            candidates = [{'individual': i, 'seed': 11, 'genome': genome, 'draw': d,
                           'evaluation_mode': mode, 'evaluation_epsilon': .05 if arm == 'B' else 0.0,
                           'action_seed': draw_seed(generation, d, namespace=11)}
                          for i, genome in enumerate(population) for d in range(draws)]
            selection = evaluator.evaluate(candidates, 'selection', generation, started + seconds)
            if len(selection) != len(candidates):
                break
            scores = aggregate_draws(selection, 48, draws)
            if arm == 'B':
                diagnostic = [{'individual': i, 'seed': 11, 'genome': genome, 'draw': 0,
                               'evaluation_mode': 'argmax', 'evaluation_epsilon': 0.0,
                               'action_seed': 0} for i, genome in enumerate(population)]
                greedy = evaluator.evaluate(diagnostic, 'argmax_diagnostic', generation, started + seconds)
                if len(greedy) != 48:
                    break
            else:
                greedy = selection
            greedy_scores = aggregate_draws(greedy, 48, 1)
            rows = [r for r, _ in greedy]
            reference = policy.source
            offsets = np.linalg.norm(population - reference, axis=1)
            pairwise = np.linalg.norm(population[:, None] - population[None, :], axis=2)
            selected = np.argsort(-scores, kind='stable')
            with (run / f'generation-{generation:04d}.npz').open('xb') as stream:
                np.savez_compressed(stream, genomes=population, selection_fitness=scores, greedy_fitness=greedy_scores)
            generation_evidence = {'generation': generation, 'selection_draws': len(selection),
                'greedy_draws': len(greedy), 'selection_best': float(scores[selected[0]]),
                'selection_mean': float(scores.mean()), 'greedy_best': float(greedy_scores.max()),
                'top8': selected[:8].tolist(), 'genome_hashes': [genome_digest(g) for g in population],
                'median_l2_from_initial': float(np.median(offsets)),
                'median_pairwise_l2': float(np.median(pairwise[np.triu_indices(48, 1)])),
                'unique_action_prefixes': len({r['action_prefix60_sha256'] for r in rows}),
                'deaths_within_60': sum(r['dead'] and r['decisions'] <= 60 for r in rows),
                'rng_state_before_reproduction': rng.bit_generator.state}
            checked_json(run / f'generation-{generation:04d}.json', generation_evidence)
            status.update(generations_completed=generation + 1, last_complete_generation=generation,
                          last_generation=generation_evidence)
            checked_json(run / 'status.json', status)
            writer.add_scalar('generation/selection_best', float(scores[selected[0]]), generation)
            writer.add_scalar('generation/greedy_best', float(greedy_scores.max()), generation)
            population = next_population(population, scores, rng, elite=8, sigma=sigma)
        status.update(state='completed', stop_reason='finite_budget_or_generation_ceiling')
    except BaseException as exc:
        status.update(state='failed', error=repr(exc))
        raise
    finally:
        if vector is not None:
            vector.close()
        if evaluator is not None:
            evaluator.ledger.close()
        writer.flush()
        writer.close()
        status['elapsed_seconds'] = time.monotonic() - started
        status['source_unchanged'] = all(digest(SOURCE / n) == h for n, h in status['source_hashes'].items())
        status['initial_head_unchanged'] = digest(HEAD) == status['initial_head_sha256']
        status['frozen_network_after'] = network_digest(policy.network)
        if (not status['source_unchanged'] or not status['initial_head_unchanged']
                or status['frozen_network_before'] != status['frozen_network_after']):
            status.update(state='failed', error='Protected source/head changed.')
        checked_json(run / 'status.json', status)
        checked_json(run / 'summary.json', status)
    return run


def run_sampling_smoke(seconds=240, calibration_review=None):
    """Check frozen argmax parity and three real epsilon draws before training."""
    verified_stage_review(calibration_review, 'calibration')
    started, deadline = time.monotonic(), time.time() + seconds
    config, policy, run, status = setup('sampling-smoke', seconds, 1200)
    candidates = [{'individual': 0, 'seed': 11, 'genome': policy.source.copy(), 'draw': 0,
                   'evaluation_mode': 'argmax', 'evaluation_epsilon': 0.0, 'action_seed': 0}]
    candidates.extend({'individual': 0, 'seed': 11, 'genome': policy.source.copy(), 'draw': d,
                       'evaluation_mode': 'epsilon', 'evaluation_epsilon': .05,
                       'action_seed': draw_seed(0, d, namespace=11)} for d in range(3))
    checked_json(run / 'protocol.json', {'stage': 'sampling-smoke', 'epsilon': .05, 'draws': 3,
        'baseline_expected_fitness': 2075.6707117629994, 'baseline_expected_decisions': 269,
        'seconds_ceiling': seconds, 'same_canonical_state': True})
    writer = SummaryWriter(str(Path(config['paths']['tensorboard']) / 'robust-smoke' / run.name))
    vector = evaluator = None
    try:
        shape = (config['environment']['frame_stack'], config['environment']['image_size'], config['environment']['image_size'])
        vector = MeasuredSubprocVecEnv([LeanWorkerFactory(config, i, str(run)) for i in range(6)], shape,
                                       shared=False, receive_order='ordered', timeout=config['parallel']['worker_timeout'], lean=True)
        evaluator = GeneticEvaluator(config, run, policy, vector, started + seconds, status, writer, deadline)
        trials = evaluator.evaluate(candidates, 'sampling-smoke', 0, started + seconds)
        if len(trials) != len(candidates):
            status.update(state='incomplete', reason='Sampling smoke timed out before every episode completed.')
        else:
            records = [record for record, _ in trials]
            baseline = next(r for r in records if r['evaluation_mode'] == 'argmax')
            if (not math.isclose(baseline['fitness_world_units'], 2075.6707117629994, abs_tol=1e-5)
                    or baseline['decisions'] != 269 or baseline['outcome'] != 'death'):
                raise ValueError('Frozen argmax control failed parity with the prior validated champion.')
            scores = aggregate_draws([(r, None) for r in records if r['evaluation_mode'] == 'epsilon'], 1, 3)
            status.update(state='completed', baseline=baseline, epsilon_mean=float(scores[0]),
                          parity_passed=True)
    except BaseException as exc:
        status.update(state='failed', error=repr(exc))
        raise
    finally:
        if vector is not None:
            vector.close()
        if evaluator is not None:
            evaluator.ledger.close()
        writer.flush()
        writer.close()
        status['elapsed_seconds'] = time.monotonic() - started
        status['source_unchanged'] = all(digest(SOURCE / n) == h for n, h in status['source_hashes'].items())
        status['initial_head_unchanged'] = digest(HEAD) == status['initial_head_sha256']
        status['frozen_network_after'] = network_digest(policy.network)
        if (not status['source_unchanged'] or not status['initial_head_unchanged']
                or status['frozen_network_before'] != status['frozen_network_after']):
            status.update(state='failed', error='Protected source/head changed.')
        checked_json(run / 'status.json', status)
        checked_json(run / 'summary.json', status)
    return run


def run_confirmation(arm_b, arm_c, seconds=1800, arm_b_review=None, arm_c_review=None):
    """Evaluate top eight of each complete arm on disjoint action seeds."""
    started, deadline = time.monotonic(), time.time() + seconds
    selections = []
    protocols = []
    for label, parent in (('B', Path(arm_b)), ('C', Path(arm_c))):
        verified_stage_review(arm_b_review if label == 'B' else arm_c_review,
                              'arm-' + label.lower(), parent)
        summary = json.loads((parent / 'summary.json').read_text(encoding='utf-8'))
        if (summary['state'] != 'completed' or summary.get('error') or not summary['source_unchanged']
                or summary['protocol']['arm'] != label or not summary['generations_completed']):
            raise ValueError('Both arms require clean, reviewed complete generations.')
        protocols.append((summary['protocol']['mutation_std'], summary['protocol']['population_seed'],
                          summary['source_hashes'], summary['initial_genome_sha256']))
        generation = summary['last_complete_generation']
        meta = json.loads((parent / f'generation-{generation:04d}.json').read_text(encoding='utf-8'))
        with np.load(parent / f'generation-{generation:04d}.npz', allow_pickle=False) as data:
            population = data['genomes'].copy()
        for index in meta['top8']:
            genome = population[index]
            if genome_digest(genome) != meta['genome_hashes'][index]:
                raise ValueError('Frozen confirmation genome hash mismatch.')
            selections.append((label, generation, index, genome))
    if protocols[0] != protocols[1]:
        raise ValueError('Arms do not share mutation, population seed, source and initial genome.')
    config, policy, run, status = setup('confirmation', seconds, 1200)
    selections.append(('baseline', -1, -1, policy.source.copy()))
    checked_json(run / 'protocol.json', {'stage': 'confirmation', 'draws': 5, 'epsilon': .05,
        'draw_namespace': 99, 'training_draw_namespace': 11, 'greedy_diagnostics': True,
        'selection': 'top eight of each last complete generation plus initial baseline',
        'wii_state': 'same canonical seed-11 state', 'source_arms': [str(arm_b), str(arm_c)],
        'no_independent_winner_test': True, 'seconds_ceiling': seconds})
    writer = SummaryWriter(str(Path(config['paths']['tensorboard']) / 'robust-confirmation' / run.name))
    vector = evaluator = None
    try:
        shape = (config['environment']['frame_stack'], config['environment']['image_size'], config['environment']['image_size'])
        vector = MeasuredSubprocVecEnv([LeanWorkerFactory(config, i, str(run)) for i in range(6)], shape,
                                       shared=False, receive_order='ordered', timeout=config['parallel']['worker_timeout'], lean=True)
        evaluator = GeneticEvaluator(config, run, policy, vector, started + seconds, status, writer, deadline)
        candidates = [{'individual': i, 'source_arm': label, 'source_generation': generation,
                       'source_individual': index, 'seed': 11, 'genome': genome, 'draw': d,
                       'evaluation_mode': 'epsilon', 'evaluation_epsilon': .05,
                       'action_seed': draw_seed(0, d, namespace=99)}
                      for i, (label, generation, index, genome) in enumerate(selections) for d in range(5)]
        evaluated = evaluator.evaluate(candidates, 'confirmation', 0, started + seconds)
        if len(evaluated) != len(candidates):
            status.update(state='incomplete', reason='Held-out evaluation budget expired.')
        else:
            scores = aggregate_draws(evaluated, len(selections), 5)
            greedy = [{'individual': i, 'source_arm': label, 'source_generation': generation,
                       'source_individual': index, 'seed': 11, 'genome': genome, 'draw': 0,
                       'evaluation_mode': 'argmax', 'evaluation_epsilon': 0.0, 'action_seed': 0}
                      for i, (label, generation, index, genome) in enumerate(selections)]
            deterministic = evaluator.evaluate(greedy, 'confirmation_argmax', 0, started + seconds)
            if len(deterministic) != len(greedy):
                status.update(state='incomplete', reason='Greedy confirmation budget expired.')
            else:
                greedy_scores = aggregate_draws(deterministic, len(selections), 1)
                rows = []
                for i, (label, generation, index, genome) in enumerate(selections):
                    draws = sorted((r for r, _ in evaluated if r['individual'] == i), key=lambda r: r['draw'])
                    rows.append({'source_arm': label, 'source_generation': generation,
                                 'source_individual': index, 'genome_sha256': genome_digest(genome),
                                 'epsilon_mean': float(scores[i]), 'epsilon_sd': float(np.std(
                                     [r['fitness_world_units'] for r in draws], ddof=1)),
                                 'greedy_fitness': float(greedy_scores[i]),
                                 'draw_fitness': [r['fitness_world_units'] for r in draws],
                                 'ram_stars': sum(r['ram_star'] for r in draws)})
                status.update(state='completed', candidates=rows,
                              interpretation='Confirmation data select a candidate; a further independent test is required.')
    except BaseException as exc:
        status.update(state='failed', error=repr(exc))
        raise
    finally:
        if vector is not None:
            vector.close()
        if evaluator is not None:
            evaluator.ledger.close()
        writer.flush()
        writer.close()
        status['elapsed_seconds'] = time.monotonic() - started
        status['source_unchanged'] = all(digest(SOURCE / n) == h for n, h in status['source_hashes'].items())
        status['initial_head_unchanged'] = digest(HEAD) == status['initial_head_sha256']
        status['frozen_network_after'] = network_digest(policy.network)
        if (not status['source_unchanged'] or not status['initial_head_unchanged']
                or status['frozen_network_before'] != status['frozen_network_after']):
            status.update(state='failed', error='Protected source/head changed.')
        checked_json(run / 'status.json', status)
        checked_json(run / 'summary.json', status)
    return run
