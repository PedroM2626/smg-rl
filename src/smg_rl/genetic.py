"""Finite visual genetic search of an IQN policy's action head; no gradient updates."""

import hashlib
import json
import math
import os
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.utils.tensorboard import SummaryWriter

from smg_rl.checkpoints import digest
from smg_rl.diagnostics import atomic_json
from smg_rl.dqn import DistributionalAgent
from smg_rl.lean_worker import LeanWorkerFactory
from smg_rl.parallel import runtime_directory
from smg_rl.plotting import plot_trajectory
from smg_rl.replay import replay_signature
from smg_rl.resource_memory import memory_safe, memory_snapshot, require_memory_headroom
from smg_rl.vector import MeasuredSubprocVecEnv


class GroundedAdvance:
    """High-water X/Z projection toward the calibrated goal; never native progress."""

    def __init__(self, start, goal, max_step):
        self.start = np.asarray(start, float)[[0, 2]]
        delta = np.asarray(goal, float)[[0, 2]] - self.start
        self.length = float(np.linalg.norm(delta))
        if not math.isfinite(self.length) or self.length <= 0 or max_step <= 0:
            raise ValueError('Fitness needs distinct finite start/goal positions and a positive jump bound.')
        self.axis, self.last, self.max_step = delta / self.length, np.asarray(start, float), max_step
        self.best, self.invalid, self.path_xz = 0., False, 0.

    def update(self, info):
        point = np.asarray(info['position'], float)
        if point.shape != (3,) or not np.isfinite(point).all() or np.linalg.norm(point - self.last) > self.max_step:
            self.invalid = True
            return self.best
        self.path_xz += float(np.linalg.norm((point - self.last)[[0, 2]]))
        self.last = point
        if not self.invalid and info.get('grounded') is True and not info.get('dead'):
            self.best = max(self.best, float(np.clip((point[[0, 2]] - self.start) @ self.axis, 0, self.length)))
        return self.best


class GeneticHead:
    """Frozen deterministic IQN features, with independently evolved linear action logits."""

    def __init__(self, agent):
        self.network = agent.online.eval().requires_grad_(False)
        self.device, self.quantiles = agent.device, agent.settings['quantiles']
        if self.network.algorithm != 'iqn':
            raise ValueError('The transferred visual head currently requires IQN.')
        head = self.network.advantage[-1]
        self.actions, self.hidden = head.weight_mu.shape
        self.source = torch.cat((head.weight_mu.detach().cpu().flatten(), head.bias_mu.detach().cpu())).numpy().copy()

    @torch.no_grad()
    def logits(self, observation, genomes):
        images = torch.as_tensor(observation['image'], device=self.device)
        scenario = torch.as_tensor(observation['scenario'], device=self.device)
        n = self.network
        features = n.encode(n.conv(images.float() / 255.))
        taus = ((torch.arange(self.quantiles, device=self.device) + .5) / self.quantiles)
        embedding = n.quantile_embedding(torch.cos(taus[:, None] * n.frequencies))
        encoded = features[:, None] * embedding[None]
        encoded = torch.cat((encoded, scenario[:, None].expand(-1, self.quantiles, -1)), -1)
        features = n.advantage[:-1](encoded).mean(1)
        genes = torch.as_tensor(np.stack(genomes), dtype=features.dtype, device=self.device)
        weights = genes[:, :self.actions * self.hidden].reshape(-1, self.actions, self.hidden)
        return (weights @ features[:, :, None]).squeeze(-1) + genes[:, -self.actions:]


def next_population(population, fitness, rng, elite=4, sigma=.02):
    population, fitness = np.asarray(population), np.asarray(fitness)
    if (population.ndim != 2 or fitness.shape != (len(population),) or not np.isfinite(population).all()
            or not np.isfinite(fitness).all() or not 1 <= elite < len(population) or not math.isfinite(sigma) or sigma <= 0):
        raise ValueError('Only a completely evaluated finite population can reproduce.')
    order = np.argsort(-fitness, kind='stable')
    result = [population[i].copy() for i in order[:elite]]
    while len(result) < len(population):
        parents = [max(rng.integers(len(population), size=3), key=lambda i: fitness[i]) for _ in range(2)]
        child = np.where(rng.random(population.shape[1]) < .5, population[parents[0]], population[parents[1]])
        child = np.clip(child + rng.normal(0, sigma, child.shape), -5, 5).astype(np.float32)
        result.append(child)
    return np.stack(result)


def genome_digest(genome):
    return hashlib.sha256(np.asarray(genome, np.float32).tobytes()).hexdigest()


def checked_json(path, value):
    if not atomic_json(path, value):
        raise OSError(f'Could not publish genetic evidence: {path}')


class GeneticEvaluator:
    def __init__(self, config, run, policy, vector, stop_at, status, writer):
        self.config, self.run, self.policy, self.vector = config, run, policy, vector
        self.stop_at, self.status, self.writer = stop_at, status, writer
        calibration = json.loads(Path(config['paths']['calibration']).read_text())
        self.bounds = calibration['variations'][config['variations'][0]['name']]
        self.ledger = (run / 'trials.jsonl').open('x', encoding='utf-8')

    def stopped(self):
        return (self.run / 'STOP').exists() or time.monotonic() >= self.stop_at

    def evaluate(self, candidates, phase, generation, end_at):
        """Keep at most six emulators occupied; never step an idle worker."""
        active, completed, pending = {}, [], iter(candidates)

        def assign(worker):
            if self.stopped() or time.monotonic() >= end_at:
                return
            candidate = next(pending, None)
            if candidate is None:
                return
            remote = self.vector.remotes[worker]
            remote.send(('reset', (candidate['seed'], {'full_start': True})))
            observation, reset = remote.recv()
            if reset.get('start_kind') != 'full_start':
                raise ValueError('Genetic fitness requires the same canonical start for every individual.')
            active[worker] = {'candidate': candidate, 'observation': observation, 'steps': 0, 'opening': 0., 'return': 0.,
                              'advance': GroundedAdvance(self.bounds['start'], self.bounds['goal'],
                                  self.config['environment']['max_step_distance'])}

        def finish(worker, outcome, info, eligible):
            item = active.pop(worker)
            advance = item['advance']
            record = {k: v for k, v in item['candidate'].items() if k != 'genome'}
            record.update(phase=phase, generation=generation, worker=worker, decisions=item['steps'],
                genome_sha256=genome_digest(item['candidate']['genome']), fitness_world_units=advance.best,
                geometric_projection_percent=100 * advance.best / advance.length,
                sampled_path_distance_xz=advance.path_xz, environment_return=item['return'],
                mapped_opening_percent=100 * item['opening'], fitness_eligible=eligible and not advance.invalid,
                invalid_position=advance.invalid, outcome=outcome, ram_star=bool(info.get('complete')),
                dead=bool(info.get('dead')), health=info.get('health'), start_kind='full_start',
                trajectory=str(self.run / f'worker-{worker:02d}' / self.config['variations'][0]['name'] /
                    f"episode-{info['worker_episode']:06d}.csv") if 'worker_episode' in info else None)
            self.ledger.write(json.dumps(record) + '\n')
            self.ledger.flush()
            if record['fitness_eligible']:
                completed.append((record, item['candidate']['genome'].copy()))
                self.status['completed_trials'] += 1
                self.status['full_start_stars'] += int(record['ram_star'])
                if phase == 'evolution' and (not self.status.get('champion') or
                        record['fitness_world_units'] > self.status['champion']['fitness_world_units']):
                    self.status['champion'] = record
                    torch.save({'source_online': self.policy.network.state_dict(),
                                'genome': torch.from_numpy(item['candidate']['genome'].copy())}, self.run / 'champion.tmp')
                    os.replace(self.run / 'champion.tmp', self.run / 'champion.pt')
                self.writer.add_scalar(f'{phase}/fitness_world_units', advance.best, self.status['completed_trials'])
                self.writer.add_scalar(f'{phase}/geometric_projection_percent', record['geometric_projection_percent'],
                                       self.status['completed_trials'])
                for key in ('sampled_path_distance_xz', 'environment_return', 'mapped_opening_percent', 'decisions'):
                    self.writer.add_scalar(f'{phase}/{key}', record[key], self.status['completed_trials'])
                self.writer.add_scalar('accounting/real_decisions', self.status['real_decisions'], self.status['completed_trials'])
            else:
                self.status['partial_trials'] += 1
            return record

        try:
            for worker in range(self.config['parallel']['workers']):
                assign(worker)
            while active and not self.stopped() and time.monotonic() < end_at:
                indices = list(active)
                observation = {k: np.stack([active[i]['observation'][k] for i in indices]) for k in ('image', 'scenario')}
                logits = self.policy.logits(observation, [active[i]['candidate']['genome'] for i in indices])
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Nonfinite genetic action logits.')
                actions = logits.argmax(-1).cpu().numpy()
                for worker, action in zip(indices, actions):
                    self.vector.remotes[worker].send(('step', int(action)))
                for worker, action in zip(indices, actions):
                    observation, reward, done, info, _ = self.vector.remotes[worker].recv()
                    item = active[worker]
                    item['observation'], item['info'] = observation, info
                    item['steps'] += 1
                    item['return'] += float(reward)
                    item['advance'].update(info)
                    item['opening'] = max(item['opening'], float(info.get('start_relative_route_progress') or 0))
                    self.status['real_decisions'] += 1
                    self.status['action_counts'][int(action)] += 1
                    self.status['nonfatal_hp_lost'] += int(info.get('health_lost', 0)) if not info.get('dead') else 0
                    if done:
                        self.status['deaths'] += int(bool(info.get('dead')))
                        outcome = 'star' if info.get('complete') else 'death' if info.get('dead') else 'timeout'
                        record = finish(worker, outcome, info, True)
                        if record['invalid_position']:
                            raise ValueError('Invalid trajectory position; no successor generation allowed.')
                        assign(worker)
                if self.status['real_decisions'] - self.status.get('last_publish_decision', -1000) >= 192:
                    if not memory_safe(memory_snapshot()):
                        raise MemoryError('Memory pressure stopped genetic evaluation; no automatic retry.')
                    self.status.update(last_publish_decision=self.status['real_decisions'],
                                       updated_at=datetime.now(timezone.utc).isoformat())
                    checked_json(self.run / 'status.json', self.status)
            return completed
        finally:
            for worker in list(active):
                item = active[worker]
                finish(worker, 'partial_budget_cut', item.get('info', {}), False)
            checked_json(self.run / 'status.json', self.status)


def train_genetic(config, source, seconds=3600, population_size=24, max_decisions=1200):
    if (not math.isfinite(seconds) or not 60 <= seconds <= 3600 or population_size < 6
            or population_size > 96 or max_decisions < 1 or max_decisions > 3000
            or config['parallel']['workers'] != 6 or len(config['variations']) != 1):
        raise ValueError('This pilot requires six workers, one mission and finite bounded dimensions/time.')
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('Genetic training requires exclusive emulator ownership; pause and audit IQN first.')
    require_memory_headroom()
    started = time.monotonic()
    config = deepcopy(config)
    signature = replay_signature(config)
    source = Path(source)
    source_hashes = {name: digest(source / name) for name in ('agent.pt', 'replay.npz', 'manifest.json')}
    torch.set_num_threads(config['training'].get('torch_threads', 1))
    agent = DistributionalAgent.load_policy(source, signature, config['training']['device'])
    policy = GeneticHead(agent)
    rng = np.random.default_rng(142)
    population = np.stack([policy.source + rng.normal(0, .02, policy.source.shape).astype(np.float32)
                           for _ in range(population_size)])
    population[0] = policy.source.copy()
    config['curriculum']['enabled'] = False
    config['parallel']['plot_every'] = 1000000
    frames = max_decisions * config['parallel']['frame_skip']
    config['regional_reward'].update(max_idle_frames=frames, max_episode_frames=frames)
    config['environment'].update(max_steps=max_decisions, idle_limit=max_decisions)
    run = Path(config['paths']['runs']) / ('genetic-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    run.mkdir(parents=True, exist_ok=False)
    config['paths']['user'] = str(runtime_directory(config, run) / 'template')
    checked_json(run / 'configuration.json', config)
    checked_json(run / 'controller.json', {'pid': os.getpid(), 'created_time': psutil.Process().create_time(),
                 'source': str(source), 'seconds': seconds, 'workers': 6})
    print(json.dumps({'run': str(run)}), flush=True)
    status = {'state': 'running', 'method': 'genetic_action_head_with_frozen_transferred_impala_iqn_features',
              'source': str(source), 'source_hashes': source_hashes, 'source_real_decisions': agent.real_decisions,
              'source_optimizer_updates': agent.updates, 'gradient_updates': 0, 'real_decisions': 0,
              'generations_completed': 0, 'completed_trials': 0, 'partial_trials': 0, 'full_start_stars': 0,
              'deaths': 0, 'nonfatal_hp_lost': 0, 'action_counts': [0] * 8, 'seconds': seconds,
              'population': population_size, 'genome_parameters': len(policy.source), 'mutation_std': .02,
              'fitness': 'Maximum grounded X/Z projection toward calibrated goal in world units; not native progress.',
              'curriculum': False, 'fitness_seed': 11, 'confirmation_seeds': [23, 47], 'error': None}
    checked_json(run / 'status.json', status)
    writer = SummaryWriter(str(Path(config['paths']['tensorboard']) / 'genetic' / run.name))
    vector, evaluator = None, None
    try:
        shape = (config['environment']['frame_stack'], config['environment']['image_size'], config['environment']['image_size'])
        vector = MeasuredSubprocVecEnv([LeanWorkerFactory(config, i, str(run)) for i in range(6)], shape,
                  shared=False, receive_order='ordered', timeout=config['parallel']['worker_timeout'], lean=True)
        evaluator = GeneticEvaluator(config, run, policy, vector, started + seconds, status, writer)
        reserve = min(300., seconds * .15)
        for generation in range(1000):
            if evaluator.stopped() or time.monotonic() >= started + seconds - reserve:
                break
            candidates = [{'individual': i, 'seed': 11, 'genome': genome} for i, genome in enumerate(population)]
            trials = evaluator.evaluate(candidates, 'evolution', generation, started + seconds - reserve)
            if len(trials) != population_size:
                break  # Never reproduce from incomplete or interrupted candidate fitness.
            by_id = {r['individual']: r for r, _ in trials}
            fitness = [by_id[i]['fitness_world_units'] for i in range(population_size)]
            np.savez_compressed(run / f'generation-{generation:04d}.npz', genomes=population, fitness=fitness)
            checked_json(run / f'generation-{generation:04d}.json', {
                'review_passed': True, 'completed_trials': len(trials), 'best_fitness_world_units': max(fitness),
                'rng_state_before_reproduction': rng.bit_generator.state,
                'trial_genome_hashes': {str(r['individual']): r['genome_sha256'] for r, _ in trials}})
            status['generations_completed'] += 1
            writer.add_scalar('generation/best_fitness_world_units', max(fitness), generation)
            writer.add_scalar('generation/mean_fitness_world_units', float(np.mean(fitness)), generation)
            population = next_population(population, fitness, rng)
            checked_json(run / 'status.json', status)
        if status.get('champion') and not (run / 'STOP').exists():
            checkpoint = torch.load(run / 'champion.pt', map_location='cpu', weights_only=True)
            champion = checkpoint['genome'].numpy()
            candidates = [{'individual': name, 'seed': seed, 'genome': genome}
                          for name, genome in [('source_baseline', policy.source), ('genetic_champion', champion)]
                          for seed in (23, 47)]
            tests = evaluator.evaluate(candidates, 'frozen_confirmation', status['generations_completed'], started + seconds)
            status['confirmation_results'] = [record for record, _ in tests]
        status['state'] = 'stopped' if (run / 'STOP').exists() else 'completed'
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
        final_hashes = {name: digest(source / name) for name in source_hashes}
        status['source_unchanged'] = final_hashes == source_hashes
        if not status['source_unchanged']:
            status.update(state='failed', error='Source checkpoint changed during isolated genetic search.')
        status['elapsed_seconds'] = time.monotonic() - started
        if status.get('champion'):
            trajectory = status['champion'].get('trajectory')
            if trajectory and Path(trajectory).exists():
                plot_trajectory(trajectory, run / 'best-trajectory.png')
            status['champion_sha256'] = digest(run / 'champion.pt')
        checked_json(run / 'status.json', status)
        checked_json(run / 'summary.json', status)
    return run
