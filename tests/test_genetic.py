import numpy as np
import pytest
import torch

from smg_rl.config import load_config
from smg_rl.dqn import DistributionalAgent
from smg_rl.genetic import GeneticHead, GroundedAdvance, genome_digest, next_population
from smg_rl.replay import replay_signature


def test_source_genome_preserves_iqn_greedy_actions_without_gradients():
    config = load_config('config/iqn-smg1.yaml')
    config['environment']['image_size'] = 40
    config['dqn'].update(channels=2, hidden=16, quantiles=4, encoder='impala')
    torch.set_num_threads(1)
    agent = DistributionalAgent(replay_signature(config), config['dqn'])
    head = GeneticHead(agent)
    observation = {'image': np.random.default_rng(11).integers(0, 256, (3, 4, 40, 40), dtype=np.uint8),
                   'scenario': np.ones((3, 1), np.float32)}
    expected, _ = agent.act(observation, evaluate=True)
    logits = head.logits(observation, [head.source] * 3)
    assert np.array_equal(logits.argmax(-1).numpy(), expected)
    assert not logits.requires_grad and all(p.grad is None and not p.requires_grad for p in agent.online.parameters())
    changed = head.source.copy()
    changed[-8:] = 0
    changed[-1] = 10000
    assert head.logits(observation, [changed] * 3).argmax(-1).tolist() == [7, 7, 7]


def test_fitness_credits_only_new_grounded_goal_projection_and_excludes_falls():
    score = GroundedAdvance((0, 0, 0), (100, 10, 0), 50)
    assert score.update({'position': [20, 0, 0], 'grounded': True}) == 20
    assert score.update({'position': [40, -10, 0], 'grounded': False}) == 20
    assert score.update({'position': [30, 0, 0], 'grounded': True, 'dead': True}) == 20
    assert score.update({'position': [10, 0, 0], 'grounded': True}) == 20
    assert score.update({'position': [45, 0, 0], 'grounded': True}) == 45
    assert score.update({'position': [90, 0, 200], 'grounded': True}) == 45
    assert score.invalid
    assert score.update({'position': [80, 0, 0], 'grounded': True}) == 45


def test_distance_fitness_evaluator_uses_nearest_point_not_axis_projection(tmp_path):
    import json
    import time
    from types import SimpleNamespace

    from smg_rl.genetic import GeneticEvaluator

    class Remote:
        def send(self, command):
            observation = {'image': np.zeros((4, 40, 40), np.uint8), 'scenario': np.ones(1, np.float32)}
            if command[0] == 'reset':
                self.reply = observation, {'start_kind': 'full_start', 'position': [0, 0, 0]}
            else:
                info = {'position': [90, 0, 100], 'grounded': True, 'complete': False,
                        'dead': False, 'worker_episode': 1}
                self.reply = observation, 999999., True, info, {}

        def recv(self):
            return self.reply

    calibration = tmp_path / 'calibration.json'
    calibration.write_text(json.dumps({'variations': {'sweet': {'start': [0, 0, 0], 'goal': [100, 0, 0]}}}))
    config = {'paths': {'calibration': str(calibration)}, 'variations': [{'name': 'sweet'}],
              'parallel': {'workers': 1}, 'environment': {'max_step_distance': 500}}
    policy = SimpleNamespace(logits=lambda observations, genomes: torch.zeros((len(genomes), 8)))
    vector = SimpleNamespace(remotes=[Remote()])
    writer = SimpleNamespace(add_scalar=lambda *args: None)
    status = {'fitness_mode': 'goal-distance-xyz-v1', 'real_decisions': 0, 'completed_trials': 0,
              'partial_trials': 0, 'full_start_stars': 0, 'deaths': 0, 'nonfatal_hp_lost': 0,
              'action_counts': [0] * 8, 'last_publish_decision': 0}
    policy.network = SimpleNamespace(state_dict=lambda: {})
    evaluator = GeneticEvaluator(config, tmp_path, policy, vector, time.monotonic() + 30, status, writer)
    records = evaluator.evaluate([{'individual': 0, 'seed': 11, 'genome': np.zeros(8, np.float32)}],
                                 'evolution', 0, time.monotonic() + 30)
    evaluator.ledger.close()
    record = records[0][0]
    assert record['fitness_world_units'] == 0
    assert record['geometric_projection_percent'] == pytest.approx(90)
    assert record['mapped_opening_percent'] is None
    assert record['fitness_mode'] == 'goal-distance-xyz-v1'


def test_distance_fitness_cannot_reuse_old_population():
    from smg_rl.genetic import train_genetic

    config = {'parallel': {'workers': 6}, 'variations': [{}]}
    with pytest.raises(ValueError, match='historical fitness cannot be reused'):
        train_genetic(config, 'unused', seconds=60, fitness_mode='goal-distance-xyz-v1', resume_run='old')


def test_distance_three_hour_dispatch_rejects_existing_identity(tmp_path, monkeypatch):
    from scripts import launch_goal_distance_three_hours as launcher

    state = tmp_path / 'existing.json'
    state.write_text('{}')
    monkeypatch.setattr(launcher, 'STATE', state)
    with pytest.raises(ValueError, match='refusing to start another job'):
        launcher.launch()


def test_population_seed_requires_a_bounded_integer():
    from smg_rl.genetic import train_genetic

    config = {'parallel': {'workers': 6}, 'variations': [{}]}
    for seed in (-1, 2**32, True):
        with pytest.raises(ValueError, match='finite bounded'):
            train_genetic(config, 'unused', seconds=60, population_seed=seed)


def test_elitism_preserves_best_genomes_and_mutations_are_reproducible_and_independent():
    population = np.arange(240, dtype=np.float32).reshape(24, 10) / 100
    original = population.copy()
    fitness = np.arange(24)
    children = next_population(population, fitness, np.random.default_rng(11))
    assert np.array_equal(children[:4], original[[23, 22, 21, 20]])
    assert np.array_equal(population, original)
    assert np.array_equal(children, next_population(population, fitness, np.random.default_rng(11)))
    assert len({genome_digest(row) for row in children}) == 24
    with pytest.raises(ValueError, match='completely evaluated'):
        next_population(population, fitness[:-1], np.random.default_rng(11))
    with pytest.raises(ValueError):
        next_population(population, np.full(24, np.nan), np.random.default_rng(11))


def test_parallel_fitness_assigns_every_candidate_once_without_using_environment_rewards(tmp_path, monkeypatch):
    import json
    import time
    from types import SimpleNamespace

    from smg_rl import genetic

    class Remote:
        def __init__(self):
            self.episode = 0
            self.seeds = []

        def send(self, command):
            name, value = command
            observation = {'image': np.zeros((4, 40, 40), np.uint8), 'scenario': np.ones(1, np.float32)}
            if name == 'reset':
                self.episode += 1
                self.steps = 0
                self.seeds.append(value)
                self.reply = observation, {'start_kind': 'full_start'}
            else:
                self.steps += 1
                info = {'position': [self.steps * 10, 0, 0], 'grounded': True,
                        'start_relative_route_progress': .1, 'worker_episode': self.episode,
                        'complete': False, 'dead': self.steps == 2}
                self.reply = observation, 999999., self.steps == 2, info, {}  # Reward cannot affect fitness.

        def recv(self):
            return self.reply

    calibration = tmp_path / 'calibration.json'
    calibration.write_text(json.dumps({'variations': {'sweet': {'start': [0, 0, 0], 'goal': [100, 0, 0]}}}))
    config = {'paths': {'calibration': str(calibration)}, 'variations': [{'name': 'sweet'}],
              'parallel': {'workers': 6}, 'environment': {'max_step_distance': 50}}
    policy = SimpleNamespace(logits=lambda observation, genomes: torch.zeros((len(genomes), 8)),
                             network=torch.nn.Linear(1, 1))
    remotes = [Remote() for _ in range(6)]
    writer = SimpleNamespace(add_scalar=lambda *args: None)
    status = dict(real_decisions=0, action_counts=[0] * 8, nonfatal_hp_lost=0, deaths=0,
                  completed_trials=0, partial_trials=0, full_start_stars=0)
    monkeypatch.setattr(genetic, 'memory_safe', lambda _: True)
    evaluator = genetic.GeneticEvaluator(config, tmp_path, policy, SimpleNamespace(remotes=remotes),
                                         time.monotonic() + 60, status, writer)
    candidates = [{'individual': i, 'seed': 11, 'genome': np.zeros(3, np.float32)} for i in range(24)]
    results = evaluator.evaluate(candidates, 'evolution', 0, time.monotonic() + 60)
    evaluator.ledger.close()
    assert sorted(r['individual'] for r, _ in results) == list(range(24))
    assert all(r['fitness_world_units'] == 10 and not r['ram_star'] for r, _ in results)
    assert status['real_decisions'] == 48 and status['completed_trials'] == 24
    assert status['partial_trials'] == 0
    assert all(seed == (11, {'full_start': True}) for remote in remotes for seed in remote.seeds)
    assert len((tmp_path / 'trials.jsonl').read_text().splitlines()) == 24


def test_ga_profile_loads_the_exact_source_policy_but_uses_isolated_output_paths():
    iqn = load_config('config/smg1-six-two-hour.yaml')
    ga = load_config('config/genetic-smg1.yaml')
    assert replay_signature(ga) == replay_signature(iqn)
    assert ga['dqn'] == iqn['dqn']
    assert ga['paths']['runs'] != iqn['paths']['runs']
    assert ga['parallel']['workers'] == 6


def test_existing_genetic_job_cannot_pause_iqn_or_launch_twice(tmp_path, monkeypatch):
    from scripts import run_genetic_hour

    state = tmp_path / 'existing.json'
    state.write_text('{}')
    monkeypatch.setattr(run_genetic_hour, 'STATE', state)
    with pytest.raises(ValueError, match='no duplicate'):
        run_genetic_hour.supervise()


def test_result_audit_recomputes_grounded_fitness_and_rejects_inflated_claims(tmp_path):
    import csv

    from scripts.audit_genetic_result import audited_trajectory

    path = tmp_path / 'fixture.csv'
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['x', 'y', 'z', 'grounded', 'dead', 'complete'])
        writer.writerows([[0, 0, 0, True, False, False], [20, 0, 0, True, False, False],
                          [40, 0, 0, False, False, False], [50, 0, 0, True, True, False]])
    record = {'trajectory': str(path), 'decisions': 3, 'fitness_world_units': 20., 'ram_star': False}
    bounds = {'start': [0, 0, 0], 'goal': [100, 0, 0]}
    _, advances = audited_trajectory(record, bounds, 50)
    assert advances == [0, 20, 20, 20]
    with pytest.raises(ValueError, match='fitness differs'):
        audited_trajectory({**record, 'fitness_world_units': 50}, bounds, 50)
    with pytest.raises(ValueError, match='completion CSV'):
        audited_trajectory({**record, 'ram_star': True}, bounds, 50)


def test_result_audit_refuses_missing_nonempty_trajectories():
    from scripts.audit_genetic_result import audited_trajectory

    with pytest.raises(ValueError, match='lacks a trajectory'):
        audited_trajectory({'trajectory': None, 'decisions': 1}, {}, 50)


def test_extended_genetic_budget_requires_a_future_timezone_qualified_deadline(monkeypatch):
    from smg_rl import genetic

    monkeypatch.setattr(genetic.time, 'time', lambda: 1000.)
    assert genetic.collection_deadline(43200, '1970-01-01T01:00:00+00:00') == 3600
    with pytest.raises(ValueError, match='absolute deadline'):
        genetic.collection_deadline(7200)
    with pytest.raises(ValueError, match='timezone'):
        genetic.collection_deadline(7200, '2026-10-27T07:00:00')
    with pytest.raises(ValueError, match='already passed'):
        genetic.collection_deadline(7200, '1970-01-01T00:00:00+00:00')
    with pytest.raises(ValueError, match='12 hours'):
        genetic.collection_deadline(50000, '2026-10-27T07:00:00-03:00')


def test_absolute_genetic_deadline_overrides_remaining_monotonic_budget(tmp_path, monkeypatch):
    from smg_rl import genetic

    evaluator = object.__new__(genetic.GeneticEvaluator)
    evaluator.run, evaluator.stop_at, evaluator.wall_deadline = tmp_path, float('inf'), 1000.
    monkeypatch.setattr(genetic.time, 'time', lambda: 1000.)
    assert evaluator.stopped()


def test_resume_restores_population_rng_and_only_completed_pending_trials(tmp_path):
    import json

    from smg_rl.checkpoints import digest
    from smg_rl.genetic import resume_population

    population = np.arange(240, dtype=np.float32).reshape(24, 10) / 100
    fitness = np.arange(24, dtype=np.float64)
    rng = np.random.default_rng(142)
    before = rng.bit_generator.state
    pending_population = next_population(population, fitness, rng)
    np.savez_compressed(tmp_path / 'generation-0000.npz', genomes=population, fitness=fitness)
    meta = {'review_passed': True, 'rng_state_before_reproduction': before,
            'trial_genome_hashes': {str(i): genome_digest(g) for i, g in enumerate(population)}}
    (tmp_path / 'generation-0000.json').write_text(json.dumps(meta))
    (tmp_path / 'champion.pt').write_bytes(b'fixture: not a trained checkpoint')
    source_hashes = {'fixture': 'source'}
    final = {'state': 'completed', 'error': None, 'source_hashes': source_hashes, 'population': 24,
             'genome_parameters': 10, 'generations_completed': 1, 'mutation_std': .02,
             'champion_sha256': digest(tmp_path / 'champion.pt')}
    (tmp_path / 'summary.json').write_text(json.dumps(final))
    trials = [{'phase': 'evolution', 'generation': 0, 'individual': i, 'fitness_eligible': True,
               'fitness_world_units': float(fitness[i])} for i in range(24)]
    trials += [{'phase': 'evolution', 'generation': 1, 'individual': i, 'fitness_eligible': i == 3,
                'genome_sha256': genome_digest(pending_population[i])} for i in (3, 4)]
    (tmp_path / 'trials.jsonl').write_text('\n'.join(json.dumps(t) for t in trials))
    restored, restored_rng, offset, inherited, _ = resume_population(tmp_path, source_hashes, 24, 10)
    assert np.array_equal(restored, pending_population)
    assert restored_rng.bit_generator.state == rng.bit_generator.state
    assert offset == 1 and list(inherited) == [3]
    with pytest.raises(ValueError, match='identity'):
        resume_population(tmp_path, {'different': 'source'}, 24, 10)


def test_report_counts_keep_partial_decisions_but_exclude_inherited_fitness():
    from smg_rl.genetic_audit import counts

    ledger = [{'decisions': 10, 'fitness_eligible': True, 'ram_star': False},
              {'decisions': 4, 'fitness_eligible': False, 'ram_star': False}]
    assert counts(ledger) == {'real_decisions': 14, 'completed_trials': 1, 'partial_trials': 1,
                              'partial_decisions': 4, 'full_start_stars': 0}
    with pytest.raises(ValueError, match='Inherited fitness'):
        counts(ledger + [{**ledger[0], 'inherited_from_run': 'fixture'}])
