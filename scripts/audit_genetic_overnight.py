"""Audit the stopped overnight GA without launching training or evaluation episodes."""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch

from smg_rl.checkpoints import digest
from smg_rl.diagnostics import atomic_json
from smg_rl.dqn_trial import alive, locate_run, read
from smg_rl.genetic import genome_digest, next_population, resume_population
from smg_rl.genetic_audit import audited_trajectory, counts

OUTPUT = Path('docs/assets/performance/genetic-overnight-result-20261027.json')
FIGURE = OUTPUT.with_suffix('.png')


def audit():
    state = read(Path('.local/genetic-overnight-state.json'))
    run = locate_run(state['genetic'])
    final = read(run / 'summary.json')
    if (state['phase'] != 'completed' or final['state'] != 'completed' or final['error']
            or final['stop_reason'] != 'scheduled_deadline' or alive(state['supervisor'])
            or alive(read(run / 'controller.json'))):
        raise ValueError('The overnight process did not stop cleanly at its scheduled deadline.')
    dolphins = [p.pid for p in psutil.process_iter(['name']) if (p.info['name'] or '').lower() == 'dolphin.exe']
    if dolphins:
        raise ValueError('Emulators are still running; inspect actual ownership before reporting closure.')
    source = Path(final['source'])
    if any(digest(source / name) != sha for name, sha in final['source_hashes'].items()):
        raise ValueError('The immutable visual source checkpoint changed.')
    if digest(run / 'champion.pt') != final['champion_sha256']:
        raise ValueError('Final genetic champion hash failed.')
    saved = torch.load(run / 'champion.pt', map_location='cpu', weights_only=True)
    original = torch.load(source / 'agent.pt', map_location='cpu', weights_only=True)
    if (saved['source_online'].keys() != original['online'].keys()
            or not all(torch.equal(v, original['online'][k]) for k, v in saved['source_online'].items())
            or genome_digest(saved['genome'].numpy()) != final['champion']['genome_sha256']):
        raise ValueError('Frozen visual tensors or champion genome differ from their declared identities.')
    trials = [json.loads(line) for line in (run / 'trials.jsonl').read_text().splitlines()]
    local = counts(trials)
    if any(local[k] != final[k] for k in ('real_decisions', 'completed_trials', 'partial_trials', 'full_start_stars')):
        raise ValueError('Local real-decision/attempt/star accounting failed.')
    if final['gradient_updates'] != 0 or sum(final['action_counts']) != final['real_decisions']:
        raise ValueError('Unexpected gradient updates or inconsistent applied-action accounting.')
    if any(t['start_kind'] != 'full_start' or t['phase'] != 'evolution' for t in trials):
        raise ValueError('Unexpected assisted or final-evaluation data in the overnight training ledger.')
    config = read(run / 'configuration.json')
    bounds = read(Path(config['paths']['calibration']))['variations']['sweet_sweet']
    latest_sample = 0.
    by_generation = defaultdict(dict)
    for trial in trials:
        rows, _ = audited_trajectory(trial, bounds, config['environment']['max_step_distance'])
        if rows:
            latest_sample = max(latest_sample, float(rows[-1]['time']))
        if trial['fitness_eligible']:
            group = by_generation[trial['generation']]
            if trial['individual'] in group:
                raise ValueError('Duplicate completed physical trial in one generation.')
            group[trial['individual']] = trial
    ancestor = Path(final['resume_genetic_run'])
    population, rng, offset, inherited, old = resume_population(ancestor, final['source_hashes'], 24, 4104)
    if offset != final['generation_offset'] or len(inherited) != final['inherited_completed_trials']:
        raise ValueError('Inherited ancestry count/offset failed.')
    trend = []
    for index in range(old['generations_completed']):
        with np.load(ancestor / f'generation-{index:04d}.npz', allow_pickle=False) as data:
            trend.append((index, float(data['fitness'].max()), float(data['fitness'].mean())))
    for index in range(offset, offset + final['generations_completed']):
        metadata = read(run / f'generation-{index:04d}.json')
        with np.load(run / f'generation-{index:04d}.npz', allow_pickle=False) as data:
            retained, fitness = data['genomes'].copy(), data['fitness'].copy()
        records = {**inherited, **by_generation[index]}
        if (len(records) != 24 or not metadata['review_passed'] or not np.array_equal(retained, population)
                or not np.isfinite(retained).all() or not np.isfinite(fitness).all()):
            raise ValueError('Generation ancestry/physical population review failed.')
        for i, genome in enumerate(retained):
            if genome_digest(genome) != records[i]['genome_sha256'] or fitness[i] != records[i]['fitness_world_units']:
                raise ValueError('Genome/fitness differs from its retained physical trial.')
        if metadata['rng_state_before_reproduction'] != rng.bit_generator.state:
            raise ValueError('Mutation/selection RNG continuity failed.')
        trend.append((index, float(fitness.max()), float(fitness.mean())))
        population = next_population(retained, fitness, rng, sigma=final['mutation_std'])
        inherited = {}
    for record in by_generation[offset + final['generations_completed']].values():
        if genome_digest(population[record['individual']]) != record['genome_sha256']:
            raise ValueError('Final partial generation has an unexpected genome.')
    prior_trials = [json.loads(line) for line in (ancestor / 'trials.jsonl').read_text().splitlines()]
    cumulative = counts(prior_trials + trials)
    evolution = counts([t for t in prior_trials + trials if t['phase'] == 'evolution'])
    best_new = max((t for t in trials if t['fitness_eligible']), key=lambda t: t['fitness_world_units'])
    best_partial = max((t for t in trials if not t['fitness_eligible']), key=lambda t: t['fitness_world_units'])
    if max(old['champion']['fitness_world_units'], best_new['fitness_world_units']) != final['champion']['fitness_world_units']:
        raise ValueError('Retained champion is not the best eligible inherited/new individual.')
    rows, advances = audited_trajectory(final['champion'], bounds, config['environment']['max_step_distance'])
    length = np.linalg.norm(np.asarray(bounds['goal'])[[0, 2]] - np.asarray(bounds['start'])[[0, 2]])
    fig, (top, plot) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    positions = np.array([[float(row[k]) for k in ('x', 'z')] for row in rows])
    top.plot(*positions.T, color='#2563eb', label='Retained best trajectory')
    peak = int(np.argmax(advances))
    top.scatter(*positions[peak], marker='D', s=55, color='#d97706', label='Peak grounded fitness')
    top.scatter(*positions[0], color='#059669', label='Canonical start')
    top.set(xlabel='World X', ylabel='World Z', title='Best trajectory (inherited from pilot)')
    top.set_aspect('equal', adjustable='datalim')
    series = np.asarray(trend)
    x = series[:, 0] + 1
    plot.plot(x, 100 * np.maximum.accumulate(series[:, 1]) / length, color='#2563eb', label='Best so far')
    plot.plot(x, 100 * series[:, 2] / length, color='#d97706', alpha=.7, label='Population mean')
    plot.axvline(offset + .5, color='#64748b', linestyle='--', label='Overnight continuation')
    plot.set(xlabel='Completed generation (1-based)', ylabel='Grounded geometric projection (%)',
             title='No new maximum during overnight evolution', ylim=(0, 10))
    for axis in (top, plot):
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle('Sweet Sweet genetic evolution: zero RAM-confirmed stars')
    fig.supxlabel('Projection is not native mission progress. Initial mapped opening is a separate metric.', fontsize=9)
    fig.savefig(FIGURE, dpi=150)
    plt.close(fig)
    names = read(source / 'manifest.json')['signature']['action_names']
    actions = {name: {'count': n, 'percent': 100 * n / final['real_decisions']}
               for name, n in zip(names, final['action_counts'])}
    deadline = datetime.fromisoformat(final['deadline_utc']).timestamp()
    iqn = state['iqn_pause_review']
    iqn_manifest = read(Path(iqn['checkpoint']) / 'manifest.json')
    if iqn_manifest['agent_sha256'] != digest(Path(iqn['checkpoint']) / 'agent.pt'):
        raise ValueError('Retained newer IQN checkpoint identity failed.')
    evidence = {'recorded_utc': datetime.now(timezone.utc).isoformat(), 'audit_passed': True, 'run': str(run),
        'stop_reason': final['stop_reason'], 'supervisor_completed_at': state['completed_at'],
        'last_emulator_sample_utc': datetime.fromtimestamp(latest_sample, timezone.utc).isoformat(),
        'last_sample_minus_deadline_seconds': latest_sample - deadline,
        'all_controllers_stopped': True, 'remaining_emulators': dolphins,
        'source_hashes_unchanged': True, 'frozen_source_tensors_equal': True,
        'champion_hash_verified': final['champion_sha256'], 'gradient_updates': final['gradient_updates'],
        'elapsed_seconds': final['elapsed_seconds'], 'local': local, 'cumulative': cumulative,
        'cumulative_evolution_only': evolution, 'new_generations': final['generations_completed'],
        'cumulative_complete_generations': old['generations_completed'] + final['generations_completed'],
        'inherited_completed_trials_not_charged_again': len(resume_population(ancestor, final['source_hashes'], 24, 4104)[3]),
        'all_local_trial_csv_fitness_recomputed': len(trials),
        'generation_lineage_verified': final['generations_completed'],
        'champion_is_inherited': final['champion_origin_run'] == str(ancestor),
        'inherited_champion': old['champion'], 'retained_champion': final['champion'],
        'best_new_complete_trial': best_new, 'best_unselected_partial_trial': best_partial,
        'deaths': final['deaths'], 'nonfatal_hp_lost': final['nonfatal_hp_lost'], 'action_occupancy': actions,
        'real_decisions_per_second': final['real_decisions'] / final['elapsed_seconds'],
        'new_frozen_confirmation_trials': 0, 'iqn_retained_real_decisions': iqn['real_decisions'],
        'interpretation': 'No improvement over the inherited pilot champion. Transferred action-head GA, not full-network evolution. Geometric projection and mapped opening are not native mission percentages; inherited fitness is reused data, not unseen testing. No new frozen policy evaluation was allocated or run.',
        'figure': str(FIGURE)}
    if not atomic_json(OUTPUT, evidence):
        raise OSError('Could not persist final overnight evidence.')
    print(json.dumps({k: evidence[k] for k in ('audit_passed', 'local', 'cumulative', 'cumulative_evolution_only',
        'elapsed_seconds', 'new_generations', 'cumulative_complete_generations', 'real_decisions_per_second',
        'last_sample_minus_deadline_seconds', 'action_occupancy')}))


if __name__ == '__main__':
    audit()
