"""Audit the disk-exhausted IQN allocation without retrying or loading its corrupt checkpoint."""

import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import psutil

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.dqn_trial import alive, audit_block, locate_run, read
from smg_rl.genetic_audit import audited_trajectory
from smg_rl.iqn_progress import scan_runs

OUTPUT = Path('docs/assets/performance/iqn-four-hour-result-20261027.json')
FIGURE = OUTPUT.with_suffix('.png')


def audit():
    state_path = Path('.local/iqn-four-hour-20261027-state.json')
    state = read(state_path)
    branch = state['games']['smg1']
    retained_source = read(Path('.local/genetic-visual-20261027-state.json'))['iqn_pause_review']
    if digest(Path(retained_source['checkpoint']) / 'agent.pt') != retained_source['agent_sha256']:
        raise ValueError('The initial retained IQN checkpoint changed.')
    if state['phase'] != 'needs_review' or alive(state['supervisor']):
        raise ValueError('Expected the stopped failed allocation; do not duplicate/retry it.')
    emulators = [p.pid for p in psutil.process_iter(['name']) if (p.info['name'] or '').lower() == 'dolphin.exe']
    if emulators:
        raise ValueError('Emulators remain; inspect ownership before reporting closure.')
    reviews, actions, acknowledged_attempts, physical_steps, hp_lost = [], Counter(), [], 0, 0
    acknowledged_real = 0
    for job in branch['jobs']:
        run = locate_run(job)
        if alive(read(run / 'controller.json')):
            raise ValueError('A controller remains active.')
        if job['phase'] == 'completed_reviewed':
            reviews.append(audit_block(run, load_config(job['config']), job['source_real'], job['source_attempts']))
        real = job['source_real']
        for line in (run / 'collector-steps.jsonl').read_text().splitlines():
            entry = json.loads(line)
            if entry['real_before_action'] != real or len(entry['actions']) != 6:
                raise ValueError('Acknowledged vector-step continuity failed.')
            real += 6
            actions.update(entry['actions'])
        acknowledged_real += real - job['source_real']
        acknowledged_attempts.extend(json.loads(line) for line in (run / 'attempts.jsonl').read_text().splitlines())
        for path in run.glob('worker-*/*/episode-*.csv'):
            with path.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            physical_steps += max(0, len(rows) - 1)
            hp_lost += sum(float(row['health_lost']) for row in rows[1:] if row['dead'] != 'True')
    last = branch['jobs'][-1]
    failed = locate_run(last)
    stale = read(failed / 'status.json')
    stderr = Path(last['stderr']).read_text()
    if ('No space left on device' not in stderr or not (failed / 'checkpoint-final/.incomplete').exists()
            or (failed / 'checkpoint-final/manifest.json').exists()):
        raise ValueError('Expected disk-exhaustion and an explicitly incomplete final checkpoint.')
    valid = reviews[-1]
    if valid['real_decisions'] != branch['real_decisions'] or valid['checkpoint'] != branch['checkpoint']:
        raise ValueError('The last resumable checkpoint does not match the supervisor ledger.')
    progress = read(Path('docs/assets/performance/iqn-four-hour-goal-progress-20261027.json'))
    config = load_config(branch['config'])
    bounds = read(Path(config['paths']['calibration']))['variations']['sweet_sweet']
    stable = scan_runs([locate_run(j) for j in branch['jobs'] if j['phase'] == 'completed_reviewed'],
                       bounds, config['environment']['max_step_distance'])
    best = progress['current_phase']['best']
    fig, (top, plot) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for record, title, color in [(progress['baseline_before_phase']['best'], 'Before allocation', '#d97706'),
                                  (best, 'Observed before disk failure', '#2563eb')]:
        rows, advances = audited_trajectory({**record, 'decisions': record['saved_decisions'],
            'fitness_world_units': record['grounded_fitness_world_units']}, bounds, config['environment']['max_step_distance'])
        positions = np.array([[float(row[k]) for k in ('x', 'z')] for row in rows])
        top.plot(*positions.T, label=title, color=color)
        top.scatter(*positions[int(np.argmax(advances))], color=color, marker='D', s=55)
        plot.plot(advances, color=color, label=f"{title}: {record['geometric_projection_percent']:.2f}%")
    top.set(xlabel='World X', ylabel='World Z', title='Recorded canonical trajectories')
    top.set_aspect('equal', adjustable='datalim')
    plot.set(xlabel='Decision step', ylabel='Grounded goal projection (%)', title='Observation gain, not a completed mission', ylim=(0, 10))
    for axis in (top, plot):
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle('IQN four-hour allocation interrupted by disk exhaustion — zero RAM stars')
    fig.supxlabel('Geometric projection is not native mission progress. Best new trace came from an uncheckpointed block.', fontsize=9)
    fig.savefig(FIGURE, dpi=150)
    plt.close(fig)
    retained = list(Path('.local/smg1-six-two-hour/parallel-profiles/RMGE01').glob('*/worker-*/StateSaves/*.retained-*'))
    reserved_bytes = sum(p.stat().st_size for p in retained)
    disk = shutil.disk_usage('.')
    disk_d = shutil.disk_usage('D:/')
    observed_total = branch['source_real'] + acknowledged_real
    last_attempts = valid['completed_attempts']
    counts = Counter(('canonical' if t['start_kind'] == 'full_start' else 'assisted') for t in acknowledged_attempts)
    stars = Counter(('canonical' if t['start_kind'] == 'full_start' else 'assisted') for t in acknowledged_attempts if t['ram_star'])
    attempted_seconds_lower_bound = sum(read(locate_run(j) / 'status.json')['elapsed_seconds'] for j in branch['jobs'])
    evidence = {'recorded_utc': datetime.now(timezone.utc).isoformat(), 'audit_passed': True,
        'training_completed_four_hours': False, 'failure': 'Disk exhaustion while staging a curriculum state; subsequent final checkpoint serialization also failed.',
        'root_cause': 'NativeDolphin.load_state retained the prior disposable slot9 on every reset, creating unbounded per-profile copies.',
        'all_controllers_stopped': True, 'remaining_emulators': emulators,
        'clean_accounted_seconds': branch['elapsed_training_seconds'],
        'attempted_accounted_seconds_latest_status_lower_bound': attempted_seconds_lower_bound,
        'remaining_allocated_seconds_upper_bound': state['training_seconds'] - attempted_seconds_lower_bound,
        'source_real_decisions': branch['source_real'], 'last_valid_checkpoint_real': valid['real_decisions'],
        'initial_source_checkpoint_hash_verified': True,
        'new_checkpointed_real_decisions': valid['real_decisions'] - branch['source_real'],
        'acknowledged_real_decisions_observed': observed_total, 'new_acknowledged_real_decisions': acknowledged_real,
        'uncheckpointed_acknowledged_decisions': observed_total - valid['real_decisions'],
        'saved_csv_decisions': physical_steps, 'saved_minus_acknowledged': physical_steps - acknowledged_real,
        'source_completed_attempts': branch['source_attempts'], 'new_acknowledged_completed_attempts': len(acknowledged_attempts),
        'observed_total_completed_attempts': branch['source_attempts'] + len(acknowledged_attempts),
        'last_valid_checkpoint_completed_attempts': last_attempts, 'new_attempts_by_start_kind': dict(counts),
        'acknowledged_ram_stars_by_start_kind': dict(stars),
        'acknowledged_deaths': sum(t['dead'] for t in acknowledged_attempts),
        'saved_csv_nonfatal_hp_lost': hp_lost, 'last_valid_checkpoint': valid['checkpoint'],
        'last_valid_checkpoint_agent_sha256': valid['agent_sha256'],
        'failed_run': str(failed), 'failed_status_is_stale_running': stale['state'] == 'running',
        'failed_status_last_real': stale['real_decisions'], 'failed_checkpoint_resumable': False,
        'failed_checkpoint_files': {p.name: p.stat().st_size for p in (failed / 'checkpoint-final').iterdir()},
        'progress_before': progress['baseline_before_phase']['best'], 'best_observed_new': best,
        'best_observed_new_is_from_failed_run': Path(best['trajectory']).is_relative_to(failed),
        'best_observed_within_clean_blocks': stable['best'],
        'retained_slot_copies': len(retained), 'retained_slot_bytes': reserved_bytes,
        'review_c_free_bytes': disk.free, 'review_d_free_bytes': disk_d.free,
        'files_deleted_or_relocated': False, 'automatic_retry_started': False,
        'actions_by_index': dict(actions), 'completed_block_audits': len(reviews),
        'interpretation': 'Only749006 real decisions are resumable. The larger846674 acknowledged total and6.5860% geometric maximum describe actual observations, not a saved final policy. Projection and opening fraction are not native mission percentage. No new frozen evaluation was allocated.',
        'figure': str(FIGURE)}
    if not atomic_json(OUTPUT, evidence):
        raise OSError('Could not persist the failure audit.')
    state['failure_review'] = {k: evidence[k] for k in ('failure', 'acknowledged_real_decisions_observed',
        'last_valid_checkpoint_real', 'failed_checkpoint_resumable', 'uncheckpointed_acknowledged_decisions',
        'attempted_accounted_seconds_latest_status_lower_bound', 'remaining_allocated_seconds_upper_bound')}
    last['phase'] = 'failed_uncheckpointed_reviewed'
    if not atomic_json(state_path, state):
        raise OSError('Could not persist the reviewed failed-phase label.')
    print(json.dumps({k: evidence[k] for k in ('audit_passed', 'new_acknowledged_real_decisions',
        'acknowledged_real_decisions_observed', 'uncheckpointed_acknowledged_decisions',
        'new_acknowledged_completed_attempts', 'observed_total_completed_attempts',
        'saved_minus_acknowledged', 'attempted_accounted_seconds_latest_status_lower_bound',
        'acknowledged_deaths', 'saved_csv_nonfatal_hp_lost')}))


if __name__ == '__main__':
    audit()
