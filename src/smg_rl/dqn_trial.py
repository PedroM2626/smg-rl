"""Finite six-hour IQN/IMPALA trial with reviewed, immutable block checkpoints.

This controller performs mechanical safety/accounting reviews, not scientific
claims of policy quality. Owner follow-ups inspect the same retained evidence.
"""

import csv
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.profiles import PROFILES, game_id
from smg_rl.replay import replay_signature

STATE = Path('.local/impala-six-hour-state.json')
STOP = Path('.local/impala-six-hour.STOP')
LOCK = threading.RLock()
SECONDS = 6 * 3600


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def alive(controller):
    try:
        return abs(psutil.Process(controller['pid']).create_time() - controller['created_time']) < .01
    except psutil.Error:
        return False


def save(state):
    if not atomic_json(STATE, state):
        raise OSError('Could not atomically record trial state; no successor may be dispatched.')


def locate_run(job):
    for line in Path(job['stdout']).read_text(encoding='utf-8').splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and 'run' in value:
            run = Path(value['run']).resolve()
            if run.parent != Path(load_config(job['config'])['paths']['runs']).resolve():
                raise ValueError('Run output is outside the configured owned directory.')
            return run
    return None


def launch(state, game, steps, seconds, resume, label):
    """The pending command is durable before process creation; never blindly retry."""
    with LOCK:
        job = {'config': f'config/impala-iqn-{game}.yaml', 'label': label, 'phase': 'pending',
               'source_real': state['games'][game].get('real_decisions', 0),
               'source_attempts': state['games'][game].get('completed_attempts', 0),
               'requested_additional': steps, 'max_seconds': seconds, 'resume': resume,
               'stdout': f'.local/impala-{game}-{label}.out.log',
               'stderr': f'.local/impala-{game}-{label}.err.log'}
        command = [sys.executable, '-m', 'smg_rl.cli', '--config', job['config'], 'train-dqn',
                   '--steps', str(steps), '--max-seconds', str(seconds)]
        if resume:
            command.extend(['--resume', resume])
        job['command'] = command
        state['games'][game]['jobs'].append(job)
        save(state)
        with open(job['stdout'], 'x', encoding='utf-8') as out, open(job['stderr'], 'x', encoding='utf-8') as err:
            process = subprocess.Popen(command, stdout=out, stderr=err,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        job.update(phase='running', pid=process.pid, created_time=psutil.Process(process.pid).create_time())
        save(state)
        return process, job


def audit_block(run, config, source_real, source_attempts):
    """Verify real counts, reward sum, RAM outcomes, checkpoint hashes and attempts."""
    status = read(run / 'status.json')
    controller = read(run / 'controller.json')
    if alive(controller) or status['state'] != 'completed' or status['error'] is not None:
        raise ValueError('Only a clean, stopped block can authorize a successor.')
    if status['source_count'] != source_real or status['imagined_transitions'] != 0:
        raise ValueError('Unexpected branch ancestry or imagined data.')
    for value in status['losses'].values():
        if not np.isfinite(value):
            raise ValueError('Nonfinite final optimization metric.')
    checkpoint = run / 'checkpoint-final'
    manifest = read(checkpoint / 'manifest.json')
    if ((checkpoint / '.incomplete').exists() or manifest['signature'] != replay_signature(config)
            or manifest['settings'] != config['dqn']
            or manifest['agent_sha256'] != digest(checkpoint / 'agent.pt')
            or manifest['replay_sha256'] != digest(checkpoint / 'replay.npz')):
        raise ValueError('Checkpoint identity or integrity failed.')
    with np.load(checkpoint / 'replay.npz', allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        if metadata['emitted'] != status['real_decisions'] or metadata['size'] != min(
                status['real_decisions'], config['dqn']['capacity']):
            raise ValueError('Real replay accounting mismatch.')
    ledger = [json.loads(line) for line in (run / 'attempts.jsonl').read_text().splitlines()]
    if len(ledger) != status['episodes_this_run'] or source_attempts + len(ledger) != status['completed_attempts']:
        raise ValueError('Attempt ancestry does not reconcile.')
    physical, episodes, best = 0, Counter(), {}
    actions = Counter()
    for path in run.glob('worker-*/*/episode-*.csv'):
        with path.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        first, last = rows[0], rows[-1]
        key = f"{first['variation']}/{first['start_kind']}"
        decisions = rows[1:]
        physical += len(decisions)
        if int(last['emulated_frames']) != config['parallel']['frame_skip'] * len(decisions):
            raise ValueError('Physical-frame accounting differs from real decisions.')
        expected_credit = float(last['shaping_potential']) - float(first['shaping_potential'])
        observed_credit = sum(float(r['regional_shaping_reward']) for r in decisions)
        if abs(expected_credit - observed_credit) > 1e-6 or not -1e-8 <= observed_credit <= 1.000001:
            raise ValueError('Incremental route credit does not telescope to the new high-water mark.')
        for row in decisions:
            actions[row['action']] += 1
            expected_reward = float(row['regional_shaping_reward']) + float(row['damage_penalty'])
            expected_reward += config['rewards']['completion'] if row['complete'] == 'True' else (
                config['rewards']['death'] if row['dead'] == 'True' else 0.)
            timeout = config['rewards']['timeout'] if row['outcome'] == 'timeout' else 0.
            if float(row['timeout_penalty']) != timeout:
                raise ValueError('Unexpected timeout penalty.')
            expected_reward += timeout
            if abs(float(row['reward']) - expected_reward) > 1e-6:
                raise ValueError('Unexpected reward component in simple objective.')
            if float(row['regional_shaping_reward']) > 1e-8 and row['grounded'] != 'True':
                raise ValueError('Airborne progress was rewarded.')
        if last['outcome'] == 'success' and (last['complete'] != 'True' or last['dead'] == 'True'
                or int(last['scene_nerve'], 16) not in PROFILES[game_id(config)].complete):
            raise ValueError('Success lacks a validated RAM terminal.')
        if last['outcome'] != 'interrupted':
            episodes[key] += 1
        fraction = max(float(r['start_relative_route_progress']) for r in rows)
        if key not in best or fraction > best[key]['opening_advance']:
            best[key] = {'opening_advance': fraction, 'trajectory': str(path), 'outcome': last['outcome']}
    if physical != status['real_decisions'] - source_real or sum(episodes.values()) != len(ledger):
        raise ValueError('CSV decisions/episodes do not reconcile with the learner.')
    return {'passed': True, 'checkpoint': str(checkpoint), 'agent_sha256': manifest['agent_sha256'],
            'real_decisions': status['real_decisions'], 'completed_attempts': status['completed_attempts'],
            'elapsed_seconds': status['elapsed_seconds'], 'decisions_per_second': status['decisions_per_second'],
            'best_collection': best, 'actions': dict(actions), 'attempts': status['attempts'],
            'reason': 'Finite metrics, exact real/replay/attempt counts, signed artifacts and simple reward controls passed.',
            'note': 'Collection maxima include random exploration. Opening fractions are not mission percentages.'}


def apply_review(state, game, job):
    if job['phase'] == 'completed_reviewed':
        return job['review']
    run = locate_run(job)
    if run is None:
        raise ValueError('The subprocess did not publish an owned run.')
    review = audit_block(run, load_config(job['config']), job['source_real'], job['source_attempts'])
    with LOCK:
        job.update(phase='completed_reviewed', run=str(run), review=review)
        branch = state['games'][game]
        branch.update(real_decisions=review['real_decisions'], completed_attempts=review['completed_attempts'],
                      checkpoint=review['checkpoint'])
        branch['elapsed_training_seconds'] += review['elapsed_seconds']
        save(state)
    return review


def start_health():
    if STATE.exists() or STOP.exists():
        raise ValueError('Trial state/stop already exists; inspect it instead of duplicating collection.')
    dolphins = [p for p in psutil.process_iter(['name']) if (p.info['name'] or '').lower() == 'dolphin.exe']
    if dolphins:
        raise ValueError('Existing Dolphin processes need ownership review before dispatch.')
    state = {'schema': 1, 'phase': 'health_running', 'started_at': datetime.now(timezone.utc).isoformat(),
             'owner_authorization': 'October 26: correct incentive first, consider IMPALA, then train for six hours with attempt metrics.',
             'selection': 'IQN + IMPALA; no evidence establishes a genetic advantage, so no GA rollout is allocated.',
             'maximum_emulators': 6, 'training_seconds_per_game': SECONDS,
             'games': {game: {'jobs': [], 'real_decisions': 0, 'completed_attempts': 0,
                              'elapsed_training_seconds': 0., 'real_cap': cap}
                       for game, cap in [('smg1', 500000), ('smg2', 1000000)]}}
    save(state)
    for game, count in [('smg1', 3072), ('smg2', 6144)]:
        launch(state, game, count, 900, None, 'health')
    print(json.dumps(state, indent=2))


def supervise():
    state = read(STATE)
    if state['phase'] != 'health_running' or STOP.exists():
        raise ValueError('Only the once-reviewed health phase may enter this finite main trial.')
    for game, branch in state['games'].items():
        job = branch['jobs'][-1]
        if alive(job):
            raise ValueError('Health block is still running; no duplicate main block allowed.')
        review = apply_review(state, game, job)
        if review['real_decisions'] != job['requested_additional']:
            raise ValueError('Health hit its time boundary early; partial data requires diagnosis.')
    state.update(phase='main_running', supervisor={'pid': os.getpid(), 'created_time': psutil.Process().create_time()})
    save(state)
    failed = threading.Event()

    def branch_work(game):
        branch = state['games'][game]
        block = 1
        try:
            while branch['elapsed_training_seconds'] < SECONDS and branch['real_decisions'] < branch['real_cap']:
                if STOP.exists() or failed.is_set():
                    break
                seconds = min(900 if block == 1 else 3600, SECONDS - branch['elapsed_training_seconds'])
                steps = branch['real_cap'] - branch['real_decisions']
                workers = load_config(f'config/impala-iqn-{game}.yaml')['parallel']['workers']
                steps -= steps % workers
                if not steps:
                    break
                process, job = launch(state, game, steps, seconds, branch['checkpoint'], f'block{block:02d}')
                while process.poll() is None:
                    run = locate_run(job)
                    if run and job.get('run') != str(run):
                        with LOCK:
                            job['run'] = str(run)
                            job['actual_controller'] = read(run / 'controller.json')
                            save(state)
                    if STOP.exists() or failed.is_set():
                        if run:
                            (run / 'STOP').touch()
                    time.sleep(5)
                if process.returncode or STOP.exists() or failed.is_set():
                    raise RuntimeError(f"Block ended with code {process.returncode}; inspect {job['stderr']} and retain partial data.")
                apply_review(state, game, job)
                block += 1
            with LOCK:
                branch['phase'] = 'stopped' if STOP.exists() or failed.is_set() else 'completed'
                save(state)
        except BaseException as exc:
            failed.set()
            with LOCK:
                branch.update(phase='needs_review', error=repr(exc))
                current = branch['jobs'][-1]
                current['phase'] = 'needs_review'
                run = locate_run(current)
                if run:
                    current['run'] = str(run)
                    if (run / 'status.json').exists():
                        current['unreviewed_final_status'] = read(run / 'status.json')
                save(state)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(branch_work, state['games']))
    state['phase'] = 'needs_review' if failed.is_set() else 'stopped' if STOP.exists() else 'completed'
    state['finished_at'] = datetime.now(timezone.utc).isoformat()
    save(state)
    print(json.dumps(state, indent=2))


def snapshot():
    state = read(STATE)
    for branch in state['games'].values():
        latest = branch['jobs'][-1]
        latest['process_alive'] = alive(latest)
        run = locate_run(latest)
        if run and (run / 'status.json').exists():
            latest.update(run=str(run), current_status=read(run / 'status.json'))
    print(json.dumps(state, indent=2))
