"""Pause/audit IQN, run one finite genetic pilot, then restore the overnight allocation."""

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import psutil

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.dqn_trial import alive, audit_block, locate_run, read

STATE = Path('.local/genetic-hour-state.json')
STOP = Path('.local/genetic-hour.STOP')
OVERNIGHT = Path('.local/smg1-overnight-state.json')
OVERNIGHT_STOP = Path('.local/smg1-overnight.STOP')
PAUSED = Path('.local/smg1-overnight-paused-for-ga.json')
CONFIG = 'config/genetic-smg1.yaml'
PAUSE_TOKEN = 'Owner-authorized one-hour genetic pilot; IQN resume remains bounded by October27 07:00 BRT.'


def save(state):
    if not atomic_json(STATE, state):
        raise OSError('Genetic dispatch state was not persisted; refuse dispatch.')


def launch(state, name, command):
    job = {'command': command, 'phase': 'pending_dispatch', 'config': CONFIG,
           'stdout': f'.local/genetic-hour-{name}.out.log', 'stderr': f'.local/genetic-hour-{name}.err.log'}
    state[name] = job
    save(state)
    environment = dict(os.environ)
    environment.update({k: '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    with open(job['stdout'], 'x', encoding='utf-8') as out, open(job['stderr'], 'x', encoding='utf-8') as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, env=environment,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    job.update(phase='running', pid=process.pid, created_time=psutil.Process(process.pid).create_time())
    save(state)
    return process, job


def supervise():
    if STATE.exists() or STOP.exists() or OVERNIGHT_STOP.exists() or PAUSED.exists():
        raise ValueError('Existing genetic/stop/pause state requires review; no duplicate pilot.')
    original = read(OVERNIGHT)
    if original['phase'] != 'running' or not alive(original['supervisor']):
        raise ValueError('The expected overnight supervisor is not healthy; diagnose before dispatch.')
    deadline = datetime.fromisoformat(original['deadline_utc']).timestamp()
    if deadline - time.time() < 4200:
        raise ValueError('Insufficient time for this pilot before the existing overnight deadline.')
    state = {'schema': 1, 'phase': 'pausing_iqn', 'seconds': 3600, 'maximum_emulators': 6,
             'owner_authorization': 'Implement and run a genetic algorithm for one hour; fitness is furthest advance.',
             'started_at': datetime.now(timezone.utc).isoformat(),
             'supervisor': {'pid': os.getpid(), 'created_time': psutil.Process().create_time()},
             'iqn_supervisor_before_pause': original['supervisor'], 'iqn_deadline_utc': original['deadline_utc']}
    save(state)
    try:
        OVERNIGHT_STOP.write_text(PAUSE_TOKEN, encoding='utf-8')
        end = time.monotonic() + 180
        while alive(original['supervisor']) and time.monotonic() < end:
            time.sleep(3)
        if alive(original['supervisor']):
            raise TimeoutError('IQN did not stop cleanly; no genetic emulator may start.')
        paused = deepcopy(read(OVERNIGHT))
        branch = paused['games']['smg1']
        job = branch['jobs'][-1]
        run = locate_run(job)
        review = audit_block(run, load_config(job['config']), job['source_real'], job['source_attempts'], paused=True)
        if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
            raise ValueError('An emulator remains after IQN pause; inspect ownership, no duplicate launch.')
        job.update(phase='paused_for_genetic_reviewed', review=review)
        branch.update(real_decisions=review['real_decisions'], completed_attempts=review['completed_attempts'],
                      checkpoint=review['checkpoint'])
        branch['elapsed_training_seconds'] += review['elapsed_seconds']
        paused.update(phase='completed', reviewed_genetic_pause=True, pause_reason=PAUSE_TOKEN)
        paused.pop('error', None)
        if not atomic_json(PAUSED, paused):
            raise OSError('Could not persist the reviewed IQN resume point.')
        state.update(phase='iqn_paused_reviewed', iqn_pause_review=review)
        save(state)
        if STOP.exists():
            state['phase'] = 'owner_stopped'
            return
        command = [sys.executable, 'scripts/train_genetic.py', '--config', CONFIG,
                   '--source', branch['checkpoint'], '--seconds', '3600']
        process, job = launch(state, 'genetic', command)
        state['phase'] = 'genetic_running'
        save(state)
        while process.poll() is None:
            run = locate_run(job)
            if run and job.get('run') != str(run):
                job['run'] = str(run)
                save(state)
            if STOP.exists() and run:
                (run / 'STOP').touch()
            time.sleep(3)
        if process.returncode or STOP.exists():
            raise RuntimeError('Genetic trial failed/stopped; preserve partial evidence and do not retry or resume automatically.')
        run = locate_run(job)
        result = read(run / 'summary.json')
        trials = [json.loads(line) for line in (run / 'trials.jsonl').read_text().splitlines()]
        if (result['state'] != 'completed' or result['error'] or not result['source_unchanged']
                or result['gradient_updates'] != 0 or result['generations_completed'] < 1
                or sum(t['decisions'] for t in trials) != result['real_decisions']
                or sum(t['fitness_eligible'] for t in trials) != result['completed_trials']
                or result['champion_sha256'] != digest(run / 'champion.pt')):
            raise ValueError('Genetic artifact/accounting review failed; no automatic successor.')
        job.update(phase='completed_reviewed', run=str(run), review=result)
        state['phase'] = 'genetic_completed_reviewed'
        save(state)
        if (STOP.exists() or time.time() >= deadline or
                OVERNIGHT_STOP.read_text(encoding='utf-8') != PAUSE_TOKEN):
            state['phase'] = 'completed_without_iqn_resume'
            return
        if alive(read(OVERNIGHT)['supervisor']):
            raise ValueError('Another overnight supervisor owns training; do not duplicate it.')
        OVERNIGHT.rename('.local/smg1-overnight-state-before-genetic.json')
        OVERNIGHT_STOP.unlink()
        _, resume = launch(state, 'iqn_resume', [sys.executable, 'scripts/run_smg1_overnight.py', 'supervise'])
        end = time.monotonic() + 60
        while not OVERNIGHT.exists() and time.monotonic() < end:
            time.sleep(2)
        resumed = read(OVERNIGHT)
        if not alive(resumed['supervisor']) or resumed['phase'] not in ('running', 'waiting_for_initial_segment'):
            raise ValueError('IQN resumption did not become healthy; inspect retained state.')
        resume['actual_supervisor'] = resumed['supervisor']
        exporter = read(Path('.local/impala-telemetry-state.json'))
        if not alive(exporter['controller']):
            launch(state, 'telemetry_resume', [sys.executable, 'scripts/export_training_metrics.py', '--watch',
                '--max-seconds', str(max(1, deadline - time.time() + 300)), '--training-state', str(OVERNIGHT)])
        state['phase'] = 'completed'
        state['completed_at'] = datetime.now(timezone.utc).isoformat()
    except BaseException as exc:
        state.update(phase='needs_review', error=repr(exc))
        raise
    finally:
        save(state)


def snapshot():
    state = read(STATE)
    job = state.get('genetic')
    if job:
        run = locate_run(job)
        if run and (run / 'status.json').exists():
            job['status'] = read(run / 'status.json')
    print(json.dumps(state, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['supervise', 'snapshot'])
    args = parser.parse_args()
    {'supervise': supervise, 'snapshot': snapshot}[args.mode]()
