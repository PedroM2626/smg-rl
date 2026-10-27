"""Owner-authorized continuous genetic continuation until October27 07:00 BRT."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.dqn_trial import alive, audit_block, locate_run, read

STATE = Path('.local/genetic-overnight-state.json')
STOP = Path('.local/genetic-overnight.STOP')
IQN = Path('.local/smg1-overnight-state.json')
IQN_STOP = Path('.local/smg1-overnight.STOP')
PILOT = Path('.local/genetic-hour-state.json')
DEADLINE = '2026-10-27T10:00:00+00:00'
CONFIG = 'config/genetic-smg1.yaml'


def save(state):
    if not atomic_json(STATE, state):
        raise OSError('Genetic overnight dispatch state must persist before any action.')


def supervise():
    if STATE.exists() or STOP.exists() or IQN_STOP.exists():
        raise ValueError('Existing continuation/stop; never duplicate genetic dispatch.')
    initial = read(IQN)
    pilot = read(PILOT)
    if initial['phase'] != 'running' or not alive(initial['supervisor']) or pilot['phase'] != 'completed':
        raise ValueError('Expected a healthy IQN and a completed genetic pilot; review first.')
    if datetime.fromisoformat(DEADLINE).timestamp() - time.time() < 300:
        raise ValueError('Insufficient time before the explicitly authorized deadline.')
    parent = locate_run(pilot['genetic'])
    source = read(parent / 'summary.json')['source']
    state = {'schema': 1, 'phase': 'pausing_iqn', 'deadline_utc': DEADLINE,
             'owner_authorization': 'Owner selected continued genetic training without scheduled pauses and progress analysis at07:00.',
             'maximum_emulators': 6, 'resume_genetic_run': str(parent), 'frozen_feature_source': source,
             'supervisor': {'pid': os.getpid(), 'created_time': psutil.Process().create_time()},
             'started_at': datetime.now(timezone.utc).isoformat(), 'iqn_auto_resume': False}
    save(state)
    try:
        IQN_STOP.write_text('Owner replaced the remaining overnight IQN allocation with genetic continuation.', encoding='utf-8')
        until = time.monotonic() + 180
        while alive(initial['supervisor']) and time.monotonic() < until:
            time.sleep(3)
        if alive(initial['supervisor']):
            raise TimeoutError('IQN did not stop; no overlapping genetic processes allowed.')
        paused = read(IQN)
        job = paused['games']['smg1']['jobs'][-1]
        review = audit_block(locate_run(job), load_config(job['config']), job['source_real'], job['source_attempts'], paused=True)
        if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
            raise ValueError('Dolphin remains after pause; inspect ownership before any successor.')
        state.update(phase='iqn_paused_reviewed', iqn_pause_review=review)
        save(state)
        if STOP.exists():
            state['phase'] = 'owner_stopped'
            return
        command = [sys.executable, 'scripts/train_genetic.py', '--config', CONFIG, '--source', source,
                   '--seconds', '43200', '--deadline-utc', DEADLINE, '--resume-run', str(parent),
                   '--confirmation-seconds', '0']
        job = {'command': command, 'config': CONFIG, 'phase': 'pending_dispatch',
               'stdout': '.local/genetic-overnight.out.log', 'stderr': '.local/genetic-overnight.err.log'}
        state['genetic'] = job
        save(state)
        environment = dict(os.environ)
        environment.update({k: '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
        with open(job['stdout'], 'x', encoding='utf-8') as out, open(job['stderr'], 'x', encoding='utf-8') as err:
            process = subprocess.Popen(command, stdout=out, stderr=err, env=environment,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        job.update(phase='running', pid=process.pid, created_time=psutil.Process(process.pid).create_time())
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
            raise RuntimeError('Genetic continuation failed/stopped; preserve evidence, no blind retry or IQN restart.')
        run = locate_run(job)
        result = read(run / 'summary.json')
        trials = [json.loads(line) for line in (run / 'trials.jsonl').read_text().splitlines()]
        if (result['state'] != 'completed' or result['error'] or not result['source_unchanged']
                or result['gradient_updates'] != 0 or sum(t['decisions'] for t in trials) != result['real_decisions']
                or sum(t['fitness_eligible'] for t in trials) != result['completed_trials']
                or result['champion_sha256'] != digest(run / 'champion.pt')):
            raise ValueError('Final genetic continuation evidence does not reconcile.')
        job.update(phase='completed_reviewed', review=result)
        state.update(phase='completed', completed_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        state.update(phase='needs_review', error=repr(exc))
        raise
    finally:
        save(state)


def snapshot():
    state = read(STATE)
    if state.get('genetic'):
        run = locate_run(state['genetic'])
        if run and (run / 'status.json').exists():
            state['genetic']['status'] = read(run / 'status.json')
    print(json.dumps(state, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['supervise', 'snapshot'])
    args = parser.parse_args()
    {'supervise': supervise, 'snapshot': snapshot}[args.mode]()
