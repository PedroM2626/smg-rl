"""Adopt the running SMG1 segment, then continue only until an absolute deadline."""

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import psutil

from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.dqn_trial import alive, audit_block, locate_run, read
from smg_rl.resource_memory import require_memory_headroom

INITIAL = Path('.local/smg1-six-two-hour-state.json')
STATE = Path('.local/smg1-overnight-state.json')
STOP = Path('.local/smg1-overnight.STOP')
DEADLINE = '2026-10-27T10:00:00+00:00'
CONFIG = 'config/smg1-six-two-hour.yaml'
RECOVERY = Path('.local/smg1-overnight-reviewed-recovery.json')


def remaining_seconds(deadline=DEADLINE):
    return max(0., datetime.fromisoformat(deadline).timestamp() - time.time())


def save(state):
    if not atomic_json(STATE, state):
        raise OSError('Overnight dispatch state was not persisted; no successor allowed.')


def supervise():
    if STATE.exists() or STOP.exists() or remaining_seconds() <= 0:
        raise ValueError('Existing state/stop or expired deadline; never duplicate overnight training.')
    initial = read(INITIAL)
    recovery = read(RECOVERY) if RECOVERY.exists() else None
    if recovery:
        if initial['phase'] != 'needs_review' or alive(initial['supervisor']):
            raise ValueError('Recovery requires the reviewed, stopped initial supervisor.')
        last = initial['games']['smg1']['jobs'][-1]
        if recovery['run'] != str(locate_run(last)) or not recovery['manual_review']:
            raise ValueError('Recovery evidence does not identify this failed block.')
        review = audit_block(locate_run(last), load_config(CONFIG), last['source_real'], last['source_attempts'], failed=True)
        last.update(phase='failed_partial_reviewed', review=review)
        branch = initial['games']['smg1']
        branch.update(real_decisions=review['real_decisions'], completed_attempts=review['completed_attempts'],
                      checkpoint=review['checkpoint'])
        branch['elapsed_training_seconds'] += review['elapsed_seconds']
        initial['phase'] = 'completed'
    if initial['phase'] not in ('running', 'completed'):
        raise ValueError('The initial segment must be healthy; no blind partial recovery.')
    state = {'schema': 1, 'phase': 'waiting_for_initial_segment', 'maximum_emulators': 6,
             'deadline_utc': DEADLINE, 'deadline_local': '2026-10-27T07:00:00-03:00',
             'owner_authorization': 'Owner extended the same six-worker SMG1 training until 07:00 and requested reviews every 3 hours.',
             'new_real_decision_cap': 4800000,
             'supervisor': {'pid': os.getpid(), 'created_time': psutil.Process().create_time()},
             'initial_supervisor': initial['supervisor'], 'games': deepcopy(initial['games'])}
    if recovery:
        state['reviewed_recovery'] = recovery
    save(state)
    try:
        while True:
            if not recovery:
                initial = read(INITIAL)
            state['games'] = deepcopy(initial['games'])
            state['initial_phase'] = initial['phase']
            save(state)
            if STOP.exists() or remaining_seconds() <= 0:
                if initial['phase'] == 'running':
                    Path('.local/smg1-six-two-hour.STOP').touch()
                state['phase'] = 'stopped' if STOP.exists() else 'completed'
                return
            if initial['phase'] == 'completed':
                break
            if initial['phase'] != 'running' or not alive(initial['supervisor']):
                raise RuntimeError('Initial segment failed/stopped; retain evidence and do not retry automatically.')
            time.sleep(3)
        branch = state['games']['smg1']
        last = branch['jobs'][-1]
        audit_block(locate_run(last), load_config(last['config']), last['source_real'], last['source_attempts'], failed=bool(recovery))
        state['phase'] = 'running'
        save(state)
        for index in range(24):
            seconds = remaining_seconds()
            steps = state['new_real_decision_cap'] - (branch['real_decisions'] - branch['source_real'])
            if recovery and index == 0:
                steps = min(3072, steps)
                seconds = min(300., seconds)
            steps -= steps % 6
            if seconds <= 0 or steps < 6 or STOP.exists():
                break
            require_memory_headroom()
            job = {'config': CONFIG, 'label': f'overnight{index:02d}', 'phase': 'pending_dispatch',
                   'source_real': branch['real_decisions'], 'source_attempts': branch['completed_attempts'],
                   'requested_additional': steps, 'max_seconds': min(3600., seconds), 'deadline_utc': DEADLINE,
                   'stdout': f'.local/smg1-overnight-{index:02d}.out.log',
                   'stderr': f'.local/smg1-overnight-{index:02d}.err.log'}
            command = [sys.executable, '-m', 'smg_rl.cli', '--config', CONFIG, 'train-dqn',
                       '--steps', str(steps), '--max-seconds', str(job['max_seconds']),
                       '--deadline-utc', DEADLINE, '--resume', branch['checkpoint']]
            job['command'] = command
            branch['jobs'].append(job)
            save(state)
            environment = dict(os.environ)
            environment.update({k: '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
            with open(job['stdout'], 'x', encoding='utf-8') as out, open(job['stderr'], 'x', encoding='utf-8') as err:
                process = subprocess.Popen(command, stdout=out, stderr=err, env=environment,
                                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            job.update(phase='running', pid=process.pid, created_time=psutil.Process(process.pid).create_time())
            save(state)
            while process.poll() is None:
                run = locate_run(job)
                if run and job.get('run') != str(run):
                    job.update(run=str(run), actual_controller=read(run / 'controller.json'))
                    save(state)
                if STOP.exists() and run:
                    (run / 'STOP').touch()
                time.sleep(3)
            if process.returncode or STOP.exists():
                raise RuntimeError('Overnight block failed/owner stopped; preserve partial work, no blind retry.')
            run = locate_run(job)
            review = audit_block(run, load_config(CONFIG), job['source_real'], job['source_attempts'])
            job.update(phase='completed_reviewed', run=str(run), review=review)
            branch.update(real_decisions=review['real_decisions'], completed_attempts=review['completed_attempts'],
                          checkpoint=review['checkpoint'])
            branch['elapsed_training_seconds'] += review['elapsed_seconds']
            save(state)
        state['phase'] = 'stopped' if STOP.exists() else 'completed'
    except BaseException as exc:
        state.update(phase='needs_review', error=repr(exc))
        raise
    finally:
        save(state)


def snapshot():
    state = read(STATE)
    job = state['games']['smg1']['jobs'][-1]
    run = locate_run(job)
    if run and (run / 'status.json').exists():
        job.update(run=str(run), status=read(run / 'status.json'))
    print(json.dumps(state, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['supervise', 'snapshot'])
    args = parser.parse_args()
    {'supervise': supervise, 'snapshot': snapshot}[args.mode]()
