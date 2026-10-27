"""Finite owner-authorized SMG1 IQN continuation with six lean collectors."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.dqn_trial import alive, audit_block, locate_run, read
from smg_rl.resource_disk import require_disk_headroom
from smg_rl.resource_memory import require_memory_headroom

STATE = Path('.local/smg1-six-two-hour-state.json')
STOP = Path('.local/smg1-six-two-hour.STOP')
CONFIG = 'config/smg1-six-two-hour.yaml'
SOURCE = Path('.local/smg1-six-source/checkpoint-six-workers')
LOG_PREFIX = 'smg1-six'


def save(state):
    if not atomic_json(STATE, state):
        raise OSError('Could not persist the finite continuation state.')


def snapshot():
    state = read(STATE)
    for job in state['games']['smg1']['jobs']:
        if job['phase'] == 'running':
            run = locate_run(job)
            job['controller_alive'] = alive(job)
            if run and (run / 'status.json').exists():
                job['run'], job['status'] = str(run), read(run / 'status.json')
    print(json.dumps(state, indent=2))


def finite_budget(training_seconds):
    if type(training_seconds) is not int or not 60 <= training_seconds <= 14400:
        raise ValueError('Use an explicit finite integer allocation from60 to14400 seconds.')
    return max(3072, training_seconds * 100), (training_seconds + 1799) // 1800 + 4


def supervise(resume_review=None, training_seconds=7200):
    real_cap, dispatch_cap = finite_budget(training_seconds)
    if STATE.exists() or STOP.exists():
        raise ValueError('Existing continuation/stop requires review; never dispatch a duplicate.')
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('Review emulator ownership before starting the six SMG1 collectors.')
    source = SOURCE
    if resume_review:
        retained = read(Path(resume_review))['iqn_pause_review']
        if not retained['passed']:
            raise ValueError('The retained IQN pause was not reviewed.')
        source = Path(retained['checkpoint'])
        prior = read(source.parent / 'status.json')
        reviewed = audit_block(source.parent, load_config(CONFIG), prior['source_count'],
                               prior['completed_attempts'] - prior['episodes_this_run'], paused=True)
        if reviewed['agent_sha256'] != retained['agent_sha256']:
            raise ValueError('The reviewed IQN resume checkpoint identity changed.')
        migration = reviewed
    else:
        migration = read(SOURCE / 'worker-migration.json')
        if migration['target_workers'] != 6 or not migration['learner_optimizer_rng_and_counters_equal']:
            raise ValueError('An audited six-worker migration checkpoint is required.')
    branch = {'config': CONFIG, 'jobs': [], 'source_real': migration['real_decisions'],
              'real_decisions': migration['real_decisions'], 'source_attempts': migration['completed_attempts'],
              'completed_attempts': migration['completed_attempts'], 'elapsed_training_seconds': 0.,
              'checkpoint': str(source.resolve())}
    state = {'schema': 1, 'phase': 'running', 'owner_authorization': f'Owner requested six SMG1 workers; finite allocation{training_seconds} seconds.',
             'supervisor': {'pid': os.getpid(), 'created_time': psutil.Process().create_time()},
             'training_seconds': training_seconds, 'new_real_decision_cap': real_cap, 'maximum_emulators': 6,
             'maximum_dispatches': dispatch_cap,
             'games': {'smg1': branch}, 'note': 'One shared policy, unchanged objective/control/start mixture. Zero stars alone does not prove a game bug.'}
    if resume_review:
        state.update(owner_authorization=f'Owner requested IQN continuation; finite{training_seconds}-second allocation with staged reviews.',
                     retained_pause_review=str(resume_review), visual_encoder='IMPALA, trainable by gradient')
    save(state)
    try:
        for block in range(dispatch_cap):  # Health,30min,60min/remainder; finite dispatch ceiling.
            remaining = training_seconds - branch['elapsed_training_seconds']
            decisions = real_cap - (branch['real_decisions'] - branch['source_real'])
            if remaining <= 0 or decisions < 6 or STOP.exists():
                break
            require_memory_headroom()
            require_disk_headroom(load_config(CONFIG))
            seconds = min(300 if block == 0 else 1800 if block == 1 else 3600, remaining)
            steps = min(3072, decisions) if block == 0 else decisions
            steps -= steps % 6
            label = f'block{block:02d}'
            job = {'config': CONFIG, 'phase': 'pending_dispatch', 'label': label,
                   'source_real': branch['real_decisions'], 'source_attempts': branch['completed_attempts'],
                   'requested_additional': steps, 'max_seconds': seconds,
                   'stdout': f'.local/{LOG_PREFIX}-{label}.out.log', 'stderr': f'.local/{LOG_PREFIX}-{label}.err.log',
                   'resume': branch['checkpoint']}
            command = [sys.executable, '-m', 'smg_rl.cli', '--config', CONFIG, 'train-dqn',
                       '--steps', str(steps), '--max-seconds', str(seconds), '--resume', branch['checkpoint']]
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
                raise RuntimeError(f'Block stopped/failed with code{process.returncode}; retain partial data, no blind retry.')
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
        if branch['jobs']:
            job = branch['jobs'][-1]
            run = locate_run(job)
            if run and (run / 'status.json').exists():
                job['unreviewed_status'] = read(run / 'status.json')
        raise
    finally:
        save(state)
    print(json.dumps(state, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['supervise', 'snapshot'])
    parser.add_argument('--resume-reviewed', help='Completed controller state holding an audited IQN pause checkpoint.')
    parser.add_argument('--state', type=Path)
    parser.add_argument('--stop', type=Path)
    parser.add_argument('--log-prefix')
    parser.add_argument('--training-seconds', type=int, default=7200)
    args = parser.parse_args()
    STATE, STOP = args.state or STATE, args.stop or STOP
    LOG_PREFIX = args.log_prefix or LOG_PREFIX
    if args.mode == 'supervise':
        supervise(args.resume_reviewed, args.training_seconds)
    else:
        snapshot()
