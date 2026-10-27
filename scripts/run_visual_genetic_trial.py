"""Pause audited IQN and run the owner-requested full-visual GA for a finite two hours."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psutil

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.dqn_trial import alive, audit_block, locate_run, read

STATE = Path('.local/genetic-visual-20261027-state.json')
STOP = Path('.local/genetic-visual-20261027.STOP')
IQN = Path('.local/iqn-return-20261027-state.json')
IQN_STOP = Path('.local/iqn-return-20261027.STOP')
ANCESTOR = Path('.local/genetic-overnight-state.json')
CONFIG = 'config/genetic-smg1.yaml'


def save(state):
    if not atomic_json(STATE, state):
        raise OSError('Full-visual genetic state must persist before any dispatch.')


def supervise():
    if STATE.exists() or STOP.exists() or IQN_STOP.exists():
        raise ValueError('Existing dispatch/stop state; never duplicate visual genetic collection.')
    iqn, prior = read(IQN), read(ANCESTOR)
    if iqn['phase'] != 'running' or not alive(iqn['supervisor']) or prior['phase'] != 'completed':
        raise ValueError('Review actual IQN/ancestral GA ownership before switching algorithms.')
    parent = locate_run(prior['genetic'])
    source = read(parent / 'summary.json')['source']
    state = {'schema': 1, 'phase': 'pausing_iqn', 'maximum_emulators': 6, 'training_seconds': 7200,
             'owner_authorization': 'Owner requested continuing genetic training with visual representation unlocked.',
             'method': 'Evolve every deterministic visual/action-path parameter; no gradients.',
             'ancestor_run': str(parent), 'feature_initialization_source': source,
             'supervisor': {'pid': os.getpid(), 'created_time': psutil.Process().create_time()},
             'started_at': datetime.now(timezone.utc).isoformat(), 'iqn_auto_resume': False}
    save(state)
    try:
        IQN_STOP.write_text('Owner switched from IQN to a full-visual genetic trial.', encoding='utf-8')
        limit = time.monotonic() + 180
        while alive(iqn['supervisor']) and time.monotonic() < limit:
            time.sleep(3)
        if alive(iqn['supervisor']):
            raise TimeoutError('IQN did not close; no genetic emulator may overlap it.')
        paused = read(IQN)
        oldjob = paused['games']['smg1']['jobs'][-1]
        oldrun = locate_run(oldjob)
        review = audit_block(oldrun, load_config(oldjob['config']), oldjob['source_real'], oldjob['source_attempts'], paused=True)
        if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
            raise ValueError('An emulator remains after the reviewed IQN stop; inspect ownership.')
        state.update(phase='iqn_paused_reviewed', iqn_pause_review=review)
        save(state)
        if STOP.exists():
            state['phase'] = 'owner_stopped'
            return
        deadline = (datetime.now(timezone.utc) + timedelta(seconds=7200)).isoformat()
        state['deadline_utc'] = deadline
        command = [sys.executable, 'scripts/train_genetic.py', '--config', CONFIG, '--source', source,
                   '--seconds', '7200', '--deadline-utc', deadline, '--resume-run', str(parent),
                   '--confirmation-seconds', '0', '--policy-mode', 'full_visual']
        job = {'phase': 'pending_dispatch', 'config': CONFIG, 'command': command,
               'stdout': '.local/genetic-visual-20261027.out.log', 'stderr': '.local/genetic-visual-20261027.err.log'}
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
            raise RuntimeError('Full-visual collection failed/stopped; retain partial evidence, no blind retry or IQN restart.')
        run = locate_run(job)
        result = read(run / 'summary.json')
        trials = [json.loads(line) for line in (run / 'trials.jsonl').read_text().splitlines()]
        if (result['state'] != 'completed' or result['error'] or not result['source_unchanged']
                or result['policy_mode'] != 'full_visual' or result['visual_evolvable_parameters'] <= 0
                or result['gradient_updates'] != 0 or sum(t['decisions'] for t in trials) != result['real_decisions']
                or sum(t['fitness_eligible'] for t in trials) != result['completed_trials']
                or digest(run / 'champion.pt') != result['champion_sha256']
                or digest(run / 'population-final.npz') != result['final_population_sha256']):
            raise ValueError('Full-visual final identity/accounting audit failed.')
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
