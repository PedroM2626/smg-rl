"""Finite, sequential SMG1 reward-scale study; never retries or extends a block."""

import ast
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import yaml

from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.dqn_trial import audit_block, locate_run
from smg_rl.resource_disk import require_disk_headroom
from smg_rl.resource_memory import require_memory_headroom
from smg_rl.route_transfer import transfer_policy

os.chdir(Path(__file__).resolve().parents[1])
ROOT = Path('.local/reward-scale-ab-20261028')
STATE = ROOT / 'state.json'
STOP = ROOT / 'STOP'
SOURCE = 'runs/model-free/smg1-six-two-hour/dqn-20261027-114555-611503/checkpoint-final'
SOURCE_CONFIG = 'config/smg1-six-two-hour.yaml'
TEMPLATE = '.local/dynamic-route-smg1-v0000.yaml'
SEEDS = (142, 143)
ARMS = (('control', 50000.0), ('candidate', 5000.0))


def save(state):
    if not atomic_json(STATE, state):
        raise OSError('Cannot persist study state.')


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(value)


def prepare():
    if ROOT.exists():
        raise FileExistsError('Existing study must be reviewed, never duplicated.')
    require_memory_headroom()
    ROOT.mkdir(parents=True)
    source_config = load_config(SOURCE_CONFIG)
    template = yaml.safe_load(Path(TEMPLATE).read_text(encoding='utf-8'))
    state = {'phase': 'preparing', 'created_utc': datetime.now(timezone.utc).isoformat(),
             'source': SOURCE, 'source_config': SOURCE_CONFIG, 'runs': [],
             'design': {'seeds': list(SEEDS), 'arms': dict(ARMS), 'new_real_per_arm': 6144,
                        'training_seconds_per_arm_cap': 900, 'workers': 6,
                        'evaluation': 'One greedy seed 11 and three stochastic action seeds 11/23/47; epsilon 0.1, at most 600 decisions each.',
                        'gate': 'Both independent training seeds must show candidate stochastic mean opening at least 5 percentage points higher than matched control; no reduction in real full-start stars. Exploratory only; no automatic longer run.',
                        'scope': 'Same source weights, reset optimizer/replay, starts, seed, route and settings. Only route normalization changes; fixed opening and no automatic extension.'}}
    save(state)
    for seed in SEEDS:
        for arm, normalization in ARMS:
            raw = yaml.safe_load(yaml.safe_dump(template))
            raw['training']['seed'] = seed
            raw['rewards']['revision'] = 'reward-scale-study-v1'
            raw['regional_reward']['normalization_distance'] = normalization
            raw['paths'].update(runs=f'runs/model-free/reward-scale-ab-20261028/{arm}-{seed}',
                                replay=f'runs/model-free/reward-scale-ab-20261028/{arm}-{seed}/replay',
                                user=str(ROOT / f'{arm}-{seed}-template'))
            config = Path(f'.local/reward-scale-{arm}-{seed}-20261028.yaml')
            write_new(config, yaml.safe_dump(raw, sort_keys=False))
            transfer = transfer_policy(SOURCE, source_config, load_config(config), ROOT / f'transfer-{arm}-{seed}')
            state['runs'].append({'arm': arm, 'seed': seed, 'normalization': normalization,
                                  'config': str(config), 'transfer': transfer, 'phase': 'prepared'})
            save(state)
    state['phase'] = 'prepared'
    save(state)
    print(json.dumps({'state': str(STATE), 'runs': len(state['runs'])}))


def call_logged(command, stdout, stderr):
    with Path(stdout).open('x', encoding='utf-8') as out, Path(stderr).open('x', encoding='utf-8') as err:
        env = dict(os.environ)
        env.update({key: '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                         'NUMEXPR_NUM_THREADS')})
        return subprocess.Popen(command, stdout=out, stderr=err, env=env,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))


def wait_process(process, job, state):
    job.update(pid=process.pid, created_time=psutil.Process(process.pid).create_time(), phase='running')
    save(state)
    while process.poll() is None:
        if STOP.exists():
            run = locate_run(job) if 'config' in job else None
            if run:
                (run / 'STOP').touch()
        time.sleep(3)
    job['exit_code'] = process.returncode
    save(state)
    if process.returncode != 0 or STOP.exists():
        raise RuntimeError('Stopped or failed process: retain partial artifacts; do not retry automatically.')


def evaluate(state, job, mode):
    label = f"{job['arm']}-{job['seed']}-{mode}"
    record = {'phase': 'pending', 'mode': mode, 'stdout': str(ROOT / f'{label}.out.log'),
              'stderr': str(ROOT / f'{label}.err.log')}
    cmd = [sys.executable, '-m', 'smg_rl.cli', '--config', job['config'], 'record-dqn',
           job['review']['checkpoint'], '--variation', '0', '--max-steps', '600', '--seeds']
    cmd += ['11'] if mode == 'greedy' else ['11', '23', '47', '--stochastic', '--evaluation-epsilon', '0.1']
    record['command'] = cmd
    job.setdefault('evaluations', []).append(record)
    save(state)
    process = call_logged(cmd, record['stdout'], record['stderr'])
    wait_process(process, record, state)
    found = None
    for line in Path(record['stdout']).read_text(encoding='utf-8').splitlines():
        try:
            value = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, dict) and 'evaluation_run' in value:
            found = Path(value['evaluation_run'])
            break
    if found is None:
        raise ValueError('Evaluation did not publish its run directory.')
    result = json.loads((found / 'recording.json').read_text(encoding='utf-8'))
    if result['state'] != 'completed' or len(result['episodes']) != (1 if mode == 'greedy' else 3):
        raise ValueError('Evaluation did not complete all declared episodes.')
    record.update(phase='reviewed', run=str(found), result=result)
    save(state)


def run():
    state = json.loads(STATE.read_text(encoding='utf-8'))
    if state['phase'] != 'prepared' or STOP.exists():
        raise ValueError('Study cannot be resumed or duplicated automatically.')
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('Existing Dolphin processes need ownership review.')
    state.update(phase='running', supervisor={'pid': os.getpid(), 'created_time': psutil.Process().create_time()})
    save(state)
    try:
        for job in state['runs']:
            require_memory_headroom()
            require_disk_headroom(load_config(job['config']))
            label = f"{job['arm']}-{job['seed']}-train"
            job.update(phase='pending', stdout=str(ROOT / f'{label}.out.log'),
                       stderr=str(ROOT / f'{label}.err.log'), source_real=0, source_attempts=0)
            cmd = [sys.executable, '-m', 'smg_rl.cli', '--config', job['config'], 'train-dqn',
                   '--steps', '6144', '--max-seconds', '900', '--resume', job['transfer']['destination']]
            job['command'] = cmd
            save(state)
            process = call_logged(cmd, job['stdout'], job['stderr'])
            wait_process(process, job, state)
            path = locate_run(job)
            if path is None:
                raise ValueError('Training process did not publish a run.')
            review = audit_block(path, load_config(job['config']), 0, 0)
            job.update(phase='training_reviewed', run=str(path), review=review)
            save(state)
            if review['real_decisions'] != 6144:
                raise ValueError('Training did not complete the matched decision budget; partial data retained.')
            evaluate(state, job, 'greedy')
            evaluate(state, job, 'stochastic')
            job['phase'] = 'completed_reviewed'
            save(state)
        state['phase'] = 'completed_needs_scientific_review'
    except Exception as exc:
        state.update(phase='failed_needs_review', error=repr(exc))
        save(state)
        raise
    state['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save(state)


def launch():
    state = json.loads(STATE.read_text(encoding='utf-8'))
    if state['phase'] != 'prepared':
        raise ValueError('Only a prepared study can start.')
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('Existing Dolphin processes need ownership review.')
    state['phase'] = 'supervisor_pending'
    save(state)
    with (ROOT / 'supervisor.out.log').open('x', encoding='utf-8') as out, (ROOT / 'supervisor.err.log').open('x', encoding='utf-8') as err:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'run'], stdout=out, stderr=err,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                                   | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0))
    print(json.dumps({'supervisor_launcher_pid': process.pid, 'state': str(STATE)}))


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('prepare', 'run', 'launch', 'snapshot'):
        raise SystemExit('Use prepare, run, launch or snapshot.')
    if sys.argv[1] == 'prepare':
        prepare()
    elif sys.argv[1] == 'run':
        state = json.loads(STATE.read_text(encoding='utf-8'))
        if state['phase'] != 'supervisor_pending':
            raise ValueError('Only the pending supervisor can own the study.')
        state['phase'] = 'prepared'
        save(state)
        run()
    elif sys.argv[1] == 'launch':
        launch()
    else:
        print(STATE.read_text(encoding='utf-8'))
