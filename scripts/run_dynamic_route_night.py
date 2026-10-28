"""Bounded canonical-start IQN trial with reviewed, versioned route extensions.

One local supervisor owns dispatch; it never retries a partial block. All physical
episodes and signed checkpoints remain on disk. See docs/dynamic-route-night.md.
"""

import json
import os
import shutil
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
from smg_rl.resource_memory import require_memory_headroom
from smg_rl.route import RouteProgress
from smg_rl.route_discovery import canonical_bytes, discover, extended_spec, spec_sha256
from smg_rl.route_transfer import transfer_policy

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
STATE = Path('.local/dynamic-route-night-20261028.json')
STOP = Path('.local/dynamic-route-night-20261028.STOP')
DIRECTORY = Path('.local/auto-route-smg1-20261028')
SOURCE_CONFIG = Path('config/smg1-six-two-hour.yaml')
SOURCE = Path('runs/model-free/smg1-six-two-hour/dqn-20261027-114555-611503/checkpoint-final')
DEADLINE = datetime.fromisoformat('2026-10-28T11:00:00+00:00')
EXPECTED_SOURCE = '4b13d0734f09d7f0b182eff3a6ae5fdeb4a094524b83e4ef447f2310f5ce5816'


def save(state):
    if not atomic_json(STATE, state):
        raise OSError('Could not atomically publish trial state.')


def write_new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(content)


def resources():
    if shutil.disk_usage(ROOT).free < 8 * 2**30:
        raise OSError('Less than 8 GiB of free space remains.')
    return require_memory_headroom()


def version_config(index, spec):
    route = DIRECTORY / f'route-v{index:04d}.json'
    config_path = Path(f'.local/dynamic-route-smg1-v{index:04d}.yaml')
    write_new(route, canonical_bytes(spec))
    raw = yaml.safe_load(SOURCE_CONFIG.read_text(encoding='utf-8'))
    raw['paths'].update(route=str(route), runs='runs/model-free/dynamic-route-smg1-20261028',
                        replay='runs/model-free/dynamic-route-smg1-20261028/replay',
                        user=str(DIRECTORY / 'template'))
    raw['rewards']['revision'] = 'expandable-regional-advance-v1'
    raw['regional_reward'].update(spec_sha256=spec_sha256(spec), normalization_distance=50000.0)
    raw['curriculum']['enabled'] = False
    write_new(config_path, yaml.safe_dump(raw, sort_keys=False).encode('utf-8'))
    return config_path


def setup():
    if STATE.exists() or STOP.exists() or DIRECTORY.exists():
        raise FileExistsError('Trial state, stop marker or version directory already exists.')
    hold = Path('.local/training-hold.json')
    if hold.exists() and json.loads(hold.read_text(encoding='utf-8')).get('hold'):
        raise ValueError('An owner training hold is active.')
    if any((p.info['name'] or '').lower() == 'dolphin.exe'
           for p in psutil.process_iter(['name'])):
        raise RuntimeError('An existing Dolphin process needs ownership review.')
    if not 0 < (DEADLINE - datetime.now(timezone.utc)).total_seconds() <= 12 * 3600:
        raise ValueError('The absolute 08:00 BRT deadline is outside the authorized window.')
    resources()
    source_manifest = json.loads((SOURCE / 'manifest.json').read_text(encoding='utf-8'))
    if source_manifest['agent_sha256'] != EXPECTED_SOURCE:
        raise ValueError('The selected original IQN checkpoint differs from the audited source.')
    spec = json.loads(Path('config/routes/smg1-opening-regional-v3.json').read_text(encoding='utf-8'))
    config_path = version_config(0, spec)
    transfer = transfer_policy(SOURCE, load_config(SOURCE_CONFIG), load_config(config_path),
                               DIRECTORY / 'transfer-v0000')
    state = {'schema': 1, 'phase': 'prepared', 'deadline_utc': DEADLINE.isoformat(),
             'created_utc': datetime.now(timezone.utc).isoformat(),
             'owner_request': 'Six SMG1 workers; reviewed bounded dynamic route experiment; report at 08:00 BRT.',
             'method': 'IQN/IMPALA, canonical starts only, no curiosity or world model.',
             'stagnation_rule': 'Stop after two completed 30-minute main blocks without a repeated valid extension or RAM-confirmed full-start star; health excluded.',
             'limits': {'health_real': 3072, 'health_seconds': 300, 'block_real': 180000,
                        'block_seconds': 1800, 'max_workers': 6, 'minimum_free_bytes': 8 * 2**30},
             'stagnant_blocks': 0, 'new_real_decisions': 0, 'new_completed_attempts': 0,
             'route_versions': [{'index': 0, 'config': str(config_path), 'route': str(DIRECTORY / 'route-v0000.json'),
                                  'sha256': spec_sha256(spec), 'length_xz': RouteProgress(spec).total_length,
                                  'transfer': transfer}], 'blocks': [],
             'checkpoint': transfer['destination'], 'branch_real': 0, 'branch_attempts': 0}
    save(state)
    return state


def run_block(state, seconds, steps):
    resources()
    index = len(state['blocks'])
    version = state['route_versions'][-1]
    job = {'index': index, 'version': version['index'], 'config': version['config'],
           'source': state['checkpoint'], 'source_real': state['branch_real'],
           'source_attempts': state['branch_attempts'], 'requested_real': steps,
           'max_seconds': seconds, 'phase': 'pending',
           'stdout': str(DIRECTORY / f'block-{index:03d}.out.log'),
           'stderr': str(DIRECTORY / f'block-{index:03d}.err.log')}
    command = [sys.executable, '-m', 'smg_rl.cli', '--config', job['config'], 'train-dqn',
               '--steps', str(steps), '--max-seconds', str(seconds),
               '--deadline-utc', DEADLINE.isoformat(), '--resume', job['source']]
    job['command'] = command
    state['blocks'].append(job)
    state['phase'] = 'block_pending'
    save(state)
    with open(job['stdout'], 'x', encoding='utf-8') as out, open(job['stderr'], 'x', encoding='utf-8') as err:
        environment = dict(os.environ)
        environment.update({key: '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                                 'NUMEXPR_NUM_THREADS')})
        process = subprocess.Popen(command, stdout=out, stderr=err, env=environment,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    job.update(phase='running', pid=process.pid, created_time=psutil.Process(process.pid).create_time())
    state['phase'] = 'block_running'
    save(state)
    while process.poll() is None:
        if STOP.exists():
            run = locate_run(job)
            if run:
                (run / 'STOP').touch()
        time.sleep(2)
    job['exit_code'] = process.returncode
    job['run'] = str(locate_run(job) or '')
    job['phase'] = 'stopped_unreviewed'
    state['phase'] = 'block_review'
    save(state)
    if process.returncode != 0 or not job['run']:
        raise RuntimeError('Block failed or did not publish a run; partial data retained without retry.')
    if STOP.exists():
        state['phase'] = 'owner_stopped_unreviewed'
        save(state)
        return None
    run = Path(job['run'])
    review = audit_block(run, load_config(job['config']), job['source_real'], job['source_attempts'])
    status = json.loads((run / 'status.json').read_text(encoding='utf-8'))
    job.update(phase='reviewed', review=review, stop_reason=status['stop_reason'],
               native_full_start_stars=status.get('full_start_stars_this_run', 0))
    state['new_real_decisions'] += review['real_decisions'] - job['source_real']
    state['new_completed_attempts'] += review['completed_attempts'] - job['source_attempts']
    state['branch_real'] = review['real_decisions']
    state['branch_attempts'] = review['completed_attempts']
    state['checkpoint'] = review['checkpoint']
    spec = json.loads(Path(version['route']).read_text(encoding='utf-8'))
    job['extension'] = discover(sorted(run.glob('worker-*/*/episode-*.csv')), spec)
    save(state)
    return job


def promote(state, job):
    previous = state['route_versions'][-1]
    spec = json.loads(Path(previous['route']).read_text(encoding='utf-8'))
    expanded = extended_spec(spec, job['extension'])
    index = previous['index'] + 1
    config_path = version_config(index, expanded)
    transfer = transfer_policy(state['checkpoint'], load_config(previous['config']), load_config(config_path),
                               DIRECTORY / f'transfer-v{index:04d}')
    state['route_versions'].append({'index': index, 'config': str(config_path),
        'route': str(DIRECTORY / f'route-v{index:04d}.json'), 'sha256': spec_sha256(expanded),
        'length_xz': RouteProgress(expanded).total_length, 'evidence': job['extension'], 'transfer': transfer})
    state.update(checkpoint=transfer['destination'], branch_real=0, branch_attempts=0)
    save(state)


def supervise(state):
    state['supervisor'] = {'pid': os.getpid(), 'created_time': psutil.Process().create_time()}
    state['phase'] = 'supervising'
    save(state)
    try:
        while datetime.now(timezone.utc) < DEADLINE and not STOP.exists():
            health = not state['blocks']
            job = run_block(state, 300 if health else 1800, 3072 if health else 180000)
            if job is None:
                return
            if job['native_full_start_stars']:
                state['phase'] = 'completed_star'
                break
            if job['extension']:
                state['stagnant_blocks'] = 0
                if datetime.now(timezone.utc) < DEADLINE:
                    promote(state, job)
            elif not health and job['stop_reason'] == 'wall_clock_budget':
                state['stagnant_blocks'] += 1
            if state['stagnant_blocks'] >= 2:
                state['phase'] = 'stopped_no_validated_extension'
                break
        else:
            state['phase'] = 'owner_stopped' if STOP.exists() else 'completed_deadline'
    except Exception as exc:
        state.update(phase='failed_needs_review', error=repr(exc))
        save(state)
        raise
    state['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save(state)


def launch():
    state = setup()
    out_path = DIRECTORY / 'supervisor.out.log'
    err_path = DIRECTORY / 'supervisor.err.log'
    state['phase'] = 'supervisor_pending'
    save(state)
    with out_path.open('x', encoding='utf-8') as out, err_path.open('x', encoding='utf-8') as err:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'supervise'],
                                   stdout=out, stderr=err,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                                   | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0))
    print(json.dumps({'supervisor_pid': process.pid, 'state': str(STATE), 'deadline_utc': DEADLINE.isoformat()}))


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('launch', 'supervise', 'snapshot'):
        raise SystemExit('Use launch, supervise, or snapshot.')
    if sys.argv[1] == 'launch':
        launch()
    elif sys.argv[1] == 'snapshot':
        print(STATE.read_text(encoding='utf-8'))
    else:
        current = json.loads(STATE.read_text(encoding='utf-8'))
        if current['phase'] != 'supervisor_pending':
            raise ValueError('Supervisor can start only once from its pending state.')
        supervise(current)
