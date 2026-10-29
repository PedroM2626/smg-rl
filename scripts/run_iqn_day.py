"""Once-only 24-hour SMG1 IQN continuation with reviewed finite subprocesses."""

import argparse
import json
import os
import socket
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
from smg_rl.replay import replay_signature
from smg_rl.resource_disk import require_disk_headroom
from smg_rl.resource_memory import memory_snapshot
from smg_rl.windows_pool import windows_pool_tags

ROOT = Path('.local/iqn-day-20261029')
STATE = ROOT / 'state.json'
CONFIG = Path('config/iqn-day-smg1-20261029.yaml')
SOURCE_AUDIT = Path('docs/assets/performance/density-four-worker-stop-20261029.json')
SECONDS = 86400
REAL_CAP = 10_000_000
DISPATCH_CAP = 56


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(state):
    state['updated_utc'] = utc()
    if not atomic_json(STATE, state):
        raise OSError('Cannot persist supervisor state; no successor is allowed.')


def block_budget(index, remaining_wall):
    return max(0, min((300 if index == 0 else 900 if index == 1 else 1800),
                      int(remaining_wall) - 180))


def pool_snapshot():
    return {'utc': utc(), 'memory': memory_snapshot(),
            'tags': [x for x in windows_pool_tags() if x['tag'] in ('EtwD', 'Etwr')]}


def pool_growth(before, after):
    def total(snapshot):
        return sum(x['paged_bytes'] + x['nonpaged_bytes'] for x in snapshot['tags'])
    return total(after) - total(before)


def assert_source(checkpoint, expected):
    manifest = read(checkpoint / 'manifest.json')
    for name in ('agent', 'replay'):
        suffix = 'pt' if name == 'agent' else 'npz'
        if digest(checkpoint / f'{name}.{suffix}') != expected[f'{name}_sha256']:
            raise ValueError('Retained source changed; stop without retry.')
    return manifest


def terminate_owned(job):
    """A last-resort bound for an unresponsive owned controller, never a retry."""
    if not alive(job):
        return []
    parent = psutil.Process(job['pid'])
    targets = parent.children(recursive=True) + [parent]
    identities = [{'pid': p.pid, 'created_time': p.create_time()} for p in targets]
    for proc in reversed(targets):
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    _, pending = psutil.wait_procs(targets, timeout=10)
    for proc in pending:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    return identities


def start_tensorboard(state):
    with socket.socket() as connection:
        if connection.connect_ex(('127.0.0.1', 6006)) == 0:
            state['tensorboard'] = {'owned': False, 'url': 'http://127.0.0.1:6006'}
            return None
    command = [sys.executable, '-m', 'tensorboard.main', '--logdir',
               'runs/model-free/tensorboard', '--host', '127.0.0.1', '--port', '6006']
    with (ROOT / 'tensorboard.stdout.log').open('x') as out, (ROOT / 'tensorboard.stderr.log').open('x') as err:
        process = subprocess.Popen(command, stdout=out, stderr=err,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    state['tensorboard'] = {'owned': True, 'pid': process.pid,
                            'created_time': psutil.Process(process.pid).create_time(),
                            'url': 'http://127.0.0.1:6006'}
    return process


def supervise():
    if STATE.exists() or (ROOT / 'STOP').exists():
        raise FileExistsError('Existing ledger or stop: inspect it, never duplicate it.')
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'supervisor-lock.json').open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'created_time': psutil.Process().create_time()}, stream)
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('Existing emulator ownership requires review.')
    config = load_config(CONFIG)
    source_record = read(SOURCE_AUDIT)
    source = Path(source_record['checkpoint'])
    manifest = assert_source(source, {'agent_sha256': source_record['agent_sha256'],
                                     'replay_sha256': read(source / 'manifest.json')['replay_sha256']})
    if (not source_record['agent_hash_ok'] or not source_record['replay_hash_ok']
            or source_record['csv_gap'] != 0 or manifest['signature'] != replay_signature(config)
            or manifest['settings'] != config['dqn'] or config['parallel']['workers'] != 4):
        raise ValueError('Reviewed source or configuration is incompatible.')
    prior = read(source.parent / 'status.json')
    reviewed = audit_block(source.parent, config, prior['source_count'],
                           prior['completed_attempts'] - prior['episodes_this_run'], resource_stopped=True)
    preflight = read(ROOT / 'preflight-result.json')
    if preflight['state'] != 'completed' or preflight['decisions'] < 100:
        raise ValueError('A completed non-learning native preflight is required.')
    start = time.time()
    state = {'schema': 1, 'phase': 'running', 'started_utc': utc(),
             'deadline_utc': datetime.fromtimestamp(start + SECONDS, timezone.utc).isoformat(),
             'report_due_utc': [datetime.fromtimestamp(start + h * 3600, timezone.utc).isoformat()
                                for h in (6, 12, 18, 24)],
             'supervisor': {'pid': os.getpid(), 'created_time': psutil.Process().create_time()},
             'config': str(CONFIG), 'config_sha256': digest(CONFIG),
             'source_audit_sha256': digest(SOURCE_AUDIT), 'source_checkpoint': str(source),
             'source_hashes': {k: manifest[k] for k in ('agent_sha256', 'replay_sha256')},
             'source_real': reviewed['real_decisions'], 'source_attempts': reviewed['completed_attempts'],
             'source_optimizer_updates': source_record['optimizer_updates'],
             'real_decisions': reviewed['real_decisions'], 'completed_attempts': reviewed['completed_attempts'],
             'optimizer_updates': source_record['optimizer_updates'], 'checkpoint': str(source),
             'maximum_seconds': SECONDS, 'new_real_cap': REAL_CAP, 'dispatch_cap': DISPATCH_CAP,
             'maximum_workers': 4, 'accounted_controller_seconds': 0., 'jobs': [],
             'pool_samples': [], 'preflight': preflight,
             'owner_authorization': 'October 29: resume training now, at most24h, report every6h.',
             'method': 'Pure model-free IQN; trainable IMPALA,4 stacked frames,8 exclusive actions. No world model or novelty.',
             'note': 'Same reward/start/control/learner signatures and exact own replay/optimizer/RNG. Audio backend corrected. Collection is not frozen evaluation.'}
    save(state)
    tensorboard = None
    active = None
    try:
        tensorboard = start_tensorboard(state)
        save(state)
        for index in range(DISPATCH_CAP):
            remaining = start + SECONDS - time.time()
            seconds = block_budget(index, remaining)
            decisions = REAL_CAP - (state['real_decisions'] - state['source_real'])
            if seconds < 60 or decisions < 4 or (ROOT / 'STOP').exists():
                break
            if digest(CONFIG) != state['config_sha256']:
                raise ValueError('Immutable run configuration changed.')
            before = pool_snapshot()
            if min(before['memory'][k] for k in ('physical_available_bytes', 'commit_available_bytes')) < 6 * 2**30:
                raise MemoryError('Dispatch requires6GiB physical and commit headroom.')
            require_disk_headroom(config)
            label = f'block-{index:02d}'
            steps = min(3072 if index == 0 else 999996, decisions)
            steps -= steps % 4
            job = {'label': label, 'phase': 'pending_dispatch', 'config': str(CONFIG),
                   'source_real': state['real_decisions'], 'source_attempts': state['completed_attempts'],
                   'resume': state['checkpoint'], 'max_seconds': seconds, 'requested_additional': steps,
                   'stdout': str(ROOT / f'{label}.stdout.log'), 'stderr': str(ROOT / f'{label}.stderr.log'),
                   'pool_before': before, 'dispatch_utc': utc()}
            command = [sys.executable, '-m', 'smg_rl.cli', '--config', str(CONFIG), 'train-dqn',
                       '--steps', str(steps), '--max-seconds', str(seconds), '--deadline-utc',
                       state['deadline_utc'], '--resume', job['resume']]
            job['command'] = command
            state['jobs'].append(job)
            save(state)
            environment = dict(os.environ)
            environment.update({k: '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                                                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
            dispatched = time.time()
            with Path(job['stdout']).open('x') as out, Path(job['stderr']).open('x') as err:
                active = subprocess.Popen(command, stdout=out, stderr=err, env=environment,
                                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            job.update(phase='running', pid=active.pid, created_time=psutil.Process(active.pid).create_time())
            save(state)
            last_publish = 0
            while active.poll() is None:
                run = locate_run(job)
                if run:
                    job['run'] = str(run)
                    if (ROOT / 'STOP').exists() or time.time() - dispatched > seconds + 180:
                        (run / 'STOP').touch(exist_ok=True)
                if time.time() - dispatched > seconds + 240:
                    job['hard_timeout_terminated'] = terminate_owned(job)
                    raise TimeoutError('Owned subprocess exceeded its cleanup bound; no retry.')
                if time.time() - last_publish >= 30:
                    if run and (run / 'status.json').exists():
                        status = read(run / 'status.json')
                        job['latest_status'] = {k: status.get(k) for k in (
                            'state', 'updated_at', 'real_decisions', 'optimizer_updates', 'completed_attempts',
                            'deaths_this_run', 'nonfatal_hp_lost_this_run', 'full_start_stars_this_run',
                            'epsilon', 'losses', 'decisions_per_second', 'stop_reason')}
                    save(state)
                    last_publish = time.time()
                time.sleep(3)
            job.update(exit_code=active.returncode, finished_utc=utc(),
                       controller_wall_seconds=time.time() - dispatched)
            state['accounted_controller_seconds'] += job['controller_wall_seconds']
            run = locate_run(job)
            save(state)
            if active.returncode or run is None:
                raise RuntimeError('Failed subprocess; preserve partial evidence, no retry.')
            status = read(run / 'status.json')
            stopped = status['state'] == 'stopped'
            resource = stopped and status['stop_reason'] == 'memory_pressure'
            paused = stopped and status['stop_reason'] == 'owner_stop'
            review = audit_block(run, config, job['source_real'], job['source_attempts'],
                                 resource_stopped=resource, paused=paused)
            after = pool_snapshot()
            job.update(phase='completed_reviewed' if not stopped else 'partial_reviewed',
                       review=review, pool_after=after, etw_growth_bytes=pool_growth(before, after))
            state.update(checkpoint=review['checkpoint'], real_decisions=review['real_decisions'],
                         completed_attempts=review['completed_attempts'], optimizer_updates=status['optimizer_updates'])
            state['pool_samples'].append(after)
            save(state)
            if stopped:
                raise RuntimeError(f'Collection stopped: {status["stop_reason"]}; no automatic retry.')
            if job['etw_growth_bytes'] > 64 * 2**20:
                raise MemoryError('ETW grew by more than64MiB in one block; diagnose before resuming.')
            if index == 0 and review['real_decisions'] - job['source_real'] != 3072:
                raise RuntimeError('Health ended before its full decision budget; review required.')
        state['phase'] = 'owner_stopped' if (ROOT / 'STOP').exists() else 'completed_needs_review'
    except BaseException as exc:
        state.update(phase='needs_review', error=repr(exc))
        raise
    finally:
        if active is not None and active.poll() is None:
            state['emergency_cleanup'] = terminate_owned(state['jobs'][-1])
        if tensorboard is not None and tensorboard.poll() is None:
            tensorboard.terminate()
            try:
                tensorboard.wait(timeout=10)
            except subprocess.TimeoutExpired:
                tensorboard.kill()
        state['retained_source_manifest'] = assert_source(source, state['source_hashes'])
        state['finished_utc'] = utc()
        save(state)


def snapshot():
    state = read(STATE)
    job = state['jobs'][-1] if state['jobs'] else {}
    run = locate_run(job) if job else None
    status = read(run / 'status.json') if run and (run / 'status.json').exists() else {}
    result = {k: state.get(k) for k in ('phase', 'started_utc', 'deadline_utc', 'report_due_utc',
              'checkpoint', 'source_real', 'source_attempts', 'real_decisions', 'completed_attempts',
              'optimizer_updates', 'accounted_controller_seconds', 'error', 'tensorboard')}
    result.update(supervisor_alive=alive(state['supervisor']), jobs=len(state['jobs']),
                  latest_job={k: job.get(k) for k in ('label', 'phase', 'pid', 'created_time', 'run', 'max_seconds')},
                  controller_alive=alive(job) if job.get('pid') else False, memory=pool_snapshot(),
                  live_status={k: status.get(k) for k in ('state', 'updated_at', 'real_decisions',
                  'optimizer_updates', 'completed_attempts', 'episodes_this_run', 'decisions_per_second',
                  'full_start_stars_this_run', 'deaths_this_run', 'nonfatal_hp_lost_this_run', 'losses',
                  'stop_reason', 'epsilon', 'visual_encoder_first_update_probe')})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('supervise', 'snapshot'))
    args = parser.parse_args()
    supervise() if args.mode == 'supervise' else snapshot()
