"""Resume the audited memory-stopped density comparison without duplicate budget."""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import yaml

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.dqn_trial import audit_block, locate_run
from smg_rl.human_route_audit import episode_landmark_candidates, load_spec
from smg_rl.replay import replay_signature
from smg_rl.resource_disk import require_disk_headroom
from smg_rl.resource_memory import require_memory_headroom

ROOT = Path('.local/human-route-density-recovery-20261028')
STATE = ROOT / 'state.json'
STOP = ROOT / 'STOP'
PREVIOUS = Path('.local/human-route-density-pair-20261028/state.json')
LANDMARKS = Path('config/evaluation/smg1-human-landmarks-v1.json')
DEADLINE = datetime.fromisoformat('2026-10-29T10:00:00+00:00')


def save(state):
    if not atomic_json(STATE, state):
        raise OSError('Recovery ledger could not be written.')


def segment_seconds(now, elapsed, cap=17100, maximum=600, reserve=120):
    return max(0, min(maximum, int(cap - elapsed) - reserve,
                      int((DEADLINE - now).total_seconds()) - reserve))


def main():
    if ROOT.exists():
        raise FileExistsError('Recovery already exists; do not repeat any block.')
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise RuntimeError('Existing Dolphin processes require ownership review.')
    old = json.loads(PREVIOUS.read_text(encoding='utf-8'))
    if old['phase'] != 'needs_review' or old.get('error') != "ValueError('Only a clean, stopped block can authorize a successor.')":
        raise ValueError('The previous density comparison is not the identified memory stop.')
    job = old['jobs'][-1]
    run = Path(job['run'])
    status = json.loads((run / 'status.json').read_text(encoding='utf-8'))
    if job['arm'] != 'candidate' or status['stop_reason'] != 'memory_pressure' or job['exit_code'] != 0:
        raise ValueError('The previous partial collection is not a clean memory stop.')
    review = audit_block(run, load_config(job['config']), job['source_real'], job['source_attempts'],
                         resource_stopped=True)
    arms = old['arms']
    candidate = arms['candidate']
    if candidate['real_decisions'] != job['source_real'] or candidate['completed_attempts'] != job['source_attempts']:
        raise ValueError('The preceding reviewed checkpoint does not reconcile with the partial block.')
    candidate.update(checkpoint=review['checkpoint'], real_decisions=review['real_decisions'],
                     completed_attempts=review['completed_attempts'],
                     elapsed_seconds=candidate['elapsed_seconds'] + review['elapsed_seconds'])
    ROOT.mkdir(parents=True)
    state = {'phase': 'preparing', 'started_utc': datetime.now(timezone.utc).isoformat(),
             'supervisor': {'pid': os.getpid(), 'created_time': psutil.Process().create_time()},
             'previous_state': str(PREVIOUS), 'previous_state_sha256': digest(PREVIOUS),
             'partial_review': review, 'deadline_utc': DEADLINE.isoformat(),
             'seconds_per_arm': old['seconds_per_arm'], 'landmarks_sha256': digest(LANDMARKS),
             'arms': arms, 'jobs': [], 'maximum_emulators': 6,
             'note': 'No duplicate partial time or decisions; equal reward signatures; full-start resets at boundaries.'}
    save(state)
    try:
        for name, arm in arms.items():
            old_config = load_config(arm['config'])
            if digest(Path(arm['config'])) != arm['config_sha256']:
                raise ValueError('Original arm profile changed.')
            raw = yaml.safe_load(Path(arm['config']).read_text(encoding='utf-8'))
            raw['resource_guard'] = {'physical_reserve_bytes': 2 * 2**30}
            path = Path(f'.local/human-route-density-recovery-{name}-20261028.yaml')
            with path.open('x', encoding='utf-8') as stream:
                stream.write(yaml.safe_dump(raw, sort_keys=False))
            new_config = load_config(path)
            if replay_signature(new_config) != replay_signature(old_config) or new_config['dqn'] != old_config['dqn']:
                raise ValueError('Recovery profile changes the learning or reward signature.')
            manifest = json.loads((Path(arm['checkpoint']) / 'manifest.json').read_text(encoding='utf-8'))
            if (manifest['signature'] != replay_signature(new_config)
                    or digest(Path(arm['checkpoint']) / 'agent.pt') != manifest['agent_sha256']
                    or digest(Path(arm['checkpoint']) / 'replay.npz') != manifest['replay_sha256']):
                raise ValueError('Recovery checkpoint integrity or signature failed.')
            arm.update(previous_config=arm['config'], config=str(path), config_sha256=digest(path))
            save(state)
        state['phase'] = 'running'
        save(state)
        spec = load_spec(LANDMARKS)
        for index in range(160):
            if STOP.exists():
                break
            eligible = [k for k, a in arms.items()
                        if segment_seconds(datetime.now(timezone.utc), a['elapsed_seconds'], state['seconds_per_arm']) >= 60]
            if not eligible:
                break
            arm_name = min(eligible, key=lambda name: arms[name]['elapsed_seconds'])
            arm = arms[arm_name]
            maximum = 300 if not any(j['arm'] == arm_name for j in state['jobs']) else 600
            seconds = segment_seconds(datetime.now(timezone.utc), arm['elapsed_seconds'], state['seconds_per_arm'], maximum)
            if digest(Path(arm['config'])) != arm['config_sha256'] or digest(LANDMARKS) != state['landmarks_sha256']:
                raise ValueError('Configuration or landmarks changed during recovery.')
            mem = require_memory_headroom()
            if mem['physical_available_bytes'] < 6 * 2**30:
                raise MemoryError('Six-worker recovery needs at least six GiB available before dispatch.')
            config = load_config(arm['config'])
            require_disk_headroom(config)
            label = f'{arm_name}-{index:03d}'
            job = {'label': label, 'arm': arm_name, 'phase': 'pending_dispatch', 'config': arm['config'],
                   'source_real': arm['real_decisions'], 'source_attempts': arm['completed_attempts'],
                   'resume': arm['checkpoint'], 'max_seconds': seconds,
                   'stdout': str(ROOT / f'{label}.stdout.log'), 'stderr': str(ROOT / f'{label}.stderr.log')}
            job['command'] = [sys.executable, '-m', 'smg_rl.cli', '--config', arm['config'], 'train-dqn',
                              '--steps', '999996', '--max-seconds', str(seconds), '--deadline-utc',
                              DEADLINE.isoformat(), '--resume', job['resume']]
            state['jobs'].append(job)
            save(state)
            env = dict(os.environ)
            env.update({k: '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
            with open(job['stdout'], 'x', encoding='utf-8') as out, open(job['stderr'], 'x', encoding='utf-8') as err:
                process = subprocess.Popen(job['command'], stdout=out, stderr=err, env=env,
                                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            job.update(phase='running', pid=process.pid, created_time=psutil.Process(process.pid).create_time())
            save(state)
            while process.poll() is None:
                owned = locate_run(job)
                if owned and job.get('run') != str(owned):
                    job['run'] = str(owned)
                    save(state)
                if STOP.exists() and owned:
                    (owned / 'STOP').touch()
                time.sleep(3)
            job['exit_code'] = process.returncode
            save(state)
            if process.returncode or STOP.exists():
                raise RuntimeError('Stopped or failed block; preserve partial artifacts for review.')
            owned = locate_run(job)
            if owned is None:
                raise ValueError('The process did not publish its owned run.')
            latest = json.loads((owned / 'status.json').read_text(encoding='utf-8'))
            if latest['state'] != 'completed':
                raise RuntimeError(f"Recovery stopped for review: {latest['stop_reason']}")
            review = audit_block(owned, config, job['source_real'], job['source_attempts'])
            observations, excluded = [], 0
            for path in owned.glob('worker-*/*/episode-*.csv'):
                try:
                    result = episode_landmark_candidates(path, spec)
                    if result['candidate_landmarks'] or result['native_star']:
                        observations.append({'trajectory': str(path), **result})
                except (KeyError, ValueError):
                    excluded += 1
            job.update(phase='completed_reviewed', run=str(owned), review=review,
                       landmark_candidates=observations, landmark_exclusions=excluded)
            arm.update(checkpoint=review['checkpoint'], real_decisions=review['real_decisions'],
                       completed_attempts=review['completed_attempts'],
                       elapsed_seconds=arm['elapsed_seconds'] + review['elapsed_seconds'])
            save(state)
            if arm['elapsed_seconds'] > state['seconds_per_arm']:
                raise ValueError('A branch exceeded its original accounted time cap.')
        state['phase'] = 'stopped' if STOP.exists() else 'completed_needs_scientific_review'
    except BaseException as exc:
        state.update(phase='needs_review', error=repr(exc))
        raise
    finally:
        state['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(state)


if __name__ == '__main__':
    main()
