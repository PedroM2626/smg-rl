"""Run one bounded SMG1 human-route continuation with reviewed blocks."""

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
from smg_rl.dqn_trial import audit_block, locate_run
from smg_rl.human_route_audit import episode_landmark_candidates, load_spec
from smg_rl.resource_disk import require_disk_headroom
from smg_rl.resource_memory import require_memory_headroom

ROOT = Path('.local/human-route-overnight-20261028')
STATE = ROOT / 'state.json'
STOP = ROOT / 'STOP'
SOURCE_STATE = Path('.local/reward-hour-recovery2-20261028/state.json')
LANDMARKS = Path('config/evaluation/smg1-human-landmarks-v1.json')
DEADLINE = datetime.fromisoformat('2026-10-29T10:00:00+00:00')


def save(state):
    if not atomic_json(STATE, state):
        raise OSError('Could not atomically record overnight state.')


def remaining_budget(now, deadline=DEADLINE, reserve=120, maximum=1800):
    return max(0, min(maximum, int((deadline - now).total_seconds()) - reserve))


def main():
    if ROOT.exists():
        raise FileExistsError('An existing overnight ledger must not be repeated.')
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('Existing Dolphin processes require ownership review.')
    original = json.loads(SOURCE_STATE.read_text(encoding='utf-8'))
    if original['phase'] != 'completed_reviewed':
        raise ValueError('The source recovery has not passed review.')
    arm = original['arms']['candidate']
    config_path, checkpoint = Path(arm['config']), Path(arm['current_checkpoint'])
    config = load_config(config_path)
    prior = json.loads((checkpoint.parent / 'status.json').read_text(encoding='utf-8'))
    reviewed = audit_block(checkpoint.parent, config, prior['source_count'],
                           prior['completed_attempts'] - prior['episodes_this_run'])
    if (Path(reviewed['checkpoint']).resolve() != checkpoint.resolve()
            or reviewed['real_decisions'] != arm['real_decisions']
            or reviewed['completed_attempts'] != arm['completed_attempts']):
        raise ValueError('The reviewed source checkpoint does not match its ledger.')
    specification = load_spec(LANDMARKS)
    state = {'phase': 'running', 'supervisor': {'pid': os.getpid(),
             'created_time': psutil.Process().create_time()}, 'deadline_utc': DEADLINE.isoformat(),
             'started_utc': datetime.now(timezone.utc).isoformat(), 'maximum_emulators': 6,
             'config': str(config_path), 'config_sha256': digest(config_path),
             'landmarks_sha256': digest(LANDMARKS), 'source_review': reviewed,
             'source_checkpoint': str(checkpoint), 'source_real': arm['real_decisions'],
             'source_attempts': arm['completed_attempts'], 'checkpoint': str(checkpoint),
             'real_decisions': arm['real_decisions'], 'completed_attempts': arm['completed_attempts'],
             'jobs': [], 'note': 'Candidate landmark proximity is not verified mission progress; collection is not frozen evaluation.'}
    ROOT.mkdir(parents=True)
    save(state)
    try:
        for index in range(24):
            seconds = remaining_budget(datetime.now(timezone.utc), maximum=300 if index == 0 else 1800)
            if seconds < 60 or STOP.exists():
                break
            if digest(config_path) != state['config_sha256'] or digest(LANDMARKS) != state['landmarks_sha256']:
                raise ValueError('The configuration or evaluation landmark specification changed.')
            mem = require_memory_headroom()
            if mem['physical_available_bytes'] < 6 * 2**30:
                raise MemoryError('Available physical RAM is below the six-worker dispatch guard.')
            require_disk_headroom(config)
            job = {'label': f'block-{index:02d}', 'phase': 'pending_dispatch', 'config': str(config_path),
                   'source_real': state['real_decisions'], 'source_attempts': state['completed_attempts'],
                   'resume': state['checkpoint'], 'max_seconds': seconds,
                   'stdout': str(ROOT / f'block-{index:02d}.stdout.log'),
                   'stderr': str(ROOT / f'block-{index:02d}.stderr.log')}
            job['command'] = [sys.executable, '-m', 'smg_rl.cli', '--config', str(config_path), 'train-dqn',
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
                run = locate_run(job)
                if run and job.get('run') != str(run):
                    job['run'] = str(run)
                    save(state)
                if STOP.exists() and run:
                    (run / 'STOP').touch()
                time.sleep(3)
            job['exit_code'] = process.returncode
            save(state)
            if process.returncode or STOP.exists():
                raise RuntimeError('Stopped or failed block; preserve partial artifacts for review.')
            run = locate_run(job)
            if run is None:
                raise ValueError('The block did not publish an owned run.')
            review = audit_block(run, config, job['source_real'], job['source_attempts'])
            observations, excluded = [], []
            for path in run.glob('worker-*/*/episode-*.csv'):
                try:
                    result = episode_landmark_candidates(path, specification)
                    if result['candidate_landmarks'] or result['native_star']:
                        observations.append({'trajectory': str(path), **result})
                except (KeyError, ValueError) as exc:
                    excluded.append({'trajectory': str(path), 'reason': str(exc)})
            job.update(phase='completed_reviewed', run=str(run), review=review,
                       landmark_candidates=observations, landmark_exclusions=excluded)
            state.update(checkpoint=review['checkpoint'], real_decisions=review['real_decisions'],
                         completed_attempts=review['completed_attempts'])
            save(state)
        state['phase'] = 'stopped' if STOP.exists() else 'completed_needs_scientific_review'
    except BaseException as exc:
        state.update(phase='needs_review', error=repr(exc))
        raise
    finally:
        state['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(state)


if __name__ == '__main__':
    main()
