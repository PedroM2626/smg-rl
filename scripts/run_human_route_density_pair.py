"""Run a bounded matched reward-density comparison with reviewed blocks."""

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
from smg_rl.resource_disk import require_disk_headroom
from smg_rl.resource_memory import require_memory_headroom
from smg_rl.route_transfer import transfer_policy

ROOT = Path('.local/human-route-density-pair-20261028')
STATE = ROOT / 'state.json'
STOP = ROOT / 'STOP'
SOURCE_STATE = Path('.local/human-route-overnight-20261028/owner-stop-review.json')
LANDMARKS = Path('config/evaluation/smg1-human-landmarks-v1.json')
DEADLINE = datetime.fromisoformat('2026-10-29T10:00:00+00:00')


def save(state):
    if not atomic_json(STATE, state):
        raise OSError('Could not atomically record overnight state.')


def remaining_budget(now, deadline=DEADLINE, reserve=120, maximum=1800):
    return max(0, min(maximum, int((deadline - now).total_seconds()) - reserve))


def main():
    if ROOT.exists():
        raise FileExistsError('An existing density comparison must never be repeated.')
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('Existing Dolphin processes require ownership review.')
    reviewed = json.loads(SOURCE_STATE.read_text(encoding='utf-8'))
    if not reviewed['passed'] or reviewed.get('resource_stopped_checkpoint') is not False:
        raise ValueError('An audited clean owner-stop source is required.')
    checkpoint = Path(reviewed['checkpoint'])
    old_config_path = Path('.local/reward-hour-pair-candidate-20261028.yaml')
    old_config = load_config(old_config_path)
    if digest(checkpoint / 'agent.pt') != reviewed['agent_sha256']:
        raise ValueError('The source checkpoint changed after review.')
    ROOT.mkdir(parents=True)
    state = {'phase': 'preparing', 'supervisor': {'pid': os.getpid(),
             'created_time': psutil.Process().create_time()}, 'deadline_utc': DEADLINE.isoformat(),
             'started_utc': datetime.now(timezone.utc).isoformat(), 'maximum_emulators': 6,
             'source_review': reviewed, 'source_review_sha256': digest(SOURCE_STATE),
             'seconds_per_arm': 17100, 'landmarks_sha256': digest(LANDMARKS),
             'arms': {}, 'jobs': [],
             'note': 'Matched single training seed; candidate route proximity is not mission progress.'}
    save(state)
    try:
        for arm in ('candidate', 'control'):
            raw = yaml.safe_load(Path('config/human-route-smg1-density-v1.yaml').read_text(encoding='utf-8'))
            if arm == 'control':
                raw['regional_reward'].update(mode='advance', scale=1.)
            raw['rewards']['revision'] = f'human-route-density-pair-{arm}-v1'
            raw['paths'].update(runs=f'runs/model-free/human-route-density-20261028/{arm}',
                                replay=f'runs/model-free/human-route-density-20261028/{arm}/replay',
                                user=str(ROOT / f'{arm}-template'))
            config_path = Path(f'.local/human-route-density-{arm}-20261028.yaml')
            with config_path.open('x', encoding='utf-8') as stream:
                stream.write(yaml.safe_dump(raw, sort_keys=False))
            config = load_config(config_path)
            require_disk_headroom(config)
            transfer = transfer_policy(checkpoint, old_config, config, ROOT / f'transfer-{arm}')
            state['arms'][arm] = {'config': str(config_path), 'config_sha256': digest(config_path),
                                  'transfer': transfer, 'checkpoint': transfer['destination'],
                                  'real_decisions': 0, 'completed_attempts': 0, 'elapsed_seconds': 0.}
            save(state)
        state['phase'] = 'running'
        save(state)
        specification = load_spec(LANDMARKS)
        # Alternate one-hour batches so either early interruption retains matched data.
        for index in range(12):
            arm_name = ('candidate', 'control')[index % 2]
            arm = state['arms'][arm_name]
            batch_remaining = min(3600., state['seconds_per_arm'] - arm['elapsed_seconds'])
            if batch_remaining <= 130 or STOP.exists():
                continue
            for segment in range(4):
                seconds = min(remaining_budget(datetime.now(timezone.utc)),
                              int(batch_remaining) - 120,
                              300 if not any(j['arm'] == arm_name for j in state['jobs']) else 1800)
                if seconds < 60 or STOP.exists():
                    break
                if digest(Path(arm['config'])) != arm['config_sha256'] or digest(LANDMARKS) != state['landmarks_sha256']:
                    raise ValueError('The profile or evaluation landmark specification changed.')
                mem = require_memory_headroom()
                if mem['physical_available_bytes'] < 6 * 2**30:
                    raise MemoryError('Available physical RAM is below the six-worker dispatch guard.')
                config = load_config(arm['config'])
                require_disk_headroom(config)
                label = f"{arm_name}-{index:02d}-{segment:02d}"
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
                observations, excluded = [], 0
                for path in run.glob('worker-*/*/episode-*.csv'):
                    try:
                        result = episode_landmark_candidates(path, specification)
                        if result['candidate_landmarks'] or result['native_star']:
                            observations.append({'trajectory': str(path), **result})
                    except (KeyError, ValueError):
                        excluded += 1
                job.update(phase='completed_reviewed', run=str(run), review=review,
                           landmark_candidates=observations, landmark_exclusions=excluded)
                arm.update(checkpoint=review['checkpoint'], real_decisions=review['real_decisions'],
                           completed_attempts=review['completed_attempts'],
                           elapsed_seconds=arm['elapsed_seconds'] + review['elapsed_seconds'])
                batch_remaining -= review['elapsed_seconds']
                save(state)
                if arm['elapsed_seconds'] > state['seconds_per_arm']:
                    raise ValueError('Arm exceeded its accounted wall-time allocation.')
        state['phase'] = 'stopped' if STOP.exists() else 'completed_needs_scientific_review'
    except BaseException as exc:
        state.update(phase='needs_review', error=repr(exc))
        raise
    finally:
        state['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(state)


if __name__ == '__main__':
    main()
