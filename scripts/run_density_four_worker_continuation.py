"""Run a bounded, guarded four-worker continuation of the reviewed density policy."""

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
from smg_rl.resource_memory import memory_snapshot
from smg_rl.windows_pool import windows_pool_tags

ROOT = Path('.local/density-four-worker-20261029')
STATE = ROOT / 'state.json'
STOP = ROOT / 'STOP'
REVIEW = Path('.local/density-frozen-review-20261029/state.json')
LANDMARKS = Path('config/evaluation/smg1-human-landmarks-v1.json')


def save(state):
    if not atomic_json(STATE, state):
        raise OSError('Cannot persist continuation state.')


def pool_snapshot():
    return {'utc': datetime.now(timezone.utc).isoformat(), 'memory': memory_snapshot(),
            'pool_tags': [x for x in windows_pool_tags() if x['tag'] in ('EtwD', 'Etwr')]}


def segment_seconds(elapsed, remaining_wall, cap=7200, maximum=600, reserve=120):
    return max(0, min(maximum, int(cap - elapsed) - reserve, int(remaining_wall) - reserve))


def main():
    if STATE.exists() or STOP.exists():
        raise FileExistsError('A continuation ledger or stop exists; never repeat it.')
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('Existing Dolphin processes require ownership review.')
    review = json.loads(REVIEW.read_text(encoding='utf-8'))
    if review['phase'] != 'completed_needs_scientific_review' or any(x['phase'] != 'reviewed' for x in review['jobs']):
        raise ValueError('Both frozen checkpoints require complete reviewed episodes.')
    source = review['arms']['candidate']
    migration = json.loads((ROOT / 'migrated' / 'worker-migration.json').read_text(encoding='utf-8'))
    if (migration['target_workers'] != 4 or migration['source_workers'] != 6
            or not migration['learner_optimizer_rng_and_counters_equal']
            or not migration['all_retained_frames_links_and_transitions_equal']
            or migration['source_agent_sha256'] != source['agent_sha256']):
        raise ValueError('Four-worker migration failed the immutable-source review.')
    source_config = load_config(source['config'])
    raw = yaml.safe_load(Path(source['config']).read_text(encoding='utf-8'))
    raw['parallel']['workers'] = 4
    raw['paths'].update(runs='runs/model-free/density-four-worker-20261029',
                        replay='runs/model-free/density-four-worker-20261029/replay',
                        user=str(ROOT / 'template'))
    config_path = Path('.local/density-four-worker-config-20261029.yaml')
    with config_path.open('x', encoding='utf-8') as stream:
        stream.write(yaml.safe_dump(raw, sort_keys=False))
    config = load_config(config_path)
    if replay_signature(config) != replay_signature(source_config) or config['dqn'] != source_config['dqn']:
        raise ValueError('Four-worker profile changed learning or reward signature.')
    manifest = json.loads((ROOT / 'migrated' / 'manifest.json').read_text(encoding='utf-8'))
    if (digest(ROOT / 'migrated' / 'agent.pt') != manifest['agent_sha256']
            or digest(ROOT / 'migrated' / 'replay.npz') != manifest['replay_sha256']):
        raise ValueError('Migrated checkpoint integrity failed.')
    start = time.time()
    deadline = datetime.fromtimestamp(start + 10800, timezone.utc).isoformat()
    state = {'phase': 'running', 'supervisor': {'pid': os.getpid(), 'created_time': psutil.Process().create_time()},
             'started_utc': datetime.now(timezone.utc).isoformat(), 'deadline_utc': deadline,
             'budget_seconds': 7200, 'maximum_workers': 4, 'config': str(config_path),
             'config_sha256': digest(config_path), 'landmarks_sha256': digest(LANDMARKS),
             'frozen_review_sha256': digest(REVIEW), 'migration': migration, 'checkpoint': str((ROOT / 'migrated').resolve()),
             'source_real': migration['real_decisions'], 'real_decisions': migration['real_decisions'],
             'source_attempts': migration['completed_attempts'], 'completed_attempts': migration['completed_attempts'],
             'elapsed_training_seconds': 0., 'jobs': [], 'pool_samples': [pool_snapshot()],
             'note': 'One bounded 2h candidate continuation, no reward or optimizer reset; four workers.'}
    save(state)
    spec = load_spec(LANDMARKS)
    try:
        for index in range(20):
            maximum = 300 if index == 0 else 600
            seconds = segment_seconds(state['elapsed_training_seconds'], start + 10800 - time.time(), maximum=maximum)
            if seconds < 60 or STOP.exists():
                break
            if digest(config_path) != state['config_sha256'] or digest(LANDMARKS) != state['landmarks_sha256']:
                raise ValueError('Continuation profile or landmarks changed.')
            snap = pool_snapshot()
            state['pool_samples'].append(snap)
            if snap['memory']['physical_available_bytes'] < 6 * 2**30 or snap['memory']['commit_available_bytes'] < 6 * 2**30:
                raise MemoryError('Insufficient memory for a four-worker dispatch.')
            require_disk_headroom(config)
            label = f'block-{index:02d}'
            job = {'label': label, 'phase': 'pending_dispatch', 'config': str(config_path),
                   'source_real': state['real_decisions'], 'source_attempts': state['completed_attempts'],
                   'resume': state['checkpoint'], 'max_seconds': seconds,
                   'stdout': str(ROOT / f'{label}.stdout.log'), 'stderr': str(ROOT / f'{label}.stderr.log')}
            job['command'] = [sys.executable, '-m', 'smg_rl.cli', '--config', str(config_path), 'train-dqn',
                              '--steps', '999996', '--max-seconds', str(seconds), '--deadline-utc', deadline,
                              '--resume', job['resume']]
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
                raise RuntimeError('Stopped or failed block; retain partial data for review.')
            run = locate_run(job)
            if run is None:
                raise ValueError('A block did not publish its run.')
            status = json.loads((run / 'status.json').read_text(encoding='utf-8'))
            if status['state'] != 'completed':
                raise RuntimeError(f"Block stopped for review: {status['stop_reason']}")
            audited = audit_block(run, config, job['source_real'], job['source_attempts'])
            observations, excluded = [], 0
            for path in run.glob('worker-*/*/episode-*.csv'):
                try:
                    result = episode_landmark_candidates(path, spec)
                    if result['candidate_landmarks'] or result['native_star']:
                        observations.append({'trajectory': str(path), **result})
                except (KeyError, ValueError):
                    excluded += 1
            job.update(phase='completed_reviewed', run=str(run), review=audited,
                       landmark_candidates=observations, landmark_exclusions=excluded)
            state.update(checkpoint=audited['checkpoint'], real_decisions=audited['real_decisions'],
                         completed_attempts=audited['completed_attempts'],
                         elapsed_training_seconds=state['elapsed_training_seconds'] + audited['elapsed_seconds'])
            state['pool_samples'].append(pool_snapshot())
            save(state)
            if state['elapsed_training_seconds'] > state['budget_seconds']:
                raise ValueError('Continuation exceeded its accounted wall budget.')
        state['phase'] = 'stopped' if STOP.exists() else 'completed_needs_scientific_review'
    except BaseException as exc:
        state.update(phase='needs_review', error=repr(exc))
        raise
    finally:
        state['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(state)


if __name__ == '__main__':
    main()
