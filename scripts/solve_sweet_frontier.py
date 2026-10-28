"""Bounded real-game archive search, starting from a physically reproduced partial path.

The search is scripted planning, not a trained policy. Each private search state
is a dynamic exploration fixture, not a validated stable curriculum state.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import psutil

from smg_rl.actions import EIGHT_ACTIONS, advance_action
from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.native import NativeDolphin
from smg_rl.parallel import runtime_directory
from smg_rl.profiles import PROFILES
from smg_rl.resource_disk import require_disk_headroom
from smg_rl.resource_memory import require_memory_headroom
from smg_rl.train import new_run

os.chdir(Path(__file__).resolve().parents[1])
ROOT = Path('.local/sweet-manual-search-stage4-20261028')
STATE = ROOT / 'state.json'
STOP = ROOT / 'STOP'
VALIDATION = Path('runs/sweet-scripted-replay-20261028/replay-20261028-100753-233880/result.json')
WORKERS, TRIALS, HORIZON, SECONDS, MAX_STATES = 6, 48, 24, 1800, 24


def save(path, data):
    if not atomic_json(path, data):
        raise OSError(f'Cannot atomically save {path}.')


def action_schedule(rng):
    """Diverse repeated exclusive controls; no unconfigured simultaneous presses."""
    result = []
    while len(result) < HORIZON:
        action = int(rng.choice(8, p=[.04, .16, .13, .36, .04, .19, .06, .02]))
        duration = int(rng.integers(1, 5)) if action in (1, 2, 3, 4) else int(rng.integers(1, 3))
        result.extend([action] * duration)
    return result[:HORIZON]


def cell(position):
    """Coarse 3D world cell; different cells do not prove course progress."""
    point = np.asarray(position, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('Cannot archive nonfinite game coordinates.')
    return tuple(np.floor(point / np.array([240., 180., 240.])).astype(int).tolist())


def source_candidate():
    validation = json.loads(VALIDATION.read_text(encoding='utf-8'))
    entry = validation['expected']
    if (validation['state'] != 'completed' or not validation['reached_exact_saved_state']
            or validation['full_start_ram_star'] or digest(entry['file']) != entry['sha256']):
        raise ValueError('The previously reproduced source state did not pass integrity checks.')
    return entry


def choose(entries, rng):
    """Favor less-visited cells while keeping earlier alternatives selectable."""
    scores = np.array([1 / np.sqrt(1 + item['visits']) for item in entries], dtype=float)
    return entries[int(rng.choice(len(entries), p=scores / scores.sum()))]


def worker(index, deadline):
    source = source_candidate()
    config = load_config('config/smg1-six-two-hour.yaml')
    config['paths']['runs'] = str(Path(config['root']) / 'runs/sweet-manual-search-stage4-20261028')
    config['parallel']['native_capture'] = 'full'
    config['replay']['enabled'] = False
    run = new_run(config, f'worker-{index:02d}')
    config['paths']['user'] = str(runtime_directory(config, run) / 'template')
    result = {'worker': index, 'state': 'running', 'run': str(run), 'real_decisions': 0,
              'trials': [], 'archive': [], 'stars': [], 'source_state_sha256': source['sha256'],
              'scope': 'Scripted, assisted dynamic-state search; no model learning or full-start success claim.'}
    save(run / 'summary.json', result)
    print(json.dumps({'run': str(run), 'worker': index}), flush=True)
    backend = NativeDolphin(config)
    try:
        backend.load_state(source['file'])
        telemetry = backend.memory.sample()
        if (telemetry.dead or telemetry.complete or telemetry.grounded is not True
                or telemetry.health != 3 or telemetry.scene_frame != source['scene_frame']
                or np.linalg.norm(np.subtract(telemetry.position, source['position'])) > .01):
            raise ValueError('Source state does not reproduce the documented physical context.')
        result['archive'].append({'file': source['file'], 'sha256': source['sha256'],
            'position': list(telemetry.position), 'scene_frame': telemetry.scene_frame,
            'cell': list(cell(telemetry.position)), 'visits': 0,
            'full_start_actions': source['full_start_actions'],
            'origin': 'previously reproduced, nonstationary dynamic search state'})
        rng = np.random.default_rng(18400 + index)
        known = {cell(telemetry.position)}
        for trial in range(TRIALS):
            if time.time() >= deadline or STOP.exists():
                break
            parent = choose(result['archive'], rng)
            parent['visits'] += 1
            if digest(parent['file']) != parent['sha256']:
                raise ValueError('Archive state hash changed.')
            backend.load_state(parent['file'])
            backend._motor_direction = ()
            before = backend.memory.sample()
            if (before.dead or before.complete or before.scene_frame != parent['scene_frame']
                    or np.linalg.norm(np.subtract(before.position, parent['position'])) > .01):
                raise ValueError('Archive restore context did not match.')
            actions, trace = [], []
            for action in action_schedule(rng):
                if time.time() >= deadline or STOP.exists():
                    break
                advance_action(backend, EIGHT_ACTIONS[action], config)
                telemetry = backend.memory.sample()
                actions.append(action)
                result['real_decisions'] += 1
                trace.append({'action': action, 'position': list(telemetry.position), 'health': telemetry.health,
                              'grounded': telemetry.grounded, 'dead': telemetry.dead, 'complete': telemetry.complete,
                              'scene_frame': telemetry.scene_frame, 'scene_nerve': telemetry.scene_nerve})
                if telemetry.complete and not telemetry.dead:
                    if telemetry.scene_nerve not in PROFILES['RMGE01'].complete:
                        raise ValueError('Completion signal is not a validated native RAM terminal.')
                    star = {'trial': trial, 'assisted': True, 'full_start_actions': parent['full_start_actions'] + actions,
                            'terminal': telemetry.to_dict(), 'worker': index}
                    result['stars'].append(star)
                    save(run / 'summary.json', result)
                    STOP.touch()
                    break
                if telemetry.dead:
                    break
                current_cell = cell(telemetry.position)
                if (telemetry.grounded is True and telemetry.health is not None and telemetry.health > 0
                        and current_cell not in known and len(result['archive']) < MAX_STATES):
                    path = run / f'state-{len(result["archive"]):03d}.state'
                    backend.save_state(path)
                    image_path = path.with_suffix('.png')
                    cv2.imwrite(str(image_path), cv2.cvtColor(backend.bridge.image(), cv2.COLOR_RGBA2BGRA))
                    result['archive'].append({'file': str(path), 'sha256': digest(path), 'image': str(image_path),
                        'position': list(telemetry.position), 'scene_frame': telemetry.scene_frame,
                        'health': telemetry.health, 'cell': list(current_cell), 'visits': 0,
                        'full_start_actions': parent['full_start_actions'] + actions,
                        'origin': 'dynamic assisted search; not validated for stationary curriculum'})
                    known.add(current_cell)
            result['trials'].append({'trial': trial, 'parent_sha256': parent['sha256'], 'actions': actions,
                                     'trace': trace, 'outcome': 'star' if telemetry.complete and not telemetry.dead
                                     else 'death' if telemetry.dead else 'bounded_cut'})
            save(run / 'summary.json', result)
        result['state'] = 'completed' if len(result['trials']) == TRIALS else 'stopped_or_deadline'
    except BaseException as exc:
        result.update(state='failed', error=repr(exc))
        raise
    finally:
        backend.close()
        save(run / 'summary.json', result)
    print(json.dumps({'run': str(run), 'state': result['state'], 'trials': len(result['trials']),
                      'real_decisions': result['real_decisions'], 'stars': len(result['stars'])}), flush=True)


def supervise():
    state = json.loads(STATE.read_text(encoding='utf-8'))
    if state['phase'] != 'supervisor_pending':
        raise ValueError('Only the pending supervisor can own this search.')
    state.update(phase='running', supervisor={'pid': os.getpid(), 'created_time': psutil.Process().create_time()})
    save(STATE, state)
    deadline = time.time() + SECONDS
    state['deadline_utc'] = datetime.fromtimestamp(deadline, timezone.utc).isoformat()
    save(STATE, state)
    processes = []
    try:
        for index in range(WORKERS):
            if STOP.exists() or time.time() >= deadline:
                break
            require_memory_headroom()
            require_disk_headroom(load_config('config/smg1-six-two-hour.yaml'))
            job = {'worker': index, 'phase': 'pending', 'stdout': str(ROOT / f'worker-{index:02d}.out.log'),
                   'stderr': str(ROOT / f'worker-{index:02d}.err.log')}
            job['command'] = [sys.executable, str(Path(__file__).resolve()), 'worker', str(index), str(deadline)]
            state['workers'].append(job)
            save(STATE, state)
            with Path(job['stdout']).open('x', encoding='utf-8') as out, Path(job['stderr']).open('x', encoding='utf-8') as err:
                env = dict(os.environ)
                env.update({key: '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
                process = subprocess.Popen(job['command'], stdout=out, stderr=err, env=env,
                                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            job.update(phase='running', pid=process.pid, created_time=psutil.Process(process.pid).create_time())
            processes.append((process, job))
            save(STATE, state)
            time.sleep(5)
        for process, job in processes:
            process.wait()
            job.update(phase='completed' if process.returncode == 0 else 'failed', exit_code=process.returncode)
            save(STATE, state)
        state['phase'] = ('failed_needs_review' if any(p.returncode for p, _ in processes)
                          else 'completed_needs_review')
    except BaseException as exc:
        STOP.touch()
        state.update(phase='failed_needs_review', error=repr(exc))
        save(STATE, state)
        for process, _ in processes:
            process.wait(timeout=120)
        raise
    state['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save(STATE, state)


def launch():
    if ROOT.exists():
        raise FileExistsError('An existing search requires review; never duplicate it.')
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('Existing Dolphin processes require ownership review.')
    source_candidate()
    require_memory_headroom()
    require_disk_headroom(load_config('config/smg1-six-two-hour.yaml'))
    ROOT.mkdir(parents=True)
    state = {'phase': 'supervisor_pending', 'created_utc': datetime.now(timezone.utc).isoformat(),
             'workers': [], 'limits': {'workers': WORKERS, 'trials_per_worker': TRIALS,
                 'decisions_per_trial': HORIZON, 'seconds': SECONDS, 'states_per_worker': MAX_STATES},
             'scope': 'Bounded scripted search from a previously reproduced nonstationary dynamic state.'}
    save(STATE, state)
    with (ROOT / 'supervisor.out.log').open('x', encoding='utf-8') as out, (ROOT / 'supervisor.err.log').open('x', encoding='utf-8') as err:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'supervise'], stdout=out, stderr=err,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                                   | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0))
    print(json.dumps({'state': str(STATE), 'supervisor_launcher_pid': process.pid}))


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('launch', 'supervise', 'worker', 'snapshot'):
        raise SystemExit('Use launch, supervise, worker INDEX DEADLINE or snapshot.')
    if sys.argv[1] == 'launch':
        launch()
    elif sys.argv[1] == 'supervise':
        supervise()
    elif sys.argv[1] == 'worker':
        if len(sys.argv) != 4 or int(sys.argv[2]) not in range(WORKERS):
            raise ValueError('Worker requires an authorized index and finite deadline.')
        worker(int(sys.argv[2]), float(sys.argv[3]))
    else:
        print(STATE.read_text(encoding='utf-8'))
