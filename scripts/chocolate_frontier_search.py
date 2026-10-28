"""Bounded scripted search from the first-chocolate assisted savestate.

This is physical exploration, not policy training or a full-start completion.
Each worker owns an isolated Dolphin profile and immutable candidate snapshots.
"""

import argparse
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

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.env import GalaxyEnv
from smg_rl.native import NativeDolphin
from smg_rl.parallel import runtime_directory
from smg_rl.resource_disk import require_disk_headroom
from smg_rl.resource_memory import require_memory_headroom
from smg_rl.train import new_run

os.chdir(Path(__file__).resolve().parents[1])
ROOT = Path('.local/chocolate-frontier-search-20261028')
STATE = ROOT / 'state.json'
STOP = ROOT / 'STOP'
CONFIG = 'config/smg1-six-two-hour.yaml'
ANCHOR = 'sweet-sweet-first-chocolate.state'
WORKERS, TRIALS, HORIZON, SECONDS = 6, 24, 48, 1800


def save(path, data):
    if not atomic_json(path, data):
        raise OSError(f'Cannot persist {path}.')


def schedule(rng):
    """Persistent exclusive motor actions, not simultaneous input combinations."""
    actions = []
    while len(actions) < HORIZON:
        action = int(rng.choice([0, 1, 2, 3, 4, 5, 6, 7], p=[.04, .12, .12, .38, .04, .20, .08, .02]))
        duration = int(rng.integers(1, 5)) if action in (1, 2, 3, 4) else int(rng.integers(1, 3))
        actions.extend([action] * duration)
    return actions[:HORIZON]


def interesting(telemetry, endpoint):
    """Candidate after the opening; does not certify reachable route or stability."""
    x, y, z = telemetry.position
    return bool(not telemetry.dead and not telemetry.complete and telemetry.grounded is True
                and telemetry.health == 3 and np.isfinite([x, y, z]).all()
                and z <= endpoint[2] - 150 and abs(x - endpoint[0]) <= 600
                and abs(y - endpoint[1]) <= 200)


def worker(index, deadline):
    config = load_config(CONFIG)
    config['model_free_run'] = 'evaluation'
    config['replay']['enabled'] = False
    config['paths']['runs'] = str(Path(config['root']) / 'runs/assisted-chocolate-search-20261028')
    config['parallel']['native_capture'] = 'full'
    run = new_run(config, f'worker-{index:02d}')
    config['paths']['user'] = str(runtime_directory(config, run) / 'template')
    summary = {'worker': index, 'state': 'running', 'run': str(run), 'trials': [],
               'physical_decisions': 0, 'candidate_snapshots': 0, 'assisted_ram_stars': 0,
               'scope': 'Scripted assisted start; no gradient, full-start success, or training replay.'}
    save(run / 'summary.json', summary)
    print(json.dumps({'run': str(run), 'worker': index}), flush=True)
    backend = NativeDolphin(config)
    env = None
    try:
        env = GalaxyEnv(config, backend, run / 'worker-00', memory=backend.memory, intrinsic=False, variation=0)
        reset_seed = next(i for i in range(10000)
                          if (item := env.curriculum_bank.choose(np.random.default_rng(i), 1))
                          and item['file'] == ANCHOR)
        rng = np.random.default_rng(9000 + index)
        endpoint = json.loads(Path(config['paths']['route']).read_text(encoding='utf-8'))['waypoints'][-1]
        for trial in range(TRIALS):
            if time.time() >= deadline or STOP.exists():
                break
            actions = schedule(rng)
            _, initial = env.reset(seed=reset_seed, options={'variation': 0})
            if env.start_kind != 'curriculum' or env.start_state_id != ANCHOR:
                raise ValueError('An attempt did not start from the declared verified anchor.')
            actual, candidate = [], None
            info = {}
            for action in actions:
                if time.time() >= deadline or STOP.exists():
                    break
                _, _, terminated, truncated, info = env.step(action)
                actual.append(action)
                summary['physical_decisions'] += 1
                if candidate is None and summary['candidate_snapshots'] < 6 and interesting(env.telemetry, endpoint):
                    destination = run / f'candidate-{trial:03d}.state'
                    backend.save_state(destination)
                    image_path = run / f'candidate-{trial:03d}.png'
                    cv2.imwrite(str(image_path), cv2.cvtColor(backend.bridge.image(), cv2.COLOR_RGBA2BGRA))
                    candidate = {'file': str(destination), 'sha256': digest(destination),
                                 'image': str(image_path), 'position': list(env.telemetry.position),
                                 'step': len(actual), 'scene_frame': env.telemetry.scene_frame,
                                 'actions': actual.copy(), 'grounded_sample': True,
                                 'screen': 'Single sampled grounded candidate; repeat, replay and idle validation required.'}
                    summary['candidate_snapshots'] += 1
                if terminated or truncated:
                    break
            if not actual:
                break
            star = bool(info.get('complete') and not info.get('dead'))
            summary['assisted_ram_stars'] += int(star)
            summary['trials'].append({'trial': trial, 'initial': initial, 'actions': actual,
                'decisions': len(actual), 'outcome': 'star' if star else 'death' if info.get('dead')
                else 'timeout' if terminated or truncated else 'budget_cut', 'candidate': candidate,
                'final': env.telemetry.to_dict()})
            save(run / 'summary.json', summary)
        summary['state'] = 'completed' if len(summary['trials']) == TRIALS else 'deadline_or_stop'
    except BaseException as exc:
        summary.update(state='failed', error=repr(exc))
        raise
    finally:
        (env or backend).close()
        save(run / 'summary.json', summary)
    print(json.dumps({'run': str(run), 'state': summary['state'], 'trials': len(summary['trials']),
                      'physical_decisions': summary['physical_decisions']}), flush=True)


def supervise():
    state = json.loads(STATE.read_text(encoding='utf-8'))
    if state['phase'] != 'supervisor_pending':
        raise ValueError('Only the pending supervisor can dispatch.')
    state.update(phase='running', supervisor={'pid': os.getpid(), 'created_time': psutil.Process().create_time()})
    save(STATE, state)
    deadline = time.time() + SECONDS
    state['deadline_utc'] = datetime.fromtimestamp(deadline, timezone.utc).isoformat()
    save(STATE, state)
    processes = []
    try:
        for i in range(WORKERS):
            if time.time() >= deadline or STOP.exists():
                break
            require_memory_headroom()
            require_disk_headroom(load_config(CONFIG))
            entry = {'worker': i, 'phase': 'pending', 'stdout': str(ROOT / f'worker-{i:02d}.out.log'),
                     'stderr': str(ROOT / f'worker-{i:02d}.err.log')}
            command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--index', str(i), '--deadline', str(deadline)]
            entry['command'] = command
            state['workers'].append(entry)
            save(STATE, state)
            with Path(entry['stdout']).open('x', encoding='utf-8') as out, Path(entry['stderr']).open('x', encoding='utf-8') as err:
                env = dict(os.environ)
                env.update({key: '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
                process = subprocess.Popen(command, stdout=out, stderr=err, env=env,
                                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            entry.update(phase='running', pid=process.pid, created_time=psutil.Process(process.pid).create_time())
            processes.append((process, entry))
            save(STATE, state)
            time.sleep(5)
        for process, entry in processes:
            process.wait()
            entry['exit_code'] = process.returncode
            entry['phase'] = 'completed' if process.returncode == 0 else 'failed'
            save(STATE, state)
        state['phase'] = ('failed_needs_review' if len(processes) != WORKERS or any(p.returncode for p, _ in processes)
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
        raise FileExistsError('A prior search exists; inspect it instead of duplicating it.')
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('An existing Dolphin process needs ownership review.')
    require_memory_headroom()
    require_disk_headroom(load_config(CONFIG))
    ROOT.mkdir(parents=True)
    state = {'phase': 'supervisor_pending', 'created_utc': datetime.now(timezone.utc).isoformat(),
             'workers': [], 'limits': {'workers': WORKERS, 'trials_per_worker': TRIALS,
                                       'decisions_per_trial': HORIZON, 'seconds': SECONDS},
             'scope': 'Scripted assisted-start search only; bounded and never automatically trains a policy.'}
    save(STATE, state)
    with (ROOT / 'supervisor.out.log').open('x', encoding='utf-8') as out, (ROOT / 'supervisor.err.log').open('x', encoding='utf-8') as err:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'supervise'], stdout=out, stderr=err,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                                   | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0))
    print(json.dumps({'state': str(STATE), 'supervisor_launcher_pid': process.pid}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('launch', 'supervise', 'worker', 'snapshot'))
    parser.add_argument('--index', type=int)
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    if args.command == 'launch':
        launch()
    elif args.command == 'supervise':
        supervise()
    elif args.command == 'worker':
        if args.index not in range(WORKERS) or not args.deadline:
            raise ValueError('A worker needs an authorized index and finite deadline.')
        worker(args.index, args.deadline)
    else:
        print(STATE.read_text(encoding='utf-8'))
