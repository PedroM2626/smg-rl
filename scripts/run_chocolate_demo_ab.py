"""Bounded, equal-update IQN comparison with and without real partial demonstrations."""

import ast
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch
import yaml

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.dqn_replay import PrioritizedFrames
from smg_rl.dqn_training import prepare_dqn
from smg_rl.replay import replay_signature
from smg_rl.resource_disk import require_disk_headroom
from smg_rl.resource_memory import require_memory_headroom

os.chdir(Path(__file__).resolve().parents[1])
ROOT = Path('.local/chocolate-demo-ab-20261028')
STATE = ROOT / 'state.json'
STOP = ROOT / 'STOP'
CONFIG = Path('config/smg1-six-two-hour.yaml')
SOURCE = Path('runs/model-free/smg1-six-two-hour/dqn-20261027-114555-611503/checkpoint-final')
CAPTURE = Path('runs/chocolate-demonstrations-20261028/capture-20261028-093321-941291/result.json')
SEEDS, UPDATES = (142, 143), 128


def save(state):
    if not atomic_json(STATE, state):
        raise OSError('Cannot atomically record demonstration study state.')


def demo_replay(config, capture, seed):
    episodes = [episode for episode in capture['episodes'] if episode['purpose'] == 'training']
    if len(episodes) != 2:
        raise ValueError('Exactly two physical full-start demonstration episodes are required.')
    replay = PrioritizedFrames(512, config['environment']['image_size'], stack=config['dqn']['frame_stack'],
        workers=1, n_step=config['dqn']['n_step'], gamma=config['dqn']['gamma'], alpha=config['dqn']['alpha'], seed=seed)
    for episode in episodes:
        path = Path(episode['file'])
        if digest(path) != episode['sha256'] or episode['outcome'] != 'nonterminal_collector_cut':
            raise ValueError('Demonstration provenance, hash or cut invalid.')
        with np.load(path, allow_pickle=False) as data:
            frames, actions, rewards = data['frames'], data['actions'], data['rewards']
        if (len(frames) != len(actions) + 1 or len(actions) != len(rewards)
                or len(actions) != episode['decisions'] or len(actions) < 1 or frames.dtype != np.uint8
                or not np.isfinite(rewards).all() or np.any(actions < 0) or np.any(actions >= 8)):
            raise ValueError('Invalid demonstration frame/action/reward lengths.')
        current = replay.add_frame(frames[0])
        for i, action in enumerate(actions):
            successor = replay.add_frame(frames[i + 1], current)
            replay.append(0, current, int(action), float(rewards[i]), successor, 0,
                          terminal=False, ended=i == len(actions) - 1, assisted=False)
            current = successor
    if replay.emitted != sum(e['decisions'] for e in episodes) or any(replay.pending):
        raise ValueError('Demonstration replay counts do not reconcile.')
    return replay


def prepare():
    if ROOT.exists():
        raise FileExistsError('Existing study must be reviewed, never duplicated.')
    capture = json.loads(CAPTURE.read_text(encoding='utf-8'))
    source_config = load_config(CONFIG)
    if (capture['state'] != 'completed' or capture['signature'] != replay_signature(source_config)
            or len([e for e in capture['episodes'] if e['purpose'] == 'held_out_timing_check']) != 1):
        raise ValueError('Capture is not a completed matching signed demonstration.')
    require_memory_headroom()
    require_disk_headroom(source_config)
    ROOT.mkdir(parents=True)
    raw = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    state = {'phase': 'preparing', 'created_utc': datetime.now(timezone.utc).isoformat(),
             'source': str(SOURCE), 'capture': str(CAPTURE), 'capture_sha256': digest(CAPTURE),
             'design': {'seeds': list(SEEDS), 'updates': UPDATES, 'demo_every_updates': 4,
                        'offline_seconds_per_case': 900,
                        'evaluation': 'One greedy draw and three noisy action draws, canonical start, max600 decisions, epsilon0.1.',
                        'gate': 'For both seeds, noisy mean alive-grounded goal-axis projection improves by >=0.5 percentage points, at least one noisy full-start episode reaches the declared grounded beyond-opening region, and no star regression.',
                        'scope': 'Standard IQN TD updates; same reward and optimizer/update budget. No model, imagination, or auto-longer-run.'}, 'cases': []}
    save(state)
    for seed in SEEDS:
        for arm in ('control', 'demonstration'):
            profile = yaml.safe_load(yaml.safe_dump(raw))
            profile['training']['seed'] = seed
            profile['paths']['runs'] = f'runs/model-free/chocolate-demo-ab-20261028/{arm}-{seed}'
            profile['paths']['user'] = str(ROOT / f'{arm}-{seed}-template')
            config_path = Path(f'.local/chocolate-demo-{arm}-{seed}-20261028.yaml')
            with config_path.open('x', encoding='utf-8') as stream:
                yaml.safe_dump(profile, stream, sort_keys=False)
            if replay_signature(load_config(config_path)) != capture['signature']:
                raise ValueError('Demonstration and study reward/action signatures do not match.')
            state['cases'].append({'arm': arm, 'seed': seed, 'config': str(config_path), 'phase': 'prepared'})
            save(state)
    state['phase'] = 'prepared'
    save(state)
    print(json.dumps({'state': str(STATE), 'cases': len(state['cases'])}))


def fit(state, case, capture):
    cfg = load_config(case['config'])
    require_memory_headroom()
    require_disk_headroom(cfg)
    case['phase'] = 'offline_fitting'
    save(state)
    agent, original = prepare_dqn(cfg, str(SOURCE))
    demo = demo_replay(cfg, capture, case['seed'] + 100000) if case['arm'] == 'demonstration' else None
    original.rng = np.random.default_rng(case['seed'])
    initial_updates, start = agent.updates, time.monotonic()
    last = None
    for update in range(UPDATES):
        if STOP.exists() or time.monotonic() - start >= 900:
            raise TimeoutError('Offline fit interrupted; retain partial evidence, never retry automatically.')
        torch.manual_seed(case['seed'] * 10000 + update)
        selected = demo if demo is not None and update % 4 == 0 else original
        last = agent.update(selected)
    if agent.updates - initial_updates != UPDATES or not all(np.isfinite(v) for v in last.values()):
        raise ValueError('Offline optimizer count or finite-metrics check failed.')
    destination = ROOT / f"{case['arm']}-{case['seed']}-checkpoint"
    agent.save(destination, original)
    case.update(phase='fit_reviewed', checkpoint=str(destination.resolve()),
                source_real_decisions=agent.real_decisions, offline_updates=UPDATES,
                demo_batches=32 if demo is not None else 0,
                demo_physical_decisions=demo.emitted if demo is not None else 0,
                last_metrics=last, checkpoint_agent_sha256=digest(destination / 'agent.pt'),
                checkpoint_replay_sha256=digest(destination / 'replay.npz'))
    save(state)
    del agent, original, demo
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate(state, case, mode):
    label = f"{case['arm']}-{case['seed']}-{mode}"
    record = {'mode': mode, 'phase': 'pending', 'stdout': str(ROOT / f'{label}.out.log'),
              'stderr': str(ROOT / f'{label}.err.log')}
    command = [sys.executable, '-m', 'smg_rl.cli', '--config', case['config'], 'record-dqn',
               case['checkpoint'], '--variation', '0', '--max-steps', '600', '--seeds']
    command += ['11'] if mode == 'greedy' else ['11', '23', '47', '--stochastic', '--evaluation-epsilon', '0.1']
    record['command'] = command
    case.setdefault('evaluations', []).append(record)
    save(state)
    with Path(record['stdout']).open('x', encoding='utf-8') as out, Path(record['stderr']).open('x', encoding='utf-8') as err:
        env = dict(os.environ)
        env.update({key: '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
        process = subprocess.Popen(command, stdout=out, stderr=err, env=env,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    record.update(phase='running', pid=process.pid, created_time=psutil.Process(process.pid).create_time())
    save(state)
    process.wait()
    record['exit_code'] = process.returncode
    save(state)
    if process.returncode != 0 or STOP.exists():
        raise RuntimeError('Evaluation stopped or failed; retain evidence without retry.')
    found = None
    for line in Path(record['stdout']).read_text(encoding='utf-8').splitlines():
        try:
            item = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue
        if isinstance(item, dict) and 'evaluation_run' in item:
            found = Path(item['evaluation_run'])
            break
    if found is None:
        raise ValueError('Evaluation did not publish its run.')
    result = json.loads((found / 'recording.json').read_text(encoding='utf-8'))
    if result['state'] != 'completed' or len(result['episodes']) != (1 if mode == 'greedy' else 3):
        raise ValueError('Evaluation did not finish its declared episodes.')
    record.update(phase='reviewed', run=str(found), result=result)
    save(state)


def supervise():
    state = json.loads(STATE.read_text(encoding='utf-8'))
    if state['phase'] != 'supervisor_pending' or STOP.exists():
        raise ValueError('Only a pending supervisor may dispatch the study.')
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('Existing Dolphin processes require ownership review.')
    capture = json.loads(CAPTURE.read_text(encoding='utf-8'))
    if digest(CAPTURE) != state['capture_sha256']:
        raise ValueError('Capture changed after preparation.')
    state.update(phase='running', supervisor={'pid': os.getpid(), 'created_time': psutil.Process().create_time()})
    save(state)
    try:
        for case in state['cases']:
            fit(state, case, capture)
            evaluate(state, case, 'greedy')
            evaluate(state, case, 'stochastic')
            case['phase'] = 'completed_reviewed'
            save(state)
        state['phase'] = 'completed_needs_scientific_review'
    except BaseException as exc:
        state.update(phase='failed_needs_review', error=repr(exc))
        save(state)
        raise
    state['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save(state)


def launch():
    state = json.loads(STATE.read_text(encoding='utf-8'))
    if state['phase'] != 'prepared' or STOP.exists():
        raise ValueError('Only a prepared study can be launched.')
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('Existing Dolphin processes require ownership review.')
    state['phase'] = 'supervisor_pending'
    save(state)
    with (ROOT / 'supervisor.out.log').open('x', encoding='utf-8') as out, (ROOT / 'supervisor.err.log').open('x', encoding='utf-8') as err:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'supervise'], stdout=out, stderr=err,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                                   | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0))
    print(json.dumps({'state': str(STATE), 'supervisor_launcher_pid': process.pid}))


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('prepare', 'launch', 'supervise', 'snapshot'):
        raise SystemExit('Use prepare, launch, supervise or snapshot.')
    if sys.argv[1] == 'prepare':
        prepare()
    elif sys.argv[1] == 'launch':
        launch()
    elif sys.argv[1] == 'supervise':
        supervise()
    else:
        print(STATE.read_text(encoding='utf-8'))
