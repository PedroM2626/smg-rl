"""Launch one owner-requested three-hour genetic distance run and persist its identity."""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psutil

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.profiles import require_training_enabled
from smg_rl.resource_disk import require_disk_headroom
from smg_rl.resource_memory import require_memory_headroom

STATE = Path('.local/genetic-goal-three-hours-20261027.json')
SOURCE = Path('runs/model-free/smg1-six-two-hour/dqn-20261026-204307-660910/checkpoint-final')
HEAD = Path('runs/model-free/genetic-goal-distance-smg1/genetic-20261027-191407-420841/champion.pt')
CONFIG = 'config/genetic-goal-distance-smg1.yaml'
STDOUT = Path('.local/genetic-goal-three-hours-20261027.out.log')
STDERR = Path('.local/genetic-goal-three-hours-20261027.err.log')


def save(state):
    if not atomic_json(STATE, state):
        raise OSError('Could not persist the three-hour dispatch identity.')


def launch():
    if STATE.exists() or STDOUT.exists() or STDERR.exists():
        raise ValueError('An existing dispatch or log requires review; refusing to start another job.')
    config = load_config(CONFIG)
    require_training_enabled(config)
    require_disk_headroom(config)
    require_memory_headroom()
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        name = (process.info['name'] or '').lower()
        command = ' '.join(process.info['cmdline'] or []).lower()
        if name == 'dolphin.exe' or (process.info['pid'] != os.getpid()
                                      and 'train_genetic.py' in command and 'smg-rl' in command):
            raise ValueError('An emulator or genetic trainer is already active; inspect ownership first.')
    parent = json.loads((HEAD.parent / 'summary.json').read_text(encoding='utf-8'))
    if (parent['state'] != 'completed' or parent['error'] or not parent['source_unchanged']
            or parent['fitness_mode'] != 'goal-distance-xyz-v1'
            or digest(HEAD) != parent['champion_sha256']
            or any(digest(SOURCE / name) != value for name, value in parent['source_hashes'].items())):
        raise ValueError('Pilot champion or source integrity failed; do not dispatch.')
    started = datetime.now(timezone.utc)
    deadline = started + timedelta(hours=3)
    command = [sys.executable, 'scripts/train_genetic.py', '--config', CONFIG,
               '--source', str(SOURCE), '--initial-head', str(HEAD), '--fitness-mode',
               'goal-distance-xyz-v1', '--policy-mode', 'head', '--seconds', '10800',
               '--deadline-utc', deadline.isoformat(), '--confirmation-seconds', '0',
               '--population-seed', '143']
    state = {'phase': 'dispatching', 'started_utc': started.isoformat(), 'deadline_utc': deadline.isoformat(),
             'seconds': 10800, 'workers': 6, 'fitness_mode': 'goal-distance-xyz-v1',
             'population_seed': 143, 'initial_head': str(HEAD), 'initial_head_sha256': parent['champion_sha256'],
             'source_hashes': parent['source_hashes'], 'command': command,
             'stdout': str(STDOUT), 'stderr': str(STDERR), 'auto_retry': False,
             'fitness_reused': False, 'frozen_confirmation_reserved': False}
    save(state)
    try:
        with STDOUT.open('x', encoding='utf-8') as out, STDERR.open('x', encoding='utf-8') as err:
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            process = subprocess.Popen(command, cwd=Path.cwd(), stdin=subprocess.DEVNULL,
                                       stdout=out, stderr=err, creationflags=flags)
        identity = psutil.Process(process.pid)
        state.update(phase='running', pid=process.pid, created_time=identity.create_time())
        save(state)
        print(json.dumps(state, indent=2))
    except BaseException as exc:
        state.update(phase='dispatch_failed', error=repr(exc))
        save(state)
        raise


if __name__ == '__main__':
    launch()
