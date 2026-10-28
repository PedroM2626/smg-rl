"""Launch exactly one bounded calibration with a persisted process identity."""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psutil

from smg_rl.diagnostics import atomic_json

STATE = Path('.local/robust-genetic-plan-20261027.json')
OUT = Path('.local/robust-genetic-calibration-20261027.out.log')
ERR = Path('.local/robust-genetic-calibration-20261027.err.log')


def main():
    if any(path.exists() for path in (STATE, OUT, ERR)):
        raise ValueError('Existing plan or log requires review; refusing duplicate dispatch.')
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('A Dolphin instance is active; do not start overlapping workers.')
    now = datetime.now(timezone.utc)
    state = {'phase': 'calibration_dispatching', 'created_utc': now.isoformat(),
             'calibration_deadline_utc': (now + timedelta(minutes=30)).isoformat(),
             'calibration_seconds': 1800, 'workers': 6, 'auto_retry': False,
             'sigma_candidates': [.02, .05, .10], 'children_per_sigma': 16,
             'selection_rule': 'Largest sigma with at least 8/16 viable and 4/16 distinct action prefixes.',
             'training_arm_seconds_each': 10800, 'stdout': str(OUT), 'stderr': str(ERR)}
    if not atomic_json(STATE, state):
        raise OSError('Failed to persist pre-dispatch state.')
    try:
        with OUT.open('x', encoding='utf-8') as out, ERR.open('x', encoding='utf-8') as err:
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            process = subprocess.Popen([sys.executable, 'scripts/calibrate_robust_genetic.py'], cwd=Path.cwd(),
                                       stdin=subprocess.DEVNULL, stdout=out, stderr=err, creationflags=flags)
        state.update(phase='calibration_running', pid=process.pid,
                     process_created_time=psutil.Process(process.pid).create_time())
        if not atomic_json(STATE, state):
            raise OSError('Failed to persist process identity after dispatch.')
        print(json.dumps(state, indent=2))
    except BaseException as exc:
        state.update(phase='dispatch_needs_review', error=repr(exc))
        atomic_json(STATE, state)
        raise


if __name__ == '__main__':
    main()
