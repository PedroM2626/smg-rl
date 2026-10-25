"""Read-only compact progress review for the finite exploration intervention."""

import json
from pathlib import Path

import psutil

from smg_rl.diagnostics import atomic_json
from smg_rl.exploration_trial import review_run


def controller_live(run):
    path = Path(run) / 'controller.json'
    if not path.exists():
        return False
    controller = json.loads(path.read_text())
    try:
        process = psutil.Process(controller['pid'])
        return abs(process.create_time() - controller['created_time']) < 0.1
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def main():
    state = json.loads(Path('.local/exploration-trial-state.json').read_text())
    result = {'scope':'Read-only staged inspection; no job dispatch', 'factorial':[], 'motor':[]}
    for case in state['cases']:
        root = Path(case['config']['paths']['runs'])
        runs = sorted(root.glob('world-model-*'))
        row = {k:case[k] for k in ('name','seed','state','new_online_decisions')}
        if runs and (runs[-1]/'status.json').exists():
            row['evidence'] = review_run(runs[-1])
            row['controller_live'] = controller_live(runs[-1])
        result['factorial'].append(row)
    for arm in ('control','route'):
        root = Path(f'runs/motor16-{arm}')
        for prefix in ('world-model-2','world-model-offline-'):
            runs = sorted(p for p in root.glob(prefix+'*') if p.is_dir())
            if runs and (runs[-1]/'status.json').exists():
                row = review_run(runs[-1])
                row.update(arm=arm, kind='offline_fit' if 'offline' in prefix else 'online',
                           controller_live=controller_live(runs[-1]))
                result['motor'].append(row)
    atomic_json('.local/intervention-latest-review.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
