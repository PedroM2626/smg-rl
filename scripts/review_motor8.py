"""Read-only snapshot of the owner-selected motor8 jobs and policy previews."""

import json
from pathlib import Path

from review_interventions import controller_live

from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.exploration_trial import review_run


def main():
    state = json.loads(Path('.local/motor8-state.json').read_text())
    result = {'state':state['state'],'jobs':[]}
    for job in state['jobs']:
        config = load_config(job['config'])
        row = {'game':job['game'],'phase':job['phase'],'recorded_new_real_decisions':job['new_real_decisions']}
        root = Path(config['paths']['runs'])
        for label,pattern in [('online','world-model-2*'),('offline','world-model-offline-*')]:
            runs = sorted(p for p in root.glob(pattern) if p.is_dir())
            if runs and (runs[-1]/'status.json').exists():
                row[label] = {'evidence':review_run(runs[-1]),'controller_live':controller_live(runs[-1])}
        previews = sorted(root.glob('eval-world-video-*/recording.json'))
        row['videos'] = [json.loads(p.read_text())['video'] for p in previews]
        result['jobs'].append(row)
    atomic_json('.local/motor8-latest-review.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
