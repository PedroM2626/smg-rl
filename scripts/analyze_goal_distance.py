"""Compare an unconnected goal-distance candidate on frozen evaluation CSVs."""

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from smg_rl.checkpoints import digest
from smg_rl.goal_distance import GroundedGoalDistance
from smg_rl.iqn_progress import episode_projection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figure', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.figure.exists():
        parser.error('Refusing to overwrite existing evidence.')
    root = Path.cwd().resolve()
    calibration = json.loads(Path('config/smg1-native-calibration.json').read_text(encoding='utf-8'))
    bounds = calibration['variations']['sweet_sweet']
    manifest = json.loads((args.checkpoint / 'manifest.json').read_text(encoding='utf-8'))
    if (digest(args.checkpoint / 'agent.pt') != manifest['agent_sha256']
            or digest(args.checkpoint / 'replay.npz') != manifest['replay_sha256']):
        raise ValueError('Checkpoint hash mismatch.')
    result = {'recorded_utc': datetime.now(timezone.utc).isoformat(),
              'checkpoint_agent_sha256': manifest['agent_sha256'],
              'checkpoint_replay_sha256': manifest['replay_sha256'],
              'scope': 'Frozen canonical-start evaluation; the goal-distance candidate is offline only.',
              'metric': 'Closest alive-grounded X/Z distance improvement from the canonical start; not native mission completion.',
              'goal_xyz': bounds['goal'], 'episodes': []}
    fig, axes = plt.subplots(1, len(args.runs), figsize=(7 * len(args.runs), 6), squeeze=False)
    for column, run in enumerate(args.runs):
        recording = json.loads((run / 'recording.json').read_text(encoding='utf-8'))
        if recording['state'] != 'completed' or recording['agent_sha256'] != manifest['agent_sha256']:
            raise ValueError('Evaluation is incomplete or uses another policy.')
        axes[0, column].set_title(recording['evaluation_mode'])
        for episode, entry in enumerate(recording['episodes'], 1):
            path = run / 'worker-00' / 'sweet_sweet' / f'episode-{episode:06d}.csv'
            with path.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) - 1 != entry['decisions'] or rows[0]['start_kind'] != 'full_start':
                raise ValueError('Evaluation trajectory/decision/start mismatch.')
            tracker = GroundedGoalDistance(bounds['start'], bounds['goal'],
                                           [float(rows[0][k]) for k in ('x', 'y', 'z')], 2500)
            initial = tracker.best_distance
            credit = 0.0
            for row in rows[1:]:
                credit += tracker.update([float(row[k]) for k in ('x', 'y', 'z')],
                                         row['grounded'] == 'True', row['dead'] == 'True',
                                         row['complete'] == 'True')
            projection = episode_projection(path, bounds, 2500)
            if tracker.invalid or 'excluded' in projection:
                raise ValueError('Invalid trajectory geometry.')
            if (entry['full_start_ram_star'] != projection['ram_star']
                    or (entry['outcome'] == 'death') != (rows[-1]['dead'] == 'True')):
                raise ValueError('Evaluation outcome does not match trajectory telemetry.')
            actions = dict(Counter(row['action'] for row in rows[1:]))
            result['episodes'].append({
                'mode': recording['evaluation_mode'], 'seed_label': entry['seed'],
                'trajectory': str(path.resolve().relative_to(root)), 'decisions': entry['decisions'],
                'outcome': entry['outcome'], 'ram_star': entry['full_start_ram_star'],
                'grounded_projection_percent': projection['geometric_projection_percent'],
                'closest_distance_improvement_world_units': initial - tracker.best_distance,
                'closest_distance_improvement_percent_of_initial': 100 * credit,
                'opening_fraction': entry['opening_fraction'], 'actions': actions,
                'episode_return': sum(float(row['reward']) for row in rows),
                'regional_reward': sum(float(row['regional_shaping_reward']) for row in rows),
                'epsilon_random_decisions': entry['epsilon_random_decisions'],
            })
            axes[0, column].plot([float(row['x']) for row in rows], [float(row['z']) for row in rows],
                                 label=f"Seed label {entry['seed']} ({entry['outcome']})", alpha=.75)
        axes[0, column].scatter(bounds['start'][0], bounds['start'][2], marker='o', color='black', label='Start')
        axes[0, column].set(xlabel='World X', ylabel='World Z')
        axes[0, column].axis('equal')
        axes[0, column].grid(alpha=.2)
        axes[0, column].legend(fontsize=8)
    fig.suptitle('Frozen IQN canonical trajectories; goal is outside this local view')
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=150)
    plt.close(fig)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'episodes': len(result['episodes']), 'stars': sum(e['ram_star'] for e in result['episodes']),
                      'output': str(args.output), 'figure': str(args.figure)}))


if __name__ == '__main__':
    main()
