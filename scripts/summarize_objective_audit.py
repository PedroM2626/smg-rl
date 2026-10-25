"""Render the dated read-only objective audit and summarize control fixtures."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from smg_rl.config import load_config
from smg_rl.rewards import ProgressReward

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'docs/assets/performance'


def main():
    names = ['SMG1 (0.01)', 'SMG1 (0.1)', 'SMG2 (0.01)']
    files = ['smg1-objective-audit-20261025.json', 'smg1-curiosity10x-objective-audit-20261025.json',
             'smg2-objective-audit-20261025.json']
    data = [json.loads((ASSETS / file).read_text()) for file in files]
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for offset, key, color, label in [(-0.18, 'remove_intrinsic_gradient_change_fraction', '#287eb5', 'Remove intrinsic bonus'),
                                      (0.18, 'lower_entropy_gradient_change_fraction', '#d58028', 'Reduce entropy 10x')]:
        values = [[100 * row[key] for row in item['rows']] for item in data]
        mean = np.array([np.mean(v) for v in values])
        error = np.array([[mean[i] - min(v) for i, v in enumerate(values)],
                          [max(v) - mean[i] for i, v in enumerate(values)]])
        axes[0].bar(np.arange(3) + offset, mean, width=0.34, yerr=error, color=color, label=label, capsize=3)
    axes[0].set(xticks=np.arange(3), xticklabels=names, ylabel='Actor gradient change / full gradient (%)',
                title='Fixed-rollout objective attribution')
    axes[0].legend(fontsize=8)
    stationary = json.loads((ASSETS / 'smg1-stationary-bonus-20261025.json').read_text())
    values = [np.mean([row['weighted_bonus'] for row in item['trace']]) for item in stationary['sequences']]
    axes[1].bar([item['input'] for item in stationary['sequences']], values, color='#4d8e7e')
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
    axes[1].set(ylabel='Calculated bonus, coefficient 0.01', title='SMG1: zero horizontal displacement in all four probes')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle('October 25, 2026: diagnostic evidence, not trained-policy improvement', fontsize=12)
    fig.savefig(ASSETS / 'objective-audit-20261025.png', dpi=160)
    plt.close(fig)
    controls = json.loads((ASSETS / 'control-probe-20261025.json').read_text())
    summary = {'physical_fixture_decisions': controls['physical_decisions'] + stationary['physical_decisions'],
               'optimizer_updates': 0, 'new_training_decisions': 0, 'scenarios': []}
    for scene in controls['scenarios']:
        config = load_config('config/staged-smg1-control.yaml' if scene['game'] == 'RMGE01'
                             else 'config/staged-smg2-terminal_reward_weight.yaml')
        calibration = json.loads(Path(config['paths']['calibration']).read_text())['variations'][scene['scenario']]
        below_start_rewards = []
        for direction in scene['held_directions']:
            progress = ProgressReward(calibration['start'], calibration['goal'],
                                      config['environment']['min_reward_y'],
                                      config['environment']['max_step_distance'])
            progress.reset(direction['initial']['position'])
            for step in direction['trace']:
                delta, _ = progress.update(step['position'], step['dead'], step['complete'])
                if delta > 0 and step['position'][1] < direction['initial']['position'][1] - 50:
                    below_start_rewards.append(config['rewards']['progress'] * delta)
        actual = np.array([a['actual_one_step_reward'] for a in scene['single_actions']])
        prediction = np.array([a['prediction_mean_latent'] for a in scene['single_actions']])
        summary['scenarios'].append({'game': scene['game'], 'scenario': scene['scenario'],
            'one_step_actual_reward_min_max': [float(actual.min()), float(actual.max())],
            'cold_mean_latent_reward_mae': float(np.mean(np.abs(actual - prediction))),
            'maximum_action_reward_contrast': float(np.ptp(actual)),
            'positive_progress_steps_below_initial_y_minus_50': len(below_start_rewards),
            'positive_progress_reward_below_initial_y_minus_50': sum(below_start_rewards),
            'held_directions': [{'direction': d['direction'], 'decisions': len(d['trace']),
                                 'maximum_xz_distance': d['maximum_xz_distance'],
                                 'ended_dead': d['trace'][-1]['dead']} for d in scene['held_directions']]})
    (ASSETS / 'objective-control-summary-20261025.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
