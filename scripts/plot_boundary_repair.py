"""Plot paired fresh prediction audits; error bars are latent-draw variation."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    data = json.loads(Path('docs/assets/performance/smg1-boundary-confirmation-20261025.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    labels = ['Stopped model\n1024 updates', 'Retained visual core\n512 updates', 'Repaired heads\n704 updates']
    metrics = [('terminal_reward_mae', .5, 'Terminal reward MAE'),
               ('terminal_brier', .2, 'Terminal continuation Brier'),
               ('open_loop_image_mse', .04, 'Eight-step image MSE')]
    for axis, (metric, limit, title) in zip(axes, metrics):
        values = [[audit['metrics']['sweet_sweet'][metric] for audit in row['audits']] for row in data['rows']]
        axis.bar(labels, [np.mean(v) for v in values], yerr=[np.std(v) for v in values],
                 color=['#7c8796', '#4c91bd', '#25896e'], capsize=4)
        axis.axhline(limit, color='#bd4545', linestyle='--', label=f'Unchanged limit {limit:g}')
        axis.set_title(title)
        axis.tick_params(axis='x', labelsize=8)
        axis.legend(fontsize=8)
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle('SMG1: paired prediction audit on four fresh confirmation episodes')
    fig.text(.5, .01, 'Same episodes and latent draws; bars are mean +/- SD of 3 draws, not independent training seeds. No mission-completion claim.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .92))
    fig.savefig('docs/assets/performance/smg1-boundary-confirmation-20261025.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
