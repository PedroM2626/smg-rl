"""Compare the completed single-seed Rainbow/IQN pilots without new learning."""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from smg_rl.diagnostics import atomic_json


def main():
    reports = {name: json.loads(Path(path).read_text()) for name, path in {
        'Rainbow': 'docs/assets/performance/model-free-final-20261026.json',
        'IQN': 'docs/assets/performance/model-free-iqn-final-20261026.json'}.items()}
    summary = {'scope': 'One paired training seed per game, three frozen epsilon-evaluation seeds. Exploratory, no significance or superiority claim.',
               'missions': [], 'throughput': {}, 'limitations': [
                   'Opening fractions are not full mission percentages.',
                   'Collection maxima are not frozen policy results.',
                   'Wall-clock workloads/visitation differ; throughput is not sample efficiency.',
                   'Inputs use the same documented recording capture path in both algorithms.']}
    for game in ('smg1', 'smg2'):
        summary['throughput'][game] = {name: r['games'][game]['status']['decisions_per_second'] for name, r in reports.items()}
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    actions = ['none', 'd', 'a', 'w', 's', 'jump', 'spin', 'crouch']
    labels = ['None', 'Right', 'Left', 'Forward', 'Backward', 'Jump', 'Spin', 'Crouch']
    colors = plt.get_cmap('tab10').colors
    for column, mission in enumerate(('sweet_sweet', 'spinning', 'double_time')):
        results = {}
        top, bottom = axes[:, column]
        for model_index, (name, report) in enumerate(reports.items()):
            item = next(e for e in report['evaluations'] if e['mission'] == mission)
            recording = item['recording']
            episodes = recording['episodes']
            results[name] = {'real_training_decisions_per_mission': 8192,
                'full_start_stars': sum(e['full_start_ram_star'] for e in episodes),
                'evaluation_decisions': sum(e['decisions'] for e in episodes),
                'outcomes': [e['outcome'] for e in episodes],
                'mapped_opening_percent_by_seed': [100 * e['opening_fraction'] for e in episodes],
                'action_counts': item['action_counts'], 'video': recording['video']}
            paths = sorted((Path(item['run']) / 'worker-00' / mission).glob('episode-*.csv'))
            assert len(paths) == 3
            for index, path in enumerate(paths):
                with path.open(newline='') as stream:
                    rows = list(csv.DictReader(stream))
                top.plot([float(r['x']) for r in rows], [float(r['z']) for r in rows],
                         color=('tab:orange' if name == 'Rainbow' else 'tab:blue'),
                         alpha=.7, label=name if index == 0 else None)
            height = 0.
            total = sum(item['action_counts'].values())
            for action, label, color in zip(actions, labels, colors):
                amount = 100 * item['action_counts'].get(action, 0) / total
                bottom.bar(model_index, amount, bottom=height, color=color,
                           label=label if column == 0 and model_index == 0 else None)
                height += amount
        summary['missions'].append({'mission': mission, 'algorithms': results})
        top.set_title(mission)
        top.set_xlabel('World X')
        top.set_ylabel('World Z')
        top.invert_yaxis()
        top.set_aspect('equal', adjustable='datalim')
        top.grid(alpha=.2)
        top.legend(fontsize=8)
        bottom.set_xticks(np.arange(2), list(reports))
        bottom.set_ylim(0, 100)
        bottom.set_ylabel('Selected logical actions (%)')
        bottom.grid(axis='y', alpha=.2)
    handles, legend_labels = axes[1, 0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc='outside lower center', ncols=8, fontsize=9)
    figure.suptitle('Frozen canonical-start evaluation: all three seeds; world paths and action occupancy\nEqual real budgets; no star; no native mission-percentage or learning-efficiency claim')
    output = Path('docs/assets/performance/model-free-comparison-20261026.json')
    figure.savefig(output.with_suffix('.png'), dpi=150)
    plt.close(figure)
    atomic_json(output, summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
