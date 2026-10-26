"""Locate a recorded exploratory high-water mark, without replaying or training."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    metadata = json.loads(Path('docs/assets/performance/sweet-518-location-20261026.json').read_text())
    with Path(metadata['training_csv']).open(newline='') as stream:
        training = list(csv.DictReader(stream))
    with Path(metadata['frozen_csv']).open(newline='') as stream:
        frozen = list(csv.DictReader(stream))
    route = json.loads(Path('config/routes/smg1-opening-regional-v3.json').read_text())['waypoints']
    peak = max(training, key=lambda r: float(r['start_relative_route_progress']))
    figure, (axis, progress) = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True)
    axis.plot([p[0] for p in route], [p[2] for p in route], '--', color='gray', label='Mapped opening route')
    prefix = training[:int(peak['step']) + 1]
    for rows, color, label in [(prefix, 'royalblue', 'Exploratory training path to peak'),
                               (frozen, 'darkorange', 'Frozen greedy evaluation')]:
        axis.plot([float(r['x']) for r in rows], [float(r['z']) for r in rows], color=color, alpha=.7, label=label)
    for x, z, label, color, offset in [(-50, -50, 'Canonical start', 'black', (-80, 18)),
        (float(peak['x']), float(peak['z']), 'Training peak: 51.80%\nX159.8 / Z-1084.1', 'green', (20, 0)),
        (float(frozen[-1]['x']), float(frozen[-1]['z']), 'Frozen stall\nX166.7 / Z-315.1', 'darkorange', (18, 4)),
        (213.23, -1413.87, 'first-pink anchor', 'purple', (14, -5))]:
        axis.scatter(x, z, color=color, s=65, zorder=5)
        axis.annotate(label, (x, z), xytext=offset, textcoords='offset points', fontsize=9)
    axis.invert_yaxis()
    axis.set_aspect('equal', adjustable='datalim')
    axis.set(xlabel='World X', ylabel='World Z', title='Top-down RAM trajectory')
    axis.grid(alpha=.2)
    axis.legend(loc='upper left', fontsize=8)
    for rows, color, label in [(training, 'royalblue', 'Training attempt657 (changing/noisy policy)'),
                               (frozen, 'darkorange', 'Final frozen greedy policy')]:
        progress.plot([int(r['step']) for r in rows], [100 * float(r['start_relative_route_progress']) for r in rows],
                      color=color, label=label)
    progress.axhline(51.7965, linestyle=':', color='green', alpha=.6)
    progress.axvline(18, linestyle=':', color='gray', alpha=.6)
    progress.set(ylim=(0, 105), xlabel='Decision within attempt', ylabel='Verified opening advance (%)',
                 title='Opening high-water mark, not full mission')
    progress.grid(alpha=.2)
    progress.legend(fontsize=8)
    figure.suptitle('Sweet Sweet: where the 51.80% training record occurred\n'
                   'Only the short mapped opening is shown; neither attempt collected a star', fontsize=12)
    figure.savefig('docs/assets/performance/sweet-518-location-20261026.png', dpi=150)
    plt.close(figure)


if __name__ == '__main__':
    main()
