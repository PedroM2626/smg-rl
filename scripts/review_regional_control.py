"""Review completed control comparisons, verify video sidecars and plot real paths."""

import argparse
import csv
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from smg_rl.checkpoints import digest
from smg_rl.diagnostics import atomic_json


def verify_video(path, condition):
    path = Path(path)
    records = [json.loads(line) for line in path.with_suffix('.inputs.jsonl').read_text().splitlines()]
    frames = 4 if condition == 'held4_carry' else 12
    groups = defaultdict(list)
    for record in records:
        groups[(record['episode'], record['decision'])].append(record)
    directions = {}
    for (episode, _), samples in groups.items():
        assert len(samples) == frames // 2
        assert len({s['policy_input_sha256'] for s in samples}) == 1
        assert samples[0]['policy_input_sha256'] is not None
        label = samples[0]['selected_action']
        assert label in ('none', 'd', 'a', 'w', 's', 'jump', 'spin', 'crouch')
        direction = directions.get(episode, [])
        if label in ('d', 'a', 'w', 's'):
            direction = [label]
        elif label in ('none', 'crouch'):
            direction = []
        directions[episode] = direction
        expected = [] if label == 'none' else [label]
        if label == 'jump' and 'carry' in condition:
            expected = direction + ['jump']
        for index, sample in enumerate(samples):
            applied = [] if condition == 'legacy12' and label in ('jump', 'spin') and index >= 4 else expected
            assert sample['applied_controls'] == applied
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_frames,width,height,avg_frame_rate', '-of', 'json', str(path)], text=True))
    assert int(probe['streams'][0]['nb_frames']) == len(records)
    return {'video': str(path), 'sha256': digest(path), 'frames': len(records),
            'verified': 'Exact action/control timing, stable actual input digest per decision, encoded frame count.',
            'probe': probe}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    reports = [json.loads((Path(r) / 'result.json').read_text()) for r in args.run]
    assert all(r['state'] == 'completed' for r in reports)
    rows, videos = [], []
    figure, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    axis_index = 0
    for report in reports:
        expected = len(report['conditions']) * len(report['seeds']) * (1 if report['game'] == 'smg1' else 2)
        assert len(report['attempts']) == expected
        grouped = defaultdict(list)
        for attempt in report['attempts']:
            assert attempt['frames'] <= 1200 and attempt['start_kind'] == 'full_start'
            grouped[(attempt['variation'], attempt['condition'])].append(attempt)
        for (variation, condition), attempts in grouped.items():
            rows.append({'game': report['game'], 'variation': variation, 'condition': condition,
                'seeds': [a['seed'] for a in attempts],
                'mean_confirmed_distance': statistics.mean(a['confirmed_distance'] for a in attempts),
                'median_confirmed_distance': statistics.median(a['confirmed_distance'] for a in attempts),
                'best_confirmed_distance': max(a['confirmed_distance'] for a in attempts),
                'deaths': sum(a['dead'] for a in attempts), 'damage': sum(a['damage'] for a in attempts),
                'ram_stars': sum(a['ram_star'] for a in attempts),
                'frames': sum(a['frames'] for a in attempts), 'decisions': sum(a['decisions'] for a in attempts),
                'median_nonvideo_frames_per_second': statistics.median(
                    a['emulated_frames_per_second'] for a in attempts if not a['video'])})
        for movie in sorted({a['video'] for a in report['attempts'] if a['video']}):
            condition = Path(movie).parent.name
            videos.append(verify_video(movie, condition))
        for variation in sorted({a['variation'] for a in report['attempts']}):
            axis = axes[axis_index]
            axis_index += 1
            for condition in report['conditions']:
                best = max((a for a in report['attempts'] if a['variation'] == variation and a['condition'] == condition),
                           key=lambda a: a['confirmed_distance'])
                folder = Path(report['run']) / condition / f"seed-{best['seed']}" / variation
                files = list(folder.glob('episode-*.csv'))
                matching = [f for f in files if len(list(csv.DictReader(f.open()))) - 1 == best['decisions']]
                # One episode per mission/seed in this experiment.
                assert len(matching) == 1
                trace = list(csv.DictReader(matching[0].open()))
                axis.plot([float(r['x']) for r in trace], [float(r['z']) for r in trace],
                          label=f"{condition}, seed {best['seed']}", linewidth=1.3)
            axis.set_title(f"{report['game'].upper()} / {variation}")
            axis.set_xlabel('World X')
            axis.set_ylabel('World Z')
            axis.invert_yaxis()
            axis.set_aspect('equal', adjustable='datalim')
            axis.legend(fontsize=7)
            axis.grid(alpha=.2)
    figure.suptitle('Best frozen-policy path per condition — canonical starts; no new learning')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix('.png'), dpi=160)
    plt.close(figure)
    evidence = {'scope': 'Frozen-policy sensitivity only; not a trained comparison or native mission percentage.',
                'runs': reports, 'summary': rows, 'videos': videos,
                'limitations': ['Three seeds only; no significance claim.',
                    'SMG1 source has one actor update; SMG2 source has134. Both are weak policies.',
                    'Four-frame recurrent inference was intentionally adapted without fitting; dynamics gates do not transfer.',
                    'Capture/encoding and fixed condition order confound absolute wall throughput.',
                    'Distances cover the validated opening. Full mission progression remains unverified.']}
    atomic_json(output, evidence)
    print(json.dumps({'attempts': sum(len(r['attempts']) for r in reports), 'verified_videos': len(videos), 'summary': rows}))


if __name__ == '__main__':
    main()
