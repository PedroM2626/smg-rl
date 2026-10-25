"""Verify exactly the declared trained motor8 evaluations; no job dispatch."""

import csv
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from smg_rl.checkpoints import digest
from smg_rl.diagnostics import atomic_json
from smg_rl.plotting import plot_trajectory


def main():
    state = json.loads(Path('.local/motor8-state.json').read_text())
    job = next(j for j in state['jobs'] if j['game'] == 'smg2')
    evaluations = job['evaluations']
    assert {e['variation'] for e in evaluations} == {'spinning', 'double_time'}
    destination = Path('runs/motor8-evaluations-20261025')
    destination.mkdir(exist_ok=True)
    result = {'scope': 'Predeclared full-start policy evaluations, seeds11/23/47, '
              'maximum600 decisions per attempt. No optimizer or replay recording. '
              'Validation seeds are not independent training replications. '
              'Geometric projection is not verified safe traversal or native mission percentage.',
              'missions': []}
    for evaluation in evaluations:
        run = Path(evaluation['run'])
        data = json.loads((run / 'recording.json').read_text())
        assert data['state'] == 'completed' and data['seeds'] == [11, 23, 47]
        assert data['variation'] == evaluation['variation']
        assert data['source_actor_updates'] == 134
        assert data['checkpoint_sha256'] == evaluation['checkpoint_sha256'] == digest(job['checkpoint'])
        episodes = data['episodes']
        assert [e['seed'] for e in episodes] == [11, 23, 47]
        assert all(1 <= e['decisions'] <= 600 for e in episodes)
        movie = Path(data['video'])
        sidecar = movie.with_suffix('.inputs.jsonl')
        trace = [json.loads(line) for line in sidecar.read_text().splitlines()]
        assert len(trace) == data['frames_written'] == 3 + 6 * sum(e['decisions'] for e in episodes)
        groups = {}
        for row in trace:
            assert len(row['applied_controls']) <= 1 and row['policy_input_sha256']
            groups.setdefault((row['episode'], row['decision']), []).append(row)
        for (_, decision), samples in groups.items():
            assert len({s['policy_input_sha256'] for s in samples}) == 1
            if not decision:
                assert len(samples) == 1
                continue
            assert len(samples) == 6
            action = samples[0]['selected_action']
            if action in ('jump', 'spin'):
                assert all(s['applied_controls'] == [action] for s in samples[:4])
                assert all(s['applied_controls'] == [] for s in samples[4:])
            else:
                controls = [] if action == 'none' else [action]
                assert all(s['applied_controls'] == controls for s in samples)
        probe = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
            'stream=width,height,codec_name,avg_frame_rate,nb_frames',
            '-show_entries', 'format=duration', '-of', 'json', str(movie)], text=True))
        assert int(probe['streams'][0]['nb_frames']) == len(trace)
        assert probe['streams'][0]['avg_frame_rate'] == '30/1'
        output = destination / f"smg2-{data['variation']}-trained.mp4"
        if output.exists():
            assert digest(output) == digest(movie)
        else:
            shutil.copy2(movie, output)
        screenshot = destination / f"smg2-{data['variation']}-trained.png"
        if not screenshot.exists():
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-ss', '2',
                            '-i', str(movie), '-frames:v', '1', str(screenshot)], check=True)
        traces = []
        for path in sorted(run.glob('worker-*/*/episode-*.csv')):
            with path.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            assert rows[0]['start_kind'] == 'full_start'
            assert len(rows) - 1 == episodes[len(traces)]['decisions']
            traces.append((path, rows))
        assert len(traces) == 3
        best_path, _ = max(traces, key=lambda item: max(float(r['best_progress']) for r in item[1]))
        graph = Path(f"docs/assets/performance/motor8-smg2-{data['variation']}-evaluation-20261025.png")
        plot_trajectory(best_path, graph)
        result['missions'].append({
            'variation': data['variation'], 'run': str(run), 'checkpoint': data['checkpoint'],
            'checkpoint_sha256': data['checkpoint_sha256'], 'actor_updates': data['source_actor_updates'],
            'episodes': episodes, 'physical_evaluation_decisions': sum(e['decisions'] for e in episodes),
            'outcomes': dict(Counter(e['outcome'] for e in episodes)),
            'ram_full_start_stars': sum(e['final_telemetry']['complete'] for e in episodes),
            'maximum_geometric_projection_percent': max(e['geometric_projection_percent'] for e in episodes),
            'video': str(output), 'video_sha256': digest(movie), 'sidecar_sha256': digest(sidecar),
            'frames': len(trace), 'probe': probe, 'best_trace': str(best_path), 'trajectory': str(graph),
            'sidecar_checks': 'Exact per-decision input digest retained, exclusive applied controls, '
            'correct8-on/4-off jump/spin samples and all declared attempts retained.'})
    atomic_json('docs/assets/performance/motor8-trained-evaluations-20261025.json', result)
    print(json.dumps([{k: m[k] for k in ('variation', 'physical_evaluation_decisions', 'outcomes',
                                       'ram_full_start_stars', 'maximum_geometric_projection_percent')}
                      for m in result['missions']], indent=2))


if __name__ == '__main__':
    main()
