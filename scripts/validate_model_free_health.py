"""Review actual model-free health blocks and bounded video/evaluation fixtures."""

import json
import subprocess
from pathlib import Path

import psutil
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.dqn_video import record_dqn


def main():
    state = json.loads(Path('.local/model-free-state.json').read_text())
    assert not [p for p in psutil.process_iter(['name']) if p.info['name'] and p.info['name'].lower() == 'dolphin.exe']
    report = {'scope': 'Finite startup/optimizer/evaluation implementation checks; not an algorithm performance comparison.',
              'maximum_evaluation_decisions_per_mission': 96, 'training_phases': {}, 'evaluations': []}
    for game, job in state['current_jobs'].items():
        assert job['review']['state'] == 'passed'
        report['training_phases'][game] = {'run': job['run'], 'status': job['status'], 'review': job['review']}
        events = EventAccumulator(job['status']['tensorboard'])
        events.Reload()
        assert 'observations/actual_four_frame_stack' in events.Tags()['images']
        config = load_config(job['config'])
        for variation in range(len(config['variations'])):
            run = record_dqn(job['config'], Path(job['run']) / 'checkpoint-final', variation,
                             seeds=(11,), maximum_steps=96)
            recording = json.loads((run / 'recording.json').read_text())
            movie = Path(recording['video'])
            trace = [json.loads(line) for line in movie.with_suffix('.inputs.jsonl').read_text().splitlines()]
            assert len(trace) == recording['frames_written']
            groups = {}
            for row in trace:
                groups.setdefault((row['episode'], row['decision']), []).append(row)
            for samples in groups.values():
                assert len(samples) == 6 and len({s['policy_input_sha256'] for s in samples}) == 1
                label = samples[0]['selected_action']
                controls = [] if label == 'none' else [label]
                for index, sample in enumerate(samples):
                    expected = [] if label in ('jump', 'spin') and index >= 4 else controls
                    assert sample['applied_controls'] == expected
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=nb_frames,width,height', '-of', 'json', str(movie)], text=True))
            assert int(probe['streams'][0]['nb_frames']) == len(trace)
            report['evaluations'].append({'game': game, 'run': str(run), 'recording': recording,
                                           'video_verified': True, 'probe': probe})
            atomic_json('docs/assets/performance/model-free-health-20261026.json', report)
    print(json.dumps({'training': {g: d['status']['real_decisions'] for g, d in report['training_phases'].items()},
                      'evaluations': report['evaluations']}))


if __name__ == '__main__':
    main()
