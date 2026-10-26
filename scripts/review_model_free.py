"""Read-only real-data, TensorBoard, checkpoint and frozen-evaluation audit."""

import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from review_regional_six import audit_episode
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.route import RouteProgress


def main():
    state = json.loads(Path('.local/model-free-state.json').read_text())
    algorithm = state.get('current_algorithm', 'rainbow')
    report = {'observed_at': datetime.now(timezone.utc).isoformat(),
              'scope': f'Completed declared {algorithm} blocks and frozen evaluations. No new training or fitting.',
              'games': {}, 'missions': [], 'evaluations': []}
    for game, job in state['current_jobs'].items():
        config = load_config(job['config'])
        run = Path(job['run'])
        status = json.loads((run / 'status.json').read_text())
        assert status['state'] == 'completed' and status['real_decisions'] == job['target_count']
        assert status['error'] is None and status['imagined_transitions'] == 0
        checkpoint = run / 'checkpoint-final'
        manifest = json.loads((checkpoint / 'manifest.json').read_text())
        assert not (checkpoint / '.incomplete').exists()
        assert digest(checkpoint / 'agent.pt') == manifest['agent_sha256']
        assert digest(checkpoint / 'replay.npz') == manifest['replay_sha256']
        with np.load(checkpoint / 'replay.npz', allow_pickle=False) as replay:
            meta = json.loads(str(replay['metadata']))
            assert meta['emitted'] == meta['size'] == status['real_decisions']
            scenarios = dict(Counter(replay['scenario'][:meta['size']].tolist()))
        route = RouteProgress(json.loads(Path(config['paths']['route']).read_text()))
        episodes = []
        actions = Counter()
        for source in (Path(job['health_run']), run):
            ledger = [json.loads(line) for line in (source / 'collector-steps.jsonl').read_text().splitlines()]
            for row in ledger:
                actions.update(config_name for index in row['actions']
                    for config_name in [manifest['signature']['action_names'][index]])
            for path in source.glob('worker-*/*/episode-*.csv'):
                episodes.append(audit_episode(path, config, route))
        assert sum(e['decisions'] for e in episodes) == status['real_decisions']
        assert sum(actions.values()) == status['real_decisions']
        for name in manifest['signature']['scenario_names']:
            group = [e for e in episodes if e['variation'] == name]
            for kind in ('full_start', 'curriculum'):
                selected = [e for e in group if e['start_kind'] == kind]
                best = max(selected, key=lambda e: e['maximum_start_relative_distance'])
                report['missions'].append({'game': game, 'variation': name, 'start_kind': kind,
                    'episodes': len(selected), 'decisions': sum(e['decisions'] for e in selected),
                    'outcomes': dict(Counter(e['outcome'] for e in selected)),
                    'ram_stars': sum(e['ram_star'] for e in selected),
                    'nonfatal_hp_lost': sum(e['nonfatal_hp_lost'] for e in selected),
                    'maximum_start_relative_opening_fraction': best['maximum_start_relative_opening_fraction'],
                    'maximum_grounded_arc_distance': best['maximum_start_relative_distance'],
                    'best_collection_trace': best['csv'],
                    'note': 'Collection maximum, not a frozen-policy result.'})
        events = EventAccumulator(status['tensorboard'])
        events.Reload()
        clip = (events.Scalars('learning/support_clipped_mass')
                if 'learning/support_clipped_mass' in events.Tags()['scalars'] else [])
        report['games'][game] = {'status': status, 'manifest': manifest,
            'scenario_transition_counts': scenarios, 'training_action_counts': dict(actions),
            'terminal_failures_audited': sum(e['outcome'] in ('death', 'timeout') for e in episodes),
            'maximum_terminal_closure_error': max(e['terminal_shaping_closure_error'] or 0 for e in episodes),
            'observed_logged_support_clip_mass_max': max((e.value for e in clip), default=None),
            'tensorboard_scalar_tags': events.Tags()['scalars']}
    if len(state['final_evaluations']) != 3:
        raise ValueError('Complete all three declared frozen mission evaluations before the final audit.')
    for name, job in state['final_evaluations'].items():
        run = Path(job['run'])
        recording = json.loads((run / 'recording.json').read_text())
        assert recording['state'] == 'completed' and len(recording['episodes']) == 3
        assert [e['seed'] for e in recording['episodes']] == [11, 23, 47]
        assert all(e['decisions'] <= 300 for e in recording['episodes'])
        movie = Path(recording['video'])
        samples = [json.loads(line) for line in movie.with_suffix('.inputs.jsonl').read_text().splitlines()]
        groups = {}
        for sample in samples:
            groups.setdefault((sample['episode'], sample['decision']), []).append(sample)
        actions = Counter()
        for group in groups.values():
            assert len(group) == 6 and len({s['policy_input_sha256'] for s in group}) == 1
            expected_q = group[0]['policy_expected_returns']
            assert len(expected_q) == 8 and np.isfinite(expected_q).all()
            label = group[0]['selected_action']
            actions[label] += 1
            for index, sample in enumerate(group):
                expected = [] if label == 'none' or label in ('jump', 'spin') and index >= 4 else [label]
                assert sample['applied_controls'] == expected and sample['policy_expected_returns'] == expected_q
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_frames,width,height', '-of', 'json', str(movie)], text=True))
        assert int(probe['streams'][0]['nb_frames']) == len(samples) == recording['frames_written']
        checkpoint = Path(recording['checkpoint'])
        assert digest(checkpoint / 'agent.pt') == recording['agent_sha256']
        report['evaluations'].append({'mission': name, 'run': str(run), 'recording': recording,
                                      'action_counts': dict(actions), 'video_verified': True, 'probe': probe})
    atomic_json(f'docs/assets/performance/model-free-{algorithm}-final-20261026.json', report)
    print(json.dumps({'missions': report['missions'], 'evaluations': [
        {'mission': e['mission'], 'episodes': e['recording']['episodes'], 'actions': e['action_counts']}
        for e in report['evaluations']]}))


if __name__ == '__main__':
    main()
