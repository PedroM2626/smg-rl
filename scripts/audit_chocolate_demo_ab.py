"""Audit frozen-policy outcomes of the completed chocolate demonstration study."""

import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
from chocolate_frontier_search import interesting

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.iqn_progress import episode_projection
from smg_rl.profiles import PROFILES
from smg_rl.replay import replay_signature

STATE = Path('.local/chocolate-demo-ab-20261028/state.json')
OUTPUT = Path('docs/assets/performance/chocolate-demo-ab-results-20261028.json')
ROOT = Path.cwd()


def pair_gate(delta, beyond, control_stars, demo_stars):
    """Apply the unchanged preregistered threshold to one paired training seed."""
    return delta >= .5 and beyond >= 1 and demo_stars >= control_stars


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace('\\', '/')


def audit():
    state = json.loads(STATE.read_text(encoding='utf-8'))
    if state['phase'] != 'completed_needs_scientific_review' or len(state['cases']) != 4:
        raise ValueError('Only a complete, unreviewed four-case study can be audited.')
    capture_path = Path(state['capture'])
    if digest(capture_path) != state['capture_sha256']:
        raise ValueError('Demonstration capture manifest changed.')
    capture = json.loads(capture_path.read_text(encoding='utf-8'))
    if capture['state'] != 'completed':
        raise ValueError('Demonstration capture did not complete.')
    for episode in capture['episodes']:
        if digest(episode['file']) != episode['sha256']:
            raise ValueError('Demonstration data changed.')
    source = Path(state['source'])
    source_manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    if (digest(source / 'agent.pt') != source_manifest['agent_sha256']
            or digest(source / 'replay.npz') != source_manifest['replay_sha256']):
        raise ValueError('Original source checkpoint or replay changed.')
    source_agent = torch.load(source / 'agent.pt', map_location='cpu', weights_only=True)
    bounds = json.loads(Path('config/smg1-native-calibration.json').read_text(encoding='utf-8'))['variations']['sweet_sweet']
    result = {'recorded_utc': datetime.now(timezone.utc).isoformat(), 'phase': 'completed',
              'capture_physical_decisions': capture['physical_decisions'],
              'demonstration_training_decisions': sum(e['decisions'] for e in capture['episodes'] if e['purpose'] == 'training'),
              'held_out_timing_decisions': sum(e['decisions'] for e in capture['episodes'] if e['purpose'] == 'held_out_timing_check'),
              'new_online_training_decisions': 0, 'offline_updates': 0, 'evaluation_decisions': 0,
              'evaluation_episodes': 0, 'cases': [], 'pairs': [], 'source_unchanged': True,
              'scope': 'Frozen canonical-start policy assessments; same restored Wii state across action seeds. Grounded geometric projections are not mission-completion percentages.'}
    for case in state['cases']:
        checkpoint = Path(case['checkpoint'])
        manifest = json.loads((checkpoint / 'manifest.json').read_text(encoding='utf-8'))
        config = load_config(case['config'])
        if (case['phase'] != 'completed_reviewed' or case['offline_updates'] != 128
                or case['demo_batches'] != (32 if case['arm'] == 'demonstration' else 0)
                or manifest['signature'] != replay_signature(config)
                or digest(checkpoint / 'agent.pt') != manifest['agent_sha256']
                or manifest['agent_sha256'] != case['checkpoint_agent_sha256']
                or digest(checkpoint / 'replay.npz') != manifest['replay_sha256']
                or manifest['replay_sha256'] != case['checkpoint_replay_sha256']
                or manifest['real_decisions'] != source_manifest['real_decisions']):
            raise ValueError('Case checkpoint, counters or replay identity mismatch.')
        saved = torch.load(checkpoint / 'agent.pt', map_location='cpu', weights_only=True)
        if saved['updates'] != source_agent['updates'] + 128 or saved['real_decisions'] != source_agent['real_decisions']:
            raise ValueError('Offline updates or real-decision ancestry mismatch.')
        result['offline_updates'] += 128
        entry = {'arm': case['arm'], 'seed': case['seed'], 'offline_updates': 128,
                 'demo_batches': case['demo_batches'], 'agent_sha256': manifest['agent_sha256'],
                 'replay_sha256': manifest['replay_sha256'], 'evaluations': []}
        for evaluation in case['evaluations']:
            run = Path(evaluation['run'])
            record = json.loads((run / 'recording.json').read_text(encoding='utf-8'))
            expected = 3 if evaluation['mode'] == 'stochastic' else 1
            epsilon = .1 if evaluation['mode'] == 'stochastic' else 0.
            if (record['state'] != 'completed' or len(record['episodes']) != expected
                    or record['agent_sha256'] != manifest['agent_sha256']
                    or record['physical_training_decisions_added'] != 0 or record['imagined_transitions'] != 0
                    or record['evaluation_epsilon'] != epsilon or not Path(record['video']).is_file()):
                raise ValueError('Frozen evaluation identity, counts or video mismatch.')
            paths = sorted((run / 'worker-00' / 'sweet_sweet').glob('episode-*.csv'))
            if len(paths) != expected:
                raise ValueError('Evaluation is missing a physical episode trace.')
            episodes = []
            for reported, path in zip(record['episodes'], paths):
                with path.open(newline='', encoding='utf-8') as stream:
                    rows = list(csv.DictReader(stream))
                projection = episode_projection(path, bounds, 2500)
                if (rows[0]['start_kind'] != 'full_start' or len(rows) - 1 != reported['decisions']
                        or 'excluded' in projection):
                    raise ValueError('Episode is not an eligible complete canonical trace.')
                star = (rows[-1]['complete'] == 'True' and rows[-1]['dead'] != 'True'
                        and int(rows[-1]['scene_nerve'], 16) in PROFILES['RMGE01'].complete)
                if star != reported['full_start_ram_star']:
                    raise ValueError('RAM terminal differs from recorded star.')
                beyond = False
                for row in rows[1:]:
                    telemetry = SimpleNamespace(position=tuple(float(row[k]) for k in ('x', 'y', 'z')),
                        dead=row['dead'] == 'True', complete=row['complete'] == 'True',
                        grounded=row['grounded'] == 'True', health=float(row['health']))
                    beyond |= interesting(telemetry, (500., -50., -1950.))
                episodes.append({'seed': reported['seed'], 'decisions': reported['decisions'],
                    'outcome': reported['outcome'], 'grounded_projection_percent': projection['geometric_projection_percent'],
                    'opening_fraction': reported['opening_fraction'], 'beyond_opening_grounded_full_health': beyond,
                    'ram_star': star, 'trajectory': relative(path)})
            count = sum(e['decisions'] for e in episodes)
            result['evaluation_decisions'] += count
            result['evaluation_episodes'] += len(episodes)
            entry['evaluations'].append({'mode': evaluation['mode'], 'epsilon': epsilon,
                'mean_grounded_projection_percent': statistics.mean(e['grounded_projection_percent'] for e in episodes),
                'episodes': episodes, 'video': relative(record['video'])})
        if {e['mode'] for e in entry['evaluations']} != {'greedy', 'stochastic'}:
            raise ValueError('Expected both declared evaluation modes.')
        result['cases'].append(entry)
    for seed in (142, 143):
        control = next(c for c in result['cases'] if c['seed'] == seed and c['arm'] == 'control')
        demo = next(c for c in result['cases'] if c['seed'] == seed and c['arm'] == 'demonstration')
        a = next(e for e in control['evaluations'] if e['mode'] == 'stochastic')
        b = next(e for e in demo['evaluations'] if e['mode'] == 'stochastic')
        delta = b['mean_grounded_projection_percent'] - a['mean_grounded_projection_percent']
        control_stars = sum(e['ram_star'] for group in control['evaluations'] for e in group['episodes'])
        demo_stars = sum(e['ram_star'] for group in demo['evaluations'] for e in group['episodes'])
        beyond = sum(e['beyond_opening_grounded_full_health'] for e in b['episodes'])
        result['pairs'].append({'seed': seed, 'control_mean_grounded_projection_percent': a['mean_grounded_projection_percent'],
            'demo_mean_grounded_projection_percent': b['mean_grounded_projection_percent'], 'delta_percentage_points': delta,
            'demo_beyond_count': beyond, 'control_stars': control_stars, 'demo_stars': demo_stars,
            'passed': pair_gate(delta, beyond, control_stars, demo_stars)})
    result['gate'] = 'passed' if all(p['passed'] for p in result['pairs']) else 'failed'
    return result


if __name__ == '__main__':
    data = audit()
    if not atomic_json(OUTPUT, data):
        raise OSError('Could not persist result audit.')
    print(json.dumps({k: data[k] for k in ('gate', 'capture_physical_decisions', 'offline_updates',
                                           'evaluation_decisions', 'evaluation_episodes', 'pairs')}, indent=2))
