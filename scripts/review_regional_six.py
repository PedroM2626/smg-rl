"""Read-only completion audit of the explicitly allocated six-collector blocks."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import psutil
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.profiles import PROFILES, game_id
from smg_rl.replay import SequenceReplay, replay_signature
from smg_rl.route import RouteProgress


def audit_episode(path, config, route):
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert rows and rows[0]['action'] == 'reset'
    initial, final = rows[0], rows[-1]
    decisions = rows[1:]
    assert all(float(r['intrinsic_reward']) == 0 and float(r['progress_reward']) == 0 for r in rows)
    assert int(final['emulated_frames']) == 12 * len(decisions)
    kind = initial['start_kind']
    assert kind in ('full_start', 'curriculum')
    star = final['outcome'] == 'success' and final['complete'] == 'True' and final['dead'] != 'True'
    star &= int(final['scene_nerve'], 16) in PROFILES[game_id(config)].complete
    relative = max(float(r['start_relative_route_progress']) for r in rows)
    assert -.000001 <= relative <= 1.000001
    start_phi = float(initial['shaping_potential'])
    discount = config['regional_reward']['gamma_per_frame'] ** 12
    closed_shaping = sum(discount ** index * float(r['regional_shaping_reward'])
                         for index, r in enumerate(decisions))
    discounted_return = sum(discount ** index * float(r['reward']) for index, r in enumerate(decisions))
    closure_error = None
    if final['outcome'] in ('death', 'timeout'):
        closure_error = abs(closed_shaping + start_phi)
        assert closure_error < 1e-7 and discounted_return < 0
        assert float(final['shaping_potential']) == 0
    return {'csv': str(path), 'sha256': digest(path), 'variation': initial['variation'],
            'start_kind': kind, 'anchor': initial['frontier_state'], 'outcome': final['outcome'],
            'decisions': len(decisions), 'physical_frames': int(final['emulated_frames']),
            'ram_star': bool(star), 'full_start_star': bool(star and kind == 'full_start'),
            'opening_exit': any(r['route_prefix_exited'] == 'True' for r in rows),
            'maximum_start_relative_opening_fraction': relative,
            'maximum_start_relative_distance': relative * route.total_length,
            'maximum_geometric_projection_percent': 100 * max(float(r['best_progress']) for r in rows),
            'nonfatal_hp_lost': sum(int(r['health_lost']) for r in decisions
                                   if r['dead'] != 'True' and r['complete'] != 'True'),
            'damage_penalty': sum(float(r['damage_penalty']) for r in decisions),
            'unknown_contact_samples': sum(r['grounded'] not in ('True', 'False') for r in decisions),
            'discounted_return': discounted_return, 'terminal_shaping_closure_error': closure_error,
            'policy_age': 'Per-episode actor-update count unavailable; maxima include prior exploration.'}


def alive(controller):
    try:
        return abs(psutil.Process(controller['pid']).create_time() - controller['created_time']) < .01
    except psutil.Error:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', default='.local/regional-six-state.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    state = json.loads(Path(args.state).read_text())
    evidence = {'observed_at': datetime.now(timezone.utc).isoformat(),
                'scope': 'Completed allocated blocks only. No collection, fitting or evaluation dispatch.',
                'games': {}, 'missions': [], 'episodes': []}
    for game, job in state['jobs'].items():
        config = load_config(job['config'])
        run = Path(job['run'])
        status = json.loads((run / 'status.json').read_text())
        controller = json.loads((run / 'controller.json').read_text())
        assert not alive(controller) and status['state'] == 'completed'
        assert status['real_decisions'] == job['target_count']
        assert not (run / 'error.txt').exists()
        checkpoint = run / 'world-final.pt'
        metadata = json.loads(checkpoint.with_suffix('.json').read_text())
        assert metadata['sha256'] == digest(checkpoint)
        counts = torch.load(checkpoint, map_location='cpu', weights_only=True)['counters']
        assert all(counts[k] == status[k] for k in counts)
        replay = SequenceReplay(config['paths']['replay'], replay_signature(config)).refresh()
        assert replay['transitions'] == job['target_count'] and not replay['rejected']
        route = RouteProgress(json.loads(Path(config['paths']['route']).read_text()))
        episodes = []
        for source in (Path(job['source_run']), run):
            for path in sorted(source.glob('worker-*/*/episode-*.csv')):
                episodes.append(audit_episode(path, config, route))
        assert sum(e['decisions'] for e in episodes) == job['target_count']
        evidence['episodes'].extend(episodes)
        grouped = defaultdict(list)
        for episode in episodes:
            grouped[(episode['variation'], episode['start_kind'])].append(episode)
        for (variation, kind), items in grouped.items():
            best = max(items, key=lambda e: e['maximum_start_relative_distance'])
            evidence['missions'].append({'game': game, 'variation': variation, 'start_kind': kind,
                'episodes': len(items), 'decisions': sum(e['decisions'] for e in items),
                'outcomes': dict(Counter(e['outcome'] for e in items)),
                'ram_stars': sum(e['ram_star'] for e in items),
                'full_start_stars': sum(e['full_start_star'] for e in items),
                'opening_exits': sum(e['opening_exit'] for e in items),
                'nonfatal_hp_lost': sum(e['nonfatal_hp_lost'] for e in items),
                'damage_penalty': sum(e['damage_penalty'] for e in items),
                'maximum_start_relative_distance': best['maximum_start_relative_distance'],
                'maximum_start_relative_opening_fraction': best['maximum_start_relative_opening_fraction'],
                'maximum_geometric_projection_percent': max(e['maximum_geometric_projection_percent'] for e in items),
                'best_trajectory': best['csv']})
        evidence['games'][game] = {'run': str(run), 'checkpoint': str(checkpoint),
            'checkpoint_sha256': digest(checkpoint), 'counts': counts, 'disk_replay': replay,
            'controller_alive': False, 'final_status': status,
            'main_additional_decisions_per_second': status['decisions_per_second'],
            'stale_final_replay_snapshot': status['diagnostics']['replay']['transitions'],
            'note': 'Final status precedes worker close/partial-chunk flush; audited disk replay is authoritative.'}
    figure, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    for axis, mission in zip(axes, [m for m in evidence['missions'] if m['start_kind'] == 'full_start']):
        with Path(mission['best_trajectory']).open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        config = load_config(state['jobs'][mission['game']]['config'])
        points = json.loads(Path(config['paths']['route']).read_text())['waypoints']
        axis.plot([p[0] for p in points], [p[2] for p in points], '--', color='gray', label='Mapped opening')
        axis.plot([float(r['x']) for r in rows], [float(r['z']) for r in rows], label='Observed path')
        axis.scatter(float(rows[0]['x']), float(rows[0]['z']), color='green', label='Canonical start')
        axis.scatter(float(rows[-1]['x']), float(rows[-1]['z']), color='red', marker='x', label='Last sample')
        axis.set_title(f"{mission['game'].upper()} / {mission['variation']}\n{mission['maximum_start_relative_distance']:.1f} grounded arc units")
        axis.set_xlabel('World X')
        axis.set_ylabel('World Z')
        axis.invert_yaxis()
        axis.set_aspect('equal', adjustable='datalim')
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    figure.suptitle('Best full-start collection trajectories — opening coverage, not mission percentage or policy evaluation')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix('.png'), dpi=150)
    plt.close(figure)
    atomic_json(output, evidence)
    print(json.dumps({'counts': {g: d['counts'] for g, d in evidence['games'].items()},
                      'missions': evidence['missions'], 'audited_episodes': len(evidence['episodes'])}))


if __name__ == '__main__':
    main()
