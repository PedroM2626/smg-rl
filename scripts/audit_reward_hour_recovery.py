"""Audit the completed reward-hour comparison without collection or fitting."""

import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.dqn_trial import audit_block, locate_run
from smg_rl.human_route_audit import first_segment_progress, load_spec, score_episode
from smg_rl.iqn_progress import episode_projection
from smg_rl.profiles import PROFILES

ORIGINAL = Path('.local/reward-hour-pair-20261028/state.json')
FIRST = Path('.local/reward-hour-recovery-20261028/state.json')
SECOND = Path('.local/reward-hour-recovery2-20261028/state.json')
OUT = Path('docs/assets/performance/reward-hour-pair-final-20261028.json')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    if OUT.exists():
        raise FileExistsError('Final audit evidence cannot be overwritten.')
    original, first, second = read(ORIGINAL), read(FIRST), read(SECOND)
    if second['phase'] != 'completed_needs_review' or digest(FIRST) != second['previous_state_sha256']:
        raise ValueError('Completed recovery and unchanged first recovery ledger required.')
    if digest(ORIGINAL) != first['original_state_sha256']:
        raise ValueError('Original ledger changed after the first recovery.')
    spec_path = Path('config/routes/smg1-human-landing-v2.json')
    if digest(spec_path) != first['original_route_sha256']:
        raise ValueError('The common evaluation route changed.')
    spec = load_spec(spec_path)
    sources = {a: original['arms'][a]['source_checkpoint'] for a in ('control', 'candidate')}
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'phase': 'completed_reviewed',
              'control_overrun_seconds': second['control_overrun_seconds'],
              'common_route_sha256': digest(spec_path), 'arms': {},
              'scope': 'Single paired training seed; real full-start collection and frozen evaluations.',
              'limitations': ['Splitting at recovery checkpoints resets emulators, so collection differs from uninterrupted training.',
                  'Equal elapsed time does not imply equal real decisions.',
                  'Action seeds restore the same Wii state; these are not independent environments.',
                  'First-segment fractions and grounded goal-axis projection are not native mission percentages.',
                  'The fixed first-segment diagnostic can saturate and the strict full-route score has a floor effect.']}
    old_stages = {x['arm']: x for x in original['stages'][:2]}
    continuation = first['stages']
    control_segments = continuation[1]['segments']
    candidate_segments = continuation[0]['segments'] + second['stages'][1]['segments']
    evals = {'control': [(1, original['stages'][0]['evaluations']), (2, second['stages'][0]['evaluations'])],
             'candidate': [(1, first['stages'][0]['evaluations']), (2, second['stages'][1]['evaluations'])]}
    for arm in ('control', 'candidate'):
        cfg = load_config(original['arms'][arm]['config'])
        bounds = read(cfg['paths']['calibration'])['variations']['sweet_sweet']
        source = Path(sources[arm])
        source_manifest = read(source / 'manifest.json')
        expected_source = original['arms'][arm]['source_manifest']
        if (source_manifest != expected_source or digest(source / 'agent.pt') != source_manifest['agent_sha256']
                or digest(source / 'replay.npz') != source_manifest['replay_sha256']):
            raise ValueError(f'{arm} source checkpoint changed.')
        jobs = [old_stages[arm]] + (control_segments if arm == 'control' else candidate_segments)
        real = source_manifest['real_decisions']
        attempts = old_stages[arm]['source_attempts']
        elapsed = 0.0
        blocks = []
        for index, job in enumerate(jobs):
            run = Path(job['run']) if job.get('run') else locate_run(job)
            if run is None:
                raise ValueError('A completed job has no run path.')
            st = read(run / 'status.json')
            resource = arm == 'candidate' and index == 0
            result = audit_block(run, cfg, real, attempts, resource_stopped=resource)
            if st['real_decisions'] != result['real_decisions']:
                raise ValueError('Real counters disagree.')
            blocks.append({'run': str(run), 'stop_reason': st['stop_reason'],
                'elapsed_seconds': st['elapsed_seconds'], 'source_real': real,
                'new_real': st['real_decisions'] - real, 'final_real': st['real_decisions'],
                'new_attempts': st['episodes_this_run'], 'deaths': st['deaths_this_run'],
                'hp_lost': st['nonfatal_hp_lost_this_run'], 'native_stars': st['full_start_stars_this_run'],
                'optimizer_updates_total': st['optimizer_updates'], 'checkpoint': result['checkpoint'],
                'agent_sha256': result['agent_sha256'], 'replay_sha256': read(run / 'checkpoint-final' / 'manifest.json')['replay_sha256'],
                'csv_decision_gap': result['csv_decision_gap']})
            real, attempts = result['real_decisions'], result['completed_attempts']
            elapsed += st['elapsed_seconds']
        if (real != second['arms'][arm]['real_decisions']
                or attempts != second['arms'][arm]['completed_attempts']
                or abs(elapsed - second['arms'][arm]['consumed_seconds']) > 1e-4):
            raise ValueError(f'{arm} totals do not reconcile across recovery ledgers.')
        evaluations = []
        for boundary, group in evals[arm]:
            if len(group) != 2:
                raise ValueError('Both evaluation modes must be present at every boundary.')
            for ev in group:
                if ev['phase'] != 'reviewed':
                    raise ValueError('An evaluation was not reviewed.')
                root = Path(ev['run'])
                rec = read(root / 'recording.json')
                checkpoint = Path(rec['checkpoint'])
                manifest = read(checkpoint / 'manifest.json')
                if (rec['state'] != 'completed' or rec['agent_sha256'] != manifest['agent_sha256']
                        or digest(checkpoint / 'agent.pt') != manifest['agent_sha256']
                        or digest(checkpoint / 'replay.npz') != manifest['replay_sha256']):
                    raise ValueError('Evaluation checkpoint failed identity or integrity.')
                if len(rec['episodes']) != (1 if ev['mode'] == 'greedy' else 3):
                    raise ValueError('Evaluation episode count differs from the protocol.')
                episodes = []
                for i, episode in enumerate(rec['episodes'], 1):
                    path = root / 'worker-00' / 'sweet_sweet' / f'episode-{i:06d}.csv'
                    q = episode_projection(path, bounds, cfg['environment']['max_step_distance'])
                    r = score_episode(path, spec)
                    if 'excluded' in q or r['native_star'] != episode['full_start_ram_star']:
                        raise ValueError('Invalid full-start trajectory or star signal.')
                    with path.open(newline='', encoding='utf-8') as stream:
                        rows = list(csv.DictReader(stream))
                    if r['native_star'] and int(rows[-1]['scene_nerve'], 16) not in PROFILES['RMGE01'].complete:
                        raise ValueError('Invalid native star terminal.')
                    episodes.append({'seed': episode['seed'], 'outcome': episode['outcome'],
                        'decisions': episode['decisions'], 'native_star': r['native_star'],
                        'first_segment_fraction': first_segment_progress(path, spec),
                        'strict_route_fraction': r['route_fraction'],
                        'grounded_goal_projection_percent': q['geometric_projection_percent'],
                        'hp_lost': sum(int(row['health_lost']) for row in rows[1:]),
                        'trajectory': str(path), 'trajectory_sha256': digest(path)})
                evaluations.append({'boundary': boundary, 'mode': ev['mode'], 'episodes': episodes,
                    'first_segment_mean': statistics.mean(e['first_segment_fraction'] for e in episodes),
                    'grounded_projection_mean_percent': statistics.mean(e['grounded_goal_projection_percent'] for e in episodes),
                    'video': rec['video'], 'checkpoint_agent_sha256': rec['agent_sha256']})
        report['arms'][arm] = {'source_real': source_manifest['real_decisions'], 'final_real': real,
            'new_real': real - source_manifest['real_decisions'], 'elapsed_seconds': elapsed,
            'new_attempts': sum(x['new_attempts'] for x in blocks), 'deaths': sum(x['deaths'] for x in blocks),
            'hp_lost': sum(x['hp_lost'] for x in blocks), 'native_stars': sum(x['native_stars'] for x in blocks),
            'final_optimizer_updates': blocks[-1]['optimizer_updates_total'],
            'blocks': blocks, 'evaluations': evaluations}
    OUT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({a: {k: v[k] for k in ('new_real', 'elapsed_seconds', 'new_attempts', 'deaths', 'native_stars')}
                      for a, v in report['arms'].items()}, indent=2))


if __name__ == '__main__':
    main()
