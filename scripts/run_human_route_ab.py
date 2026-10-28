"""Owner-authorized finite, paired SMG1 opening versus human-route study."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import run_reward_scale_ab as runner
import yaml

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.human_route_audit import load_spec, score_episode
from smg_rl.resource_memory import require_memory_headroom
from smg_rl.route_transfer import transfer_policy

os.chdir(Path(__file__).resolve().parents[1])
ROOT = Path('.local/human-route-ab-fixed-20261028')
STATE = ROOT / 'state.json'
SOURCE = 'runs/model-free/smg1-six-two-hour/dqn-20261027-114555-611503/checkpoint-final'
SOURCE_CONFIG = 'config/smg1-six-two-hour.yaml'
ROUTE = 'config/routes/smg1-human-landing-v2.json'
runner.ROOT, runner.STATE, runner.STOP = ROOT, STATE, ROOT / 'STOP'


def prepare():
    if ROOT.exists():
        raise FileExistsError('Study already exists; inspect it instead of duplicating.')
    require_memory_headroom()
    ROOT.mkdir(parents=True)
    source = load_config(SOURCE_CONFIG)
    templates = {'control': yaml.safe_load(Path(SOURCE_CONFIG).read_text()),
                 'candidate': yaml.safe_load(Path('config/human-route-smg1-landing-v2.yaml').read_text())}
    state = {'phase': 'preparing', 'created_utc': datetime.now(timezone.utc).isoformat(),
             'source': SOURCE, 'source_config': SOURCE_CONFIG, 'route': ROUTE,
             'route_sha256': digest(ROUTE), 'runs': [],
             'design': {'seeds': [142, 143], 'arms': ['control', 'candidate'],
                        'workers': 6, 'new_real_per_arm': 6144, 'seconds_per_arm_cap': 900,
                        'evaluation': 'One greedy action seed 11 and three stochastic 11/23/47, epsilon 0.1, max 600 real decisions each.',
                        'gate': 'Candidate stochastic mean full-route fraction exceeds control by >=0.005 in both paired training seeds, no reduction in RAM-confirmed stars and no serious faults. Exploratory, not significance testing.',
                        'scope': 'Same source weights, fresh optimizer/target/replay, eight actions and canonical starts. The route and normalization both change.'}}
    runner.save(state)
    for seed in (142, 143):
        for arm in ('control', 'candidate'):
            raw = yaml.safe_load(yaml.safe_dump(templates[arm]))
            raw['training']['seed'] = seed
            raw['curriculum']['enabled'] = False
            raw['rewards']['revision'] = f'human-route-ab-{arm}-v1'
            raw['paths'].update(runs=f'runs/model-free/human-route-ab-fixed-20261028/{arm}-{seed}',
                                replay=f'runs/model-free/human-route-ab-fixed-20261028/{arm}-{seed}/replay',
                                user=str(ROOT / f'{arm}-{seed}-template'))
            # load_config treats the config's grandparent as the workspace root.
            config = Path('.local') / f'human-route-ab-fixed-{arm}-{seed}.yaml'
            runner.write_new(config, yaml.safe_dump(raw, sort_keys=False))
            resolved = load_config(config)
            if Path(resolved['root']) != Path.cwd() or not (Path(resolved['paths']['assets']) / 'RMGE01.s01').is_file():
                raise ValueError('Generated profile is not rooted at the verified workspace/assets.')
            transfer = transfer_policy(SOURCE, source, resolved, ROOT / f'transfer-{arm}-{seed}')
            state['runs'].append({'arm': arm, 'seed': seed, 'config': str(config),
                                  'transfer': transfer, 'phase': 'prepared'})
            runner.save(state)
    state['phase'] = 'prepared'
    runner.save(state)
    print(json.dumps({'state': str(STATE), 'runs': len(state['runs'])}), flush=True)


def analyze():
    state = json.loads(STATE.read_text(encoding='utf-8'))
    if state['phase'] != 'completed_needs_scientific_review':
        raise ValueError('All four reviewed blocks are required before analysis.')
    if digest(ROUTE) != state['route_sha256']:
        raise ValueError('Evaluation route changed during the study.')
    source_manifest = json.loads((Path(SOURCE) / 'manifest.json').read_text())
    if (digest(Path(SOURCE) / 'agent.pt') != source_manifest['agent_sha256']
            or digest(Path(SOURCE) / 'replay.npz') != source_manifest['replay_sha256']):
        raise ValueError('Original source checkpoint changed.')
    specification = load_spec(ROUTE)
    scores = []
    for job in state['runs']:
        if job['phase'] != 'completed_reviewed' or job['review']['real_decisions'] != 6144:
            raise ValueError('Training budget or accounting review did not complete.')
        for ev in job['evaluations']:
            if ev['phase'] != 'reviewed':
                raise ValueError('Incomplete evaluation.')
            run = Path(ev['run'])
            result = ev['result']
            if result['agent_sha256'] != job['review']['agent_sha256']:
                raise ValueError('Evaluation did not use the reviewed weights.')
            episodes = []
            for i, episode in enumerate(result['episodes'], 1):
                csv = run / 'worker-00' / 'sweet_sweet' / f'episode-{i:06d}.csv'
                score = score_episode(csv, specification)
                if score['native_star'] != episode['full_start_ram_star']:
                    raise ValueError('Native completion disagrees with trajectory.')
                episodes.append({**score, 'seed': episode['seed'], 'outcome': episode['outcome'],
                                 'trajectory': str(csv), 'telemetry': episode['telemetry']})
            scores.append({'arm': job['arm'], 'training_seed': job['seed'], 'mode': ev['mode'],
                           'episodes': episodes, 'video': result['video']})
    pairs = []
    for seed in (142, 143):
        c = next(x for x in scores if x['training_seed'] == seed and x['arm'] == 'control' and x['mode'] == 'stochastic')
        t = next(x for x in scores if x['training_seed'] == seed and x['arm'] == 'candidate' and x['mode'] == 'stochastic')
        cm = sum(x['route_fraction'] for x in c['episodes']) / 3
        tm = sum(x['route_fraction'] for x in t['episodes']) / 3
        cs = sum(x['native_star'] for result in scores if result['training_seed'] == seed
                 and result['arm'] == 'control' for x in result['episodes'])
        ts = sum(x['native_star'] for result in scores if result['training_seed'] == seed
                 and result['arm'] == 'candidate' for x in result['episodes'])
        pairs.append({'seed': seed, 'control_mean': cm, 'candidate_mean': tm, 'delta': tm - cm,
                      'control_stars': cs, 'candidate_stars': ts,
                      'passed': tm - cm >= .005 and ts >= cs})
    state['scientific_review'] = {'gate': 'passed' if all(x['passed'] for x in pairs) else 'failed',
                                  'pairs': pairs, 'scores': scores,
                                  'limitation': 'Action seeds share the same Wii savestate; route fraction is not native mission progress.'}
    state['phase'] = 'scientific_review_complete'
    runner.save(state)
    print(json.dumps(state['scientific_review'], indent=2), flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Use prepare, run, analyze or snapshot.')
    if sys.argv[1] == 'prepare':
        prepare()
    elif sys.argv[1] == 'run':
        runner.run()
    elif sys.argv[1] == 'analyze':
        analyze()
    elif sys.argv[1] == 'snapshot':
        print(STATE.read_text(encoding='utf-8'))
    else:
        raise SystemExit('Use prepare, run, analyze or snapshot.')
