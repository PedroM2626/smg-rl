"""Run one additional hour per reward, in reviewed sequential 30-minute blocks."""

import ast
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import psutil
import run_reward_scale_ab as helper
import yaml

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.dqn_trial import audit_block, locate_run
from smg_rl.human_route_audit import episode_landmark_candidates, load_spec, score_episode
from smg_rl.iqn_progress import episode_projection
from smg_rl.replay import replay_signature
from smg_rl.resource_disk import require_disk_headroom
from smg_rl.resource_memory import require_memory_headroom

os.chdir(Path(__file__).resolve().parents[1])
ROOT = Path('.local/reward-hour-pair-20261028')
STATE = ROOT / 'state.json'
STOP = ROOT / 'STOP'
SOURCE = Path('.local/human-route-ab-fixed-20261028/state.json')
ROUTE = Path('config/routes/smg1-human-landing-v2.json')
helper.ROOT, helper.STATE, helper.STOP = ROOT, STATE, STOP


def save(state):
    if not atomic_json(STATE, state):
        raise OSError('Study state could not be persisted.')


def prepare():
    if ROOT.exists():
        raise FileExistsError('A study already exists; review it before any dispatch.')
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('An existing Dolphin process requires ownership review.')
    require_memory_headroom()
    old = json.loads(SOURCE.read_text(encoding='utf-8'))
    if old['phase'] != 'scientific_review_complete':
        raise ValueError('The earlier A/B must be complete and reviewed.')
    ROOT.mkdir(parents=True)
    state = {'phase': 'preparing', 'created_utc': datetime.now(timezone.utc).isoformat(),
             'source_state': str(SOURCE), 'source_state_sha256': digest(SOURCE),
             'common_route': str(ROUTE), 'common_route_sha256': digest(ROUTE),
             'design': {'training_seed': 142, 'arms': ['control', 'candidate'], 'workers': 6,
                        'additional_training_seconds_per_arm': 3600, 'block_seconds': 1800,
                        'decision_ceiling_per_block': 999996,
                        'evaluation': 'One greedy and three stochastic full-start episodes at each reviewed half, action seeds 11/23/47, epsilon 0.1, max 600 decisions each.',
                        'predeclared_metrics': 'RAM-confirmed full-start stars; first-segment projection (later withdrawn as invalid); strict human-route fraction; grounded start-to-goal projection, HP, deaths, timeout, attempts, real decisions and optimizer updates.',
                        'current_evaluation_metrics': 'Ordered sampled XYZ landmark candidates pending visual review; unvalidated reward-tracker fraction; grounded goal projection as geometry only.',
                        'warning': 'Single training seed; equal wall time does not ensure equal decisions or optimizer updates. Frozen seeds share the same Wii state. No causal or mission-percentage claim.'},
             'arms': {}, 'stages': []}
    save(state)
    for arm in ('control', 'candidate'):
        previous = next(j for j in old['runs'] if j['arm'] == arm and j['seed'] == 142)
        if previous['phase'] != 'completed_reviewed':
            raise ValueError('The prior source checkpoint was not reviewed.')
        checkpoint = Path(previous['review']['checkpoint'])
        manifest = json.loads((checkpoint / 'manifest.json').read_text(encoding='utf-8'))
        if (digest(checkpoint / 'agent.pt') != manifest['agent_sha256']
                or digest(checkpoint / 'replay.npz') != manifest['replay_sha256']):
            raise ValueError('The prior checkpoint or replay failed integrity.')
        raw = yaml.safe_load(Path(previous['config']).read_text(encoding='utf-8'))
        raw['paths'].update(runs=f'runs/model-free/reward-hour-pair-20261028/{arm}',
                            replay=f'runs/model-free/reward-hour-pair-20261028/{arm}/replay',
                            user=str(ROOT / f'{arm}-template'))
        path = Path('.local') / f'reward-hour-pair-{arm}-20261028.yaml'
        helper.write_new(path, yaml.safe_dump(raw, sort_keys=False))
        config = load_config(path)
        if (Path(config['root']) != Path.cwd() or not (Path(config['paths']['assets']) / 'RMGE01.s01').is_file()
                or replay_signature(config) != manifest['signature'] or config['dqn'] != manifest['settings']):
            raise ValueError('Continuation profile is not an exact signed resume.')
        require_disk_headroom(config)
        state['arms'][arm] = {'config': str(path), 'source_checkpoint': str(checkpoint),
                              'source_manifest': manifest, 'current_checkpoint': str(checkpoint),
                              'real_decisions': manifest['real_decisions'],
                              'completed_attempts': previous['review']['completed_attempts'],
                              'prior_evaluations': [v['run'] for v in previous['evaluations']]}
        save(state)
    for block in (1, 2):
        for arm in ('control', 'candidate'):
            state['stages'].append({'arm': arm, 'block': block, 'phase': 'prepared'})
    state['phase'] = 'prepared'
    save(state)
    print(json.dumps({'state': str(STATE), 'stages': len(state['stages'])}), flush=True)


def evaluate(state, stage, mode):
    arm = state['arms'][stage['arm']]
    label = f"{stage['arm']}-block{stage['block']}-{mode}"
    job = {'phase': 'pending', 'mode': mode, 'stdout': str(ROOT / f'{label}.out.log'),
           'stderr': str(ROOT / f'{label}.err.log')}
    command = [sys.executable, '-m', 'smg_rl.cli', '--config', arm['config'], 'record-dqn',
               stage['review']['checkpoint'], '--variation', '0', '--max-steps', '600', '--seeds']
    command += ['11'] if mode == 'greedy' else ['11', '23', '47', '--stochastic', '--evaluation-epsilon', '0.1']
    job['command'] = command
    stage.setdefault('evaluations', []).append(job)
    save(state)
    process = helper.call_logged(command, job['stdout'], job['stderr'])
    helper.wait_process(process, job, state)
    run = None
    for line in Path(job['stdout']).read_text(encoding='utf-8').splitlines():
        try:
            value = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, dict) and 'evaluation_run' in value:
            run = Path(value['evaluation_run'])
            break
    if run is None:
        raise ValueError('Evaluation did not publish its run.')
    result = json.loads((run / 'recording.json').read_text(encoding='utf-8'))
    checkpoint = Path(stage['review']['checkpoint'])
    if (result['state'] != 'completed' or len(result['episodes']) != (1 if mode == 'greedy' else 3)
            or result['agent_sha256'] != stage['review']['agent_sha256']
            or digest(checkpoint / 'agent.pt') != stage['review']['agent_sha256']):
        raise ValueError('Frozen evaluation or weights failed review.')
    specification = load_spec(ROUTE)
    config = load_config(arm['config'])
    bounds = json.loads(Path(config['paths']['calibration']).read_text(encoding='utf-8'))['variations']['sweet_sweet']
    scores = []
    for index, episode in enumerate(result['episodes'], 1):
        path = run / 'worker-00' / 'sweet_sweet' / f'episode-{index:06d}.csv'
        route = score_episode(path, specification)
        projection = episode_projection(path, bounds, config['environment']['max_step_distance'])
        if 'excluded' in projection or route['native_star'] != episode['full_start_ram_star']:
            raise ValueError('Evaluation telemetry or star failed validation.')
        scores.append({'seed': episode['seed'], 'outcome': episode['outcome'],
                       'decisions': episode['decisions'], 'native_star': route['native_star'],
                       'observed_landmark_candidates': episode_landmark_candidates(
                           path, load_spec('config/evaluation/smg1-human-landmarks-v1.json'))['candidate_landmarks'],
                       'reward_tracker_fraction_unvalidated': route['route_fraction'],
                       'grounded_goal_projection_percent': projection['geometric_projection_percent'],
                       'trajectory': str(path), 'trajectory_sha256': digest(path),
                       'telemetry': episode['telemetry']})
    job.update(phase='reviewed', run=str(run), scores=scores, video=result['video'],
               decisions=sum(x['decisions'] for x in scores), agent_sha256=result['agent_sha256'])
    save(state)


def run():
    state = json.loads(STATE.read_text(encoding='utf-8'))
    if state['phase'] != 'prepared' or STOP.exists():
        raise ValueError('A started, stopped or failed study may not be resumed automatically.')
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('An existing Dolphin process requires ownership review.')
    state.update(phase='running', supervisor={'pid': os.getpid(), 'created_time': psutil.Process().create_time()})
    save(state)
    try:
        for stage in state['stages']:
            arm = state['arms'][stage['arm']]
            config = load_config(arm['config'])
            require_memory_headroom()
            require_disk_headroom(config)
            label = f"{stage['arm']}-block{stage['block']}-train"
            stage.update(phase='pending', config=arm['config'], source_real=arm['real_decisions'],
                         source_attempts=arm['completed_attempts'], resume=arm['current_checkpoint'],
                         stdout=str(ROOT / f'{label}.out.log'), stderr=str(ROOT / f'{label}.err.log'))
            stage['command'] = [sys.executable, '-m', 'smg_rl.cli', '--config', arm['config'], 'train-dqn',
                                '--steps', '999996', '--max-seconds', '1800', '--resume', stage['resume']]
            save(state)
            process = helper.call_logged(stage['command'], stage['stdout'], stage['stderr'])
            helper.wait_process(process, stage, state)
            run_path = locate_run(stage)
            if run_path is None:
                raise ValueError('Training did not publish its owned run.')
            review = audit_block(run_path, config, stage['source_real'], stage['source_attempts'])
            status = json.loads((run_path / 'status.json').read_text(encoding='utf-8'))
            if status['stop_reason'] != 'wall_clock_budget' or status['real_decisions'] <= stage['source_real']:
                raise ValueError('The block did not finish its authorized wall-clock budget cleanly.')
            stage.update(phase='training_reviewed', run=str(run_path), review=review,
                         status_summary={k: status[k] for k in ('stop_reason', 'elapsed_seconds',
                             'optimizer_updates', 'real_decisions', 'episodes_this_run',
                             'deaths_this_run', 'nonfatal_hp_lost_this_run', 'full_start_stars_this_run')})
            arm.update(current_checkpoint=review['checkpoint'], real_decisions=review['real_decisions'],
                       completed_attempts=review['completed_attempts'])
            save(state)
            evaluate(state, stage, 'greedy')
            evaluate(state, stage, 'stochastic')
            stage['phase'] = 'completed_reviewed'
            save(state)
        state.update(phase='completed_needs_review', finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        state.update(phase='failed_needs_review', error=repr(exc))
        save(state)
        raise
    save(state)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Use prepare, run or snapshot.')
    if sys.argv[1] == 'prepare':
        prepare()
    elif sys.argv[1] == 'run':
        run()
    elif sys.argv[1] == 'snapshot':
        print(STATE.read_text(encoding='utf-8'))
    else:
        raise SystemExit('Use prepare, run or snapshot.')
