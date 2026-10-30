"""Finite SMG1 objective, direction-carry and human-backward curriculum study."""

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import yaml

from smg_rl.actions import advance_action
from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.dqn_trial import alive, audit_block, locate_run, read
from smg_rl.native import NativeDolphin
from smg_rl.resource_disk import require_disk_headroom
from smg_rl.resource_memory import memory_snapshot
from smg_rl.route_transfer import transfer_policy
from smg_rl.windows_pool import windows_pool_tags

ROOT = Path('.local/exploration-study-20261029')
STATE = ROOT / 'state.json'
BANK = ROOT / 'human-bank' / 'bank.json'
DAY = Path('.local/iqn-day-20261029/state.json')
ARM_SECONDS = 14400
ARMS = ('C', 'A', 'B')


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(state):
    state['updated_utc'] = utc()
    if not atomic_json(STATE, state):
        raise OSError('Study state publication failed; no successor may start.')


def resources():
    return {'utc': utc(), 'memory': memory_snapshot(),
            'tags': [t for t in windows_pool_tags() if t['tag'] in ('EtwD', 'Etwr')]}


def probe_controls():
    output = ROOT / 'control-probe.json'
    if output.exists():
        raise FileExistsError('Never duplicate a recorded control fixture.')
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('Emulator ownership conflict.')
    bank = read(BANK)
    entry = next(e for e in bank['entries'] if e['route_waypoint'] == 1)
    target = np.array(read('config/routes/smg1-human-landing-v2.json')['waypoints'][2])
    config = load_config('config/iqn-day-smg1-20261029.yaml')
    config['paths']['user'] = str((ROOT / 'control-profile' / 'template').resolve())
    config['parallel']['workers'] = 1
    report = {'state': 'running', 'bank_sha256': digest(BANK), 'target': target.tolist(),
              'trials': [], 'scope': 'Scripted assisted control fixtures, not learned progress.'}
    backend = None
    deadline = time.monotonic() + 600
    try:
        backend = NativeDolphin(config)
        for profile in ('motor8-v1', 'motor8-carry-v1'):
            config['actions']['profile'] = profile
            for delay in range(1, 9):
                backend.load_state(BANK.parent / entry['file'])
                backend._motor_direction = ()
                trial = {'profile': profile, 'jump_after_decisions': delay, 'samples': [],
                         'target_alive_grounded': False, 'minimum_target_distance_3d': None}
                for step in range(40):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Finite control fixture boundary.')
                    names = ('jump',) if step == delay else ('w',)
                    advance_action(backend, names, config)
                    t = backend.memory.sample()
                    distance = float(np.linalg.norm(np.array(t.position) - target))
                    trial['samples'].append({'decision': step + 1, 'logical_action': list(names),
                                            'telemetry': t.to_dict()})
                    if t.grounded and not t.dead and not t.complete:
                        previous = trial['minimum_target_distance_3d']
                        trial['minimum_target_distance_3d'] = min(distance, previous) if previous is not None else distance
                        trial['target_alive_grounded'] |= distance <= 100 and t.health == entry['health']
                    if t.dead or t.complete or trial['target_alive_grounded']:
                        break
                trial['decisions'] = len(trial['samples'])
                report['trials'].append(trial)
                atomic_json(output, report)
        report['state'] = 'completed'
    except BaseException as exc:
        report.update(state='failed', error=repr(exc))
        raise
    finally:
        if backend:
            backend.close()
        atomic_json(output, report)
    print(json.dumps({'state': report['state'], 'trials': len(report['trials']),
        'clearances': {p: sum(t['target_alive_grounded'] for t in report['trials'] if t['profile'] == p)
                       for p in ('motor8-v1', 'motor8-carry-v1')}}))


def prepare():
    if STATE.exists():
        raise FileExistsError('Existing study; do not duplicate preparations.')
    day = read(DAY)
    if alive(day['supervisor']) or any(alive(j) for j in day['jobs'] if j.get('pid')):
        raise ValueError('The old day allocation has not stopped.')
    last = day['jobs'][-1]
    if last['phase'] != 'partial_reviewed' or not last['review']['passed']:
        raise ValueError('The owner-stopped checkpoint needs an accounting audit.')
    if read(ROOT / 'control-probe.json')['state'] != 'completed':
        raise ValueError('Control fixtures need complete preserved evidence.')
    preparation = read(BANK.parent / 'preparation.json')
    if preparation['state'] != 'completed' or preparation['accepted'] < 3:
        raise ValueError('Validated human anchors are required.')
    source = Path(day['checkpoint'])
    original = load_config('config/iqn-day-smg1-20261029.yaml')
    raw = yaml.safe_load(Path('config/iqn-day-smg1-20261029.yaml').read_text())
    manifest = read(source / 'manifest.json')
    state = {'phase': 'preparing', 'source_checkpoint': str(source),
             'source_hashes': {k: manifest[k] for k in ('agent_sha256', 'replay_sha256')},
             'source_real_decisions': last['review']['real_decisions'],
             'source_attempts': last['review']['completed_attempts'], 'source_day_state_sha256': digest(DAY),
             'bank_sha256': digest(BANK), 'control_probe_sha256': digest(ROOT / 'control-probe.json'),
             'deadline_utc': day['deadline_utc'], 'maximum_arm_seconds': ARM_SECONDS,
             'maximum_workers': 4, 'order': list(ARMS), 'arms': {},
             'plateau_decisions': 1_000_000,
             'protocol': 'C first: B plus backward curriculum; A: corrected objective baseline; B: A plus direction-carry and prepared long jump with8 labels. Each arm has fresh replay/optimizer/target and identical transferred online weights, seed142,250k replay,4h maximum. A is an objective/retention package, not an isolated gamma ablation.',
             'canonical_evaluations': 'One greedy seed11 and three stochastic11/23/47, epsilon0.1,600 decisions at baseline and after each arm. Same Wii state, not independent environments.'}
    save(state)

    for arm in ARMS:
        case = deepcopy(raw)
        case['paths'].update(runs=f'runs/model-free/exploration-study-20261029/{arm}',
                             replay=f'runs/model-free/exploration-study-20261029/{arm}/replay',
                             user=str(ROOT / arm / 'template'))
        case['dqn'].update(gamma=.997, capacity=250000, max_replay_bytes=4_000_000_000)
        case['rewards'].update(timeout=0., revision='discounted-route-bootstrap-study-v1')
        case['regional_reward'].update(mode='discounted_distance_advance',
                                     gamma_per_frame=.997 ** (1 / 12), timeout_mode='truncation_bootstrap')
        case['study'] = {'arm': arm, 'source_real_decisions': state['source_real_decisions'],
                         'source_agent_sha256': manifest['agent_sha256']}
        if arm in ('B', 'C'):
            case['actions']['profile'] = 'motor8-longjump-v1'
        if arm == 'C':
            case['paths'].update(curriculum=str(BANK),
                                 curriculum_progress=str(ROOT / arm / 'curriculum-progress.json'))
            case['curriculum'].update(enabled=True, mode='human_backward_v2', probability=.5,
                                      spec_sha256=digest(BANK))
        config_path = Path(f'config/exploration-study-{arm.lower()}-20261029.yaml')
        with config_path.open('x', encoding='utf-8') as stream:
            yaml.safe_dump(case, stream, sort_keys=False)
        transfer = transfer_policy(source, original, load_config(config_path), ROOT / f'transfer-{arm}',
                                   allow_settings_change=True)
        state['arms'][arm] = {'config': str(config_path), 'config_sha256': digest(config_path),
                              'phase': 'prepared', 'transfer': transfer, 'checkpoint': transfer['destination'],
                              'real_decisions': 0, 'completed_attempts': 0, 'controller_seconds': 0.,
                              'best_full_start_credit': next(iter(read(source.parent / 'status.json')['attempts'].values()))['best_opening_advance'],
                              'last_improvement_real': 0, 'ram_stars_seen': 0, 'frontier_events_seen': 0,
                              'jobs': [], 'evaluations': []}
        save(state)
    state['phase'] = 'prepared'
    save(state)


def validate_motor():
    """Validate the actual eight-label state machine, beyond raw button fixtures."""
    output = ROOT / 'motor-validation-v2.json'
    if output.exists():
        raise FileExistsError('Existing motor validation ledger.')
    bank = read(BANK)
    entry = next(e for e in bank['entries'] if e['route_waypoint'] == 1)
    target = np.array(read('config/routes/smg1-human-landing-v2.json')['waypoints'][2])
    config = load_config('config/iqn-day-smg1-20261029.yaml')
    config['paths']['user'] = str((ROOT / 'motor-validation-profile' / 'template').resolve())
    config['actions']['profile'] = 'motor8-longjump-v1'
    config['parallel']['workers'] = 1
    report = {'state': 'running', 'trials': [], 'training': False,
              'scope': 'Eight-label scripted assisted control verification, no agent learning.'}
    backend = None
    deadline = time.monotonic() + 300
    try:
        backend = NativeDolphin(config)
        for delay in (2, 3, 4):
            backend.load_state(BANK.parent / entry['file'])
            backend._motor_direction, backend._motor_prepare = (), False
            trial = {'jump_after_decisions': delay, 'target_alive_grounded': False, 'samples': []}
            for step in range(40):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Motor verification deadline.')
                names = ('crouch',) if step == delay - 1 else ('jump',) if step == delay else ('w',)
                advance_action(backend, names, config)
                t = backend.memory.sample()
                trial['samples'].append({'logical_controls': names, 'telemetry': t.to_dict()})
                trial['target_alive_grounded'] |= bool(t.grounded is True and not t.dead and not t.complete
                    and t.health == entry['health'] and np.linalg.norm(np.array(t.position) - target) <= 100)
                if t.dead or t.complete or trial['target_alive_grounded']:
                    break
            report['trials'].append(trial)
        if not all(t['target_alive_grounded'] for t in report['trials']):
            raise ValueError('The signed logical motor profile did not reproduce the positive fixture.')
        report['state'] = 'completed'
    except BaseException as exc:
        report.update(state='failed', error=repr(exc))
        raise
    finally:
        if backend:
            backend.close()
        atomic_json(output, report)
    print(json.dumps({'state': report['state'], 'passed': len(report['trials'])}))

def logged(state, job, command):
    job['command'] = command
    save(state)
    env = dict(os.environ)
    env.update({k: '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    with Path(job['stdout']).open('x') as out, Path(job['stderr']).open('x') as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, env=env,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    job.update(phase='running', pid=process.pid, created_time=psutil.Process(process.pid).create_time())
    save(state)
    return process


def wait(state, job, process, maximum):
    start = time.time()
    while process.poll() is None:
        if time.time() - start > maximum:
            from run_iqn_day import terminate_owned
            job['terminated_after_cleanup_bound'] = terminate_owned(job)
            save(state)
            raise TimeoutError('Owned study subprocess exceeded its finite boundary.')
        if job.get('role') == 'training':
            run = locate_run(job)
            if run:
                job['run'] = str(run)
                if (ROOT / 'STOP').exists():
                    (run / 'STOP').touch(exist_ok=True)
        time.sleep(3)
    job.update(exit_code=process.returncode, controller_seconds=time.time() - start)
    save(state)
    if process.returncode or (ROOT / 'STOP').exists():
        raise RuntimeError('Stopped/failed study subprocess; preserve partial data without retry.')


def evaluate(state, arm, label):
    branch = state['arms'][arm]
    checkpoint = Path(branch['checkpoint'])
    before = {name: digest(checkpoint / filename) for name, filename in [('agent', 'agent.pt'), ('replay', 'replay.npz')]}
    for mode in ('greedy', 'stochastic'):
        prefix = f'{arm}-{label}-{mode}'
        job = {'role': 'evaluation', 'label': prefix, 'phase': 'pending',
               'checkpoint': str(checkpoint), 'stdout': str(ROOT / f'{prefix}.stdout.log'),
               'stderr': str(ROOT / f'{prefix}.stderr.log')}
        branch['evaluations'].append(job)
        command = [sys.executable, '-m', 'smg_rl.cli', '--config', branch['config'],
                   'record-dqn', str(checkpoint), '--variation', '0', '--seeds',
                   '11']
        if mode == 'stochastic':
            command += ['23', '47']
        command += ['--max-steps', '600']
        if mode == 'stochastic':
            command += ['--stochastic', '--evaluation-epsilon', '.1']
        process = logged(state, job, command)
        wait(state, job, process, 900)
        results = [p for p in Path(load_config(branch['config'])['paths']['runs']).glob('eval-dqn-video-*/recording.json')
                   if read(p)['checkpoint'] == str(checkpoint)]
        seen = {x.get('recording') for x in branch['evaluations'] if x.get('recording')}
        results = [p for p in results if str(p) not in seen and read(p)['evaluation_mode'] == ('noisynet_epsilon' if mode == 'stochastic' else 'greedy')]
        if len(results) != 1 or read(results[0])['state'] != 'completed':
            raise ValueError('Frozen episode identity/completeness mismatch.')
        record = read(results[0])
        if digest(checkpoint / 'agent.pt') != before['agent'] or digest(checkpoint / 'replay.npz') != before['replay']:
            raise ValueError('Evaluation changed a retained policy or replay.')
        job.update(phase='reviewed', recording=str(results[0]), result=record,
                   checkpoint_hashes_unchanged=True)
        save(state)


def run():
    state = read(STATE)
    if state['phase'] != 'prepared' or (ROOT / 'STOP').exists():
        raise ValueError('Only a fresh prepared study may run.')
    with (ROOT / 'supervisor-lock.json').open('x') as stream:
        json.dump({'pid': os.getpid(), 'created_time': psutil.Process().create_time()}, stream)
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('Unexpected existing emulator.')
    state.update(phase='running', started_utc=utc(),
                 supervisor={'pid': os.getpid(), 'created_time': psutil.Process().create_time()})
    save(state)
    deadline = datetime.fromisoformat(state['deadline_utc']).timestamp()
    if deadline - time.time() < len(ARMS) * ARM_SECONDS + 1800:
        raise ValueError('Not enough remaining absolute time for all declared arm ceilings and evaluation.')
    active = None
    try:
        # Baselines use unchanged source online weights; C has priority for scarce later-region data.
        for arm in ARMS:
            evaluate(state, arm, 'baseline')
            branch = state['arms'][arm]
            branch['phase'] = 'training'
            config = load_config(branch['config'])
            for index in range(12):
                seconds = min(300 if index == 0 else 900 if index == 1 else 1800,
                              int(ARM_SECONDS - branch['controller_seconds']) - 150,
                              int(deadline - time.time()) - 150)
                if seconds < 60:
                    break
                if digest(branch['config']) != branch['config_sha256'] or digest(BANK) != state['bank_sha256']:
                    raise ValueError('Study signatures changed.')
                snapshot = resources()
                if min(snapshot['memory'][k] for k in ('physical_available_bytes', 'commit_available_bytes')) < 6 * 2**30:
                    raise MemoryError('Insufficient physical/commit headroom.')
                require_disk_headroom(config)
                label = f'{arm}-block-{index:02d}'
                job = {'role': 'training', 'label': label, 'phase': 'pending_dispatch', 'config': branch['config'],
                       'resume': branch['checkpoint'], 'source_real': branch['real_decisions'],
                       'source_attempts': branch['completed_attempts'], 'max_seconds': seconds,
                       'stdout': str(ROOT / f'{label}.stdout.log'), 'stderr': str(ROOT / f'{label}.stderr.log'),
                       'resources_before': snapshot}
                branch['jobs'].append(job)
                command = [sys.executable, '-m', 'smg_rl.cli', '--config', branch['config'], 'train-dqn',
                           '--steps', '3072' if index == 0 else '999996', '--max-seconds', str(seconds),
                           '--deadline-utc', state['deadline_utc'], '--resume', branch['checkpoint']]
                active = logged(state, job, command)
                wait(state, job, active, seconds + 180)
                run_path = locate_run(job)
                if run_path is None:
                    raise ValueError('Training did not publish an owned run.')
                status = read(run_path / 'status.json')
                if status['state'] != 'completed':
                    raise RuntimeError(f'Partial stop: {status["stop_reason"]}; no automatic retry.')
                review = audit_block(run_path, config, job['source_real'], job['source_attempts'])
                after = resources()
                growth = sum(t['paged_bytes'] + t['nonpaged_bytes'] for t in after['tags']) - sum(
                    t['paged_bytes'] + t['nonpaged_bytes'] for t in snapshot['tags'])
                job.update(phase='completed_reviewed', review=review, resources_after=after, etw_growth_bytes=growth)
                branch.update(checkpoint=review['checkpoint'], real_decisions=review['real_decisions'],
                              completed_attempts=review['completed_attempts'],
                              controller_seconds=branch['controller_seconds'] + job['controller_seconds'])
                canonical = review['best_collection'].get('sweet_sweet/full_start', {}).get('opening_advance', 0.)
                stars = sum(v['ram_stars'] for v in status['attempts'].values())
                progress_file = config['paths'].get('curriculum_progress')
                frontier_events = len(read(progress_file)['events']) if progress_file and Path(progress_file).exists() else 0
                improved = (canonical > branch['best_full_start_credit'] + 1e-6
                            or stars > branch['ram_stars_seen'] or frontier_events > branch['frontier_events_seen'])
                if improved:
                    branch['last_improvement_real'] = branch['real_decisions']
                branch.update(best_full_start_credit=max(canonical, branch['best_full_start_credit']),
                              ram_stars_seen=stars, frontier_events_seen=frontier_events)
                save(state)
                if growth > 64 * 2**20 or branch['controller_seconds'] > ARM_SECONDS:
                    raise RuntimeError('Resource or finite arm-time boundary requires review.')
                if index == 0 and review['real_decisions'] != 3072:
                    raise ValueError('Health decision budget was incomplete.')
                if branch['real_decisions'] - branch['last_improvement_real'] >= state['plateau_decisions']:
                    branch['early_stop_reason'] = 'one_million_decisions_without_new_route_credit_star_or_frontier'
                    break
            if not branch['jobs']:
                raise TimeoutError('No remaining finite time for this arm.')
            evaluate(state, arm, 'final')
            branch['phase'] = 'completed_reviewed'
            save(state)
        state['phase'] = 'completed_needs_scientific_review'
    except BaseException as exc:
        state.update(phase='needs_review', error=repr(exc))
        raise
    finally:
        if active is not None and active.poll() is None:
            from run_iqn_day import terminate_owned
            state['emergency_cleanup'] = terminate_owned(job)
        source = Path(state['source_checkpoint'])
        state['source_unchanged'] = all(digest(source / filename) == state['source_hashes'][key]
            for key, filename in [('agent_sha256', 'agent.pt'), ('replay_sha256', 'replay.npz')])
        state['finished_utc'] = utc()
        save(state)


def snapshot():
    state = read(STATE)
    result = {k: state.get(k) for k in ('phase', 'started_utc', 'updated_utc', 'deadline_utc', 'error')}
    result['supervisor_alive'] = alive(state['supervisor']) if state.get('supervisor') else False
    result['arms'] = {}
    for arm, branch in state['arms'].items():
        summary = {k: branch[k] for k in ('phase', 'real_decisions', 'completed_attempts', 'controller_seconds', 'checkpoint')}
        if branch['jobs']:
            job = branch['jobs'][-1]
            summary['latest_job'] = {k: job.get(k) for k in ('phase', 'pid', 'created_time', 'run')}
            run_path = locate_run(job)
            if run_path and (run_path / 'status.json').exists():
                status = read(run_path / 'status.json')
                summary['live_status'] = {k: status.get(k) for k in ('state', 'updated_at', 'real_decisions',
                    'optimizer_updates', 'completed_attempts', 'full_start_stars_this_run',
                    'assisted_stars_this_run', 'deaths_this_run', 'epsilon', 'losses', 'stop_reason')}
        result['arms'][arm] = summary
    result['resources'] = resources()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('probe-controls', 'validate-motor', 'prepare', 'run', 'snapshot'))
    args = parser.parse_args()
    {'probe-controls': probe_controls, 'validate-motor': validate_motor,
     'prepare': prepare, 'run': run, 'snapshot': snapshot}[args.mode]()
