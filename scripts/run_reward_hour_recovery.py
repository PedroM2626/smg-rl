"""Once-only owner-authorized recovery of the interrupted reward comparison."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import psutil
import run_reward_hour_pair as pair

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.dqn_trial import audit_block, locate_run
from smg_rl.resource_disk import require_disk_headroom
from smg_rl.resource_memory import memory_snapshot, require_memory_headroom

os.chdir(Path(__file__).resolve().parents[1])
ROOT = Path('.local/reward-hour-recovery-20261028')
STATE = ROOT / 'state.json'
STOP = ROOT / 'STOP'
ORIGINAL = Path('.local/reward-hour-pair-20261028/state.json')
PARTIAL = Path('runs/model-free/reward-hour-pair-20261028/candidate/dqn-20261028-154214-457197')
pair.ROOT, pair.STATE, pair.STOP = ROOT, STATE, STOP
pair.helper.ROOT, pair.helper.STATE, pair.helper.STOP = ROOT, STATE, STOP


def save(state):
    if not atomic_json(STATE, state):
        raise OSError('Recovery state could not be recorded.')


def prepare():
    if ROOT.exists():
        raise FileExistsError('Recovery already exists; never charge the partial budget twice.')
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('Existing Dolphin process requires ownership review.')
    snapshot = require_memory_headroom()
    if snapshot['physical_available_bytes'] < 6 * 2**30:
        raise MemoryError('Recovery requires at least 6 GiB free physical RAM before dispatch.')
    old = json.loads(ORIGINAL.read_text(encoding='utf-8'))
    if old['phase'] != 'failed_needs_review' or old['stages'][0]['phase'] != 'completed_reviewed':
        raise ValueError('Original interrupted state is not eligible for once-only recovery.')
    partial_job = old['stages'][1]
    partial = audit_block(PARTIAL, load_config(partial_job['config']), partial_job['source_real'],
                          partial_job['source_attempts'], resource_stopped=True)
    partial_status = json.loads((PARTIAL / 'status.json').read_text())
    control_review = old['stages'][0]['review']
    control_time = old['stages'][0]['status_summary']['elapsed_seconds']
    candidate_time = partial_status['elapsed_seconds']
    if not (0 < control_time < 3600 and 0 < candidate_time < 1800):
        raise ValueError('Consumed time is outside the original allocation.')
    arms = old['arms']
    arms['control'].update(current_checkpoint=control_review['checkpoint'],
                           real_decisions=control_review['real_decisions'],
                           completed_attempts=control_review['completed_attempts'], consumed_seconds=control_time)
    arms['candidate'].update(current_checkpoint=str((PARTIAL / 'checkpoint-final').resolve()),
                             real_decisions=partial['real_decisions'],
                             completed_attempts=partial['completed_attempts'], consumed_seconds=candidate_time)
    state = {'phase': 'prepared', 'created_utc': datetime.now(timezone.utc).isoformat(),
             'original_state': str(ORIGINAL), 'original_state_sha256': digest(ORIGINAL),
             'original_route_sha256': old['common_route_sha256'], 'arms': arms,
             'partial_review': partial, 'partial_status': {k: partial_status[k] for k in (
                 'state', 'stop_reason', 'elapsed_seconds', 'source_count', 'real_decisions',
                 'optimizer_updates', 'episodes_this_run', 'deaths_this_run', 'full_start_stars_this_run')},
             'control_first_evaluations': old['stages'][0]['evaluations'],
             'limits': {'total_seconds_per_arm': 3600, 'first_candidate_boundary': 1800,
                        'max_segment_seconds': 600, 'first_recovery_health_seconds': 120,
                        'shutdown_reserve_seconds': 20, 'max_workers': 6},
             'note': 'Every saved elapsed time including cleanup is charged; short collector restarts reset environments.',
             'stages': [{'arm': 'candidate', 'block': 1, 'target_consumed_seconds': 1800,
                         'phase': 'prepared', 'segments': []},
                        {'arm': 'control', 'block': 2, 'target_consumed_seconds': 3600,
                         'phase': 'prepared', 'segments': []},
                        {'arm': 'candidate', 'block': 2, 'target_consumed_seconds': 3600,
                         'phase': 'prepared', 'segments': []}]}
    ROOT.mkdir(parents=True)
    save(state)
    print(json.dumps({'state': str(STATE), 'control_consumed': control_time,
                      'candidate_consumed': candidate_time}), flush=True)


def run():
    state = json.loads(STATE.read_text(encoding='utf-8'))
    if state['phase'] != 'prepared' or STOP.exists():
        raise ValueError('Recovery already started, stopped or failed.')
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('Existing Dolphin process requires ownership review.')
    state.update(phase='running', supervisor={'pid': os.getpid(),
                                               'created_time': psutil.Process().create_time()})
    save(state)
    try:
        serial = 0
        for stage in state['stages']:
            arm = state['arms'][stage['arm']]
            config = load_config(arm['config'])
            stage['phase'] = 'running'
            save(state)
            while True:
                remaining = stage['target_consumed_seconds'] - arm['consumed_seconds']
                if remaining <= 30:
                    break
                serial += 1
                limit = min(120 if serial == 1 else 600, remaining - 20)
                if limit <= 0:
                    break
                mem = require_memory_headroom()
                if mem['physical_available_bytes'] < 6 * 2**30:
                    raise MemoryError('Physical RAM below recovery pre-dispatch reserve; no successor launched.')
                require_disk_headroom(config)
                label = f"{stage['arm']}-block{stage['block']}-segment{serial:02d}"
                job = {'phase': 'pending', 'label': label, 'config': arm['config'],
                       'source_real': arm['real_decisions'], 'source_attempts': arm['completed_attempts'],
                       'resume': arm['current_checkpoint'], 'max_seconds': limit,
                       'memory_before': mem, 'stdout': str(ROOT / f'{label}.out.log'),
                       'stderr': str(ROOT / f'{label}.err.log')}
                job['command'] = [sys.executable, '-m', 'smg_rl.cli', '--config', arm['config'],
                                  'train-dqn', '--steps', '999996', '--max-seconds', str(limit),
                                  '--resume', job['resume']]
                stage['segments'].append(job)
                save(state)
                process = pair.helper.call_logged(job['command'], job['stdout'], job['stderr'])
                pair.helper.wait_process(process, job, state)
                run_path = locate_run(job)
                if run_path is None:
                    raise ValueError('Recovery segment did not publish its run.')
                review = audit_block(run_path, config, job['source_real'], job['source_attempts'])
                status = json.loads((run_path / 'status.json').read_text())
                if status['stop_reason'] != 'wall_clock_budget' or status['real_decisions'] <= job['source_real']:
                    raise ValueError('Recovery segment did not end cleanly at its wall boundary.')
                arm['consumed_seconds'] += status['elapsed_seconds']
                if arm['consumed_seconds'] > 3600:
                    raise ValueError('Elapsed recovery time exceeded the original allocation.')
                arm.update(current_checkpoint=review['checkpoint'], real_decisions=review['real_decisions'],
                           completed_attempts=review['completed_attempts'])
                job.update(phase='reviewed', run=str(run_path), review=review,
                           status_summary={k: status[k] for k in ('elapsed_seconds', 'stop_reason',
                               'source_count', 'real_decisions', 'optimizer_updates', 'episodes_this_run',
                               'deaths_this_run', 'nonfatal_hp_lost_this_run', 'full_start_stars_this_run',
                               'memory_guard_snapshot')}, memory_after=memory_snapshot())
                save(state)
            if not stage['segments']:
                raise ValueError('A recovery boundary had no reviewed segment.')
            stage['review'] = stage['segments'][-1]['review']
            stage['phase'] = 'training_reviewed'
            save(state)
            pair.evaluate(state, stage, 'greedy')
            pair.evaluate(state, stage, 'stochastic')
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
