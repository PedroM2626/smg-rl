"""Continue only the unspent candidate budget after checkpoint-cleanup overrun."""

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
ROOT = Path('.local/reward-hour-recovery2-20261028')
STATE = ROOT / 'state.json'
STOP = ROOT / 'STOP'
PREVIOUS = Path('.local/reward-hour-recovery-20261028/state.json')
pair.ROOT, pair.STATE, pair.STOP = ROOT, STATE, STOP
pair.helper.ROOT, pair.helper.STATE, pair.helper.STOP = ROOT, STATE, STOP


def segment_limit(consumed, ceiling=3600.0, reserve=120.0, maximum=600.0):
    """Leave a conservative reserve for collector close and checkpoint I/O."""
    remaining = ceiling - consumed
    return max(0.0, min(maximum, remaining - reserve)) if remaining > reserve + 10 else 0.0


def save(state):
    if not atomic_json(STATE, state):
        raise OSError('Recovery state could not be recorded.')


def prepare():
    if ROOT.exists():
        raise FileExistsError('Recovery already exists; never repeat a segment or budget.')
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('Existing Dolphin processes require ownership review.')
    snapshot = require_memory_headroom()
    if snapshot['physical_available_bytes'] < 6 * 2**30:
        raise MemoryError('Insufficient physical RAM for a recovery dispatch.')
    old = json.loads(PREVIOUS.read_text(encoding='utf-8'))
    if old['phase'] != 'failed_needs_review' or old.get('error') != "ValueError('Elapsed recovery time exceeded the original allocation.')":
        raise ValueError('The prior state is not the reviewed cleanup-overrun case.')
    stage = old['stages'][1]
    job = stage['segments'][-1]
    if job['phase'] != 'running' or job.get('exit_code') != 0:
        raise ValueError('The last control segment is not a clean completed process.')
    run = locate_run(job)
    review = audit_block(run, load_config(job['config']), job['source_real'], job['source_attempts'])
    status = json.loads((run / 'status.json').read_text(encoding='utf-8'))
    if status['stop_reason'] != 'wall_clock_budget':
        raise ValueError('Control did not finish at the authorized wall boundary.')
    arms = old['arms']
    consumed = arms['control']['consumed_seconds']
    previous_total = sum([old['stages'][1]['segments'][0]['status_summary']['elapsed_seconds'],
                          old['stages'][1]['segments'][1]['status_summary']['elapsed_seconds'],
                          status['elapsed_seconds'],
                          json.loads(Path(old['original_state']).read_text())['stages'][0]['status_summary']['elapsed_seconds']])
    if not (3600 < consumed < 3660 and abs(consumed - previous_total) < 1e-4):
        raise ValueError('Control consumed-time accounting does not reconcile.')
    arms['control'].update(current_checkpoint=review['checkpoint'], real_decisions=review['real_decisions'],
                           completed_attempts=review['completed_attempts'])
    if not 0 < arms['candidate']['consumed_seconds'] < 3600:
        raise ValueError('Candidate has no remaining authorized time.')
    state = {'phase': 'prepared', 'created_utc': datetime.now(timezone.utc).isoformat(),
             'previous_state': str(PREVIOUS), 'previous_state_sha256': digest(PREVIOUS),
             'original_state': old['original_state'], 'arms': arms,
             'control_final_segment_review': review,
             'control_overrun_seconds': consumed - 3600,
             'note': 'Control collection stopped on budget but cleanup exceeded its reserve; no further control training. Candidate has an increased shutdown reserve.',
             'limits': {'candidate_total_seconds': 3600, 'shutdown_reserve_seconds': 120,
                        'max_segment_seconds': 600, 'max_workers': 6},
             'stages': [{'arm': 'control', 'block': 2, 'phase': 'training_reviewed', 'review': review},
                        {'arm': 'candidate', 'block': 2, 'phase': 'prepared', 'segments': []}]}
    ROOT.mkdir(parents=True)
    save(state)
    print(json.dumps({'state': str(STATE), 'control_overrun_seconds': consumed - 3600,
                      'candidate_consumed_seconds': arms['candidate']['consumed_seconds']}), flush=True)


def run():
    state = json.loads(STATE.read_text(encoding='utf-8'))
    if state['phase'] != 'prepared' or STOP.exists():
        raise ValueError('Recovery already started or stopped.')
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('Existing Dolphin processes require ownership review.')
    state.update(phase='running', supervisor={'pid': os.getpid(), 'created_time': psutil.Process().create_time()})
    save(state)
    try:
        control = state['stages'][0]
        pair.evaluate(state, control, 'greedy')
        pair.evaluate(state, control, 'stochastic')
        control['phase'] = 'completed_reviewed'
        save(state)
        stage = state['stages'][1]
        arm = state['arms']['candidate']
        config = load_config(arm['config'])
        stage['phase'] = 'running'
        save(state)
        serial = 0
        while (limit := segment_limit(arm['consumed_seconds'])) > 0:
            serial += 1
            mem = require_memory_headroom()
            if mem['physical_available_bytes'] < 6 * 2**30:
                raise MemoryError('Physical RAM below recovery dispatch reserve.')
            require_disk_headroom(config)
            label = f'candidate-block2-recovery2-segment{serial:02d}'
            job = {'phase': 'pending', 'label': label, 'config': arm['config'],
                   'source_real': arm['real_decisions'], 'source_attempts': arm['completed_attempts'],
                   'resume': arm['current_checkpoint'], 'max_seconds': limit, 'memory_before': mem,
                   'stdout': str(ROOT / f'{label}.out.log'), 'stderr': str(ROOT / f'{label}.err.log')}
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
                raise ValueError('Candidate did not end cleanly at its wall boundary.')
            arm['consumed_seconds'] += status['elapsed_seconds']
            if arm['consumed_seconds'] > 3600:
                raise ValueError('Candidate elapsed time exceeded its original allocation.')
            arm.update(current_checkpoint=review['checkpoint'], real_decisions=review['real_decisions'],
                       completed_attempts=review['completed_attempts'])
            job.update(phase='reviewed', run=str(run_path), review=review,
                       status_summary={k: status[k] for k in ('elapsed_seconds', 'stop_reason',
                           'source_count', 'real_decisions', 'optimizer_updates', 'episodes_this_run',
                           'deaths_this_run', 'nonfatal_hp_lost_this_run', 'full_start_stars_this_run',
                           'memory_guard_snapshot')}, memory_after=memory_snapshot())
            save(state)
        if not stage['segments']:
            raise ValueError('No candidate continuation was reviewed.')
        stage.update(phase='training_reviewed', review=stage['segments'][-1]['review'])
        save(state)
        pair.evaluate(state, stage, 'greedy')
        pair.evaluate(state, stage, 'stochastic')
        stage['phase'] = 'completed_reviewed'
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
