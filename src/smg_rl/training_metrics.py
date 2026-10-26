"""Read-only episode telemetry export; never changes collection or rewards."""

import csv
import json
import time
from pathlib import Path

import numpy as np
import psutil
from torch.utils.tensorboard import SummaryWriter

from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.dqn_trial import STATE, STOP, alive, locate_run, read
from smg_rl.profiles import game_id
from smg_rl.route import RouteProgress

EXPORT_STATE = Path('.local/impala-telemetry-state.json')
EXPORT_STOP = Path('.local/impala-telemetry.STOP')


def episode_metrics(rows, route_length, max_step_distance):
    """Observed world-unit distances, excluding terminal jumps/discontinuities.

    Path distance includes backtracking and normal airborne motion. Ordered,
    grounded route advance is a separate metric already validated by the env.
    Neither metric is a native mission-completion percentage or a meter scale.
    """
    if len(rows) < 2 or rows[0]['action'] != 'reset':
        raise ValueError('Expected an actual episode with its reset baseline.')
    paths = np.zeros(2)
    first = previous = last = None
    excluded, discontinuous = 0, False
    for row in rows:
        point = np.array([float(row[k]) for k in ('x', 'y', 'z')])
        valid = np.isfinite(point).all() and row['dead'] != 'True' and row['complete'] != 'True'
        if not valid:
            previous = None
            excluded += 1
            discontinuous |= row['dead'] != 'True' and row['complete'] != 'True'
            continue
        if first is None:
            first = point
        if previous is not None:
            delta = point - previous
            if np.linalg.norm(delta) <= max_step_distance:
                paths += [np.linalg.norm(delta[[0, 2]]), np.linalg.norm(delta)]
            else:
                excluded += 1
                discontinuous = True
        previous = last = point
    advance = max(float(row['start_relative_route_progress']) for row in rows)
    if not np.isfinite(advance) or not -1e-8 <= advance <= 1.000001:
        raise ValueError('Invalid start-relative verified route advance.')
    reward = sum(float(row['reward']) for row in rows[1:])
    if not np.isfinite(reward):
        raise ValueError('Nonfinite episode reward.')
    result = {'episode_reward': reward, 'path_distance_xz': float(paths[0]),
            'path_distance_xyz': float(paths[1]),
            'valid_route_advance': max(0., advance) * route_length,
            'opening_advance_percent': 100 * max(0., advance),
            'excluded_position_samples': excluded}
    if not discontinuous and first is not None:
        result['net_displacement_xz'] = float(np.linalg.norm((last - first)[[0, 2]]))
    return result


class MetricsExporter:
    def __init__(self, state_path=EXPORT_STATE):
        self.path = Path(state_path)
        self.data = read(self.path) if self.path.exists() else {'schema': 1, 'groups': {}, 'exported_attempts': 0}
        self.writers = {}

    def export(self, training):
        new = 0
        for game, branch in training['games'].items():
            for job in branch['jobs']:
                run = (Path(job['run']) if job.get('run') else
                       locate_run(job) if Path(job['stdout']).exists() else None)
                if run is None or not (run / 'attempts.jsonl').exists():
                    continue
                config = load_config(job['config'])
                route = RouteProgress(read(config['paths']['route']))
                logdir = Path(config['paths']['tensorboard']) / 'telemetry' / game_id(config) / 'simple-impala'
                writer = self.writers.setdefault(str(logdir), None)
                if writer is None:
                    writer = self.writers[str(logdir)] = SummaryWriter(str(logdir), flush_secs=10)
                    writer.add_text('metric_definitions', 'Read-only recorded episodes. Distances are world units, not meters. '
                        'Path distance includes backtracking; valid_route_advance is ordered grounded opening credit. '
                        'Full-start and assisted attempts remain separate; collection is not frozen evaluation.')
                for line in (run / 'attempts.jsonl').read_text().splitlines():
                    try:
                        attempt = json.loads(line)
                    except ValueError:
                        break  # An active writer may not have finished its last line yet.
                    mission, kind = attempt['mission'], attempt['start_kind']
                    key = f'{game}/{mission}/{kind}'
                    group = self.data['groups'].setdefault(key, {'last_attempt': 0, 'recent': [], 'best_route': 0.})
                    if attempt['attempt'] <= group['last_attempt']:
                        continue
                    if attempt['attempt'] != group['last_attempt'] + 1:
                        raise ValueError(f'Missing attempt before {key}/{attempt["attempt"]}; refusing misleading axes.')
                    path = run / f'worker-{attempt["worker"]:02d}' / mission / f'episode-{attempt["worker_episode"]:06d}.csv'
                    with path.open(newline='', encoding='utf-8') as stream:
                        rows = list(csv.DictReader(stream))
                    metrics = episode_metrics(rows, route.total_length, config['environment']['max_step_distance'])
                    if abs(metrics['episode_reward'] - attempt['return']) > 1e-4:
                        raise ValueError('Recorded return differs from the episode CSV.')
                    group['recent'] = (group['recent'] + [metrics])[-100:]
                    group['best_route'] = max(group['best_route'], metrics['valid_route_advance'])
                    values = {**metrics, 'best_valid_route_advance': group['best_route']}
                    for name in ('episode_reward', 'path_distance_xz', 'valid_route_advance'):
                        values[f'{name}_mean_last100'] = float(np.mean([m[name] for m in group['recent']]))
                    for axis, step in [('by_attempt', attempt['attempt']), ('by_real_decision', attempt['real_decisions'])]:
                        for name, value in values.items():
                            writer.add_scalar(f'{axis}/{mission}/{kind}/{name}', value, step,
                                              walltime=float(rows[-1]['time']))
                    group['last_attempt'] = attempt['attempt']
                    self.data['exported_attempts'] += 1
                    new += 1
        for writer in self.writers.values():
            writer.flush()
        self.data.update(updated_at=time.time(), training_phase=training['phase'], new_attempts_last_pass=new)
        if not atomic_json(self.path, self.data):
            raise OSError('Could not persist telemetry export progress.')
        return new

    def close(self):
        for writer in self.writers.values():
            writer.close()


def export_metrics(watch=False, max_seconds=None, interval=15, training_state=STATE):
    if watch and (max_seconds is None or not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError('Watching requires a finite positive wall-clock budget.')
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError('The refresh interval must be positive and finite.')
    if EXPORT_STATE.exists():
        owner = read(EXPORT_STATE).get('controller')
        if owner and alive(owner):
            raise ValueError('A telemetry exporter is already active; do not duplicate it.')
    exporter = MetricsExporter()
    process = psutil.Process()
    exporter.data['controller'] = {'pid': process.pid, 'created_time': process.create_time()}
    exporter.data['state'] = 'running'
    exporter.data['training_state'] = str(training_state)
    started = time.perf_counter()
    try:
        while True:
            training = read(training_state)
            exporter.export(training)
            if (not watch or training['phase'] in ('completed', 'stopped', 'needs_review') or STOP.exists()
                    or EXPORT_STOP.exists()
                    or time.perf_counter() - started >= max_seconds):
                break
            time.sleep(max(0., min(interval, max_seconds - (time.perf_counter() - started))))
        exporter.data['state'] = 'completed'
    except BaseException as exc:
        exporter.data.update(state='failed', error=repr(exc))
        raise
    finally:
        exporter.close()
        atomic_json(EXPORT_STATE, exporter.data)
    print(json.dumps({'state': exporter.data['state'], 'exported_attempts': exporter.data['exported_attempts'],
                      'logdirs': list(exporter.writers)}))
