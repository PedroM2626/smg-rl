"""Score recorded full-start traces using one fixed, signed route specification."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from smg_rl.route import RouteProgress


def score_episode(path, specification):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or rows[0]['action'] != 'reset' or rows[0]['start_kind'] != 'full_start':
        raise ValueError('A recorded canonical reset is required for this score.')

    def sample(row):
        def boolean(name):
            if row[name] not in ('True', 'False'):
                raise ValueError(f'Unknown {name} telemetry.')
            return row[name] == 'True'

        return SimpleNamespace(position=tuple(float(row[k]) for k in ('x', 'y', 'z')),
                               health=int(row['health']) if row['health'] else None,
                               grounded=boolean('grounded'), dead=boolean('dead'),
                               complete=boolean('complete'))

    route = RouteProgress(specification)
    route.reset(sample(rows[0]))
    if not route.eligible:
        raise ValueError('Recorded reset does not match the full route start.')
    for row in rows[1:]:
        route.update(sample(row))
    return {'route_fraction': route.best, 'eligible': bool(route.eligible),
            'route_exit': bool(route.exited), 'decisions': len(rows) - 1,
            'native_star': rows[-1]['complete'] == 'True',
            'death': rows[-1]['dead'] == 'True'}


def load_spec(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def first_segment_progress(path, specification, corridor=750.0, height_tolerance=500.0,
                           max_step=2500.0):
    """Diagnostic grounded high-water projection on only the first human segment.

    This score is not an RL reward and is not full-course completion. The wide
    corridor permits distinct paths and therefore does not prove route traversal.
    """
    if not all(np.isfinite(v) and v > 0 for v in (corridor, height_tolerance, max_step)):
        raise ValueError('Diagnostic tolerances must be positive and finite.')
    with Path(path).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or rows[0]['action'] != 'reset' or rows[0]['start_kind'] != 'full_start':
        raise ValueError('A recorded canonical reset is required for this score.')
    start, end = np.asarray(specification['waypoints'][:2], dtype=float)
    axis = (end - start)[[0, 2]]
    length2 = float(axis @ axis)
    if not np.isfinite(length2) or length2 <= 0:
        raise ValueError('Invalid first segment.')
    initial = np.array([float(rows[0][k]) for k in ('x', 'y', 'z')])
    if not np.isfinite(initial).all() or np.linalg.norm(initial - start) > specification['waypoint_radius']:
        raise ValueError('Episode does not begin at the canonical route start.')
    previous, best = initial, 0.0
    for row in rows[1:]:
        pos = np.array([float(row[k]) for k in ('x', 'y', 'z')])
        if not np.isfinite(pos).all() or np.linalg.norm(pos - previous) > max_step:
            raise ValueError('Nonfinite or discontinuous position in diagnostic trajectory.')
        previous = pos
        if row['dead'] != 'False' or row['complete'] != 'False' or row['grounded'] != 'True':
            continue
        t = float(np.clip((pos - start)[[0, 2]] @ axis / length2, 0, 1))
        nearest = start + t * (end - start)
        if (np.linalg.norm((pos - nearest)[[0, 2]]) <= corridor
                and abs(pos[1] - nearest[1]) <= height_tolerance):
            best = max(best, t)
    return best
