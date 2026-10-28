"""Score recorded full-start traces using one fixed, signed route specification."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from smg_rl.profiles import PROFILES
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
                           max_step=2500.0, *, allow_unsafe_projection=False):
    """Reproduce a withdrawn projection for historical audit only.

    The projection reported positive values for positions on the starting
    platform. It is not an observed visit or a valid progress measurement.
    """
    if not allow_unsafe_projection:
        raise ValueError('Withdrawn metric: a projection does not prove that the route was reached.')
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


def observed_landmark_candidates(rows, specification):
    """Find ordered, sampled XYZ proximity to human-observed landmarks.

    Matches are candidates for visual review, not verified traversal or a
    percentage of the level. Interpolated samples never count as visits.
    """
    if specification.get('schema') != 'observed-landmarks-v1':
        raise ValueError('Unsupported observed-landmark specification.')
    start = np.asarray(specification['start'], dtype=float)
    targets = specification['landmarks']
    if start.shape != (3,) or not np.isfinite(start).all() or not targets:
        raise ValueError('Invalid observed-landmark route.')
    positions = [np.asarray(t['position'], dtype=float) for t in targets]
    if (any(p.shape != (3,) or not np.isfinite(p).all() for p in positions)
            or any(not np.isfinite(t['radius_3d']) or not 0 < t['radius_3d'] <= 100 for t in targets)
            or not 0 < specification['start_radius_3d'] <= 1
            or not 0 < specification['max_step_distance_3d'] <= 500
            or any(a['human_sample'] >= b['human_sample'] for a, b in zip(targets, targets[1:]))
            or any(np.linalg.norm(a - b) <= targets[i]['radius_3d'] + targets[i + 1]['radius_3d']
                   for i, (a, b) in enumerate(zip(positions, positions[1:])))):
        raise ValueError('Invalid or overly broad observed-landmark bounds.')
    if not rows:
        raise ValueError('A full-start observation sequence is required.')
    initial = np.array([float(rows[0][k]) for k in ('x', 'y', 'z')])
    if (not np.isfinite(initial).all() or np.linalg.norm(initial - start) > specification['start_radius_3d']
            or rows[0]['grounded'] != 'True' or rows[0]['dead'] != 'False' or rows[0]['complete'] != 'False'):
        raise ValueError('The sequence does not begin at its canonical route start.')
    previous, matches = initial, []
    for i, row in enumerate(rows[1:], 1):
        point = np.array([float(row[k]) for k in ('x', 'y', 'z')])
        if (not np.isfinite(point).all()
                or np.linalg.norm(point - previous) > specification['max_step_distance_3d']):
            raise ValueError('Nonfinite or discontinuous landmark observation.')
        previous = point
        if any(row[k] not in ('True', 'False') for k in ('grounded', 'dead', 'complete')):
            raise ValueError('Landmark telemetry must include valid boolean flags.')
        if row['dead'] == 'True' or row['complete'] == 'True' or row['grounded'] != 'True':
            continue
        if len(matches) == len(targets):
            continue
        target = targets[len(matches)]
        distance = float(np.linalg.norm(point - positions[len(matches)]))
        if distance <= target['radius_3d']:
            matches.append({'name': target['name'], 'observation_index': i,
                            'position': point.tolist(), 'distance_3d': distance,
                            'human_sample': target['human_sample']})
    return matches


def episode_landmark_candidates(path, specification):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if (not rows or rows[0].get('action') != 'reset' or rows[0].get('start_kind') != 'full_start'
            or any(row.get('variation') != specification['variation'] for row in rows)):
        raise ValueError('The matching canonical mission reset is required.')
    matches = observed_landmark_candidates(rows, specification)
    terminal = rows[-1]
    star = terminal['complete'] == 'True' and terminal['dead'] == 'False'
    if star and int(terminal['scene_nerve'], 16) not in PROFILES[specification['game_id']].complete:
        raise ValueError('A completion needs a matching native RAM terminal.')
    eligible = [row for row in rows if row['grounded'] == 'True'
                and row['dead'] == 'False' and row['complete'] == 'False']
    landmark = np.asarray(specification['landmarks'][0]['position'], dtype=float)
    distances = [float(np.linalg.norm(np.array([float(row[k]) for k in ('x', 'y', 'z')]) - landmark))
                 for row in eligible]
    return {'candidate_landmarks': matches, 'native_star': star,
            'minimum_grounded_3d_distance_to_first_landmark': min(distances) if distances else None,
            'scope': 'XYZ proximity candidates require visual review; not a mission percentage.'}
