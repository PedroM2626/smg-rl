"""Conservative, offline route-extension proposals from actual episode traces."""

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from smg_rl.checkpoints import digest
from smg_rl.route import RouteProgress

DIRECTION_ACTIONS = {'w', 'a', 's', 'd'}


def point_segment_distance(point, a, b):
    point, a, b = (np.asarray(x, dtype=float)[[0, 2]] for x in (point, a, b))
    axis = b - a
    t = float(np.clip((point - a) @ axis / (axis @ axis), 0, 1))
    return float(np.linalg.norm(point - (a + t * axis)))


def proposals_for_episode(path, spec):
    """Return only continuous, alive-grounded movement beyond the old route.

    The narrow detector intentionally rejects jumps, uncertain contact, damage,
    discontinuities, backward movement along prior segments and partial starts.
    It is a candidate heuristic, not proof of a traversable or complete mission.
    """
    with Path(path).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if (not rows or rows[0].get('start_kind') != 'full_start'
            or rows[-1].get('outcome') not in ('success', 'death', 'timeout')):
        return []
    points = np.asarray(spec['waypoints'], dtype=float)
    old_end = points[-1]
    prefix = []
    result = []
    begun = False
    for row in rows[1:]:
        if not begun:
            if row.get('route_prefix_exited') != 'True':
                continue
            begun = True
        try:
            point = np.array([float(row[k]) for k in ('x', 'y', 'z')])
            health = float(row['health'])
            step = int(row['step'])
        except (ValueError, KeyError):
            break
        if (not np.isfinite(point).all() or not np.isfinite(health) or health <= 0
                or row.get('grounded') != 'True' or row.get('dead') != 'False'
                or row.get('complete') != 'False' or row.get('health_lost', '0') not in ('0', '0.0', '')):
            break
        if prefix and np.linalg.norm(point - prefix[-1][0]) > 150:
            break
        prefix.append((point, row.get('action'), step))
        if len(prefix) < 3 or np.linalg.norm((prefix[-1][0] - prefix[0][0])[[0, 2]]) < 50:
            continue
        if sum(action in DIRECTION_ACTIONS for _, action, _ in prefix[-3:]) < 2:
            continue
        old_distance = min(point_segment_distance(point, a, b) for a, b in zip(points[:-1], points[1:]))
        from_end = float(np.linalg.norm((point - old_end)[[0, 2]]))
        if not 150 <= old_distance or not 150 <= from_end <= 450:
            continue
        if any(point_segment_distance(p, old_end, point) > 100 or abs(p[1] - point[1]) > 40
               for p, _, _ in prefix):
            continue
        result.append({'position': point.tolist(), 'step': step, 'episode': str(Path(path).resolve()),
                       'episode_sha256': digest(path), 'history_samples': len(prefix),
                       'distance_from_existing_route': old_distance, 'distance_from_old_endpoint': from_end})
        break  # One earliest independently reproducible extension per physical episode.
    return result


def discover(paths, spec):
    """Require two physically distinct full-start traces near the same candidate."""
    RouteProgress(spec)  # Reject malformed base map.
    candidates = []
    for path in paths:
        candidates.extend(proposals_for_episode(path, spec))
    for i, first in enumerate(candidates):
        for second in candidates[i + 1:]:
            if first['episode'] == second['episode']:
                continue
            a, b = np.asarray(first['position']), np.asarray(second['position'])
            if np.linalg.norm((a - b)[[0, 2]]) <= 80 and abs(a[1] - b[1]) <= 30:
                proposal = {'candidate': first, 'confirmation': second,
                        'independent_game_states': False,
                        'scope': 'Two separate physical full-start episodes from potentially identical game state; conservative sampled-contact heuristic.'}
                expanded = extended_spec(spec, proposal)
                old_length = RouteProgress(spec).total_length
                credit = [replay_route_distance(item['episode'], expanded) for item in (first, second)]
                if all(value > old_length + 1 for value in credit):
                    proposal['replayed_route_distances'] = credit
                    return proposal
    return None


def replay_route_distance(path, specification):
    """Check the actual route implementation can award both saved crossings."""
    with Path(path).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))

    def telemetry(row):
        return SimpleNamespace(position=[float(row[k]) for k in ('x', 'y', 'z')],
                               health=float(row['health']), grounded=row['grounded'] == 'True',
                               dead=row['dead'] == 'True', complete=row['complete'] == 'True')

    route = RouteProgress(specification)
    route.reset(telemetry(rows[0]))
    for row in rows[1:]:
        route.update(telemetry(row))
    return float(route.best * route.total_length)


def extended_spec(spec, proposal):
    """Append an actual observed point; never mutate the input or old waypoints."""
    out = json.loads(json.dumps(spec))
    out['waypoints'].append(proposal['candidate']['position'])
    out['name'] = spec['name'] + ' + validated extension'
    out['scope'] = 'Opening and conservatively repeated extension; not a complete course or native mission percentage.'
    route = RouteProgress(out)
    if route.total_length >= 50000:
        raise ValueError('Route reached the fixed reward-normalization limit; no silent saturation.')
    return out


def canonical_bytes(spec):
    return (json.dumps(spec, indent=2, sort_keys=True) + '\n').encode('utf-8')


def spec_sha256(spec):
    return hashlib.sha256(canonical_bytes(spec)).hexdigest()
