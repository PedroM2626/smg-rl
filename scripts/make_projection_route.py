"""Build a schema-6 projection route from a recorded human trajectory.

Usage: python scripts/make_projection_route.py TRAJECTORY.csv SPACING GAME_ID VARIATION OUT.json

The trajectory's x, y, z columns are resampled every SPACING units of 3D arc length (the first and the
last point are kept). The route pays by projecting a position onto this polyline inside a corridor, so the
waypoints only set how finely progress is counted.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

from smg_rl.checkpoints import digest


def resample(points, spacing):
    points = np.asarray(points, dtype=float)
    kept, since = [points[0]], 0.0
    for previous, point in zip(points, points[1:]):
        since += float(np.linalg.norm(point - previous))
        if since >= spacing:
            # Consecutive kept points must differ horizontally, as every route schema requires.
            if np.linalg.norm((point - kept[-1])[[0, 2]]) >= 1.0:
                kept.append(point)
                since = 0.0
    last = points[-1]
    if np.linalg.norm(last - kept[-1]) > 0 and np.linalg.norm((last - kept[-1])[[0, 2]]) >= 1.0:
        kept.append(last)
    return np.asarray(kept)


def projection_route(rows, spacing, game_id, variation, corridor_radius=500.0):
    kept = resample([[float(r['x']), float(r['y']), float(r['z'])] for r in rows], spacing)
    return {'schema': 6, 'game_id': game_id, 'variation': variation,
            'name': f'Human full-start projection route, {spacing:g} unit spacing',
            'scope': 'Dense polyline for projection credit; the route end is not a RAM star.',
            'waypoints': [[round(float(v), 3) for v in p] for p in kept],
            'waypoint_radius': 100, 'corridor_radius': corridor_radius, 'height_tolerance': 500,
            'exit_confirmations': 1, 'max_sample_distance': 1000, 'requires_grounded': True,
            'requires_full_health': False, 'lookahead': 1500.0, 'lookbehind': 300.0}


if __name__ == '__main__':
    source, spacing, game_id, variation, out = Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4], Path(sys.argv[5])
    with source.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    route = projection_route(rows, spacing, game_id, variation)
    route['source_trajectory_sha256'] = digest(source)
    out.write_text(json.dumps(route, indent=1) + '\n')
    print(json.dumps({'out': str(out), 'waypoints': len(route['waypoints']), 'sha256': digest(out)}))
