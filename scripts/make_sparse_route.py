"""Derive a sparser schema-5 route from a human landing route by keeping a subset of its waypoints.

Usage: python scripts/make_sparse_route.py SOURCE.json COUNT OUT.json
COUNT is the number of waypoints after the start (so COUNT + 1 points are kept). The kept points
are real human landing points chosen so that their cumulative route lengths are as even as
possible; the first and the last point are always kept.
"""
import json
import sys
from pathlib import Path

import numpy as np

from smg_rl.checkpoints import digest


def sparse_route(specification, count):
    points = np.asarray(specification['waypoints'], dtype=float)
    if type(count) is not int or not 1 <= count < len(points) - 1:
        raise ValueError('The kept waypoint count must be smaller than the source route.')
    cumulative = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(points[:, [0, 2]], axis=0), axis=1))])
    chosen = [0]
    for k in range(1, count):
        target = cumulative[-1] * k / count
        # Nearest unused point strictly after the previous choice, leaving room for the rest.
        low, high = chosen[-1] + 1, len(points) - 1 - (count - k)
        chosen.append(int(low + np.argmin(np.abs(cumulative[low:high + 1] - target))))
    chosen.append(len(points) - 1)
    result = dict(specification)
    result['waypoints'] = [specification['waypoints'][i] for i in chosen]
    result['name'] = f"{specification['name']} (sparse, {count} waypoints)"
    result['scope'] = specification['scope'] + ' Sparse subset of the source landing points.'
    result['kept_source_indices'] = chosen
    return result


if __name__ == '__main__':
    source, count, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    spec = json.loads(source.read_text())
    sparse = sparse_route(spec, count)
    sparse['derived_from_sha256'] = digest(source)
    out.write_text(json.dumps(sparse, indent=1) + '\n')
    print(json.dumps({'out': str(out), 'sha256': digest(out), 'kept': sparse['kept_source_indices']}))
