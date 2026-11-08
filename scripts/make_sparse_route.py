"""Derive a sparser schema-5 route from a human landing route by keeping a subset of its waypoints.

Usage: python scripts/make_sparse_route.py SOURCE.json COUNT OUT.json [HEAD]
COUNT is the number of waypoints after the start (so COUNT + 1 points are kept). The kept points
are real human landing points chosen so that their cumulative route lengths are as even as
possible; the first and the last point are always kept.
"""
import json
import sys
from pathlib import Path

import numpy as np

from smg_rl.checkpoints import digest


def sparse_route(specification, count, head=0):
    """Keep `count` waypoints after the start; the first `head` of them are the source's own first ones."""
    points = np.asarray(specification['waypoints'], dtype=float)
    if (type(count) is not int or not 1 <= count < len(points) - 1 or type(head) is not int
            or not 0 <= head < count):
        raise ValueError('The kept waypoint count must be smaller than the source route and exceed the head.')
    cumulative = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(points[:, [0, 2]], axis=0), axis=1))])
    chosen = list(range(head + 1))
    rest = count - head
    base, span = cumulative[head], cumulative[-1] - cumulative[head]
    for k in range(1, rest):
        target = base + span * k / rest
        # Nearest unused point strictly after the previous choice, leaving room for the rest.
        low, high = chosen[-1] + 1, len(points) - 1 - (rest - k)
        chosen.append(int(low + np.argmin(np.abs(cumulative[low:high + 1] - target))))
    chosen.append(len(points) - 1)
    result = dict(specification)
    result['waypoints'] = [specification['waypoints'][i] for i in chosen]
    result['name'] = f"{specification['name']} (sparse, {count} waypoints, {head} dense at the start)"
    result['scope'] = specification['scope'] + ' Sparse subset of the source landing points.'
    result['kept_source_indices'] = chosen
    return result


if __name__ == '__main__':
    source, count, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    head = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    spec = json.loads(source.read_text())
    sparse = sparse_route(spec, count, head)
    sparse['derived_from_sha256'] = digest(source)
    out.write_text(json.dumps(sparse, indent=1) + '\n')
    print(json.dumps({'out': str(out), 'sha256': digest(out), 'kept': sparse['kept_source_indices']}))
