"""Score recorded full-start traces using one fixed, signed route specification."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

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
