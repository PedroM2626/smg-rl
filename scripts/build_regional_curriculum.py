"""Publish an immutable local curriculum from reviewed, reachable probe results."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from smg_rl.checkpoints import digest
from smg_rl.diagnostics import atomic_json
from smg_rl.memory import Telemetry
from smg_rl.route import RouteProgress


def build(report_path, route_path, destination):
    report_path, route_path, destination = map(Path, (report_path, route_path, destination))
    report = json.loads(report_path.read_text())
    route = RouteProgress(json.loads(route_path.read_text()))
    if report['state'] != 'completed' or route.schema != 3:
        raise ValueError('Only completed preparation and schema3 routes can be published.')
    entries, rejections = [], []
    for mission in report['missions']:
        if not all(mission['fixtures']['contact'].values()):
            raise ValueError('Contact fixture failed.')
        for candidate in mission['candidates']:
            if not candidate['accepted']:
                rejections.append({'variation': mission['variation'], **candidate})
                continue
            source = Path(candidate['file'])
            if digest(source) != candidate['sha256']:
                raise ValueError('Preparation source changed.')
            point = np.asarray(candidate['position'])
            distances = []
            for region in range(len(route.lengths)):
                start, end = route.points[region:region + 2]
                axis = (end - start)[[0, 2]]
                t = np.clip((point - start)[[0, 2]] @ axis / (axis @ axis), 0, 1)
                distances.append(np.linalg.norm(point - (start + t * (end - start))))
            region = int(np.argmin(distances))
            telemetry = Telemetry(candidate['position'], candidate['scenario'], 'fixture', 1, 1, 1,
                                  candidate['scene_frame'], False, False,
                                  health=candidate['health'], grounded=True)
            route.reset(telemetry, segment=region)
            # The preparation schedule name was aspirational; this position is before the obstacle.
            name = candidate['name'].replace('beyond-obstacle', 'obstacle-jump-approach')
            filename = f"{mission['variation'].replace('_', '-')}-{name}.state"
            entries.append({**candidate, 'preparation_name': candidate['name'], 'name': name,
                            'file': filename, 'source_file': str(source), 'region': region,
                            'route_sha256': digest(route_path)})
    if not entries:
        raise ValueError('No eligible starts survived review.')
    destination.mkdir(parents=True, exist_ok=False)
    for entry in entries:
        shutil.copy2(entry['source_file'], destination / entry['file'])
    bank = {'schema': 1, 'game_id': route.specification['game_id'],
            'emulator_sha256': report['emulator_sha256'], 'preparation': str(report_path),
            'preparation_sha256': digest(report_path), 'route_sha256': digest(route_path),
            'scope': 'Bounded idle-safe assisted starts; not guaranteed safe indefinitely.',
            'entries': entries, 'rejected_candidates': rejections}
    atomic_json(destination / 'bank.json', bank)
    return destination / 'bank.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', required=True)
    parser.add_argument('--route', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = build(args.preparation, args.route, args.output)
    print(json.dumps({'bank': str(result), 'sha256': digest(result)}))
