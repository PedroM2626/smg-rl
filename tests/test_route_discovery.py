import csv
from copy import deepcopy

from smg_rl.route_discovery import discover, extended_spec, proposals_for_episode


def write_episode(path, points, start='full_start', grounded=True, damage=False):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['step', 'start_kind', 'route_prefix_exited', 'x', 'y', 'z',
                         'health', 'grounded', 'dead', 'complete', 'health_lost', 'action', 'outcome'])
        writer.writerow([0, start, False, 0, 0, 0, 3, True, False, False, 0, 'reset', 'running'])
        for i, x in enumerate(points, 1):
            writer.writerow([i, start, True, x, 0, 0, 3, grounded, False, False,
                             int(damage), 'w', 'timeout' if i == len(points) else 'running'])


def spec():
    return {'schema': 3, 'name': 'Test opening', 'waypoints': [[0, 0, 0], [100, 0, 0]],
            'waypoint_radius': 100, 'corridor_radius': 100, 'height_tolerance': 60,
            'exit_confirmations': 2, 'requires_grounded': True}


def test_extension_needs_two_separate_full_start_traces_and_preserves_old_map(tmp_path):
    a, b = tmp_path / 'a.csv', tmp_path / 'b.csv'
    write_episode(a, [100, 180, 260, 300])
    write_episode(b, [100, 170, 250, 300])
    original = spec()
    assert discover([a], original) is None
    proposal = discover([a, b], original)
    assert proposal and proposal['independent_game_states'] is False
    frozen = deepcopy(original)
    updated = extended_spec(original, proposal)
    assert original == frozen
    assert updated['waypoints'][:-1] == original['waypoints']
    assert updated['waypoints'][-1] == proposal['candidate']['position']


def test_rejects_assisted_airborne_damaged_or_discontinuous_samples(tmp_path):
    for name, kwargs, points in [
        ('assisted', {'start': 'frontier'}, [100, 180, 260]),
        ('airborne', {'grounded': False}, [100, 180, 260]),
        ('damaged', {'damage': True}, [100, 180, 260]),
        ('teleport', {}, [100, 300, 380]),
        ('backtrack', {}, [100, 60, 0]),
    ]:
        path = tmp_path / f'{name}.csv'
        write_episode(path, points, **kwargs)
        assert proposals_for_episode(path, spec()) == []


def test_proposals_must_also_cross_the_ordered_old_route(tmp_path):
    a, b = tmp_path / 'a.csv', tmp_path / 'b.csv'
    write_episode(a, [100, 180, 260, 300])
    write_episode(b, [100, 170, 250, 300])
    inaccessible = spec()
    inaccessible['waypoints'] = [[-1000, 0, 0], [100, 0, 0]]
    # A claimed exit flag alone cannot validate a route whose start was never reached.
    assert discover([a, b], inaccessible) is None
