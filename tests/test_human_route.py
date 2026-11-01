import csv
from dataclasses import replace

import pytest

from smg_rl.config import load_config
from smg_rl.env import GalaxyEnv
from smg_rl.human_route_audit import first_segment_progress, observed_landmark_candidates
from smg_rl.memory import Telemetry
from smg_rl.replay import replay_signature
from smg_rl.route import RouteProgress


def test_human_route_requires_order_motion_and_landing():
    spec = {'schema': 4, 'waypoints': [[0, 0, 0], [100, 0, 0], [200, 0, 0]],
            'waypoint_radius': 10, 'corridor_radius': 10, 'height_tolerance': 100,
            'exit_confirmations': 1, 'max_sample_distance': 150, 'requires_grounded': True}
    route = RouteProgress(spec)
    start = Telemetry((0, 0, 0), 1, 'BeltConveyerExGalaxy', 1, 2, 3, 0,
                      False, False, health=3, grounded=True)
    route.reset(start)
    assert route.update(start) == 0
    assert route.update(replace(start, position=(110, 20, 0), grounded=False)) == 0
    assert route.best == 0
    assert route.update(replace(start, position=(110, 0, 0))) == pytest.approx(.5)
    assert route.update(replace(start, position=(110, 0, 0))) == 0
    assert route.update(replace(start, position=(0, 0, 0))) == 0
    assert route.update(replace(start, position=(110, 0, 0))) == 0
    assert route.update(replace(start, position=(200, 0, 0))) == pytest.approx(.5)
    assert route.exited
    route.reset(start)
    assert route.update(replace(start, position=(200, 0, 0))) == 0
    assert not route.eligible  # Teleports cannot recover eligibility later.
    assert route.update(replace(start, position=(100, 0, 0))) == 0
    route.reset(replace(start, position=(200, 0, 0)))
    assert not route.eligible  # A mid-level start is not the canonical full start.


def test_schema5_lookahead_skips_a_missed_waypoint_only_when_a_later_one_is_swept():
    spec = {'schema': 5, 'waypoints': [[0, 0, 0], [100, 0, 0], [200, 0, 0], [300, 0, 0], [400, 0, 0]],
            'waypoint_radius': 10, 'corridor_radius': 10, 'height_tolerance': 100,
            'exit_confirmations': 1, 'max_sample_distance': 150, 'requires_grounded': True,
            'landing_radius': 45}
    start = Telemetry((0, 0, 0), 1, 'BeltConveyerExGalaxy', 1, 2, 3, 0,
                      False, False, health=3, grounded=True)
    strict, skipping = RouteProgress(spec), RouteProgress({**spec, 'waypoint_lookahead': 1})
    # The path passes 50 units beside waypoint 1, then sweeps waypoint 2.
    path = [(60, 0, 50), (120, 0, 50), (200, 0, 0)]
    for route in (strict, skipping):
        route.reset(start)
        route.update(start)
        paid = sum(route.update(replace(start, position=p)) for p in path)
        assert route.eligible
        if route is strict:
            assert paid == 0 and route.committed_segment == 0
        else:
            assert paid == pytest.approx(.5) and route.committed_segment == 2
    # Lookahead never jumps past more than its bound.
    skipping.reset(start)
    skipping.update(start)
    assert skipping.update(replace(start, position=(100, 0, 60))) == 0
    assert skipping.update(replace(start, position=(240, 0, 60))) == 0
    assert skipping.update(replace(start, position=(300, 0, 0))) == 0
    with pytest.raises(ValueError):
        RouteProgress({**spec, 'waypoint_lookahead': 7})
    with pytest.raises(ValueError):
        RouteProgress({**spec, 'schema': 4, 'waypoint_lookahead': 1})


def test_schema5_does_not_pay_airborne_visit_after_return_to_start():
    spec = {'schema': 5, 'waypoints': [[0, 0, 0], [100, 0, 0], [200, 0, 0]],
            'waypoint_radius': 10, 'corridor_radius': 10, 'height_tolerance': 100,
            'exit_confirmations': 1, 'max_sample_distance': 200, 'landing_radius': 450,
            'requires_grounded': True}
    route = RouteProgress(spec)
    start = Telemetry((0, 0, 0), 1, 'BeltConveyerExGalaxy', 1, 2, 3, 0,
                      False, False, health=3, grounded=True)
    route.reset(start)
    assert route.update(replace(start, position=(110, 20, 0), grounded=False)) == 0
    assert route.update(start) == 0
    assert route.best == 0 and route.pending_segment == 0
    assert route.update(replace(start, position=(110, 20, 0), grounded=False)) == 0
    assert route.update(replace(start, position=(110, 0, 0))) == pytest.approx(.5)
    assert route.update(replace(start, position=(0, 0, 0))) == 0
    assert route.update(replace(start, position=(110, 0, 0))) == 0
    assert route.best == pytest.approx(.5)
    route.reset(start)
    assert route.update(replace(start, position=(110, 20, 0), grounded=False)) == 0
    assert route.update(replace(start, position=(110, 0, 0), dead=True)) == 0
    assert route.best == 0
    route.reset(start)
    assert route.update(replace(start, position=(500, 0, 0))) == 0
    assert not route.eligible
    route.reset(start)
    assert route.update(replace(start, position=(110, 0, 0), health=2)) == pytest.approx(.5)
    # Damage and route progression are separate signals; the environment penalizes HP loss.


def test_candidate_profile_has_distinct_signed_reward_and_safe_objective(tmp_path):
    original = load_config('config/smg1-six-two-hour.yaml')
    candidate = load_config('config/human-route-smg1-candidate.yaml')
    assert replay_signature(candidate) != replay_signature(original)
    assert candidate['actions'] == original['actions']
    assert candidate['parallel']['frame_skip'] == 12
    assert not candidate['curriculum']['enabled']
    assert candidate['training']['enabled']
    env = GalaxyEnv(candidate, backend=None, run_dir=tmp_path, memory=object(), intrinsic=False)
    assert env.route.schema == 4


@pytest.mark.parametrize(('variation', 'scenario'), [('spinning', 1), ('double-time', 2)])
def test_smg2_candidates_are_separately_signed_and_mission_specific(tmp_path, variation, scenario):
    config = load_config(f'config/human-route-smg2-{variation}-candidate.yaml')
    env = GalaxyEnv(config, backend=None, run_dir=tmp_path, memory=object(), intrinsic=False)
    assert env.route.schema == 4 and config['game']['id'] == 'SB4E01'
    assert len(config['variations']) == 1 and config['variations'][0]['scenario'] == scenario
    assert len(env.actions) == 8 and not config['curriculum']['enabled']


def test_initial_progress_counts_only_nearby_live_grounded_observations(tmp_path):
    path = tmp_path / 'episode.csv'
    columns = ['action', 'start_kind', 'x', 'y', 'z', 'grounded', 'dead', 'complete']
    rows = [
        ['reset', 'full_start', 0, 0, 0, True, False, False],
        ['w', 'full_start', 50, 0, 0, True, False, False],
        ['w', 'full_start', 100, 0, 1000, True, False, False],  # outside corridor
        ['jump', 'full_start', 100, 0, 0, False, False, False],  # airborne
        ['w', 'full_start', 100, 0, 0, True, True, False],  # death
    ]
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)
    spec = {'waypoints': [[0, 0, 0], [100, 0, 0]], 'waypoint_radius': 10}
    with pytest.raises(ValueError, match='Withdrawn metric'):
        first_segment_progress(path, spec)
    assert first_segment_progress(path, spec, allow_unsafe_projection=True) == pytest.approx(.5)


def test_observed_landmarks_reject_nearby_parallel_platform_and_skipped_visits():
    spec = {'schema': 'observed-landmarks-v1', 'start': [0, 0, 0], 'start_radius_3d': 1,
            'max_step_distance_3d': 500,
            'landmarks': [{'name': 'first', 'human_sample': 10, 'position': [100, 0, 0], 'radius_3d': 20},
                          {'name': 'second', 'human_sample': 20, 'position': [200, 0, 0], 'radius_3d': 20}]}

    def row(x, y=0, z=0, grounded=True, dead=False, complete=False):
        return {'x': x, 'y': y, 'z': z, 'grounded': str(grounded),
                'dead': str(dead), 'complete': str(complete)}

    assert observed_landmark_candidates([row(0), row(50, z=50)], spec) == []
    assert observed_landmark_candidates([row(0), row(200)], spec) == []  # no skipped first visit
    assert observed_landmark_candidates([row(0), row(100, grounded=False)], spec) == []
    assert observed_landmark_candidates([row(0), row(100, dead=True)], spec) == []
    matches = observed_landmark_candidates([row(0), row(100), row(200)], spec)
    assert [v['name'] for v in matches] == ['first', 'second']
    with pytest.raises(ValueError, match='discontinuous'):
        observed_landmark_candidates([row(0), row(600)], spec)


def test_real_starting_platform_example_is_not_a_landmark_visit():
    start = [-50.0, 45.0000114440918, -50.0]
    # This actual final-evaluation position previously produced 51.23% by projection.
    end = [453.1246337890625, 45.0, -532.9596557617188]
    spec = {'schema': 'observed-landmarks-v1', 'start': start, 'start_radius_3d': 1,
            'max_step_distance_3d': 500,
            'landmarks': [{'name': 'First pink belt', 'human_sample': 120,
                           'position': [-19.230239868164062, -50.00019836425781, -1022.8365478515625],
                           'radius_3d': 100}]}
    rows = []
    for i in range(16):
        point = [start[j] + i / 15 * (end[j] - start[j]) for j in range(3)]
        rows.append({'x': point[0], 'y': point[1], 'z': point[2], 'grounded': 'True',
                     'dead': 'False', 'complete': 'False'})
    assert observed_landmark_candidates(rows, spec) == []
