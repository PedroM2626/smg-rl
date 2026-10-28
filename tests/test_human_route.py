from dataclasses import replace

import pytest

from smg_rl.config import load_config
from smg_rl.env import GalaxyEnv
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
