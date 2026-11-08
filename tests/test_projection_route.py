import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from smg_rl.config import load_config
from smg_rl.route import RouteProgress

ROOT = Path(__file__).resolve().parents[1]


def tele(x, y=0., z=0., dead=False, complete=False):
    return SimpleNamespace(position=(x, y, z), grounded=True, dead=dead, complete=complete, health=3)


def line_route(**extra):
    return RouteProgress({'schema': 6, 'waypoints': [[0, 0, -100 * i] for i in range(11)], 'waypoint_radius': 100,
                          'corridor_radius': 150, 'height_tolerance': 500, 'exit_confirmations': 1,
                          'requires_grounded': True, 'max_sample_distance': 1000, 'lookahead': 400,
                          'lookbehind': 100, **extra})


def test_projection_pays_continuously_in_the_air_and_counts_passed_waypoints():
    route = line_route()
    route.reset(tele(0, 0, 0))
    assert route.committed_segment == 0 and route.eligible
    assert route.update(tele(10, 300, -250)) == 0.0  # 300 units above the line is outside the 150-unit corridor
    assert route.update(tele(10, 50, -250)) == pytest.approx(0.25) and route.committed_segment == 2
    assert route.update(tele(0, 0, -250)) == 0.0  # no credit for the same arc length twice
    assert route.update(tele(0, 0, -100)) == 0.0  # going back pays nothing and takes nothing away
    assert route.best == pytest.approx(0.25)


def test_projection_ignores_jumps_past_the_lookahead_off_corridor_and_teleports():
    route = line_route()
    route.reset(tele(0, 0, 0))
    assert route.update(tele(0, 0, -800)) == 0.0  # beyond the 400-unit lookahead window
    assert route.update(tele(500, 0, -200)) == 0.0  # outside the corridor
    assert route.update(tele(0, 0, -300)) == pytest.approx(0.3)
    far = line_route(max_sample_distance=200)
    far.reset(tele(0, 0, 0))
    assert far.update(tele(0, 0, -350)) == 0.0 and not far.eligible  # a teleport disables credit
    assert far.update(tele(0, 0, -400)) == 0.0


def test_projection_completes_at_the_last_waypoint_and_a_dead_or_off_route_start_is_not_eligible():
    route = line_route(lookahead=2000)
    route.reset(tele(0, 0, 0))
    route.update(tele(0, 0, -1000))
    assert route.committed_segment == 10 and route.exited and route.best == pytest.approx(1.0)
    assert not RouteProgress({**json.loads(json.dumps(route.specification))}).eligible
    off = line_route()
    off.reset(tele(5000, 0, 0))
    assert not off.eligible and off.update(tele(5000, 0, -100)) == 0.0
    dead = line_route()
    dead.reset(tele(0, 0, 0))
    assert dead.update(tele(0, 0, -100, dead=True)) == 0.0


def test_builder_resamples_by_arc_length_and_keeps_the_ends(tmp_path):
    spec = importlib.util.spec_from_file_location('make_projection_route', ROOT / 'scripts/make_projection_route.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{'x': 0, 'y': 0, 'z': -10 * i} for i in range(101)]
    route = module.projection_route(rows, 250, 'RMGE01', 'sweet_sweet')
    points = np.array(route['waypoints'])
    assert route['schema'] == 6 and points[0].tolist() == [0, 0, 0] and points[-1].tolist() == [0, 0, -1000]
    assert 4 <= len(points) <= 6 and np.all(np.abs(np.diff(points[:, 2])) >= 240)
    assert route['corridor_radius'] == 500 and route['lookahead'] == 1500
    RouteProgress(route)


def test_projection_arms_use_the_same_reward_scale_as_arm_f_and_build(tmp_path):
    from smg_rl.env import GalaxyEnv
    control = load_config(ROOT / 'config/algorithm-benchmark-f-iqn-20261106.yaml')
    for name in ('f-proj', 'btr-proj', 'altbtr-proj'):
        arm = load_config(ROOT / f'config/algorithm-benchmark-{name}-20261108.yaml')
        env = GalaxyEnv(arm, backend=None, run_dir=tmp_path / name, memory=object(), intrinsic=False)
        assert env.route.schema == 6 and len(env.route.lengths) == 278
        assert arm['rewards']['completion'] == 20 and arm['rewards']['death'] == control['rewards']['death']
        assert env.potential.scale == pytest.approx(16.68)
        assert env.potential.scale < arm['rewards']['completion']
    assert load_config(ROOT / 'config/algorithm-benchmark-altbtr-proj-20261108.yaml')['dqn']['gamma'] == 0.99


def test_projection_arm_pays_through_environment_steps_and_logs_a_consistent_potential(tmp_path):
    from smg_rl.env import GalaxyEnv
    from smg_rl.memory import Telemetry
    config = load_config(ROOT / 'config/algorithm-benchmark-f-proj-20261108.yaml')
    config['actions'] = {'profile': 'legacy54'}  # the fake backend has no native bridge
    points = json.loads((ROOT / config['paths']['route']).read_text())['waypoints']
    path = [points[0]]
    for start, end in zip(points, points[1:]):
        count = int(np.ceil(np.linalg.norm(np.subtract(end, start)) / 120))
        path += [list(np.add(start, np.subtract(end, start) * k / count)) for k in range(1, count + 1)]

    class Walker:
        last_frame_count = 12

        def __init__(self):
            self.steps = 0

        def load_slot(self, slot):
            self.steps = 0

        def advance(self, names):
            self.steps += 1

        def image(self):
            return np.zeros((84, 84), np.uint8)

        def sample(self):
            x, y, z = path[min(self.steps, len(path) - 1)]
            return Telemetry((x, y, z), 1, 'BeltConveyerExGalaxy', 1, 2, 3, self.steps, False, False,
                             health=3, max_health=3, grounded=False)

        def close(self):
            pass

    backend = Walker()
    (tmp_path / 'c.json').write_text(json.dumps({'validated': True, 'game_id': 'RMGE01', 'variations': {
        'sweet_sweet': {'start': points[0], 'goal': points[-1]}}}))
    config['paths']['calibration'] = str(tmp_path / 'c.json')
    env = GalaxyEnv(config, backend, tmp_path / 'run', memory=backend, intrinsic=False)
    env.reset(options={'variation': 0, 'full_start': True})
    rewards = [env.step(0)[1] for _ in range(len(path) - 1)]
    # Airborne throughout (grounded False): the projection route still pays, unlike the landing route.
    assert sum(rewards) == pytest.approx(16.68) and min(rewards) >= 0
    assert env.potential.value == pytest.approx(16.68) and env.route.committed_segment == 278


def test_no_failure_cost_projection_arm_differs_from_the_btr_projection_arm_only_in_reward_scale(tmp_path):
    from smg_rl.env import GalaxyEnv
    base = load_config(ROOT / 'config/algorithm-benchmark-btr-proj-20261108.yaml')
    free = load_config(ROOT / 'config/algorithm-benchmark-btr-proj-free-20261108.yaml')
    assert free['dqn'] == base['dqn'] and free['training'] == base['training'] and free['parallel'] == base['parallel']
    assert {k for k in base['rewards'] if base['rewards'][k] != free['rewards'][k]} == {'completion', 'damage', 'death'}
    assert free['rewards']['death'] == 0 and free['rewards']['damage'] == 0 and free['rewards']['timeout'] == 0
    changed = {k for k in base['regional_reward'].keys() | free['regional_reward'].keys()
               if base['regional_reward'].get(k) != free['regional_reward'].get(k)}
    assert changed == {'scale', 'waypoint_bonus', 'allow_unpenalized_failure'}
    env = GalaxyEnv(free, backend=None, run_dir=tmp_path, memory=object(), intrinsic=False)
    assert env.potential.scale == pytest.approx(27.8)
    assert env.potential.scale < free['rewards']['completion']
