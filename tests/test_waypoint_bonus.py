import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from smg_rl.config import load_config
from smg_rl.env import GalaxyEnv
from smg_rl.rewards import RegionalWaypointBonus
from smg_rl.route import RouteProgress

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'config/routes/smg1-human-landing-v2.json'
CONFIG = ROOT / 'config/algorithm-benchmark-btr-waypoint-20261107.yaml'


def sparse():
    import importlib.util
    spec = importlib.util.spec_from_file_location('make_sparse_route', ROOT / 'scripts/make_sparse_route.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sparse_route


def test_waypoint_bonus_pays_each_new_waypoint_once_and_never_takes_credit_back():
    bonus = RegionalWaypointBonus(1.0, 15)
    bonus.reset(0)
    assert [bonus.update(c, 12) for c in (0, 1, 1, 3, 2, 3, 15)] == [0, 1, 0, 2, 0, 0, 12]
    assert bonus.update(15, 12, terminal=True) == 0 and bonus.scale == 15
    for bad in ((0, 15), (1, 0), (np.nan, 15), (6, 15), (1, 2001)):
        with pytest.raises(ValueError):
            RegionalWaypointBonus(*bad)
    with pytest.raises(ValueError):
        bonus.update(16, 12)
    with pytest.raises(ValueError):
        bonus.update(1.5, 12)


def test_sparse_route_keeps_real_landing_points_evenly_spaced_and_is_a_valid_route():
    source = json.loads(SOURCE.read_text())
    route = sparse()(source, 15)
    kept = route['kept_source_indices']
    assert len(kept) == 16 and kept[0] == 0 and kept[-1] == len(source['waypoints']) - 1
    assert kept == sorted(set(kept))
    assert route['waypoints'] == [source['waypoints'][i] for i in kept]
    full, thin = RouteProgress(source), RouteProgress(route)
    assert len(thin.lengths) == 15 and 0.9 * full.total_length < thin.total_length <= full.total_length
    assert thin.lengths.max() < 2.5 * thin.lengths.mean()
    with pytest.raises(ValueError):
        sparse()(source, len(source['waypoints']))


def test_waypoint_arm_builds_an_environment_whose_total_bonus_stays_below_the_star(tmp_path):
    config = load_config(CONFIG)
    env = GalaxyEnv(config, backend=None, run_dir=tmp_path / 'a', memory=object(), intrinsic=False)
    assert isinstance(env.potential, RegionalWaypointBonus) and len(env.route.lengths) == 15
    assert env.potential.scale == 15 and config['rewards']['completion'] == 20
    assert config['rewards']['death'] == -3 and config['rewards']['damage'] == -1
    greedy = deepcopy(config)
    greedy['regional_reward']['waypoint_bonus'] = 2.0
    with pytest.raises(ValueError, match='Discounted distance objective'):
        GalaxyEnv(greedy, backend=None, run_dir=tmp_path / 'b', memory=object(), intrinsic=False)


def test_waypoint_arm_differs_from_the_btr_arm_only_in_reward_and_paths():
    btr = load_config(ROOT / 'config/algorithm-benchmark-btr-20261106.yaml')
    arm = load_config(CONFIG)
    assert arm['dqn'] == btr['dqn'] and arm['training'] == btr['training'] and arm['parallel'] == btr['parallel']
    differing = {k for k in btr['rewards'] if btr['rewards'][k] != arm['rewards'][k]}
    assert differing == {'death', 'damage'} and arm['rewards']['death'] == -3 and arm['rewards']['damage'] == -1
    assert arm['rewards']['completion'] == btr['rewards']['completion'] == 20
    changed = {k for k in btr['regional_reward'] if btr['regional_reward'][k] != arm['regional_reward'].get(k)}
    assert changed == {'spec_sha256', 'scale', 'mode'} and arm['regional_reward']['waypoint_bonus'] == 1.0
    assert arm['paths']['runs'] != btr['paths']['runs'] and arm['paths']['route'] != btr['paths']['route']


def test_unpenalized_failure_needs_an_explicit_opt_in_and_still_rejects_positive_failure_pay():
    import pytest

    from smg_rl.rewards import validate_discounted_objective
    weights = {'step': 0., 'death': 0., 'timeout': 0., 'completion': 20., 'damage': 0.}
    settings = {'scale': 17., 'timeout_mode': 'truncation_bootstrap'}
    with pytest.raises(ValueError, match='Discounted distance objective'):
        validate_discounted_objective(settings, weights, .997)
    validate_discounted_objective({**settings, 'allow_unpenalized_failure': True}, weights, .997)
    for bad in ({'death': .5}, {'damage': .5}, {'step': -.1}, {'timeout': -1.}):
        with pytest.raises(ValueError, match='Discounted distance objective'):
            validate_discounted_objective({**settings, 'allow_unpenalized_failure': True}, {**weights, **bad}, .997)
    validate_discounted_objective(settings, {**weights, 'death': -2., 'damage': -.25}, .997)


def test_no_penalty_probe_differs_from_arm_f_only_in_the_two_failure_weights(tmp_path):
    f = load_config(ROOT / 'config/algorithm-benchmark-f-iqn-20261106.yaml')
    probe = load_config(ROOT / 'config/algorithm-benchmark-f-nopenalty-20261107.yaml')
    assert probe['dqn'] == f['dqn'] and probe['training'] == f['training'] and probe['parallel'] == f['parallel']
    assert {k for k in f['rewards'] if f['rewards'][k] != probe['rewards'][k]} == {'death', 'damage'}
    assert probe['rewards']['death'] == 0 and probe['rewards']['damage'] == 0
    assert probe['regional_reward']['allow_unpenalized_failure'] is True
    GalaxyEnv(probe, backend=None, run_dir=tmp_path, memory=object(), intrinsic=False)


def test_dense_head_keeps_the_sources_first_waypoints_then_spreads_the_rest():
    source = json.loads(SOURCE.read_text())
    route = sparse()(source, 15, 8)
    kept = route['kept_source_indices']
    assert kept[:9] == list(range(9)) and kept[-1] == len(source['waypoints']) - 1 and len(kept) == 16
    assert kept == sorted(set(kept))
    for bad in (15, -1, 2.5):
        with pytest.raises(ValueError):
            sparse()(source, 15, bad)


def test_waypoint_arm_pays_through_real_environment_steps_and_audits_its_credit(tmp_path):
    from types import SimpleNamespace

    from smg_rl.memory import Telemetry
    config = load_config(CONFIG)
    config['actions'] = {'profile': 'legacy54'}  # the fake backend has no native bridge
    route = json.loads((ROOT / config['paths']['route']).read_text())
    points = route['waypoints']
    # A scripted walk that lands on every waypoint in order, in steps under the route's sample limit.
    path = [points[0]]
    for start, end in zip(points, points[1:]):
        count = int(np.ceil(np.linalg.norm(np.subtract(end, start)) / 800))
        path += [list(np.add(start, (np.subtract(end, start)) * k / count)) for k in range(1, count + 1)]

    class Walker:
        last_frame_count = 12

        def __init__(self):
            self.steps, self.slot = 0, 1

        def load_slot(self, slot):
            self.steps = 0

        def advance(self, names):
            self.steps += 1

        def image(self):
            return np.zeros((84, 84), np.uint8)

        def sample(self):
            x, y, z = path[min(self.steps, len(path) - 1)]
            return Telemetry((x, y, z), 1, 'BeltConveyerExGalaxy', 1, 2, 3, self.steps, False, False,
                             health=3, max_health=3, grounded=True)

        def close(self):
            pass

    backend = Walker()
    calibration = {'validated': True, 'game_id': 'RMGE01', 'variations': {'sweet_sweet': {'start': points[0], 'goal': points[-1]}}}
    (tmp_path / 'c.json').write_text(json.dumps(calibration))
    config['paths']['calibration'] = str(tmp_path / 'c.json')
    env = GalaxyEnv(config, backend, tmp_path / 'run', memory=backend, intrinsic=False)
    env.reset(options={'variation': 0, 'full_start': True})
    rewards = [env.step(0)[1] for _ in range(len(path) - 1)]
    assert set(np.round(rewards, 6)) <= {0.0, 1.0, 2.0}
    assert sum(rewards) == pytest.approx(15) and env.potential.value == pytest.approx(15)
    assert SimpleNamespace(**env.rows[-1]).shaping_potential == pytest.approx(15)
