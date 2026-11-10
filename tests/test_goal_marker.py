import json

import numpy as np
import pytest

from smg_rl.config import load_config
from smg_rl.env import GalaxyEnv
from smg_rl.memory import Telemetry


class Walker:
    """Fake emulator whose position and grounded flag follow a script."""

    def __init__(self, path, grounded=True):
        self.path, self.grounded, self.steps = path, grounded, 0

    def load_slot(self, slot):
        self.steps = 0

    def advance(self, names):
        self.steps += 1

    def image(self):
        return np.zeros((84, 84), np.uint8)

    def sample(self):
        return Telemetry(tuple(self.path(self.steps)), 1, 'SecretAthleticGalaxy', 0x81458EC0, 0x807D4E34, 0x807D5310,
                         self.steps, False, False, None, None, self.grounded)

    def close(self):
        pass


def make_env(tmp_path, path, grounded=True, enabled=True):
    config = load_config()
    calibration = {'validated': True, 'variations': {n: {'start': [0, -1150, 500], 'goal': [0, -1150, -700],
                                                         'goal_radius_3d': 300.0} for n in ('spinning', 'double_time')}}
    (tmp_path / 'c.json').write_text(json.dumps(calibration))
    config['paths']['calibration'] = str(tmp_path / 'c.json')
    config['rewards'] = {**config['rewards'], 'step': 0.0, 'progress': 0.0, 'intrinsic': 0.0, 'timeout': 0.0,
                         'completion': 20.0}
    config['environment']['idle_limit'] = 50
    config['goal_marker'] = {'enabled': enabled}
    backend = Walker(path, grounded)
    return GalaxyEnv(config, backend, tmp_path / 'run', memory=backend, intrinsic=False), backend


def walk(t):
    return (0, -1150, 500 - 400 * t)


def test_reaching_the_marker_alive_and_grounded_ends_the_attempt_as_a_success_without_a_star(tmp_path):
    env, _ = make_env(tmp_path, walk)
    env.reset(options={'variation': 0})
    outcomes = [env.step(0) for _ in range(3)]
    assert [o[2] for o in outcomes] == [False, False, True]
    reward, info = outcomes[-1][1], outcomes[-1][4]
    assert reward == pytest.approx(20.0)
    assert info['goal_reached'] is True and info['is_success'] is True and info['complete'] is False


def test_the_marker_needs_the_ground_and_does_nothing_when_disabled(tmp_path):
    env, _ = make_env(tmp_path, walk, grounded=False)
    env.reset(options={'variation': 0})
    assert not any(env.step(0)[2] for _ in range(3))
    env, _ = make_env(tmp_path, walk, enabled=False)
    env.reset(options={'variation': 0})
    assert not any(env.step(0)[2] for _ in range(3))


def test_a_teleport_sized_jump_into_the_radius_is_not_a_success(tmp_path):
    env, _ = make_env(tmp_path, lambda t: (0, -1150, 500) if t == 0 else (0, -1150, -700 + 9000 * (t == 99)))
    config = env.config
    config['environment']['max_step_distance'] = 2500.0
    env.reset(options={'variation': 0})
    env.backend.steps = 98
    env.backend.path = lambda t: (0, -1150, 3000) if t < 99 else (0, -1150, -700)
    env.telemetry = env.memory.sample()
    assert env.step(0)[4]['goal_reached'] is False


def test_a_marker_without_a_calibrated_radius_is_refused(tmp_path):
    config = load_config()
    (tmp_path / 'c.json').write_text(json.dumps({'validated': True, 'variations': {
        n: {'start': [0, 0, 0], 'goal': [0, 0, 100]} for n in ('spinning', 'double_time')}}))
    config['paths']['calibration'] = str(tmp_path / 'c.json')
    config['goal_marker'] = {'enabled': True}
    with pytest.raises(ValueError, match='goal marker'):
        GalaxyEnv(config, Walker(walk), tmp_path / 'run', memory=Walker(walk), intrinsic=False)
