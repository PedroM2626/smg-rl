import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from smg_rl.config import load_config
from smg_rl.env import GalaxyEnv
from smg_rl.memory import Telemetry
from smg_rl.replay import replay_signature

ROOT = Path(__file__).resolve().parents[1]


class Walker:
    """Fake emulator whose position follows a script; a missing entry means standing still."""

    def __init__(self, path):
        self.path, self.steps, self.slot = path, 0, 1

    def load_slot(self, slot):
        self.slot, self.steps = slot, 0

    def advance(self, names):
        self.steps += 1

    def image(self):
        return np.zeros((84, 84), np.uint8)

    def sample(self):
        x, y, z = self.path(self.steps)
        return Telemetry((x, y, z), 1, 'SecretAthleticGalaxy', 0x81458EC0, 0x807D4E34, 0x807D5310,
                         self.steps, False, False)

    def close(self):
        pass


def make_env(tmp_path, path, **novelty):
    config = load_config()
    calibration = {'validated': True, 'variations': {n: {'start': [0, -1150, 500], 'goal': [0, -450, -10000]}
                                                      for n in ('spinning', 'double_time')}}
    (tmp_path / 'c.json').write_text(json.dumps(calibration))
    config['paths']['calibration'] = str(tmp_path / 'c.json')
    config['position_novelty'] = {'enabled': True, 'cell_xz': 100.0, 'cell_y': 250.0, 'bonus': 0.1, 'power': 0.5, **novelty}
    config['rewards'] = {**config['rewards'], 'step': 0.0, 'progress': 0.0, 'intrinsic': 0.0, 'timeout': 0.0}
    config['environment']['idle_limit'] = 5
    backend = Walker(path)
    return GalaxyEnv(config, backend, tmp_path / 'run', memory=backend, intrinsic=False), backend


def test_new_cells_pay_a_decaying_bonus_that_persists_across_episodes(tmp_path):
    env, _ = make_env(tmp_path, lambda t: (0, -1150, 500 - 100 * t))
    env.reset(options={'variation': 0})
    first = [env.step(0)[1] for _ in range(3)]
    assert first == pytest.approx([0.1, 0.1, 0.1])
    env.reset(options={'variation': 0})
    second = [env.step(0)[1] for _ in range(3)]
    assert second == pytest.approx([0.1 / np.sqrt(2)] * 3)
    assert env.rows[-1]['position_novelty_reward'] == pytest.approx(0.1 / np.sqrt(2))


def test_standing_still_pays_nothing_and_ends_the_episode_by_the_idle_limit(tmp_path):
    env, _ = make_env(tmp_path, lambda t: (0, -1150, 500))
    env.reset(options={'variation': 0})
    outcomes = [env.step(0) for _ in range(5)]
    assert [r for _, r, _, _, _ in outcomes] == [0.0] * 5
    assert outcomes[-1][3] is True and outcomes[-1][2] is False  # truncated by the idle cut, not terminated
    assert all(not o[3] for o in outcomes[:-1])


def test_a_fall_below_the_limit_is_not_rewarded(tmp_path):
    env, _ = make_env(tmp_path, lambda t: (0, -1150 if t < 3 else -9000, 500 - 100 * t))
    env.reset(options={'variation': 0})
    rewards = [env.step(0)[1] for _ in range(3)]
    assert rewards[:2] == pytest.approx([0.1, 0.1]) and rewards[2] == 0.0


def test_novelty_is_off_by_default_and_changes_the_replay_signature_when_on(tmp_path):
    config = load_config(ROOT / 'config/algorithm-benchmark-f-iqn-20261106.yaml')
    assert 'position_novelty' not in replay_signature(config)
    arm = load_config(ROOT / 'config/algorithm-benchmark-exploration-20261107.yaml')
    assert replay_signature(arm)['position_novelty']['bonus'] == 0.05


@pytest.mark.parametrize('change', [{'bonus': 0.0}, {'bonus': 2.0}, {'cell_xz': -1.0}, {'power': 0.2}, {'power': 1.5},
                                    {'bonus': True}])
def test_invalid_novelty_settings_are_rejected(tmp_path, change):
    with pytest.raises(ValueError, match='Position novelty'):
        make_env(tmp_path, lambda t: (0, -1150, 500), **change)


def test_exploration_arm_excludes_curricula_and_demonstrations_and_builds(tmp_path):
    config = load_config(ROOT / 'config/algorithm-benchmark-exploration-20261107.yaml')
    assert not config['regional_reward']['enabled'] and not config['curriculum']['enabled']
    assert 'demonstration' not in config and config['position_novelty']['bonus'] == 0.05
    GalaxyEnv(config, backend=None, run_dir=tmp_path / 'ok', memory=object(), intrinsic=False)
    for key, value in (('curriculum', {'enabled': True}), ('demonstration', {'sequences': {}})):
        broken = deepcopy(config)
        broken[key] = value
        with pytest.raises(ValueError):
            GalaxyEnv(broken, backend=None, run_dir=tmp_path / key, memory=object(), intrinsic=False)


def test_exploration_arm_differs_from_arm_f_in_reward_only():
    f = load_config(ROOT / 'config/algorithm-benchmark-f-iqn-20261106.yaml')
    arm = load_config(ROOT / 'config/algorithm-benchmark-exploration-20261107.yaml')
    assert arm['dqn'] == f['dqn'] and arm['training'] == f['training'] and arm['parallel'] == f['parallel']
    assert {k for k in f['rewards'] if f['rewards'][k] != arm['rewards'][k]} == {'revision'}


def test_center_novelty_arm_adds_only_the_novelty_bonus_to_the_plain_center_arm(tmp_path):
    plain = load_config(ROOT / 'config/center-altbtr-start-20261110.yaml')
    arm = load_config(ROOT / 'config/center-altbtr-novelty-20261113.yaml')
    assert arm['position_novelty']['enabled'] and arm['regional_reward'] == plain['regional_reward']
    assert arm['dqn'] == plain['dqn'] and arm['rewards'] == plain['rewards'] and arm['goal_marker'] == plain['goal_marker']
    assert arm['paths']['runs'] != plain['paths']['runs'] and arm['paths']['replay'] != plain['paths']['replay']
    assert replay_signature(arm)['position_novelty']['bonus'] == 0.02
    GalaxyEnv(arm, backend=None, run_dir=tmp_path / 'ok', memory=object(), intrinsic=False)
    broken = deepcopy(arm)
    broken['demonstration'] = {'sequences': {}}
    with pytest.raises(ValueError, match='Position novelty excludes'):
        GalaxyEnv(broken, backend=None, run_dir=tmp_path / 'bad', memory=object(), intrinsic=False)


def test_center_novelty_arm_fall_limit_is_below_the_center_start_height():
    arm = load_config(ROOT / 'config/center-altbtr-novelty-20261113.yaml')
    calibration = json.loads((ROOT / 'config/smg1-center-calibration.json').read_text())
    start_height = list(calibration['variations'].values())[0]['start'][1]
    assert arm['environment']['min_reward_y'] < start_height  # the Sweet Sweet limit of -1800 paid nothing here
