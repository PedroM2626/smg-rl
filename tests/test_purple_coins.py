import hashlib
import json

import numpy as np
import pytest

from smg_rl.config import load_config
from smg_rl.env import GalaxyEnv
from smg_rl.memory import MemoryFault, Telemetry
from smg_rl.purple_coins import PurpleCoinReward

STAR = (100.0, 0.0, 0.0)
SETTINGS = {'coin_reward': 1.0, 'threshold': 100, 'approach_total': 10.0, 'max_gain_per_decision': 25}


def test_each_coin_pays_one_and_a_drop_is_never_paid():
    reward = PurpleCoinReward(SETTINGS, STAR)
    reward.reset(0, (0, 0, 0))
    assert reward.update(1, (0, 0, 0)) == (1, 1.0, 0.0)
    assert reward.update(3, (0, 0, 0)) == (2, 2.0, 0.0)
    assert reward.update(0, (0, 0, 0)) == (0, 0.0, 0.0)  # respawn
    assert reward.update(1, (0, 0, 0))[1] == 1.0


def test_the_approach_credit_starts_after_the_threshold_and_totals_at_most_the_setting():
    reward = PurpleCoinReward(SETTINGS, STAR)
    reward.reset(99, (0, 0, 0))
    assert reward.update(99, (50, 0, 0))[2] == 0.0  # not armed yet
    assert reward.update(100, (50, 0, 0))[2] == 0.0  # armed on this decision, pays from the next
    assert reward.span == pytest.approx(50.0)
    paid = [reward.update(100, (x, 0, 0))[2] for x in (60, 55, 80, 80, 40, 100)]  # distances 40, 45, 20, 20, 60, 0
    assert paid[0] == pytest.approx(2.0)
    assert paid[1] == 0.0 and paid[3] == 0.0 and paid[4] == 0.0  # retreat and regaining distance pay nothing
    assert paid[2] == pytest.approx(4.0) and paid[5] == pytest.approx(4.0)
    assert sum(paid) == pytest.approx(10.0)


def test_a_counter_jump_larger_than_the_limit_is_rejected():
    reward = PurpleCoinReward(SETTINGS, STAR)
    reward.reset(0, (0, 0, 0))
    with pytest.raises(ValueError):
        reward.update(60, (0, 0, 0))


def test_settings_must_be_finite_positive_numbers():
    for bad in ({**SETTINGS, 'coin_reward': 0}, {**SETTINGS, 'threshold': 0}, {**SETTINGS, 'approach_total': float('nan')},
                {**SETTINGS, 'threshold': True}):
        with pytest.raises(ValueError):
            PurpleCoinReward(bad, STAR)


class Script:
    """Fake emulator reading (position, coins, health, complete, dead) from a script of decisions."""

    def __init__(self, frames, error_on_load=None):
        self.frames, self.index, self.error_on_load = frames, 0, error_on_load
        self.loaded, self.paused = [], False

    def load_state(self, path):
        self.loaded.append(path)
        self.index = 0
        if self.error_on_load:
            raise RuntimeError(self.error_on_load)

    def load_slot(self, slot):
        raise AssertionError('The owner state must be used instead of a slot.')

    def advance(self, names):
        self.index = min(self.index + 1, len(self.frames) - 1)

    def image(self):
        return np.zeros((84, 84), np.uint8)

    def sample(self):
        position, coins, health, complete, dead = self.frames[self.index]
        return Telemetry(position, 2, 'MarioOnMarioGalaxy', 0x8122D4E0, 0x807D4E34, 0x807D5310, self.index, dead, complete,
                         health, 3, True, coins)

    def close(self):
        pass


def make_env(tmp_path, frames, **kwargs):
    state = tmp_path / 'owner.state'
    state.write_bytes(b'owner savestate')
    config = load_config()
    calibration = {'validated': True, 'variations': {n: {'start': [0, 0, 0], 'goal': list(STAR)}
                                                     for n in ('spinning', 'double_time')}}
    (tmp_path / 'c.json').write_text(json.dumps(calibration))
    config['paths']['calibration'] = str(tmp_path / 'c.json')
    config['rewards'] = {**config['rewards'], 'step': 0.0, 'progress': 0.0, 'intrinsic': 0.0, 'timeout': 0.0,
                         'death': 0.0, 'damage': -1.0, 'completion': 50.0}
    config['environment'].update(idle_limit=5, max_steps=100)
    config['game'] = {**config.get('game', {}), 'signals': {'purple_coins': {'address': 1, 'limit': 120}}}
    config['purple_coin_reward'] = {'enabled': True, **SETTINGS}
    config['variations'] = [{**config['variations'][1], 'start_state': str(state),
                             'start_state_sha256': hashlib.sha256(b'owner savestate').hexdigest()}]
    config['variations'][0]['name'] = 'double_time'
    backend = Script(frames, **kwargs)
    env = GalaxyEnv(config, backend, tmp_path / 'run', memory=backend, intrinsic=False)
    return env, backend


def frame(x, coins, health=3, complete=False, dead=False):
    return ((x, 0.0, 0.0), coins, health, complete, dead)


def test_coins_damage_approach_and_the_star_are_paid_and_death_costs_nothing(tmp_path):
    frames = [frame(0, 0), frame(0, 2), frame(0, 2, health=2), frame(0, 100, health=2), frame(0, 100, 2),
              frame(50, 100, 2), frame(100, 100, 2, complete=True)]
    env, backend = make_env(tmp_path, frames)
    env.purple.maximum = 100  # the script jumps 98 coins at once
    env.reset(options={'variation': 0})
    assert backend.loaded == [str(tmp_path / 'owner.state')]
    rewards = []
    for _ in range(6):
        _, reward, terminated, _, info = env.step(0)
        rewards.append(round(reward, 6))
    assert rewards[0] == pytest.approx(2.0)                      # two coins
    assert rewards[1] == pytest.approx(-1.0)                     # one point of damage
    assert rewards[2] == pytest.approx(98.0)                     # 98 coins, the threshold is reached
    assert rewards[3] == 0.0                                     # standing still
    assert rewards[4] == pytest.approx(5.0)                      # half of the way to the star
    assert rewards[5] == pytest.approx(5.0 + 50.0)               # the rest plus the star
    assert terminated and info['is_success'] and info['purple_coins_best'] == 100
    assert sum(rewards) == pytest.approx(2 - 1 + 98 + 10 + 50)


def test_dying_costs_nothing_beyond_the_lost_coins_opportunity(tmp_path):
    env, _ = make_env(tmp_path, [frame(0, 0), frame(0, 1), frame(0, 0, health=0, dead=True)])
    env.reset(options={'variation': 0})
    assert env.step(0)[1] == pytest.approx(1.0)
    _, reward, terminated, _, info = env.step(0)
    assert terminated and reward == 0.0 and info['dead'] is True


def test_idle_means_no_coin_and_no_approach(tmp_path):
    env, _ = make_env(tmp_path, [frame(0, 0), frame(0, 0)])
    env.reset(options={'variation': 0})
    outcomes = [env.step(0) for _ in range(5)]
    assert outcomes[-1][3] is True  # five idle decisions end the attempt


def test_the_owner_state_hash_is_pinned_and_a_bridge_error_two_is_tolerated(tmp_path):
    env, backend = make_env(tmp_path, [frame(0, 0), frame(0, 0)], error_on_load='native bridge error 2')
    env.reset(options={'variation': 0})
    assert backend.paused is True
    env, _ = make_env(tmp_path, [frame(0, 0)])
    env.variation_hash = None
    env.config['variations'][0]['start_state_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        env.reset(options={'variation': 0})
    env, _ = make_env(tmp_path, [frame(0, 0)], error_on_load='something else broke')
    with pytest.raises(RuntimeError):
        env.reset(options={'variation': 0})


def test_the_reward_refuses_other_shaping_a_missing_signal_and_a_missing_penalty(tmp_path):
    env, backend = make_env(tmp_path, [frame(0, 0)])
    for change in ({'rewards': {**env.config['rewards'], 'damage': 0.0}},
                   {'rewards': {**env.config['rewards'], 'death': -1.0}},
                   {'game': {'signals': {}}},
                   {'goal_marker': {'enabled': True}}):
        config = {**env.config, **change}
        with pytest.raises(ValueError):
            GalaxyEnv(config, backend, tmp_path / 'again', memory=backend, intrinsic=False)


def test_the_counter_jumping_is_a_memory_fault(tmp_path):
    env, _ = make_env(tmp_path, [frame(0, 0), frame(0, 60)])
    env.reset(options={'variation': 0})
    with pytest.raises(MemoryFault):
        env.step(0)


def test_the_luigi_configs_load_and_pin_the_owner_hash():
    config = load_config('config/luigi-mario-squared-altbtr-20261110.yaml')
    assert config['purple_coin_reward']['enabled'] and config['rewards']['damage'] == -1.0
    assert config['rewards']['completion'] == 50.0 and config['rewards']['death'] == 0.0
    variation = config['variations'][0]
    assert variation['start_state'] == 'data/luigi-coins/SB4E01.s02'
    assert variation['start_state_sha256'] == json.loads(
        open('.local/luigi-coins/original-hashes.json').read()).get('SB4E01.s02', variation['start_state_sha256'])
    assert config['dqn']['algorithm'] == 'iqn' and config['game']['signals']['purple_coins']['address'] == 0x80A75D08


def test_every_stage_has_an_altbtr_from_start_config_with_the_same_learner():
    reference = load_config('config/algorithm-benchmark-altbtr-pf-20261110.yaml')
    for path, game, gate in (('config/double-time-altbtr-start-20261110.yaml', 'SB4E01', False),
                             ('config/center-altbtr-start-20261110.yaml', 'RMGE01', True),
                             ('config/luigi-mario-squared-altbtr-20261110.yaml', 'SB4E01', False)):
        config = load_config(path)
        assert config['dqn'] == reference['dqn'] and config['game']['id'] == game
        assert bool(config.get('goal_marker', {}).get('enabled')) is gate
        assert config['curriculum']['enabled'] is False if 'curriculum' in config else True
        assert 'BBF' not in json.dumps(config)


def test_the_toy_time_config_is_an_evaluation_only_zero_shot_test_of_the_mario_squared_agent():
    from smg_rl.dqn_video import checkpoint_signature
    from smg_rl.replay import replay_signature

    source = load_config('config/luigi-mario-squared-altbtr-20261110.yaml')
    target = load_config('config/luigi-toy-time-zero-shot-20261110.yaml')
    assert target['training']['enabled'] is False and target['dqn'] == source['dqn']
    assert target['game']['id'] == 'RMGE01' and target['variations'][0]['scenario'] == 5
    assert target['rewards'] == source['rewards'] and target['purple_coin_reward'] == source['purple_coin_reward']
    assert checkpoint_signature(target) == replay_signature(source)
    assert checkpoint_signature(source) == replay_signature(source)
    target['actions'] = {'profile': 'legacy54'}
    with pytest.raises(ValueError):
        checkpoint_signature(target)
    target = load_config('config/luigi-toy-time-zero-shot-20261110.yaml')
    target['training']['enabled'] = True
    with pytest.raises(ValueError):
        checkpoint_signature(target)
