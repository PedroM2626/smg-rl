from pathlib import Path

import pytest

from smg_rl.config import load_config
from smg_rl.env import GalaxyEnv

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / '.local/double-time-anchors-20261110/bank.json'


def arm(name):
    return load_config(ROOT / f'config/double-time-bbf-{name}-20261110.yaml')


def test_both_double_time_arms_share_the_bbf_learner_and_the_free_reward():
    start, anchors = arm('start'), arm('anchors')
    reference = load_config(ROOT / 'config/algorithm-benchmark-bbf-pf-20261110.yaml')
    for config in (start, anchors):
        assert config['dqn'] == reference['dqn'] and config['game']['id'] == 'SB4E01'
        assert config['rewards'] == reference['rewards'] and 'demonstration' not in config
        assert config['variations'] == [{'goal_slot': 4, 'name': 'double_time', 'scenario': 2, 'start_slot': 3,
                                         'title': 'Turning Turning Double Time'}]
        assert Path(config['paths']['route']).name == 'smg2-double-time-projection-v1.json'
        regional = config['regional_reward']
        assert regional['mode'] == 'discounted_waypoint_bonus' and regional['scale'] == pytest.approx(26.5)
        assert regional['scale'] < config['rewards']['completion']
    assert not start['curriculum']['enabled'] and 'start_mix' not in start
    assert anchors['curriculum']['sampling'] == 'uniform' and anchors['start_mix']['full_start_fraction'] == 0.25


@pytest.mark.skipif(not BANK.exists(), reason='The local anchor bank is not present.')
@pytest.mark.parametrize('name', ('start', 'anchors'))
def test_double_time_arms_build_their_environment(name, tmp_path):
    env = GalaxyEnv(arm(name), backend=None, run_dir=tmp_path, memory=object(), intrinsic=False)
    assert env.route.schema == 6 and env.potential.scale == pytest.approx(26.5)
