from pathlib import Path

import pytest

from smg_rl.config import load_config
from smg_rl.env import GalaxyEnv

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / '.local/center-anchors-20261110/bank.json'


def arm(name):
    return load_config(ROOT / f'config/center-bbf-{name}-20261110.yaml')


def test_center_arms_use_the_marker_gate_the_bbf_learner_and_the_free_reward():
    reference = load_config(ROOT / 'config/algorithm-benchmark-bbf-pf-20261110.yaml')
    for name in ('start', 'anchors'):
        config = arm(name)
        assert config['dqn'] == reference['dqn'] and config['rewards'] == reference['rewards']
        assert config['goal_marker'] == {'enabled': True, 'require_grounded': True} and 'demonstration' not in config
        assert config['game']['stage'] == 'KoopaBattleVs3Galaxy' and config['training']['enabled'] is True
        assert config['regional_reward']['scale'] < config['rewards']['completion']
    assert not arm('start')['curriculum']['enabled'] and 'start_mix' not in arm('start')
    assert arm('anchors')['curriculum']['sampling'] == 'uniform' and arm('anchors')['start_mix']['full_start_fraction'] == 0.25


@pytest.mark.skipif(not BANK.exists(), reason='The local anchor bank is not present.')
@pytest.mark.parametrize('name', ('start', 'anchors'))
def test_center_arms_build_their_environment(name, tmp_path):
    env = GalaxyEnv(arm(name), backend=None, run_dir=tmp_path, memory=object(), intrinsic=False)
    assert env.route.schema == 6 and env.potential.scale == pytest.approx(27.3)
