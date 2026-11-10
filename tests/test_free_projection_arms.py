from pathlib import Path

import pytest

from smg_rl.config import load_config
from smg_rl.env import GalaxyEnv

ROOT = Path(__file__).resolve().parents[1] / 'config'
NAMES = ('f-pf', 'altbtr-pf', 'officialbtr-pf', 'bbf-pf', 'random-pf', 'backward-pf')


def arm(name):
    return load_config(ROOT / f'algorithm-benchmark-{name}-20261110.yaml')


@pytest.mark.parametrize('name', NAMES)
def test_every_arm_has_the_btr_free_reward_and_builds_its_environment(name, tmp_path):
    reference = load_config(ROOT / 'algorithm-benchmark-btr-proj-free-20261108.yaml')
    config = arm(name)
    assert config['rewards'] == reference['rewards'] and config['paths']['route'] == reference['paths']['route']
    expected = {k: v for k, v in reference['regional_reward'].items() if k != 'gamma_per_frame'}
    assert {k: v for k, v in config['regional_reward'].items() if k != 'gamma_per_frame'} == expected
    assert 'demonstration' not in config and config['training']['seed'] == reference['training']['seed']
    assert config['parallel'] == reference['parallel']
    env = GalaxyEnv(config, backend=None, run_dir=tmp_path, memory=object(), intrinsic=False)
    assert env.route.schema == 6 and env.potential.scale == pytest.approx(27.8)


def test_learner_arms_keep_the_defining_learner_settings():
    assert arm('f-pf')['dqn']['quantiles'] == 32 and arm('f-pf')['dqn']['epsilon'] == 0.1
    alt = arm('altbtr-pf')
    assert alt['dqn']['gamma'] == 0.99 and alt['regional_reward']['gamma_per_frame'] == pytest.approx(0.99 ** (1 / 12))
    bbf = arm('bbf-pf')['dqn']
    assert bbf['channels'] == 64 and bbf['noisy'] is False and bbf['reset'] == {'interval': 40000, 'shrink': 0.5}
    assert arm('officialbtr-pf')['official_btr']['min_sampling_size'] == 20000


def test_start_distribution_arms_differ_only_in_how_episodes_start():
    uniform, backward = arm('random-pf'), arm('backward-pf')
    base = load_config(ROOT / 'algorithm-benchmark-btr-proj-free-20261108.yaml')
    for config in (uniform, backward):
        assert config['dqn'] == base['dqn'] and config['curriculum']['enabled'] and 'frontier' not in config['curriculum']
    assert uniform['curriculum']['sampling'] == 'uniform' and uniform['start_mix']['full_start_fraction'] == 0.25
    assert 'frontier_rule' not in uniform
    assert backward['start_mix']['full_start_fraction'] == 0.0 and backward['frontier_rule']['mode'] == 'greedy_hybrid'
