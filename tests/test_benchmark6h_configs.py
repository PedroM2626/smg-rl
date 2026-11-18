"""The 6 h Sweet Sweet benchmark arms differ from the earlier 1 h arms only where intended."""
from copy import deepcopy
from pathlib import Path

import pytest

from smg_rl.config import load_config
from smg_rl.env import GalaxyEnv
from smg_rl.replay import replay_signature
from smg_rl.world_model import ImaginedAgent

ROOT = Path(__file__).resolve().parents[1]
STAMP = '20261118'
LEARNER_ARMS = {'altbtr-novelty': 'altbtr-pf', 'altbtr-plain': 'altbtr-pf', 'bbf-novelty': 'bbf-pf',
                'officialbtr-novelty': 'officialbtr-pf'}


def arm(key):
    return load_config(ROOT / f'config/benchmark6h-{key}-{STAMP}.yaml')


def earlier(name):
    return load_config(ROOT / f'config/algorithm-benchmark-{name}-20261110.yaml')


def without(config, *keys):
    return {k: v for k, v in config.items() if k not in keys}


def test_paths_are_unique_and_outside_the_earlier_benchmark():
    paths = [arm(key)['paths'] for key in (*LEARNER_ARMS, 'worldmodel')]
    for field in ('runs', 'replay', 'user'):
        values = [p[field] for p in paths]
        assert len(set(values)) == len(values) and all('benchmark-6h' in v for v in values)


@pytest.mark.parametrize('key,base', LEARNER_ARMS.items())
def test_learner_arms_keep_the_earlier_learner_reward_and_environment(key, base):
    config, old = arm(key), earlier(base)
    for block in ('dqn', 'rewards', 'regional_reward', 'environment', 'actions', 'variations', 'parallel',
                  'training', 'official_btr'):
        assert config.get(block) == old.get(block), block
    assert config['study']['budget_seconds'] == 21600 and config['study']['seeds'] == 1
    assert ('position_novelty' in config) is key.endswith('novelty')


def test_novelty_arms_add_the_same_bonus_as_the_center_run_and_the_control_has_none():
    center = load_config(ROOT / 'config/center-altbtr-novelty-20261113.yaml')['position_novelty']
    for key in ('altbtr-novelty', 'bbf-novelty', 'officialbtr-novelty'):
        assert arm(key)['position_novelty'] == center
        assert replay_signature(arm(key))['position_novelty'] == center
    assert 'position_novelty' not in arm('altbtr-plain')
    assert 'position_novelty' not in replay_signature(arm('altbtr-plain'))


def test_control_and_novelty_arm_differ_only_in_novelty_paths_and_study():
    plain, novelty = arm('altbtr-plain'), arm('altbtr-novelty')
    assert without(plain, 'paths', 'study') == without(novelty, 'paths', 'study', 'position_novelty')


def test_novelty_arm_builds_an_environment(tmp_path):
    GalaxyEnv(arm('altbtr-novelty'), backend=None, run_dir=tmp_path / 'run', memory=object(), intrinsic=False)


def test_world_model_arm_keeps_reward_actions_and_every_quality_gate(tmp_path):
    config, old = arm('worldmodel'), earlier('altbtr-pf')
    assert 'dqn' not in config and 'position_novelty' not in config and config['replay']['enabled']
    for block in ('rewards', 'regional_reward', 'environment', 'actions', 'variations'):
        assert config[block] == old[block], block
    gates = load_config(ROOT / 'config/smg1-world-model.yaml')['world_model']
    assert config['world_model'] == gates
    assert config['study']['budget_seconds'] == 21600
    signature = replay_signature(config)
    assert len(signature['action_names']) == 8
    agent = ImaginedAgent(signature, deepcopy(config['world_model']), 'cpu')
    assert agent.real_decisions == 0 and agent.imagined_transitions == 0
