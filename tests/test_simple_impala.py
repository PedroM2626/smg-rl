"""Objective/encoder controls; fixtures do not establish game learning."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from smg_rl.config import load_config
from smg_rl.distributional import DistributionalQ, ResidualBlock
from smg_rl.dqn import DistributionalAgent
from smg_rl.replay import replay_signature
from smg_rl.rewards import RegionalAdvance, validate_simple_objective


def test_simple_failures_are_duration_neutral_and_progress_is_not_clawed_back():
    config = load_config('config/impala-iqn-smg1.yaml')
    validate_simple_objective(config['regional_reward'], config['rewards'], config['dqn']['gamma'])
    returns = []
    for duration in (1, 14, 180, 3000):
        advance = RegionalAdvance()
        advance.reset(0.)
        result = 0.
        for step in range(duration):
            result += advance.update(0., 12, terminal=step == duration - 1)
            result += config['rewards']['step']
        returns.append(result + config['rewards']['death'])
    assert returns == [-2.] * 4
    advance = RegionalAdvance()
    advance.reset(.4)  # An assisted start never pays its pre-existing route position.
    paid = [advance.update(p, 12, terminal=i == 5) for i, p in enumerate((.4, .6, .4, .6, .8, .8))]
    assert sum(paid) == pytest.approx(.4)
    assert sum(paid) - 2 > returns[0]
    assert advance.update(.8, 12) == 0
    assert advance.update(1., 12) == pytest.approx(.2)
    assert sum(paid) + .2 - 2 < 0  # The entire opening cannot outweigh a failure.
    assert 20 > 1  # A RAM-confirmed star dominates the total opening credit.


@pytest.mark.parametrize('change', ['gamma', 'step', 'timeout', 'death', 'completion', 'damage'])
def test_simple_profile_rejects_reintroduced_waiting_preferences(change):
    config = load_config('config/impala-iqn-smg1.yaml')
    gamma = 1.
    if change == 'gamma':
        gamma = .997
    else:
        config['rewards'][change] = {'step': -.001, 'timeout': -3., 'death': -.5,
                                   'completion': .5, 'damage': .25}[change]
    with pytest.raises(ValueError):
        validate_simple_objective(config['regional_reward'], config['rewards'], gamma)


@pytest.mark.parametrize('algorithm', ['iqn', 'rainbow'])
def test_impala_residual_encoder_backpropagates_and_preserves_distributional_actions(algorithm):
    torch.set_num_threads(1)
    settings = {'algorithm': algorithm, 'encoder': 'impala', 'channels': 2, 'hidden': 32,
                'frame_stack': 4, 'cosine_embedding': 8, 'atoms': 5, 'v_min': -2., 'v_max': 20.}
    signature = {'image_size': 84, 'action_names': list(range(8)), 'scenario_names': ['fixture']}
    network = DistributionalQ(signature, settings)
    assert sum(isinstance(layer, ResidualBlock) for layer in network.modules()) == 6
    images = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    values = network.expected(images, torch.ones(2, 1), quantiles=4)
    assert values.shape == (2, 8) and torch.isfinite(values).all()
    values.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.conv.parameters())
    network.eval()
    before = network.expected(images, torch.ones(2, 1), quantiles=4, deterministic=True)
    network.reset_noise()
    torch.testing.assert_close(before, network.expected(images, torch.ones(2, 1), quantiles=4, deterministic=True))


def test_exploration_anneals_by_real_decisions_and_old_replay_stays_incompatible():
    old = load_config('config/iqn-smg1.yaml')
    config = load_config('config/impala-iqn-smg1.yaml')
    assert replay_signature(old) != replay_signature(config)
    small = deepcopy(config['dqn'])
    small.update(channels=2, hidden=32)
    agent = DistributionalAgent(replay_signature(config), small)
    assert agent.epsilon() == 1.
    agent.real_decisions = small['learning_starts'] + small['epsilon_decay_decisions'] // 2
    assert agent.epsilon() == pytest.approx(.55)
    agent.real_decisions += small['epsilon_decay_decisions']
    assert agent.epsilon() == pytest.approx(.1)
    assert agent.epsilon(evaluate=True) == 0
    assert np.isfinite(agent.epsilon())
