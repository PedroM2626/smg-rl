"""Attribution is a counterfactual gradient calculation, never optimizer training."""

import torch
from torch.distributions import Categorical

from smg_rl.actions import ACTIONS
from smg_rl.objective_audit import _gradient, actor_objective, audit_batch
from smg_rl.world_model import ImaginedAgent, lambda_returns, weighted_mean


def test_fixed_rollout_matches_training_objective_and_zero_bonus_has_zero_effect():
    torch.set_num_threads(1)
    torch.manual_seed(5)
    agent = ImaginedAgent({"image_size": 16, "scenario_names": ["test"],
                           "action_names": [str(i) for i in ACTIONS]},
                          {"deterministic_size": 8, "stochastic_size": 2,
                           "embedding_size": 4, "hidden_size": 8, "channels": 2})
    features = torch.randn(3, 4, agent.world.features)
    actions = torch.randint(54, (3, 4))
    reward = torch.randn(3, 4)
    bonus = torch.zeros_like(reward)
    discounts = torch.full_like(reward, 0.97)
    values, bootstrap = torch.randn(3, 4), torch.randn(4)
    args = (features, actions, reward, bonus, discounts, values, bootstrap)
    loss, returns = actor_objective(agent, *args, 0.01, 0.003)
    target = lambda_returns(reward, discounts, torch.cat([values[1:], bootstrap[None]]), bootstrap).clamp(-100, 100)
    advantage = target - values
    advantage = (advantage / advantage.std(unbiased=False).clamp_min(1)).clamp(-10, 10)
    weights = torch.cat([torch.ones_like(discounts[:1]), discounts[:-1]]).cumprod(0)
    dist = Categorical(logits=agent.actor(features))
    expected = -weighted_mean(dist.log_prob(actions) * advantage + 0.003 * dist.entropy(), weights)
    torch.testing.assert_close(returns, target)
    torch.testing.assert_close(loss, expected)
    before = {k: v.clone() for k, v in agent.actor.state_dict().items()}
    full, _ = _gradient(agent, *args, 0.01, 0.003)
    zero, _ = _gradient(agent, *args, 0.0, 0.003)
    torch.testing.assert_close(full, zero)
    assert all(torch.equal(v, before[k]) for k, v in agent.actor.state_dict().items())
    assert all(p.grad is None for p in agent.actor.parameters())
    assert agent.actor_updates == agent.real_decisions == 0


def test_full_audit_preserves_modules_and_counts_on_portable_visual_batch():
    import numpy as np

    torch.set_num_threads(1)
    agent = ImaginedAgent({"image_size": 16, "scenario_names": ["test"],
                           "action_names": [str(i) for i in ACTIONS]},
                          {"deterministic_size": 8, "stochastic_size": 2,
                           "embedding_size": 4, "hidden_size": 8, "channels": 2,
                           "imagination_horizon": 2, "imagination_batch": 4})
    batch = {"images": np.zeros((2, 4, 16, 16), np.uint8),
             "actions": np.zeros((2, 3), np.int64), "rewards": np.zeros((2, 3), np.float32),
             "terminated": np.zeros((2, 3), bool), "mask": np.ones((2, 3), np.float32),
             "is_first": np.array([[True, False, False, False]] * 2),
             "scenario": np.ones((2, 1), np.float32)}
    before = {k: v.clone() for k, v in agent.world.state_dict().items()}
    row = audit_batch(agent, batch, 11)
    assert row["transitions"] == 8
    assert 0 <= row["special_action_probability"] <= 1
    assert row["remove_intrinsic_gradient_cosine"] <= 1.000001
    assert all(torch.equal(v, before[k]) for k, v in agent.world.state_dict().items())
    assert agent.actor_updates == agent.real_decisions == agent.imagined_transitions == 0
