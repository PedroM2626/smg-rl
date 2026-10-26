import numpy as np
import pytest
import torch

from smg_rl.distributional import (
    DistributionalQ,
    NoisyLinear,
    categorical_projection,
    quantile_huber,
)
from smg_rl.dqn import DistributionalAgent
from smg_rl.dqn_replay import PrioritizedFrames


def settings(algorithm='rainbow'):
    return {'algorithm': algorithm, 'frame_stack': 4, 'hidden': 32, 'channels': 2,
        'atoms': 3, 'v_min': -2., 'v_max': 2., 'quantiles': 4, 'cosine_embedding': 8,
        'learning_rate': .001, 'learning_starts': 1, 'epsilon': .01,
        'beta_start': .4, 'beta_steps': 1000, 'batch_size': 4,
        'max_grad_norm': 10., 'priority_epsilon': 1e-6, 'target_update_interval': 1}


def signature():
    return {'image_size': 40, 'scenario_names': ['fixture'], 'action_names': list('01234567')}


def replay_fixture():
    replay = PrioritizedFrames(16, 40, n_step=1, gamma=1.)
    for value in range(4):
        start = replay.add_frame(np.full((40, 40), value, np.uint8))
        end = replay.add_frame(np.full((40, 40), value + 1, np.uint8), start)
        replay.append(0, start, 0, 0., end, 0, False, True)
    return replay


def test_noisy_layers_resample_in_training_and_disable_noise_for_evaluation():
    layer = NoisyLinear(4, 2)
    x = torch.ones(3, 4)
    before = layer(x).detach()
    layer.reset_noise()
    assert not torch.equal(before, layer(x))
    layer.eval()
    before = layer(x).detach()
    layer.reset_noise()
    torch.testing.assert_close(before, layer(x))


def test_categorical_projection_preserves_exact_bin_mass_and_terminal_targets():
    support = torch.linspace(-2, 2, 5)
    probabilities = torch.tensor([[.1, .2, .4, .2, .1]])
    torch.testing.assert_close(categorical_projection(probabilities, torch.zeros(1), torch.ones(1), support), probabilities)
    for reward, expected in [(0., [0, 0, 1, 0, 0]), (.5, [0, 0, .5, .5, 0]), (99., [0, 0, 0, 0, 1])]:
        projected = categorical_projection(probabilities, torch.tensor([reward]), torch.zeros(1), support)
        torch.testing.assert_close(projected, torch.tensor([expected], dtype=torch.float32))
        assert projected.sum() == 1


def test_quantile_huber_has_asymmetric_tails_and_correct_gradient():
    prediction = torch.zeros(1, 1, requires_grad=True)
    tau = torch.tensor([[.25]])
    positive = quantile_huber(prediction, torch.ones(1, 1), tau)
    negative = quantile_huber(prediction, -torch.ones(1, 1), tau)
    assert positive.item() == pytest.approx(.125)
    assert negative.item() == pytest.approx(.375)
    positive.sum().backward()
    assert prediction.grad.item() < 0


@pytest.mark.parametrize('algorithm', ['rainbow', 'iqn'])
def test_both_distributional_learners_update_and_checkpoint_replay_rng(tmp_path, algorithm):
    torch.set_num_threads(1)
    agent = DistributionalAgent(signature(), settings(algorithm))
    replay = replay_fixture()
    metrics = agent.update(replay)
    assert np.isfinite(metrics['loss']) and agent.updates == agent.target_updates == 1
    agent.real_decisions = replay.emitted
    folder = tmp_path / 'checkpoint'
    agent.save(folder, replay)
    policy = DistributionalAgent.load_policy(folder, signature())
    assert policy.real_decisions == 4 and policy.updates == 1
    expected_sample = replay.sample(4, .7)['indices']
    observation = {'image': np.zeros((1, 4, 40, 40), np.uint8), 'scenario': np.ones((1, 1), np.float32)}
    expected_action, _ = agent.act(observation)
    restored, buffer = DistributionalAgent.load(folder, signature(), settings(algorithm),
        {'capacity': 16, 'image_size': 40, 'stack': 4, 'workers': 1, 'n_step': 1, 'gamma': 1., 'alpha': .5})
    np.testing.assert_array_equal(buffer.sample(4, .7)['indices'], expected_sample)
    action, _ = restored.act(observation)
    np.testing.assert_array_equal(action, expected_action)
    assert restored.real_decisions == 4
    with pytest.raises(FileExistsError):
        agent.save(folder, replay)
    with (folder / 'agent.pt').open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='integrity'):
        DistributionalAgent.load(folder, signature(), settings(algorithm), {})


def test_double_q_selects_online_action_instead_of_target_greedy(monkeypatch):
    agent = DistributionalAgent(signature(), settings())
    bias = agent.online.advantage[-1].bias_mu
    with torch.no_grad():
        bias.zero_()
    monkeypatch.setattr(agent.online, 'expected', lambda images, scenario: torch.tensor([[1.] + [0.] * 7]).expand(len(images), -1))
    monkeypatch.setattr(agent.online, 'forward', lambda images, scenario: bias.reshape(1, 8, 3).expand(len(images), -1, -1))
    target_logits = torch.full((1, 8, 3), -50.)
    target_logits[0, 0, 0] = 50.  # Online-selected action0 returns -2.
    target_logits[0, 1:, 2] = 50.  # A target-greedy bug would choose +2.
    monkeypatch.setattr(agent.target, 'forward', lambda images, scenario: target_logits.expand(len(images), -1, -1))
    agent.update(replay_fixture())
    assert bias[0].item() > 0 and bias[2].item() < 0


def test_iqn_explicit_quantiles_and_evaluation_values_are_deterministic():
    network = DistributionalQ(signature(), settings('iqn')).eval()
    images, scenario = torch.zeros(2, 4, 40, 40), torch.ones(2, 1)
    with pytest.raises(ValueError):
        network(images, scenario)
    values = network.expected(images, scenario, 8, deterministic=True)
    network.reset_noise()
    torch.testing.assert_close(values, network.expected(images, scenario, 8, deterministic=True))


def test_seeded_stochastic_act_changes_actions_without_updates_or_weight_changes():
    torch.set_num_threads(1)
    s = settings('iqn')
    s.update(epsilon=0., evaluation_epsilon=0.)
    agent = DistributionalAgent(signature(), s)
    agent.real_decisions = 1000
    before = {name: tensor.clone() for name, tensor in agent.online.state_dict().items()}
    observation = {'image': np.zeros((1, 4, 40, 40), np.uint8), 'scenario': np.ones((1, 1), np.float32)}
    def trace(seed):
        torch.manual_seed(seed)
        agent.rng = np.random.default_rng(seed)
        result = []
        for _ in range(16):
            action, metrics = agent.act(observation, evaluate=False, return_values=True)
            assert not metrics['epsilon_random_mask'][0]
            result.append(int(action[0]))
        return result
    first = trace(11)
    assert len(set(first)) > 1 and trace(11) == first and trace(23) != first
    assert agent.updates == 0 and agent.real_decisions == 1000
    for name, value in agent.online.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_explicit_random_action_provenance_is_not_confused_with_noisynet():
    s = settings()
    s.update(evaluation_epsilon=1.)
    agent = DistributionalAgent(signature(), s)
    observation = {'image': np.zeros((2, 4, 40, 40), np.uint8), 'scenario': np.ones((2, 1), np.float32)}
    _, metrics = agent.act(observation, evaluate=True, return_values=True)
    assert metrics['epsilon_random_mask'].all() and not agent.online.training
