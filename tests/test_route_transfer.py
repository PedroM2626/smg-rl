from copy import deepcopy

import pytest
import torch

from smg_rl.config import load_config
from smg_rl.dqn import DistributionalAgent
from smg_rl.dqn_replay import PrioritizedFrames
from smg_rl.replay import replay_signature
from smg_rl.route_transfer import transfer_policy


def test_signed_route_transfer_resets_replay_and_optimizer(tmp_path):
    original = load_config('config/smg1-six-two-hour.yaml')
    original['environment']['image_size'] = 40
    original['parallel']['workers'] = 1
    original['dqn'].update(capacity=32, batch_size=2, learning_starts=8, channels=2, hidden=16, quantiles=4)
    settings = original['dqn']
    source_agent = DistributionalAgent(replay_signature(original), settings, 'cpu', 11)
    replay = PrioritizedFrames(32, 40, stack=4, workers=1, n_step=3, gamma=1, alpha=.5)
    source = tmp_path / 'source'
    source_agent.save(source, replay)
    updated = deepcopy(original)
    updated['rewards']['revision'] = 'regional-advance-expandable-v1'
    destination = tmp_path / 'transfer'
    record = transfer_policy(source, original, updated, destination)
    assert record['optimizer_reset'] and record['replay_reset'] and record['new_real_decisions'] == 0
    loaded = DistributionalAgent.load_policy(destination, replay_signature(updated), 'cpu')
    assert all(torch.equal(value, loaded.online.state_dict()[name])
               for name, value in source_agent.online.state_dict().items())
    with pytest.raises(FileExistsError):
        transfer_policy(source, original, updated, destination)
    with pytest.raises(ValueError, match='changed reward signature'):
        transfer_policy(source, original, original, tmp_path / 'same')
