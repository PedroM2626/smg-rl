from copy import deepcopy

import pytest
import torch

from smg_rl.config import load_config
from smg_rl.regional_transfer import rebase_regions
from smg_rl.replay import replay_signature
from smg_rl.world_model import ImaginedAgent


def test_regional_warm_start_preserves_vision_but_resets_reward_policy_and_readiness():
    config = load_config('config/regional-smg1-12.yaml')
    config['environment']['image_size'] = 16
    signature = replay_signature(config)
    old = deepcopy(signature)
    old.pop('regional_reward')
    old['reward_revision'] = 'historical'
    settings = {'deterministic_size': 8, 'stochastic_size': 2, 'embedding_size': 4,
                'hidden_size': 8, 'channels': 2}
    source = ImaginedAgent(old, settings)
    source.real_decisions, source.actor_updates = 4096, 1
    with torch.no_grad():
        source.world.reward[-1].bias.fill_(9)
        source.world.continuation[-1].bias.fill_(9)
        source.world.reward_high.fill_(20)
    target = rebase_regions(source, signature, settings, 42)
    for name, value in target.world.state_dict().items():
        if name.startswith(('reward.', 'continuation.')) or name in ('reward_low', 'reward_high'):
            continue
        torch.testing.assert_close(value, source.world.state_dict()[name])
    assert target.world.reward[-1].bias.item() != 9
    assert target.world.continuation[-1].bias.item() != 9
    assert target.world.reward_high.item() == 0
    assert target.real_decisions == target.actor_updates == target.model_updates == target.imagined_transitions == 0
    assert not target.model_optimizer.state and target.replay_rng_state is None
    assert source.real_decisions == 4096 and source.actor_updates == 1
    probabilities = target.actor(torch.zeros(1, target.world.features)).softmax(-1)[0]
    torch.testing.assert_close(probabilities[:5].sum(), torch.tensor(.8))
    changed = deepcopy(signature)
    changed['frame_skip'] = 4
    with pytest.raises(ValueError, match='interfaces'):
        rebase_regions(source, changed, settings, 42)
