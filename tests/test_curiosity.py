import numpy as np
import torch

from smg_rl.curiosity import EpisodicMemory, IntrinsicReward


def test_stationary_revisits_decay_and_episode_reset_restores_novelty():
    memory = EpisodicMemory()
    embedding = np.ones(64)
    first = memory.reward(embedding)
    for _ in range(20):
        last = memory.reward(embedding)
    assert last < first
    memory.reset()
    assert memory.reward(embedding) == first


def test_curiosity_checkpoint_roundtrip_and_frozen_embedding():
    torch.set_num_threads(1)
    config = dict(embedding_size=16, neighbors=5, capacity=20, rnd_clip=5)
    curiosity = IntrinsicReward(54, config)
    image = np.full((84, 84), 100, dtype=np.uint8)
    curiosity.reset(image)
    frozen = {k: v.clone() for k, v in curiosity.episodic_encoder.state_dict().items()}
    reward, metrics = curiosity.step(image, image + 10, 5)
    assert np.isfinite(reward) and metrics["rnd_error"] >= 0
    assert all(
        torch.equal(v, frozen[k]) for k, v in curiosity.episodic_encoder.state_dict().items()
    )
    restored = IntrinsicReward(54, config)
    restored.load_state_dict(curiosity.state_dict())
    assert restored.error_stats.count == 1
    assert all(
        torch.equal(a, b)
        for a, b in zip(restored.target.parameters(), curiosity.target.parameters())
    )
