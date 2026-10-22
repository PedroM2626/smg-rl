import numpy as np
import torch

from smg_rl.curiosity import EpisodicMemory, IntrinsicReward


def module():
    torch.set_num_threads(1)
    return IntrinsicReward(54, dict(embedding_size=16, neighbors=5, capacity=20,
                                    rnd_clip=5, update_every=4))


def test_minibatch_delays_updates_but_computes_rewards_and_freezes_episode_encoder():
    m = module()
    image = np.full((84, 84), 100, np.uint8)
    m.reset(image)
    old = {k: v.clone() for k, v in m.encoder.state_dict().items()}
    frozen = {k: v.clone() for k, v in m.episodic_encoder.state_dict().items()}
    for i in range(3):
        reward, info = m.step(image, image + i + 1, i)
        assert np.isfinite(reward) and info["curiosity_update"] == 0
    assert all(torch.equal(v, old[k]) for k, v in m.encoder.state_dict().items())
    _, info = m.step(image, image + 4, 4)
    assert info["curiosity_batch_size"] == 4 and m.updates == 1
    assert any(not torch.equal(v, old[k]) for k, v in m.encoder.state_dict().items())
    assert all(torch.equal(v, frozen[k]) for k, v in m.episodic_encoder.state_dict().items())


def test_pending_minibatch_survives_safe_checkpoint_and_flushes_before_episode_reset(tmp_path):
    m = module()
    image = np.full((84, 84), 100, np.uint8)
    m.reset(image)
    m.step(image, image + 1, 2)
    path = tmp_path / "curiosity.pt"
    torch.save(m.state_dict(), path)
    restored = module()
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert len(restored.pending) == 1 and restored.error_stats.count == 1
    restored.reset(image)
    assert not restored.pending and restored.updates == 1
    assert all(torch.equal(a, b) for a, b in zip(
        restored.encoder.parameters(), restored.episodic_encoder.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(m.target.parameters(), restored.target.parameters()))


def test_circular_knn_storage_preserves_nearest_set_at_capacity():
    memory = EpisodicMemory(neighbors=2, capacity=3)
    for i in range(8):
        memory.reward(np.full(4, i, np.float32))
    assert memory.count == 3
    assert set(memory.values[:, 0]) == {5, 6, 7}
    for _ in range(20):
        last = memory.reward(np.full(4, 7, np.float32))
    assert last < 1
