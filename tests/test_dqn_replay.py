import numpy as np
import pytest

from smg_rl.dqn_replay import PrioritizedFrames


def frame(value):
    return np.full((4, 4), value, np.uint8)


def test_n_step_worker_boundaries_terminal_truncation_and_collector_cut():
    replay = PrioritizedFrames(8, 4, workers=2, n_step=3, gamma=.5)
    a = replay.add_frame(frame(10))
    b = replay.add_frame(frame(11), a)
    c = replay.add_frame(frame(12), b)
    x = replay.add_frame(frame(200))
    y = replay.add_frame(frame(201), x)
    replay.append(0, a, 1, 1., b, 0, False, False)
    replay.append(1, x, 2, 10., y, 1, False, True, assisted=True)  # Truncation bootstraps.
    replay.append(0, b, 1, 2., c, 0, True, True)
    assert replay.arrays['reward'][:3].tolist() == [10, 2, 2]
    assert replay.arrays['discount'][:3].tolist() == [.5, 0, 0]
    assert replay.arrays['horizon'][:3].tolist() == [1, 2, 1]
    assert replay.observation(b)[:, 0, 0].tolist() == [10, 10, 10, 11]
    assert replay.observation(y)[:, 0, 0].tolist() == [200, 200, 200, 201]
    start = replay.add_frame(frame(99))
    end = replay.add_frame(frame(100), start)
    replay.append(0, start, 0, 4., end, 0, False, False)
    replay.flush_cuts()
    assert replay.arrays['reward'][3] == 4 and replay.arrays['discount'][3] == .5
    assert replay.emitted == 4


def test_prioritized_sampling_importance_weights_and_duplicate_priority_order():
    replay = PrioritizedFrames(2, 4, n_step=1, alpha=1.)
    for value in (0, 1):
        start = replay.add_frame(frame(value))
        end = replay.add_frame(frame(value + 1), start)
        replay.append(0, start, 0, 0., end, 0, True, True)
    replay.priorities([0, 1], [1., 4.])
    batch = replay.sample(100, 1.)
    assert np.count_nonzero(batch['indices'] == 1) == 80
    np.testing.assert_allclose(batch['weights'][batch['indices'] == 1], .25)
    np.testing.assert_allclose(batch['weights'][batch['indices'] == 0], 1.)
    replay.priorities([0, 0], [10., 1.])
    assert replay.tree[replay.base] == 10
    with pytest.raises(ValueError):
        replay.priorities([0], [np.nan])


def test_frame_ring_wrap_does_not_cross_episode_or_worker_histories():
    replay = PrioritizedFrames(3, 4, workers=2, n_step=3)
    for iteration in range(100):
        for worker in range(2):
            start = replay.add_frame(frame(worker * 100))
            current = start
            for step in range(4):
                next_frame = replay.add_frame(frame(worker * 100 + step + 1), current)
                replay.append(worker, current, 0, 0., next_frame, worker, step == 3, step == 3)
                current = next_frame
        sample = replay.sample(4, .4)
        assert sample['images'].min() >= 100  # Last3 transitions belong to worker1.
        assert sample['next_images'].min() >= 100
    with pytest.raises(ValueError, match='Stale'):
        replay.observation(0)


def test_worker_expansion_remaps_wrapped_frame_ids_and_preserves_sampling_rng(tmp_path):
    replay = PrioritizedFrames(8, 4, workers=2, n_step=3)
    for episode in range(70):
        current = replay.add_frame(frame(episode))
        for step in range(4):
            following = replay.add_frame(frame(episode + step + 1), current)
            replay.append(episode % 2, current, step, float(step), following, 0, step == 3, step == 3)
            current = following
    before = {int(i): replay.observation(int(i)) for key in ('state', 'next') for i in replay.arrays[key][:replay.size]}
    expanded = replay.expanded_workers(6)
    assert expanded.workers == len(expanded.pending) == 6 and not any(expanded.pending)
    assert expanded.emitted == replay.emitted and expanded.next_frame == replay.next_frame
    for identity, pixels in before.items():
        np.testing.assert_array_equal(expanded.observation(identity), pixels)
    expected, actual = replay.sample(16, .6), expanded.sample(16, .6)
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key])
    path = tmp_path / 'expanded.npz'
    expanded.save(path)
    restored = PrioritizedFrames.load(path, {'capacity':8,'image_size':4,'stack':4,'workers':6,
        'n_step':3,'gamma':.997,'alpha':.5})
    for worker in range(6):
        first = restored.add_frame(frame(100 + worker))
        last = restored.add_frame(frame(110 + worker), first)
        restored.append(worker, first, 0, 0., last, 0, True, True)
    assert restored.emitted == replay.emitted + 6


def test_worker_expansion_refuses_pending_transitions_and_shrinking():
    replay = PrioritizedFrames(8, 4, workers=2)
    first, last = replay.add_frame(frame(0)), replay.add_frame(frame(1))
    replay.append(0, first, 0, 0., last, 0, False, False)
    with pytest.raises(ValueError, match='flushed'):
        replay.expanded_workers(6)
    replay.flush_cuts()
    with pytest.raises(ValueError):
        replay.expanded_workers(1)
