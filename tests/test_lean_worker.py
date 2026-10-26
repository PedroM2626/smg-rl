"""Real spawned collectors, but synthetic environments; no game-completion claim."""

import sys
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces


class PureEnvironment(gym.Env):
    def __init__(self):
        self.steps = 0
        self.observation_space = spaces.Dict({'image': spaces.Box(0, 255, (4, 8, 8), np.uint8),
                                              'scenario': spaces.Box(0, 1, (1,), np.float32)})
        self.action_space = spaces.Discrete(8)

    def observation(self):
        return {'image': np.full((4, 8, 8), self.steps, np.uint8), 'scenario': np.ones(1, np.float32)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self.observation(), {'seed': seed}

    def step(self, action):
        self.steps += 1
        return self.observation(), float(action), self.steps == 2, False, {'action': int(action)}

    def runtime(self):
        import os

        return {'torch': 'torch' in sys.modules, 'sb3': 'stable_baselines3' in sys.modules,
                'blas_threads': os.environ.get('OPENBLAS_NUM_THREADS')}


@dataclass
class PureFactory:
    def __call__(self):
        return PureEnvironment()


def test_spawned_lean_collector_preserves_autoreset_terminal_pixels_without_ml_imports():
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from smg_rl.vector import MeasuredSubprocVecEnv

    lean = MeasuredSubprocVecEnv([PureFactory()], (4, 8, 8), shared=False, lean=True)
    baseline = SubprocVecEnv([PureFactory()], start_method='spawn')
    try:
        lean.seed(17)
        baseline.seed(17)
        first, expected = lean.reset(), baseline.reset()
        np.testing.assert_array_equal(first['image'], expected['image'])
        assert lean.reset_infos == baseline.reset_infos
        assert lean.env_method('runtime') == [{'torch': False, 'sb3': False, 'blas_threads': '1'}]
        for _ in range(4):
            actual, reference = lean.step([3]), baseline.step([3])
            np.testing.assert_array_equal(actual[0]['image'], reference[0]['image'])
            np.testing.assert_array_equal(actual[1], reference[1])
            np.testing.assert_array_equal(actual[2], reference[2])
            assert actual[3][0]['TimeLimit.truncated'] == reference[3][0]['TimeLimit.truncated']
            if actual[2][0]:
                np.testing.assert_array_equal(actual[3][0]['terminal_observation']['image'],
                                              reference[3][0]['terminal_observation']['image'])
        assert not first['image'].any()
    finally:
        lean.close()
        baseline.close()


def test_lean_transport_rejects_incompatible_shared_pixel_mode():
    from smg_rl.vector import MeasuredSubprocVecEnv

    with pytest.raises(ValueError, match='owned pipe'):
        MeasuredSubprocVecEnv([PureFactory()], (4, 8, 8), shared=True, lean=True)


def test_commit_guard_can_fail_even_when_physical_ram_looks_available():
    from smg_rl.resource_memory import memory_safe

    assert not memory_safe({'commit_available_bytes': 2**30, 'physical_available_bytes': 12 * 2**30})
    assert not memory_safe({'commit_available_bytes': 12 * 2**30, 'physical_available_bytes': 128 * 2**20})
    assert memory_safe({'commit_available_bytes': 12 * 2**30, 'physical_available_bytes': 2 * 2**30})
