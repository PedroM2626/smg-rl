import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.vec_env import SubprocVecEnv

from smg_rl.vector import MeasuredSubprocVecEnv


class TinyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, delay):
        self.delay = delay
        self.steps = 0
        self.observation_space = spaces.Dict({"image": spaces.Box(0, 255, (4, 8, 8), np.uint8),
                                              "scenario": spaces.Box(0, 1, (1,), np.float32)})
        self.action_space = spaces.Discrete(3)

    def observation(self):
        return {"image": np.full((4, 8, 8), self.steps, np.uint8),
                "scenario": np.ones(1, np.float32)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self.observation(), {"seed": seed}

    def step(self, action):
        time.sleep(self.delay)
        self.steps += 1
        return self.observation(), float(action), False, self.steps == 2, {"action": int(action)}


@dataclass
class TinyFactory:
    delay: float

    def __call__(self):
        return TinyEnv(self.delay)


@pytest.mark.parametrize("shared,order", [(False, "ordered"), (False, "ready"), (True, "ready")])
def test_measured_collector_matches_sb3_autoreset_terminal_observations_and_ownership(shared, order):
    factories = [TinyFactory(0), TinyFactory(0)]
    expected = SubprocVecEnv(factories, start_method="spawn")
    actual = MeasuredSubprocVecEnv(factories, (4, 8, 8), shared=shared, receive_order=order)
    try:
        expected.seed(17)
        actual.seed(17)
        original = actual.reset()
        baseline = expected.reset()
        assert np.array_equal(original["image"], baseline["image"])
        assert actual.reset_infos == expected.reset_infos
        for _ in range(4):
            a = actual.step(np.array([1, 2]))
            b = expected.step(np.array([1, 2]))
            for key in a[0]:
                assert np.array_equal(a[0][key], b[0][key])
            assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
            if a[2][0]:
                assert a[3][0]["TimeLimit.truncated"]
                assert np.array_equal(a[3][0]["terminal_observation"]["image"],
                                      b[3][0]["terminal_observation"]["image"])
        assert not original["image"].any()
        assert actual.get_attr("steps") == expected.get_attr("steps")
        assert actual.env_method("observation")[0]["image"].shape == (4, 8, 8)
        assert actual.timing_snapshot()["stages"]["barrier"]["count"] == 4
    finally:
        expected.close()
        actual.close()


def test_worker_barrier_identifies_slow_environment_and_collects_ready_first():
    env = MeasuredSubprocVecEnv([TinyFactory(0.1), TinyFactory(0)], (4, 8, 8))
    try:
        env.reset()
        env.step(np.array([0, 0]))
        result = env.last_statistics
        assert result["slowest_worker"] == 0
        assert result["receive_order"] == [1, 0]
        assert result["barrier_idle_ms"][1] > 50
        assert result["compute_ms"][0] > result["compute_ms"][1]
    finally:
        env.close()
