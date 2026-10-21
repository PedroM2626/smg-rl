import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from smg_rl.config import load_config
from smg_rl.env import GalaxyEnv
from smg_rl.memory import Telemetry


class FakeDolphin:
    def __init__(self):
        self.slot = 1
        self.steps = 0
        self.closed = False

    def load_slot(self, slot):
        self.slot, self.steps = slot, 0

    def advance(self, names):
        self.steps += 1

    def image(self):
        return np.full((84, 84), self.steps, dtype=np.uint8)

    def sample(self):
        return Telemetry(
            (0, -1150, 500 - 100 * self.steps),
            1 if self.slot == 1 else 2,
            "SecretAthleticGalaxy",
            0x81458EC0,
            0x807D4E34,
            0x807D5310,
            self.steps,
            False,
            False,
        )

    def close(self):
        self.closed = True


def make_env(tmp_path):
    config = load_config()
    calibration = {"validated": True, "variations": {}}
    for name in ("spinning", "double_time"):
        calibration["variations"][name] = {"start": [0, -1150, 500], "goal": [0, -450, -10000]}
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration))
    config["paths"]["calibration"] = str(path)
    backend = FakeDolphin()
    env = GalaxyEnv(config, backend, tmp_path / "run", memory=backend, intrinsic=False)
    return env, backend


def test_policy_observation_excludes_memory_and_encodes_scenario(tmp_path):
    env, backend = make_env(tmp_path)
    try:
        obs, info = env.reset(options={"variation": 1})
        assert set(obs) == {"image", "scenario"}
        assert env.observation_space.contains(obs)
        assert obs["scenario"].tolist() == [0, 1]
        assert "position" in info
        obs, reward, terminated, truncated, info = env.step(1)
        assert reward > 0 and not terminated and not truncated
        assert not np.array_equal(obs["image"][0], obs["image"][-1])
    finally:
        env.close()
    assert backend.closed
    assert list((tmp_path / "run").rglob("*.png"))


def test_terminal_death_penalty_is_logged_and_no_intrinsic_is_used(tmp_path):
    env, backend = make_env(tmp_path)
    try:
        env.reset(options={"variation": 0})
        baseline = backend.sample
        backend.sample = lambda: replace(baseline(), dead=True)
        _, reward, terminated, truncated, info = env.step(1)
        assert reward < -1 and terminated and not truncated
        assert info["intrinsic_reward"] == 0
        assert not env.rows
    finally:
        env.close()


def test_live_snapshot_includes_in_progress_high_water(tmp_path):
    env, _ = make_env(tmp_path)
    try:
        env.reset(options={"variation": 0})
        for _ in range(16):
            env.step(1)
        import csv

        with (env.run_dir / "live.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 17
        assert rows[0]["action"] == "reset"
        assert float(rows[-1]["best_progress"]) > 0
        assert rows[-1]["outcome"] == "running"
    finally:
        env.close()


def test_viewer_file_lock_does_not_abort_collection(tmp_path, monkeypatch):
    env, _ = make_env(tmp_path)
    original = Path.replace

    def locked(path, target):
        if path.name == "live.tmp":
            raise PermissionError("Viewer is reading the previous live snapshot")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", locked)
    try:
        env.reset(options={"variation": 0})
        env.step(1)
    finally:
        env.close()


def test_single_scenario_smg1_observation_has_its_own_shape(tmp_path):
    config = load_config("config/smg1-training.yaml")
    backend = FakeDolphin()
    original = backend.sample
    backend.sample = lambda: replace(
        original(), position=(-50, 45, -50 - 100 * backend.steps), stage="BeltConveyerExGalaxy"
    )
    env = GalaxyEnv(config, backend, tmp_path / "single", memory=backend, intrinsic=False)
    try:
        obs, _ = env.reset()
        assert obs["scenario"].tolist() == [1]
        assert env.observation_space.contains(obs)
        _, _, terminated, truncated, _ = env.step(1)
        assert not terminated and not truncated
        assert backend.slot == 1
    finally:
        env.close()
