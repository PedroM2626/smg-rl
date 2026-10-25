import json
from dataclasses import replace

import numpy as np
import pytest

from smg_rl.frontier import FrontierArchive
from smg_rl.memory import Telemetry
from smg_rl.rewards import ProgressReward


class SnapshotBackend:
    def __init__(self, falling=False):
        self.telemetry = Telemetry((1000, 45, 0), 1, "fixture", 1, 1, 1, 0, False, False, 3, 3)
        self.memory = self
        self.falling = falling

    def sample(self):
        return self.telemetry

    def save_state(self, path):
        path.write_text(json.dumps(list(self.telemetry.position)))

    def load_state(self, path):
        self.telemetry = replace(self.telemetry, position=tuple(json.loads(path.read_text())))

    def advance(self, names):
        if self.falling:
            self.telemetry = replace(self.telemetry, position=(1000, -100, 0))


def test_archive_has_one_writer_and_releases_lock_on_close(tmp_path):
    first = FrontierArchive(tmp_path, {}, {"minimum_height": 0})
    with pytest.raises(RuntimeError, match="Another worker"):
        FrontierArchive(tmp_path, {}, {"minimum_height": 0})
    first.close()
    second = FrontierArchive(tmp_path, {}, {"minimum_height": 0})
    second.close()


def test_archive_probe_restore_integrity_and_no_reset_progress_reward(tmp_path):
    backend = SnapshotBackend()
    settings = {"minimum_height": 0, "reset_probability": 1}
    archive = FrontierArchive(tmp_path, {"game_id": "fixture", "scenario": 1}, settings)
    original = backend.sample()
    entry, probes = archive.consider(backend, original, [original] * 6, 0.1, {"episode": 1})
    assert entry["accepted"] and probes == 5 and backend.sample() == original
    assert archive.choose(np.random.default_rng(11))["file"] == entry["file"]
    progress = ProgressReward((0, 45, 0), (2000, 45, 0))
    progress.reset(original.position)
    assert progress.update(original.position)[0] == 0  # No reward for teleporting to a frontier.
    archive.state_path(entry).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="integrity"):
        archive.choose(np.random.default_rng(11))
    with pytest.raises(ValueError, match="filename"):
        archive.state_path({**entry, "file": "../outside.state"})


def test_falling_probe_is_quarantined_and_original_state_restored(tmp_path):
    backend = SnapshotBackend(falling=True)
    original = backend.sample()
    archive = FrontierArchive(tmp_path, {}, {"minimum_height": 0, "reset_probability": 1})
    entry, count = archive.consider(backend, original, [original] * 6, 0.1, {})
    assert not entry["accepted"] and count == 1
    assert backend.sample() == original
    assert archive.choose(np.random.default_rng(11)) is None
    assert (tmp_path / entry["file"]).exists()  # Rejected states are retained, never silently reused.
    unstable = [replace(original, position=(1000, 100 * i, 0)) for i in range(6)]
    assert archive.consider(backend, original, unstable, 0.2, {}) == (None, 0)
