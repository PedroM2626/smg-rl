import json
import zipfile
from pathlib import Path

import pytest

from smg_rl.supervision import newest_pair, supervised_train


def pair(root, steps):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"model-{steps}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps({"num_timesteps": steps}))
    with zipfile.ZipFile(path.with_suffix(".intrinsic.pt"), "w") as archive:
        archive.writestr("archive/data.pkl", b"checkpoint metadata test")
    return path


def test_recovery_ignores_incomplete_or_corrupt_pairs(tmp_path):
    expected = pair(tmp_path, 10)
    incomplete = pair(tmp_path, 20)
    incomplete.with_suffix(".intrinsic.pt").unlink()
    corrupt = pair(tmp_path, 30)
    corrupt.write_bytes(b"broken zip")
    assert newest_pair(tmp_path) == (10, expected)


def test_supervisor_restarts_at_last_complete_pair_and_preserves_total_budget(
    tmp_path, monkeypatch
):
    config = {
        "root": str(tmp_path),
        "paths": {"runs": str(tmp_path / "runs")},
        "training": {"enabled": True, "steps": 8},
        "parallel": {"workers": 1},
    }
    monkeypatch.setattr("smg_rl.supervision.load_config", lambda _: config)
    monkeypatch.setattr("smg_rl.supervision.time.sleep", lambda _: None)
    ticks = iter(range(0, 10000, 100))
    monkeypatch.setattr("smg_rl.supervision.time.monotonic", lambda: next(ticks))
    commands = []

    class Child:
        # Deliberately different from the controller PID, like Windows venv's launcher.
        pid = 9988

        def __init__(self, command, **kwargs):
            commands.append(command)
            attempt = int(kwargs["env"]["SMG_RL_SUPERVISOR_ATTEMPT"])
            self.code, self.polls = (1 if attempt == 1 else 0), 0
            run = Path(config["paths"]["runs"]) / f"parallel-{attempt}"
            pair(run, attempt * 4)
            (run / "controller.json").write_text(
                json.dumps(
                    {
                        "pid": 1122,
                        "supervisor": kwargs["env"]["SMG_RL_SUPERVISOR"],
                        "supervisor_attempt": attempt,
                    }
                )
            )
            (run / "status.json").write_text(
                json.dumps({"state": "failed" if self.code else "completed"})
            )

        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else self.code

        def wait(self, **kwargs):
            return self.code

    monkeypatch.setattr("smg_rl.supervision.subprocess.Popen", Child)
    folder = supervised_train(tmp_path / "config.yaml")
    status = json.loads((folder / "status.json").read_text())
    assert status["state"] == "completed" and status["attempts"] == 2
    assert status["confirmed_checkpoint_timesteps"] == 8
    assert commands[0][commands[0].index("--steps") + 1] == "8"
    assert commands[1][commands[1].index("--steps") + 1] == "4"
    assert "--resume" in commands[1]


def test_supervisor_bounds_permanent_failures_without_checkpoint_progress(tmp_path, monkeypatch):
    config = {"root": str(tmp_path), "paths": {"runs": str(tmp_path / "runs")},
              "training": {"enabled": True, "steps": 8},
              "parallel": {"workers": 1, "max_stalled_attempts": 2}}
    monkeypatch.setattr("smg_rl.supervision.load_config", lambda _: config)
    monkeypatch.setattr("smg_rl.supervision.time.sleep", lambda _: None)
    ticks = iter(range(0, 10000, 100))
    monkeypatch.setattr("smg_rl.supervision.time.monotonic", lambda: next(ticks))

    class Child:
        pid = 9988

        def __init__(self, *args, **kwargs):
            pass

        def poll(self):
            return 1

        def wait(self):
            return 1

    monkeypatch.setattr("smg_rl.supervision.subprocess.Popen", Child)
    with pytest.raises(RuntimeError, match="2 attempts without checkpoint progress"):
        supervised_train(tmp_path / "config.yaml")
    status = json.loads(next(Path(config["paths"]["runs"]).glob("supervised-*/status.json")).read_text())
    assert status["state"] == "failed" and status["attempts"] == 2
