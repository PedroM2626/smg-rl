import json
from pathlib import Path

from smg_rl.diagnostics import StageProfiler, atomic_json


def test_locked_status_keeps_previous_snapshot_and_does_not_abort(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    path.write_text('{"state": "previous"}')
    original = Path.replace

    def locked(source, destination):
        if destination == path:
            raise PermissionError("Viewer or antivirus owns the destination")
        return original(source, destination)

    monkeypatch.setattr(Path, "replace", locked)
    assert not atomic_json(path, {"state": "running"})
    assert json.loads(path.read_text())["state"] == "previous"
    assert list(tmp_path.iterdir()) == [path]


def test_transient_windows_lock_retries_and_publishes_complete_json(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    original = Path.replace
    calls = []

    def briefly_locked(source, destination):
        calls.append(source)
        if len(calls) < 3:
            raise PermissionError("Temporary file lock")
        return original(source, destination)

    monkeypatch.setattr(Path, "replace", briefly_locked)
    assert atomic_json(path, {"state": "running"})
    assert len(calls) == 3
    assert json.loads(path.read_text()) == {"state": "running"}


def test_stage_storage_is_bounded_and_failed_calls_are_measured():
    profiler = StageProfiler()
    for _ in range(600):
        with profiler.measure("capture"):
            pass
    try:
        with profiler.measure("reset"):
            raise RuntimeError("Failed operation")
    except RuntimeError:
        pass
    assert len(profiler.values["capture"]) == 512
    assert profiler.snapshot()["capture"]["count"] == 600
    assert profiler.snapshot()["reset"]["count"] == 1
