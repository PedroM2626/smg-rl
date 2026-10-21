import pytest

from smg_rl.dolphin import DesktopDolphin, prepare_profile


def test_pause_retries_and_rejects_unacknowledged_input():
    backend = object.__new__(DesktopDolphin)
    backend.paused = False
    backend.con = type("Keys", (), {"VK_F10": 121})
    events = []
    backend.detect_pause = lambda: False
    backend.hotkey = lambda key: events.append(key)
    backend.release = lambda: events.append("release")
    with pytest.raises(RuntimeError, match="acknowledge"):
        backend.pause(True)
    assert events == [121, 121, 121, "release"]


def test_held_controller_inputs_never_overlap_pause_shortcuts(monkeypatch):
    monkeypatch.setattr("smg_rl.dolphin.time.sleep", lambda _: None)
    backend = object.__new__(DesktopDolphin)
    backend.config = {"environment": {"action_seconds": 0.1}}
    backend.held = {"crouch", "jump"}
    actions = []
    backend.release = lambda: backend.held.clear()

    def set_action(names):
        backend.held = set(names)
        actions.append(set(names))

    def pause(value):
        assert not backend.held

    backend.set_action = set_action
    backend.pause = pause
    backend.advance(["w", "crouch", "jump"])
    assert actions == [{"w", "crouch", "jump"}]
    assert not backend.held


def test_profile_preparation_preserves_original_and_is_repeatable(tmp_path):
    source = tmp_path / "original"
    (source / "Config").mkdir(parents=True)
    original = "[Core]\nEmulationSpeed = 0.5\n"
    (source / "Config" / "Dolphin.ini").write_text(original)
    (source / "Config" / "WiimoteNew.ini").write_text("[Wiimote1]\nButtons/A = `Click 0`\n")
    (source / "Wii").mkdir()
    (source / "Wii" / "play_rec.dat").write_text("disposable")
    assets = tmp_path / "assets"
    assets.mkdir()
    for slot in range(1, 5):
        (assets / f"SB4E01.s{slot:02d}").write_bytes(b"local-state")
    config = {
        "paths": {
            "user": str(tmp_path / "profile"),
            "source_user": str(source),
            "assets": str(assets),
        }
    }
    target = prepare_profile(config)
    assert (source / "Config" / "Dolphin.ini").read_text() == original
    assert "F10" in (target / "Config" / "Hotkeys.ini").read_text()
    assert not (target / "Wii" / "play_rec.dat").exists()
    assert prepare_profile(config) == target
    assert len(list((target / "StateSaves").iterdir())) == 4


def test_preparation_refuses_to_modify_unowned_directory(tmp_path):
    with pytest.raises(RuntimeError, match="unowned"):
        prepare_profile({"paths": {"user": str(tmp_path)}})
