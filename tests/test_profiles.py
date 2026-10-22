import struct
from types import SimpleNamespace

import pytest
from test_memory import RAM

from smg_rl.background import BackgroundDolphin
from smg_rl.config import load_config
from smg_rl.dolphin import prepare_profile
from smg_rl.memory import GameMemory, MemoryFault
from smg_rl.profiles import require_training_enabled


def smg1_ram():
    ram = RAM()
    ram.put(0x80000000, b"RMGE01")
    ram.put(0x802E1520, bytes.fromhex("8063000c4e800020"))
    for address, value in [
        (0x806A1228, 0x80001000),
        (0x80001024, 0x80002000),
        (0x800020AC, 0x80003000),
        (0x80003000, 0x805BD054),
        (0x80003010, 0x80004000),
        (0x80004050, 0x80005000),
        (0x8000500C, 0x80006000),
        (0x80006050, 0x80007000),
        (0x80003004, 0x80008000),
        (0x80007004, 0x8069F038),
        (0x80008004, 0x806A1BD0),
        (0x80002040, 1),
        (0x80006380, 3),
        (0x800063E0, 3),
    ]:
        ram.word(address, value)
    ram.put(0x80002020, b"BeltConveyerExGalaxy\0")
    ram.put(0x8000600C, struct.pack(">fff", -50, 45, -50))
    return ram


def test_smg1_independent_layout_and_completion():
    ram = smg1_ram()
    memory = GameMemory(ram, "RMGE01")
    first = memory.sample()
    assert first.position == (-50, 45, -50)
    assert not first.dead and not first.complete
    ram.word(0x80008008, 0x806A1BD8)
    assert memory.sample().complete
    # The other game's completion address must not satisfy this profile.
    ram.word(0x80008008, 0x807D5318)
    assert not memory.sample().complete


def test_smg1_live_verified_falling_death_and_cross_game_rejection():
    ram = smg1_ram()
    ram.word(0x80007008, 0x8069F040)
    assert GameMemory(ram, "RMGE01").sample().dead
    with pytest.raises(MemoryFault):
        GameMemory(ram)


def test_smg1_training_guard_precedes_environment_creation():
    config = load_config("config/smg1.yaml")
    with pytest.raises(ValueError, match="validation-only"):
        require_training_enabled(config)


def test_explicit_smg1_training_configuration_is_enabled_and_isolated(tmp_path):
    config = load_config("config/smg1-training.yaml")
    config["root"] = str(tmp_path)
    require_training_enabled(config)
    assert config["parallel"]["workers"] == 1
    assert config["variations"][0]["start_slot"] == 1
    assert config["training"]["enabled"]
    assert config["paths"]["runs"].endswith("runs\\smg1") or config["paths"]["runs"].endswith(
        "runs/smg1"
    )


@pytest.mark.parametrize("command", ["train", "train-parallel", "train-supervised"])
def test_smg1_cli_rejects_training_before_launch(command, monkeypatch):
    from smg_rl.cli import main

    monkeypatch.setattr("sys.argv", ["smg-rl", "--config", "config/smg1.yaml", command])
    with pytest.raises(ValueError, match="validation-only"):
        main()


def test_smg1_profile_requires_only_its_two_states(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (tmp_path / "source" / "Config").mkdir(parents=True)
    for slot in (1, 2):
        (assets / f"RMGE01.s{slot:02d}").write_bytes(b"local test state")
    config = load_config("config/smg1.yaml")
    config["paths"].update(
        assets=str(assets), user=str(tmp_path / "owned"), source_user=str(tmp_path / "source")
    )
    root = prepare_profile(config)
    assert sorted(p.name for p in (root / "StateSaves").iterdir()) == ["RMGE01.s01", "RMGE01.s02"]


def test_pause_reconciles_actual_state_without_unnecessary_toggle():
    backend = object.__new__(BackgroundDolphin)
    backend.paused = False
    backend.detect_pause = lambda: True
    backend._pulse = lambda _: pytest.fail("Already paused; must not toggle")
    backend.pause(True)
    assert backend.paused


def test_short_render_stall_is_not_classified_as_pause(monkeypatch):
    backend = object.__new__(BackgroundDolphin)
    samples = iter([0, 0, 0, 1, 2])
    backend.memory = SimpleNamespace(
        sample=lambda: SimpleNamespace(scene_frame=next(samples), scene_nerve=123)
    )
    monkeypatch.setattr("smg_rl.background.time.sleep", lambda _: None)
    assert not backend.detect_pause()


def test_one_queued_screenshot_frame_does_not_masquerade_as_unpaused(monkeypatch):
    backend = object.__new__(BackgroundDolphin)
    samples = iter([0, 1, 1, 1, 1, 1, 1])
    backend.memory = SimpleNamespace(
        sample=lambda: SimpleNamespace(scene_frame=next(samples), scene_nerve=123)
    )
    monkeypatch.setattr("smg_rl.background.time.sleep", lambda _: None)
    assert backend.detect_pause()
