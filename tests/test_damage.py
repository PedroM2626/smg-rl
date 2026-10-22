import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_env import make_env
from test_profiles import smg1_ram

from smg_rl.memory import GameMemory, MemoryFault
from smg_rl.rewards import DamageTracker


def test_real_layout_reads_health_and_zero_health_terminates_before_nerve_change():
    ram = smg1_ram()
    memory = GameMemory(ram, "RMGE01")
    assert memory.sample().health == memory.sample().max_health == 3
    ram.word(0x80006380, 2)
    assert memory.sample().health == 2 and not memory.sample().dead
    ram.word(0x80006380, 0)
    assert memory.sample().dead


@pytest.mark.parametrize("health,maximum", [(4, 3), (7, 6), (1, 0), (3, 9)])
def test_implausible_health_is_rejected(health, maximum):
    ram = smg1_ram()
    ram.word(0x80006380, health)
    ram.word(0x800063E0, maximum)
    with pytest.raises(MemoryFault, match="health telemetry"):
        GameMemory(ram, "RMGE01").sample()


def test_damage_is_not_repeated_on_hurt_frames_or_reset_and_healing_is_not_rewarded():
    tracker = DamageTracker()
    tracker.reset(3)
    assert [tracker.update(h) for h in (2, 2, 2, 3, 2)] == [1, 0, 0, 0, 1]
    tracker.reset(3)
    assert tracker.update(3) == 0
    tracker.reset(None)
    assert tracker.update(None) == 0


def test_nonlethal_damage_penalty_and_no_duplicate_death_penalty(tmp_path):
    env, backend = make_env(tmp_path)
    hp = [3]
    original = backend.sample
    backend.sample = lambda: replace(original(), health=hp[0], max_health=3, dead=hp[0] == 0)
    env.config["rewards"]["damage"] = -0.25
    try:
        obs, _ = env.reset(options={"variation": 0})
        assert set(obs) == {"image", "scenario"}
        hp[0] = 2
        _, reward, terminated, _, info = env.step(0)
        assert info["health_lost"] == 1 and info["damage_penalty"] == -0.25
        assert not terminated
        assert reward == pytest.approx(info["progress_reward"] - 0.2501)
        _, _, _, _, info = env.step(0)
        assert info["health_lost"] == 0 and info["damage_penalty"] == 0
        hp[0] = 0
        _, reward, terminated, _, info = env.step(0)
        assert terminated and info["health_lost"] == 2
        assert info["damage_penalty"] == 0
        assert reward == pytest.approx(-2.0001)
    finally:
        env.close()


def test_smg2_does_not_gain_health_detection_or_damage_penalties(tmp_path):
    env, _ = make_env(tmp_path)
    try:
        env.reset(options={"variation": 0})
        _, _, _, _, info = env.step(0)
        assert info["health"] is None
        assert info["health_lost"] == 0 and info["damage_penalty"] == 0
    finally:
        env.close()


def test_measured_live_damage_fixture_and_reward_replay_are_consistent():
    evidence = json.loads(
        Path("docs/assets/smg1/damage-validation.json").read_text(encoding="utf-8")
    )
    assert evidence["baseline"]["health"] == 3
    assert evidence["hit"]["health"] == 2 and not evidence["hit"]["dead"]
    losses = [r for r in evidence["reward_replay"] if r["health_lost"]]
    assert len(losses) == 1 and losses[0]["damage_penalty"] == -0.25
    assert evidence["restored_baseline"]["health"] == 3
