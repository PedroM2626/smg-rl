"""Executable-specific, read-only telemetry layouts. Addresses are Wii addresses."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemoryProfile:
    game_id: str
    root: int
    signature_address: int
    signature: bytes
    scene_offset: int
    stage_offset: int
    scenario_offset: int
    scene_vtable: int
    mario_holder_offset: int
    actor_offset: int
    position_offset: int
    actor_spine_offset: int
    playing: int
    complete: tuple[int, ...]
    death: tuple[int, ...]
    game_over: int
    stage: str
    scenarios: tuple[int, ...]
    health_offset: int | None = None
    max_health_offset: int | None = None
    mario_offset: int | None = None
    movement_flags_offset: int | None = None
    grounded_mask: int | None = None
    grounded_exclusion_offset: int | None = None
    extra_stages: tuple[str, ...] = ()
    extra_scenarios: tuple[int, ...] = ()


PROFILES = {
    "SB4E01": MemoryProfile(
        "SB4E01",
        0x807D0DA4,
        0x8045A970,
        bytes.fromhex("806d9a8480630024"),
        0x12C,
        0x40,
        0x80,
        0x806EFA28,
        0x54,
        0x14,
        0x14,
        0x58,
        0x807D5310,
        (0x807D5318, 0x807D531C),
        (0x807D4E38, 0x807D4E3C),
        0x807D5320,
        "SecretAthleticGalaxy",
        (1, 2),
        mario_offset=0x584,
        movement_flags_offset=0x0C,
        grounded_mask=0x40000000,
        grounded_exclusion_offset=0xE24,
        extra_stages=("MarioOnMarioGalaxy",),
    ),
    "RMGE01": MemoryProfile(
        "RMGE01",
        0x806A1228,
        0x802E1520,
        bytes.fromhex("8063000c4e800020"),
        0xAC,
        0x20,
        0x40,
        0x805BD054,
        0x50,
        0x0C,
        0x0C,
        0x50,
        0x806A1BD0,
        (0x806A1BD8, 0x806A1BDC),
        tuple(range(0x8069F03C, 0x8069F058, 4)),
        0x806A1BE0,
        "BeltConveyerExGalaxy",
        (1,),
        health_offset=0x380,
        max_health_offset=0x3E0,
        mario_offset=0x230,
        movement_flags_offset=8,
        grounded_mask=0x40000000,
        extra_stages=("KoopaBattleVs3Galaxy", "FactoryGalaxy"),
        extra_scenarios=(5,),
    ),
}


def game_id(config):
    return config.get("game", {}).get("id", "SB4E01")


def require_training_enabled(config):
    if not config.get("training", {}).get("enabled", True):
        raise ValueError("This configuration is validation-only. Training is explicitly disabled.")
    if (Path(config.get("root", ".")) / ".local" / "training-hold.json").exists():
        raise ValueError("Training is on hold by explicit user request. Await authorization to resume.")
