import struct

import pytest

from smg_rl.memory import GameMemory, MemoryFault


class RAM:
    def __init__(self):
        self.data = {}

    def put(self, address, data):
        self.data.update({address + i: value for i, value in enumerate(data)})

    def word(self, address, value):
        self.put(address, struct.pack(">I", value))

    def read_bytes(self, address, size):
        return bytes(self.data.get(address + i, 0) for i in range(size))


@pytest.fixture
def ram():
    ram = RAM()
    ram.put(0x80000000, b"SB4E01")
    ram.put(0x8045A970, bytes.fromhex("806d9a8480630024"))
    for address, value in [
        (GameMemory.ROOT, 0x80001000),
        (0x80001024, 0x80002000),
        (0x8000212C, 0x80003000),
        (0x80003000, 0x806EFA28),
        (0x80003010, 0x80004000),
        (0x80004054, 0x80005000),
        (0x80005014, 0x80006000),
        (0x80006058, 0x80007000),
        (0x80003004, 0x80008000),
        (0x80007004, 0x807D4E34),
        (0x80008004, GameMemory.PLAYING),
        (0x80002080, 1),
        (0x80006000 + 0x584, 0x80009000),
        (0x8000900C, 0x40000000),
    ]:
        ram.word(address, value)
    ram.put(0x80002040, b"SecretAthleticGalaxy\0")
    ram.put(0x80006014, struct.pack(">fff", 10, -1150, -4000))
    return ram


def test_live_baseline_is_not_completion(ram):
    telemetry = GameMemory(ram).sample()
    assert telemetry.position == (10, -1150, -4000)
    assert not telemetry.dead and not telemetry.complete


@pytest.mark.parametrize("scenario", [1, 2])
def test_completion_uses_same_nerve_in_both_variations(ram, scenario):
    ram.word(0x80002080, scenario)
    ram.word(0x80008008, 0x807D5318)
    assert GameMemory(ram).sample().complete


def test_pending_death_nerve_overrides_playable_nerve(ram):
    ram.word(0x80007008, 0x807D4E3C)
    telemetry = GameMemory(ram).sample()
    assert telemetry.dead and not telemetry.complete


def test_resolves_actor_after_heap_relocation(ram):
    memory = GameMemory(ram)
    assert memory.sample().actor == 0x80006000
    ram.word(0x80005014, 0x90001000)
    ram.word(0x90001058, 0x80007000)
    ram.word(0x90001000 + 0x584, 0x80009000)
    ram.put(0x90001014, struct.pack(">fff", 20, -450, -37000))
    assert memory.sample().position == (20, -450, -37000)


def test_smg2_ground_contact_is_game_specific_and_excludes_alternate_actor_mode(ram):
    assert GameMemory(ram).sample().grounded is True
    ram.word(0x8000900C, 0)
    assert GameMemory(ram).sample().grounded is False
    ram.put(0x80006000 + 0xE24, b'\1')
    assert GameMemory(ram).sample().grounded is None


@pytest.mark.parametrize("mutation", ["game", "code", "pointer", "stage", "scene", "nan"])
def test_rejects_unsafe_telemetry(ram, mutation):
    if mutation == "game":
        ram.put(0x80000000, b"SB4P01")
    elif mutation == "code":
        ram.put(0x8045A970, b"\0" * 8)
    elif mutation == "pointer":
        ram.word(0x80005014, 0)
    elif mutation == "stage":
        ram.put(0x80002040, b"WrongStage\0")
    elif mutation == "scene":
        ram.word(0x80003000, 0x806E0000)
    else:
        ram.put(0x80006014, struct.pack(">fff", float("nan"), 0, 0))
    with pytest.raises(MemoryFault):
        GameMemory(ram).sample()


def test_purple_coin_signal_and_signal_health_are_read_only_when_configured(ram):
    ram.word(0x80006000 + 0x6B0, 2)
    ram.word(0x80A75D08, 41)
    plain = GameMemory(ram).sample()
    assert plain.purple_coins is None and plain.health is None
    signals = {'purple_coins': {'address': 0x80A75D08, 'limit': 100}, 'health': {'actor_offset': 0x6B0, 'max': 3}}
    configured = GameMemory(ram, signals=signals).sample()
    assert configured.purple_coins == 41 and configured.health == 2 and configured.max_health == 3


def test_implausible_purple_coin_counts_and_health_are_refused(ram):
    ram.word(0x80A75D08, 5000)
    with pytest.raises(MemoryFault, match='purple coin'):
        GameMemory(ram, signals={'purple_coins': {'address': 0x80A75D08, 'limit': 100}}).sample()
    ram.word(0x80006000 + 0x6B0, 9)
    with pytest.raises(MemoryFault, match='health'):
        GameMemory(ram, signals={'health': {'actor_offset': 0x6B0, 'max': 3}}).sample()


def test_extra_stages_are_limited_to_known_profile_stages(ram):
    assert GameMemory(ram, stage='MarioOnMarioGalaxy').expected_stage == 'MarioOnMarioGalaxy'
    with pytest.raises(ValueError, match='stage'):
        GameMemory(ram, stage='UnknownGalaxy')
