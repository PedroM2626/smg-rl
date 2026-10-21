"""Read-only SB4E01 telemetry, resolved through the game's object graph.

Offsets are derived from Garigari SB4E01 symbols and the live PowerPC code.
No host addresses, game patches, writes, or emulated function calls are used.
"""

import math
import struct
from dataclasses import asdict, dataclass
from typing import Protocol


class MemoryFault(RuntimeError):
    """Telemetry is unavailable or does not match the supported game."""


class Reader(Protocol):
    def read_bytes(self, address: int, size: int) -> bytes: ...


@dataclass(frozen=True)
class Telemetry:
    position: tuple[float, float, float]
    scenario: int
    stage: str
    actor: int
    actor_nerve: int
    scene_nerve: int
    scene_frame: int
    dead: bool
    complete: bool

    def to_dict(self):
        return asdict(self)


class GameMemory:
    ROOT = 0x807D0DA4
    PLAYING = 0x807D5310
    COMPLETE = (0x807D5318, 0x807D531C)
    # forceKill switches to +0x38; forceGameOverAbyss switches to +0x3C.
    DEATH = (0x807D4E38, 0x807D4E3C)

    def __init__(self, reader: Reader | None = None):
        if reader is None:
            import dolphin_memory_engine as engine

            engine.hook()
            if not engine.is_hooked():
                raise MemoryFault("Start exactly one Dolphin process and boot the game first.")
            reader = engine
        self.reader = reader
        if self.reader.read_bytes(0x80000000, 6) != b"SB4E01":
            raise MemoryFault("Only the USA SB4E01 game is supported by this address profile.")
        # Fail closed on another executable revision even if the game ID matches.
        expected = bytes.fromhex("806d9a8480630024")
        if self.reader.read_bytes(0x8045A970, 8) != expected:
            raise MemoryFault("The executable does not match the verified SB4E01 object graph.")

    def u32(self, address: int) -> int:
        return struct.unpack(">I", self.reader.read_bytes(address, 4))[0]

    def pointer(self, address: int) -> int:
        result = self.u32(address)
        if not (0x80000000 <= result < 0x81800000 or 0x90000000 <= result < 0x94000000):
            raise MemoryFault(f"Invalid pointer at 0x{address:08X}: 0x{result:08X}")
        if result % 4:
            raise MemoryFault(f"Unaligned pointer: 0x{result:08X}")
        return result

    def nerve(self, spine: int) -> int:
        # Spine::getCurrentNerve returns the pending nerve when non-null.
        return self.u32(spine + 8) or self.u32(spine + 4)

    def sample(self) -> Telemetry:
        try:
            root = self.pointer(self.ROOT)
            controller = self.pointer(root + 0x24)
            scene = self.pointer(controller + 0x12C)
            stage = self.reader.read_bytes(controller + 0x40, 64).split(b"\0")[0].decode("ascii")
            scenario = self.u32(controller + 0x80)
            if stage != "SecretAthleticGalaxy" or scenario not in (1, 2):
                raise MemoryFault(f"Unexpected stage/scenario: {stage}/{scenario}")
            # Verify the current scene is actually GameScene, not a stale heap object.
            if self.u32(scene) != 0x806EFA28:
                raise MemoryFault(f"Unexpected scene vtable: 0x{self.u32(scene):08X}")
            holder = self.pointer(scene + 0x10)
            mario_holder = self.pointer(holder + 0x54)
            actor = self.pointer(mario_holder + 0x14)
            position = struct.unpack(">fff", self.reader.read_bytes(actor + 0x14, 12))
            if not all(math.isfinite(v) and abs(v) < 1e7 for v in position):
                raise MemoryFault("Non-finite or implausible Mario coordinates.")
            actor_nerve = self.nerve(self.pointer(actor + 0x58))
            scene_spine = self.pointer(scene + 4)
            scene_nerve = self.nerve(scene_spine)
            return Telemetry(
                position,
                scenario,
                stage,
                actor,
                actor_nerve,
                scene_nerve,
                self.u32(scene_spine + 12),
                actor_nerve in self.DEATH or scene_nerve == 0x807D5320,
                scene_nerve in self.COMPLETE,
            )
        except MemoryFault:
            raise
        except Exception as exc:
            raise MemoryFault(f"Could not read game telemetry: {exc}") from exc
