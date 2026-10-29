import os
import socket
import struct
import zlib

import cv2
import numpy as np
import pytest

from smg_rl.background import BackgroundDolphin, prepare_worker
from smg_rl.dsu import DSUServer, packet, pad_data
from smg_rl.memory import MemoryFault
from smg_rl.process_memory import ProcessMemory


def test_dsu_wire_packet_has_valid_crc_and_length():
    response = packet(0x100001, b"example")
    magic, version, length, checksum, _, kind = struct.unpack_from("<4sHHIII", response)
    assert (magic, version, length, kind) == (b"DSUS", 1001, len(response) - 16, 0x100001)
    unsigned = bytearray(response)
    struct.pack_into("<I", unsigned, 8, 0)
    assert zlib.crc32(unsigned) == checksum


def test_dsu_game_controls_are_separate_from_emulator_commands():
    state = pad_data({"w", "a", "jump", "crouch", "spin"}, {"pause", "load3"}, 7)
    assert state[16] == 1
    assert state[18:22] == bytes([1, 1, 0, 0])
    assert list(state[28:33]) == [0, 0, 255, 0, 255]
    released = pad_data(set(), set(), 8)
    assert released[16:20] == bytes(4)
    assert released[20:24] == bytes([128] * 4)


def test_loopback_servers_have_independent_ports_and_states():
    first, second = DSUServer(), DSUServer()
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(1)
    try:
        assert first.port != second.port
        first.set({"jump"})
        request = bytearray(packet(0x100002, bytes(8)))
        request[:4] = b"DSUC"
        struct.pack_into("<I", request, 8, 0)
        struct.pack_into("<I", request, 8, zlib.crc32(request))
        client.sendto(request, ("127.0.0.1", first.port))
        state, _ = client.recvfrom(256)
        assert state[:4] == b"DSUS" and state[20 + 18] == 1
        assert not second.names
    finally:
        first.close()
        second.close()
        client.close()


def test_worker_configuration_enables_unfocused_hotkeys_without_modifying_source(
    tmp_path, monkeypatch
):
    root = tmp_path / "worker"
    (root / "Config").mkdir(parents=True)
    (root / "Config/Dolphin.ini").write_text(
        "[Core]\nEmulationSpeed = 1.0\n[DSP]\nBackend = Cubeb\nVolume = 75\n"
    )
    monkeypatch.setattr("smg_rl.background.prepare_profile", lambda _: root)
    config = {"paths": {"user": str(tmp_path / "template")}, "parallel": {"emulation_speed": 2.0}}
    prepare_worker(config, 0, 29000)
    assert "HotkeysRequireFocus = False" in (root / "Config/Dolphin.ini").read_text()
    assert "BackgroundInput = True" in (root / "Config/Dolphin.ini").read_text()
    assert "Backend = No Audio Output" in (root / "Config/Dolphin.ini").read_text()
    assert "29000" in (root / "Config/DSUClient.ini").read_text()
    assert "DSUClient/0/SMG-RL" in (root / "Config/WiimoteNew.ini").read_text()
    assert config["paths"]["user"] == str(tmp_path / "template")


def test_capture_rejects_old_png_instead_of_reusing_stale_vision(tmp_path):
    old = tmp_path / "old.png"
    cv2.imwrite(str(old), np.arange(84 * 84, dtype=np.uint8).reshape(84, 84))
    os.utime(old, (1, 1))
    backend = object.__new__(BackgroundDolphin)
    backend.config = {"parallel": {"timeout": 0.01}}
    backend.screenshots = tmp_path
    backend.last_signature = None
    backend.index = 0
    backend._pulse = lambda _: None
    with pytest.raises(RuntimeError, match="fresh rendered screenshot"):
        backend.capture()


@pytest.mark.parametrize("address,size", [(0x7FFFFFFF, 1), (0x817FFFFF, 2), (0x90000000, 4)])
def test_pid_memory_reader_rejects_unmapped_ranges(address, size):
    reader = object.__new__(ProcessMemory)
    reader.pid, reader.mem1, reader.mem2 = 123, 0x100000000, None
    with pytest.raises(MemoryFault):
        reader.read_bytes(address, size)
