"""Loopback-only DSU gamepad server; each Dolphin worker has its own UDP port."""

import socket
import struct
import threading
import time
import zlib


def valid_request(data):
    """Reject malformed lengths, versions and checksums before dispatch."""
    if len(data) < 20 or data[:4] != b"DSUC":
        return False
    version, length, checksum = struct.unpack_from("<HHI", data, 4)
    if version != 1001 or length != len(data) - 16:
        return False
    checked = bytearray(data)
    checked[8:12] = bytes(4)
    return zlib.crc32(checked) == checksum


def packet(kind, payload, uid=0x534D4752):
    data = bytearray(
        struct.pack("<4sHHII I", b"DSUS", 1001, 4 + len(payload), 0, uid, kind) + payload
    )
    struct.pack_into("<I", data, 8, zlib.crc32(data))
    return bytes(data)


def port_info(index=0):
    return struct.pack("<BBBB6sBB", index, 2 if index == 0 else 0, 2, 1, b"SMGRL0", 5, 0)


def pad_data(names, commands, count):
    data = bytearray(80)
    data[:12] = port_info()
    data[11] = 1
    struct.pack_into("<I", data, 12, count)
    # Button byte 1: Share/L3/R3/Options are dedicated emulator commands.
    data[16] = (1 if "pause" in commands else 0) | (2 if "capture" in commands else 0)
    data[16] |= 4 if "frame" in commands else 0
    data[18] = int("jump" in names)
    data[19] = int("crouch" in names)
    data[20:24] = bytes(
        [
            255 if "d" in names else 0 if "a" in names else 128,
            0 if "w" in names else 255 if "s" in names else 128,
            128,
            128,
        ]
    )
    # Four analog face buttons load the four local states; R1 shakes the Wiimote.
    for index in range(4):
        data[28 + index] = 255 if f"load{index + 1}" in commands else 0
    data[32] = 255 if "spin" in names else 0
    struct.pack_into("<Q", data, 48, time.monotonic_ns() // 1000)
    struct.pack_into("<fff", data, 56, 0.0, -1.0, 0.0)
    return bytes(data)


class DSUServer:
    def __init__(self, port=0):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", port))
        self.socket.settimeout(0.004)
        self.port = self.socket.getsockname()[1]
        self.names, self.commands = frozenset(), frozenset()
        self.clients = {}
        self.count = 0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def set(self, names=(), commands=()):
        self.names, self.commands = frozenset(names), frozenset(commands)

    def _loop(self):
        next_send = 0
        while self.running:
            try:
                request, peer = self.socket.recvfrom(256)
                kind = struct.unpack_from("<I", request, 16)[0] if valid_request(request) else None
                if kind == 0x100000:
                    self.socket.sendto(packet(kind, struct.pack("<H2x", 1001)), peer)
                elif kind == 0x100001:
                    for index in range(4):
                        self.socket.sendto(packet(kind, port_info(index)), peer)
                elif kind == 0x100002:
                    if len(request) >= 28 and (peer in self.clients or len(self.clients) < 8):
                        self.clients[peer] = time.monotonic()
            except socket.timeout:
                pass
            except OSError:
                if not self.running:
                    break
            now = time.monotonic()
            if now >= next_send:
                self.count = (self.count + 1) & 0xFFFFFFFF
                frame = packet(0x100002, pad_data(self.names, self.commands, self.count))
                for peer, seen in tuple(self.clients.items()):
                    if now - seen < 10:
                        try:
                            self.socket.sendto(frame, peer)
                        except OSError:
                            self.clients.pop(peer, None)
                    else:
                        self.clients.pop(peer, None)
                next_send = now + 0.004

    def close(self):
        self.running = False
        self.thread.join(timeout=1)
        self.socket.close()
