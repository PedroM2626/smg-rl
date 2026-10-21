"""PID-scoped, read-only access to mapped Dolphin RAM on 64-bit Windows."""

import ctypes as ct
from ctypes import wintypes as wt

from smg_rl.memory import MemoryFault
from smg_rl.profiles import PROFILES


class Region(ct.Structure):
    _fields_ = [
        ("base", ct.c_void_p),
        ("allocation", ct.c_void_p),
        ("allocation_protect", wt.DWORD),
        ("partition", wt.WORD),
        ("size", ct.c_size_t),
        ("state", wt.DWORD),
        ("protect", wt.DWORD),
        ("kind", wt.DWORD),
    ]


class ProcessMemory:
    def __init__(self, pid, game_id="SB4E01"):
        profile = PROFILES[game_id]
        self.pid = pid
        self.api = ct.WinDLL("kernel32", use_last_error=True)
        self.api.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
        self.api.OpenProcess.restype = wt.HANDLE
        self.api.VirtualQueryEx.argtypes = [wt.HANDLE, ct.c_void_p, ct.POINTER(Region), ct.c_size_t]
        self.api.VirtualQueryEx.restype = ct.c_size_t
        self.api.ReadProcessMemory.argtypes = [
            wt.HANDLE,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_size_t,
            ct.POINTER(ct.c_size_t),
        ]
        self.api.CloseHandle.argtypes = [wt.HANDLE]
        # No PROCESS_VM_WRITE or PROCESS_VM_OPERATION permission is requested.
        self.handle = self.api.OpenProcess(0x0400 | 0x0010, False, pid)
        if not self.handle:
            raise MemoryFault(f"Cannot open Dolphin PID {pid} for reading.")
        self.mem1, self.mem2 = None, None
        regions = []
        address = 0
        region = Region()
        while self.api.VirtualQueryEx(self.handle, address, ct.byref(region), ct.sizeof(region)):
            if region.state == 0x1000 and not region.protect & (0x100 | 0x01):
                regions.append((region.base, region.size))
                if region.size >= 0x1800000:
                    try:
                        if (
                            self._read(region.base, 6) == profile.game_id.encode("ascii")
                            and self._read(region.base + profile.signature_address - 0x80000000, 8)
                            == profile.signature
                        ):
                            self.mem1 = region.base
                    except MemoryFault:
                        pass
            address = (region.base or 0) + region.size
        if self.mem1 is None:
            self.close()
            raise MemoryFault(f"Could not locate verified {game_id} RAM in PID {pid}.")
        candidates = [
            base
            for base, size in regions
            if size == 0x4000000 and abs(base - self.mem1) < 0x10000000
        ]
        if len(candidates) == 1:
            self.mem2 = candidates[0]

    def _read(self, address, size):
        buffer = ct.create_string_buffer(size)
        received = ct.c_size_t()
        if (
            not self.api.ReadProcessMemory(self.handle, address, buffer, size, ct.byref(received))
            or received.value != size
        ):
            raise MemoryFault(
                f"Cannot read {size} bytes at host address 0x{address:X} in PID {self.pid}."
            )
        return buffer.raw

    def read_bytes(self, address, size):
        if 0x80000000 <= address and address + size <= 0x81800000:
            return self._read(self.mem1 + address - 0x80000000, size)
        if self.mem2 and 0x90000000 <= address and address + size <= 0x94000000:
            return self._read(self.mem2 + address - 0x90000000, size)
        raise MemoryFault(f"Unmapped emulated range 0x{address:X} in PID {self.pid}.")

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None
