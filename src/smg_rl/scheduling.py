"""Optional Windows hybrid-CPU affinity, discovered rather than hard-coded."""

import ctypes
import os
import struct

import psutil


def cpu_topology():
    if os.name != "nt":
        return []
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel.GetSystemCpuSetInformation
    query.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
                      ctypes.c_void_p, ctypes.c_ulong]
    query.restype = ctypes.c_int
    required = ctypes.c_ulong()
    query(None, 0, ctypes.byref(required), None, 0)
    if not required.value:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_string_buffer(required.value)
    if not query(buffer, len(buffer), ctypes.byref(required), None, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    return parse_cpu_sets(buffer.raw[:required.value])


def parse_cpu_sets(data):
    result, offset = [], 0
    while offset < len(data):
        if len(data) - offset < 8:
            raise ValueError("Truncated CPU set header.")
        size, kind = struct.unpack_from("<II", data, offset)
        if size < 8 or offset + size > len(data):
            raise ValueError("Invalid CPU set record length.")
        if kind == 0:
            if size < 32:
                raise ValueError("Truncated CPU set record.")
            _, group, logical, core, _, _, efficiency, flags = struct.unpack_from(
                "<IH6B", data, offset + 8)
            result.append(dict(group=group, logical=logical, core=core,
                               efficiency=efficiency, flags=flags))
        offset += size
    return result


def performance_cpus(topology, allowed):
    # psutil's mask is group-relative; do not silently mis-pin multi-group hosts.
    if not topology or any(cpu["group"] != 0 for cpu in topology):
        return list(allowed)
    usable = [cpu for cpu in topology if cpu["logical"] in allowed
              and not (cpu["flags"] & 2 and not cpu["flags"] & 4)]
    if not usable:
        return list(allowed)
    fastest = max(cpu["efficiency"] for cpu in usable)
    return [cpu["logical"] for cpu in usable if cpu["efficiency"] == fastest]


def apply_affinity(pid, mode="auto"):
    if mode not in ("auto", "performance"):
        raise ValueError("CPU affinity must be auto or performance.")
    process = psutil.Process(pid)
    allowed = process.cpu_affinity()
    chosen = performance_cpus(cpu_topology(), allowed) if mode == "performance" else allowed
    if chosen != allowed:
        process.cpu_affinity(chosen)
    return {"mode": mode, "cpus": chosen}
