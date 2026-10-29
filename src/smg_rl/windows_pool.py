"""Read Windows system pool-tag counters without changing tracing sessions."""

import ctypes
import os
import struct


def parse_pool_tags(data):
    if len(data) < 8:
        raise ValueError('Truncated pool-tag header.')
    count = struct.unpack_from('<I', data)[0]
    entry = struct.Struct('<4sII4xQIIQ')
    if count > 100000 or 8 + count * entry.size > len(data):
        raise ValueError('Invalid pool-tag record count.')
    result = []
    for index in range(count):
        tag, paged_alloc, paged_free, paged_bytes, nonpaged_alloc, nonpaged_free, nonpaged_bytes = entry.unpack_from(data, 8 + index * entry.size)
        result.append({'tag': tag.decode('ascii', errors='replace'), 'paged_bytes': paged_bytes,
                       'nonpaged_bytes': nonpaged_bytes, 'paged_outstanding': paged_alloc - paged_free,
                       'nonpaged_outstanding': nonpaged_alloc - nonpaged_free})
    return result


def windows_pool_tags():
    if os.name != 'nt' or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise RuntimeError('Pool-tag diagnostics require 64-bit Windows.')
    library = ctypes.WinDLL('ntdll')
    query = library.NtQuerySystemInformation
    query.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
    query.restype = ctypes.c_long
    size = 1024 * 1024
    for _ in range(6):
        buffer = ctypes.create_string_buffer(size)
        needed = ctypes.c_uint32()
        status = query(22, buffer, size, ctypes.byref(needed))
        if status == 0:
            return parse_pool_tags(buffer.raw)
        if status & 0xffffffff != 0xc0000004:
            raise OSError(f'Windows pool-tag query failed: 0x{status & 0xffffffff:08X}')
        size = max(size * 2, needed.value + 1024)
        if size > 64 * 1024 * 1024:
            break
    raise MemoryError('Pool-tag diagnostics exceeded their finite buffer bound.')
