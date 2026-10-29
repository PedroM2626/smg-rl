"""Windows commit headroom is distinct from free physical RAM."""

import ctypes
import os
from ctypes import wintypes

import psutil


def memory_snapshot():
    memory = psutil.virtual_memory()
    result = {'physical_available_bytes': memory.available, 'physical_total_bytes': memory.total}
    if os.name == 'nt':
        class PerformanceInfo(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in (
                'CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal', 'PhysicalAvailable',
                'SystemCache', 'KernelTotal', 'KernelPaged', 'KernelNonpaged', 'PageSize')] + [
                (name, wintypes.DWORD) for name in ('HandleCount', 'ProcessCount', 'ThreadCount')]
        info = PerformanceInfo()
        info.cb = ctypes.sizeof(info)
        library = ctypes.WinDLL('psapi', use_last_error=True)
        library.GetPerformanceInfo.argtypes = [ctypes.POINTER(PerformanceInfo), wintypes.DWORD]
        library.GetPerformanceInfo.restype = wintypes.BOOL
        if not library.GetPerformanceInfo(ctypes.byref(info), info.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        result.update(commit_total_bytes=info.CommitTotal * info.PageSize,
            commit_limit_bytes=info.CommitLimit * info.PageSize,
            commit_available_bytes=(info.CommitLimit - info.CommitTotal) * info.PageSize,
            kernel_paged_bytes=info.KernelPaged * info.PageSize,
            kernel_nonpaged_bytes=info.KernelNonpaged * info.PageSize)
    else:
        result['commit_available_bytes'] = memory.available
    return result


def memory_safe(snapshot, reserve_bytes=6 * 2**30, physical_reserve_bytes=512 * 2**20):
    if type(physical_reserve_bytes) is not int or physical_reserve_bytes < 512 * 2**20:
        raise ValueError('The physical memory guard cannot be weakened.')
    return (snapshot['commit_available_bytes'] >= reserve_bytes
            and snapshot['physical_available_bytes'] >= physical_reserve_bytes)


def require_memory_headroom():
    snapshot = memory_snapshot()
    if not memory_safe(snapshot):
        raise MemoryError(f'Insufficient Windows commit/physical headroom for a safe block: {snapshot}')
    return snapshot
