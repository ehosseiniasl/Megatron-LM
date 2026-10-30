# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Release unused CPU memory via PyTorch or pinned NGC 26.07 mimalloc; no rebuild."""

import ctypes
import functools
import hashlib
import logging
import os
import platform
import struct
from pathlib import Path

import torch

from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)

TORCH_VERSION = '2.13.0a0+9186a08b2c.nv26.07'
C10_SHA256 = '50aca75b5b0f214aaf2c42f3d2febae78508286ff6e35d59a6536e7e09f30135'


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f'--release-cpu-memory-after-optimizer-init: {message}')


class _LinkMap(ctypes.Structure):
    """Public prefix of glibc's struct link_map from <link.h>."""

    _fields_ = [('l_addr', ctypes.c_size_t), ('l_name', ctypes.c_char_p)]


@functools.cache
def _resolve_mimalloc_collect():
    """Resolve mi_collect and allocator getters once per process; does not collect memory."""
    _require(platform.machine() == 'aarch64', 'expected AArch64')
    _require(torch.__version__ == TORCH_VERSION, f'expected PyTorch {TORCH_VERSION}')
    path = (Path(torch.__file__).resolve().parent / 'lib/libc10.so').resolve()
    data = path.read_bytes()
    _require(hashlib.sha256(data).hexdigest() == C10_SHA256, 'libc10 hash mismatch')
    header = struct.unpack_from('<16sHHIQQQIHHHHHH', data)
    _require(
        header[0][:6] == b'\x7fELF\x02\x01' and header[1:3] == (3, 183), 'unexpected ELF format'
    )
    programs = [
        struct.unpack_from('<IIQQQQQQ', data, header[5] + i * header[9]) for i in range(header[10])
    ]
    sections = [
        struct.unpack_from('<IIQQQQIIQQ', data, header[6] + i * header[11])
        for i in range(header[12])
    ]
    symbols = []
    for section in sections:
        if section[1] != 2:
            continue
        names_section = sections[section[6]]
        names = data[names_section[4] : names_section[4] + names_section[5]]
        for offset in range(section[4], section[4] + section[5], section[9]):
            name, info, _, index, value, size = struct.unpack_from('<IBBHQQ', data, offset)
            if names[name : names.find(b'\0', name)] == b'mi_collect':
                _require(info & 15 == 2 and index != 0 and size >= 4, 'invalid mi_collect symbol')
                symbols.append((value, size))
    _require(len(symbols) == 1, 'expected one mi_collect symbol')
    value, size = symbols[0]
    segments = [
        p
        for p in programs
        if p[0] == 1 and p[1] & 1 and p[3] <= value and value + size <= p[3] + p[5]
    ]
    _require(len(segments) == 1, 'mi_collect is not in one executable segment')
    segment = segments[0]
    file_offset = segment[2] + value - segment[3]
    # Take a handle to the existing allocator; never load another mimalloc instance.
    library = ctypes.CDLL(str(path), mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
    dlinfo = ctypes.CDLL(None).dlinfo
    dlinfo.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    dlinfo.restype = ctypes.c_int
    link_map = ctypes.POINTER(_LinkMap)()
    # RTLD_DI_LINKMAP = 2. l_addr is the load bias, including BOLT's ELF layout.
    _require(dlinfo(library._handle, 2, ctypes.byref(link_map)) == 0, 'dlinfo failed')
    _require(bool(link_map), 'missing libc10 link map')
    _require(Path(os.fsdecode(link_map.contents.l_name)).resolve() == path, 'wrong loaded libc10')
    address = link_map.contents.l_addr + value
    _require(
        ctypes.string_at(address, size) == data[file_offset : file_offset + size],
        'loaded mi_collect bytes differ from the file',
    )
    getters = []
    # Exported C++ getters declared in c10/core/CPUAllocator.h. Their pointers
    # identify whether the active allocator is one of the supported built-ins.
    for name in (
        '_ZN3c1015GetCPUAllocatorEv',  # c10::GetCPUAllocator()
        '_ZN3c1022GetDefaultCPUAllocatorEv',  # c10::GetDefaultCPUAllocator()
        '_ZN3c1028GetDefaultMobileCPUAllocatorEv',  # c10::GetDefaultMobileCPUAllocator()
    ):
        getter = getattr(library, name)
        getter.argtypes = []
        getter.restype = ctypes.c_void_p
        getters.append(getter)
    collect = ctypes.CFUNCTYPE(None, ctypes.c_bool)(address)
    return library, getters, collect


def _get_cpu_memory_release():
    purge_delay = os.environ.get('MIMALLOC_PURGE_DELAY')
    if purge_delay is not None:
        try:
            delay = int(purge_delay)
        except ValueError as error:
            raise RuntimeError('MIMALLOC_PURGE_DELAY must be an integer or unset.') from error
        _require(delay >= 0, 'MIMALLOC_PURGE_DELAY < 0 disables purging; unset it before launch.')

    release = getattr(torch.cpu, 'release_unused_memory', None)
    if callable(release):
        return release

    # Private fallback: only the exact, qualified vendor binary is safe to call.
    _, (current, default, mobile), collect = _resolve_mimalloc_collect()
    _require(
        current() in (default(), mobile()), 'custom CPU allocator does not support collection.'
    )
    return functools.partial(collect, True)  # mi_collect returns void, not a reclaimed-byte count.


def validate_cpu_memory_release() -> None:
    """Check support before model construction, without collecting memory."""
    _get_cpu_memory_release()


def release_cpu_memory_after_setup() -> None:
    """Collect unused pages on the calling thread after optimizer and checkpoint setup."""
    release = _get_cpu_memory_release()
    if release() is False:
        raise RuntimeError(
            'The active CPU allocator does not support torch.cpu.release_unused_memory().'
        )
    log_single_rank(
        logger, logging.INFO, 'Requested CPU memory release after model/optimizer setup.'
    )
