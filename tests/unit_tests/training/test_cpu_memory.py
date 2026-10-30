# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reject unsupported CPU collection instead of logging a successful request."""

from unittest import mock

import pytest

from megatron.training import cpu_memory


@pytest.mark.parametrize(
    'failure, message',
    [
        ('disabled_purge', 'MIMALLOC_PURGE_DELAY < 0 disables purging'),
        ('unsupported_build', 'libc10 hash mismatch'),
        ('custom_allocator', 'custom CPU allocator'),
        ('native_rejection', 'active CPU allocator does not support'),
    ],
)
def test_release_fails_loudly(monkeypatch, failure, message):
    monkeypatch.delenv('MIMALLOC_PURGE_DELAY', raising=False)
    native_release = mock.Mock(return_value=False)
    monkeypatch.setattr(
        cpu_memory.torch.cpu, 'release_unused_memory', native_release, raising=False
    )
    collect = mock.Mock()
    resolve = mock.Mock(return_value=(None, (lambda: 3, lambda: 1, lambda: 2), collect))
    monkeypatch.setattr(cpu_memory, '_resolve_mimalloc_collect', resolve)
    if failure == 'disabled_purge':
        monkeypatch.setenv('MIMALLOC_PURGE_DELAY', '-1')
    elif failure != 'native_rejection':
        monkeypatch.delattr(cpu_memory.torch.cpu, 'release_unused_memory')
        if failure == 'unsupported_build':
            resolve.side_effect = RuntimeError('libc10 hash mismatch')

    if failure != 'native_rejection':
        with pytest.raises(RuntimeError, match=message):
            cpu_memory.validate_cpu_memory_release()
    else:
        cpu_memory.validate_cpu_memory_release()
        native_release.assert_not_called()  # Validation must not collect memory.
        resolve.assert_not_called()  # Prefer the native API over the pinned fallback.
    with pytest.raises(RuntimeError, match=message):
        cpu_memory.release_cpu_memory_after_setup()
    collect.assert_not_called()
