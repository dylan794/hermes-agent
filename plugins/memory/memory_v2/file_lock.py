"""Small fail-closed cross-platform advisory file-lock helpers."""

from __future__ import annotations

import errno
import os
from typing import BinaryIO

try:  # POSIX
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    _fcntl = None

try:  # Windows
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - exercised on POSIX
    _msvcrt = None


class FileLockUnavailableError(RuntimeError):
    """Raised when the runtime has no supported process-lock primitive."""


def _prepare_windows_lock_byte(handle: BinaryIO) -> None:
    """Ensure byte zero exists because ``msvcrt.locking`` locks a byte range."""
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
        os.fsync(handle.fileno())
    handle.seek(0)


def try_acquire_exclusive(handle: BinaryIO) -> bool:
    """Try to acquire an exclusive lock, returning false only for contention."""
    if _fcntl is not None:
        try:
            _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False
    if _msvcrt is not None:
        _prepare_windows_lock_byte(handle)
        try:
            _msvcrt.locking(handle.fileno(), _msvcrt.LK_NBLCK, 1)
            return True
        except OSError as exc:
            # EACCES/EAGAIN are the documented contention signals. Windows
            # may expose ERROR_LOCK_VIOLATION as winerror 33 or 36 depending
            # on the Python/runtime combination.
            if exc.errno in {errno.EACCES, errno.EAGAIN} or getattr(exc, "winerror", None) in {33, 36}:
                return False
            raise
    raise FileLockUnavailableError(
        "no supported cross-process file-lock primitive is available"
    )


def release_exclusive(handle: BinaryIO) -> None:
    """Release a lock previously acquired with :func:`try_acquire_exclusive`."""
    if _fcntl is not None:
        _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)
        return
    if _msvcrt is not None:
        handle.seek(0)
        _msvcrt.locking(handle.fileno(), _msvcrt.LK_UNLCK, 1)
        return
    raise FileLockUnavailableError(
        "no supported cross-process file-lock primitive is available"
    )
