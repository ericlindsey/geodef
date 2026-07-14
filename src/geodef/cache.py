"""Hash-based disk caching for expensive matrix computations.

Caches Green's matrices and stress kernels to `.npz` files keyed by a
SHA-256 hash of all computation inputs. Identical inputs always find the
cache; changed inputs always recompute.

Usage::

    import geodef
    geodef.cache.set_dir("path/to/cache")
    geodef.cache.clear()
    geodef.cache.disable()
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Version stamp for the numerical kernels feeding cached results. Bump this
# whenever a change to any engine (okada85/okada92/tri), Green's assembly, or
# projection alters numerical output: every existing cache entry then misses,
# so a stale pre-fix matrix can never be served. Record bumps in CHANGELOG.md.
KERNEL_VERSION: int = 1


@dataclasses.dataclass
class _CacheConfig:
    """Internal mutable state for the cache module."""

    directory: Path = dataclasses.field(default_factory=lambda: Path(".geodef_cache"))
    enabled: bool = True


_config = _CacheConfig()


# ====================================================================
# Module-level configuration API
# ====================================================================


def set_dir(path: str | Path) -> None:
    """Set the cache directory.

    Args:
        path: Directory path for cached ``.npz`` files.
    """
    _config.directory = Path(path)


def get_dir() -> Path:
    """Return the current cache directory.

    Returns:
        Cache directory as a ``Path``.
    """
    return _config.directory


def enable() -> None:
    """Enable disk caching (the default)."""
    _config.enabled = True


def disable() -> None:
    """Disable disk caching. All computations will run without cache."""
    _config.enabled = False


def is_enabled() -> bool:
    """Return whether caching is currently enabled.

    Returns:
        ``True`` if caching is active.
    """
    return _config.enabled


def clear() -> None:
    """Remove all cached files from the cache directory."""
    d = _config.directory
    if d.exists():
        shutil.rmtree(d)


def info() -> dict[str, int]:
    """Return cache statistics.

    Returns:
        Dict with ``n_files`` (number of cached ``.npz`` files)
        and ``total_bytes`` (total size on disk).
    """
    d = _config.directory
    if not d.exists():
        return {"n_files": 0, "total_bytes": 0}
    files = list(d.rglob("*.npz"))
    return {
        "n_files": len(files),
        "total_bytes": sum(f.stat().st_size for f in files),
    }


# ====================================================================
# Hash computation
# ====================================================================


def compute_hash(key_data: dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hex digest from a dict of inputs.

    Args:
        key_data: Dict mapping names to values. Supported value types:
            ``np.ndarray``, ``str``, ``int``, ``float``, ``None``.

    Returns:
        64-character lowercase hex digest string.
    """
    h = hashlib.sha256()
    for key in sorted(key_data):
        h.update(key.encode("utf-8"))
        h.update(b"\x00")
        val = key_data[key]
        if isinstance(val, np.ndarray):
            h.update(val.dtype.str.encode("utf-8"))
            h.update(np.array(val.shape, dtype=np.int64).tobytes())
            h.update(val.tobytes())
        elif isinstance(val, str):
            h.update(b"s")
            h.update(val.encode("utf-8"))
        elif val is None:
            h.update(b"None")
        elif isinstance(val, (int, float)):
            h.update(repr(val).encode("utf-8"))
        else:
            raise TypeError(f"Unsupported type in key_data: {type(val)}")
    return h.hexdigest()


# ====================================================================
# Core caching primitive
# ====================================================================


def _compute_context() -> dict[str, Any]:
    """Implicit inputs that affect every cached computation.

    The kernel version stamp plus the active backend and precision are merged
    into every cache key so that a kernel fix, a backend switch, or a
    float32/float64 change can never serve a stale entry.
    """
    from geodef import backend

    return {
        "__kernel_version": KERNEL_VERSION,
        "__backend": backend.get_backend(),
        "__precision": backend.get_precision(),
    }


def cached_compute(
    key_data: dict[str, Any],
    compute_fn: Callable[[], np.ndarray],
) -> np.ndarray:
    """Compute a matrix, caching the result to disk.

    On the first call with a given set of inputs, ``compute_fn`` is invoked
    and the result is saved as a compressed ``.npz`` file. Subsequent calls
    with identical ``key_data`` load from disk without recomputing.

    The hash additionally covers the module's ``KERNEL_VERSION`` stamp and
    the active backend name/precision, so entries written under a different
    compute context are never returned. Entries orphaned by a context change
    stay on disk until ``clear()`` removes them.

    Args:
        key_data: Dict describing all explicit inputs that affect the
            result. Used to build a deterministic hash for the cache
            filename. Keys starting with ``__`` are reserved for the
            implicit compute context.
        compute_fn: Zero-argument callable that returns the matrix.

    Returns:
        The computed (or cached) numpy array.
    """
    if not _config.enabled:
        return compute_fn()

    hex_hash = compute_hash({**key_data, **_compute_context()})
    cache_path = _config.directory / hex_hash[:2] / f"{hex_hash}.npz"

    if cache_path.exists():
        logger.debug("Cache hit: %s", hex_hash[:12])
        return np.load(cache_path)["data"]

    result = compute_fn()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, data=result)
    logger.debug("Cache save: %s", hex_hash[:12])

    return result
