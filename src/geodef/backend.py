"""Array-backend selection for GeoDef's accelerated compute paths.

GeoDef's Green's-function kernels are pure, vectorized array math and can
run on either NumPy (the default) or JAX. The JAX backend JIT-compiles the
kernels through XLA on CPUs and offloads to a GPU when one is available,
and it unlocks automatic differentiation of the forward model.

NumPy remains the default everywhere: nothing changes for existing users
unless a backend is explicitly selected here or via the ``GEODEF_BACKEND``
environment variable.

Computations default to float64. GPU hardware is typically much faster in
float32, so a lower-precision mode is available as an explicit opt-in;
the kernels are sensitive near the fault, so expect reduced accuracy there.

Usage::

    import geodef
    geodef.backend.set_backend("jax")        # pip install geodef[jax]
    geodef.backend.set_precision("float32")  # optional, GPU-friendly
"""

from __future__ import annotations

import dataclasses
import logging
import os
from types import ModuleType

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

_VALID_BACKENDS = ("numpy", "jax")
_VALID_PRECISIONS = ("float64", "float32")


@dataclasses.dataclass
class _BackendConfig:
    """Internal mutable state for the backend module."""

    name: str = "numpy"
    precision: str = "float64"


_config = _BackendConfig()


# ====================================================================
# Module-level configuration API
# ====================================================================


def set_backend(name: str) -> None:
    """Select the array backend used by the compute kernels.

    Args:
        name: Backend name, ``'numpy'`` (default) or ``'jax'``.

    Raises:
        ValueError: If ``name`` is not a known backend.
        ImportError: If ``'jax'`` is requested but JAX is not installed.
    """
    if name not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown backend {name!r}; expected one of {_VALID_BACKENDS}."
        )
    if name == "jax":
        try:
            import jax  # noqa: F401
        except ImportError as err:
            raise ImportError(
                "The 'jax' backend requires JAX. "
                "Install it with: pip install geodef[jax]"
            ) from err
    _config.name = name
    _apply_jax_precision()


def get_backend() -> str:
    """Return the name of the active backend.

    Returns:
        ``'numpy'`` or ``'jax'``.
    """
    return _config.name


def namespace() -> ModuleType:
    """Return the array namespace of the active backend.

    Kernels call this at compute time (not import time) so that backend
    switches take effect immediately.

    Returns:
        ``numpy`` or ``jax.numpy``.
    """
    if _config.name == "jax":
        import jax.numpy as jnp

        return jnp
    return np


def set_precision(precision: str) -> None:
    """Set the floating-point precision for backend computations.

    Args:
        precision: ``'float64'`` (default) or ``'float32'``. Selecting
            float32 trades accuracy near the fault for GPU throughput.

    Raises:
        ValueError: If ``precision`` is not a supported precision.
    """
    if precision not in _VALID_PRECISIONS:
        raise ValueError(
            f"Unknown precision {precision!r}; expected one of {_VALID_PRECISIONS}."
        )
    _config.precision = precision
    _apply_jax_precision()


def get_precision() -> str:
    """Return the active floating-point precision.

    Returns:
        ``'float64'`` or ``'float32'``.
    """
    return _config.precision


def default_dtype() -> np.dtype:
    """Return the default floating-point dtype for the active precision.

    Returns:
        ``np.dtype('float64')`` or ``np.dtype('float32')``.
    """
    return np.dtype(_config.precision)


def to_numpy(array: npt.ArrayLike) -> np.ndarray:
    """Convert a backend array to a NumPy array at a module boundary.

    Args:
        array: Array from any active backend (NumPy or JAX).

    Returns:
        The equivalent ``np.ndarray`` (zero-copy where possible).
    """
    return np.asarray(array)


def _apply_jax_precision() -> None:
    """Sync JAX's x64 flag with the configured precision."""
    if _config.name != "jax":
        return
    import jax

    jax.config.update("jax_enable_x64", _config.precision == "float64")


_env_backend = os.environ.get("GEODEF_BACKEND")
if _env_backend:
    try:
        set_backend(_env_backend)
    except (ValueError, ImportError) as err:
        logger.warning("Ignoring GEODEF_BACKEND=%r: %s", _env_backend, err)
