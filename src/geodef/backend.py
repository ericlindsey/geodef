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
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any

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


class _NamespaceProxy:
    """Array-namespace handle that re-resolves the backend at every access.

    Kernels import this once as ``xp`` and write ``xp.cos(...)`` etc.; each
    attribute access looks up the active backend, so ``set_backend`` takes
    effect immediately without re-importing the kernels.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(namespace(), name)


xp = _NamespaceProxy()


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


def masked_eval(
    func: Callable[..., tuple],
    mask: npt.NDArray[np.bool_],
    args: Sequence[npt.ArrayLike],
    n_out: int,
    fill: float = np.nan,
) -> tuple:
    """Evaluate a vectorized function only on the lanes where ``mask`` is True.

    The dislocation kernels select between artefact-free formula
    configurations per observation point. On the NumPy backend the True
    lanes are gathered, ``func`` runs once on the compressed arrays, and
    the results are scattered back, so no work is spent on masked-out
    lanes. On the JAX backend ``func`` runs on the full arrays and the
    result is selected with ``where``, because data-dependent shapes
    cannot be traced or JIT-compiled.

    Args:
        func: Vectorized callable returning a tuple of ``n_out`` arrays
            whose trailing axis matches its inputs'.
        mask: Boolean mask over the trailing axis of each argument.
        args: Arrays passed to ``func``, each maskable along its trailing
            axis.
        n_out: Number of arrays ``func`` returns.
        fill: Value assigned to lanes where ``mask`` is False.

    Returns:
        Tuple of ``n_out`` arrays shaped like ``mask``, holding ``func``'s
        results on the True lanes and ``fill`` elsewhere.
    """
    if _config.name == "jax":
        xp = namespace()
        outs = func(*(xp.asarray(a) for a in args))
        return tuple(xp.where(mask, out, fill) for out in outs)
    mask = np.asarray(mask, dtype=bool)
    full = tuple(np.full(mask.shape, fill) for _ in range(n_out))
    if mask.any():
        outs = func(*(np.asarray(a)[..., mask] for a in args))
        for dst, src in zip(full, outs):
            dst[mask] = src
    return full


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
