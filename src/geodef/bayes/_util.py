"""Shared helpers for the geodef.bayes package.

Private submodule of :mod:`geodef.bayes`: the JAX requirement guard, the
prior-mode vocabularies, and the positivity slip transform shared by the
collapsed and joint-slip posteriors.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from geodef import backend

_VALID_MODES = ("hierarchical", "weak", "profiled")
_VALID_SLIP_MODES = ("hierarchical", "weak", "fixed")


def _require_jax() -> Any:
    """Return the jax module, or raise if the JAX backend is not active."""
    if backend.get_backend() != "jax":
        raise RuntimeError(
            "geodef.bayes requires the JAX backend; "
            "call geodef.backend.set_backend('jax') first."
        )
    import jax

    return jax


def _slip_transform(
    z: Any,
    mu0: Any,
    L0: Any,
    mask: Any,
    sigma_ref: float,
    logJ_affine: float,
) -> tuple:
    """Whitened-softplus slip map ``z -> (m, logJ)`` (traceable).

    Pushes a standard-normal-scaled ``z`` through the affine whitening
    ``v = mu0 + sigma_ref * L0^-T z`` (``L0`` the lower-Cholesky factor of
    a fixed reference precision), then a per-component softplus where
    ``mask`` selects a positivity constraint, returning the slip vector
    ``m`` and the log-Jacobian of the whole map. Shared by
    :class:`SlipPosterior` and :class:`RectPosterior`'s positivity path.

    Args:
        z: Whitened slip coordinates, shape (n_slip,).
        mu0: Reference ridge slip (affine offset), shape (n_slip,).
        L0: Lower-triangular Cholesky factor of the reference precision.
        mask: Bool array, True where the component is non-negative.
        sigma_ref: Reference noise scale used to build the affine map.
        logJ_affine: Constant log-Jacobian of the affine part,
            ``n_slip * log(sigma_ref) - sum(log diag L0)``.

    Returns:
        Tuple ``(m, logJ)`` of the slip vector and the scalar
        log-Jacobian of the ``z -> m`` map.
    """
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import solve_triangular

    v = jnp.asarray(mu0) + sigma_ref * solve_triangular(
        jnp.asarray(L0).T, z, lower=False
    )
    mask = jnp.asarray(mask)
    m = jnp.where(mask, jax.nn.softplus(v), v)
    logJ = jnp.sum(jnp.where(mask, jax.nn.log_sigmoid(v), 0.0)) + logJ_affine
    return m, logJ


def _parse_positive(
    positive: str | npt.ArrayLike | None,
    components: str,
    n_patches: int,
    n_params: int,
) -> np.ndarray:
    """Resolve the ``positive`` argument of :class:`SlipPosterior` to a mask.

    Args:
        positive: None, 'strike', 'dip', 'both', or a bool array.
        components: The posterior's sampled slip components.
        n_patches: Number of fault patches (one block's width).
        n_params: Total number of sampled slip components.

    Returns:
        Bool array, shape (n_params,), True where positivity-constrained.

    Raises:
        ValueError: If ``positive`` names a block absent from
            ``components``, is an unrecognized string, or is an array
            of the wrong length.
    """
    if positive is None:
        return np.zeros(n_params, dtype=bool)
    if isinstance(positive, str):
        if positive == "both":
            return np.ones(n_params, dtype=bool)
        if positive == "strike":
            if components not in ("both", "strike"):
                raise ValueError(
                    "positive='strike' requires a sampled strike-slip block "
                    f"(components={components!r})"
                )
            mask = np.zeros(n_params, dtype=bool)
            if components == "both":
                mask[:n_patches] = True
            else:
                mask[:] = True
            return mask
        if positive == "dip":
            if components not in ("both", "dip"):
                raise ValueError(
                    "positive='dip' requires a sampled dip-slip block "
                    f"(components={components!r})"
                )
            mask = np.zeros(n_params, dtype=bool)
            if components == "both":
                mask[n_patches:] = True
            else:
                mask[:] = True
            return mask
        raise ValueError(
            "positive must be None, 'strike', 'dip', 'both', or a bool array, "
            f"got {positive!r}"
        )
    positive_mask = np.asarray(positive, dtype=bool)
    if positive_mask.shape != (n_params,):
        raise ValueError(
            f"positive array must have shape ({n_params},), got {positive_mask.shape}"
        )
    return positive_mask
