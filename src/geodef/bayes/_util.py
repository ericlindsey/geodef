"""Shared helpers for the geodef.bayes package.

Private submodule of :mod:`geodef.bayes`: the JAX requirement guard, the
prior-mode vocabularies, and the positivity slip transform shared by the
collapsed and joint-slip posteriors.
"""

from __future__ import annotations

from typing import Any

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
