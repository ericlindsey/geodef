"""Differentiable forward models: gradients of predicted displacements.

Exposes the rectangular (okada85) and triangular (Nikkhoo & Walter)
half-space forward models as functions that JAX can differentiate, plus
Jacobian helpers built on forward-mode autodiff (``jax.jacfwd``), which
suits the many-outputs / few-parameters shape of geometry inversion and
avoids the reverse-mode NaN-through-``where`` pitfall in the triangular
kernel's configuration selection.

Differentiation variables follow the engines' native parameterizations:

- Rectangles: ``theta = [e0, n0, depth, strike, dip, length, width]``
  (fault-centroid position, orientation in degrees, and size).
- Triangles: the three vertex coordinates. Gradients with respect to
  derived parameters (trace position, dip of a planar mesh, ...) follow
  by composing a ``theta -> vertices`` builder with these functions and
  letting JAX chain through it.

Requires the JAX backend::

    import geodef
    geodef.backend.set_backend("jax")
    d, d_dtheta, d_dslip = geodef.gradients.rect_displacement_jacobian(
        theta, slip, e_obs, n_obs
    )
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from geodef import backend, okada85, tri
from geodef.backend import xp


def _require_jax():
    """Return the jax module, or raise if the JAX backend is not active."""
    if backend.get_backend() != "jax":
        raise RuntimeError(
            "geodef.gradients requires the JAX backend; "
            "call geodef.backend.set_backend('jax') first."
        )
    import jax

    return jax


def rect_displacement(
    theta: npt.ArrayLike,
    slip: npt.ArrayLike,
    e_obs: np.ndarray,
    n_obs: np.ndarray,
    nu: float = 0.25,
) -> np.ndarray:
    """Surface displacement from one rectangular fault, traceable in theta.

    Args:
        theta: Geometry parameters ``[e0, n0, depth, strike, dip, length,
            width]`` — centroid easting/northing offset, centroid depth,
            strike and dip in degrees, and fault dimensions.
        slip: Slip components ``[strike_slip, dip_slip, opening]``.
        e_obs: Observation eastings (nobs,).
        n_obs: Observation northings (nobs,).
        nu: Poisson's ratio.

    Returns:
        Displacements of shape (nobs, 3), columns [E, N, U].
    """
    theta_a = xp.asarray(theta)
    slip_a = xp.asarray(slip)
    e0, n0, depth, strike, dip, length, width = (theta_a[i] for i in range(7))
    de = e_obs - e0
    dn = n_obs - n0

    geom = (de, dn, depth, strike, dip, length, width)
    u_ss = xp.stack(okada85.displacement(*geom, 0.0, 1.0, 0.0, nu), axis=-1)
    u_ds = xp.stack(okada85.displacement(*geom, 90.0, 1.0, 0.0, nu), axis=-1)
    u_op = xp.stack(okada85.displacement(*geom, 0.0, 0.0, 1.0, nu), axis=-1)
    return slip_a[0] * u_ss + slip_a[1] * u_ds + slip_a[2] * u_op


def tri_displacement(
    vertices: npt.ArrayLike,
    slip: npt.ArrayLike,
    obs: np.ndarray,
    nu: float = 0.25,
) -> np.ndarray:
    """Half-space displacement from one triangular dislocation.

    Traceable in the vertex coordinates — the engine's native geometry
    parameterization.

    Args:
        vertices: Triangle vertex coordinates, shape (3, 3); each row is
            [x, y, z] with z <= 0.
        slip: Slip components ``[strike_slip, dip_slip, tensile]``.
        obs: Observation coordinates, shape (nobs, 3), z <= 0.
        nu: Poisson's ratio.

    Returns:
        Displacements of shape (nobs, 3), columns [E, N, U].
    """
    return tri.TDdispHS(obs, xp.asarray(vertices), xp.asarray(slip), nu)


def rect_displacement_jacobian(
    theta: npt.ArrayLike,
    slip: npt.ArrayLike,
    e_obs: np.ndarray,
    n_obs: np.ndarray,
    nu: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward model and its Jacobians for a rectangular fault.

    Args:
        theta: Geometry parameters, see :func:`rect_displacement`.
        slip: Slip components ``[strike_slip, dip_slip, opening]``.
        e_obs: Observation eastings (nobs,).
        n_obs: Observation northings (nobs,).
        nu: Poisson's ratio.

    Returns:
        Tuple ``(d, d_dtheta, d_dslip)``: predicted displacements
        (nobs, 3), geometry Jacobian (nobs, 3, 7), and slip Jacobian
        (nobs, 3, 3).

    Raises:
        RuntimeError: If the JAX backend is not active.
    """
    jax = _require_jax()
    theta = xp.asarray(theta, dtype=float)
    slip = xp.asarray(slip, dtype=float)
    d = rect_displacement(theta, slip, e_obs, n_obs, nu)
    d_dtheta, d_dslip = jax.jacfwd(rect_displacement, argnums=(0, 1))(
        theta, slip, e_obs, n_obs, nu
    )
    return d, d_dtheta, d_dslip


def tri_displacement_jacobian(
    vertices: npt.ArrayLike,
    slip: npt.ArrayLike,
    obs: np.ndarray,
    nu: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward model and its Jacobians for a triangular dislocation.

    Args:
        vertices: Triangle vertex coordinates, shape (3, 3), z <= 0.
        slip: Slip components ``[strike_slip, dip_slip, tensile]``.
        obs: Observation coordinates, shape (nobs, 3), z <= 0.
        nu: Poisson's ratio.

    Returns:
        Tuple ``(d, d_dvertices, d_dslip)``: predicted displacements
        (nobs, 3), vertex Jacobian (nobs, 3, 3, 3) with the trailing axes
        indexing (vertex, coordinate), and slip Jacobian (nobs, 3, 3).

    Raises:
        RuntimeError: If the JAX backend is not active.
    """
    jax = _require_jax()
    vertices = xp.asarray(vertices, dtype=float)
    slip = xp.asarray(slip, dtype=float)
    d = tri_displacement(vertices, slip, obs, nu)
    d_dvertices, d_dslip = jax.jacfwd(tri_displacement, argnums=(0, 1))(
        vertices, slip, obs, nu
    )
    return d, d_dvertices, d_dslip
