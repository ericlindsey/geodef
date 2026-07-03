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


def _planar_patches(theta, n_length: int, n_width: int):
    """Traced mirror of ``Fault.planar``: patch centers from geometry theta.

    Reproduces the same patch ordering (down-dip index slowest,
    along-strike fastest) and the same center/depth formulas, but in
    local Cartesian coordinates and traceable in ``theta``.

    Args:
        theta: ``[e0, n0, depth, strike, dip, length, width]`` of the
            fault centroid.
        n_length: Number of patches along strike (static).
        n_width: Number of patches down dip (static).

    Returns:
        Tuple ``(e_c, n_c, depth_c, strike, dip, patch_L, patch_W)`` with
        per-patch center arrays of length ``n_length * n_width``.
    """
    theta_a = xp.asarray(theta)
    e0, n0, depth, strike, dip, length, width = (theta_a[i] for i in range(7))
    patch_L = length / n_length
    patch_W = width / n_width

    sin_str = xp.sin(xp.radians(strike))
    cos_str = xp.cos(xp.radians(strike))
    sin_dip = xp.sin(xp.radians(dip))
    cos_dip = xp.cos(xp.radians(dip))

    # Offset from center to top-left corner of the fault
    fault_e0 = -0.5 * length * sin_str - 0.5 * width * cos_dip * cos_str
    fault_n0 = -0.5 * length * cos_str + 0.5 * width * cos_dip * sin_str
    fault_u0 = -0.5 * width * sin_dip

    jj_grid, ii_grid = np.meshgrid(
        np.arange(n_width), np.arange(n_length), indexing="ij"
    )
    ii = ii_grid.ravel()
    jj = jj_grid.ravel()

    e_c = (
        e0
        + fault_e0
        + (ii + 0.5) * patch_L * sin_str
        + (jj + 0.5) * patch_W * cos_dip * cos_str
    )
    n_c = (
        n0
        + fault_n0
        + (ii + 0.5) * patch_L * cos_str
        - (jj + 0.5) * patch_W * cos_dip * sin_str
    )
    depth_c = depth - (fault_u0 + (jj + 0.5) * patch_W * sin_dip)

    return e_c, n_c, depth_c, strike, dip, patch_L, patch_W


def rect_greens(
    theta: npt.ArrayLike,
    e_obs: np.ndarray,
    n_obs: np.ndarray,
    n_length: int = 1,
    n_width: int = 1,
    nu: float = 0.25,
) -> np.ndarray:
    """Displacement Green's matrix G(theta) for a discretized planar fault.

    Traceable in ``theta``, so ``jax.jacfwd(rect_greens)`` gives the
    sensitivity of every Green's coefficient to the fault geometry. The
    layout matches ``greens.displacement_greens``: each set of 3 rows is
    [E, N, U] for one observation point; columns ``[:N]`` are strike-slip
    and ``[N:]`` dip-slip, ordered as in ``Fault.planar``.

    Args:
        theta: ``[e0, n0, depth, strike, dip, length, width]`` of the
            fault centroid, in local Cartesian coordinates.
        e_obs: Observation eastings (nobs,).
        n_obs: Observation northings (nobs,).
        n_length: Number of patches along strike (static).
        n_width: Number of patches down dip (static).
        nu: Poisson's ratio.

    Returns:
        G of shape (3*nobs, 2*N) with N = n_length*n_width patches.
    """
    e_c, n_c, depth_c, strike, dip, patch_L, patch_W = _planar_patches(
        theta, n_length, n_width
    )
    nobs = len(np.atleast_1d(e_obs))
    npatch = n_length * n_width

    de = e_obs[None, :] - e_c[:, None]
    dn = n_obs[None, :] - n_c[:, None]
    geom = (de, dn, depth_c[:, None], strike, dip, patch_L, patch_W)

    blocks = []
    for rake in (0.0, 90.0):
        ue, un, uz = okada85.displacement(*geom, rake, 1.0, 0.0, nu)
        comp = xp.stack([ue, un, uz], axis=0)  # (3, npatch, nobs)
        blocks.append(xp.transpose(comp, (2, 0, 1)).reshape(3 * nobs, npatch))
    return xp.concatenate(blocks, axis=1)


def tri_greens(
    vertices: npt.ArrayLike,
    obs: np.ndarray,
    nu: float = 0.25,
) -> np.ndarray:
    """Displacement Green's matrix G(vertices) for a triangular mesh.

    Traceable in the vertex coordinates. Column and row layout matches
    ``greens.tri_displacement_greens``: each set of 3 rows is [E, N, U]
    for one observation point; columns ``[:ntri]`` are strike-slip and
    ``[ntri:]`` dip-slip.

    The triangles are evaluated in a Python loop, so this traces (and
    differentiates) eagerly but does not yet support ``jax.jit``/``vmap``
    over the mesh axis.

    Args:
        vertices: Mesh vertex coordinates, shape (ntri, 3, 3), z <= 0.
        obs: Observation coordinates, shape (nobs, 3), z <= 0.
        nu: Poisson's ratio.

    Returns:
        G of shape (3*nobs, 2*ntri).
    """
    vertices_a = xp.asarray(vertices)
    ntri = int(np.asarray(np.shape(vertices))[0])
    ss = xp.asarray([1.0, 0.0, 0.0])
    ds = xp.asarray([0.0, 1.0, 0.0])
    cols_ss = [
        xp.reshape(tri.TDdispHS(obs, vertices_a[i], ss, nu), (-1,)) for i in range(ntri)
    ]
    cols_ds = [
        xp.reshape(tri.TDdispHS(obs, vertices_a[i], ds, nu), (-1,)) for i in range(ntri)
    ]
    return xp.stack(cols_ss + cols_ds, axis=1)


def los_project(G: npt.ArrayLike, look: npt.ArrayLike) -> np.ndarray:
    """Project a displacement Green's matrix onto per-point look vectors.

    Maps a (3*nobs, ncols) matrix with interleaved [E, N, U] rows to the
    (nobs, ncols) line-of-sight matrix ``los = look_e*ue + look_n*un +
    look_u*uz``, matching ``InSAR.project``. Traceable, so it composes
    with :func:`rect_greens` / :func:`tri_greens` under autodiff.

    Args:
        G: Displacement Green's matrix, shape (3*nobs, ncols).
        look: Unit look vectors, shape (nobs, 3), columns [E, N, U].

    Returns:
        Projected matrix of shape (nobs, ncols).
    """
    G_a = xp.asarray(G)
    look_a = xp.asarray(look)
    ncols = G_a.shape[1]
    return xp.einsum("nc,nck->nk", look_a, G_a.reshape(-1, 3, ncols))


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
