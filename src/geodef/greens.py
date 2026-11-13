"""Green's matrix assembly, dataset projection, and Laplacian operators.

Three levels, from most to least convenient:

- :func:`matrix` builds ``G`` for a ``Fault`` and its datasets, in each
  dataset's observation layout.
- ``Fault.greens_matrix`` builds the raw three-component ``G`` at geographic
  points.
- :func:`displacement_greens`, :func:`strain_greens`, and their ``tri_``
  counterparts work purely in local Cartesian meters and call the kernels
  in :mod:`geodef.kernels`. Geography stops at ``Fault``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.sparse

from geodef import backend
from geodef import cache as _cache
from geodef.kernels import okada85, tri

if TYPE_CHECKING:
    from geodef.data import DataSet
    from geodef.fault import Fault

logger = logging.getLogger(__name__)


# ======================================================================
# Green's matrix assembly for rectangular patches (local Cartesian)
# ======================================================================


def _displacement_batch(
    e: np.ndarray,
    n: np.ndarray,
    depth: np.ndarray,
    strike: np.ndarray,
    dip: np.ndarray,
    L: np.ndarray,
    W: np.ndarray,
    rake: float,
    nu: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate okada85.displacement for all patches at once.

    The kernel is pure elementwise math, so it broadcasts over a leading
    patch axis: ``e``/``n`` are (npatch, nobs) patch-relative coordinates
    and the patch parameters are (npatch,) columns.

    Returns:
        Tuple of (ue, un, uz), each of shape (npatch, nobs).
    """
    return okada85.displacement(
        e,
        n,
        depth[:, None],
        strike[:, None],
        dip[:, None],
        L[:, None],
        W[:, None],
        rake,
        1.0,
        0.0,
        nu,
    )


def _strain_surface_batch(
    e: np.ndarray,
    n: np.ndarray,
    depth: np.ndarray,
    strike: np.ndarray,
    dip: np.ndarray,
    L: np.ndarray,
    W: np.ndarray,
    rake: float,
    nu: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate okada85.strain for all patches at once.

    Same broadcasting scheme as ``_displacement_batch``: ``e``/``n`` are
    (npatch, nobs) and patch parameters are (npatch,) columns.

    Returns:
        Tuple of (unn, une, uen, uee), each of shape (npatch, nobs).
    """
    return okada85.strain(
        e,
        n,
        depth[:, None],
        strike[:, None],
        dip[:, None],
        L[:, None],
        W[:, None],
        rake,
        1.0,
        0.0,
        nu,
    )


def _strain_depth_batch(
    e: np.ndarray,
    n: np.ndarray,
    z: np.ndarray,
    depth: np.ndarray,
    strike: np.ndarray,
    dip: np.ndarray,
    L: np.ndarray,
    W: np.ndarray,
    disl1: float,
    disl2: float,
    disl3: float,
    nu: float,
) -> np.ndarray:
    """Evaluate okada92 strain at depth for all patches at once.

    Mirrors ``okada92.okada92`` (centroid transform, DC3D, rotation to
    geographic) without the eager singular-point checks so the whole
    batch JIT-compiles; singular lanes come back NaN, matching
    ``allow_singular=True``. Uses unit shear modulus, as the loop path
    does.

    Returns:
        Geographic strain tensors of shape (npatch, nobs, 3, 3).
    """
    from geodef.kernels import okada92

    depth_c = depth[:, None]
    strike_c = strike[:, None]
    dip_c = dip[:, None]
    L_c = L[:, None]
    W_c = W[:, None]

    lam = 2 * nu / (1 - 2 * nu)  # G = 1
    alpha = (lam + 1.0) / (lam + 2.0)

    cs = backend.xp.cos(backend.xp.radians(strike_c))
    ss = backend.xp.sin(backend.xp.radians(strike_c))
    cd = backend.xp.cos(backend.xp.radians(dip_c))
    sd = backend.xp.sin(backend.xp.radians(dip_c))

    d_top = depth_c + sd * W_c / 2
    ec = e + cs * cd * W_c / 2
    nc = n - ss * cd * W_c / 2
    x_dc3d = cs * nc + ss * ec + L_c / 2
    y_dc3d = ss * nc - cs * ec + cd * W_c

    displacement, strain, _ = okada92.DC3D(
        alpha,
        x_dc3d,
        y_dc3d,
        z,
        d_top,
        dip_c,
        0.0,
        L_c,
        0.0,
        W_c,
        disl1,
        disl2,
        disl3,
    )
    _, strain_geo = okada92._rotate_to_geographic(displacement, strain, ss, cs)
    return strain_geo


_jitted_batch_kernels: dict = {}


def _get_batch_kernel(name: str):
    """Return a batched assembly kernel, JIT-compiled on the JAX backend.

    Args:
        name: One of ``'displacement'``, ``'strain_surface'``,
            ``'strain_depth'``.
    """
    kernels: dict[str, tuple[Callable, tuple[int, ...]]] = {
        "displacement": (_displacement_batch, ()),
        "strain_surface": (_strain_surface_batch, ()),
        # slip components select code blocks inside DC3D, so they must be
        # static under JIT (two compilations: strike-slip and dip-slip)
        "strain_depth": (_strain_depth_batch, (8, 9, 10)),
    }
    func, static_argnums = kernels[name]
    if backend.get_backend() != "jax":
        return func
    if name not in _jitted_batch_kernels:
        import jax

        _jitted_batch_kernels[name] = jax.jit(func, static_argnums=static_argnums)
    return _jitted_batch_kernels[name]


# The largest (patches x observations) block evaluated at once on NumPy.
_BLOCK_ELEMENTS = 250_000


def _patch_blocks(n_patches: int, n_obs: int) -> list[slice]:
    """Split the patches into blocks whose kernel arrays stay small.

    The kernels broadcast over (patches, observations). On NumPy that
    array is bounded so a large fault does not exhaust memory; on JAX the
    whole problem is one block so the kernel compiles once.
    """
    if backend.get_backend() == "jax":
        return [slice(0, n_patches)]
    size = max(1, _BLOCK_ELEMENTS // max(n_obs, 1))
    return [slice(i, min(i + size, n_patches)) for i in range(0, n_patches, size)]


def _rectangles(
    obs_east: npt.ArrayLike,
    obs_north: npt.ArrayLike,
    east: npt.ArrayLike,
    north: npt.ArrayLike,
    depth: npt.ArrayLike,
    strike: npt.ArrayLike,
    dip: npt.ArrayLike,
    length: npt.ArrayLike,
    width: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    """Validate rectangular-patch inputs and return them as float arrays."""
    obs = [np.atleast_1d(np.asarray(a, dtype=float)) for a in (obs_east, obs_north)]
    if obs[0].shape != obs[1].shape or obs[0].ndim != 1:
        raise ValueError("obs_east and obs_north must be 1-D arrays of equal length")
    patches = [
        np.atleast_1d(np.asarray(a, dtype=float))
        for a in (east, north, depth, strike, dip, length, width)
    ]
    if len({a.shape for a in patches}) != 1 or patches[0].ndim != 1:
        raise ValueError(
            "east, north, depth, strike, dip, length, and width must be 1-D "
            "arrays of equal length"
        )
    return obs[0], obs[1], patches[0], patches[1], tuple(patches[2:])


def displacement_greens(
    *,
    obs_east: npt.ArrayLike,
    obs_north: npt.ArrayLike,
    east: npt.ArrayLike,
    north: npt.ArrayLike,
    depth: npt.ArrayLike,
    strike: npt.ArrayLike,
    dip: npt.ArrayLike,
    length: npt.ArrayLike,
    width: npt.ArrayLike,
    nu: float = 0.25,
) -> np.ndarray:
    """Build the surface-displacement Green's matrix for rectangular patches.

    Everything is in one local Cartesian frame, in meters and degrees. Each
    set of 3 rows is the ``[east, north, up]`` displacement at one
    observation point; the first ``N`` columns are unit strike-slip on each
    patch, the next ``N`` unit dip-slip.

    Args:
        obs_east: Observation East coordinates, shape (M,).
        obs_north: Observation North coordinates, shape (M,).
        east: Patch-center East coordinates, shape (N,).
        north: Patch-center North coordinates, shape (N,).
        depth: Patch-center depths, positive down, shape (N,).
        strike: Patch strikes in degrees clockwise from North, shape (N,).
        dip: Patch dips in degrees from horizontal, shape (N,).
        length: Along-strike patch lengths, shape (N,).
        width: Down-dip patch widths, shape (N,).
        nu: Poisson's ratio.

    Returns:
        Green's matrix of shape (3*M, 2*N).
    """
    obs_e, obs_n, east, north, shape = _rectangles(
        obs_east, obs_north, east, north, depth, strike, dip, length, width
    )
    n_obs, n_patches = obs_e.size, east.size
    kernel = _get_batch_kernel("displacement")
    G = np.empty((3 * n_obs, 2 * n_patches))
    for block in _patch_blocks(n_patches, n_obs):
        e = obs_e[None, :] - east[block, None]
        n = obs_n[None, :] - north[block, None]
        patch = [a[block] for a in shape]
        for first_column, rake in ((0, 0.0), (n_patches, 90.0)):
            columns = slice(first_column + block.start, first_column + block.stop)
            for row, component in enumerate(kernel(e, n, *patch, rake, nu)):
                G[row::3, columns] = backend.to_numpy(component).T
    return G


def strain_greens(
    *,
    obs_east: npt.ArrayLike,
    obs_north: npt.ArrayLike,
    east: npt.ArrayLike,
    north: npt.ArrayLike,
    depth: npt.ArrayLike,
    strike: npt.ArrayLike,
    dip: npt.ArrayLike,
    length: npt.ArrayLike,
    width: npt.ArrayLike,
    nu: float = 0.25,
    obs_depth: npt.ArrayLike | None = None,
) -> np.ndarray:
    """Build the horizontal-strain Green's matrix for rectangular patches.

    Each set of 4 rows is ``[NN, NE, EN, EE]`` (displacement gradients) at
    one observation point. Columns are blocked as in
    :func:`displacement_greens`.

    Args:
        obs_east: Observation East coordinates, shape (M,).
        obs_north: Observation North coordinates, shape (M,).
        east: Patch-center East coordinates, shape (N,).
        north: Patch-center North coordinates, shape (N,).
        depth: Patch-center depths, positive down, shape (N,).
        strike: Patch strikes in degrees clockwise from North, shape (N,).
        dip: Patch dips in degrees from horizontal, shape (N,).
        length: Along-strike patch lengths, shape (N,).
        width: Down-dip patch widths, shape (N,).
        nu: Poisson's ratio.
        obs_depth: Observation depths, positive down, shape (M,). ``None``
            evaluates at the surface with Okada (1985); otherwise Okada
            (1992) is used for internal strain.

    Returns:
        Green's matrix of shape (4*M, 2*N).
    """
    obs_e, obs_n, east, north, shape = _rectangles(
        obs_east, obs_north, east, north, depth, strike, dip, length, width
    )
    n_obs, n_patches = obs_e.size, east.size
    G = np.empty((4 * n_obs, 2 * n_patches))
    z_obs = None if obs_depth is None else -np.asarray(obs_depth, dtype=float)
    kernel = _get_batch_kernel("strain_surface" if z_obs is None else "strain_depth")
    # (row, column) of NN, NE, EN, EE in the 3x3 displacement-gradient tensor
    gradient_index = ((1, 1), (1, 0), (0, 1), (0, 0))
    for block in _patch_blocks(n_patches, n_obs):
        e = obs_e[None, :] - east[block, None]
        n = obs_n[None, :] - north[block, None]
        patch = [a[block] for a in shape]
        for first_column, rake, unit_slip in (
            (0, 0.0, (1.0, 0.0, 0.0)),
            (n_patches, 90.0, (0.0, 1.0, 0.0)),
        ):
            columns = slice(first_column + block.start, first_column + block.stop)
            if z_obs is None:
                components = kernel(e, n, *patch, rake, nu)
            else:
                tensor = backend.to_numpy(kernel(e, n, z_obs, *patch, *unit_slip, nu))
                components = [tensor[:, :, i, j] for i, j in gradient_index]
            for row, component in enumerate(components):
                G[row::4, columns] = backend.to_numpy(component).T
    return G


# ======================================================================
# Green's matrix assembly for triangular patches (local Cartesian)
# ======================================================================


def _tri_greens(
    kernel: Callable[..., np.ndarray],
    obs_east: npt.ArrayLike,
    obs_north: npt.ArrayLike,
    obs_depth: npt.ArrayLike | None,
    vertices: npt.ArrayLike,
    nu: float,
) -> np.ndarray:
    """Assemble a triangular Green's matrix from a per-triangle kernel."""
    obs_e = np.atleast_1d(np.asarray(obs_east, dtype=float))
    obs_n = np.atleast_1d(np.asarray(obs_north, dtype=float))
    obs_up = np.zeros_like(obs_e) if obs_depth is None else -np.asarray(obs_depth)
    obs = np.column_stack([obs_e, obs_n, obs_up])
    triangles = np.asarray(vertices, dtype=float)
    blocks = []
    for unit_slip in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
        # each kernel call returns (M, n_components); one column per triangle
        columns = [
            kernel(obs, triangle, unit_slip, nu).ravel() for triangle in triangles
        ]
        blocks.append(np.column_stack(columns))
    return np.hstack(blocks)


def tri_displacement_greens(
    *,
    obs_east: npt.ArrayLike,
    obs_north: npt.ArrayLike,
    vertices: npt.ArrayLike,
    nu: float = 0.25,
) -> np.ndarray:
    """Build the surface-displacement Green's matrix for triangular patches.

    Args:
        obs_east: Observation East coordinates, shape (M,).
        obs_north: Observation North coordinates, shape (M,).
        vertices: Triangle corners as ``[east, north, up]`` in the same
            frame, shape (N, 3, 3).
        nu: Poisson's ratio.

    Returns:
        Green's matrix of shape (3*M, 2*N): rows are ``[east, north, up]``
        per observation point; columns are strike-slip then dip-slip.
    """
    return _tri_greens(tri.TDdispHS, obs_east, obs_north, None, vertices, nu)


def tri_strain_greens(
    *,
    obs_east: npt.ArrayLike,
    obs_north: npt.ArrayLike,
    vertices: npt.ArrayLike,
    nu: float = 0.25,
    obs_depth: npt.ArrayLike | None = None,
) -> np.ndarray:
    """Build the strain Green's matrix for triangular patches.

    Args:
        obs_east: Observation East coordinates, shape (M,).
        obs_north: Observation North coordinates, shape (M,).
        vertices: Triangle corners as ``[east, north, up]`` in the same
            frame, shape (N, 3, 3).
        nu: Poisson's ratio.
        obs_depth: Observation depths, positive down, shape (M,). ``None``
            evaluates at the surface.

    Returns:
        Green's matrix of shape (6*M, 2*N): rows are ``[xx, yy, zz, xy, xz,
        yz]`` per observation point; columns are strike-slip then dip-slip.
    """
    return _tri_greens(tri.TDstrainHS, obs_east, obs_north, obs_depth, vertices, nu)


# ======================================================================
# Polymorphic Green's matrix assembly (Fault + DataSet)
# ======================================================================


def _build_greens_key(fault: Fault, data: DataSet) -> dict:
    """Build the cache key dict for a fault + dataset combination."""
    from geodef.data import GNSS, InSAR

    key: dict = {
        **fault._cache_key(),
        "obs_lat": data.lat,
        "obs_lon": data.lon,
        "data_class": type(data).__name__,
        "greens_type": data.greens_type,
    }
    if isinstance(data, InSAR):
        key["look_e"] = data.look_e
        key["look_n"] = data.look_n
        key["look_u"] = data.look_u
    if isinstance(data, GNSS):
        key["components"] = data.components
    return key


def select_slip_columns(
    G_full: np.ndarray,
    n_patches: int,
    components: str,
    rake: float | None = None,
    fault_strike: np.ndarray | None = None,
    slip_azimuth: float | None = None,
    plate_rake: float | np.ndarray | None = None,
) -> np.ndarray:
    """Project a two-component Green's matrix onto the requested slip basis.

    The full matrix has ``2*N`` columns blocked as ``[strike-slip | dip-slip]``.
    This reduces them to the columns for the requested component(s), matching
    the semantics used by :func:`geodef.invert`.

    Args:
        G_full: Full Green's (or stress-kernel) matrix, shape (M, 2*N).
        n_patches: Number of fault patches N.
        components: ``'both'`` (no reduction), ``'strike'``, ``'dip'``,
            ``'rake'`` (fixed rake, all patches), or ``'azimuth'`` (fixed
            geographic slip azimuth, per-patch local rake), or ``'plate'``
            (two components parallel/perpendicular to a per-patch plate rake).
        rake: Fixed rake angle in degrees, required for ``'rake'``.
        fault_strike: Per-patch strike angles in degrees from true north
            (``Fault.geographic_strike``), shape (N,), required for
            ``'azimuth'``.
        slip_azimuth: Geographic slip azimuth in degrees CW from North,
            required for ``'azimuth'``.
        plate_rake: Large-scale plate direction expressed as local rake in
            degrees, scalar or shape (N,), required for ``'plate'``.

    Returns:
        Reduced matrix: shape (M, 2*N) for ``'both'`` and ``'plate'``, else
        (M, N).

    Raises:
        ValueError: If required angles for the chosen basis are missing.
    """
    if components == "both":
        return G_full
    if components == "strike":
        return G_full[:, :n_patches]
    if components == "dip":
        return G_full[:, n_patches:]
    if components == "plate":
        if plate_rake is None:
            raise ValueError("components='plate' requires plate_rake")
        theta = np.deg2rad(np.broadcast_to(plate_rake, (n_patches,)))
        cosine = np.cos(theta)
        sine = np.sin(theta)
        strike = G_full[:, :n_patches]
        dip = G_full[:, n_patches:]
        parallel = strike * cosine + dip * sine
        perpendicular = -strike * sine + dip * cosine
        return np.hstack([parallel, perpendicular])
    if components == "rake":
        if rake is None:
            raise ValueError("components='rake' requires a rake angle in degrees")
        theta = np.deg2rad(rake)  # scalar
    elif components == "azimuth":
        if fault_strike is None or slip_azimuth is None:
            raise ValueError(
                "components='azimuth' requires fault_strike and slip_azimuth"
            )
        theta = np.deg2rad(slip_azimuth - fault_strike)  # shape (N,)
    else:
        raise ValueError(f"Unknown slip components {components!r}")
    return G_full[:, :n_patches] * np.cos(theta) + G_full[:, n_patches:] * np.sin(theta)


def matrix(
    fault: Fault,
    datasets: DataSet | list[DataSet],
    *,
    components: str = "both",
    rake: float | None = None,
    slip_azimuth: float | None = None,
    plate_rake: float | np.ndarray | None = None,
) -> np.ndarray:
    """Build a projected Green's matrix for one or more datasets.

    Computes the raw Green's matrix from the fault at each dataset's
    observation locations, then projects each column through
    ``data.project()`` to map into the dataset's observation space.

    Args:
        fault: A ``Fault`` instance.
        datasets: A single ``DataSet`` or a list of them.
        components: Slip basis for the returned columns: ``'both'``
            (default, ``2*N`` columns), ``'strike'``, ``'dip'``, ``'rake'``
            (fixed rake), ``'azimuth'`` (fixed geographic slip azimuth), or
            ``'plate'`` (plate-rake-parallel/perpendicular coordinates).
            The reduction uses the same semantics as
            :func:`geodef.solve`.
        rake: Fixed rake angle in degrees, required for ``components='rake'``.
        slip_azimuth: Geographic slip azimuth in degrees CW from North,
            required for ``components='azimuth'``.
        plate_rake: Large-scale direction as local rake, scalar or shape (N,),
            required for ``components='plate'``.

    Returns:
        Projected Green's matrix. For a single dataset with M observations
        and a fault with N patches: shape (M_obs, 2*N) for ``'both'``, else
        (M_obs, N). Columns for ``'both'`` are blocked ``[:N]`` strike-slip,
        ``[N:]`` dip-slip. For multiple datasets rows are vertically stacked.
    """
    from geodef.data import DataSet

    if isinstance(datasets, DataSet):
        datasets = [datasets]

    blocks = []
    for data in datasets:
        key = _build_greens_key(fault, data)
        G_proj = _cache.cached_compute(
            key,
            lambda: project(
                data,
                fault.greens_matrix(lon=data.lon, lat=data.lat, kind=data.greens_type),
            ),
        )
        blocks.append(G_proj)

    G_full = np.vstack(blocks)
    return select_slip_columns(
        G_full,
        fault.n_patches,
        components,
        rake,
        fault_strike=fault.geographic_strike,
        slip_azimuth=slip_azimuth,
        plate_rake=plate_rake,
    )


def stack_obs(datasets: DataSet | list[DataSet]) -> np.ndarray:
    """Concatenate observation vectors from one or more datasets.

    Args:
        datasets: A single ``DataSet`` or a list of them.

    Returns:
        1-D observation vector, shape (total_n_obs,).
    """
    from geodef.data import DataSet

    if isinstance(datasets, DataSet):
        datasets = [datasets]
    return np.concatenate([d.obs for d in datasets])


def stack_weights(datasets: DataSet | list[DataSet]) -> np.ndarray:
    """Build a block-diagonal inverse-covariance weight matrix.

    Each dataset contributes a diagonal block equal to the inverse of
    its covariance matrix. The result can be used as a weight matrix
    in least-squares inversion: ``W = stack_weights(datasets)``.

    Args:
        datasets: A single ``DataSet`` or a list of them.

    Returns:
        Block-diagonal weight matrix, shape (total_n_obs, total_n_obs).
    """
    from geodef.data import DataSet

    if isinstance(datasets, DataSet):
        datasets = [datasets]

    blocks = []
    for d in datasets:
        blocks.append(np.linalg.inv(d.covariance))

    return scipy.linalg.block_diag(*blocks)


def project(data: DataSet, G_raw: np.ndarray) -> np.ndarray:
    """Project a raw Green's matrix through a dataset's projection.

    For displacement Green's matrices (3 components per station), reshapes
    each column from (3*M,) to (M, 3), unpacks to (ue, un, uz), and calls
    ``data.project(ue, un, uz)``.

    Args:
        data: A ``DataSet`` instance.
        G_raw: Raw Green's matrix, shape (n_comp*M, 2*N).

    Returns:
        Projected Green's matrix, shape (data.n_obs, 2*N).
    """
    nrows, ncols = G_raw.shape
    nsta = data.n_stations

    if data.greens_type == "displacement":
        n_comp = 3
    elif data.greens_type == "strain":
        n_comp = 6
    else:
        raise ValueError(f"Unknown greens_type: {data.greens_type!r}")

    if nrows != n_comp * nsta:
        raise ValueError(
            f"G_raw has {nrows} rows but expected {n_comp}*{nsta} = {n_comp * nsta}"
        )

    G_proj = np.empty((data.n_obs, ncols))
    for col in range(ncols):
        components = G_raw[:, col].reshape(nsta, n_comp)
        G_proj[:, col] = data.project(*[components[:, c] for c in range(n_comp)])

    return G_proj


def laplacian(fault: Fault) -> np.ndarray:
    """Return the fault's patch Laplacian regularization matrix.

    Args:
        fault: Rectangular or triangular fault.

    Returns:
        Dense Laplacian matrix with shape ``(N, N)``.
    """
    return fault.laplacian


def resolution_matrix(G: np.ndarray) -> np.ndarray:
    """Compute resolution matrix R = pinv(G) @ G.

    Args:
        G: Green's matrix.

    Returns:
        Resolution matrix of shape (ncols, ncols).
    """
    return np.linalg.pinv(G) @ G


# ======================================================================
# Regularization operators
# ======================================================================


def build_laplacian_knn(
    coords: np.ndarray,
    k: int = 4,
) -> scipy.sparse.csc_matrix:
    """Build a distance-weighted graph Laplacian from K nearest neighbors.

    For each point, finds the *k* nearest neighbors and assigns weights
    inversely proportional to distance.  The resulting matrix is
    symmetrized (union of neighbor graphs) so that ``L = L^T``, and
    every row sums to zero.

    This is suitable for unstructured meshes (triangular patches or
    rectangular patches with non-uniform sizing) where no grid topology
    is available.

    Args:
        coords: Patch centroid coordinates, shape ``(n, 3)``.
            Typically ``[east, north, depth]`` in meters.
        k: Number of nearest neighbors per point.

    Returns:
        Sparse Laplacian matrix of shape ``(n, n)``.

    Raises:
        ValueError: If *k* < 1 or *k* >= *n*.

    References:
        Inspired by the Huiskamp (1991) surface Laplacian used in the
        unicycle ``compute_laplacian.m``, simplified to use inverse-distance
        weighting without angular correction terms.
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]

    if k < 1 or k >= n:
        raise ValueError(f"k must satisfy 1 <= k < n_points ({n}), got k={k}")

    # Pairwise squared distances via broadcasting: (n,1,3) - (1,n,3)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)

    # For each point, find k nearest neighbors (exclude self)
    # argsort along axis=1; column 0 is self (distance=0)
    idx_sorted = np.argsort(dist, axis=1)
    knn_idx = idx_sorted[:, 1 : k + 1]  # (n, k)
    knn_dist = np.take_along_axis(dist, knn_idx, axis=1)  # (n, k)

    # Inverse-distance weights: w_ij = 1 / d_ij
    weights = 1.0 / knn_dist  # (n, k)

    # Build sparse weight matrix W (asymmetric KNN graph)
    rows = np.repeat(np.arange(n), k)
    cols = knn_idx.ravel()
    vals = weights.ravel()
    W = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(n, n))

    # Symmetrize: W_sym = (W + W^T) / 2
    W_sym = (W + W.T) / 2.0
    W_sym = W_sym.tocsc()

    # Laplacian: L = W_sym - D, where D is the diagonal degree matrix
    diag_sums = np.asarray(W_sym.sum(axis=1)).ravel()
    D = scipy.sparse.diags(diag_sums, format="csc")
    return W_sym - D


def build_laplacian_2d(nL: int, nW: int) -> np.ndarray:
    """Build a 2-D finite-difference Laplacian for a rectangular fault grid.

    Uses central differences for interior points and forward/backward
    differences at boundaries, ensuring every row sums to zero.

    Args:
        nL: Number of patches along strike.
        nW: Number of patches down dip.

    Returns:
        Laplacian matrix of shape (nL*nW, nL*nW).

    Raises:
        ValueError: If nL < 3 or nW < 3.
    """
    if nL < 3 or nW < 3:
        raise ValueError(
            "Laplacian calculation requires at least 3 patches in each dimension."
        )

    npatch = nL * nW
    Lap = np.zeros((npatch, npatch))

    for j in range(nW):
        for i in range(nL):
            k = j * nL + i

            if 0 < i < nL - 1:
                Lap[k, k - 1] += 1.0
                Lap[k, k] -= 2.0
                Lap[k, k + 1] += 1.0
            elif i == 0:
                Lap[k, k] += 1.0
                Lap[k, k + 1] -= 2.0
                Lap[k, k + 2] += 1.0
            elif i == nL - 1:
                Lap[k, k] += 1.0
                Lap[k, k - 1] -= 2.0
                Lap[k, k - 2] += 1.0

            if 0 < j < nW - 1:
                Lap[k, k - nL] += 1.0
                Lap[k, k] -= 2.0
                Lap[k, k + nL] += 1.0
            elif j == 0:
                Lap[k, k] += 1.0
                Lap[k, k + nL] -= 2.0
                Lap[k, k + 2 * nL] += 1.0
            elif j == nW - 1:
                Lap[k, k] += 1.0
                Lap[k, k - nL] -= 2.0
                Lap[k, k - 2 * nL] += 1.0

    return Lap


def build_laplacian_2d_simple(nL: int, nW: int) -> np.ndarray:
    """Build a simple 2-D Laplacian with free boundary conditions.

    The diagonal weight equals the number of available neighbors,
    ensuring every row sums to zero.

    Args:
        nL: Number of patches along strike.
        nW: Number of patches down dip.

    Returns:
        Laplacian matrix of shape (nL*nW, nL*nW).
    """
    npatch = nL * nW
    Lap = np.zeros((npatch, npatch))

    for j in range(nW):
        for i in range(nL):
            k = j * nL + i
            num_neighbors = 0

            if i > 0:
                Lap[k, k - 1] = 1.0
                num_neighbors += 1
            if i < nL - 1:
                Lap[k, k + 1] = 1.0
                num_neighbors += 1
            if j > 0:
                Lap[k, k - nL] = 1.0
                num_neighbors += 1
            if j < nW - 1:
                Lap[k, k + nL] = 1.0
                num_neighbors += 1

            Lap[k, k] = -float(num_neighbors)

    return Lap
