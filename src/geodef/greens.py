"""Green's matrix assembly and fault geometry utilities.

Combines displacement/strain Green's matrix construction (from okada_greens)
with patch grid generation, component Green's functions, and Laplacian
regularization operators (from okada_utils). Also provides the polymorphic
``greens()`` function for assembling projected Green's matrices from
``Fault`` and ``DataSet`` objects.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg

from geodef import cache as _cache
from geodef import okada85, transforms, tri

if TYPE_CHECKING:
    from geodef.data import DataSet
    from geodef.fault import Fault

logger = logging.getLogger(__name__)


# ======================================================================
# Green's matrix assembly (geographic coordinates)
# ======================================================================

def displacement_greens(
    lat: np.ndarray,
    lon: np.ndarray,
    lat0: np.ndarray,
    lon0: np.ndarray,
    depth: np.ndarray,
    strike: np.ndarray,
    dip: np.ndarray,
    L: np.ndarray,
    W: np.ndarray,
    nu: float = 0.25,
) -> np.ndarray:
    """Build displacement Green's matrix for rectangular fault patches.

    Each set of 3 rows corresponds to [E, N, U] displacements for one
    observation point. Each set of 2 columns corresponds to unit
    [strike-slip, dip-slip] on one fault patch.

    Args:
        lat: Observation latitudes (nobs,).
        lon: Observation longitudes (nobs,).
        lat0: Patch center latitudes (npatch,).
        lon0: Patch center longitudes (npatch,).
        depth: Patch centroid depths (npatch,).
        strike: Patch strike angles in degrees (npatch,).
        dip: Patch dip angles in degrees (npatch,).
        L: Along-strike patch lengths (npatch,).
        W: Down-dip patch widths (npatch,).
        nu: Poisson's ratio.

    Returns:
        G matrix of shape (3*nobs, 2*npatch).
    """
    alt = np.zeros_like(np.asarray(lon, dtype=float))
    nobs, npatch = _check_lengths(lat, lon, lat0, lon0, depth, strike, dip, L, W)

    G = np.zeros((3 * nobs, 2 * npatch))

    for ipatch in range(npatch):
        e, n, _ = transforms.geod2enu(
            lat, lon, alt, lat0[ipatch], lon0[ipatch], 0.0
        )

        str_e, str_n, str_u = okada85.displacement(
            e, n, float(depth[ipatch]), float(strike[ipatch]),
            float(dip[ipatch]), float(L[ipatch]), float(W[ipatch]),
            0.0, 1.0, 0.0, nu,
        )
        dip_e, dip_n, dip_u = okada85.displacement(
            e, n, float(depth[ipatch]), float(strike[ipatch]),
            float(dip[ipatch]), float(L[ipatch]), float(W[ipatch]),
            90.0, 1.0, 0.0, nu,
        )

        gstr = np.zeros(3 * nobs)
        gdip = np.zeros(3 * nobs)
        gstr[::3] = str_e
        gstr[1::3] = str_n
        gstr[2::3] = str_u
        gdip[::3] = dip_e
        gdip[1::3] = dip_n
        gdip[2::3] = dip_u
        G[:, 2 * ipatch] = gstr
        G[:, 2 * ipatch + 1] = gdip

    return G


def strain_greens(
    lat: np.ndarray,
    lon: np.ndarray,
    lat0: np.ndarray,
    lon0: np.ndarray,
    depth: np.ndarray,
    strike: np.ndarray,
    dip: np.ndarray,
    L: np.ndarray,
    W: np.ndarray,
    nu: float = 0.25,
    obs_depth: np.ndarray | None = None,
) -> np.ndarray:
    """Build strain Green's matrix for rectangular fault patches.

    Each set of 4 rows corresponds to [NN, NE, EN, EE] strain components
    for one observation point.

    Args:
        lat: Observation latitudes (nobs,).
        lon: Observation longitudes (nobs,).
        lat0: Patch center latitudes (npatch,).
        lon0: Patch center longitudes (npatch,).
        depth: Patch centroid depths (npatch,), positive down.
        strike: Patch strike angles in degrees (npatch,).
        dip: Patch dip angles in degrees (npatch,).
        L: Along-strike patch lengths (npatch,).
        W: Down-dip patch widths (npatch,).
        nu: Poisson's ratio.
        obs_depth: Observation depths (nobs,), positive down. If None,
            observations are at the surface (uses okada85). If provided,
            uses okada92 (DC3D) for internal deformation at depth.

    Returns:
        G matrix of shape (4*nobs, 2*npatch).
    """
    alt = np.zeros_like(np.asarray(lon, dtype=float))
    nobs, npatch = _check_lengths(lat, lon, lat0, lon0, depth, strike, dip, L, W)

    G = np.zeros((4 * nobs, 2 * npatch))

    if obs_depth is None:
        for ipatch in range(npatch):
            e, n, _ = transforms.geod2enu(
                lat, lon, alt, lat0[ipatch], lon0[ipatch], 0.0
            )

            str_nn, str_ne, str_en, str_ee = okada85.strain(
                e, n, float(depth[ipatch]), float(strike[ipatch]),
                float(dip[ipatch]), float(L[ipatch]), float(W[ipatch]),
                0.0, 1.0, 0.0, nu,
            )
            dip_nn, dip_ne, dip_en, dip_ee = okada85.strain(
                e, n, float(depth[ipatch]), float(strike[ipatch]),
                float(dip[ipatch]), float(L[ipatch]), float(W[ipatch]),
                90.0, 1.0, 0.0, nu,
            )

            gstr = np.zeros(4 * nobs)
            gdip = np.zeros(4 * nobs)
            gstr[::4] = str_nn
            gstr[1::4] = str_ne
            gstr[2::4] = str_en
            gstr[3::4] = str_ee
            gdip[::4] = dip_nn
            gdip[1::4] = dip_ne
            gdip[2::4] = dip_en
            gdip[3::4] = dip_ee
            G[:, 2 * ipatch] = gstr
            G[:, 2 * ipatch + 1] = gdip
    else:
        from geodef import okada92
        obs_depth = np.asarray(obs_depth, dtype=float)
        G_mu = 1.0  # unit shear modulus; actual scaling done by caller
        for ipatch in range(npatch):
            e, n, _ = transforms.geod2enu(
                lat, lon, alt, lat0[ipatch], lon0[ipatch], 0.0
            )
            for iobs in range(nobs):
                z_obs = -float(obs_depth[iobs])  # okada92 convention: Z <= 0
                _, strain_ss = okada92.okada92(
                    float(e[iobs]), float(n[iobs]), z_obs,
                    float(depth[ipatch]), float(strike[ipatch]),
                    float(dip[ipatch]), float(L[ipatch]), float(W[ipatch]),
                    1.0, 0.0, 0.0, G_mu, nu, allow_singular=True,
                )
                _, strain_ds = okada92.okada92(
                    float(e[iobs]), float(n[iobs]), z_obs,
                    float(depth[ipatch]), float(strike[ipatch]),
                    float(dip[ipatch]), float(L[ipatch]), float(W[ipatch]),
                    0.0, 1.0, 0.0, G_mu, nu, allow_singular=True,
                )
                row = 4 * iobs
                # NN, NE, EN, EE from the 3x3 gradient tensor
                G[row, 2 * ipatch] = strain_ss[1, 1]      # NN
                G[row + 1, 2 * ipatch] = strain_ss[1, 0]  # NE
                G[row + 2, 2 * ipatch] = strain_ss[0, 1]  # EN
                G[row + 3, 2 * ipatch] = strain_ss[0, 0]  # EE
                G[row, 2 * ipatch + 1] = strain_ds[1, 1]
                G[row + 1, 2 * ipatch + 1] = strain_ds[1, 0]
                G[row + 2, 2 * ipatch + 1] = strain_ds[0, 1]
                G[row + 3, 2 * ipatch + 1] = strain_ds[0, 0]

    return G


# ======================================================================
# Green's matrix assembly for triangular patches (geographic coordinates)
# ======================================================================

def tri_displacement_greens(
    lat: np.ndarray,
    lon: np.ndarray,
    lat0: np.ndarray,
    lon0: np.ndarray,
    depth: np.ndarray,
    vertices: np.ndarray,
    nu: float = 0.25,
) -> np.ndarray:
    """Build displacement Green's matrix for triangular fault patches.

    Each set of 3 rows corresponds to [E, N, U] displacements for one
    observation point. Each set of 2 columns corresponds to unit
    [strike-slip, dip-slip] on one fault patch.

    Args:
        lat: Observation latitudes (nobs,).
        lon: Observation longitudes (nobs,).
        lat0: Patch center latitudes (npatch,).
        lon0: Patch center longitudes (npatch,).
        depth: Patch centroid depths (npatch,), positive down.
        vertices: Triangle vertices in local ENU, shape (npatch, 3, 3).
        nu: Poisson's ratio.

    Returns:
        G matrix of shape (3*nobs, 2*npatch).
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    alt = np.zeros_like(lon)
    nobs = lat.shape[0]
    npatch = vertices.shape[0]
    ref_lat = float(np.mean(lat0))
    ref_lon = float(np.mean(lon0))

    # Convert observation points to local ENU relative to fault centroid
    obs_e, obs_n, _ = transforms.geod2enu(lat, lon, alt, ref_lat, ref_lon, 0.0)
    obs = np.column_stack([obs_e, obs_n, np.zeros(nobs)])

    G = np.zeros((3 * nobs, 2 * npatch))

    for ipatch in range(npatch):
        tri_verts = vertices[ipatch]  # (3, 3) in local ENU

        # Strike-slip: slip = [1, 0, 0]
        disp_ss = tri.TDdispHS(obs, tri_verts, np.array([1.0, 0.0, 0.0]), nu)
        # Dip-slip: slip = [0, 1, 0]
        disp_ds = tri.TDdispHS(obs, tri_verts, np.array([0.0, 1.0, 0.0]), nu)

        gstr = np.zeros(3 * nobs)
        gdip = np.zeros(3 * nobs)
        gstr[::3] = disp_ss[:, 0]   # East
        gstr[1::3] = disp_ss[:, 1]  # North
        gstr[2::3] = disp_ss[:, 2]  # Up
        gdip[::3] = disp_ds[:, 0]
        gdip[1::3] = disp_ds[:, 1]
        gdip[2::3] = disp_ds[:, 2]
        G[:, 2 * ipatch] = gstr
        G[:, 2 * ipatch + 1] = gdip

    return G


def tri_strain_greens(
    lat: np.ndarray,
    lon: np.ndarray,
    lat0: np.ndarray,
    lon0: np.ndarray,
    depth: np.ndarray,
    vertices: np.ndarray,
    nu: float = 0.25,
    obs_depth: np.ndarray | None = None,
) -> np.ndarray:
    """Build strain Green's matrix for triangular fault patches.

    Each set of 6 rows corresponds to [xx, yy, zz, xy, xz, yz] strain
    components for one observation point.

    Args:
        lat: Observation latitudes (nobs,).
        lon: Observation longitudes (nobs,).
        lat0: Patch center latitudes (npatch,).
        lon0: Patch center longitudes (npatch,).
        depth: Patch centroid depths (npatch,), positive down.
        vertices: Triangle vertices in local ENU, shape (npatch, 3, 3).
        nu: Poisson's ratio.
        obs_depth: Observation depths (nobs,), positive down. If None,
            observations are at the surface. If provided, the z-coordinate
            of observation points is set to ``-obs_depth`` (negative = below
            surface in the ENU frame used by TDstrainHS).

    Returns:
        G matrix of shape (6*nobs, 2*npatch).
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    alt = np.zeros_like(lon)
    nobs = lat.shape[0]
    npatch = vertices.shape[0]
    ref_lat = float(np.mean(lat0))
    ref_lon = float(np.mean(lon0))

    obs_e, obs_n, _ = transforms.geod2enu(lat, lon, alt, ref_lat, ref_lon, 0.0)
    if obs_depth is not None:
        obs_z = -np.asarray(obs_depth, dtype=float)
    else:
        obs_z = np.zeros(nobs)
    obs = np.column_stack([obs_e, obs_n, obs_z])

    G = np.zeros((6 * nobs, 2 * npatch))

    for ipatch in range(npatch):
        tri_verts = vertices[ipatch]

        strain_ss = tri.TDstrainHS(obs, tri_verts, np.array([1.0, 0.0, 0.0]), nu)
        strain_ds = tri.TDstrainHS(obs, tri_verts, np.array([0.0, 1.0, 0.0]), nu)

        gstr = np.zeros(6 * nobs)
        gdip = np.zeros(6 * nobs)
        for c in range(6):
            gstr[c::6] = strain_ss[:, c]
            gdip[c::6] = strain_ds[:, c]
        G[:, 2 * ipatch] = gstr
        G[:, 2 * ipatch + 1] = gdip

    return G


# ======================================================================
# Polymorphic Green's matrix assembly (Fault + DataSet)
# ======================================================================

def _build_greens_key(fault: Fault, data: DataSet) -> dict:
    """Build the cache key dict for a fault + dataset combination."""
    from geodef.data import GNSS, InSAR

    key: dict = {
        "fault_lat": fault._lat,
        "fault_lon": fault._lon,
        "fault_depth": fault._depth,
        "fault_strike": fault._strike,
        "fault_dip": fault._dip,
        "engine": fault.engine,
        "obs_lat": data.lat,
        "obs_lon": data.lon,
        "data_class": type(data).__name__,
        "greens_type": data.greens_type,
    }
    if fault._length is not None:
        key["fault_length"] = fault._length
        key["fault_width"] = fault._width
    if fault._vertices is not None:
        key["fault_vertices"] = fault._vertices
    if isinstance(data, InSAR):
        key["look_e"] = data._look_e
        key["look_n"] = data._look_n
        key["look_u"] = data._look_u
    if isinstance(data, GNSS):
        key["components"] = data.components
    return key


def greens(fault: Fault, datasets: DataSet | list[DataSet]) -> np.ndarray:
    """Build a projected Green's matrix for one or more datasets.

    Computes the raw Green's matrix from the fault at each dataset's
    observation locations, then projects each column through
    ``data.project()`` to map into the dataset's observation space.

    Args:
        fault: A ``Fault`` instance.
        datasets: A single ``DataSet`` or a list of them.

    Returns:
        Projected Green's matrix. For a single dataset with M observations
        and a fault with N patches: shape (M_obs, 2*N). For multiple
        datasets: rows are vertically stacked.
    """
    from geodef.data import DataSet

    if isinstance(datasets, DataSet):
        datasets = [datasets]

    blocks = []
    for data in datasets:
        key = _build_greens_key(fault, data)
        G_proj = _cache.cached_compute(
            key,
            lambda d=data: _project_greens(
                d, fault.greens_matrix(d.lat, d.lon, kind=d.greens_type)
            ),
        )
        blocks.append(G_proj)

    return np.vstack(blocks)


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


def _project_greens(data: DataSet, G_raw: np.ndarray) -> np.ndarray:
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


def resolution(G: np.ndarray) -> np.ndarray:
    """Compute resolution matrix R = pinv(G) @ G.

    Args:
        G: Green's matrix.

    Returns:
        Resolution matrix of shape (ncols, ncols).
    """
    return np.linalg.pinv(G) @ G


def _check_lengths(
    lat: np.ndarray,
    lon: np.ndarray,
    lat0: np.ndarray,
    lon0: np.ndarray,
    depth: np.ndarray,
    strike: np.ndarray,
    dip: np.ndarray,
    L: np.ndarray,
    W: np.ndarray,
) -> tuple[int, int]:
    """Validate that observation and patch arrays have consistent lengths.

    Args:
        lat, lon: Observation coordinate arrays.
        lat0, lon0, depth, strike, dip, L, W: Patch parameter arrays.

    Returns:
        Tuple (nobs, npatch).

    Raises:
        ValueError: If array lengths are inconsistent.
    """
    nobs = len(lat)
    npatch = len(strike)
    if len(lon) != nobs:
        raise ValueError("lat and lon must have the same length")
    patch_lens = {len(lat0), len(lon0), len(depth), len(dip), len(L), len(W)}
    if len(patch_lens) > 1 or npatch not in patch_lens:
        raise ValueError(
            "lat0, lon0, depth, strike, dip, L, W must all have the same length"
        )
    return nobs, npatch


# ======================================================================
# Regularization operators
# ======================================================================

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
