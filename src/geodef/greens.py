"""Green's matrix assembly and fault geometry utilities.

Combines displacement/strain Green's matrix construction (from okada_greens)
with patch grid generation, component Green's functions, and Laplacian
regularization operators (from okada_utils).
"""

import logging

import numpy as np

from geodef import okada85, transforms

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
) -> np.ndarray:
    """Build strain Green's matrix for rectangular fault patches.

    Each set of 4 rows corresponds to [NN, NE, EN, EE] strain components
    for one observation point.

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
        G matrix of shape (4*nobs, 2*npatch).
    """
    alt = np.zeros_like(np.asarray(lon, dtype=float))
    nobs, npatch = _check_lengths(lat, lon, lat0, lon0, depth, strike, dip, L, W)

    G = np.zeros((4 * nobs, 2 * npatch))

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

    return G


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
# Fault geometry utilities (local Cartesian coordinates)
# ======================================================================

def fault_outline(
    depth_m: float,
    dip_deg: float,
    length_m: float,
    width_m: float,
    strike_deg: float,
    centroid_E_m: float,
    centroid_N_m: float,
) -> np.ndarray:
    """Compute 2-D outline of a rectangular fault patch.

    Args:
        depth_m: Centroid depth in meters.
        dip_deg: Dip angle in degrees.
        length_m: Along-strike length in meters.
        width_m: Down-dip width in meters.
        strike_deg: Strike angle in degrees.
        centroid_E_m: Easting of centroid in meters.
        centroid_N_m: Northing of centroid in meters.

    Returns:
        Array of shape (5, 2) with corner coordinates (closed polygon).
    """
    strike = np.deg2rad(strike_deg)
    dip = np.deg2rad(dip_deg)
    u_strike = np.array([np.sin(strike), np.cos(strike)])
    u_dip_h = np.array([np.sin(strike + 0.5 * np.pi), np.cos(strike + 0.5 * np.pi)])

    half_L = 0.5 * length_m * u_strike
    half_W_h = 0.5 * width_m * np.cos(dip) * u_dip_h

    C = np.array([centroid_E_m, centroid_N_m])
    top_center = C - half_W_h
    bot_center = C + half_W_h

    top_left = top_center - half_L
    top_right = top_center + half_L
    bot_left = bot_center - half_L
    bot_right = bot_center + half_L

    return np.vstack([top_left, top_right, bot_right, bot_left, top_left])


def build_patch_grid(
    e0: float,
    n0: float,
    z0: float,
    strike_deg: float,
    dip_deg: float,
    fault_L: float,
    fault_W: float,
    nL: int,
    nW: int,
) -> list[dict]:
    """Build a grid of rectangular fault patch centers and geometry.

    Patches are indexed (i, j) where i increases along strike and j
    increases down dip.

    Args:
        e0: Easting of fault centroid.
        n0: Northing of fault centroid.
        z0: Depth of fault centroid.
        strike_deg: Strike angle in degrees.
        dip_deg: Dip angle in degrees.
        fault_L: Total along-strike length.
        fault_W: Total down-dip width.
        nL: Number of patches along strike.
        nW: Number of patches down dip.

    Returns:
        List of dicts, each with keys: i, j, e, n, depth, strike, dip, L, W.
    """
    patch_L = fault_L / nL
    patch_W = fault_W / nW

    strike = np.deg2rad(strike_deg)
    dip = np.deg2rad(dip_deg)
    sin_str, cos_str = np.sin(strike), np.cos(strike)
    sin_dip, cos_dip = np.sin(dip), np.cos(dip)

    fault_eoffset = -0.5 * fault_L * sin_str - 0.5 * fault_W * cos_dip * cos_str
    fault_noffset = -0.5 * fault_L * cos_str + 0.5 * fault_W * cos_dip * sin_str
    fault_uoffset = -0.5 * fault_W * sin_dip

    patches = []
    for j in range(nW):
        for i in range(nL):
            e = e0 + fault_eoffset + (i + 0.5) * patch_L * sin_str + (j + 0.5) * patch_W * cos_dip * cos_str
            n = n0 + fault_noffset + (i + 0.5) * patch_L * cos_str - (j + 0.5) * patch_W * cos_dip * sin_str
            u = fault_uoffset + (j + 0.5) * patch_W * sin_dip
            depth = z0 - u

            patches.append({
                'i': i, 'j': j,
                'e': float(e), 'n': float(n), 'depth': float(depth),
                'strike': float(strike_deg), 'dip': float(dip_deg),
                'L': float(patch_L), 'W': float(patch_W),
            })
    return patches


def build_component_greens(
    obs_e: np.ndarray,
    obs_n: np.ndarray,
    patches: list[dict],
    rake_deg: float = 90.0,
    nu: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-component Green's matrices (GE, GN, GU).

    Each column is the unit-slip response of one patch at the given rake.

    Args:
        obs_e: Observation eastings.
        obs_n: Observation northings.
        patches: List of patch dicts (from build_patch_grid).
        rake_deg: Rake angle in degrees.
        nu: Poisson's ratio.

    Returns:
        Tuple (GE, GN, GU), each of shape (nobs, npatch).
    """
    nobs = len(obs_e)
    npatch = len(patches)
    GE = np.zeros((nobs, npatch))
    GN = np.zeros((nobs, npatch))
    GU = np.zeros((nobs, npatch))

    for p, patch in enumerate(patches):
        e_rel = obs_e - patch['e']
        n_rel = obs_n - patch['n']
        uE, uN, uU = okada85.displacement(
            e_rel, n_rel,
            patch['depth'], patch['strike'], patch['dip'],
            patch['L'], patch['W'],
            rake_deg, 1.0, 0.0, nu,
        )
        GE[:, p] = uE
        GN[:, p] = uN
        GU[:, p] = uU

    return GE, GN, GU


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
