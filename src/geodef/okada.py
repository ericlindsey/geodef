"""Unified Okada displacement dispatcher.

Auto-selects okada85 (fast, surface-only) when all observation points are
at z=0, and okada92 (full 3-D) otherwise. Users typically import this module
rather than okada85/okada92 directly.
"""

import numpy as np

from geodef import okada85
from geodef.okada92 import okada92 as _okada92


def displacement(
    e: np.ndarray,
    n: np.ndarray,
    z: float | np.ndarray,
    depth: float,
    strike: float,
    dip: float,
    length: float,
    width: float,
    rake: float,
    slip: float,
    opening: float,
    nu: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute displacement due to a rectangular dislocation in a half-space.

    Automatically selects okada85 (surface) or okada92 (at depth) based on
    observation depth z.

    Args:
        e: Easting of observation points relative to fault centroid.
        n: Northing of observation points relative to fault centroid.
        z: Observation depth (z <= 0, z=0 is surface). Scalar or array.
        depth: Depth of fault centroid (positive down).
        strike: Strike angle in degrees from North.
        dip: Dip angle in degrees from horizontal.
        length: Along-strike fault length.
        width: Down-dip fault width.
        rake: Rake angle in degrees.
        slip: Slip magnitude.
        opening: Tensile dislocation component.
        nu: Poisson's ratio (default 0.25).

    Returns:
        Tuple (ue, un, uz) of displacement arrays in geographic coordinates.

    Raises:
        ValueError: If any z > 0.
    """
    e = np.atleast_1d(np.asarray(e, dtype=float))
    n = np.atleast_1d(np.asarray(n, dtype=float))
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))

    if np.any(z_arr > 0):
        raise ValueError(
            "Observation points above the surface (z > 0) are not allowed."
        )

    # Broadcast scalar z to match observation arrays
    if z_arr.size == 1:
        z_arr = np.broadcast_to(z_arr, e.shape)

    if np.all(z_arr == 0.0):
        return okada85.displacement(
            e, n, depth, strike, dip, length, width, rake, slip, opening, nu
        )

    # Decompose rake+slip into strike-slip and dip-slip components
    rake_rad = np.radians(rake)
    strike_slip = slip * np.cos(rake_rad)
    dip_slip = slip * np.sin(rake_rad)

    # Use okada92 for each observation point (it takes scalar inputs)
    # G=1 gives displacement directly (consistent with okada85 convention)
    ue = np.empty(e.shape)
    un = np.empty(e.shape)
    uz = np.empty(e.shape)

    for i in range(e.size):
        disp, _ = _okada92(
            e.flat[i], n.flat[i], z_arr.flat[i],
            depth, strike, dip, length, width,
            strike_slip, dip_slip, opening,
            G=1.0, nu=nu,
        )
        ue.flat[i] = disp[0, 0]
        un.flat[i] = disp[1, 0]
        uz.flat[i] = disp[2, 0]

    return ue, un, uz
