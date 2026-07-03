"""Euler pole estimation and rigid-block velocity prediction.

Rigid-plate motion on a sphere is a rotation about an Euler pole. Given
horizontal GNSS velocities on a block, :func:`best_fit_pole` recovers the
pole (latitude, longitude, rotation rate) by weighted least squares, and
:func:`pole_velocity` predicts the velocity field a pole produces.

Conventions:
- Latitudes/longitudes are geodetic degrees; the rotation itself is computed
  on a sphere (geodetic latitudes are converted with ``transforms.geod2spher``).
- Rotation rate is in degrees per million years (deg/Myr).
- Velocities are horizontal (East, North) in mm/yr.

Ported and vectorized from the ``shakeout_v2`` reference ``euler_calc.py``.
"""

import numpy as np
import scipy.linalg

from geodef import transforms

_EARTH_RADIUS_KM = 6371.0


def euler_vector(lat_p: float, lon_p: float, rate: float) -> np.ndarray:
    """Convert a geodetic Euler pole to a scaled Cartesian rotation vector.

    The vector is scaled so that multiplying it by :func:`euler_rot_matrix`
    (built from unit-sphere coordinates) yields velocities in mm/yr.

    Args:
        lat_p: Pole latitude in geodetic degrees.
        lon_p: Pole longitude in degrees.
        rate: Rotation rate in deg/Myr.

    Returns:
        Cartesian rotation vector ``[px, py, pz]``, shape (3,).
    """
    lat_c = np.radians(transforms.geod2spher(lat_p))
    lon_r = np.radians(lon_p)
    px = np.cos(lon_r) * np.cos(lat_c)
    py = np.sin(lon_r) * np.cos(lat_c)
    pz = np.sin(lat_c)
    scale = _EARTH_RADIUS_KM * rate * (np.pi / 180.0)
    return scale * np.array([px, py, pz])


def euler_location(omega: np.ndarray) -> tuple[float, float, float]:
    """Convert a Cartesian rotation vector back to a geodetic pole.

    Args:
        omega: Cartesian rotation vector ``[px, py, pz]``, shape (3,).

    Returns:
        ``(lat, lon, rate)`` with geodetic latitude/longitude in degrees and
        rate in deg/Myr.
    """
    px, py, pz = omega
    mag = np.sqrt(px**2 + py**2 + pz**2)
    rate = float(mag / (_EARTH_RADIUS_KM * np.pi / 180.0))
    lat_c = np.degrees(np.arctan2(pz, np.sqrt(px**2 + py**2)))
    lat = float(transforms.spher2geod(lat_c))
    lon = float(np.degrees(np.arctan2(py, px)))
    return lat, lon, rate


def euler_rot_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Build the ``(2n, 3)`` design matrix mapping a rotation vector to velocities.

    Each station contributes a ``(2, 3)`` block that performs the cross product
    of the rotation vector with the station's position, projected onto the
    local East and North directions.

    Args:
        lat: Station latitudes in geodetic degrees, shape (n,).
        lon: Station longitudes in degrees, shape (n,).

    Returns:
        Design matrix ``Rx``, shape (2n, 3). Rows alternate East, North.
    """
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    lat_r = np.radians(transforms.geod2spher(lat))
    lon_r = np.radians(lon)

    sin_lat, cos_lat = np.sin(lat_r), np.cos(lat_r)
    sin_lon, cos_lon = np.sin(lon_r), np.cos(lon_r)
    zeros = np.zeros_like(lat_r)

    east_rows = np.column_stack([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    north_rows = np.column_stack([sin_lon, -cos_lon, zeros])

    n = lat_r.shape[0]
    rx = np.empty((2 * n, 3))
    rx[0::2] = east_rows
    rx[1::2] = north_rows
    return rx


def pole_velocity(
    lat: np.ndarray,
    lon: np.ndarray,
    lat_p: float,
    lon_p: float,
    rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict horizontal velocities produced by an Euler pole.

    Args:
        lat: Station latitudes in geodetic degrees, shape (n,).
        lon: Station longitudes in degrees, shape (n,).
        lat_p: Pole latitude in geodetic degrees.
        lon_p: Pole longitude in degrees.
        rate: Rotation rate in deg/Myr.

    Returns:
        ``(ve, vn)`` predicted East and North velocities in mm/yr, each (n,).
    """
    omega = euler_vector(lat_p, lon_p, rate)
    vout = euler_rot_matrix(lat, lon) @ omega
    return vout[0::2], vout[1::2]


def _euler_jacobian(omega: np.ndarray) -> np.ndarray:
    """Jacobian of ``(lat, lon, rate)`` with respect to the Cartesian vector."""
    px, py, pz = omega
    mag = np.sqrt(px**2 + py**2 + pz**2)
    horiz = np.sqrt(px**2 + py**2)
    horiz_sq = px**2 + py**2
    return np.array(
        [
            [
                (-1 / mag**2) * (px * pz / horiz),
                (-1 / mag**2) * (py * pz / horiz),
                (-1 / mag**2) * horiz,
            ],
            [-py / horiz_sq, -px / horiz_sq, 0.0],
            [px / mag, py / mag, pz / mag],
        ]
    )


def _covariance_2d(
    sig_e: np.ndarray, sig_n: np.ndarray, rho: np.ndarray | float
) -> np.ndarray:
    """Block-diagonal 2-D velocity covariance from sigmas and E-N correlation."""
    sig_e = np.atleast_1d(np.asarray(sig_e, dtype=float))
    sig_n = np.atleast_1d(np.asarray(sig_n, dtype=float))
    rho = np.broadcast_to(np.asarray(rho, dtype=float), sig_e.shape)
    blocks = [
        np.array([[e * e, e * n * r], [e * n * r, n * n]])
        for e, n, r in zip(sig_e, sig_n, rho)
    ]
    return scipy.linalg.block_diag(*blocks)


def best_fit_pole(
    lat: np.ndarray,
    lon: np.ndarray,
    ve: np.ndarray,
    vn: np.ndarray,
    sig_e: np.ndarray,
    sig_n: np.ndarray,
    rho: np.ndarray | float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit the best Euler pole to horizontal GNSS velocities by weighted LS.

    Args:
        lat: Station latitudes in geodetic degrees, shape (n,).
        lon: Station longitudes in degrees, shape (n,).
        ve: East velocities in mm/yr, shape (n,).
        vn: North velocities in mm/yr, shape (n,).
        sig_e: East 1-sigma uncertainties, shape (n,).
        sig_n: North 1-sigma uncertainties, shape (n,).
        rho: East-North correlation coefficient, scalar or shape (n,).

    Returns:
        ``(pole, cov_pole, chi2_reduced)`` where ``pole`` is
        ``[lat, lon, rate]`` (deg, deg, deg/Myr), ``cov_pole`` is the 3x3
        covariance of those geodetic pole parameters, and ``chi2_reduced``
        is the reduced chi-squared misfit.

    Raises:
        ValueError: If fewer than two stations are supplied (the pole has
            three parameters and each station gives two equations).
    """
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    if lat.shape[0] < 2:
        raise ValueError("best_fit_pole requires at least two stations")

    vel = np.ravel(np.column_stack([ve, vn]))
    weight = np.linalg.inv(_covariance_2d(sig_e, sig_n, rho))
    rx = euler_rot_matrix(lat, lon)

    normal_inv = np.linalg.inv(rx.T @ weight @ rx)
    fitpole = normal_inv @ (rx.T @ weight @ vel)
    lat_p, lon_p, rate = euler_location(fitpole)

    pred_e, pred_n = pole_velocity(lat, lon, lat_p, lon_p, rate)
    resid = np.ravel(np.column_stack([ve - pred_e, vn - pred_n]))
    dof = 2 * lat.shape[0] - 3
    sigma0_sq = float(resid.T @ weight @ resid) / dof
    jac = _euler_jacobian(fitpole)
    cov_pole = jac @ (sigma0_sq * normal_inv) @ jac.T

    uncert_inv = np.ravel(
        np.column_stack(
            [np.asarray(sig_e, dtype=float) ** -2, np.asarray(sig_n, dtype=float) ** -2]
        )
    )
    chi2_red = float(resid**2 @ uncert_inv) / dof

    return np.array([lat_p, lon_p, rate]), cov_pole, chi2_red


def remove_pole(
    lat: np.ndarray,
    lon: np.ndarray,
    ve: np.ndarray,
    vn: np.ndarray,
    lat_p: float,
    lon_p: float,
    rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract an Euler-pole rotation from a velocity field.

    Useful for viewing velocities in a block-fixed reference frame, e.g.
    removing rigid plate motion to isolate near-fault deformation.

    Args:
        lat: Station latitudes in geodetic degrees, shape (n,).
        lon: Station longitudes in degrees, shape (n,).
        ve: East velocities in mm/yr, shape (n,).
        vn: North velocities in mm/yr, shape (n,).
        lat_p: Pole latitude in geodetic degrees.
        lon_p: Pole longitude in degrees.
        rate: Rotation rate in deg/Myr.

    Returns:
        ``(ve_res, vn_res)`` residual velocities in mm/yr, each (n,).
    """
    pred_e, pred_n = pole_velocity(lat, lon, lat_p, lon_p, rate)
    return ve - pred_e, vn - pred_n
