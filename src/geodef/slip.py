"""Slip-vector packing and basis conversion functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from geodef.fault import Fault

__all__ = [
    "from_azimuth",
    "from_plate",
    "from_rake",
    "magnitude",
    "pack",
    "plate_rake_from_euler",
    "rake",
    "to_plate",
    "unpack",
]


def _as_1d(values: npt.ArrayLike, name: str) -> np.ndarray:
    """Return a finite one-dimensional float array."""
    array = np.atleast_1d(np.asarray(values, dtype=float))
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _matching_components(
    first: npt.ArrayLike,
    second: npt.ArrayLike,
    first_name: str,
    second_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate two component arrays with the same shape."""
    first_array = _as_1d(first, first_name)
    second_array = _as_1d(second, second_name)
    if first_array.shape != second_array.shape:
        raise ValueError(
            f"{first_name} and {second_name} must have the same shape, got "
            f"{first_array.shape} and {second_array.shape}"
        )
    return first_array, second_array


def _angle(values: npt.ArrayLike, size: int, name: str) -> np.ndarray:
    """Broadcast a finite scalar or per-patch angle array."""
    angle = np.asarray(values, dtype=float)
    try:
        result = np.broadcast_to(angle, (size,))
    except ValueError as exc:
        raise ValueError(f"{name} must be scalar or have shape ({size},)") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def pack(strike_slip: npt.ArrayLike, dip_slip: npt.ArrayLike) -> np.ndarray:
    """Pack physical slip components into GeoDef's blocked vector.

    Args:
        strike_slip: Strike-slip value per patch, shape ``(N,)``.
        dip_slip: Dip-slip value per patch, shape ``(N,)``.

    Returns:
        Blocked ``[strike_slip | dip_slip]`` vector, shape ``(2N,)``.

    Raises:
        ValueError: If either component is non-finite, not one-dimensional, or
            has a different shape.
    """
    strike_array, dip_array = _matching_components(
        strike_slip, dip_slip, "strike_slip", "dip_slip"
    )
    return np.concatenate([strike_array, dip_array])


def unpack(vector: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Unpack a blocked two-component slip vector.

    Args:
        vector: Blocked ``[strike_slip | dip_slip]`` vector, shape ``(2N,)``.

    Returns:
        ``(strike_slip, dip_slip)`` arrays, each shape ``(N,)``.

    Raises:
        ValueError: If ``vector`` is empty, non-finite, not one-dimensional, or
            does not contain an even number of entries.
    """
    array = _as_1d(vector, "vector")
    if array.size == 0 or array.size % 2:
        raise ValueError("vector must contain a non-empty even number of entries")
    midpoint = array.size // 2
    return array[:midpoint], array[midpoint:]


def from_rake(
    amplitude: npt.ArrayLike, rake_degrees: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Convert signed amplitudes along rake to strike/dip components.

    Args:
        amplitude: Signed slip amplitude per patch, shape ``(N,)``.
        rake_degrees: Local rake in degrees, scalar or shape ``(N,)``.

    Returns:
        ``(strike_slip, dip_slip)`` arrays, each shape ``(N,)``.
    """
    amplitude_array = _as_1d(amplitude, "amplitude")
    angle = np.deg2rad(_angle(rake_degrees, amplitude_array.size, "rake_degrees"))
    return amplitude_array * np.cos(angle), amplitude_array * np.sin(angle)


def from_azimuth(
    amplitude: npt.ArrayLike,
    azimuth_degrees: float,
    fault_strike_degrees: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert slip along a geographic azimuth to patch-local components.

    Args:
        amplitude: Signed slip amplitude per patch, shape ``(N,)``.
        azimuth_degrees: Geographic direction clockwise from North.
        fault_strike_degrees: Strike of each patch in degrees, shape ``(N,)``.

    Returns:
        ``(strike_slip, dip_slip)`` arrays, each shape ``(N,)``.
    """
    amplitude_array = _as_1d(amplitude, "amplitude")
    strike = _angle(fault_strike_degrees, amplitude_array.size, "fault_strike_degrees")
    return from_rake(amplitude_array, float(azimuth_degrees) - strike)


def from_plate(
    parallel: npt.ArrayLike,
    perpendicular: npt.ArrayLike,
    plate_rake_degrees: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate plate-coordinate slip into patch-local strike/dip components.

    Args:
        parallel: Plate-rake-parallel slip per patch, shape ``(N,)``.
        perpendicular: Plate-rake-perpendicular slip per patch, shape ``(N,)``.
        plate_rake_degrees: Plate direction in local rake coordinates, scalar or
            shape ``(N,)``.

    Returns:
        ``(strike_slip, dip_slip)`` arrays, each shape ``(N,)``.
    """
    parallel_array, perpendicular_array = _matching_components(
        parallel, perpendicular, "parallel", "perpendicular"
    )
    angle = np.deg2rad(
        _angle(
            plate_rake_degrees,
            parallel_array.size,
            "plate_rake_degrees",
        )
    )
    strike_slip = parallel_array * np.cos(angle) - perpendicular_array * np.sin(angle)
    dip_slip = parallel_array * np.sin(angle) + perpendicular_array * np.cos(angle)
    return strike_slip, dip_slip


def to_plate(
    strike_slip: npt.ArrayLike,
    dip_slip: npt.ArrayLike,
    plate_rake_degrees: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate patch-local strike/dip slip into plate coordinates.

    Args:
        strike_slip: Strike-slip value per patch, shape ``(N,)``.
        dip_slip: Dip-slip value per patch, shape ``(N,)``.
        plate_rake_degrees: Plate direction in local rake coordinates, scalar or
            shape ``(N,)``.

    Returns:
        ``(parallel, perpendicular)`` arrays, each shape ``(N,)``.
    """
    strike_array, dip_array = _matching_components(
        strike_slip, dip_slip, "strike_slip", "dip_slip"
    )
    angle = np.deg2rad(
        _angle(plate_rake_degrees, strike_array.size, "plate_rake_degrees")
    )
    parallel = strike_array * np.cos(angle) + dip_array * np.sin(angle)
    perpendicular = -strike_array * np.sin(angle) + dip_array * np.cos(angle)
    return parallel, perpendicular


def magnitude(strike_slip: npt.ArrayLike, dip_slip: npt.ArrayLike) -> np.ndarray:
    """Return unsigned physical slip magnitude per patch.

    Args:
        strike_slip: Strike-slip value per patch, shape ``(N,)``.
        dip_slip: Dip-slip value per patch, shape ``(N,)``.

    Returns:
        Slip magnitude, shape ``(N,)``.
    """
    strike_array, dip_array = _matching_components(
        strike_slip, dip_slip, "strike_slip", "dip_slip"
    )
    return np.hypot(strike_array, dip_array)


def rake(strike_slip: npt.ArrayLike, dip_slip: npt.ArrayLike) -> np.ndarray:
    """Return physical local rake in degrees per patch.

    Args:
        strike_slip: Strike-slip value per patch, shape ``(N,)``.
        dip_slip: Dip-slip value per patch, shape ``(N,)``.

    Returns:
        Rake in degrees, shape ``(N,)``.
    """
    strike_array, dip_array = _matching_components(
        strike_slip, dip_slip, "strike_slip", "dip_slip"
    )
    return np.degrees(np.arctan2(dip_array, strike_array))


def plate_rake_from_euler(fault: Fault, pole: tuple[float, float, float]) -> np.ndarray:
    """Compute plate direction in each patch's local rake coordinates.

    Args:
        fault: Fault whose patch centers and strikes define the basis.
        pole: ``(latitude, longitude, rate)`` in degrees, degrees, deg/Myr.

    Returns:
        Local plate rake in degrees for each patch, shape ``(N,)``.

    Raises:
        ValueError: If a patch center lies on the Euler axis and therefore has
            no defined velocity direction.
    """
    from geodef.euler import pole_velocity

    centers = fault.centers_geo
    east, north = pole_velocity(centers[:, 1], centers[:, 0], pole[0], pole[1], pole[2])
    if np.any(np.hypot(east, north) == 0.0):
        raise ValueError("Euler pole produces zero velocity at a patch center")
    azimuth = np.degrees(np.arctan2(east, north))
    return np.asarray(azimuth - fault.strike)
