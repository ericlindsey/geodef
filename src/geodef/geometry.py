"""Coordinate frames and geometry-array conversion functions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from geodef import transforms
from geodef.validation import check_finite_scalar, check_positive, check_range

DEFAULT_PROJECTION = "wgs84-enu"
"""Current geographic-to-local projection identifier."""

PLANAR_PARAMETER_NAMES = (
    "e0",
    "n0",
    "depth",
    "strike",
    "dip",
    "length",
    "width",
)
"""Order of the expert planar-geometry parameter vector."""

__all__ = [
    "DEFAULT_PROJECTION",
    "PLANAR_PARAMETER_NAMES",
    "LocalFrame",
    "as_planar_vector",
    "planar_parameter_dict",
    "triangle_strike_dip",
    "vertices_from_nodes",
]


@dataclass(frozen=True)
class LocalFrame:
    """A local East-North-Up frame tied to a geographic origin.

    The only projection currently supported is ``"wgs84-enu"``: geographic
    WGS84 coordinates are converted through Earth-centered Earth-fixed (ECEF)
    coordinates and rotated into the tangent ENU frame at the origin.

    Args:
        origin_lat: Origin latitude in degrees.
        origin_lon: Origin longitude in degrees.
        origin_alt: Origin ellipsoidal altitude in meters.
        projection: Geographic-to-local projection identifier. Currently only
            ``"wgs84-enu"`` is supported.

    Raises:
        ValueError: If an origin value is non-finite or out of range, or the
            projection is unsupported.
    """

    origin_lat: float
    origin_lon: float
    origin_alt: float = 0.0
    projection: str = DEFAULT_PROJECTION

    def __post_init__(self) -> None:
        for name in ("origin_lat", "origin_lon", "origin_alt"):
            value = float(getattr(self, name))
            check_finite_scalar(
                name, value, unit="degrees" if name != "origin_alt" else "meters"
            )
            object.__setattr__(self, name, value)
        check_range("origin_lat", self.origin_lat, -90.0, 90.0, unit="degrees")
        check_range("origin_lon", self.origin_lon, -360.0, 360.0, unit="degrees")
        if self.projection != DEFAULT_PROJECTION:
            raise ValueError(
                f"projection must be {DEFAULT_PROJECTION!r}; got {self.projection!r}"
            )

    def to_enu(
        self,
        *,
        lon: npt.ArrayLike,
        lat: npt.ArrayLike,
        alt: npt.ArrayLike,
    ) -> np.ndarray:
        """Convert geographic coordinates to ENU meters in this frame.

        Args:
            lon: Longitude in degrees.
            lat: Latitude in degrees.
            alt: Ellipsoidal altitude in meters.

        Returns:
            Array with final axis ``[east_m, north_m, up_m]``.
        """
        lon_array, lat_array, alt_array = np.broadcast_arrays(
            np.asarray(lon, dtype=float),
            np.asarray(lat, dtype=float),
            np.asarray(alt, dtype=float),
        )
        if not np.all(
            np.isfinite(np.stack([lon_array, lat_array, alt_array], axis=-1))
        ):
            raise ValueError("lon, lat, and alt must contain only finite values")
        shape = lon_array.shape
        east, north, up = transforms.geod2enu(
            lat_array.ravel(),
            lon_array.ravel(),
            alt_array.ravel(),
            self.origin_lat,
            self.origin_lon,
            self.origin_alt,
        )
        return np.stack([east, north, up], axis=-1).reshape((*shape, 3))

    def to_geographic(
        self,
        *,
        east: npt.ArrayLike,
        north: npt.ArrayLike,
        up: npt.ArrayLike,
    ) -> np.ndarray:
        """Convert ENU meters in this frame to geographic coordinates.

        Args:
            east: East coordinate in meters.
            north: North coordinate in meters.
            up: Up coordinate in meters.

        Returns:
            Array with final axis ``[longitude_degrees, latitude_degrees,
            altitude_m]``.
        """
        east_array, north_array, up_array = np.broadcast_arrays(
            np.asarray(east, dtype=float),
            np.asarray(north, dtype=float),
            np.asarray(up, dtype=float),
        )
        if not np.all(
            np.isfinite(np.stack([east_array, north_array, up_array], axis=-1))
        ):
            raise ValueError("east, north, and up must contain only finite values")
        shape = east_array.shape
        lat, lon, alt = transforms.enu2geod(
            east_array.ravel(),
            north_array.ravel(),
            up_array.ravel(),
            self.origin_lat,
            self.origin_lon,
            self.origin_alt,
        )
        return np.stack([lon, lat, alt], axis=-1).reshape((*shape, 3))

    def transform_enu(
        self, coordinates: npt.ArrayLike, *, target: LocalFrame
    ) -> np.ndarray:
        """Explicitly re-express ENU coordinates in another local frame.

        Args:
            coordinates: ENU coordinates with shape ``(..., 3)``.
            target: Destination frame.

        Returns:
            Coordinates in ``target`` with the same shape.

        Raises:
            ValueError: If ``coordinates`` does not have a final axis of 3.
        """
        array = np.asarray(coordinates, dtype=float)
        if array.ndim == 0 or array.shape[-1] != 3:
            raise ValueError(f"coordinates must have shape (..., 3), got {array.shape}")
        geographic = self.to_geographic(
            east=array[..., 0], north=array[..., 1], up=array[..., 2]
        )
        return target.to_enu(
            lon=geographic[..., 0],
            lat=geographic[..., 1],
            alt=geographic[..., 2],
        )

    def is_compatible(self, other: LocalFrame) -> bool:
        """Return whether ``other`` is exactly the same coordinate frame."""
        return self == other

    def require_compatible(self, other: LocalFrame) -> None:
        """Raise if ``other`` is not exactly the same coordinate frame.

        Args:
            other: Frame that will be combined with this one.

        Raises:
            ValueError: If the origins or projection differ.
        """
        if not self.is_compatible(other):
            raise ValueError(
                "incompatible local frames: explicitly transform coordinates "
                f"from {other!r} to {self!r} before combining them"
            )


def as_planar_vector(
    parameters: npt.ArrayLike | Mapping[str, float],
) -> np.ndarray:
    """Validate and return the expert planar-geometry vector.

    Args:
        parameters: Either a mapping with the keys in
            :data:`PLANAR_PARAMETER_NAMES` or an array already ordered as
            ``[e0, n0, depth, strike, dip, length, width]``.

    Returns:
        Validated float array with shape ``(7,)``.

    Raises:
        ValueError: If keys, shape, finiteness, or physical ranges are invalid.
    """
    if isinstance(parameters, Mapping):
        missing = [name for name in PLANAR_PARAMETER_NAMES if name not in parameters]
        extra = [name for name in parameters if name not in PLANAR_PARAMETER_NAMES]
        if missing or extra:
            raise ValueError(
                f"planar parameters have missing keys {missing} and extra keys {extra}"
            )
        values = np.array([parameters[name] for name in PLANAR_PARAMETER_NAMES])
    else:
        values = np.asarray(parameters, dtype=float)
    if values.shape != (7,):
        raise ValueError(f"planar parameters must have shape (7,), got {values.shape}")
    if not np.all(np.isfinite(values)):
        bad_index = int(np.flatnonzero(~np.isfinite(values))[0])
        name = PLANAR_PARAMETER_NAMES[bad_index]
        raise ValueError(f"{name} must be finite")

    e0, n0, depth, strike, dip, length, width = map(float, values)
    check_finite_scalar("e0", e0, unit="meters")
    check_finite_scalar("n0", n0, unit="meters")
    if depth < 0.0:
        raise ValueError(f"depth must be non-negative (meters); got {depth:g}")
    if not 0.0 <= strike < 360.0:
        raise ValueError(f"strike must lie in [0, 360) degrees; got {strike:g}")
    check_range("dip", dip, 0.0, 90.0, unit="degrees")
    check_positive("length", length, unit="meters")
    check_positive("width", width, unit="meters")
    return np.array(values, dtype=float, copy=True)


def planar_parameter_dict(
    parameters: npt.ArrayLike | Mapping[str, float],
) -> dict[str, float]:
    """Return named planar parameters from a mapping or expert vector.

    Args:
        parameters: Planar parameter mapping or seven-element expert vector.

    Returns:
        Dictionary keyed by :data:`PLANAR_PARAMETER_NAMES`.
    """
    values = as_planar_vector(parameters)
    return dict(zip(PLANAR_PARAMETER_NAMES, map(float, values), strict=True))


def vertices_from_nodes(
    nodes_enu: npt.ArrayLike, triangles: npt.ArrayLike
) -> np.ndarray:
    """Expand shared ENU nodes into per-triangle vertices.

    Args:
        nodes_enu: Shared nodes with shape ``(M, 3)``.
        triangles: Integer node indices with shape ``(N, 3)``.

    Returns:
        Per-triangle ENU vertices with shape ``(N, 3, 3)``.

    Raises:
        ValueError: If array shapes, finiteness, or indices are invalid.
    """
    nodes = np.asarray(nodes_enu, dtype=float)
    connectivity = np.asarray(triangles, dtype=int)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError(
            f"nodes_enu node array must have shape (M, 3), got {nodes.shape}"
        )
    if not np.all(np.isfinite(nodes)):
        raise ValueError("nodes_enu must contain only finite values")
    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise ValueError(f"triangles must have shape (N, 3), got {connectivity.shape}")
    if connectivity.size and (
        connectivity.min() < 0 or connectivity.max() >= nodes.shape[0]
    ):
        raise ValueError("triangles index out of range for nodes_enu")
    return np.array(nodes[connectivity], copy=True)


def triangle_strike_dip(
    vertices_enu: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive right-hand-rule strike and dip from triangle vertices.

    Args:
        vertices_enu: Per-triangle vertices with shape ``(N, 3, 3)``.

    Returns:
        ``(strike_degrees, dip_degrees)`` arrays, each shape ``(N,)``.

    Raises:
        ValueError: If vertices have the wrong shape or contain non-finite
            values.
    """
    vertices = np.asarray(vertices_enu, dtype=float)
    if vertices.ndim != 3 or vertices.shape[1:] != (3, 3):
        raise ValueError(
            f"vertices_enu must have shape (N, 3, 3), got {vertices.shape}"
        )
    if not np.all(np.isfinite(vertices)):
        raise ValueError("vertices_enu must contain only finite values")
    edge1 = vertices[:, 1, :] - vertices[:, 0, :]
    edge2 = vertices[:, 2, :] - vertices[:, 0, :]
    normal = np.cross(edge1, edge2)
    normal[normal[:, 2] < 0.0] *= -1.0
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)

    dip = np.degrees(np.arccos(np.clip(np.abs(normal[:, 2]), 0.0, 1.0)))
    strike = np.zeros(vertices.shape[0], dtype=float)
    dipping = dip > 0.1
    updip_azimuth = (
        np.degrees(np.arctan2(normal[dipping, 0], normal[dipping, 1])) % 360.0
    )
    strike[dipping] = (updip_azimuth + 90.0) % 360.0
    return strike, dip


def _resolve_frame(
    frame: LocalFrame | None,
    ref_lat: float | None,
    ref_lon: float | None,
) -> LocalFrame:
    """Resolve an explicit or legacy local frame."""
    if (ref_lat is None) != (ref_lon is None):
        raise ValueError("ref_lat and ref_lon must be provided together")
    legacy = (
        LocalFrame(ref_lat, ref_lon)
        if ref_lat is not None and ref_lon is not None
        else None
    )
    if frame is not None and legacy is not None:
        frame.require_compatible(legacy)
    selected = frame if frame is not None else legacy
    if selected is None:
        raise ValueError("provide frame or both ref_lat and ref_lon")
    return selected
