"""Named fault geometries and explicit local coordinate frames.

Local arrays use East, North, Up (ENU) meters. Geographic arrays use
``[longitude_degrees, latitude_degrees, depth_m]`` with depth positive down.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from geodef import transforms
from geodef.validation import check_finite_scalar, check_positive, check_range

DEFAULT_PROJECTION = "wgs84-enu"
"""Current geographic-to-local projection identifier."""

__all__ = [
    "DEFAULT_PROJECTION",
    "LocalFrame",
    "PlanarGeometry",
    "TriGeometry",
]


def _readonly(array: np.ndarray) -> np.ndarray:
    """Return an owned, read-only float array."""
    owned = np.array(array, dtype=float, copy=True)
    owned.flags.writeable = False
    return owned


@dataclass(frozen=True)
class LocalFrame:
    """A local East-North-Up frame tied to a geographic origin and projection.

    The only projection currently supported is ``"wgs84-enu"``: geographic
    WGS84 coordinates are converted through Earth-centered Earth-fixed (ECEF)
    coordinates and rotated into the tangent ENU frame at the origin. The
    explicit ``projection`` field makes coordinate provenance inspectable and
    leaves room for future projected-coordinate implementations.

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
            Array with final axis ``[east_m, north_m, up_m]``. Scalar inputs
            return shape ``(3,)``; one-dimensional inputs return ``(N, 3)``.
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
        result = np.stack([east, north, up], axis=-1).reshape((*shape, 3))
        return result

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
        result = np.stack([lon, lat, alt], axis=-1).reshape((*shape, 3))
        return result

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


@dataclass(frozen=True)
class PlanarGeometry:
    """Named geometry of one planar rectangular fault in a local frame.

    Args:
        center: Fault-centroid ``(east_m, north_m)`` in ``frame``.
        depth: Fault-centroid depth in meters, positive down.
        strike: Strike clockwise from north in degrees, in ``[0, 360)``.
        dip: Dip from horizontal in degrees, in ``[0, 90]``.
        length: Total along-strike length in meters.
        width: Total down-dip width in meters.
        frame: Local coordinate frame that gives ``center`` meaning.
    """

    center: tuple[float, float]
    depth: float
    strike: float
    dip: float
    length: float
    width: float
    frame: LocalFrame

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise ValueError(
                "center must contain two finite values (east_m, north_m); "
                f"got shape {center.shape}"
            )
        object.__setattr__(self, "center", (float(center[0]), float(center[1])))
        for name, unit in (
            ("depth", "meters"),
            ("strike", "degrees"),
            ("dip", "degrees"),
            ("length", "meters"),
            ("width", "meters"),
        ):
            value = float(getattr(self, name))
            check_finite_scalar(name, value, unit=unit)
            object.__setattr__(self, name, value)
        if self.depth < 0.0:
            raise ValueError(
                "depth must be non-negative (positive down, meters); "
                f"got {self.depth:g}"
            )
        if not 0.0 <= self.strike < 360.0:
            raise ValueError(
                f"strike must lie in [0, 360) degrees; got {self.strike:g}"
            )
        check_range("dip", self.dip, 0.0, 90.0, unit="degrees")
        check_positive("length", self.length, unit="meters")
        check_positive("width", self.width, unit="meters")

    @property
    def theta(self) -> np.ndarray:
        """Expert/JAX view ``[east, north, depth, strike, dip, length, width]``."""
        return _readonly(
            np.array(
                [
                    self.center[0],
                    self.center[1],
                    self.depth,
                    self.strike,
                    self.dip,
                    self.length,
                    self.width,
                ]
            )
        )

    def to_enu(self) -> np.ndarray:
        """Return the centroid as ``[east_m, north_m, up_m]``."""
        return _readonly(np.array([self.center[0], self.center[1], -self.depth]))

    def to_geographic(self) -> np.ndarray:
        """Return the centroid as ``[lon_degrees, lat_degrees, depth_m]``."""
        geographic = self.frame.to_geographic(
            east=self.center[0], north=self.center[1], up=0.0
        )
        return _readonly(
            np.array([geographic[0], geographic[1], self.depth], dtype=float)
        )

    @classmethod
    def from_theta(cls, theta: npt.ArrayLike, *, frame: LocalFrame) -> PlanarGeometry:
        """Construct from the seven-element expert/JAX parameter vector.

        Args:
            theta: ``[east, north, depth, strike, dip, length, width]``.
            frame: Frame defining the east and north coordinates.

        Returns:
            Named planar geometry.
        """
        values = np.asarray(theta, dtype=float)
        if values.shape != (7,):
            raise ValueError(f"theta must have shape (7,), got {values.shape}")
        return cls(
            center=(float(values[0]), float(values[1])),
            depth=float(values[2]),
            strike=float(values[3]),
            dip=float(values[4]),
            length=float(values[5]),
            width=float(values[6]),
            frame=frame,
        )

    @classmethod
    def from_geographic(
        cls,
        *,
        lon: float,
        lat: float,
        depth: float,
        strike: float,
        dip: float,
        length: float,
        width: float,
        frame: LocalFrame | None = None,
    ) -> PlanarGeometry:
        """Construct from a geographic centroid and named dimensions.

        Args:
            lon: Centroid longitude in degrees.
            lat: Centroid latitude in degrees.
            depth: Centroid depth in meters, positive down.
            strike: Strike clockwise from north in degrees.
            dip: Dip from horizontal in degrees.
            length: Total along-strike length in meters.
            width: Total down-dip width in meters.
            frame: Frame for the local representation. Defaults to a
                ``wgs84-enu`` frame centered horizontally on the fault.

        Returns:
            Named planar geometry.
        """
        selected_frame = LocalFrame(lat, lon) if frame is None else frame
        enu = selected_frame.to_enu(lon=lon, lat=lat, alt=selected_frame.origin_alt)
        return cls(
            center=(float(enu[0]), float(enu[1])),
            depth=depth,
            strike=strike,
            dip=dip,
            length=length,
            width=width,
            frame=selected_frame,
        )

    def to_frame(self, frame: LocalFrame) -> PlanarGeometry:
        """Return this geometry explicitly re-expressed in ``frame``."""
        if self.frame.is_compatible(frame):
            return self
        geographic = self.frame.to_geographic(
            east=self.center[0], north=self.center[1], up=0.0
        )
        enu = frame.to_enu(lon=geographic[0], lat=geographic[1], alt=geographic[2])
        return PlanarGeometry(
            center=(float(enu[0]), float(enu[1])),
            depth=self.depth,
            strike=self.strike,
            dip=self.dip,
            length=self.length,
            width=self.width,
            frame=frame,
        )


@dataclass(frozen=True, eq=False)
class TriGeometry:
    """Triangular fault vertices in an explicit local ENU frame.

    Per-triangle strike and dip are derived from the physical vertices. A
    large-scale plate-rake basis is intentionally separate from geometry and
    can use these orientations when defining slip coordinates.

    Args:
        vertices_enu: Per-triangle corners with shape ``(N, 3, 3)``; the final
            axis is ``[east_m, north_m, up_m]``.
        frame: Frame defining the vertex coordinates.

    Raises:
        ValueError: If vertices have the wrong shape or contain non-finite
            values.
    """

    vertices_enu: np.ndarray
    frame: LocalFrame

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_enu, dtype=float)
        if vertices.ndim != 3 or vertices.shape[1:] != (3, 3):
            raise ValueError(
                f"vertices_enu must have shape (N, 3, 3); got {vertices.shape}"
            )
        if not np.all(np.isfinite(vertices)):
            raise ValueError("vertices_enu must contain only finite values")
        object.__setattr__(self, "vertices_enu", _readonly(vertices))

    @property
    def n_triangles(self) -> int:
        """Number of triangular patches."""
        return self.vertices_enu.shape[0]

    @property
    def centers_enu(self) -> np.ndarray:
        """Triangle centroids as ``[east_m, north_m, up_m]``."""
        return np.mean(self.vertices_enu, axis=1)

    @property
    def centers_geographic(self) -> np.ndarray:
        """Triangle centroids as ``[lon_degrees, lat_degrees, depth_m]``."""
        centers = self.centers_enu
        geographic = self.frame.to_geographic(
            east=centers[:, 0], north=centers[:, 1], up=centers[:, 2]
        )
        return np.column_stack([geographic[:, 0], geographic[:, 1], -geographic[:, 2]])

    @property
    def strike(self) -> np.ndarray:
        """Per-triangle right-hand-rule strike in degrees, shape ``(N,)``."""
        return _triangle_strike_dip(self.vertices_enu)[0]

    @property
    def dip(self) -> np.ndarray:
        """Per-triangle dip from horizontal in degrees, shape ``(N,)``."""
        return _triangle_strike_dip(self.vertices_enu)[1]

    def to_enu(self) -> np.ndarray:
        """Return the immutable ENU vertex array."""
        return self.vertices_enu

    def to_geographic(self) -> np.ndarray:
        """Return vertices as ``[lon_degrees, lat_degrees, depth_m]``."""
        vertices = self.vertices_enu
        geographic = self.frame.to_geographic(
            east=vertices[..., 0],
            north=vertices[..., 1],
            up=vertices[..., 2],
        )
        result = np.stack(
            [geographic[..., 0], geographic[..., 1], -geographic[..., 2]],
            axis=-1,
        )
        return _readonly(result)

    @classmethod
    def from_nodes(
        cls,
        nodes_enu: npt.ArrayLike,
        triangles: npt.ArrayLike,
        *,
        frame: LocalFrame,
    ) -> TriGeometry:
        """Construct from shared ENU nodes and triangle connectivity.

        Args:
            nodes_enu: Shared nodes with shape ``(M, 3)``.
            triangles: Integer node indices with shape ``(N, 3)``.
            frame: Frame defining the node coordinates.

        Returns:
            Expanded immutable triangular geometry.
        """
        nodes = np.asarray(nodes_enu, dtype=float)
        connectivity = np.asarray(triangles, dtype=int)
        if nodes.ndim != 2 or nodes.shape[1] != 3:
            raise ValueError(f"nodes_enu must have shape (M, 3), got {nodes.shape}")
        if connectivity.ndim != 2 or connectivity.shape[1] != 3:
            raise ValueError(
                f"triangles must have shape (N, 3), got {connectivity.shape}"
            )
        if connectivity.size and (
            connectivity.min() < 0 or connectivity.max() >= nodes.shape[0]
        ):
            raise ValueError("triangles index out of range for nodes_enu")
        return cls(nodes[connectivity], frame)

    @classmethod
    def from_geographic(
        cls,
        *,
        lon: npt.ArrayLike,
        lat: npt.ArrayLike,
        depth: npt.ArrayLike,
        triangles: npt.ArrayLike,
        frame: LocalFrame | None = None,
    ) -> TriGeometry:
        """Construct from geographic nodes and triangle connectivity.

        Args:
            lon: Node longitudes in degrees, shape ``(M,)``.
            lat: Node latitudes in degrees, shape ``(M,)``.
            depth: Node depths in meters, positive down, shape ``(M,)``.
            triangles: Integer node indices, shape ``(N, 3)``.
            frame: Local representation. Defaults to a ``wgs84-enu`` frame
                at the mean node latitude and longitude.

        Returns:
            Triangular geometry in the selected frame.
        """
        lon_array, lat_array, depth_array = np.broadcast_arrays(
            np.asarray(lon, dtype=float),
            np.asarray(lat, dtype=float),
            np.asarray(depth, dtype=float),
        )
        if lon_array.ndim != 1:
            raise ValueError("lon, lat, and depth must be one-dimensional arrays")
        if lon_array.size == 0:
            raise ValueError("geographic nodes must not be empty")
        if not np.all(
            np.isfinite(np.stack([lon_array, lat_array, depth_array], axis=-1))
        ):
            raise ValueError("lon, lat, and depth must contain only finite values")
        selected_frame = (
            LocalFrame(float(np.mean(lat_array)), float(np.mean(lon_array)))
            if frame is None
            else frame
        )
        nodes = selected_frame.to_enu(lon=lon_array, lat=lat_array, alt=-depth_array)
        return cls.from_nodes(nodes, triangles, frame=selected_frame)

    def to_frame(self, frame: LocalFrame) -> TriGeometry:
        """Return vertices explicitly re-expressed in ``frame``."""
        if self.frame.is_compatible(frame):
            return self
        transformed = self.frame.transform_enu(self.vertices_enu, target=frame)
        return TriGeometry(transformed, frame)


def _triangle_strike_dip(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derive right-hand-rule strike and dip from triangle vertices."""
    edge1 = vertices[:, 1, :] - vertices[:, 0, :]
    edge2 = vertices[:, 2, :] - vertices[:, 0, :]
    normal = np.cross(edge1, edge2)
    normal[normal[:, 2] < 0.0] *= -1.0
    magnitude = np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    normal /= magnitude

    dip = np.degrees(np.arccos(np.clip(np.abs(normal[:, 2]), 0.0, 1.0)))
    strike = np.zeros(vertices.shape[0], dtype=float)
    dipping = dip > 0.1
    updip_azimuth = (
        np.degrees(np.arctan2(normal[dipping, 0], normal[dipping, 1])) % 360.0
    )
    strike[dipping] = (updip_azimuth + 90.0) % 360.0
    return strike, dip


def _resolve_planar_geometry(
    value: npt.ArrayLike | PlanarGeometry,
    *,
    frame: LocalFrame | None = None,
    ref_lat: float | None = None,
    ref_lon: float | None = None,
) -> PlanarGeometry:
    """Resolve named or legacy planar geometry with strict frame checks."""
    if (ref_lat is None) != (ref_lon is None):
        raise ValueError("ref_lat and ref_lon must be provided together")
    legacy_frame = (
        LocalFrame(ref_lat, ref_lon)
        if ref_lat is not None and ref_lon is not None
        else None
    )
    if frame is not None and legacy_frame is not None:
        frame.require_compatible(legacy_frame)
    selected_frame = frame if frame is not None else legacy_frame

    if isinstance(value, PlanarGeometry):
        if selected_frame is not None:
            value.frame.require_compatible(selected_frame)
        return value
    if selected_frame is None:
        raise ValueError(
            "array geometry requires frame or both ref_lat and ref_lon; "
            "prefer passing PlanarGeometry"
        )
    return PlanarGeometry.from_theta(value, frame=selected_frame)
