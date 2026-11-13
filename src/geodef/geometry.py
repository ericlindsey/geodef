"""Fault-geometry array helpers: planar parameter vectors and triangles.

Coordinate frames and geodetic conversions live in :mod:`geodef.transforms`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from geodef.validation import check_finite_scalar, check_positive, check_range

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
    "PLANAR_PARAMETER_NAMES",
    "as_planar_vector",
    "planar_parameter_dict",
    "orient_triangles",
    "triangle_strike_dip",
    "vertices_from_nodes",
]


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


def orient_triangles(vertices_enu: npt.ArrayLike) -> np.ndarray:
    """Reorder triangle corners so every normal points up.

    The triangular dislocation kernel takes its strike and dip directions
    from the order of the corners. With corners counterclockwise seen from
    above (normal pointing up), its strike follows the right-hand rule and
    positive dip slip is thrust, matching rectangular patches. Listed the
    other way round, positive dip slip would be normal faulting. Vertical
    triangles, whose normal is horizontal, are left as given.

    Args:
        vertices_enu: Per-triangle corners ``[east, north, up]``, shape
            ``(N, 3, 3)``.

    Returns:
        A copy with the second and third corners swapped wherever the
        normal pointed down.
    """
    vertices = np.array(vertices_enu, dtype=float)
    normal = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    downward = normal[:, 2] < 0.0
    vertices[downward] = vertices[downward][:, [0, 2, 1]]
    return vertices


def triangle_strike_dip(
    vertices_enu: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive right-hand-rule strike and dip from triangle vertices.

    Strike is the direction the triangular kernel uses for strike slip once
    the triangle is oriented with :func:`orient_triangles`: horizontal, with
    the plane dipping to its right. A horizontal triangle has strike 0.

    Args:
        vertices_enu: Per-triangle vertices with shape ``(N, 3, 3)``.

    Returns:
        ``(strike_degrees, dip_degrees)`` arrays, each shape ``(N,)``; strike
        in ``[0, 360)`` clockwise from the frame's North axis.

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
    vertices = orient_triangles(vertices)
    normal = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    dip = np.degrees(np.arccos(np.clip(np.abs(normal[:, 2]), 0.0, 1.0)))
    # strike direction = vertical x normal = (-n_north, n_east, 0)
    strike_east, strike_north = -normal[:, 1], normal[:, 0]
    horizontal = np.hypot(strike_east, strike_north) < 1e-12
    strike = np.degrees(np.arctan2(strike_east, strike_north)) % 360.0
    strike[horizontal] = 0.0
    return strike, dip
