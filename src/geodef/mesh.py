"""Triangular mesh generation and manipulation for fault geometries.

Provides the ``Mesh`` dataclass for representing triangular meshes in
geographic coordinates, with factory functions for creating meshes from
surface traces, polygons, scattered points, and slab2.0 grids.

Requires optional dependencies for mesh generation:
- ``meshpy`` for triangle meshing (``pip install meshpy``)
- ``netCDF4`` for slab2.0 grid loading (``pip install netCDF4``)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import interpolate

from geodef import transforms

logger = logging.getLogger(__name__)


# ======================================================================
# Mesh dataclass
# ======================================================================


@dataclass(frozen=True)
class Mesh:
    """Immutable triangular mesh in geographic coordinates.

    Args:
        lon: Node longitudes, shape (N,).
        lat: Node latitudes, shape (N,).
        depth: Node depths in meters (positive down), shape (N,).
        triangles: Triangle connectivity as indices into nodes, shape (M, 3).
    """

    lon: np.ndarray
    lat: np.ndarray
    depth: np.ndarray
    triangles: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "lon", np.asarray(self.lon, dtype=float))
        object.__setattr__(self, "lat", np.asarray(self.lat, dtype=float))
        object.__setattr__(self, "depth", np.asarray(self.depth, dtype=float))
        object.__setattr__(
            self, "triangles", np.asarray(self.triangles, dtype=int)
        )

    @property
    def n_nodes(self) -> int:
        """Number of mesh nodes."""
        return len(self.lon)

    @property
    def n_triangles(self) -> int:
        """Number of triangles."""
        return len(self.triangles)

    @property
    def centers_geo(self) -> np.ndarray:
        """Triangle centroids as (M, 3) array of [lon, lat, depth]."""
        tri = self.triangles
        clon = np.mean(self.lon[tri], axis=1)
        clat = np.mean(self.lat[tri], axis=1)
        cdep = np.mean(self.depth[tri], axis=1)
        return np.column_stack([clon, clat, cdep])

    @property
    def areas(self) -> np.ndarray:
        """Triangle areas in m^2, shape (M,)."""
        ref_lat = float(np.mean(self.lat))
        ref_lon = float(np.mean(self.lon))
        verts = self.vertices_enu(ref_lat, ref_lon)
        edge1 = verts[:, 1, :] - verts[:, 0, :]
        edge2 = verts[:, 2, :] - verts[:, 0, :]
        return 0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1)

    def vertices_enu(
        self, ref_lat: float, ref_lon: float
    ) -> np.ndarray:
        """Triangle vertices in local ENU meters, shape (M, 3, 3).

        Each triangle has 3 vertices, each with [east, north, up] coordinates
        relative to the reference point. Depth is converted to up (z = -depth).

        Args:
            ref_lat: Reference latitude for ENU origin.
            ref_lon: Reference longitude for ENU origin.

        Returns:
            Array of shape (M, 3, 3) suitable for ``Fault.__init__(vertices=...)``.
        """
        e, n, u = transforms.geod2enu(
            self.lat, self.lon, -self.depth,
            ref_lat, ref_lon, 0.0,
        )
        tri = self.triangles
        verts = np.empty((self.n_triangles, 3, 3), dtype=float)
        for k in range(3):
            verts[:, k, 0] = e[tri[:, k]]
            verts[:, k, 1] = n[tri[:, k]]
            verts[:, k, 2] = u[tri[:, k]]
        return verts

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save(self, fname: str, format: str = "ned") -> None:
        """Save mesh to files.

        Args:
            fname: Base filename (without extension).
            format: File format. Currently only ``"ned"`` (unicycle
                ``.ned`` + ``.tri`` files) is supported.

        The ``.ned`` file stores nodes as ``index lat lon depth`` (1-indexed).
        The ``.tri`` file stores ``index v1 v2 v3 rake`` (1-indexed).
        """
        if format != "ned":
            raise ValueError(
                f"Unsupported format {format!r}; only 'ned' is supported"
            )
        ned_path = Path(f"{fname}.ned")
        tri_path = Path(f"{fname}.tri")

        with open(ned_path, "w") as f:
            for i in range(self.n_nodes):
                f.write(
                    f"{i + 1} {self.lat[i]:.8f} {self.lon[i]:.8f}"
                    f" {self.depth[i]:.4f}\n"
                )

        with open(tri_path, "w") as f:
            for i in range(self.n_triangles):
                v = self.triangles[i] + 1  # 1-indexed
                f.write(f"{i + 1} {v[0]} {v[1]} {v[2]} 90.00\n")

    @classmethod
    def load(cls, fname: str, format: str = "ned") -> Mesh:
        """Load mesh from files.

        Args:
            fname: Base filename (without extension).
            format: File format. Currently only ``"ned"`` is supported.

        Returns:
            Loaded ``Mesh`` instance.

        Raises:
            FileNotFoundError: If the required files do not exist.
            ValueError: If the format is not supported.
        """
        if format != "ned":
            raise ValueError(
                f"Unsupported format {format!r}; only 'ned' is supported"
            )
        ned_path = Path(f"{fname}.ned")
        tri_path = Path(f"{fname}.tri")

        if not ned_path.exists():
            raise FileNotFoundError(f"Node file not found: {ned_path}")
        if not tri_path.exists():
            raise FileNotFoundError(f"Triangle file not found: {tri_path}")

        # Parse .ned: index lat lon depth
        ned_data = np.loadtxt(ned_path, comments="#")
        lat = ned_data[:, 1]
        lon = ned_data[:, 2]
        depth = ned_data[:, 3]

        # Parse .tri: index v1 v2 v3 rake
        tri_data = np.loadtxt(tri_path, comments="#")
        triangles = tri_data[:, 1:4].astype(int) - 1  # 0-indexed

        return cls(lon=lon, lat=lat, depth=depth, triangles=triangles)


# ======================================================================
# Internal helpers
# ======================================================================


def _compute_strike_dip(
    vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive strike and dip angles from triangle vertices in local ENU.

    Args:
        vertices: Triangle vertices, shape (N, 3, 3), in local ENU meters
            where columns are [east, north, up].

    Returns:
        strike: Strike angles in degrees [0, 360), shape (N,).
            Measured clockwise from north, with the fault dipping to
            the right when looking along strike.
        dip: Dip angles in degrees [0, 90], shape (N,).
    """
    edge1 = vertices[:, 1, :] - vertices[:, 0, :]
    edge2 = vertices[:, 2, :] - vertices[:, 0, :]
    normal = np.cross(edge1, edge2)

    # Ensure normal points upward (positive z component)
    flip = normal[:, 2] < 0
    normal[flip] *= -1

    norm_mag = np.linalg.norm(normal, axis=1, keepdims=True)
    norm_mag = np.maximum(norm_mag, 1e-30)
    normal = normal / norm_mag

    # Dip = angle between normal and vertical (z-axis)
    dip = np.degrees(np.arccos(np.clip(np.abs(normal[:, 2]), 0, 1)))

    # For dip ≈ 0 (horizontal), strike is undefined; default to 0
    strike = np.zeros(len(vertices), dtype=float)

    not_horizontal = dip > 0.1
    if np.any(not_horizontal):
        # Updip direction: steepest ascent projected to horizontal
        # For a plane with upward normal (nx, ny, nz), the horizontal
        # component of the updip direction is (-nx, -ny) (opposite of
        # the horizontal projection of the normal for upward-pointing normals
        # is actually the downdip direction, so updip = horizontal normal dir)
        # Actually: the updip direction on the plane is the direction of
        # steepest ascent. The gradient of z on the plane points updip.
        # Horizontal updip = (nx, ny) / sqrt(nx^2 + ny^2) when normal points up.
        nx = normal[not_horizontal, 0]
        ny = normal[not_horizontal, 1]

        # Updip horizontal azimuth (degrees from north, CW positive)
        updip_az = np.degrees(np.arctan2(nx, ny)) % 360

        # Strike = updip azimuth + 90 (right-hand rule: dip to the right)
        strike[not_horizontal] = (updip_az + 90) % 360

    return strike, dip


def _nan_aware_griddata(
    xin: np.ndarray,
    yin: np.ndarray,
    zin: np.ndarray,
    xout: np.ndarray,
    yout: np.ndarray,
) -> np.ndarray:
    """Interpolate scattered data with NaN handling.

    Uses cubic interpolation first, then fills remaining NaN values
    (typically at convex hull edges) with nearest-neighbor interpolation.

    Args:
        xin: Input x coordinates, shape (K,).
        yin: Input y coordinates, shape (K,).
        zin: Input z values, shape (K,). May contain NaNs.
        xout: Output x coordinates.
        yout: Output y coordinates.

    Returns:
        Interpolated z values at output points, with no NaNs.
    """
    valid = ~np.isnan(zin)
    points = np.column_stack([xin[valid], yin[valid]])
    values = zin[valid]

    zout = interpolate.griddata(points, values, (xout, yout), method="cubic")

    nans = np.isnan(zout)
    if np.any(nans):
        zout[nans] = interpolate.griddata(
            points, values, (xout[nans], yout[nans]), method="nearest"
        )

    return zout


def _snap_boundary_nodes(
    mesh_pts_2d: np.ndarray,
    mesh_pts_3d: np.ndarray,
    boundary_2d: np.ndarray,
    boundary_3d: np.ndarray,
    boundary_depth: np.ndarray,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Snap mesh nodes on boundary edges to exact 3D boundary positions.

    After PCA-based meshing, Steiner points added by meshpy on boundary
    edges land on the best-fit plane rather than the true fault surface.
    This function detects such nodes and interpolates their 3D position
    along the original boundary edge, ensuring exact geometry (especially
    depth = 0 for surface trace edges).

    Args:
        mesh_pts_2d: Mesh node positions in 2D PCA space, shape (N, 2).
        mesh_pts_3d: Mesh node positions in 3D ENU (from unproject),
            shape (N, 3).
        boundary_2d: Original boundary in 2D PCA space, shape (K, 2).
        boundary_3d: Original boundary in 3D ENU, shape (K, 3).
        boundary_depth: Original boundary depths in meters (positive down),
            shape (K,). Used to interpolate exact depth for boundary nodes,
            avoiding ellipsoid curvature artifacts.
        tolerance: Distance threshold (relative to bounding box diagonal)
            for detecting boundary nodes.

    Returns:
        snapped_3d: Corrected 3D ENU positions, shape (N, 3).
        snapped_depth: Corrected depth values, shape (N,). For interior
            nodes, depth is ``-pts_3d[:, 2]``; for boundary nodes, depth
            is interpolated from the original boundary depths.
    """
    snapped = mesh_pts_3d.copy()
    snapped_depth = -mesh_pts_3d[:, 2].copy()  # default: ENU up → depth
    n_boundary = len(boundary_2d)

    # Scale tolerance relative to boundary size
    bbox_diag = np.sqrt(
        np.ptp(boundary_2d[:, 0]) ** 2 + np.ptp(boundary_2d[:, 1]) ** 2
    )
    abs_tol = tolerance * bbox_diag

    for i in range(len(mesh_pts_2d)):
        pt = mesh_pts_2d[i]
        best_dist = np.inf
        best_pos = None
        best_depth = None

        for j in range(n_boundary):
            j_next = (j + 1) % n_boundary
            a2 = boundary_2d[j]
            b2 = boundary_2d[j_next]

            # Project pt onto segment a→b, get parameter t and distance
            ab = b2 - a2
            ab_len2 = np.dot(ab, ab)
            if ab_len2 < 1e-30:
                continue
            t = np.dot(pt - a2, ab) / ab_len2
            t = np.clip(t, 0.0, 1.0)
            closest = a2 + t * ab
            dist = np.linalg.norm(pt - closest)

            if dist < best_dist:
                best_dist = dist
                if dist < abs_tol:
                    # Interpolate along the original 3D boundary edge
                    a3 = boundary_3d[j]
                    b3 = boundary_3d[j_next]
                    best_pos = (1.0 - t) * a3 + t * b3
                    # Interpolate depth from original boundary depths
                    best_depth = (
                        (1.0 - t) * boundary_depth[j]
                        + t * boundary_depth[j_next]
                    )

        if best_pos is not None:
            snapped[i] = best_pos
            snapped_depth[i] = best_depth

    return snapped, snapped_depth


def _trace_grid_boundary(
    X: np.ndarray,
    Y: np.ndarray,
    valid: np.ndarray,
    subsample: int = 1,
) -> np.ndarray:
    """Trace the boundary of a non-NaN region on a regular grid.

    Identifies valid cells adjacent to invalid (or off-grid) cells, then
    walks the boundary using Moore neighborhood tracing (8-connected) to
    produce an ordered polygon. This correctly handles concave slab
    geometries where a convex hull would include regions outside the data.

    Args:
        X: Longitude grid, shape (nrows, ncols).
        Y: Latitude grid, shape (nrows, ncols).
        valid: Boolean mask, shape (nrows, ncols). True = valid data.
        subsample: Keep every nth boundary point (default 1 = all).

    Returns:
        Boundary polygon as (K, 2) array of [lon, lat] vertices, ordered
        as a closed contour.

    Raises:
        ValueError: If the valid region is empty.
    """
    # Find boundary cells: valid with at least one 4-neighbor invalid/off-grid
    padded = np.pad(valid, 1, constant_values=False)
    neighbor_invalid = (
        ~padded[:-2, 1:-1]
        | ~padded[2:, 1:-1]
        | ~padded[1:-1, :-2]
        | ~padded[1:-1, 2:]
    )
    boundary_mask = valid & neighbor_invalid
    boundary_coords = np.argwhere(boundary_mask)

    if len(boundary_coords) == 0:
        raise ValueError("No boundary found -- valid region may be empty")

    boundary_set = set(map(tuple, boundary_coords))

    # Start from topmost-then-leftmost boundary cell
    start_idx = np.lexsort((boundary_coords[:, 1], boundary_coords[:, 0]))[0]
    start = tuple(boundary_coords[start_idx])

    # Moore neighborhood tracing (clockwise).
    # Directions indexed 0-7: E, SE, S, SW, W, NW, N, NE
    dirs = [(0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1)]

    ordered = [start]
    current = start
    # We entered from above (the start is the topmost cell), so the
    # "backtrack" direction is N (index 6). Moore tracing searches
    # clockwise starting from the cell after the backtrack direction.
    backtrack = 6

    for _ in range(len(boundary_set) + 1):
        search_start = (backtrack + 1) % 8
        moved = False
        for offset in range(8):
            di = (search_start + offset) % 8
            dr, dc = dirs[di]
            nb = (current[0] + dr, current[1] + dc)

            if nb not in boundary_set:
                continue

            if nb == start and len(ordered) > 2:
                # Closed the loop
                moved = True
                current = nb
                break

            if nb != start:
                ordered.append(nb)
                current = nb
                # Backtrack direction = opposite of the step we just took
                backtrack = (di + 4) % 8
                moved = True
                break

        if not moved or current == start:
            break

    rows = np.array([rc[0] for rc in ordered])
    cols = np.array([rc[1] for rc in ordered])
    boundary_lon = X[rows, cols]
    boundary_lat = Y[rows, cols]

    if subsample > 1 and len(boundary_lon) > subsample:
        indices = np.arange(0, len(boundary_lon), subsample)
        if indices[-1] != len(boundary_lon) - 1:
            indices = np.append(indices, len(boundary_lon) - 1)
        boundary_lon = boundary_lon[indices]
        boundary_lat = boundary_lat[indices]

    return np.column_stack([boundary_lon, boundary_lat])


def _simplify_boundary(
    boundary_xy: np.ndarray,
    spacing_func: "Callable[[float, float], float]",
) -> np.ndarray:
    """Thin a boundary polygon so point spacing matches a target function.

    Walks the boundary and keeps a point only when the distance from the
    last kept point exceeds the local target spacing evaluated at the
    midpoint of the segment.

    Args:
        boundary_xy: Ordered boundary polygon, shape (K, 2) of [lon, lat].
        spacing_func: Callable ``f(lon, lat) -> spacing`` returning the
            target point spacing (in the same units as boundary_xy) at a
            given location.

    Returns:
        Simplified boundary polygon, shape (K', 2) with K' <= K.
    """
    keep = [0]
    last_kept = 0

    for i in range(1, len(boundary_xy)):
        mid = 0.5 * (boundary_xy[last_kept] + boundary_xy[i])
        target = spacing_func(mid[0], mid[1])

        dx = boundary_xy[i, 0] - boundary_xy[last_kept, 0]
        dy = boundary_xy[i, 1] - boundary_xy[last_kept, 1]
        dist = np.sqrt(dx**2 + dy**2)

        if dist >= target:
            keep.append(i)
            last_kept = i

    if keep[-1] != len(boundary_xy) - 1:
        keep.append(len(boundary_xy) - 1)

    return boundary_xy[np.array(keep)]


def _polygon_to_facets(n: int) -> list[tuple[int, int]]:
    """Create closed polygon edge connectivity for meshpy.

    Args:
        n: Number of polygon vertices.

    Returns:
        List of (i, j) edge tuples forming a closed polygon.
    """
    return [(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)]


def _fix_vertical_edges(
    x: np.ndarray,
    y: np.ndarray,
    tolerance: float = 1e-8,
    offset: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Adjust polygon vertices to avoid vertical (zero-dx) segments.

    meshpy can fail on exactly vertical polygon edges. This nudges
    vertices by a tiny offset to break exact verticality.

    Args:
        x: Polygon x-coordinates, shape (N,).
        y: Polygon y-coordinates, shape (N,).
        tolerance: Threshold for detecting vertical segments.
        offset: Amount to shift x-coordinate.

    Returns:
        Adjusted (x, y) arrays.
    """
    x_fixed = np.array(x, dtype=float)
    n = len(x_fixed)
    for i in range(n):
        j = (i + 1) % n
        if abs(x_fixed[j] - x_fixed[i]) < tolerance:
            x_fixed[j] = x_fixed[i] + offset
            logger.debug(
                "Adjusted vertex %d x from %.8f to %.8f to avoid "
                "vertical segment",
                j, x[j], x_fixed[j],
            )
    return x_fixed, np.asarray(y, dtype=float)


def _mesh_polygon_2d(
    boundary_xy: np.ndarray,
    max_area: float | None = None,
    max_refinements: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Mesh a 2D polygon interior using meshpy.triangle.

    Args:
        boundary_xy: Polygon vertices, shape (K, 2).
        max_area: Maximum triangle area in the meshing coordinate system.
            If None, uses meshpy defaults (no refinement constraint).
        max_refinements: Safety limit on refinement iterations.

    Returns:
        points: Mesh node coordinates, shape (N, 2).
        triangles: Triangle connectivity, shape (M, 3).
    """
    tri_mod = _require_meshpy()

    x_fixed, y_fixed = _fix_vertical_edges(
        boundary_xy[:, 0], boundary_xy[:, 1]
    )
    points = list(zip(x_fixed, y_fixed))
    facets = _polygon_to_facets(len(points))

    info = tri_mod.MeshInfo()
    info.set_points(points)
    info.set_facets(facets)

    if max_area is not None:
        count = [0]

        def refinement(vertices, area):
            count[0] += 1
            if count[0] > max_refinements:
                logger.warning("Maximum refinements (%d) reached", max_refinements)
                return False
            return area > max_area

        mesh = tri_mod.build(info, refinement_func=refinement)
    else:
        mesh = tri_mod.build(info)

    return np.array(mesh.points), np.array(mesh.elements)


def _project_to_plane(
    points_3d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D points onto their best-fit 2D plane via PCA.

    Args:
        points_3d: Points in 3D, shape (N, 3).

    Returns:
        points_2d: Projected coordinates, shape (N, 2).
        basis: Orthonormal basis vectors, shape (2, 3).
        origin: Mean of the input points, shape (3,).
    """
    origin = np.mean(points_3d, axis=0)
    centered = points_3d - origin
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2]  # first two principal components
    points_2d = centered @ basis.T
    return points_2d, basis, origin


def _unproject_from_plane(
    points_2d: np.ndarray,
    basis: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    """Convert 2D projected coordinates back to 3D.

    Args:
        points_2d: Coordinates in the 2D plane, shape (N, 2).
        basis: Orthonormal basis, shape (2, 3).
        origin: 3D origin point, shape (3,).

    Returns:
        Points in 3D, shape (N, 3).
    """
    return points_2d @ basis + origin


# ======================================================================
# Public factory functions
# ======================================================================


def from_polygon(
    lon: np.ndarray,
    lat: np.ndarray,
    depth: np.ndarray | None = None,
    *,
    depth_func: "Callable | None" = None,
    target_length: float | None = None,
    max_area: float | None = None,
    max_refinements: int = 100_000,
) -> Mesh:
    """Create a triangular mesh from a polygon boundary.

    Supports two modes:

    1. **3D polygon**: provide ``lon``, ``lat``, and ``depth`` arrays for all
       boundary vertices. The polygon is projected onto its best-fit 2D plane
       for meshing, then interior node depths are interpolated from the
       boundary points.

    2. **2D polygon + depth function**: provide ``lon`` and ``lat`` only, plus
       a ``depth_func(lon, lat) -> depth`` callable. Meshing occurs in the
       lon/lat plane and depth is assigned via the callback.

    Args:
        lon: Polygon vertex longitudes, shape (K,).
        lat: Polygon vertex latitudes, shape (K,).
        depth: Polygon vertex depths in meters (positive down), shape (K,).
            Required for 3D polygon mode.
        depth_func: Callable ``f(lon, lat) -> depth`` for 2D polygon mode.
            Mutually exclusive with ``depth``.
        target_length: Target triangle edge length in meters. Converted
            to an equivalent area constraint internally.
        max_area: Maximum triangle area in the meshing coordinate system.
            Mutually exclusive with ``target_length``.
        max_refinements: Safety limit on refinement iterations.

    Returns:
        A ``Mesh`` with the polygon interior triangulated.

    Raises:
        ValueError: If neither ``depth`` nor ``depth_func`` is provided,
            or if ``target_length`` and ``max_area`` are both given.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    if target_length is not None and max_area is not None:
        raise ValueError("target_length and max_area are mutually exclusive")

    if depth is None and depth_func is None:
        raise ValueError(
            "Either depth array or depth_func must be provided"
        )

    if depth is not None and depth_func is not None:
        raise ValueError(
            "Provide either depth or depth_func, not both"
        )

    if depth is not None:
        depth = np.asarray(depth, dtype=float)
        return _from_polygon_3d(
            lon, lat, depth, target_length, max_area, max_refinements
        )
    else:
        return _from_polygon_2d(
            lon, lat, depth_func, target_length, max_area, max_refinements
        )


def _from_polygon_3d(
    lon: np.ndarray,
    lat: np.ndarray,
    depth: np.ndarray,
    target_length: float | None,
    max_area: float | None,
    max_refinements: int,
) -> Mesh:
    """Mesh a 3D polygon by projecting to its best-fit plane.

    After meshing, boundary nodes are snapped back to their exact 3D
    positions along the original polygon edges, ensuring that surface
    trace nodes remain at depth = 0.
    """
    ref_lat = float(np.mean(lat))
    ref_lon = float(np.mean(lon))

    # Convert boundary to local ENU
    e, n, u = transforms.geod2enu(lat, lon, -depth, ref_lat, ref_lon, 0.0)
    boundary_3d = np.column_stack([e, n, u])

    # Project to best-fit 2D plane
    boundary_2d, basis, origin = _project_to_plane(boundary_3d)

    # Convert target_length to max_area in the 2D plane coordinate system
    area = _resolve_area(target_length, max_area, boundary_2d)

    # Mesh in 2D
    pts_2d, tris = _mesh_polygon_2d(boundary_2d, max_area=area, max_refinements=max_refinements)

    # Unproject to 3D ENU
    pts_3d = _unproject_from_plane(pts_2d, basis, origin)

    # Snap boundary nodes to exact 3D positions along original edges
    pts_3d, out_depth = _snap_boundary_nodes(
        pts_2d, pts_3d, boundary_2d, boundary_3d, depth
    )

    # Convert back to geographic
    out_lat, out_lon, _ = transforms.enu2geod(
        pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2],
        ref_lat, ref_lon, 0.0,
    )

    return Mesh(lon=out_lon, lat=out_lat, depth=out_depth, triangles=tris)


def _from_polygon_2d(
    lon: np.ndarray,
    lat: np.ndarray,
    depth_func: "Callable",
    target_length: float | None,
    max_area: float | None,
    max_refinements: int,
) -> Mesh:
    """Mesh a 2D lon/lat polygon, assigning depth via callback."""
    boundary_xy = np.column_stack([lon, lat])

    # For target_length, convert meters to approximate degrees
    if target_length is not None:
        deg_per_m = 1.0 / 111_000.0
        length_deg = target_length * deg_per_m
        area = (np.sqrt(3) / 4) * length_deg**2
    else:
        area = max_area

    pts_2d, tris = _mesh_polygon_2d(
        boundary_xy, max_area=area, max_refinements=max_refinements
    )

    out_lon = pts_2d[:, 0]
    out_lat = pts_2d[:, 1]
    out_depth = np.asarray(depth_func(out_lon, out_lat), dtype=float)

    return Mesh(lon=out_lon, lat=out_lat, depth=out_depth, triangles=tris)


def _resolve_area(
    target_length: float | None,
    max_area: float | None,
    boundary_2d: np.ndarray,
) -> float | None:
    """Convert target_length/max_area to an area constraint in 2D coords.

    Args:
        target_length: Target edge length in meters.
        max_area: Direct area threshold (in 2D coord units).
        boundary_2d: 2D boundary points for computing scale.

    Returns:
        Area threshold, or None for no constraint.
    """
    if target_length is not None:
        # target_length is in meters; boundary_2d is already in meters (ENU)
        return (np.sqrt(3) / 4) * target_length**2
    return max_area


def from_trace(
    trace_lon: np.ndarray,
    trace_lat: np.ndarray,
    max_depth: float,
    dip: float | "Callable[[float], float]",
    *,
    dip_direction: float | None = None,
    n_downdip: int = 20,
    target_length: float | None = None,
    max_area: float | None = None,
    max_refinements: int = 100_000,
) -> Mesh:
    """Create a triangular mesh from a surface trace and dip specification.

    Meshes in a natural (along-strike, depth) coordinate system, then maps
    each node back to 3D. This guarantees that all surface nodes lie exactly
    on the interpolated trace at depth = 0.

    Args:
        trace_lon: Surface trace longitudes, shape (K,).
        trace_lat: Surface trace latitudes, shape (K,).
        max_depth: Maximum fault depth in meters (positive down).
        dip: Dip angle. Either a scalar (constant dip in degrees) or a
            callable ``dip(depth_m) -> angle_deg`` for variable (listric)
            geometry. Users with array data can use
            ``lambda z: np.interp(z, depths, dips)``.
        dip_direction: Azimuth of the dip direction in degrees from north.
            If None, inferred as strike + 90 (right-hand rule).
        n_downdip: Number of depth steps for projecting the down-dip edge.
            Controls polygon resolution along the sides.
        target_length: Target triangle edge length in meters.
        max_area: Maximum triangle area in m^2. Mutually exclusive with
            ``target_length``.
        max_refinements: Safety limit on refinement iterations.

    Returns:
        A ``Mesh`` covering the fault surface from trace to ``max_depth``.
    """
    trace_lon = np.asarray(trace_lon, dtype=float)
    trace_lat = np.asarray(trace_lat, dtype=float)

    if target_length is not None and max_area is not None:
        raise ValueError("target_length and max_area are mutually exclusive")

    ref_lat = float(np.mean(trace_lat))
    ref_lon = float(np.mean(trace_lon))

    # Convert trace to local ENU
    e, n, u = transforms.geod2enu(
        trace_lat, trace_lon, np.zeros_like(trace_lat),
        ref_lat, ref_lon, 0.0,
    )
    trace_e = np.asarray(e, dtype=float)
    trace_n = np.asarray(n, dtype=float)

    # Cumulative along-strike distance
    de = np.diff(trace_e)
    dn = np.diff(trace_n)
    seg_lengths = np.sqrt(de**2 + dn**2)
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    trace_length = cum_dist[-1]

    # Compute average strike from trace segments
    if len(trace_lon) >= 2:
        seg_azimuths = np.degrees(np.arctan2(de, dn)) % 360
        avg_strike = float(np.mean(seg_azimuths))
    else:
        avg_strike = 0.0

    # Dip direction
    if dip_direction is None:
        dip_direction = (avg_strike + 90) % 360

    dip_dir_rad = np.radians(dip_direction)
    dip_e = np.sin(dip_dir_rad)
    dip_n = np.cos(dip_dir_rad)

    # Compute down-dip horizontal offset profile
    is_callable = callable(dip)
    depths = np.linspace(0, max_depth, n_downdip + 1)
    horiz_offsets = np.zeros(n_downdip + 1)
    for i in range(n_downdip):
        z_mid = 0.5 * (depths[i] + depths[i + 1])
        dz = depths[i + 1] - depths[i]
        dip_angle = float(dip(z_mid)) if is_callable else float(dip)
        dip_rad = np.radians(dip_angle)
        if abs(np.tan(dip_rad)) > 1e-10:
            horiz_offsets[i + 1] = horiz_offsets[i] + dz / np.tan(dip_rad)
        else:
            horiz_offsets[i + 1] = horiz_offsets[i]

    # Compute cumulative down-dip distance (arc length along the dip profile)
    dd = np.diff(depths)
    dh = np.diff(horiz_offsets)
    downdip_segs = np.sqrt(dd**2 + dh**2)
    cum_downdip = np.concatenate([[0.0], np.cumsum(downdip_segs)])
    downdip_length = cum_downdip[-1]

    # Meshing domain: rectangle [0, trace_length] × [0, downdip_length]
    rect = np.array([
        [0.0, 0.0],
        [trace_length, 0.0],
        [trace_length, downdip_length],
        [0.0, downdip_length],
    ])

    # Convert target_length to max_area in the rectangle coordinate system
    if target_length is not None:
        area = (np.sqrt(3) / 4) * target_length**2
    else:
        area = max_area

    pts_2d, tris = _mesh_polygon_2d(
        rect, max_area=area, max_refinements=max_refinements
    )

    # Map each (s, d) mesh node to 3D ENU
    s_vals = pts_2d[:, 0]  # along-strike distance
    d_vals = pts_2d[:, 1]  # down-dip distance

    # Interpolate trace position at each along-strike distance
    out_e = np.interp(s_vals, cum_dist, trace_e)
    out_n = np.interp(s_vals, cum_dist, trace_n)

    # Interpolate depth and horizontal offset at each down-dip distance
    out_depth = np.interp(d_vals, cum_downdip, depths)
    out_horiz = np.interp(d_vals, cum_downdip, horiz_offsets)

    # Add dip-direction horizontal offset
    out_e = out_e + out_horiz * dip_e
    out_n = out_n + out_horiz * dip_n
    out_u = -out_depth

    # Convert to geographic (use out_depth directly to avoid ellipsoid
    # curvature artifacts in the enu2geod round-trip)
    out_lat, out_lon, _ = transforms.enu2geod(
        out_e, out_n, out_u, ref_lat, ref_lon, 0.0,
    )

    return Mesh(
        lon=out_lon, lat=out_lat, depth=out_depth,
        triangles=tris,
    )


def _km_to_deg(km: float, lat: float) -> float:
    """Convert a distance in km to approximate degrees at a given latitude.

    Uses the mean of the longitude and latitude scale factors so that the
    result is reasonable for both directions.

    Args:
        km: Distance in kilometers.
        lat: Representative latitude in degrees.

    Returns:
        Approximate equivalent in degrees.
    """
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat))
    km_per_deg = 0.5 * (km_per_deg_lat + km_per_deg_lon)
    return km / km_per_deg


def from_slab2(
    fname: str,
    bounds: tuple[float, float, float, float],
    *,
    target_length: float = 50.0,
    depth_growth: float = 1.0,
    boundary_subsample: int = 1,
    max_refinements: int = 100_000,
) -> Mesh:
    """Create a triangular mesh from a slab2.0 NetCDF depth grid.

    Loads the slab2.0 grid, crops to the specified bounds, traces the
    concave boundary of the non-NaN region, meshes the interior, and
    interpolates depth from the grid onto mesh nodes.

    Args:
        fname: Path to slab2.0 ``.grd`` file (NetCDF format).
        bounds: ``(lon_min, lon_max, lat_min, lat_max)`` cropping region.
        target_length: Target triangle edge length in km at the shallowest
            point. Default 50 km.
        depth_growth: Ratio of edge length at the deepest point to edge
            length at the shallowest. Values > 1 produce coarser triangles
            at depth (e.g. 2.0 means twice as coarse). Default 1.0
            (uniform).
        boundary_subsample: Keep every nth boundary point when tracing the
            grid boundary. Increase to simplify the polygon (e.g. 3 or 5).
            Default 1 (all boundary points).
        max_refinements: Safety limit on refinement iterations.

    Returns:
        A ``Mesh`` with node depths interpolated from the slab2.0 grid.

    Raises:
        ImportError: If ``netCDF4`` is not installed.
        ValueError: If ``depth_growth`` is less than 1.
    """
    if depth_growth < 1.0:
        raise ValueError(
            f"depth_growth must be >= 1.0, got {depth_growth}"
        )

    NCDataset = _require_netcdf4()
    _require_meshpy()

    # Load and crop
    with NCDataset(fname, mode="r") as fh:
        lons = np.asarray(fh.variables["x"][:])
        lats = np.asarray(fh.variables["y"][:])
        Z = np.asarray(fh.variables["z"][:])

    X, Y = np.meshgrid(lons, lats)

    lon_min, lon_max, lat_min, lat_max = bounds
    row_mask = (lats > lat_min) & (lats <= lat_max)
    col_mask = (lons > lon_min) & (lons <= lon_max)
    idx = np.ix_(row_mask, col_mask)
    Xc, Yc, Zc = X[idx], Y[idx], Z[idx]

    # Offset so shallowest point is at zero depth
    Zc = Zc - np.nanmax(Zc)

    # Extract boundary from non-NaN region
    valid = ~np.isnan(Zc)
    valid_lons = Xc[valid]
    valid_lats = Yc[valid]
    valid_depths = -Zc[valid] * 1000  # km → m, flip sign (positive down)

    points_2d = np.column_stack([valid_lons, valid_lats])

    # Trace the concave boundary of the valid grid region
    boundary_xy = _trace_grid_boundary(
        Xc, Yc, valid, subsample=boundary_subsample
    )

    # Build depth interpolator and compute scale factors
    depth_interp = interpolate.LinearNDInterpolator(
        points_2d, valid_depths
    )
    max_depth = float(np.max(valid_depths))
    if max_depth < 1.0:
        max_depth = 1.0

    ref_lat = 0.5 * (lat_min + lat_max)
    base_length_deg = _km_to_deg(target_length, ref_lat)
    sqrt3_4 = np.sqrt(3.0) / 4.0
    base_area = sqrt3_4 * base_length_deg**2

    def _local_length_deg(lon: float, lat: float) -> float:
        """Target edge length in degrees at a given point."""
        d = depth_interp(lon, lat)
        if d is None or np.isnan(d):
            return base_length_deg
        frac = float(np.clip(d / max_depth, 0.0, 1.0))
        return base_length_deg * (1.0 + (depth_growth - 1.0) * frac)

    # Simplify boundary to match local target edge length
    boundary_xy = _simplify_boundary(boundary_xy, _local_length_deg)

    if depth_growth > 1.0:
        tri_mod = _require_meshpy()
        x_fixed, y_fixed = _fix_vertical_edges(
            boundary_xy[:, 0], boundary_xy[:, 1]
        )
        pts = list(zip(x_fixed, y_fixed))
        facets = _polygon_to_facets(len(pts))
        info = tri_mod.MeshInfo()
        info.set_points(pts)
        info.set_facets(facets)

        count = [0]

        def refinement(vertices, area):
            count[0] += 1
            if count[0] > max_refinements:
                logger.warning(
                    "Maximum refinements (%d) reached", max_refinements
                )
                return False
            bary = np.mean(vertices, axis=0)
            local_len = _local_length_deg(bary[0], bary[1])
            return area > sqrt3_4 * local_len**2

        mesh_result = tri_mod.build(info, refinement_func=refinement)
        mesh_pts = np.array(mesh_result.points)
        mesh_tris = np.array(mesh_result.elements)
    else:
        mesh_pts, mesh_tris = _mesh_polygon_2d(
            boundary_xy, max_area=base_area,
            max_refinements=max_refinements,
        )

    # Interpolate depth onto mesh nodes
    out_lon = mesh_pts[:, 0]
    out_lat = mesh_pts[:, 1]
    out_depth = _nan_aware_griddata(
        valid_lons, valid_lats, valid_depths, out_lon, out_lat
    )

    return Mesh(lon=out_lon, lat=out_lat, depth=out_depth, triangles=mesh_tris)


def from_points(
    lon: np.ndarray,
    lat: np.ndarray,
    depth: np.ndarray,
    *,
    boundary: np.ndarray | None = None,
    target_length: float | None = None,
    max_area: float | None = None,
    max_refinements: int = 100_000,
) -> Mesh:
    """Create a triangular mesh from scattered 3D points.

    Extracts a boundary polygon (convex hull or user-supplied), meshes the
    interior, and interpolates depth from the input points onto mesh nodes.

    Args:
        lon: Point longitudes, shape (K,).
        lat: Point latitudes, shape (K,).
        depth: Point depths in meters (positive down), shape (K,).
        boundary: Optional boundary polygon as (B, 2) array of
            ``[lon, lat]`` vertices. If None, the convex hull is used.
        target_length: Target triangle edge length in meters.
        max_area: Maximum triangle area. In meters^2 if ``boundary`` is
            provided (uses ENU projection), otherwise in degrees^2 (uses
            convex hull in lon/lat). Mutually exclusive with ``target_length``.
        max_refinements: Safety limit on refinement iterations.

    Returns:
        A ``Mesh`` with depths interpolated from the input points.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    depth = np.asarray(depth, dtype=float)

    if target_length is not None and max_area is not None:
        raise ValueError("target_length and max_area are mutually exclusive")

    ref_lat = float(np.mean(lat))
    ref_lon = float(np.mean(lon))

    # Convert to local ENU
    e, n, u = transforms.geod2enu(
        lat, lon, -depth, ref_lat, ref_lon, 0.0
    )
    pts_3d = np.column_stack([e, n, u])

    # Get or compute boundary
    if boundary is not None:
        boundary = np.asarray(boundary, dtype=float)
        boundary_depth_vals = np.zeros(len(boundary))
        b_e, b_n, b_u = transforms.geod2enu(
            boundary[:, 1], boundary[:, 0],
            np.zeros(len(boundary)),
            ref_lat, ref_lon, 0.0,
        )
        boundary_3d = np.column_stack([b_e, b_n, b_u])
    else:
        from scipy.spatial import ConvexHull

        pts_2d_ll = np.column_stack([e, n])
        hull = ConvexHull(pts_2d_ll)
        boundary_3d = pts_3d[hull.vertices]
        boundary_depth_vals = depth[hull.vertices]

    # Project to best-fit plane and mesh
    boundary_2d, basis, origin = _project_to_plane(boundary_3d)
    area = _resolve_area(target_length, max_area, boundary_2d)

    mesh_pts_2d, tris = _mesh_polygon_2d(
        boundary_2d, max_area=area, max_refinements=max_refinements
    )

    # Unproject to 3D ENU
    mesh_pts_3d = _unproject_from_plane(mesh_pts_2d, basis, origin)

    # Snap boundary nodes to exact 3D positions and depths
    mesh_pts_3d, mesh_depth = _snap_boundary_nodes(
        mesh_pts_2d, mesh_pts_3d, boundary_2d, boundary_3d,
        boundary_depth_vals,
    )

    # Override interior node depths with interpolation from input points
    # (boundary nodes keep their exact snapped depths)
    interp_depth = _nan_aware_griddata(
        e, n, depth, mesh_pts_3d[:, 0], mesh_pts_3d[:, 1]
    )
    # Identify interior nodes (those NOT on any boundary edge)
    n_bnd = len(boundary_3d)
    bbox_diag = np.sqrt(
        np.ptp(boundary_2d[:, 0]) ** 2 + np.ptp(boundary_2d[:, 1]) ** 2
    )
    abs_tol = 1e-6 * bbox_diag
    for i in range(len(mesh_pts_2d)):
        pt = mesh_pts_2d[i]
        on_boundary = False
        for j in range(n_bnd):
            j_next = (j + 1) % n_bnd
            a2 = boundary_2d[j]
            b2 = boundary_2d[j_next]
            ab = b2 - a2
            ab_len2 = np.dot(ab, ab)
            if ab_len2 < 1e-30:
                continue
            t = np.clip(np.dot(pt - a2, ab) / ab_len2, 0.0, 1.0)
            closest = a2 + t * ab
            if np.linalg.norm(pt - closest) < abs_tol:
                on_boundary = True
                break
        if not on_boundary:
            mesh_depth[i] = interp_depth[i]

    # Convert mesh nodes to geographic
    out_lat, out_lon, _ = transforms.enu2geod(
        mesh_pts_3d[:, 0], mesh_pts_3d[:, 1], -mesh_depth,
        ref_lat, ref_lon, 0.0,
    )

    return Mesh(lon=out_lon, lat=out_lat, depth=mesh_depth, triangles=tris)


def _require_meshpy():
    """Lazy import of meshpy.triangle with a clear error message."""
    try:
        import meshpy.triangle as tri
        return tri
    except ImportError:
        raise ImportError(
            "meshpy is required for mesh generation. "
            "Install it with: pip install meshpy"
        ) from None


def _require_netcdf4():
    """Lazy import of netCDF4 with a clear error message."""
    try:
        from netCDF4 import Dataset
        return Dataset
    except ImportError:
        raise ImportError(
            "netCDF4 is required for loading slab2.0 grids. "
            "Install it with: pip install netCDF4"
        ) from None
