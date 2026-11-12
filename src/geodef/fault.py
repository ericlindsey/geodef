"""Fault geometry class for rectangular and triangular fault patches.

Provides the `Fault` class with factory classmethods for creating faults
from geometric parameters, files, or slab2.0 grids. Supports forward
modeling via Green's function matrices and seismic moment calculation.
"""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from geodef import _fault_io, transforms
from geodef import greens as _greens
from geodef._fault_io import _seg_to_patches as _seg_to_patches
from geodef.geometry import (
    LocalFrame,
    as_planar_vector,
    triangle_strike_dip,
    vertices_from_nodes,
)
from geodef.medium import DEFAULT_MEDIUM, ElasticMedium
from geodef.validation import (
    ValidationReport,
    _ReportBuilder,
    as_1d_floats,
    check_finite_scalar,
    check_positive,
    check_range,
)

if TYPE_CHECKING:
    from geodef.mesh import Mesh


class Fault:
    """Immutable collection of fault patches with forward-modeling methods.

    A fault is either a set of rectangular patches (Okada dislocations,
    ``engine == "okada"``) or a set of triangles (``engine == "tri"``).
    Create one with a factory classmethod rather than the constructor:

    - ``Fault.planar(...)``: a planar fault divided into a patch grid
    - ``Fault.planar_from_corner(...)``: the same, anchored at a corner
    - ``Fault.from_triangles(...)`` / ``Fault.from_mesh(...)``: triangles
    - ``Fault.load(fname)``: from a text file

    Geometry is stored in public read-only arrays: ``lon``, ``lat``,
    ``depth``, ``strike``, ``dip`` (one value per patch), plus ``length``
    and ``width`` for rectangles or ``vertices`` for triangles.

    Args:
        lon: Patch center longitudes in degrees, shape (N,).
        lat: Patch center latitudes in degrees, shape (N,).
        depth: Patch centroid depths in meters (positive down), shape (N,).
        strike: Strike angles in degrees, shape (N,).
        dip: Dip angles in degrees, shape (N,).
        length: Along-strike patch lengths in meters, shape (N,).
            Rectangular faults only.
        width: Down-dip patch widths in meters, shape (N,). Rectangular
            faults only.
        vertices: Triangle corners in local East-North-Up meters, shape
            (N, 3, 3). Giving vertices makes the fault triangular.
        grid_shape: ``(n_length, n_width)`` for structured rectangular grids.
        medium: Elastic half-space parameters used by Green's functions,
            stress kernels, and moment. Defaults to
            ``geodef.medium.DEFAULT_MEDIUM`` (30 GPa Poisson solid).
        frame: Local frame for local-coordinate views. Defaults to a frame
            centered on the mean patch location.
    """

    def __init__(
        self,
        *,
        lon: npt.ArrayLike,
        lat: npt.ArrayLike,
        depth: npt.ArrayLike,
        strike: npt.ArrayLike,
        dip: npt.ArrayLike,
        length: npt.ArrayLike | None = None,
        width: npt.ArrayLike | None = None,
        vertices: npt.ArrayLike | None = None,
        grid_shape: tuple[int, int] | None = None,
        medium: ElasticMedium | None = None,
        frame: LocalFrame | None = None,
    ) -> None:
        lon = as_1d_floats("lon", np.atleast_1d(lon), unit="degrees")
        n = lon.shape[0]
        lat = as_1d_floats("lat", np.atleast_1d(lat), n=n, unit="degrees")
        depth = as_1d_floats("depth", np.atleast_1d(depth), n=n, unit="meters")
        strike = as_1d_floats("strike", np.atleast_1d(strike), n=n, unit="degrees")
        dip = as_1d_floats("dip", np.atleast_1d(dip), n=n, unit="degrees")
        check_range("lat", lat, -90.0, 90.0, unit="degrees")
        check_range("lon", lon, -360.0, 360.0, unit="degrees")
        check_range("dip", dip, 0.0, 90.0, unit="degrees")

        if vertices is None:
            if length is None or width is None:
                raise ValueError(
                    "a fault needs either length and width (rectangular "
                    "patches) or vertices (triangular patches)"
                )
            # Rectangular centroid depths feed the Okada kernel directly and
            # must be below the surface. Triangular centroid depths are
            # derived from the vertices, so validate() reports those instead.
            if n and np.min(depth) < 0.0:
                raise ValueError(
                    "depth is positive down (meters); centroid depths must be "
                    f">= 0, got minimum {np.min(depth):g}"
                )
            length = as_1d_floats("length", np.atleast_1d(length), n=n, unit="meters")
            width = as_1d_floats("width", np.atleast_1d(width), n=n, unit="meters")
            check_positive("length", length, unit="meters")
            check_positive("width", width, unit="meters")
        else:
            if length is not None or width is not None:
                raise ValueError("give either length and width, or vertices; not both")
            vertices = np.array(vertices, dtype=float)
            if vertices.shape != (n, 3, 3):
                raise ValueError("vertices must have shape (N, 3, 3)")

        if grid_shape is not None and grid_shape[0] * grid_shape[1] != n:
            raise ValueError(
                f"grid_shape {grid_shape} implies "
                f"{grid_shape[0] * grid_shape[1]} patches, but got {n}"
            )

        self.lon = lon
        self.lat = lat
        self.depth = depth
        self.strike = strike
        self.dip = dip
        self.length: np.ndarray | None = length
        self.width: np.ndarray | None = width
        self.vertices: np.ndarray | None = vertices
        self.grid_shape = grid_shape
        self.medium = DEFAULT_MEDIUM if medium is None else medium
        self.frame = (
            LocalFrame(float(np.mean(lat)), float(np.mean(lon)))
            if frame is None
            else frame
        )
        for array in (lon, lat, depth, strike, dip, length, width, vertices):
            if array is not None:
                array.flags.writeable = False

        self._centers_local: np.ndarray | None = None
        self._laplacian: np.ndarray | None = None

    # ==================================================================
    # Factory classmethods
    # ==================================================================

    @classmethod
    def planar(
        cls,
        *,
        lat: float,
        lon: float,
        depth: float,
        strike: float,
        dip: float,
        length: float,
        width: float,
        n_length: int = 1,
        n_width: int = 1,
        medium: ElasticMedium | None = None,
        frame: LocalFrame | None = None,
    ) -> "Fault":
        """Create a discretized planar fault from its center.

        Args:
            lat: Latitude of fault centroid.
            lon: Longitude of fault centroid.
            depth: Depth of fault centroid in meters (positive down).
            strike: Strike angle in degrees from North.
            dip: Dip angle in degrees from horizontal.
            length: Total along-strike length in meters.
            width: Total down-dip width in meters.
            n_length: Number of patches along strike.
            n_width: Number of patches down dip.
            medium: Elastic half-space parameters. Defaults to the 30 GPa
                Poisson solid ``geodef.medium.DEFAULT_MEDIUM``.
            frame: Local frame for local-coordinate views. Defaults to a frame
                centered horizontally on the fault.

        Returns:
            A Fault with ``n_length * n_width`` rectangular patches.
        """
        if n_length < 1 or n_width < 1:
            raise ValueError("n_length and n_width must be positive integers")
        selected_frame = LocalFrame(lat, lon) if frame is None else frame
        center = selected_frame.to_enu(lon=lon, lat=lat, alt=selected_frame.origin_alt)
        as_planar_vector([center[0], center[1], depth, strike, dip, length, width])
        patch_L = length / n_length
        patch_W = width / n_width

        sin_str = np.sin(np.radians(strike))
        cos_str = np.cos(np.radians(strike))
        sin_dip = np.sin(np.radians(dip))
        cos_dip = np.cos(np.radians(dip))

        # Offset from center to top-left corner of the fault
        fault_e0 = -0.5 * length * sin_str - 0.5 * width * cos_dip * cos_str
        fault_n0 = -0.5 * length * cos_str + 0.5 * width * cos_dip * sin_str
        fault_u0 = -0.5 * width * sin_dip

        # Vectorized grid of patch center offsets
        i_idx = np.arange(n_length)
        j_idx = np.arange(n_width)
        jj_grid, ii_grid = np.meshgrid(j_idx, i_idx, indexing="ij")
        ii = ii_grid.ravel()
        jj = jj_grid.ravel()

        e_offsets = (
            fault_e0
            + (ii + 0.5) * patch_L * sin_str
            + (jj + 0.5) * patch_W * cos_dip * cos_str
        )
        n_offsets = (
            fault_n0
            + (ii + 0.5) * patch_L * cos_str
            - (jj + 0.5) * patch_W * cos_dip * sin_str
        )
        u_offsets = fault_u0 + (jj + 0.5) * patch_W * sin_dip

        geographic = selected_frame.to_geographic(
            east=center[0] + e_offsets,
            north=center[1] + n_offsets,
            up=np.zeros_like(e_offsets),
        )
        lon_c = geographic[:, 0]
        lat_c = geographic[:, 1]
        depth_c = depth - u_offsets

        n_patches = n_length * n_width
        strike_arr = np.full(n_patches, strike)
        dip_arr = np.full(n_patches, dip)
        length_arr = np.full(n_patches, patch_L)
        width_arr = np.full(n_patches, patch_W)

        return cls(
            lon=lon_c,
            lat=lat_c,
            depth=depth_c,
            strike=strike_arr,
            dip=dip_arr,
            length=length_arr,
            width=width_arr,
            grid_shape=(n_length, n_width),
            medium=medium,
            frame=selected_frame,
        )

    @classmethod
    def planar_from_corner(
        cls,
        *,
        lat: float,
        lon: float,
        depth: float,
        strike: float,
        dip: float,
        length: float,
        width: float,
        n_length: int = 1,
        n_width: int = 1,
        medium: ElasticMedium | None = None,
    ) -> "Fault":
        """Create a discretized planar fault from its top-left corner.

        The top-left corner is the shallowest point at the start of the
        along-strike direction.

        Args:
            lat: Latitude of top-left corner.
            lon: Longitude of top-left corner.
            depth: Depth of top-left corner in meters (positive down).
            strike: Strike angle in degrees from North.
            dip: Dip angle in degrees from horizontal.
            length: Total along-strike length in meters.
            width: Total down-dip width in meters.
            n_length: Number of patches along strike.
            n_width: Number of patches down dip.

        Returns:
            A Fault with ``n_length * n_width`` rectangular patches.
        """
        for pname, val, unit in (
            ("lat", lat, "degrees"),
            ("lon", lon, "degrees"),
            ("depth", depth, "meters"),
            ("strike", strike, "degrees"),
            ("dip", dip, "degrees"),
            ("length", length, "meters"),
            ("width", width, "meters"),
        ):
            check_finite_scalar(pname, val, unit=unit)
        sin_str = np.sin(np.radians(strike))
        cos_str = np.cos(np.radians(strike))
        sin_dip = np.sin(np.radians(dip))
        cos_dip = np.cos(np.radians(dip))

        # Compute center of the fault from the corner
        center_e = 0.5 * length * sin_str + 0.5 * width * cos_dip * cos_str
        center_n = 0.5 * length * cos_str - 0.5 * width * cos_dip * sin_str
        center_u = 0.5 * width * sin_dip

        center_lat, center_lon, _ = transforms.translate_flat(
            lat,
            lon,
            0.0,
            center_e,
            center_n,
            0.0,
        )
        center_depth = depth + center_u  # deeper = more positive

        # Delegate to planar() which handles the grid generation
        # Note: we negate center_u because depth convention is positive-down
        return cls.planar(
            lat=float(center_lat),
            lon=float(center_lon),
            depth=float(center_depth),
            strike=strike,
            dip=dip,
            length=length,
            width=width,
            n_length=n_length,
            n_width=n_width,
            medium=medium,
        )

    @classmethod
    def from_triangles(
        cls,
        vertices: npt.ArrayLike,
        *,
        triangles: npt.ArrayLike | None = None,
        frame: LocalFrame | None = None,
        ref_lat: float | None = None,
        ref_lon: float | None = None,
        medium: ElasticMedium | None = None,
    ) -> "Fault":
        """Create a triangular fault from local East-North-Up vertices.

        Per-triangle strike, dip, and geographic centers are derived from
        the vertex geometry. Two input forms are supported:

        - **Explicit vertices** (``triangles=None``): ``vertices`` is an
          ``(N, 3, 3)`` array giving the three corners of each triangle.
        - **Nodes plus connectivity** (``triangles`` given): ``vertices`` is
          a shared ``(M, 3)`` node array and ``triangles`` is an ``(N, 3)``
          index array, preserving the patch order of an imported mesh.

        Args:
            vertices: Triangle corners with shape (N, 3, 3), or a node array
                with shape (M, 3). Rows are ``[east, north, up]`` in meters.
            triangles: Optional connectivity indices into ``vertices``, shape
                (N, 3).
            frame: Local frame the vertices are expressed in. Defaults to a
                frame at longitude 0, latitude 0.
            ref_lat: Shorthand for ``frame=LocalFrame(ref_lat, ref_lon)``.
            ref_lon: See ``ref_lat``.
            medium: Elastic half-space parameters. Defaults to the 30 GPa
                Poisson solid ``geodef.medium.DEFAULT_MEDIUM``.

        Returns:
            A triangular Fault (``engine == "tri"``).

        Raises:
            ValueError: If the array shapes are inconsistent, or a triangle
                index is out of range.
        """
        if (ref_lat is None) != (ref_lon is None):
            raise ValueError("ref_lat and ref_lon must be provided together")
        if ref_lat is not None and ref_lon is not None:
            if frame is not None:
                frame.require_compatible(LocalFrame(ref_lat, ref_lon))
            frame = LocalFrame(ref_lat, ref_lon)
        selected_frame = LocalFrame(0.0, 0.0) if frame is None else frame
        if triangles is None:
            vertices_array = np.asarray(vertices, dtype=float)
            if vertices_array.ndim != 3 or vertices_array.shape[1:] != (3, 3):
                raise ValueError(
                    "without triangles, vertices must have shape (N, 3, 3)"
                )
            if not np.all(np.isfinite(vertices_array)):
                raise ValueError("vertices must contain only finite values")
        else:
            vertices_array = vertices_from_nodes(vertices, triangles)

        centers_enu = np.mean(vertices_array, axis=1)
        centers_geo = selected_frame.to_geographic(
            east=centers_enu[:, 0],
            north=centers_enu[:, 1],
            up=centers_enu[:, 2],
        )
        strike, dip = triangle_strike_dip(vertices_array)

        return cls(
            lon=centers_geo[:, 0],
            lat=centers_geo[:, 1],
            depth=-centers_geo[:, 2],
            strike=strike,
            dip=dip,
            vertices=vertices_array,
            medium=medium,
            frame=selected_frame,
        )

    @classmethod
    def from_mesh(
        cls,
        mesh: "Mesh",
        *,
        medium: ElasticMedium | None = None,
    ) -> "Fault":
        """Create a triangular fault from a ``Mesh`` object.

        Converts mesh geographic coordinates to local ENU vertices and
        derives per-triangle strike and dip from the vertex geometry.

        Args:
            mesh: A ``geodef.mesh.Mesh`` instance.
            medium: Elastic half-space parameters. Defaults to the 30 GPa
                Poisson solid ``geodef.medium.DEFAULT_MEDIUM``.

        Returns:
            A triangular Fault with ``engine="tri"``.
        """
        frame = mesh.frame
        if frame is None:
            raise ValueError("mesh must define a local frame")
        nodes_enu = frame.to_enu(
            lon=mesh.lon,
            lat=mesh.lat,
            alt=-mesh.depth,
        )
        return cls.from_triangles(
            nodes_enu,
            triangles=mesh.triangles,
            frame=frame,
            medium=medium,
        )

    @classmethod
    def load(
        cls,
        fname: str,
        *,
        format: str | None = None,
        ref_lat: float = 0.0,
        ref_lon: float = 0.0,
        medium: ElasticMedium | None = None,
    ) -> "Fault":
        """Load a fault model from a text file.

        Args:
            fname: Path to the file.
            format: File format. ``"center"`` for center-defined patches,
                ``"topleft"`` for top-left corner-defined patches,
                ``"seg"`` for unicycle segment format. If None,
                patches are assumed center-defined.
            ref_lat: Reference latitude for formats that use local Cartesian
                coordinates (e.g. ``"seg"``). Ignored for geographic formats.
            ref_lon: Reference longitude for local Cartesian formats.
            medium: Elastic half-space parameters. Fault files store geometry
                only, so the medium is always supplied at load time; defaults
                to ``geodef.medium.DEFAULT_MEDIUM``.

        Returns:
            A Fault object.

        Raises:
            ValueError: If format is unknown.

        Notes:
            The ``"seg"`` format uses local Cartesian coordinates (meters).
            The ``ref_lat`` and ``ref_lon`` parameters place the fault
            geographically. The coordinate transform currently uses
            ``transforms.translate_flat`` (simple flat-earth approximation);
            for large faults a more accurate projection may be needed.
        """
        if format is None:
            format = "center"

        if format == "ned":
            from geodef.mesh import Mesh

            mesh = Mesh.load(fname, format="ned")
            return cls.from_mesh(mesh, medium=medium)

        if format == "seg":
            fault = _fault_io.load_seg(fname, ref_lat, ref_lon)
        else:
            filedata = np.loadtxt(fname, ndmin=2)
            if format == "center":
                fault = _fault_io.load_center(filedata)
            elif format == "topleft":
                fault = _fault_io.load_topleft(filedata)
            else:
                raise ValueError(f"Unknown format: {format!r}")
        return fault if medium is None else fault.with_medium(medium)

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def n_patches(self) -> int:
        """Number of fault patches."""
        return self.lon.shape[0]

    @property
    def engine(self) -> str:
        """Dislocation type: ``"okada"`` (rectangles) or ``"tri"`` (triangles)."""
        return "okada" if self.vertices is None else "tri"

    @property
    def centers_geo(self) -> np.ndarray:
        """Patch centroids as an (N, 3) array of ``[lon, lat, depth]``.

        Depth is in meters, positive down.
        """
        return np.column_stack([self.lon, self.lat, self.depth])

    @property
    def centers_local(self) -> np.ndarray:
        """Patch centroids as ``[east, north, up]`` meters in :attr:`frame`."""
        if self._centers_local is None:
            if self.vertices is not None:
                self._centers_local = np.mean(self.vertices, axis=1)
            else:
                enu = self.frame.to_enu(
                    lon=self.lon,
                    lat=self.lat,
                    alt=np.full(self.n_patches, self.frame.origin_alt),
                )
                self._centers_local = np.column_stack(
                    [enu[:, 0], enu[:, 1], -self.depth]
                )
        return self._centers_local

    @property
    def corners(self) -> np.ndarray:
        """Patch corners as ``[east, north, up]`` meters in :attr:`frame`.

        Shape (N, 4, 3) for rectangles, ordered top-left, top-right,
        bottom-right, bottom-left looking along strike with "top" the
        shallow edge; shape (N, 3, 3) for triangles.
        """
        if self.vertices is not None:
            return self.vertices
        assert self.length is not None and self.width is not None
        strike = np.radians(self.strike)[:, None]
        dip = np.radians(self.dip)[:, None]
        along_strike = 0.5 * self.length[:, None] * np.array([-1.0, 1.0, 1.0, -1.0])
        up_dip = 0.5 * self.width[:, None] * np.array([1.0, 1.0, -1.0, -1.0])
        offsets = np.stack(
            [
                along_strike * np.sin(strike) - up_dip * np.cos(dip) * np.cos(strike),
                along_strike * np.cos(strike) + up_dip * np.cos(dip) * np.sin(strike),
                up_dip * np.sin(dip),
            ],
            axis=-1,
        )
        return self.centers_local[:, None, :] + offsets

    @property
    def corners_geo(self) -> np.ndarray:
        """Patch corners as ``[lon, lat, depth]``, shape (N, 3 or 4, 3).

        Depth is in meters, positive down. Corner order follows
        :attr:`corners`.
        """
        corners = self.corners
        if self.vertices is not None:
            geographic = self.frame.to_geographic(
                east=corners[..., 0], north=corners[..., 1], up=corners[..., 2]
            )
            geographic[..., 2] *= -1.0
            return geographic
        geographic = self.frame.to_geographic(
            east=corners[..., 0],
            north=corners[..., 1],
            up=np.full(corners.shape[:-1], self.frame.origin_alt),
        )
        geographic[..., 2] = -corners[..., 2]
        return geographic

    @property
    def areas(self) -> np.ndarray:
        """Patch areas in square meters, shape (N,)."""
        if self.vertices is None:
            assert self.length is not None and self.width is not None
            return self.length * self.width
        edge1 = self.vertices[:, 1, :] - self.vertices[:, 0, :]
        edge2 = self.vertices[:, 2, :] - self.vertices[:, 0, :]
        return 0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1)

    @property
    def laplacian(self) -> np.ndarray:
        """Discrete Laplacian smoothing operator, shape (N, N).

        Structured rectangular grids use finite-difference stencils;
        unstructured meshes use a distance-weighted six-nearest-neighbor
        Laplacian. Computed on first use and cached.
        """
        if self._laplacian is None:
            if self.grid_shape is not None:
                self._laplacian = _greens.build_laplacian_2d(*self.grid_shape)
            else:
                self._laplacian = _greens.build_laplacian_knn(
                    self.centers_local, k=6
                ).toarray()
        return self._laplacian

    def validate(self) -> ValidationReport:
        """Check the geometry for physically invalid or suspicious setups.

        Errors: patch material above the free surface, degenerate
        (zero-area) triangles. Warnings: extreme patch aspect ratios,
        strike angles outside [0, 360).

        Returns:
            A :class:`geodef.validation.ValidationReport`.
        """
        b = _ReportBuilder()
        if self.vertices is None:
            assert self.length is not None and self.width is not None
            top_edge = self.depth - 0.5 * self.width * np.sin(np.radians(self.dip))
            n_above = int(np.sum(top_edge < -1.0))
            if n_above:
                b.error(
                    "depth/width/dip",
                    f"{n_above} patch(es) extend above the free surface "
                    f"(shallowest top edge {np.min(top_edge):.1f} m): the "
                    "half-space solution is invalid there. Deepen the fault "
                    "or reduce its width.",
                )
            ratio = self.length / self.width
            if np.max(ratio) > 50.0 or np.min(ratio) < 1.0 / 50.0:
                b.warning(
                    "length/width",
                    f"extreme patch aspect ratio (max {np.max(ratio):.3g}, "
                    f"min {np.min(ratio):.3g}); very elongated patches "
                    "resolve slip poorly across their short dimension",
                )
        else:
            up = self.vertices[:, :, 2]
            n_above = int(np.sum(np.any(up > 1.0, axis=1)))
            if n_above:
                b.error(
                    "vertices",
                    f"{n_above} triangle(s) have vertices above the free "
                    f"surface (highest {np.max(up):.1f} m up): the "
                    "half-space solution is invalid there",
                )
            tiny = np.flatnonzero(self.areas < 1.0)
            if tiny.size:
                b.error(
                    "vertices",
                    f"{tiny.size} degenerate triangle(s) with area < 1 m^2 "
                    f"(first indices {tiny[:5].tolist()})",
                )
        if np.any(self.strike < 0.0) or np.any(self.strike >= 360.0):
            b.warning(
                "strike",
                "strike angles outside [0, 360) are accepted but usually "
                "indicate a sign or convention slip",
            )
        return b.report()

    def with_medium(self, medium: ElasticMedium) -> "Fault":
        """Return a copy of this fault with different elastic parameters.

        Args:
            medium: Elastic half-space parameters for the new fault.

        Returns:
            A new ``Fault`` with the same geometry and the given medium.
        """
        return self._replace(medium=medium)

    def to_frame(self, frame: LocalFrame) -> "Fault":
        """Return this fault re-expressed in another local frame.

        The physical (geographic) geometry is unchanged; triangular
        vertices are transformed into the new frame.

        Args:
            frame: Destination local frame.

        Returns:
            A fault with the same physical geometry and medium in ``frame``.
        """
        if self.frame.is_compatible(frame):
            return self
        if self.vertices is not None:
            vertices = self.frame.transform_enu(self.vertices, target=frame)
            return Fault.from_triangles(vertices, frame=frame, medium=self.medium)
        return self._replace(frame=frame)

    def _replace(self, **changes: object) -> "Fault":
        """Return a copy with some constructor arguments replaced."""
        arguments: dict[str, object] = {
            "lon": self.lon,
            "lat": self.lat,
            "depth": self.depth,
            "strike": self.strike,
            "dip": self.dip,
            "length": self.length,
            "width": self.width,
            "vertices": self.vertices,
            "grid_shape": self.grid_shape,
            "medium": self.medium,
            "frame": self.frame,
        }
        arguments.update(changes)
        return Fault(**arguments)  # type: ignore[arg-type]

    # ==================================================================
    # Forward modeling
    # ==================================================================

    def greens_matrix(
        self,
        *,
        lon: npt.ArrayLike,
        lat: npt.ArrayLike,
        kind: str = "displacement",
        depth: npt.ArrayLike | None = None,
    ) -> np.ndarray:
        """Compute the Green's function matrix at observation points.

        Column ``j`` holds the response at every observation point to one
        meter of slip on one patch. Columns are blocked: the first ``N``
        are strike-slip, the next ``N`` are dip-slip.

        Args:
            lon: Observation longitudes in degrees, shape (M,).
            lat: Observation latitudes in degrees, shape (M,).
            kind: ``"displacement"`` or ``"strain"``.
            depth: Observation depths in meters (positive down), shape (M,).
                Only used for ``kind="strain"``; defaults to the surface.

        Returns:
            Green's matrix. Displacement: shape (3*M, 2*N), rows
            interleaved ``[east, north, up]`` per point. Strain: shape
            (4*M, 2*N) for rectangles or (6*M, 2*N) for triangles.

        Raises:
            ValueError: If ``kind`` is unknown.
        """
        if kind not in {"displacement", "strain"}:
            raise ValueError(f"Unknown kind: {kind!r}. Use 'displacement' or 'strain'.")
        lon = np.atleast_1d(np.asarray(lon, dtype=float))
        lat = np.atleast_1d(np.asarray(lat, dtype=float))
        if depth is not None:
            depth = np.atleast_1d(np.asarray(depth, dtype=float))
        nu = self.medium.poisson_ratio
        centers = (self.lat, self.lon, self.depth)
        if self.vertices is not None:
            if kind == "displacement":
                return _greens.tri_displacement_greens(
                    lat, lon, *centers, self.vertices, nu=nu, frame=self.frame
                )
            return _greens.tri_strain_greens(
                lat,
                lon,
                *centers,
                self.vertices,
                nu=nu,
                obs_depth=depth,
                frame=self.frame,
            )
        assert self.length is not None and self.width is not None
        patches = (*centers, self.strike, self.dip, self.length, self.width)
        if kind == "displacement":
            return _greens.displacement_greens(lat, lon, *patches, nu=nu)
        return _greens.strain_greens(lat, lon, *patches, nu=nu, obs_depth=depth)

    def displacement(
        self,
        *,
        lon: npt.ArrayLike,
        lat: npt.ArrayLike,
        strike_slip: float | npt.ArrayLike = 0.0,
        dip_slip: float | npt.ArrayLike = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute surface displacements from a slip distribution.

        Args:
            lon: Observation longitudes in degrees, any shape.
            lat: Observation latitudes in degrees, broadcastable with ``lon``.
            strike_slip: Strike-slip per patch in meters (left-lateral
                positive), a scalar applied to every patch or shape (N,).
            dip_slip: Dip-slip per patch in meters (thrust positive), a
                scalar or shape (N,).

        Returns:
            ``(east, north, up)`` displacements in meters, each with the
            broadcast shape of ``lon`` and ``lat``.

        Examples:
            >>> east, north, up = fault.displacement(lon=lon, lat=lat, dip_slip=1.0)
        """
        lon_array, lat_array = np.broadcast_arrays(
            np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
        )
        slip = np.concatenate(
            [
                np.broadcast_to(np.asarray(strike_slip, dtype=float), self.n_patches),
                np.broadcast_to(np.asarray(dip_slip, dtype=float), self.n_patches),
            ]
        )
        G = self.greens_matrix(lon=lon_array.ravel(), lat=lat_array.ravel())
        east, north, up = (G @ slip).reshape(-1, 3).T
        shape = lon_array.shape
        return east.reshape(shape), north.reshape(shape), up.reshape(shape)

    # ==================================================================
    # Stress kernel
    # ==================================================================

    def stress_kernel(self, mu: float | None = None) -> np.ndarray:
        """Compute the stress interaction kernel for the fault.

        Evaluates strain Green's functions at patch centroid depths using
        okada92 (DC3D) for internal deformation.

        Args:
            mu: Shear modulus in Pa. Defaults to this fault's
                ``medium.shear_modulus``.

        Returns:
            Stress kernel matrix K, shape (4*N, 2*N).
        """
        from geodef import cache as _cache

        if mu is None:
            mu = self.medium.shear_modulus
        return _cache.cached_compute(
            {"kind": "stress_kernel", "mu": mu, **self._cache_key()},
            lambda: (
                mu
                * self.greens_matrix(
                    lon=self.lon, lat=self.lat, kind="strain", depth=self.depth
                )
            ),
        )

    def _cache_key(self) -> dict[str, object]:
        """Everything about this fault that affects a Green's matrix."""
        key: dict[str, object] = {
            "fault_lat": self.lat,
            "fault_lon": self.lon,
            "fault_depth": self.depth,
            "fault_strike": self.strike,
            "fault_dip": self.dip,
            "engine": self.engine,
            "nu": self.medium.poisson_ratio,
        }
        if self.vertices is None:
            key["fault_length"] = self.length
            key["fault_width"] = self.width
        else:
            key["fault_vertices"] = self.vertices
            key["frame_origin_lat"] = self.frame.origin_lat
            key["frame_origin_lon"] = self.frame.origin_lon
            key["frame_origin_alt"] = self.frame.origin_alt
            key["frame_projection"] = self.frame.projection
        return key

    # ==================================================================
    # Moment and magnitude
    # ==================================================================

    def moment(self, slip: npt.ArrayLike, mu: float | None = None) -> float:
        """Compute scalar seismic moment.

        Args:
            slip: Slip magnitude per patch, shape (N,), in meters.
            mu: Shear modulus in Pa. Defaults to this fault's
                ``medium.shear_modulus``.

        Returns:
            Seismic moment in N-m.
        """
        if mu is None:
            mu = self.medium.shear_modulus
        slip_array = np.asarray(slip, dtype=float)
        return float(mu * np.sum(slip_array * self.areas))

    def magnitude(self, slip: npt.ArrayLike, mu: float | None = None) -> float:
        """Compute moment magnitude from a slip distribution.

        Args:
            slip: Slip magnitude per patch, shape (N,), in meters.
            mu: Shear modulus in Pa. Defaults to this fault's
                ``medium.shear_modulus``.

        Returns:
            Moment magnitude Mw.
        """
        return moment_to_magnitude(self.moment(slip, mu))

    # ==================================================================
    # Grid utilities
    # ==================================================================

    def patch_index(self, strike_idx: int, dip_idx: int) -> int:
        """Convert grid indices to flat patch index.

        Patches are stored in row-major order: dip index varies slowest,
        strike index varies fastest.

        Args:
            strike_idx: Along-strike index (0 to n_length-1).
            dip_idx: Down-dip index (0 to n_width-1).

        Returns:
            Flat index into the patch arrays.

        Raises:
            ValueError: If the fault has no structured grid.
        """
        if self.grid_shape is None:
            raise ValueError("patch_index requires a structured grid (grid_shape)")
        nL, _ = self.grid_shape
        return dip_idx * nL + strike_idx

    def reshape_patches(self, values: npt.ArrayLike) -> np.ndarray:
        """Reshape a patch-first array into ``[dip, strike, ...]`` grid axes.

        The helper makes the storage convention explicit: strike index varies
        fastest, so a flat vector becomes a grid of shape
        ``(n_width, n_length)``. Trailing value dimensions are preserved.

        Args:
            values: Array whose first axis has length ``n_patches``.

        Returns:
            Array with leading axes ``(n_width, n_length)``.

        Raises:
            ValueError: If the fault is unstructured or the leading dimension
                does not match the patch count.
        """
        if self.grid_shape is None:
            raise ValueError("reshape_patches requires a structured grid")
        array = np.asarray(values)
        if array.ndim == 0 or array.shape[0] != self.n_patches:
            raise ValueError(
                f"values must have leading dimension {self.n_patches}, got "
                f"shape {array.shape}"
            )
        n_length, n_width = self.grid_shape
        return array.reshape((n_width, n_length, *array.shape[1:]))

    def flatten_patches(self, values: npt.ArrayLike) -> np.ndarray:
        """Flatten ``[dip, strike, ...]`` grid axes into patch storage order.

        Args:
            values: Array with leading shape ``(n_width, n_length)``.

        Returns:
            Patch-first array with leading dimension ``n_patches``.

        Raises:
            ValueError: If the fault is unstructured or leading grid axes do
                not match the fault.
        """
        if self.grid_shape is None:
            raise ValueError("flatten_patches requires a structured grid")
        array = np.asarray(values)
        n_length, n_width = self.grid_shape
        expected = (n_width, n_length)
        if array.ndim < 2 or array.shape[:2] != expected:
            raise ValueError(
                f"values must have leading shape {expected}, got shape {array.shape}"
            )
        return array.reshape((self.n_patches, *array.shape[2:]))

    # ==================================================================
    # File I/O
    # ==================================================================

    def save(
        self,
        fname: str,
        *,
        format: str | None = None,
        ref_lat: float = 0.0,
        ref_lon: float = 0.0,
        vpl: float = 1.0,
        rake: float = 90.0,
    ) -> None:
        """Save fault model to a file.

        For rectangular faults the default format is ``"center"``.
        For triangular faults the default (and only supported) format is
        ``"ned"`` (unicycle ``.ned`` + ``.tri`` pair).

        Args:
            fname: Output file path (base name for ``"ned"`` format).
            format: ``"center"`` or ``"seg"`` for rectangular faults;
                ``"ned"`` for triangular faults.  ``None`` auto-selects
                ``"ned"`` for triangular and ``"center"`` for rectangular.
            ref_lat: Reference latitude for ``"seg"`` format.
            ref_lon: Reference longitude for ``"seg"`` format.
            vpl: Plate velocity for ``"seg"`` format header.
            rake: Rake angle for ``"seg"`` format header.

        Raises:
            ValueError: If the format is unknown or incompatible with the
                fault engine.
        """
        if format is None:
            format = "ned" if self.engine == "tri" else "center"

        if format == "ned":
            if self.engine != "tri":
                raise ValueError(
                    "ned format requires a triangular fault (engine='tri'); "
                    "use format='center' or 'seg' for rectangular faults"
                )
            _fault_io.save_tri_ned(self, fname)
        elif format in ("center", "seg"):
            if self.engine != "okada":
                raise ValueError(
                    f"format={format!r} requires a rectangular fault "
                    f"(engine='okada'); use format='ned' for triangular faults"
                )
            if format == "center":
                _fault_io.save_center(self, fname)
            else:
                _fault_io.save_seg(self, fname, ref_lat, ref_lon, vpl, rake)
        else:
            raise ValueError(
                f"Unknown format: {format!r}. "
                "Use 'center' or 'seg' for rectangular, 'ned' for triangular."
            )

    def to_gmt(
        self,
        fname: str,
        values: np.ndarray | None = None,
    ) -> None:
        """Export fault patches as a GMT multi-segment polygon file.

        Each patch becomes one closed polygon segment with a ``> -Z{value}``
        header line.  The file is compatible with GMT's ``psxy -L`` (closed
        polygons) and ``-Z`` coloring.

        Args:
            fname: Output file path.
            values: Scalar value per patch (e.g. slip magnitude), shape (N,).
                Defaults to zeros if not provided.

        Raises:
            ValueError: If ``values`` is provided but has the wrong length.
        """
        _fault_io.write_gmt(self, fname, values)

    # ==================================================================
    # Repr
    # ==================================================================

    def __repr__(self) -> str:
        grid = f", grid_shape={self.grid_shape}" if self.grid_shape else ""
        return f"Fault(n_patches={self.n_patches}, engine={self.engine!r}{grid})"


# ======================================================================
# Module-level utilities
# ======================================================================


def moment_to_magnitude(moment: float) -> float:
    """Convert seismic moment to moment magnitude.

    Args:
        moment: Seismic moment in N-m.

    Returns:
        Moment magnitude Mw.
    """
    return (2.0 / 3.0) * np.log10(moment) - 6.07


def magnitude_to_moment(mw: float) -> float:
    """Convert moment magnitude to seismic moment.

    Args:
        mw: Moment magnitude.

    Returns:
        Seismic moment in N-m.
    """
    return 10.0 ** (1.5 * (mw + 6.07))


# ======================================================================
# Seg format helpers
# ======================================================================
