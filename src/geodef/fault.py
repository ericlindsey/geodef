"""Fault geometry class for rectangular and triangular fault patches.

Provides the `Fault` class with factory classmethods for creating faults
from geometric parameters, files, or slab2.0 grids. Supports forward
modeling via Green's function matrices and seismic moment calculation.
"""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from geodef import _engines, _fault_io, transforms
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

    Create via factory classmethods rather than calling ``__init__`` directly:

    - ``Fault.planar(...)`` — discretized planar fault from center parameters
    - ``Fault.planar_from_corner(...)`` — from top-left corner
    - ``Fault.load(fname)`` — from a text file

    Args:
        lat: Patch center latitudes, shape (N,).
        lon: Patch center longitudes, shape (N,).
        depth: Patch centroid depths in meters (positive down), shape (N,).
        strike: Strike angles in degrees, shape (N,).
        dip: Dip angles in degrees, shape (N,).
        length: Along-strike lengths in meters, shape (N,). None for triangular.
        width: Down-dip widths in meters, shape (N,). None for triangular.
        vertices: Triangle vertices in local ENU, shape (N, 3, 3). None for rectangular.
        grid_shape: ``(n_length, n_width)`` for structured rectangular grids.
        engine: Green's function engine, ``"okada"`` or ``"tri"``.
        medium: Elastic half-space parameters used by Green's functions,
            stress kernels, and moment. Defaults to
            ``geodef.medium.DEFAULT_MEDIUM`` (30 GPa Poisson solid).
        frame: Local frame for local-coordinate views. Inferred from mean patch
            coordinates for direct legacy construction.
    """

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        depth: np.ndarray,
        strike: np.ndarray,
        dip: np.ndarray,
        length: np.ndarray | None,
        width: np.ndarray | None,
        *,
        vertices: np.ndarray | None = None,
        grid_shape: tuple[int, int] | None = None,
        engine: str = "okada",
        medium: ElasticMedium | None = None,
        frame: LocalFrame | None = None,
    ) -> None:
        lat = as_1d_floats("lat", np.atleast_1d(lat), unit="degrees")
        n = lat.shape[0]
        lon = as_1d_floats("lon", np.atleast_1d(lon), n=n, unit="degrees")
        depth = as_1d_floats("depth", np.atleast_1d(depth), n=n, unit="meters")
        strike = as_1d_floats("strike", np.atleast_1d(strike), n=n, unit="degrees")
        dip = as_1d_floats("dip", np.atleast_1d(dip), n=n, unit="degrees")

        check_range("lat", lat, -90.0, 90.0, unit="degrees")
        check_range("lon", lon, -360.0, 360.0, unit="degrees")
        check_range("dip", dip, 0.0, 90.0, unit="degrees")

        if engine not in ("okada", "tri"):
            raise ValueError(f"engine must be 'okada' or 'tri', got {engine!r}")

        # Rectangular centroid depths feed the Okada kernel directly and must
        # be below the surface. Triangular centroid depths are derived
        # metadata (the kernel uses the ENU vertices) and legitimately carry
        # meter-scale ellipsoidal-curvature offsets near the surface, so
        # above-surface *vertices* are reported by validate() instead.
        if engine == "okada" and n and np.min(depth) < 0.0:
            raise ValueError(
                "depth is positive down (meters); centroid depths must be "
                f">= 0, got minimum {np.min(depth):g}"
            )

        if engine == "okada":
            if length is None or width is None:
                raise ValueError("Rectangular faults require length and width arrays")
            length = as_1d_floats("length", np.atleast_1d(length), n=n, unit="meters")
            width = as_1d_floats("width", np.atleast_1d(width), n=n, unit="meters")
            check_positive("length", length, unit="meters")
            check_positive("width", width, unit="meters")
        else:
            if vertices is None:
                raise ValueError("Triangular faults require a vertices array")
            vertices = np.asarray(vertices, dtype=float)
            if vertices.shape != (n, 3, 3):
                raise ValueError("vertices must have shape (N, 3, 3)")

        if grid_shape is not None:
            nL, nW = grid_shape
            if nL * nW != n:
                raise ValueError(
                    f"grid_shape {grid_shape} implies {nL * nW} patches, but got {n}"
                )

        self._lat = lat
        self._lon = lon
        self._depth = depth
        self.strike = strike
        self.dip = dip
        self._length = length
        self._width = width
        self._vertices = vertices
        self._grid_shape = grid_shape
        self._engine = engine
        self._medium = DEFAULT_MEDIUM if medium is None else medium
        inferred_frame = LocalFrame(float(np.mean(lat)), float(np.mean(lon)))
        self._frame = inferred_frame if frame is None else frame

        # Make arrays read-only
        for arr in (self._lat, self._lon, self._depth, self.strike, self.dip):
            arr.flags.writeable = False
        if self._length is not None and self._width is not None:
            self._length.flags.writeable = False
            self._width.flags.writeable = False
        if self._vertices is not None:
            self._vertices.flags.writeable = False

        # Lazy caches
        self._centers_local: np.ndarray | None = None
        self._laplacian: np.ndarray | None = None
        self._ref_lat = self._frame.origin_lat
        self._ref_lon = self._frame.origin_lon

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
            lat_c,
            lon_c,
            depth_c,
            strike_arr,
            dip_arr,
            length_arr,
            width_arr,
            grid_shape=(n_length, n_width),
            engine="okada",
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
        vertices: np.ndarray,
        *,
        ref_lat: float | None = None,
        ref_lon: float | None = None,
        frame: LocalFrame | None = None,
        triangles: np.ndarray | None = None,
        medium: ElasticMedium | None = None,
    ) -> "Fault":
        """Create a triangular fault from ENU vertex coordinates.

        Derives per-triangle strike, dip, and geographic centers automatically
        from the vertex geometry.

        Two input forms are supported:

        - **Explicit vertices** (``triangles=None``): ``vertices`` is a
          ``(N, 3, 3)`` array giving all three ENU corners of each triangle.
        - **Node array + connectivity** (``triangles`` given): ``vertices`` is
          a shared ``(M, 3)`` node array and ``triangles`` is an ``(N, 3)``
          index array. This preserves the exact patch order of an imported
          mesh and its node sharing.

        Args:
            vertices: Per-triangle corners with shape (N, 3, 3), or a shared
                node array with shape (M, 3). Numeric rows are [east, north,
                up] in meters.
            ref_lat: Reference latitude for the ENU origin.
            ref_lon: Reference longitude for the ENU origin.
            frame: Explicit local frame. Mutually exclusive with legacy
                ``ref_lat``/``ref_lon``.
            triangles: Optional connectivity indices into ``vertices``, shape
                (N, 3). When given, ``vertices`` is treated as a node array.
            medium: Elastic half-space parameters. Defaults to the 30 GPa
                Poisson solid ``geodef.medium.DEFAULT_MEDIUM``.

        Returns:
            A triangular Fault with ``engine="tri"``.

        Raises:
            ValueError: If the array shapes are inconsistent, or a triangle
                index is out of range.
        """
        legacy_frame: LocalFrame | None = None
        if (ref_lat is None) != (ref_lon is None):
            raise ValueError("ref_lat and ref_lon must be provided together")
        if ref_lat is not None and ref_lon is not None:
            legacy_frame = LocalFrame(ref_lat, ref_lon)
        if frame is not None and legacy_frame is not None:
            frame.require_compatible(legacy_frame)

        selected_frame = frame if frame is not None else legacy_frame
        if selected_frame is None:
            selected_frame = LocalFrame(0.0, 0.0)
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
            centers_geo[:, 1],
            centers_geo[:, 0],
            -centers_geo[:, 2],
            strike,
            dip,
            None,
            None,
            vertices=vertices_array,
            engine="tri",
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
        return self._lat.shape[0]

    @property
    def centers_geo(self) -> np.ndarray:
        """Patch centroids as (N, 3) array of [lon, lat, depth].

        Follows the documented geographic ordering (x, y order; matches
        ``Mesh.centers_geo``). Depth is in meters, positive down.
        """
        return np.column_stack([self._lon, self._lat, self._depth])

    @property
    def centers_local(self) -> np.ndarray:
        """Patch centroids in local Cartesian [east, north, up] in meters.

        Coordinates are expressed in :attr:`frame`.
        """
        if self._centers_local is None:
            if self._vertices is not None:
                self._centers_local = np.mean(self._vertices, axis=1)
            else:
                enu = self._frame.to_enu(
                    lon=self._lon,
                    lat=self._lat,
                    alt=np.full(self.n_patches, self._frame.origin_alt),
                )
                self._centers_local = np.column_stack(
                    [enu[:, 0], enu[:, 1], -self._depth]
                )
        return self._centers_local

    @property
    def frame(self) -> LocalFrame:
        """Local coordinate frame for :attr:`centers_local` and vertices."""
        return self._frame

    @property
    def areas(self) -> np.ndarray:
        """Patch areas in square meters, shape (N,)."""
        if self._engine == "okada":
            assert self._length is not None and self._width is not None
            return self._length * self._width
        # Triangular: area from cross product of two edge vectors
        v = self._vertices
        assert v is not None
        edge1 = v[:, 1, :] - v[:, 0, :]
        edge2 = v[:, 2, :] - v[:, 0, :]
        return 0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1)

    @property
    def engine(self) -> str:
        """Green's function engine: ``"okada"`` or ``"tri"``."""
        return self._engine

    def validate(self) -> ValidationReport:
        """Check the geometry for physically invalid or suspicious setups.

        Errors: patch material above the free surface, degenerate
        (zero-area) triangles. Warnings: extreme patch aspect ratios,
        strike angles outside [0, 360).

        Returns:
            A :class:`geodef.validation.ValidationReport`.
        """
        b = _ReportBuilder()
        if self._engine == "okada":
            assert self._length is not None and self._width is not None
            top_edge = self._depth - 0.5 * self._width * np.sin(np.radians(self.dip))
            n_above = int(np.sum(top_edge < -1.0))
            if n_above:
                b.error(
                    "depth/width/dip",
                    f"{n_above} patch(es) extend above the free surface "
                    f"(shallowest top edge {np.min(top_edge):.1f} m): the "
                    "half-space solution is invalid there. Deepen the fault "
                    "or reduce its width.",
                )
            ratio = self._length / self._width
            if np.max(ratio) > 50.0 or np.min(ratio) < 1.0 / 50.0:
                b.warning(
                    "length/width",
                    f"extreme patch aspect ratio (max {np.max(ratio):.3g}, "
                    f"min {np.min(ratio):.3g}); very elongated patches "
                    "resolve slip poorly across their short dimension",
                )
        else:
            assert self._vertices is not None
            up = self._vertices[:, :, 2]
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

    @property
    def medium(self) -> ElasticMedium:
        """Elastic half-space parameters used by this fault's computations."""
        return self._medium

    def with_medium(self, medium: ElasticMedium) -> "Fault":
        """Return a copy of this fault with different elastic parameters.

        The geometry arrays are shared (they are immutable); only the medium
        differs.

        Args:
            medium: Elastic half-space parameters for the new fault.

        Returns:
            A new ``Fault`` with the same geometry and the given medium.
        """
        return Fault(
            self._lat,
            self._lon,
            self._depth,
            self.strike,
            self.dip,
            self._length,
            self._width,
            vertices=self._vertices,
            grid_shape=self._grid_shape,
            engine=self._engine,
            medium=medium,
            frame=self._frame,
        )

    def to_frame(self, frame: LocalFrame) -> "Fault":
        """Return this fault explicitly re-expressed in another local frame.

        Geographic rectangular patch coordinates remain unchanged. Triangular
        vertices are transformed so their physical geographic positions remain
        unchanged rather than being reinterpreted in the new frame.

        Args:
            frame: Destination local frame.

        Returns:
            A fault with the same physical geometry and elastic medium in
            ``frame``.
        """
        if self._frame.is_compatible(frame):
            return self
        if self._vertices is not None:
            vertices = self._frame.transform_enu(self._vertices, target=frame)
            return Fault.from_triangles(vertices, frame=frame, medium=self._medium)
        return Fault(
            self._lat,
            self._lon,
            self._depth,
            self.strike,
            self.dip,
            self._length,
            self._width,
            grid_shape=self._grid_shape,
            engine=self._engine,
            medium=self._medium,
            frame=frame,
        )

    @property
    def vertices(self) -> np.ndarray | None:
        """Triangle vertices in local ENU, shape (N, 3, 3).

        ``None`` for rectangular faults; use :attr:`vertices_3d` /
        :attr:`vertices_2d` for their corner geometry instead.
        """
        return self._vertices

    @property
    def grid_shape(self) -> tuple[int, int] | None:
        """Grid dimensions ``(n_length, n_width)`` for structured grids, else None."""
        return self._grid_shape

    @property
    def laplacian(self) -> np.ndarray:
        """Discrete Laplacian smoothing operator, shape (N, N).

        For structured rectangular grids, uses finite-difference stencils.
        For unstructured meshes (no ``grid_shape``), uses a distance-weighted
        K-nearest-neighbors Laplacian with ``k=6`` (returned as a dense array
        for API consistency).

        Computed lazily and cached.
        """
        if self._laplacian is None:
            if self._grid_shape is not None:
                nL, nW = self._grid_shape
                self._laplacian = _greens.build_laplacian_2d(nL, nW)
            else:
                L_sparse = _greens.build_laplacian_knn(self.centers_local, k=6)
                self._laplacian = L_sparse.toarray()
        return self._laplacian

    # ==================================================================
    # Forward modeling
    # ==================================================================

    def greens_matrix(
        self,
        obs_lat: np.ndarray,
        obs_lon: np.ndarray,
        kind: str = "displacement",
        obs_depth: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute Green's function matrix at observation points.

        Args:
            obs_lat: Observation latitudes, shape (M,).
            obs_lon: Observation longitudes, shape (M,).
            kind: ``"displacement"`` or ``"strain"``.
            obs_depth: Observation depths (M,), positive down, in meters.
                Only used for ``kind="strain"`` to compute internal
                deformation via okada92/DC3D. If None, observations
                are at the surface.

        Returns:
            Green's matrix G. For displacement: shape (3*M, 2*N).
            For strain (okada): shape (4*M, 2*N).
            For strain (tri): shape (6*M, 2*N).

        Raises:
            ValueError: If kind is unknown or engine doesn't support it.
        """
        nu = self._medium.poisson_ratio
        spec = _engines.get(self._engine)
        if kind == "displacement":
            fn = _engines.require(spec, "displacement_greens")
            return fn(self, obs_lat, obs_lon, nu=nu)
        if kind == "strain":
            fn = _engines.require(spec, "strain_greens")
            return fn(self, obs_lat, obs_lon, nu=nu, obs_depth=obs_depth)
        raise ValueError(f"Unknown kind: {kind!r}. Use 'displacement' or 'strain'.")

    def displacement(
        self,
        obs_lat: np.ndarray,
        obs_lon: np.ndarray,
        slip_strike: float | np.ndarray,
        slip_dip: float | np.ndarray = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute surface displacements from a slip distribution.

        Args:
            obs_lat: Observation latitudes, shape (M,).
            obs_lon: Observation longitudes, shape (M,).
            slip_strike: Strike-slip component per patch. Values may be scalar
                (broadcast to all patches) or shape (N,).
            slip_dip: Dip-slip component per patch. Scalar or array of shape (N,).

        Returns:
            ``(east, north, up)`` displacement arrays, each shape (M,).
        """
        obs_lat = np.atleast_1d(np.asarray(obs_lat, dtype=float))
        obs_lon = np.atleast_1d(np.asarray(obs_lon, dtype=float))

        slip_s = np.broadcast_to(
            np.asarray(slip_strike, dtype=float), (self.n_patches,)
        )
        slip_d = np.broadcast_to(np.asarray(slip_dip, dtype=float), (self.n_patches,))

        G = self.greens_matrix(obs_lat, obs_lon, kind="displacement")

        # Build slip vector: blocked [ss0, ..., ssN, ds0, ..., dsN]
        m = np.empty(2 * self.n_patches)
        m[: self.n_patches] = slip_s
        m[self.n_patches :] = slip_d

        d = G @ m

        ue = d[0::3]
        un = d[1::3]
        uz = d[2::3]
        return ue, un, uz

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
            mu = self._medium.shear_modulus
        key = _build_stress_key(self, mu)
        return _cache.cached_compute(
            key,
            lambda: (
                mu
                * self.greens_matrix(
                    self._lat,
                    self._lon,
                    kind="strain",
                    obs_depth=self._depth,
                )
            ),
        )

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
            mu = self._medium.shear_modulus
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
        if self._grid_shape is None:
            raise ValueError("patch_index requires a structured grid (grid_shape)")
        nL, _ = self._grid_shape
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
        if self._grid_shape is None:
            raise ValueError("reshape_patches requires a structured grid")
        array = np.asarray(values)
        if array.ndim == 0 or array.shape[0] != self.n_patches:
            raise ValueError(
                f"values must have leading dimension {self.n_patches}, got "
                f"shape {array.shape}"
            )
        n_length, n_width = self._grid_shape
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
        if self._grid_shape is None:
            raise ValueError("flatten_patches requires a structured grid")
        array = np.asarray(values)
        n_length, n_width = self._grid_shape
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
            format = "ned" if self._engine == "tri" else "center"

        if format == "ned":
            if self._engine != "tri":
                raise ValueError(
                    "ned format requires a triangular fault (engine='tri'); "
                    "use format='center' or 'seg' for rectangular faults"
                )
            _fault_io.save_tri_ned(self, fname)
        elif format in ("center", "seg"):
            if self._engine != "okada":
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
    # Vertex computation
    # ==================================================================

    @property
    def vertices_2d(self) -> np.ndarray:
        """Patch corner coordinates as (N, 4, 2) array of [lon, lat].

        Vertices are ordered: top-left, top-right, bottom-right, bottom-left
        (where "top" is the shallowest edge).
        """
        v3d = self.vertices_3d
        return v3d[:, :, :2]

    @property
    def patch_outlines(self) -> np.ndarray:
        """Closed patch outlines for plotting, shape (N, 5, 2) as [lon, lat].

        Each outline is a closed polygon (first vertex repeated at end),
        suitable for use with ``matplotlib.collections.PolyCollection``.
        """
        v2d = self.vertices_2d  # (N, 4, 2)
        return np.concatenate([v2d, v2d[:, :1, :]], axis=1)

    @property
    def vertices_3d(self) -> np.ndarray:
        """Patch corner coordinates as (N, 4, 3) array of [lon, lat, depth_km].

        Vertices are ordered: top-left, top-right, bottom-right, bottom-left.
        """
        if self._engine != "okada":
            raise NotImplementedError(
                "vertices_3d is only implemented for rectangular faults"
            )
        assert self._length is not None and self._width is not None

        n = self.n_patches
        sin_dip = np.sin(np.radians(self.dip))
        cos_dip = np.cos(np.radians(self.dip))
        sin_str = np.sin(np.radians(self.strike))
        cos_str = np.cos(np.radians(self.strike))

        half_L = self._length / 2
        half_W = self._width / 2

        # 4 corner offsets in ENU: [top-left, top-right, bottom-right, bottom-left]
        # "top" = updip (shallower), strike direction is positive along-strike
        e_offsets = np.column_stack(
            [
                -half_L * sin_str + half_W * cos_dip * cos_str,
                +half_L * sin_str + half_W * cos_dip * cos_str,
                +half_L * sin_str - half_W * cos_dip * cos_str,
                -half_L * sin_str - half_W * cos_dip * cos_str,
            ]
        )  # (N, 4)

        n_offsets = np.column_stack(
            [
                -half_L * cos_str - half_W * cos_dip * sin_str,
                +half_L * cos_str - half_W * cos_dip * sin_str,
                +half_L * cos_str + half_W * cos_dip * sin_str,
                -half_L * cos_str + half_W * cos_dip * sin_str,
            ]
        )  # (N, 4)

        depth_offsets = np.column_stack(
            [
                +half_W * sin_dip,
                +half_W * sin_dip,
                -half_W * sin_dip,
                -half_W * sin_dip,
            ]
        )  # (N, 4)

        # Convert ENU offsets to lat/lon using local scale factors
        lat_rad = np.radians(self._lat)
        m_per_deg_lat = (
            111132.92 - 559.82 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
        )
        m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)

        verts = np.empty((n, 4, 3))
        for corner in range(4):
            verts[:, corner, 0] = self._lon + e_offsets[:, corner] / m_per_deg_lon
            verts[:, corner, 1] = self._lat + n_offsets[:, corner] / m_per_deg_lat
            verts[:, corner, 2] = (self._depth + depth_offsets[:, corner]) * 1e-3

        return verts

    # ==================================================================
    # Repr
    # ==================================================================

    def __repr__(self) -> str:
        grid_str = f", grid={self._grid_shape}" if self._grid_shape else ""
        return f"Fault(n_patches={self.n_patches}, engine={self._engine!r}{grid_str})"


def _build_stress_key(fault: Fault, mu: float) -> dict:
    """Build the cache key dict for a stress kernel computation."""
    key: dict = {
        "kind": "stress_kernel",
        "mu": mu,
        "nu": fault._medium.poisson_ratio,
        "fault_lat": fault._lat,
        "fault_lon": fault._lon,
        "fault_depth": fault._depth,
        "fault_strike": fault.strike,
        "fault_dip": fault.dip,
        "engine": fault.engine,
    }
    if fault._length is not None:
        key["fault_length"] = fault._length
        key["fault_width"] = fault._width
    if fault._vertices is not None:
        key["fault_vertices"] = fault._vertices
    return key


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


