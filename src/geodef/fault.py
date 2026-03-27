"""Fault geometry class for rectangular and triangular fault patches.

Provides the `Fault` class with factory classmethods for creating faults
from geometric parameters, files, or slab2.0 grids. Supports forward
modeling via Green's function matrices and seismic moment calculation.
"""

import numpy as np

from geodef import greens as _greens
from geodef import transforms


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
    ) -> None:
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        depth = np.asarray(depth, dtype=float)
        strike = np.asarray(strike, dtype=float)
        dip = np.asarray(dip, dtype=float)

        n = lat.shape[0]
        if not all(arr.shape == (n,) for arr in (lon, depth, strike, dip)):
            raise ValueError(
                "lat, lon, depth, strike, dip must all be 1-D arrays of the same length"
            )

        if engine not in ("okada", "tri"):
            raise ValueError(f"engine must be 'okada' or 'tri', got {engine!r}")

        if engine == "okada":
            if length is None or width is None:
                raise ValueError("Rectangular faults require length and width arrays")
            length = np.asarray(length, dtype=float)
            width = np.asarray(width, dtype=float)
            if length.shape != (n,) or width.shape != (n,):
                raise ValueError("length and width must have shape (N,)")
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
        self._strike = strike
        self._dip = dip
        self._length = length
        self._width = width
        self._vertices = vertices
        self._grid_shape = grid_shape
        self._engine = engine

        # Make arrays read-only
        for arr in (self._lat, self._lon, self._depth, self._strike, self._dip):
            arr.flags.writeable = False
        if self._length is not None:
            self._length.flags.writeable = False
            self._width.flags.writeable = False
        if self._vertices is not None:
            self._vertices.flags.writeable = False

        # Lazy caches
        self._centers_local: np.ndarray | None = None
        self._laplacian: np.ndarray | None = None
        self._ref_lat = float(np.mean(lat))
        self._ref_lon = float(np.mean(lon))

    # ==================================================================
    # Factory classmethods
    # ==================================================================

    @classmethod
    def planar(
        cls,
        lat: float,
        lon: float,
        depth: float,
        strike: float,
        dip: float,
        length: float,
        width: float,
        n_length: int = 1,
        n_width: int = 1,
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

        Returns:
            A Fault with ``n_length * n_width`` rectangular patches.
        """
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
        jj, ii = np.meshgrid(j_idx, i_idx, indexing="ij")
        ii = ii.ravel()
        jj = jj.ravel()

        e_offsets = fault_e0 + (ii + 0.5) * patch_L * sin_str + (jj + 0.5) * patch_W * cos_dip * cos_str
        n_offsets = fault_n0 + (ii + 0.5) * patch_L * cos_str - (jj + 0.5) * patch_W * cos_dip * sin_str
        u_offsets = fault_u0 + (jj + 0.5) * patch_W * sin_dip

        lat_c, lon_c, _ = transforms.translate_flat(
            lat, lon, 0.0, e_offsets, n_offsets, 0.0,
        )
        depth_c = depth - u_offsets

        n_patches = n_length * n_width
        strike_arr = np.full(n_patches, strike)
        dip_arr = np.full(n_patches, dip)
        length_arr = np.full(n_patches, patch_L)
        width_arr = np.full(n_patches, patch_W)

        return cls(
            lat_c, lon_c, depth_c, strike_arr, dip_arr, length_arr, width_arr,
            grid_shape=(n_length, n_width), engine="okada",
        )

    @classmethod
    def planar_from_corner(
        cls,
        lat: float,
        lon: float,
        depth: float,
        strike: float,
        dip: float,
        length: float,
        width: float,
        n_length: int = 1,
        n_width: int = 1,
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
        sin_str = np.sin(np.radians(strike))
        cos_str = np.cos(np.radians(strike))
        sin_dip = np.sin(np.radians(dip))
        cos_dip = np.cos(np.radians(dip))

        # Compute center of the fault from the corner
        center_e = 0.5 * length * sin_str + 0.5 * width * cos_dip * cos_str
        center_n = 0.5 * length * cos_str - 0.5 * width * cos_dip * sin_str
        center_u = 0.5 * width * sin_dip

        center_lat, center_lon, _ = transforms.translate_flat(
            lat, lon, 0.0, center_e, center_n, 0.0,
        )
        center_depth = depth + center_u  # deeper = more positive

        # Delegate to planar() which handles the grid generation
        # Note: we negate center_u because depth convention is positive-down
        return cls.planar(
            float(center_lat), float(center_lon), float(center_depth),
            strike, dip, length, width, n_length, n_width,
        )

    @classmethod
    def load(
        cls,
        fname: str,
        *,
        format: str | None = None,
        ref_lat: float = 0.0,
        ref_lon: float = 0.0,
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

        if format == "seg":
            return cls._load_seg(fname, ref_lat, ref_lon)

        filedata = np.loadtxt(fname, ndmin=2)
        if format == "center":
            return cls._load_center(filedata)
        elif format == "topleft":
            return cls._load_topleft(filedata)
        raise ValueError(f"Unknown format: {format!r}")

    @classmethod
    def _load_center(cls, filedata: np.ndarray) -> "Fault":
        """Load patches defined by center coordinates.

        Expected columns: [id, dipid, strikeid, lon, lat, depth, L, W, strike, dip].
        """
        lon_c = filedata[:, 3]
        lat_c = filedata[:, 4]
        depth = filedata[:, 5]
        length = filedata[:, 6]
        width = filedata[:, 7]
        strike = filedata[:, 8]
        dip = filedata[:, 9]

        n_length = int(filedata[:, 2].max()) + 1
        n_width = int(filedata[:, 1].max()) + 1
        n = len(lat_c)
        grid_shape = (n_length, n_width) if n_length * n_width == n else None

        return cls(
            lat_c, lon_c, depth, strike, dip, length, width,
            grid_shape=grid_shape, engine="okada",
        )

    @classmethod
    def _load_topleft(cls, filedata: np.ndarray) -> "Fault":
        """Load patches defined by top-left corner.

        Expected columns: [id, dipid, strikeid, lon, lat, depth, L, W, strike, dip].
        """
        lon = filedata[:, 3]
        lat = filedata[:, 4]
        depth = filedata[:, 5]
        length = filedata[:, 6]
        width = filedata[:, 7]
        strike = filedata[:, 8]
        dip = filedata[:, 9]

        sin_str = np.sin(np.radians(strike))
        cos_str = np.cos(np.radians(strike))
        sin_dip = np.sin(np.radians(dip))
        cos_dip = np.cos(np.radians(dip))

        e_offset = (length / 2) * sin_str + (width / 2) * cos_dip * cos_str
        n_offset = (length / 2) * cos_str - (width / 2) * cos_dip * sin_str
        u_offset = (width / 2) * sin_dip

        lat_c, lon_c, _ = transforms.translate_flat(lat, lon, 0.0, e_offset, n_offset, 0.0)
        depth_c = depth + u_offset

        n_length = int(filedata[:, 2].max()) + 1
        n_width = int(filedata[:, 1].max()) + 1
        n = len(lat_c)
        grid_shape = (n_length, n_width) if n_length * n_width == n else None

        return cls(
            lat_c, lon_c, depth_c, strike, dip, length, width,
            grid_shape=grid_shape, engine="okada",
        )

    @classmethod
    def _load_seg(cls, fname: str, ref_lat: float, ref_lon: float) -> "Fault":
        """Load patches from a unicycle ``.seg`` file.

        Each line in the file defines a fault segment that is subdivided
        into patches using geometric growth factors. The ``.seg`` format
        uses local Cartesian coordinates (North, East, depth in meters).

        Expected columns (14-column format):
            n, Vpl, x1, x2, x3, Length, Width, Strike, Dip, Rake, L0, W0, qL, qW

        Or (13-column format, no Vpl):
            n, x1, x2, x3, Length, Width, Strike, Dip, Rake, L0, W0, qL, qW

        Args:
            fname: Path to the ``.seg`` file.
            ref_lat: Reference latitude for geographic placement.
            ref_lon: Reference longitude for geographic placement.
        """
        # Read file, skipping comment lines
        filedata = np.loadtxt(fname, comments="#", ndmin=2)
        ncols = filedata.shape[1]

        if ncols == 14:
            # With Vpl column
            x1 = filedata[:, 2]       # North position
            x2 = filedata[:, 3]       # East position
            x3 = filedata[:, 4]       # Depth (positive down)
            seg_L = filedata[:, 5]    # Total length
            seg_W = filedata[:, 6]    # Total width
            strike = filedata[:, 7]
            dip = filedata[:, 8]
            # rake = filedata[:, 9]   # stored but not used for geometry
            L0 = filedata[:, 10]      # Initial patch length
            W0 = filedata[:, 11]      # Initial patch width
            qL = filedata[:, 12]      # Length growth factor
            qW = filedata[:, 13]      # Width growth factor
        elif ncols == 13:
            # Without Vpl column
            x1 = filedata[:, 1]
            x2 = filedata[:, 2]
            x3 = filedata[:, 3]
            seg_L = filedata[:, 4]
            seg_W = filedata[:, 5]
            strike = filedata[:, 6]
            dip = filedata[:, 7]
            # rake = filedata[:, 8]
            L0 = filedata[:, 9]
            W0 = filedata[:, 10]
            qL = filedata[:, 11]
            qW = filedata[:, 12]
        else:
            raise ValueError(
                f"Seg file has {ncols} columns; expected 13 or 14"
            )

        # Process each segment and collect all patches
        all_patches = []
        for k in range(len(x1)):
            origin = np.array([x1[k], x2[k], x3[k]])  # North, East, Depth
            patches = _seg_to_patches(
                origin, seg_L[k], seg_W[k],
                strike[k], dip[k],
                L0[k], W0[k], qL[k], qW[k],
            )
            all_patches.append(patches)

        patches = np.vstack(all_patches)
        # patches columns: [north, east, depth, length, width, strike, dip]
        # These are upper-left corner positions in local Cartesian

        p_north = patches[:, 0]
        p_east = patches[:, 1]
        p_depth = patches[:, 2]
        p_length = patches[:, 3]
        p_width = patches[:, 4]
        p_strike = patches[:, 5]
        p_dip = patches[:, 6]

        # Convert upper-left corner to centroid
        sin_str = np.sin(np.radians(p_strike))
        cos_str = np.cos(np.radians(p_strike))
        sin_dip = np.sin(np.radians(p_dip))
        cos_dip = np.cos(np.radians(p_dip))

        # Strike vector: [cos(strike), sin(strike), 0] in (North, East, Down)
        # Dip vector: [-cos(dip)*sin(strike), cos(dip)*cos(strike), sin(dip)]
        # Center = corner + L/2 * strike_vec + W/2 * dip_vec
        # (following unicycle convention from flt2flt.m)
        c_north = p_north + (p_length / 2) * cos_str + (p_width / 2) * (-cos_dip * sin_str)
        c_east = p_east + (p_length / 2) * sin_str + (p_width / 2) * (cos_dip * cos_str)
        c_depth = p_depth + (p_width / 2) * sin_dip

        # Convert local Cartesian (East, North) to geographic
        lat_c, lon_c, _ = transforms.translate_flat(
            ref_lat, ref_lon, 0.0, c_east, c_north, 0.0,
        )
        depth_c = c_depth

        # Detect if this is a uniform grid (all patches same size, qL=qW=1)
        grid_shape = None
        if len(x1) == 1 and np.allclose(qL[0], 1.0) and np.allclose(qW[0], 1.0):
            n_width = round(seg_W[0] / W0[0])
            n_length = round(seg_L[0] / L0[0])
            if n_length * n_width == len(lat_c):
                grid_shape = (n_length, n_width)

        return cls(
            lat_c, lon_c, depth_c, p_strike, p_dip, p_length, p_width,
            grid_shape=grid_shape, engine="okada",
        )

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def n_patches(self) -> int:
        """Number of fault patches."""
        return self._lat.shape[0]

    @property
    def centers(self) -> np.ndarray:
        """Patch centroids as (N, 3) array of [lat, lon, depth]."""
        return np.column_stack([self._lat, self._lon, self._depth])

    @property
    def centers_local(self) -> np.ndarray:
        """Patch centroids in local Cartesian [east, north, up] in meters.

        Computed relative to the fault centroid (mean lat/lon).
        """
        if self._centers_local is None:
            alt = np.zeros(self.n_patches)
            e, n, u = transforms.geod2enu(
                self._lat, self._lon, alt,
                self._ref_lat, self._ref_lon, 0.0,
            )
            self._centers_local = np.column_stack([e, n, -self._depth])
        return self._centers_local

    @property
    def areas(self) -> np.ndarray:
        """Patch areas in square meters, shape (N,)."""
        if self._engine == "okada":
            return self._length * self._width
        # Triangular: area from cross product of two edge vectors
        v = self._vertices
        edge1 = v[:, 1, :] - v[:, 0, :]
        edge2 = v[:, 2, :] - v[:, 0, :]
        return 0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1)

    @property
    def engine(self) -> str:
        """Green's function engine: ``"okada"`` or ``"tri"``."""
        return self._engine

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
        if self._engine == "okada":
            if kind == "displacement":
                return _greens.displacement_greens(
                    obs_lat, obs_lon,
                    self._lat, self._lon, self._depth,
                    self._strike, self._dip, self._length, self._width,
                )
            elif kind == "strain":
                return _greens.strain_greens(
                    obs_lat, obs_lon,
                    self._lat, self._lon, self._depth,
                    self._strike, self._dip, self._length, self._width,
                    obs_depth=obs_depth,
                )
            raise ValueError(f"Unknown kind: {kind!r}. Use 'displacement' or 'strain'.")

        if self._engine == "tri":
            if kind == "displacement":
                return _greens.tri_displacement_greens(
                    obs_lat, obs_lon,
                    self._lat, self._lon, self._depth,
                    self._vertices,
                )
            elif kind == "strain":
                return _greens.tri_strain_greens(
                    obs_lat, obs_lon,
                    self._lat, self._lon, self._depth,
                    self._vertices,
                    obs_depth=obs_depth,
                )
            raise ValueError(f"Unknown kind: {kind!r}. Use 'displacement' or 'strain'.")

        raise ValueError(f"Unknown engine: {self._engine!r}")

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
            slip_strike: Strike-slip component per patch. Scalar (broadcast
                to all patches) or array of shape (N,).
            slip_dip: Dip-slip component per patch. Scalar or array of shape (N,).

        Returns:
            Tuple (ue, un, uz) of displacement arrays, each shape (M,).
        """
        obs_lat = np.atleast_1d(np.asarray(obs_lat, dtype=float))
        obs_lon = np.atleast_1d(np.asarray(obs_lon, dtype=float))
        nobs = obs_lat.shape[0]

        slip_s = np.broadcast_to(np.asarray(slip_strike, dtype=float), (self.n_patches,))
        slip_d = np.broadcast_to(np.asarray(slip_dip, dtype=float), (self.n_patches,))

        G = self.greens_matrix(obs_lat, obs_lon, kind="displacement")

        # Build slip vector: blocked [ss0, ..., ssN, ds0, ..., dsN]
        m = np.empty(2 * self.n_patches)
        m[:self.n_patches] = slip_s
        m[self.n_patches:] = slip_d

        d = G @ m

        ue = d[0::3]
        un = d[1::3]
        uz = d[2::3]
        return ue, un, uz

    # ==================================================================
    # Stress kernel
    # ==================================================================

    def stress_kernel(self, mu: float = 30e9) -> np.ndarray:
        """Compute the stress interaction kernel for the fault.

        Evaluates strain Green's functions at patch centroid depths using
        okada92 (DC3D) for internal deformation.

        Args:
            mu: Shear modulus in Pa (default 30 GPa).

        Returns:
            Stress kernel matrix K, shape (4*N, 2*N).
        """
        from geodef import cache as _cache

        key = _build_stress_key(self, mu)
        return _cache.cached_compute(
            key,
            lambda: mu * self.greens_matrix(
                self._lat, self._lon, kind="strain",
                obs_depth=self._depth,
            ),
        )

    # ==================================================================
    # Moment and magnitude
    # ==================================================================

    def moment(self, slip: np.ndarray, mu: float = 30e9) -> float:
        """Compute scalar seismic moment.

        Args:
            slip: Slip magnitude per patch, shape (N,), in meters.
            mu: Shear modulus in Pa (default 30 GPa).

        Returns:
            Seismic moment in N-m.
        """
        slip = np.asarray(slip, dtype=float)
        return float(mu * np.sum(slip * self.areas))

    def magnitude(self, slip: np.ndarray, mu: float = 30e9) -> float:
        """Compute moment magnitude from a slip distribution.

        Args:
            slip: Slip magnitude per patch, shape (N,), in meters.
            mu: Shear modulus in Pa (default 30 GPa).

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

    # ==================================================================
    # File I/O
    # ==================================================================

    def save(
        self,
        fname: str,
        *,
        format: str = "center",
        ref_lat: float = 0.0,
        ref_lon: float = 0.0,
        vpl: float = 1.0,
        rake: float = 90.0,
    ) -> None:
        """Save fault model to a text file.

        Args:
            fname: Output file path.
            format: ``"center"`` for center-defined patches, or ``"seg"``
                for unicycle segment format.
            ref_lat: Reference latitude for ``"seg"`` format (used to convert
                geographic coordinates back to local Cartesian).
            ref_lon: Reference longitude for ``"seg"`` format.
            vpl: Plate velocity for ``"seg"`` format header.
            rake: Rake angle for ``"seg"`` format header.

        Raises:
            ValueError: If format is unknown or fault is not rectangular.
        """
        if self._engine != "okada":
            raise ValueError("save() is only supported for rectangular faults")

        if format == "center":
            self._save_center(fname)
        elif format == "seg":
            self._save_seg(fname, ref_lat, ref_lon, vpl, rake)
        else:
            raise ValueError(f"Unknown format: {format!r}")

    def _save_center(self, fname: str) -> None:
        """Save in center-defined format."""
        if self._grid_shape is not None:
            nL, _ = self._grid_shape
            strike_ids = np.arange(self.n_patches) % nL
            dip_ids = np.arange(self.n_patches) // nL
        else:
            strike_ids = np.zeros(self.n_patches)
            dip_ids = np.arange(self.n_patches)

        outdata = np.column_stack((
            np.arange(self.n_patches),
            dip_ids,
            strike_ids,
            self._lon,
            self._lat,
            self._depth,
            self._length,
            self._width,
            self._strike,
            self._dip,
        ))
        np.savetxt(fname, outdata, fmt="%10.5f")

    def _save_seg(
        self, fname: str, ref_lat: float, ref_lon: float,
        vpl: float, rake: float,
    ) -> None:
        """Save as a unicycle ``.seg`` file.

        Writes one segment line that covers all patches. For faults with
        uniform patch sizes (qL=qW=1), this round-trips exactly. For
        non-uniform faults, L0/W0 are taken from the smallest patch.
        """
        # Convert geographic centers back to local Cartesian
        alt = np.zeros(self.n_patches)
        east, north, _ = transforms.geod2enu(
            self._lat, self._lon, alt, ref_lat, ref_lon, 0.0,
        )

        # Compute upper-left corner of each patch
        sin_str = np.sin(np.radians(self._strike))
        cos_str = np.cos(np.radians(self._strike))
        sin_dip = np.sin(np.radians(self._dip))
        cos_dip = np.cos(np.radians(self._dip))

        corner_north = north - (self._length / 2) * cos_str - (self._width / 2) * (-cos_dip * sin_str)
        corner_east = east - (self._length / 2) * sin_str - (self._width / 2) * (cos_dip * cos_str)
        corner_depth = self._depth - (self._width / 2) * sin_dip

        # Find the overall segment bounding box
        # x1 (North), x2 (East), x3 (Depth) of the segment origin
        # = upper-left corner of the shallowest, most-negative-strike patch
        x1 = float(np.min(corner_north))
        x2 = float(np.min(corner_east))
        x3 = float(np.min(corner_depth))

        # Total length and width of the segment
        strike_val = float(self._strike[0])
        dip_val = float(self._dip[0])

        # Project all corners onto strike/dip directions to get total extent
        str_rad = np.radians(strike_val)
        dip_rad = np.radians(dip_val)
        strike_vec = np.array([np.cos(str_rad), np.sin(str_rad)])
        dip_vec_h = np.array([-np.cos(dip_rad) * np.sin(str_rad),
                               np.cos(dip_rad) * np.cos(str_rad)])

        # Use patch properties to determine total extent
        # Sum unique lengths along strike and widths along dip
        total_L = float(np.sum(self._length[:1])) if self._grid_shape is None else float(self._length[0] * self._grid_shape[0])
        total_W = float(np.sum(self._width[:1])) if self._grid_shape is None else float(self._width[0] * self._grid_shape[1])

        # If we don't have grid_shape, estimate from the patches
        if self._grid_shape is None:
            # Sum all unique widths (down-dip) and max strike extent
            total_L = float(np.max(self._length) * len(self._length))
            total_W = float(np.sum(np.unique(self._width)))

        L0 = float(np.min(self._length))
        W0 = float(np.min(self._width))

        # Detect growth factors
        if self._grid_shape is not None:
            qL = 1.0
            qW = 1.0
        else:
            unique_widths = np.unique(self._width)
            if len(unique_widths) > 1:
                qW = float(unique_widths[1] / unique_widths[0])
            else:
                qW = 1.0
            unique_lengths = np.unique(self._length)
            if len(unique_lengths) > 1:
                qL = float(unique_lengths[1] / unique_lengths[0])
            else:
                qL = 1.0

        with open(fname, "w") as f:
            f.write("# Unicycle .seg file generated by geodef\n")
            f.write("# n  Vpl  x1  x2  x3  Length  Width  Strike  Dip  Rake  L0  W0  qL  qW\n")
            f.write(
                f"1 {vpl:.9f} {x1:.9f} {x2:.9f} {x3:.9f} "
                f"{total_L:.9f} {total_W:.9f} {strike_val:.9f} {dip_val:.9f} "
                f"{rake:.9f} {L0:.9f} {W0:.9f} {qL:.9f} {qW:.9f}\n"
            )

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
            raise NotImplementedError("vertices_3d is only implemented for rectangular faults")

        n = self.n_patches
        sin_dip = np.sin(np.radians(self._dip))
        cos_dip = np.cos(np.radians(self._dip))
        sin_str = np.sin(np.radians(self._strike))
        cos_str = np.cos(np.radians(self._strike))

        half_L = self._length / 2
        half_W = self._width / 2

        # 4 corner offsets in ENU: [top-left, top-right, bottom-right, bottom-left]
        # "top" = updip (shallower), strike direction is positive along-strike
        e_offsets = np.column_stack([
            -half_L * sin_str + half_W * cos_dip * cos_str,
            +half_L * sin_str + half_W * cos_dip * cos_str,
            +half_L * sin_str - half_W * cos_dip * cos_str,
            -half_L * sin_str - half_W * cos_dip * cos_str,
        ])  # (N, 4)

        n_offsets = np.column_stack([
            -half_L * cos_str - half_W * cos_dip * sin_str,
            +half_L * cos_str - half_W * cos_dip * sin_str,
            +half_L * cos_str + half_W * cos_dip * sin_str,
            -half_L * cos_str + half_W * cos_dip * sin_str,
        ])  # (N, 4)

        depth_offsets = np.column_stack([
            +half_W * sin_dip,
            +half_W * sin_dip,
            -half_W * sin_dip,
            -half_W * sin_dip,
        ])  # (N, 4)

        # Convert ENU offsets to lat/lon using local scale factors
        lat_rad = np.radians(self._lat)
        m_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
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
        "fault_lat": fault._lat,
        "fault_lon": fault._lon,
        "fault_depth": fault._depth,
        "fault_strike": fault._strike,
        "fault_dip": fault._dip,
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

def _seg_to_patches(
    origin: np.ndarray,
    total_L: float,
    total_W: float,
    strike: float,
    dip: float,
    L0: float,
    W0: float,
    alpha_l: float,
    alpha_w: float,
) -> np.ndarray:
    """Subdivide a fault segment into patches with geometric growth.

    Port of unicycle's ``flt2flt.m``. Patch widths grow geometrically
    with depth (down-dip). For each dip row, patch lengths also grow
    geometrically but are distributed evenly across the strike direction.

    Args:
        origin: Upper-left corner [North, East, Depth] in meters.
        total_L: Total along-strike length in meters.
        total_W: Total down-dip width in meters.
        strike: Strike angle in degrees.
        dip: Dip angle in degrees.
        L0: Initial (smallest) patch length.
        W0: Initial (smallest) patch width.
        alpha_l: Geometric growth factor for length (1.0 = uniform).
        alpha_w: Geometric growth factor for width (1.0 = uniform).

    Returns:
        Array of shape (N, 7) with columns
        [north, east, depth, length, width, strike, dip]
        where (north, east, depth) is the upper-left corner of each patch.
    """
    # Step 1: Compute down-dip width distribution
    remaining_W = total_W
    widths = []
    k = 0
    while remaining_W > 0:
        wt = W0 * alpha_w ** k
        if wt > remaining_W / 2:
            wt = remaining_W
        wt = min(wt, remaining_W)
        widths.append(wt)
        remaining_W -= wt
        k += 1

    # Step 2: Strike and dip direction unit vectors
    # Unicycle convention: strike_vec = [cos(strike), sin(strike), 0]
    # dip_vec = [-cos(dip)*sin(strike), cos(dip)*cos(strike), sin(dip)]
    str_rad = np.radians(strike)
    dip_rad = np.radians(dip)
    strike_vec = np.array([np.cos(str_rad), np.sin(str_rad), 0.0])
    dip_vec = np.array([
        -np.cos(dip_rad) * np.sin(str_rad),
        np.cos(dip_rad) * np.cos(str_rad),
        np.sin(dip_rad),
    ])

    # Step 3: Build patches row by row
    patches = []
    cumulative_w = 0.0
    for j, wj in enumerate(widths):
        # Patch length for this row
        lt = L0 * alpha_l ** j
        n_along = int(np.ceil(total_L / lt))
        lt = total_L / n_along  # distribute evenly

        for i in range(n_along):
            corner = origin + i * lt * strike_vec + cumulative_w * dip_vec
            patches.append([
                corner[0], corner[1], corner[2],
                lt, wj, strike, dip,
            ])
        cumulative_w += wj

    return np.array(patches)
