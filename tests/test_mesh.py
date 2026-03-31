"""Tests for geodef.mesh — Mesh dataclass, I/O, and helpers."""

import numpy as np
import numpy.testing as npt
import pytest

from geodef.mesh import Mesh, _compute_strike_dip


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def simple_mesh():
    """Two-triangle mesh forming a rectangle in geographic coords.

    Nodes:
        0: (100.0, 0.0, 0)      1: (100.1, 0.0, 0)
        2: (100.0, 0.1, 10000)   3: (100.1, 0.1, 10000)

    Triangles: [0,1,3], [0,3,2]
    """
    lon = np.array([100.0, 100.1, 100.0, 100.1])
    lat = np.array([0.0, 0.0, 0.1, 0.1])
    depth = np.array([0.0, 0.0, 10000.0, 10000.0])
    triangles = np.array([[0, 1, 3], [0, 3, 2]])
    return Mesh(lon=lon, lat=lat, depth=depth, triangles=triangles)


@pytest.fixture
def vertical_mesh():
    """Two-triangle mesh on a vertical N-S striking plane (dip=90, strike=0).

    All nodes at lon=100.0, varying lat and depth.
    Nodes in local ENU: x=0 (east), y varies (north), z varies (up, negative = depth).
    """
    lon = np.array([100.0, 100.0, 100.0, 100.0])
    lat = np.array([0.0, 0.1, 0.0, 0.1])
    depth = np.array([0.0, 0.0, 10000.0, 10000.0])
    triangles = np.array([[0, 1, 3], [0, 3, 2]])
    return Mesh(lon=lon, lat=lat, depth=depth, triangles=triangles)


@pytest.fixture
def horizontal_mesh():
    """Two-triangle mesh on a horizontal plane at 5 km depth (dip=0)."""
    lon = np.array([100.0, 100.1, 100.0, 100.1])
    lat = np.array([0.0, 0.0, 0.1, 0.1])
    depth = np.array([5000.0, 5000.0, 5000.0, 5000.0])
    triangles = np.array([[0, 1, 3], [0, 3, 2]])
    return Mesh(lon=lon, lat=lat, depth=depth, triangles=triangles)


# ======================================================================
# Mesh dataclass — construction and properties
# ======================================================================


class TestMeshConstruction:
    def test_basic_construction(self, simple_mesh):
        assert simple_mesh.n_nodes == 4
        assert simple_mesh.n_triangles == 2

    def test_arrays_stored(self, simple_mesh):
        assert simple_mesh.lon.shape == (4,)
        assert simple_mesh.lat.shape == (4,)
        assert simple_mesh.depth.shape == (4,)
        assert simple_mesh.triangles.shape == (2, 3)

    def test_frozen(self, simple_mesh):
        with pytest.raises(AttributeError):
            simple_mesh.lon = np.zeros(4)

    def test_centers_geo_shape(self, simple_mesh):
        centers = simple_mesh.centers_geo
        assert centers.shape == (2, 3)

    def test_centers_geo_values(self, simple_mesh):
        centers = simple_mesh.centers_geo
        # Triangle 0: nodes [0,1,3] -> mean lon=(100+100.1+100.1)/3, etc.
        npt.assert_allclose(centers[0, 0], (100.0 + 100.1 + 100.1) / 3)
        npt.assert_allclose(centers[0, 1], (0.0 + 0.0 + 0.1) / 3)
        npt.assert_allclose(centers[0, 2], (0.0 + 0.0 + 10000.0) / 3)

    def test_areas_positive(self, simple_mesh):
        areas = simple_mesh.areas
        assert areas.shape == (2,)
        assert np.all(areas > 0)

    def test_areas_roughly_equal_for_rectangle_split(self, simple_mesh):
        areas = simple_mesh.areas
        npt.assert_allclose(areas[0], areas[1], rtol=0.05)

    def test_n_nodes_n_triangles(self, simple_mesh):
        assert simple_mesh.n_nodes == len(simple_mesh.lon)
        assert simple_mesh.n_triangles == len(simple_mesh.triangles)


class TestMeshVerticesENU:
    def test_shape(self, simple_mesh):
        verts = simple_mesh.vertices_enu(ref_lat=0.0, ref_lon=100.0)
        assert verts.shape == (2, 3, 3)

    def test_z_convention(self, simple_mesh):
        """Depth positive down → z negative in ENU (up-positive)."""
        verts = simple_mesh.vertices_enu(ref_lat=0.0, ref_lon=100.0)
        # Nodes at depth=0 should have z≈0, nodes at depth=10000 should have z≈-10000
        # Triangle 0 has nodes 0,1,3 → depths 0, 0, 10000
        assert verts[0, 0, 2] == pytest.approx(0.0, abs=1.0)
        assert verts[0, 2, 2] == pytest.approx(-10000.0, abs=100.0)

    def test_enu_relative_to_ref(self, simple_mesh):
        """Vertices at ref point should have ~zero ENU."""
        verts = simple_mesh.vertices_enu(ref_lat=0.05, ref_lon=100.05)
        # All vertices should be within ~10 km of origin
        for i in range(2):
            for j in range(3):
                assert abs(verts[i, j, 0]) < 20000  # east
                assert abs(verts[i, j, 1]) < 20000  # north


# ======================================================================
# _compute_strike_dip helper
# ======================================================================


class TestComputeStrikeDip:
    def test_horizontal_plane_dip_zero(self, horizontal_mesh):
        """Horizontal triangles should have dip ≈ 0."""
        verts = horizontal_mesh.vertices_enu(ref_lat=0.05, ref_lon=100.05)
        strike, dip = _compute_strike_dip(verts)
        npt.assert_allclose(dip, 0.0, atol=1.0)

    def test_vertical_plane_dip_90(self, vertical_mesh):
        """Vertical N-S plane should have dip ≈ 90."""
        verts = vertical_mesh.vertices_enu(ref_lat=0.05, ref_lon=100.0)
        strike, dip = _compute_strike_dip(verts)
        npt.assert_allclose(dip, 90.0, atol=2.0)

    def test_vertical_ns_strike_zero_or_180(self, vertical_mesh):
        """Vertical N-S plane should have strike ≈ 0 or 180."""
        verts = vertical_mesh.vertices_enu(ref_lat=0.05, ref_lon=100.0)
        strike, dip = _compute_strike_dip(verts)
        # Strike should be ~0 or ~180 for a N-S plane
        for s in strike:
            assert s == pytest.approx(0.0, abs=5.0) or s == pytest.approx(
                180.0, abs=5.0
            )

    def test_known_45_degree_dip(self):
        """Construct a triangle with known 45-degree dip."""
        # Triangle on a plane dipping 45 degrees to the east, striking N-S
        # In ENU: the plane contains points along north (y) and
        # going equal parts east (x) and down (-z)
        v = np.array(
            [
                [
                    [0, 0, 0],
                    [0, 10000, 0],
                    [5000, 0, -5000],
                ]
            ],
            dtype=float,
        )
        strike, dip = _compute_strike_dip(v)
        npt.assert_allclose(dip[0], 45.0, atol=1.0)

    def test_output_shapes(self, simple_mesh):
        verts = simple_mesh.vertices_enu(ref_lat=0.0, ref_lon=100.0)
        strike, dip = _compute_strike_dip(verts)
        assert strike.shape == (2,)
        assert dip.shape == (2,)

    def test_dip_range(self, simple_mesh):
        """Dip should be in [0, 90]."""
        verts = simple_mesh.vertices_enu(ref_lat=0.0, ref_lon=100.0)
        strike, dip = _compute_strike_dip(verts)
        assert np.all(dip >= 0)
        assert np.all(dip <= 90)

    def test_strike_range(self, simple_mesh):
        """Strike should be in [0, 360)."""
        verts = simple_mesh.vertices_enu(ref_lat=0.0, ref_lon=100.0)
        strike, dip = _compute_strike_dip(verts)
        assert np.all(strike >= 0)
        assert np.all(strike < 360)


# ======================================================================
# Mesh I/O — save / load round-trip
# ======================================================================


class TestMeshIO:
    def test_save_creates_files(self, simple_mesh, tmp_path):
        fname = str(tmp_path / "test_mesh")
        simple_mesh.save(fname)
        assert (tmp_path / "test_mesh.ned").exists()
        assert (tmp_path / "test_mesh.tri").exists()

    def test_round_trip(self, simple_mesh, tmp_path):
        fname = str(tmp_path / "test_mesh")
        simple_mesh.save(fname)
        loaded = Mesh.load(fname)
        npt.assert_allclose(loaded.lon, simple_mesh.lon)
        npt.assert_allclose(loaded.lat, simple_mesh.lat)
        npt.assert_allclose(loaded.depth, simple_mesh.depth)
        npt.assert_array_equal(loaded.triangles, simple_mesh.triangles)

    def test_round_trip_preserves_n(self, simple_mesh, tmp_path):
        fname = str(tmp_path / "test_mesh")
        simple_mesh.save(fname)
        loaded = Mesh.load(fname)
        assert loaded.n_nodes == simple_mesh.n_nodes
        assert loaded.n_triangles == simple_mesh.n_triangles

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Mesh.load(str(tmp_path / "nonexistent"))

    def test_ned_format_is_default(self, simple_mesh, tmp_path):
        fname = str(tmp_path / "test_mesh")
        simple_mesh.save(fname, format="ned")
        loaded = Mesh.load(fname, format="ned")
        npt.assert_allclose(loaded.lon, simple_mesh.lon, atol=1e-4)

    def test_save_unsupported_format_raises(self, simple_mesh, tmp_path):
        fname = str(tmp_path / "test_mesh")
        with pytest.raises(ValueError, match="format"):
            simple_mesh.save(fname, format="xyz")

    def test_load_unsupported_format_raises(self, tmp_path):
        with pytest.raises(ValueError, match="format"):
            Mesh.load(str(tmp_path / "test_mesh"), format="xyz")


# ======================================================================
# _nan_aware_griddata
# ======================================================================


# ======================================================================
# Fault.from_triangles() and Fault.from_mesh()
# ======================================================================


class TestFaultFromTriangles:
    def test_basic_construction(self):
        from geodef.fault import Fault

        # Two triangles forming a vertical N-S plane
        verts = np.array(
            [
                [[0, 0, 0], [0, 10000, 0], [0, 0, -10000]],
                [[0, 10000, 0], [0, 10000, -10000], [0, 0, -10000]],
            ],
            dtype=float,
        )
        fault = Fault.from_triangles(verts, ref_lat=0.0, ref_lon=100.0)
        assert fault.n_patches == 2
        assert fault.engine == "tri"

    def test_strike_dip_derived(self):
        from geodef.fault import Fault

        # Vertical N-S plane: dip=90, strike=0 or 180
        verts = np.array(
            [
                [[0, 0, 0], [0, 10000, 0], [0, 0, -10000]],
            ],
            dtype=float,
        )
        fault = Fault.from_triangles(verts, ref_lat=0.0, ref_lon=100.0)
        assert fault._dip[0] == pytest.approx(90.0, abs=2.0)

    def test_centers_geographic(self):
        from geodef.fault import Fault

        # Single triangle centered near origin
        verts = np.array(
            [
                [[-5000, -5000, -5000], [5000, -5000, -5000], [0, 5000, -5000]],
            ],
            dtype=float,
        )
        fault = Fault.from_triangles(verts, ref_lat=0.0, ref_lon=100.0)
        # Centers should be near ref point
        assert fault._lat[0] == pytest.approx(0.0, abs=0.1)
        assert fault._lon[0] == pytest.approx(100.0, abs=0.1)

    def test_areas_positive(self):
        from geodef.fault import Fault

        verts = np.array(
            [
                [[0, 0, 0], [10000, 0, 0], [0, 10000, 0]],
            ],
            dtype=float,
        )
        fault = Fault.from_triangles(verts, ref_lat=0.0, ref_lon=100.0)
        assert fault.areas[0] > 0

    def test_vertices_stored(self):
        from geodef.fault import Fault

        verts = np.array(
            [
                [[0, 0, 0], [10000, 0, 0], [0, 10000, 0]],
            ],
            dtype=float,
        )
        fault = Fault.from_triangles(verts, ref_lat=0.0, ref_lon=100.0)
        assert fault._vertices is not None
        assert fault._vertices.shape == (1, 3, 3)

    def test_depth_from_vertices(self):
        from geodef.fault import Fault

        # Triangle at 10 km depth (z = -10000 in ENU)
        verts = np.array(
            [
                [[0, 0, -10000], [10000, 0, -10000], [0, 10000, -10000]],
            ],
            dtype=float,
        )
        fault = Fault.from_triangles(verts, ref_lat=0.0, ref_lon=100.0)
        assert fault._depth[0] == pytest.approx(10000.0, abs=100.0)


class TestFaultFromMesh:
    def test_basic(self, simple_mesh):
        from geodef.fault import Fault

        fault = Fault.from_mesh(simple_mesh)
        assert fault.n_patches == simple_mesh.n_triangles
        assert fault.engine == "tri"

    def test_vertices_shape(self, simple_mesh):
        from geodef.fault import Fault

        fault = Fault.from_mesh(simple_mesh)
        assert fault._vertices.shape == (2, 3, 3)

    def test_strike_dip_reasonable(self, simple_mesh):
        from geodef.fault import Fault

        fault = Fault.from_mesh(simple_mesh)
        assert np.all(fault._dip >= 0)
        assert np.all(fault._dip <= 90)
        assert np.all(fault._strike >= 0)
        assert np.all(fault._strike < 360)

    def test_centers_within_mesh(self, simple_mesh):
        from geodef.fault import Fault

        fault = Fault.from_mesh(simple_mesh)
        assert np.all(fault._lat >= -0.1) and np.all(fault._lat <= 0.2)
        assert np.all(fault._lon >= 99.9) and np.all(fault._lon <= 100.2)

    def test_horizontal_mesh_dip_zero(self, horizontal_mesh):
        from geodef.fault import Fault

        fault = Fault.from_mesh(horizontal_mesh)
        npt.assert_allclose(fault._dip, 0.0, atol=1.0)

    def test_forward_modeling_works(self, simple_mesh):
        """Smoke test: mesh → fault → displacement doesn't error."""
        from geodef.fault import Fault

        fault = Fault.from_mesh(simple_mesh)
        obs_lat = np.array([0.05])
        obs_lon = np.array([100.05])
        slip_s = np.ones(fault.n_patches)
        ux, uy, uz = fault.displacement(obs_lat, obs_lon, slip_s)
        assert ux.shape == (1,)


class TestFaultLoadNed:
    def test_load_ned_format(self, simple_mesh, tmp_path):
        from geodef.fault import Fault

        fname = str(tmp_path / "test_mesh")
        simple_mesh.save(fname)
        fault = Fault.load(
            fname, format="ned", ref_lat=0.05, ref_lon=100.05
        )
        assert fault.engine == "tri"
        assert fault.n_patches == 2

    def test_load_ned_requires_ref(self, simple_mesh, tmp_path):
        from geodef.fault import Fault

        fname = str(tmp_path / "test_mesh")
        simple_mesh.save(fname)
        with pytest.raises(ValueError, match="ref_lat.*ref_lon"):
            Fault.load(fname, format="ned")


# ======================================================================
# _nan_aware_griddata
# ======================================================================


# ======================================================================
# from_polygon()
# ======================================================================


class TestFromPolygon:
    def test_simple_rectangle(self):
        from geodef.mesh import from_polygon

        lon = np.array([100.0, 100.1, 100.1, 100.0])
        lat = np.array([0.0, 0.0, 0.1, 0.1])
        depth = np.array([0.0, 0.0, 10000.0, 10000.0])
        mesh = from_polygon(lon, lat, depth, target_length=5000.0)
        assert mesh.n_triangles > 0
        assert mesh.n_nodes > 4

    def test_all_triangles_positive_area(self):
        from geodef.mesh import from_polygon

        lon = np.array([100.0, 100.1, 100.1, 100.0])
        lat = np.array([0.0, 0.0, 0.1, 0.1])
        depth = np.array([0.0, 0.0, 10000.0, 10000.0])
        mesh = from_polygon(lon, lat, depth, target_length=5000.0)
        assert np.all(mesh.areas > 0)

    def test_no_degenerate_triangles(self):
        from geodef.mesh import from_polygon

        lon = np.array([100.0, 100.1, 100.1, 100.0])
        lat = np.array([0.0, 0.0, 0.1, 0.1])
        depth = np.array([0.0, 0.0, 10000.0, 10000.0])
        mesh = from_polygon(lon, lat, depth, target_length=5000.0)
        assert np.all(mesh.areas > 1e-6)

    def test_nodes_within_bounds(self):
        from geodef.mesh import from_polygon

        lon = np.array([100.0, 100.1, 100.1, 100.0])
        lat = np.array([0.0, 0.0, 0.1, 0.1])
        depth = np.array([0.0, 0.0, 10000.0, 10000.0])
        mesh = from_polygon(lon, lat, depth, target_length=5000.0)
        assert np.all(mesh.lon >= 99.99)
        assert np.all(mesh.lon <= 100.11)
        assert np.all(mesh.lat >= -0.01)
        assert np.all(mesh.lat <= 0.11)

    def test_depth_func_mode(self):
        from geodef.mesh import from_polygon

        lon = np.array([100.0, 100.1, 100.1, 100.0])
        lat = np.array([0.0, 0.0, 0.1, 0.1])

        def depth_func(lo, la):
            return la * 100000.0

        mesh = from_polygon(
            lon, lat, depth_func=depth_func, target_length=5000.0
        )
        assert mesh.n_triangles > 0
        assert mesh.depth.max() > mesh.depth.min()

    def test_max_area_parameter(self):
        from geodef.mesh import from_polygon

        lon = np.array([100.0, 100.1, 100.1, 100.0])
        lat = np.array([0.0, 0.0, 0.1, 0.1])
        depth = np.array([0.0, 0.0, 10000.0, 10000.0])
        # max_area is in m^2 (ENU coords) for 3D polygon mode
        coarse = from_polygon(lon, lat, depth, max_area=50e6)
        fine = from_polygon(lon, lat, depth, max_area=10e6)
        assert fine.n_triangles > coarse.n_triangles

    def test_target_length_and_max_area_exclusive(self):
        from geodef.mesh import from_polygon

        lon = np.array([100.0, 100.1, 100.1, 100.0])
        lat = np.array([0.0, 0.0, 0.1, 0.1])
        depth = np.array([0.0, 0.0, 10000.0, 10000.0])
        with pytest.raises(ValueError, match="mutually exclusive"):
            from_polygon(
                lon, lat, depth, target_length=5000.0, max_area=1e-3
            )

    def test_requires_depth_or_depth_func(self):
        from geodef.mesh import from_polygon

        lon = np.array([100.0, 100.1, 100.1, 100.0])
        lat = np.array([0.0, 0.0, 0.1, 0.1])
        with pytest.raises(ValueError, match="depth"):
            from_polygon(lon, lat)

    def test_surface_edge_at_zero_depth(self):
        """Polygon with a surface edge: nodes on that edge must be depth=0."""
        from geodef.mesh import from_polygon

        lon = np.array([100.0, 100.1, 100.1, 100.0])
        lat = np.array([0.0, 0.0, 0.1, 0.1])
        depth = np.array([0.0, 0.0, 10000.0, 10000.0])
        mesh = from_polygon(lon, lat, depth, target_length=3000.0)
        # Nodes near lat=0 (the surface edge) must have depth ≈ 0
        surface_mask = mesh.depth < 1.0
        assert np.sum(surface_mask) >= 2
        npt.assert_allclose(mesh.depth[surface_mask], 0.0, atol=0.01)

    def test_surface_edge_exact_steep_dip(self):
        """Steep polygon (nearly vertical): surface edge must be exact."""
        from geodef.mesh import from_polygon

        # Nearly vertical fault: surface at depth=0, bottom at 30km
        lon = np.array([100.0, 100.2, 100.201, 100.001])
        lat = np.array([0.0, 0.0, 0.0, 0.0])
        depth = np.array([0.0, 0.0, 30000.0, 30000.0])
        mesh = from_polygon(lon, lat, depth, target_length=5000.0)
        surface_mask = mesh.depth < 1.0
        assert np.sum(surface_mask) >= 2
        npt.assert_allclose(mesh.depth[surface_mask], 0.0, atol=0.01)

    def test_mesh_to_fault_pipeline(self):
        from geodef.fault import Fault
        from geodef.mesh import from_polygon

        lon = np.array([100.0, 100.1, 100.1, 100.0])
        lat = np.array([0.0, 0.0, 0.1, 0.1])
        depth = np.array([0.0, 0.0, 10000.0, 10000.0])
        mesh = from_polygon(lon, lat, depth, target_length=5000.0)
        fault = Fault.from_mesh(mesh)
        assert fault.engine == "tri"
        assert fault.n_patches == mesh.n_triangles


# ======================================================================
# _nan_aware_griddata
# ======================================================================


# ======================================================================
# from_trace()
# ======================================================================


class TestFromTrace:
    def test_simple_constant_dip(self):
        from geodef.mesh import from_trace

        trace_lon = np.array([100.0, 100.2])
        trace_lat = np.array([0.0, 0.0])
        mesh = from_trace(
            trace_lon, trace_lat,
            max_depth=30.0,
            dip=30.0,
            target_length=10000.0,
        )
        assert mesh.n_triangles > 0
        assert mesh.n_nodes > 2

    def test_all_areas_positive(self):
        from geodef.mesh import from_trace

        trace_lon = np.array([100.0, 100.2])
        trace_lat = np.array([0.0, 0.0])
        mesh = from_trace(
            trace_lon, trace_lat,
            max_depth=30.0,
            dip=30.0,
            target_length=10000.0,
        )
        assert np.all(mesh.areas > 0)

    def test_max_depth_honored(self):
        from geodef.mesh import from_trace

        trace_lon = np.array([100.0, 100.2])
        trace_lat = np.array([0.0, 0.0])
        max_depth = 30.0
        mesh = from_trace(
            trace_lon, trace_lat,
            max_depth=max_depth,
            dip=45.0,
            target_length=10000.0,
        )
        assert mesh.depth.max() <= max_depth * 1000 * 1.1

    def test_callable_dip(self):
        """Variable dip (listric): dip increases with depth."""
        from geodef.mesh import from_trace

        trace_lon = np.array([100.0, 100.2])
        trace_lat = np.array([0.0, 0.0])
        mesh = from_trace(
            trace_lon, trace_lat,
            max_depth=30.0,
            dip=lambda z: 10 + 40 * z / 30000,  # 10° at surface, 50° at depth
            target_length=10000.0,
        )
        assert mesh.n_triangles > 0

    def test_explicit_dip_direction(self):
        from geodef.mesh import from_trace

        trace_lon = np.array([100.0, 100.2])
        trace_lat = np.array([0.0, 0.0])
        mesh = from_trace(
            trace_lon, trace_lat,
            max_depth=30.0,
            dip=30.0,
            dip_direction=0.0,  # dip to the north
            target_length=10000.0,
        )
        assert mesh.n_triangles > 0
        # Nodes should extend northward (positive lat)
        assert mesh.lat.max() > 0.01

    def test_curved_trace(self):
        from geodef.mesh import from_trace

        # L-shaped trace
        trace_lon = np.array([100.0, 100.1, 100.1])
        trace_lat = np.array([0.0, 0.0, 0.1])
        mesh = from_trace(
            trace_lon, trace_lat,
            max_depth=20.0,
            dip=30.0,
            target_length=10000.0,
        )
        assert mesh.n_triangles > 0

    def test_mesh_to_fault(self):
        from geodef.fault import Fault
        from geodef.mesh import from_trace

        trace_lon = np.array([100.0, 100.2])
        trace_lat = np.array([0.0, 0.0])
        mesh = from_trace(
            trace_lon, trace_lat,
            max_depth=30.0,
            dip=30.0,
            target_length=10000.0,
        )
        fault = Fault.from_mesh(mesh)
        assert fault.engine == "tri"
        assert fault.n_patches == mesh.n_triangles

    def test_surface_trace_at_zero_depth(self):
        from geodef.mesh import from_trace

        trace_lon = np.array([100.0, 100.2])
        trace_lat = np.array([0.0, 0.0])
        mesh = from_trace(
            trace_lon, trace_lat,
            max_depth=30.0,
            dip=30.0,
            target_length=10000.0,
        )
        # Surface nodes must have exactly zero depth (within float precision)
        surface_mask = mesh.depth < 1.0  # nodes near the surface
        assert np.sum(surface_mask) >= 2  # at least the trace endpoints
        npt.assert_allclose(mesh.depth[surface_mask], 0.0, atol=0.01)

    def test_surface_trace_exact_for_steep_dip(self):
        """Even with steep dip (where PCA plane would be badly tilted),
        surface trace nodes must be exactly at depth=0."""
        from geodef.mesh import from_trace

        trace_lon = np.array([100.0, 100.1, 100.2])
        trace_lat = np.array([0.0, 0.05, 0.0])
        mesh = from_trace(
            trace_lon, trace_lat,
            max_depth=50.0,
            dip=80.0,
            target_length=5000.0,
        )
        surface_mask = mesh.depth < 1.0
        assert np.sum(surface_mask) >= 3
        npt.assert_allclose(mesh.depth[surface_mask], 0.0, atol=0.01)

    def test_surface_trace_exact_for_curved_trace(self):
        """Curved trace: all surface nodes must be at depth=0."""
        from geodef.mesh import from_trace

        # Arc-shaped trace
        theta = np.linspace(0, np.pi / 2, 10)
        trace_lon = 100.0 + 0.2 * np.cos(theta)
        trace_lat = 0.2 * np.sin(theta)
        mesh = from_trace(
            trace_lon, trace_lat,
            max_depth=20.0,
            dip=45.0,
            target_length=5000.0,
        )
        surface_mask = mesh.depth < 1.0
        assert np.sum(surface_mask) >= 5
        npt.assert_allclose(mesh.depth[surface_mask], 0.0, atol=0.01)

    def test_surface_trace_exact_for_listric_dip(self):
        """Listric (variable) dip: surface nodes must be at depth=0."""
        from geodef.mesh import from_trace

        trace_lon = np.array([100.0, 100.3])
        trace_lat = np.array([0.0, 0.0])
        mesh = from_trace(
            trace_lon, trace_lat,
            max_depth=40.0,
            dip=lambda z: 10 + 60 * z / 40000,
            target_length=8000.0,
        )
        surface_mask = mesh.depth < 1.0
        assert np.sum(surface_mask) >= 2
        npt.assert_allclose(mesh.depth[surface_mask], 0.0, atol=0.01)


# ======================================================================
# _trace_grid_boundary()
# ======================================================================


class TestTraceGridBoundary:
    def test_rectangle(self):
        """Rectangular valid region should trace all 4 edges."""
        from geodef.mesh import _trace_grid_boundary

        X, Y = np.meshgrid(np.arange(5.0), np.arange(4.0))
        valid = np.ones((4, 5), dtype=bool)
        boundary = _trace_grid_boundary(X, Y, valid)
        assert boundary.shape[1] == 2
        # All edge cells should be boundary (4+5+4+5 - 4 corners = 14)
        # but the polygon is ordered, so we get 14 unique points
        assert len(boundary) == 14

    def test_concave_l_shape(self):
        """L-shaped region should produce a concave boundary."""
        from geodef.mesh import _trace_grid_boundary

        # L-shape: top-left quadrant is invalid
        valid = np.ones((6, 6), dtype=bool)
        valid[:3, :3] = False
        X, Y = np.meshgrid(np.arange(6.0), np.arange(6.0))
        boundary = _trace_grid_boundary(X, Y, valid)

        # Verify the boundary is concave: the centroid of the L's
        # empty quadrant should NOT be inside the polygon
        from matplotlib.path import Path

        path = Path(boundary)
        # Point in the missing top-left area
        assert not path.contains_point((1.0, 1.0))
        # Point in the valid bottom-right area
        assert path.contains_point((4.0, 4.0))

    def test_concave_arc_shape(self):
        """Arc-shaped region (like a curved slab) should be concave."""
        from geodef.mesh import _trace_grid_boundary

        # Create a concave arc: valid region only on one side of a curve
        X, Y = np.meshgrid(np.arange(20.0), np.arange(20.0))
        valid = np.zeros((20, 20), dtype=bool)
        for row in range(20):
            # The left boundary curves inward (concave)
            left = int(5 + 5 * np.sin(np.pi * row / 19))
            valid[row, left:18] = True

        boundary = _trace_grid_boundary(X, Y, valid)

        from matplotlib.path import Path

        path = Path(boundary)
        # Point in the concave indentation (left of the curve) should be outside
        assert not path.contains_point((2.0, 10.0))
        # Point in the valid region should be inside
        assert path.contains_point((15.0, 10.0))

    def test_subsample(self):
        """Subsampling should reduce boundary points."""
        from geodef.mesh import _trace_grid_boundary

        X, Y = np.meshgrid(np.arange(10.0), np.arange(10.0))
        valid = np.ones((10, 10), dtype=bool)
        full = _trace_grid_boundary(X, Y, valid)
        sub3 = _trace_grid_boundary(X, Y, valid, subsample=3)
        assert len(sub3) < len(full)

    def test_empty_raises(self):
        """Empty valid region should raise ValueError."""
        from geodef.mesh import _trace_grid_boundary

        X, Y = np.meshgrid(np.arange(5.0), np.arange(5.0))
        valid = np.zeros((5, 5), dtype=bool)
        with pytest.raises(ValueError, match="boundary"):
            _trace_grid_boundary(X, Y, valid)

    def test_slab_like_concavity(self):
        """Slab-like shape (curved trace + downdip) is smaller than hull."""
        from geodef.mesh import _trace_grid_boundary

        # Simulate a slab: left boundary curves inward significantly
        X, Y = np.meshgrid(np.linspace(90, 100, 50), np.linspace(-5, 5, 50))
        valid = np.zeros((50, 50), dtype=bool)
        for row in range(50):
            # Left edge curves in by up to 10 columns at the center
            left = int(10 * np.sin(np.pi * row / 49) ** 2)
            valid[row, left:45] = True

        boundary = _trace_grid_boundary(X, Y, valid)
        boundary_area = _shoelace_area(boundary)

        from scipy.spatial import ConvexHull

        valid_pts = np.column_stack([X[valid], Y[valid]])
        hull = ConvexHull(valid_pts)
        hull_area = hull.volume

        # Traced boundary should have noticeably less area than convex hull
        assert boundary_area < hull_area * 0.90


def _shoelace_area(polygon: np.ndarray) -> float:
    """Compute polygon area via the shoelace formula."""
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
                      + x[-1] * y[0] - x[0] * y[-1])


# ======================================================================
# _simplify_boundary()
# ======================================================================


class TestSimplifyBoundary:
    def test_variable_spacing_preserves_dense_region(self):
        """Points in the small-spacing region should be kept denser."""
        from geodef.mesh import _simplify_boundary

        n = 100
        top = np.column_stack([np.linspace(0, 1, n // 2), np.zeros(n // 2)])
        right = np.column_stack([np.ones(2), [0.0, 1.0]])
        bottom = np.column_stack([
            np.linspace(1, 0, n // 2), np.ones(n // 2)
        ])
        left = np.column_stack([np.zeros(2), [1.0, 0.0]])
        boundary = np.vstack([top, right[1:], bottom[1:], left[1:]])

        # Small spacing near y=0, large near y=1
        def spacing(lon, lat):
            return 0.05 + 0.5 * lat

        result = _simplify_boundary(boundary, spacing)
        shallow = np.sum(result[:, 1] < 0.1)
        deep = np.sum(result[:, 1] > 0.9)
        assert shallow > deep

    def test_constant_spacing_thins(self):
        """Constant spacing function should thin a dense polygon."""
        from geodef.mesh import _simplify_boundary

        n = 200
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        boundary = np.column_stack([np.cos(theta), np.sin(theta)])

        result = _simplify_boundary(boundary, lambda lon, lat: 0.5)
        assert len(result) < n
        assert len(result) >= 3

    def test_always_keeps_first_and_last(self):
        """First and last boundary points are always retained."""
        from geodef.mesh import _simplify_boundary

        boundary = np.array([
            [0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0],
            [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        ])
        result = _simplify_boundary(boundary, lambda lon, lat: 0.5)
        npt.assert_array_equal(result[0], boundary[0])
        npt.assert_array_equal(result[-1], boundary[-1])


# ======================================================================
# from_slab2()
# ======================================================================


class TestKmToDeg:
    def test_equator(self):
        """At the equator, 111 km ≈ 1 degree."""
        from geodef.mesh import _km_to_deg

        result = _km_to_deg(111.0, 0.0)
        npt.assert_allclose(result, 1.0, atol=0.01)

    def test_high_latitude(self):
        """At 60N, 111 km should be more than 1 degree."""
        from geodef.mesh import _km_to_deg

        result = _km_to_deg(111.0, 60.0)
        assert result > 1.0

    def test_scales_linearly(self):
        """Double the km should give double the degrees."""
        from geodef.mesh import _km_to_deg

        npt.assert_allclose(
            _km_to_deg(200.0, 30.0),
            2.0 * _km_to_deg(100.0, 30.0),
        )


class TestFromSlab2:
    def test_requires_netcdf4(self, monkeypatch):
        """from_slab2 should give a clear error when netCDF4 is missing."""
        import geodef.mesh as mesh_mod

        def mock_require():
            raise ImportError("netCDF4 is required")

        monkeypatch.setattr(mesh_mod, "_require_netcdf4", mock_require)
        with pytest.raises(ImportError, match="netCDF4"):
            mesh_mod.from_slab2("fake.grd", bounds=(0, 1, 0, 1))

    def test_missing_file_raises(self):
        """from_slab2 should raise on nonexistent file."""
        from geodef.mesh import from_slab2

        try:
            from netCDF4 import Dataset  # noqa: F401
        except ImportError:
            pytest.skip("netCDF4 not installed")

        with pytest.raises(Exception):
            from_slab2("/nonexistent/file.grd", bounds=(0, 1, 0, 1))

    def test_depth_growth_validation(self):
        """depth_growth < 1 should raise ValueError."""
        from geodef.mesh import from_slab2

        with pytest.raises(ValueError, match="depth_growth"):
            from_slab2("fake.grd", bounds=(0, 1, 0, 1), depth_growth=0.5)

    def test_max_depth_clips_valid_region(self, monkeypatch):
        """max_depth should NaN-out cells deeper than the threshold."""
        import geodef.mesh as mesh_mod

        # Synthetic slab: depth increases linearly with latitude
        lons = np.linspace(99, 101, 30)
        lats = np.linspace(-1, 1, 30)
        X, Y = np.meshgrid(lons, lats)
        # Depth in km, negative = down: 0 at lat=1, -200 at lat=-1
        Z = (Y - 1.0) * 100.0  # ranges from 0 to -200 km

        def mock_netcdf4():
            """Return a mock Dataset class."""
            class FakeVar:
                def __init__(self, data):
                    self._data = data
                def __getitem__(self, key):
                    return self._data[key]
            class FakeDS:
                def __init__(self, fname, mode="r"):
                    self.variables = {
                        "x": FakeVar(lons),
                        "y": FakeVar(lats),
                        "z": FakeVar(Z),
                    }
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return FakeDS

        monkeypatch.setattr(mesh_mod, "_require_netcdf4", mock_netcdf4)

        # Without max_depth: full slab, max depth ≈ 200 km
        mesh_full = mesh_mod.from_slab2(
            "fake.grd", bounds=(99, 101, -1, 1), target_length=30.0,
        )
        # With max_depth=100 km: should clip roughly in half
        mesh_clipped = mesh_mod.from_slab2(
            "fake.grd", bounds=(99, 101, -1, 1),
            target_length=30.0, max_depth=100.0,
        )
        assert mesh_clipped.depth.max() <= 105_000  # 100 km + tolerance
        assert mesh_clipped.depth.max() < mesh_full.depth.max()

    def test_max_depth_reduces_extent(self, monkeypatch):
        """max_depth should reduce the spatial extent of the mesh."""
        import geodef.mesh as mesh_mod

        lons = np.linspace(99, 101, 30)
        lats = np.linspace(-1, 1, 30)
        X, Y = np.meshgrid(lons, lats)
        Z = (Y - 1.0) * 100.0

        def mock_netcdf4():
            class FakeVar:
                def __init__(self, data):
                    self._data = data
                def __getitem__(self, key):
                    return self._data[key]
            class FakeDS:
                def __init__(self, fname, mode="r"):
                    self.variables = {
                        "x": FakeVar(lons),
                        "y": FakeVar(lats),
                        "z": FakeVar(Z),
                    }
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return FakeDS

        monkeypatch.setattr(mesh_mod, "_require_netcdf4", mock_netcdf4)

        mesh_full = mesh_mod.from_slab2(
            "fake.grd", bounds=(99, 101, -1, 1), target_length=30.0,
        )
        mesh_clipped = mesh_mod.from_slab2(
            "fake.grd", bounds=(99, 101, -1, 1),
            target_length=30.0, max_depth=50.0,
        )
        # Clipped mesh should cover less latitude range
        lat_range_full = mesh_full.lat.max() - mesh_full.lat.min()
        lat_range_clip = mesh_clipped.lat.max() - mesh_clipped.lat.min()
        assert lat_range_clip < lat_range_full * 0.8

    def test_surface_trace_extends_to_zero(self, monkeypatch):
        """surface_trace should extend the mesh up to depth=0."""
        import geodef.mesh as mesh_mod

        lons = np.linspace(99, 101, 30)
        lats = np.linspace(-1, 1, 30)
        X, Y = np.meshgrid(lons, lats)
        # Slab that doesn't reach the surface: starts at -20 km
        Z = (Y - 1.0) * 100.0 - 20.0  # -20 to -220 km

        def mock_netcdf4():
            class FakeVar:
                def __init__(self, data):
                    self._data = data
                def __getitem__(self, key):
                    return self._data[key]
            class FakeDS:
                def __init__(self, fname, mode="r"):
                    self.variables = {
                        "x": FakeVar(lons),
                        "y": FakeVar(lats),
                        "z": FakeVar(Z),
                    }
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return FakeDS

        monkeypatch.setattr(mesh_mod, "_require_netcdf4", mock_netcdf4)

        # Surface trace: a line at lat≈1.2, beyond the slab's shallow edge
        trace_lon = np.array([99.2, 100.0, 100.8])
        trace_lat = np.array([1.3, 1.3, 1.3])

        mesh = mesh_mod.from_slab2(
            "fake.grd", bounds=(99, 101, -1, 1),
            target_length=30.0,
            surface_trace=(trace_lon, trace_lat),
        )
        # Should have nodes at depth ≈ 0
        assert mesh.depth.min() < 1000  # within 1 km of surface
        # The trace nodes should be in the mesh
        assert mesh.lat.max() > 1.2

    def test_surface_trace_with_max_depth(self, monkeypatch):
        """surface_trace and max_depth should compose correctly."""
        import geodef.mesh as mesh_mod

        lons = np.linspace(99, 101, 30)
        lats = np.linspace(-1, 1, 30)
        X, Y = np.meshgrid(lons, lats)
        Z = (Y - 1.0) * 100.0 - 20.0  # -20 to -220 km

        def mock_netcdf4():
            class FakeVar:
                def __init__(self, data):
                    self._data = data
                def __getitem__(self, key):
                    return self._data[key]
            class FakeDS:
                def __init__(self, fname, mode="r"):
                    self.variables = {
                        "x": FakeVar(lons),
                        "y": FakeVar(lats),
                        "z": FakeVar(Z),
                    }
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return FakeDS

        monkeypatch.setattr(mesh_mod, "_require_netcdf4", mock_netcdf4)

        trace_lon = np.array([99.2, 100.0, 100.8])
        trace_lat = np.array([1.3, 1.3, 1.3])

        mesh = mesh_mod.from_slab2(
            "fake.grd", bounds=(99, 101, -1, 1),
            target_length=30.0,
            max_depth=100.0,
            surface_trace=(trace_lon, trace_lat),
        )
        # Extended to surface
        assert mesh.depth.min() < 1000
        assert mesh.lat.max() > 1.2
        # Clipped at depth
        assert mesh.depth.max() <= 105_000


# ======================================================================
# from_points()
# ======================================================================


class TestFromPoints:
    def test_basic(self):
        from geodef.mesh import from_points

        # Scattered points on a tilted plane
        rng = np.random.default_rng(42)
        lon = 100.0 + 0.1 * rng.random(20)
        lat = 0.1 * rng.random(20)
        depth = lat * 100000.0  # depth increases with lat
        mesh = from_points(lon, lat, depth, target_length=5000.0)
        assert mesh.n_triangles > 0

    def test_all_areas_positive(self):
        from geodef.mesh import from_points

        rng = np.random.default_rng(42)
        lon = 100.0 + 0.1 * rng.random(20)
        lat = 0.1 * rng.random(20)
        depth = lat * 100000.0
        mesh = from_points(lon, lat, depth, target_length=5000.0)
        assert np.all(mesh.areas > 0)

    def test_user_supplied_boundary(self):
        from geodef.mesh import from_points

        rng = np.random.default_rng(42)
        lon = 100.0 + 0.08 * rng.random(30)
        lat = 0.02 + 0.06 * rng.random(30)
        depth = lat * 100000.0
        boundary = np.array([
            [100.0, 0.0],
            [100.1, 0.0],
            [100.1, 0.1],
            [100.0, 0.1],
        ])
        mesh = from_points(
            lon, lat, depth, boundary=boundary, target_length=5000.0
        )
        assert mesh.n_triangles > 0

    def test_interpolation_quality(self):
        from geodef.mesh import from_points

        # Dense grid of points on a known plane: depth = 100000 * lat
        rng = np.random.default_rng(123)
        lon = 100.0 + 0.1 * rng.random(50)
        lat = 0.1 * rng.random(50)
        depth = lat * 100000.0
        mesh = from_points(lon, lat, depth, target_length=5000.0)
        # Interior depths should follow the linear trend
        for i in range(mesh.n_nodes):
            expected = mesh.lat[i] * 100000.0
            assert mesh.depth[i] == pytest.approx(expected, abs=1000.0)

    def test_surface_boundary_at_zero_depth(self):
        """Points with a surface boundary edge: nodes on that edge at depth=0."""
        from geodef.mesh import from_points

        rng = np.random.default_rng(42)
        lon = 100.0 + 0.08 * rng.random(30)
        lat = 0.02 + 0.06 * rng.random(30)
        depth = lat * 100000.0  # depth ~ 0 at lat=0
        # Boundary with a surface edge at lat=0 (depth=0)
        boundary = np.array([
            [100.0, 0.0],
            [100.1, 0.0],
            [100.1, 0.1],
            [100.0, 0.1],
        ])
        mesh = from_points(
            lon, lat, depth, boundary=boundary, target_length=5000.0
        )
        # Nodes on the lat=0 boundary edge should have depth ≈ 0
        surface_mask = mesh.depth < 1.0
        assert np.sum(surface_mask) >= 2
        npt.assert_allclose(mesh.depth[surface_mask], 0.0, atol=0.01)

    def test_mesh_to_fault(self):
        from geodef.fault import Fault
        from geodef.mesh import from_points

        rng = np.random.default_rng(42)
        lon = 100.0 + 0.1 * rng.random(20)
        lat = 0.1 * rng.random(20)
        depth = lat * 100000.0
        mesh = from_points(lon, lat, depth, target_length=5000.0)
        fault = Fault.from_mesh(mesh)
        assert fault.engine == "tri"


# ======================================================================
# _nan_aware_griddata
# ======================================================================


# ======================================================================
# End-to-end integration
# ======================================================================


class TestIntegration:
    def test_from_trace_to_fault_to_greens(self):
        """Full pipeline: from_trace → Fault → greens matrix."""
        from geodef.fault import Fault
        from geodef.mesh import from_trace
        from geodef.data import GNSS
        import geodef.greens as greens_mod

        mesh = from_trace(
            trace_lon=np.array([100.0, 100.2]),
            trace_lat=np.array([0.0, 0.0]),
            max_depth=30.0,
            dip=30.0,
            target_length=15000.0,
        )
        fault = Fault.from_mesh(mesh)

        # Create a simple GNSS dataset
        gnss = GNSS(
            lat=np.array([0.05]),
            lon=np.array([100.1]),
            ve=np.array([0.01]),
            vn=np.array([0.02]),
            vu=np.array([0.0]),
            se=np.array([0.001]),
            sn=np.array([0.001]),
            su=np.array([0.001]),
        )

        G = greens_mod.greens(fault, gnss)
        assert G.shape[0] == 3  # 3 components
        assert G.shape[1] == fault.n_patches * 2  # ss + ds

    def test_from_polygon_save_load_roundtrip(self, tmp_path):
        """Mesh → save → load → Fault round-trip."""
        from geodef.fault import Fault
        from geodef.mesh import from_polygon

        lon = np.array([100.0, 100.1, 100.1, 100.0])
        lat = np.array([0.0, 0.0, 0.1, 0.1])
        depth = np.array([5000.0, 5000.0, 20000.0, 20000.0])
        mesh = from_polygon(lon, lat, depth, target_length=8000.0)

        fname = str(tmp_path / "mesh")
        mesh.save(fname)
        loaded = Mesh.load(fname)

        fault1 = Fault.from_mesh(mesh)
        fault2 = Fault.from_mesh(loaded)
        assert fault1.n_patches == fault2.n_patches
        npt.assert_allclose(fault1.areas, fault2.areas, rtol=1e-3)

    def test_mesh_module_accessible(self):
        """geodef.mesh is importable from the top-level package."""
        import geodef

        assert hasattr(geodef, "mesh")
        assert hasattr(geodef.mesh, "Mesh")
        assert hasattr(geodef.mesh, "from_trace")
        assert hasattr(geodef.mesh, "from_polygon")
        assert hasattr(geodef.mesh, "from_points")
        assert hasattr(geodef.mesh, "from_slab2")

    def test_fault_from_mesh_docstring(self):
        """Fault.from_mesh has a docstring."""
        from geodef.fault import Fault

        assert Fault.from_mesh.__doc__ is not None

    def test_fault_from_triangles_docstring(self):
        """Fault.from_triangles has a docstring."""
        from geodef.fault import Fault

        assert Fault.from_triangles.__doc__ is not None


# ======================================================================
# _nan_aware_griddata
# ======================================================================


class TestNanAwareGriddata:
    def test_basic_interpolation(self):
        from geodef.mesh import _nan_aware_griddata

        xin = np.array([0.0, 1.0, 0.0, 1.0])
        yin = np.array([0.0, 0.0, 1.0, 1.0])
        zin = np.array([0.0, 1.0, 1.0, 2.0])
        xout = np.array([0.5])
        yout = np.array([0.5])
        zout = _nan_aware_griddata(xin, yin, zin, xout, yout)
        npt.assert_allclose(zout, [1.0], atol=0.1)

    def test_handles_nan_input(self):
        from geodef.mesh import _nan_aware_griddata

        xin = np.array([0.0, 1.0, 0.0, 1.0, 0.5])
        yin = np.array([0.0, 0.0, 1.0, 1.0, 0.5])
        zin = np.array([0.0, 1.0, 1.0, 2.0, np.nan])
        xout = np.array([0.5])
        yout = np.array([0.5])
        zout = _nan_aware_griddata(xin, yin, zin, xout, yout)
        assert not np.any(np.isnan(zout))

    def test_no_nan_output(self):
        from geodef.mesh import _nan_aware_griddata

        # Points outside convex hull should use nearest neighbor
        xin = np.array([0.0, 1.0, 0.0, 1.0])
        yin = np.array([0.0, 0.0, 1.0, 1.0])
        zin = np.array([0.0, 1.0, 1.0, 2.0])
        xout = np.array([2.0])  # outside hull
        yout = np.array([0.5])
        zout = _nan_aware_griddata(xin, yin, zin, xout, yout)
        assert not np.any(np.isnan(zout))
