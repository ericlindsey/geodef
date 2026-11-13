"""Tests for the geodef.Fault class.

Covers: construction, factory classmethods, properties, forward modeling,
moment/magnitude, laplacian, file I/O, and vertex computation.
"""

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from geodef.fault import (
    Fault,
    _seg_to_patches,
    magnitude_to_moment,
    moment_to_magnitude,
)
from geodef.medium import ElasticMedium
from geodef.transforms import LocalFrame

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def simple_fault():
    """A 10x5 planar fault for general testing."""
    return Fault.planar(
        lat=0.0,
        lon=100.0,
        depth=15e3,
        strike=320.0,
        dip=15.0,
        length=100e3,
        width=50e3,
        n_length=10,
        n_width=5,
    )


@pytest.fixture
def single_patch():
    """A single-patch fault for simple forward model tests."""
    return Fault.planar(
        lat=0.0,
        lon=100.0,
        depth=10e3,
        strike=0.0,
        dip=90.0,
        length=10e3,
        width=10e3,
        n_length=1,
        n_width=1,
    )


# ======================================================================
# 1. Construction and validation
# ======================================================================


class TestConstruction:
    """Test Fault.__init__ validation."""

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            Fault(
                lat=np.array([0.0, 1.0]),
                lon=np.array([100.0]),  # wrong length
                depth=np.array([10e3, 10e3]),
                strike=np.array([0.0, 0.0]),
                dip=np.array([90.0, 90.0]),
                length=np.array([10e3, 10e3]),
                width=np.array([10e3, 10e3]),
            )

    def test_requires_length_width_or_vertices(self):
        with pytest.raises(ValueError, match="length and width"):
            Fault(
                lat=np.array([0.0]),
                lon=np.array([100.0]),
                depth=np.array([10e3]),
                strike=np.array([0.0]),
                dip=np.array([90.0]),
            )

    def test_grid_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="grid_shape"):
            Fault(
                lat=np.array([0.0, 1.0]),
                lon=np.array([100.0, 100.0]),
                depth=np.array([10e3, 10e3]),
                strike=np.array([0.0, 0.0]),
                dip=np.array([90.0, 90.0]),
                length=np.array([10e3, 10e3]),
                width=np.array([10e3, 10e3]),
                grid_shape=(3, 3),  # 9 != 2
            )

    def test_arrays_are_read_only(self, simple_fault):
        with pytest.raises(ValueError):
            simple_fault.lat[0] = 999.0

    def test_direct_fault_has_explicit_frame(self, single_patch):
        assert single_patch.frame.lat == pytest.approx(0.0)
        assert single_patch.frame.lon == pytest.approx(100.0)


# ======================================================================
# 2. Fault.planar() factory
# ======================================================================


class TestPlanarFrame:
    """Fault.planar owns its optional local frame without another wrapper."""

    def test_scalar_call_uses_centered_frame_by_default(self):
        fault = Fault.planar(
            lat=-2.0,
            lon=100.0,
            depth=15_000.0,
            strike=315.0,
            dip=25.0,
            length=80_000.0,
            width=40_000.0,
        )

        assert fault.frame.lat == -2.0
        assert fault.frame.lon == 100.0
        npt.assert_allclose(np.mean(fault.centers_local, axis=0)[:2], 0.0, atol=1.0)

    def test_explicit_frame_survives_medium_copy(self):
        frame = LocalFrame(lon=100.0, lat=0.0)
        fault = Fault.planar(
            lat=0.01,
            lon=100.02,
            depth=10_000.0,
            strike=0.0,
            dip=30.0,
            length=20_000.0,
            width=10_000.0,
            frame=frame,
        )

        changed = fault.with_medium(ElasticMedium(shear_modulus=40e9))

        assert changed.frame is frame
        npt.assert_allclose(changed.centers_geo, fault.centers_geo)

    def test_to_frame_preserves_geographic_geometry(self):
        fault = Fault.planar(
            lat=0.0,
            lon=100.0,
            depth=10_000.0,
            strike=0.0,
            dip=30.0,
            length=20_000.0,
            width=10_000.0,
            n_length=2,
            n_width=2,
        )
        target = LocalFrame(lon=100.2, lat=0.1)

        transformed = fault.to_frame(target)

        assert transformed.frame is target
        npt.assert_allclose(transformed.centers_geo, fault.centers_geo)


class TestPlanar:
    """Test Fault.planar() factory classmethod."""

    def test_n_patches(self, simple_fault):
        assert simple_fault.n_patches == 50

    def test_grid_shape(self, simple_fault):
        assert simple_fault.grid_shape == (10, 5)

    def test_engine(self, simple_fault):
        assert simple_fault.engine == "okada"

    def test_single_patch(self, single_patch):
        assert single_patch.n_patches == 1
        assert single_patch.grid_shape == (1, 1)

    def test_centroid_near_input(self, simple_fault):
        """Mean of patch centers should be close to the input center."""
        mean_lat = np.mean(simple_fault.lat)
        mean_lon = np.mean(simple_fault.lon)
        mean_depth = np.mean(simple_fault.depth)
        assert abs(mean_lat - 0.0) < 0.5
        assert abs(mean_lon - 100.0) < 0.5
        assert abs(mean_depth - 15e3) < 1e3

    def test_uniform_patch_sizes(self, simple_fault):
        """All patches should have the same L and W."""
        np.testing.assert_allclose(simple_fault.length, 10e3)
        np.testing.assert_allclose(simple_fault.width, 10e3)

    def test_uniform_strike_dip(self, simple_fault):
        np.testing.assert_allclose(simple_fault.strike, 320.0)
        np.testing.assert_allclose(simple_fault.dip, 15.0)

    def test_depth_varies_with_dip(self):
        """Patches should span a depth range consistent with dip."""
        fault = Fault.planar(
            lat=0.0,
            lon=100.0,
            depth=20e3,
            strike=0.0,
            dip=45.0,
            length=50e3,
            width=50e3,
            n_length=5,
            n_width=5,
        )
        # Patches span from shallowest to deepest across the dip direction
        patch_W = 50e3 / 5
        # Range should be (nW - 1) * patchW * sin(dip)
        actual_range = fault.depth.max() - fault.depth.min()
        expected = 4 * patch_W * np.sin(np.radians(45.0))
        np.testing.assert_allclose(actual_range, expected, rtol=0.01)

    def test_repr(self, simple_fault):
        r = repr(simple_fault)
        assert "Fault" in r
        assert "50" in r
        assert "okada" in r


# ======================================================================
# 3. Properties
# ======================================================================


class TestProperties:
    """Test computed properties."""

    def test_centers_shape(self, simple_fault):
        assert simple_fault.centers_geo.shape == (50, 3)

    def test_centers_local_shape(self, simple_fault):
        assert simple_fault.centers_local.shape == (50, 3)

    def test_areas_shape(self, simple_fault):
        assert simple_fault.areas.shape == (50,)

    def test_areas_values(self, simple_fault):
        np.testing.assert_allclose(simple_fault.areas, 10e3 * 10e3)

    def test_areas_total(self, simple_fault):
        np.testing.assert_allclose(np.sum(simple_fault.areas), 100e3 * 50e3)


# ======================================================================
# 4. Forward modeling
# ======================================================================


class TestForwardModeling:
    """Test greens_matrix() and displacement()."""

    def test_greens_matrix_shape(self, simple_fault):
        obs_lat = np.array([0.5, -0.5])
        obs_lon = np.array([100.5, 99.5])
        G = simple_fault.greens_matrix(lon=obs_lon, lat=obs_lat)
        assert G.shape == (6, 100)  # 3*2 x 2*50

    def test_strain_greens_shape(self, simple_fault):
        obs_lat = np.array([0.5])
        obs_lon = np.array([100.5])
        G = simple_fault.greens_matrix(lon=obs_lon, lat=obs_lat, kind="strain")
        assert G.shape == (4, 100)  # 4*1 x 2*50

    def test_displacement_returns_three_arrays(self, single_patch):
        obs_lat = np.array([0.1])
        obs_lon = np.array([100.0])
        ue, un, uz = single_patch.displacement(
            lon=obs_lon, lat=obs_lat, strike_slip=1.0
        )
        assert ue.shape == (1,)
        assert un.shape == (1,)
        assert uz.shape == (1,)

    def test_zero_slip_gives_zero_displacement(self, simple_fault):
        obs_lat = np.array([0.5, -0.5])
        obs_lon = np.array([100.5, 99.5])
        ue, un, uz = simple_fault.displacement(
            lon=obs_lon, lat=obs_lat, strike_slip=0.0, dip_slip=0.0
        )
        np.testing.assert_allclose(ue, 0.0, atol=1e-15)
        np.testing.assert_allclose(un, 0.0, atol=1e-15)
        np.testing.assert_allclose(uz, 0.0, atol=1e-15)

    def test_displacement_linearity(self, single_patch):
        """Doubling slip should double displacement."""
        obs_lat = np.array([0.1])
        obs_lon = np.array([100.0])
        ue1, un1, uz1 = single_patch.displacement(
            lon=obs_lon, lat=obs_lat, strike_slip=1.0
        )
        ue2, un2, uz2 = single_patch.displacement(
            lon=obs_lon, lat=obs_lat, strike_slip=2.0
        )
        np.testing.assert_allclose(ue2, 2.0 * ue1, rtol=1e-10)
        np.testing.assert_allclose(un2, 2.0 * un1, rtol=1e-10)
        np.testing.assert_allclose(uz2, 2.0 * uz1, rtol=1e-10)

    def test_displacement_per_patch_slip(self, simple_fault):
        """Per-patch slip array should work."""
        obs_lat = np.array([0.5])
        obs_lon = np.array([100.5])
        slip = np.ones(50)
        ue, un, uz = simple_fault.displacement(
            lon=obs_lon, lat=obs_lat, strike_slip=slip
        )
        assert np.any(np.abs(ue) > 0)

    def test_invalid_kind_raises(self, simple_fault):
        obs_lat = np.array([0.5])
        obs_lon = np.array([100.5])
        with pytest.raises(ValueError, match="Unknown kind"):
            simple_fault.greens_matrix(lon=obs_lon, lat=obs_lat, kind="bogus")


# ======================================================================
# 5. Moment and magnitude
# ======================================================================


class TestMomentMagnitude:
    """Test moment/magnitude calculations."""

    def test_moment_single_patch(self, single_patch):
        """M0 = mu * area * slip."""
        slip = np.array([1.0])
        m0 = single_patch.moment(slip)
        expected = 30e9 * 10e3 * 10e3 * 1.0
        np.testing.assert_allclose(m0, expected)

    def test_moment_custom_mu(self, single_patch):
        slip = np.array([1.0])
        m0 = single_patch.moment(slip, mu=40e9)
        expected = 40e9 * 10e3 * 10e3 * 1.0
        np.testing.assert_allclose(m0, expected)

    def test_magnitude_known_value(self):
        """Mw 7.0 corresponds to M0 ≈ 3.53e19 N-m."""
        m0 = magnitude_to_moment(7.0)
        mw = moment_to_magnitude(m0)
        np.testing.assert_allclose(mw, 7.0, atol=0.01)

    def test_moment_magnitude_roundtrip(self):
        for mw in [5.0, 6.0, 7.0, 8.0, 9.0]:
            m0 = magnitude_to_moment(mw)
            mw_back = moment_to_magnitude(m0)
            np.testing.assert_allclose(mw_back, mw, atol=1e-10)

    def test_fault_magnitude_method(self, single_patch):
        slip = np.array([1.0])
        mw = single_patch.magnitude(slip)
        m0 = single_patch.moment(slip)
        np.testing.assert_allclose(mw, moment_to_magnitude(m0))


# ======================================================================
# 6. Laplacian
# ======================================================================


class TestLaplacian:
    """Test laplacian property."""

    def test_laplacian_shape(self, simple_fault):
        L = simple_fault.laplacian
        assert L.shape == (50, 50)

    def test_laplacian_rows_sum_to_zero(self, simple_fault):
        L = simple_fault.laplacian
        np.testing.assert_allclose(L.sum(axis=1), 0.0, atol=1e-12)

    def test_laplacian_cached(self, simple_fault):
        L1 = simple_fault.laplacian
        L2 = simple_fault.laplacian
        assert L1 is L2

    def test_laplacian_unstructured_uses_knn(self):
        """Unstructured fault (no grid_shape) should use KNN Laplacian."""
        n = 20
        lats = np.linspace(0, 1, n)
        lons = np.full(n, 100.0)
        fault = Fault(
            lat=lats,
            lon=lons,
            depth=np.linspace(5e3, 30e3, n),
            strike=np.zeros(n),
            dip=np.full(n, 45.0),
            length=np.full(n, 10e3),
            width=np.full(n, 10e3),
            grid_shape=None,
        )
        L = fault.laplacian
        assert L.shape == (n, n)
        np.testing.assert_allclose(L.sum(axis=1), 0.0, atol=1e-10)

    def test_laplacian_unstructured_too_few_patches_raises(self):
        """Unstructured fault with fewer patches than k should raise."""
        fault = Fault(
            lat=np.array([0.0, 1.0]),
            lon=np.array([100.0, 100.0]),
            depth=np.array([10e3, 10e3]),
            strike=np.array([0.0, 0.0]),
            dip=np.array([90.0, 90.0]),
            length=np.array([10e3, 10e3]),
            width=np.array([10e3, 10e3]),
            grid_shape=None,
        )
        with pytest.raises(ValueError, match="k"):
            _ = fault.laplacian


# ======================================================================
# 7. Patch index
# ======================================================================


class TestPatchIndex:
    """Test patch_index for structured grids."""

    def test_patch_index_corners(self, simple_fault):
        assert simple_fault.patch_index(0, 0) == 0
        assert simple_fault.patch_index(9, 0) == 9
        assert simple_fault.patch_index(0, 4) == 40
        assert simple_fault.patch_index(9, 4) == 49

    def test_patch_index_no_grid_raises(self):
        fault = Fault(
            lat=np.array([0.0]),
            lon=np.array([100.0]),
            depth=np.array([10e3]),
            strike=np.array([0.0]),
            dip=np.array([90.0]),
            length=np.array([10e3]),
            width=np.array([10e3]),
            grid_shape=None,
        )
        with pytest.raises(ValueError, match="structured grid"):
            fault.patch_index(0, 0)

    def test_reshape_and_flatten_patches_roundtrip(self, simple_fault):
        values = np.arange(simple_fault.n_patches)

        grid = simple_fault.reshape_patches(values)

        assert grid.shape == (5, 10)
        np.testing.assert_array_equal(grid[0], np.arange(10))
        np.testing.assert_array_equal(simple_fault.flatten_patches(grid), values)

    def test_reshape_and_flatten_preserve_trailing_dimensions(self, simple_fault):
        values = np.arange(simple_fault.n_patches * 2).reshape(-1, 2)

        grid = simple_fault.reshape_patches(values)

        assert grid.shape == (5, 10, 2)
        np.testing.assert_array_equal(simple_fault.flatten_patches(grid), values)

    def test_reshape_patches_rejects_unstructured_fault(self):
        fault = Fault(
            lat=np.array([0.0]),
            lon=np.array([100.0]),
            depth=np.array([10e3]),
            strike=np.array([0.0]),
            dip=np.array([90.0]),
            length=np.array([10e3]),
            width=np.array([10e3]),
            grid_shape=None,
        )
        with pytest.raises(ValueError, match="structured grid"):
            fault.reshape_patches([1.0])


# ======================================================================
# 8. File I/O
# ======================================================================


class TestFileIO:
    """Test save/load round-trip."""

    def test_save_load_roundtrip(self, simple_fault):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            fname = f.name

        simple_fault.save(fname)
        loaded = Fault.load(fname, format="center")

        np.testing.assert_allclose(loaded.lat, simple_fault.lat, atol=1e-4)
        np.testing.assert_allclose(loaded.lon, simple_fault.lon, atol=1e-4)
        np.testing.assert_allclose(loaded.depth, simple_fault.depth, atol=1e-1)
        np.testing.assert_allclose(loaded.strike, simple_fault.strike, atol=1e-4)
        np.testing.assert_allclose(loaded.dip, simple_fault.dip, atol=1e-4)
        np.testing.assert_allclose(loaded.length, simple_fault.length, atol=1e-1)
        np.testing.assert_allclose(loaded.width, simple_fault.width, atol=1e-1)
        assert loaded.n_patches == simple_fault.n_patches

        Path(fname).unlink()

    def test_save_unknown_format_raises(self, simple_fault):
        with pytest.raises(ValueError, match="Unknown format"):
            simple_fault.save("/tmp/test.txt", format="bogus")


# ======================================================================
# 9. Vertex computation
# ======================================================================


class TestCorners:
    """Test the patch-corner accessors ``corners`` and ``corners_geo``."""

    def test_shapes(self, simple_fault):
        assert simple_fault.corners.shape == (50, 4, 3)
        assert simple_fault.corners_geo.shape == (50, 4, 3)

    def test_corners_average_to_the_patch_center(self, simple_fault):
        np.testing.assert_allclose(
            simple_fault.corners.mean(axis=1), simple_fault.centers_local, atol=1e-6
        )

    def test_first_two_corners_are_the_shallow_edge(self):
        fault = Fault.planar(
            lon=100.0, lat=0.0, depth=20e3, strike=30.0, dip=40.0,
            length=20e3, width=10e3,
        )  # fmt: skip
        depth = fault.corners_geo[0, :, 2]
        half_height = 5e3 * np.sin(np.radians(40.0))
        np.testing.assert_allclose(depth[:2], 20e3 - half_height)
        np.testing.assert_allclose(depth[2:], 20e3 + half_height)

    def test_fault_dips_to_the_right_of_strike(self):
        """Strike north, so the deep edge lies to the east of the shallow edge."""
        fault = Fault.planar(
            lon=100.0, lat=0.0, depth=20e3, strike=0.0, dip=30.0,
            length=20e3, width=10e3,
        )  # fmt: skip
        east = fault.corners[0, :, 0]
        assert np.all(east[2:] > east[:2].max())

    def test_corners_geo_brackets_the_center(self, single_patch):
        corners = single_patch.corners_geo[0]
        assert corners[:, 0].min() <= single_patch.lon[0] <= corners[:, 0].max()
        assert corners[:, 1].min() <= single_patch.lat[0] <= corners[:, 1].max()

    def test_neighboring_rows_share_an_edge(self):
        fault = Fault.planar(
            lon=100.0, lat=0.0, depth=20e3, strike=20.0, dip=35.0,
            length=20e3, width=20e3, n_length=1, n_width=2,
        )  # fmt: skip
        np.testing.assert_allclose(
            fault.corners[0, [3, 2]], fault.corners[1, [0, 1]], atol=0.05
        )

    def test_triangular_corners_are_the_vertices(self):
        vertices = np.array([[[0.0, 0.0, -1e3], [1e3, 0.0, -1e3], [0.0, 1e3, -2e3]]])
        fault = Fault.from_triangles(vertices)
        np.testing.assert_array_equal(fault.corners, vertices)
        np.testing.assert_allclose(
            fault.corners_geo[0, :, 2], [1e3, 1e3, 2e3], atol=1.0
        )


class TestPlanarGridGeometry:
    """A subdivided planar fault must be the same plane as its single patch."""

    KWARGS = dict(
        lon=100.0, lat=0.0, depth=20e3, strike=25.0, dip=30.0,
        length=40e3, width=40e3,
    )  # fmt: skip

    def test_plane_reaching_above_the_surface_is_rejected(self):
        with pytest.raises(ValueError, match="above the surface"):
            Fault.planar(
                lon=100.0, lat=0.0, depth=5e3, strike=0.0, dip=50.0,
                length=40e3, width=25e3, n_width=4,
            )  # fmt: skip

    def test_first_row_is_the_shallowest(self):
        fault = Fault.planar(n_length=2, n_width=3, **self.KWARGS)
        depth = fault.reshape_patches(fault.depth)
        assert np.all(np.diff(depth, axis=0) > 0)
        patch_height = 40e3 / 3 * np.sin(np.radians(30.0))
        np.testing.assert_allclose(
            depth[:, 0], 20e3 + patch_height * np.array([-1, 0, 1])
        )

    def test_deeper_rows_lie_to_the_right_of_strike(self):
        fault = Fault.planar(n_length=1, n_width=2, **{**self.KWARGS, "strike": 0.0})
        east = fault.centers_local[:, 0]
        assert east[1] > east[0]
        assert fault.depth[1] > fault.depth[0]

    @pytest.mark.parametrize("n_length, n_width", [(1, 2), (3, 1), (2, 4)])
    def test_uniform_slip_is_independent_of_the_subdivision(self, n_length, n_width):
        lon = np.linspace(99.6, 100.4, 9)
        lat = np.linspace(-0.3, 0.3, 9)
        whole = Fault.planar(**self.KWARGS)
        pieces = Fault.planar(n_length=n_length, n_width=n_width, **self.KWARGS)
        for slip in ({"strike_slip": 1.0}, {"dip_slip": 1.0}):
            expected = whole.displacement(lon=lon, lat=lat, **slip)
            actual = pieces.displacement(lon=lon, lat=lat, **slip)
            np.testing.assert_allclose(actual, expected, atol=1e-11)


class TestSharedFrame:
    """Rectangles and triangles are flattened by the same local frame."""

    KWARGS = dict(
        lon=100.0, lat=40.0, depth=60e3, strike=25.0, dip=30.0,
        length=300e3, width=150e3,
    )  # fmt: skip
    LON = np.linspace(98.0, 102.0, 7)
    LAT = np.linspace(38.5, 41.5, 7)

    def test_subdivision_is_exact_for_a_large_high_latitude_fault(self):
        whole = Fault.planar(**self.KWARGS)
        pieces = Fault.planar(n_length=5, n_width=4, **self.KWARGS)
        expected = whole.displacement(lon=self.LON, lat=self.LAT, dip_slip=1.0)
        actual = pieces.displacement(lon=self.LON, lat=self.LAT, dip_slip=1.0)
        np.testing.assert_allclose(actual, expected, atol=1e-11)

    def test_a_rectangle_equals_the_two_triangles_that_tile_it(self):
        rectangle = Fault.planar(**{**self.KWARGS, "length": 80e3, "width": 40e3})
        corners = rectangle.corners[0]
        # counterclockwise seen from above, the winding the triangle kernel
        # needs for thrust-positive dip slip
        triangles = Fault.from_triangles(
            np.array([corners[[0, 2, 1]], corners[[0, 3, 2]]]), frame=rectangle.frame
        )
        for slip in ({"strike_slip": 1.0}, {"dip_slip": 1.0}):
            expected = rectangle.displacement(lon=self.LON, lat=self.LAT, **slip)
            actual = triangles.displacement(lon=self.LON, lat=self.LAT, **slip)
            np.testing.assert_allclose(actual, expected, atol=1e-9)

    @pytest.mark.parametrize("winding", ["counterclockwise", "clockwise"])
    def test_triangles_match_the_rectangle_whatever_their_winding(self, winding):
        """Thrust is positive dip slip and strike follows the right-hand rule
        for triangles, as for rectangles, whichever way the corners are
        listed."""
        rectangle = Fault.planar(**{**self.KWARGS, "length": 80e3, "width": 40e3})
        corners = rectangle.corners[0]
        order = [[0, 2, 1], [0, 3, 2]] if winding == "counterclockwise" else [
            [0, 1, 2], [0, 2, 3]]  # fmt: skip
        triangles = Fault.from_triangles(
            np.array([corners[i] for i in order]), frame=rectangle.frame
        )
        np.testing.assert_allclose(triangles.strike, 25.0, atol=1e-9)
        np.testing.assert_allclose(triangles.dip, 30.0, atol=1e-9)
        for slip in ({"strike_slip": 1.0}, {"dip_slip": 1.0}):
            expected = rectangle.displacement(lon=self.LON, lat=self.LAT, **slip)
            actual = triangles.displacement(lon=self.LON, lat=self.LAT, **slip)
            np.testing.assert_allclose(actual, expected, atol=1e-9)

    @pytest.mark.skipif(
        importlib.util.find_spec("meshpy") is None,
        reason="requires meshpy (install with the 'mesh' extra)",
    )
    def test_generated_meshes_are_thrust_positive(self):
        """A mesh dipping east under a north trace is a north-striking fault."""
        from geodef import mesh

        dipping_east = mesh.from_trace(
            np.array([100.0, 100.0]), np.array([39.7, 40.3]), 20e3, 30.0,
            dip_direction=90.0, n_downdip=3, target_length=10e3,
        )  # fmt: skip
        fault = Fault.from_mesh(dipping_east)
        np.testing.assert_allclose(fault.strike, 0.0, atol=0.5)
        # thrust on an east-dipping fault lifts the hanging wall, to the east
        _, _, up_east = fault.displacement(lon=100.12, lat=40.0, dip_slip=1.0)
        _, _, up_west = fault.displacement(lon=99.85, lat=40.0, dip_slip=1.0)
        assert up_east > 0.0 > up_west

    def test_planar_from_corner_is_the_same_plane(self):
        center = Fault.planar(n_length=3, n_width=2, **self.KWARGS)
        top_left = center.corners_geo[0, 0]
        corner = Fault.planar_from_corner(
            lon=top_left[0], lat=top_left[1], depth=top_left[2],
            strike=25.0, dip=30.0, length=300e3, width=150e3,
            n_length=3, n_width=2, frame=center.frame,
        )  # fmt: skip
        np.testing.assert_allclose(
            corner.centers_local, center.centers_local, atol=1e-5
        )

    def test_to_frame_keeps_positions_and_converts_strike(self):
        fault = Fault.planar(n_length=4, n_width=2, **self.KWARGS)
        other = LocalFrame(lon=101.5, lat=41.0)
        moved = fault.to_frame(other)
        np.testing.assert_array_equal(moved.lon, fault.lon)
        np.testing.assert_array_equal(moved.depth, fault.depth)
        assert np.all(np.abs(moved.strike - fault.strike) > 0.5)
        np.testing.assert_allclose(
            moved.to_frame(fault.frame).strike, fault.strike, atol=1e-9
        )

    def test_center_file_round_trip_preserves_the_model(self, tmp_path):
        fault = Fault.planar(n_length=4, n_width=3, **self.KWARGS)
        path = tmp_path / "fault.txt"
        fault.save(str(path), format="center")
        loaded = Fault.load(str(path), format="center")
        expected = fault.displacement(lon=self.LON, lat=self.LAT, dip_slip=1.0)
        actual = loaded.displacement(lon=self.LON, lat=self.LAT, dip_slip=1.0)
        np.testing.assert_allclose(actual, expected, atol=2e-4)

    def test_geographic_strike_follows_the_meridians(self):
        fault = Fault.planar(n_length=5, n_width=1, **{**self.KWARGS, "strike": 90.0})
        # an east-striking fault: true north rotates against the frame's North
        # axis by about dlon * sin(lat) at each patch
        dlon = fault.lon - 100.0
        np.testing.assert_allclose(
            fault.geographic_strike, 90.0 + dlon * np.sin(np.radians(40.0)), atol=0.02
        )
        np.testing.assert_allclose(fault.strike, 90.0)

    def test_file_strikes_are_true_north_azimuths(self, tmp_path):
        """A file of due-north strikes follows the meridians, which converge."""
        lon = np.array([98.0, 100.0, 102.0])
        rows = np.column_stack(
            [np.arange(3), np.zeros(3), np.arange(3), lon, np.full(3, 40.0),
             np.full(3, 20e3), np.full(3, 10e3), np.full(3, 10e3), np.zeros(3),
             np.full(3, 60.0)]
        )  # fmt: skip
        path = tmp_path / "meridians.txt"
        np.savetxt(path, rows)
        loaded = Fault.load(str(path), format="center")
        expected = 2.0 * np.sin(np.radians(40.0))
        np.testing.assert_allclose(loaded.strike, [expected, 0.0, -expected], atol=2e-3)


# ======================================================================
# 10. Planar from corner
# ======================================================================


class TestPlanarFromCorner:
    """Test Fault.planar_from_corner() factory."""

    def test_same_n_patches(self):
        f = Fault.planar_from_corner(
            lat=0.0,
            lon=100.0,
            depth=0.0,
            strike=0.0,
            dip=45.0,
            length=50e3,
            width=30e3,
            n_length=5,
            n_width=3,
        )
        assert f.n_patches == 15

    def test_shallowest_patch_near_corner_depth(self):
        """Shallowest patches should be near the corner depth."""
        f = Fault.planar_from_corner(
            lat=0.0,
            lon=100.0,
            depth=1000.0,
            strike=0.0,
            dip=45.0,
            length=50e3,
            width=30e3,
            n_length=5,
            n_width=3,
        )
        # Shallowest patches (first row, j=0)
        min_depth = np.min(f.depth)
        # Should be close to corner_depth + half_patchW * sin(dip)
        patch_W = 30e3 / 3
        expected_min = 1000.0 + 0.5 * patch_W * np.sin(np.radians(45.0))
        np.testing.assert_allclose(min_depth, expected_min, rtol=0.01)


# ======================================================================
# 11. Stress kernel
# ======================================================================


class TestStressKernel:
    """Test stress_kernel method."""

    def test_stress_kernel_shape(self):
        fault = Fault.planar(
            lat=0.0,
            lon=100.0,
            depth=15e3,
            strike=0.0,
            dip=45.0,
            length=30e3,
            width=20e3,
            n_length=3,
            n_width=2,
        )
        K = fault.stress_kernel()
        assert K.shape == (24, 12)  # 4*6 x 2*6


# ======================================================================
# 12. Cross-validation with old FaultModel
# ======================================================================


class TestCrossValidation:
    """Verify that Fault.planar() produces the same geometry as the old code."""

    def test_planar_matches_old_centered_geometry(self):
        """Compare patch centers with manually-computed expected values.

        Uses the same trigonometric formulas as the old
        create_planar_model_centered to verify equivalence.
        """
        lat0, lon0, depth0 = 0.0, 100.0, 20e3
        strike, dip = 45.0, 30.0
        length, width = 60e3, 40e3
        nL, nW = 3, 2

        fault = Fault.planar(
            lat=lat0,
            lon=lon0,
            depth=depth0,
            strike=strike,
            dip=dip,
            length=length,
            width=width,
            n_length=nL,
            n_width=nW,
        )

        patchW = width / nW
        sin_dip = np.sin(np.radians(dip))
        fault_u0 = -0.5 * width * sin_dip

        expected_depths = []
        for j in range(nW):
            for i in range(nL):
                u_offset = fault_u0 + (j + 0.5) * patchW * sin_dip
                # The legacy code passes depth as translate_flat's altitude
                # argument, so the offset is added: row 0 is the shallowest.
                expected_depths.append(depth0 + u_offset)

        np.testing.assert_allclose(
            fault.depth,
            expected_depths,
            rtol=1e-10,
        )


# ======================================================================
# 13. Seg format (_seg_to_patches algorithm)
# ======================================================================


class TestSegToPatches:
    """Test the flt2flt port: _seg_to_patches()."""

    def test_uniform_patches(self):
        """With qL=qW=1, patches should be uniform."""
        origin = np.array([0.0, 0.0, 0.0])
        patches = _seg_to_patches(origin, 60e3, 30e3, 0.0, 30.0, 20e3, 10e3, 1.0, 1.0)
        assert patches.shape[1] == 7
        # 60/20 = 3 along strike, 30/10 = 3 down dip
        assert len(patches) == 9
        np.testing.assert_allclose(patches[:, 3], 20e3)
        np.testing.assert_allclose(patches[:, 4], 10e3)

    def test_total_width_preserved(self):
        """Total width must equal the segment width."""
        origin = np.array([0.0, 0.0, 0.0])
        patches = _seg_to_patches(origin, 60e3, 100e3, 0.0, 30.0, 20e3, 5e3, 1.0, 1.5)
        unique_widths = np.unique(patches[:, 4])
        np.testing.assert_allclose(np.sum(unique_widths), 100e3)

    def test_total_length_preserved(self):
        """Each row should span the full strike length."""
        origin = np.array([0.0, 0.0, 0.0])
        patches = _seg_to_patches(origin, 60e3, 30e3, 0.0, 30.0, 10e3, 5e3, 1.5, 1.5)
        # Group by width (each row has a distinct width)
        for w in np.unique(patches[:, 4]):
            row = patches[patches[:, 4] == w]
            row_total_L = np.sum(row[:, 3])
            np.testing.assert_allclose(row_total_L, 60e3)

    def test_geometric_width_growth(self):
        """Widths should grow by factor alpha_w."""
        origin = np.array([0.0, 0.0, 0.0])
        patches = _seg_to_patches(origin, 60e3, 60e3, 0.0, 30.0, 20e3, 5e3, 1.0, 2.0)
        unique_widths = np.sort(np.unique(patches[:, 4]))
        # First few widths should double: 5000, 10000, 20000, then remainder
        assert unique_widths[0] == 5e3
        assert unique_widths[1] == 10e3
        assert unique_widths[2] == 20e3

    def test_origin_position(self):
        """First patch corner should match the origin."""
        origin = np.array([1000.0, 2000.0, 500.0])
        patches = _seg_to_patches(origin, 60e3, 30e3, 0.0, 30.0, 20e3, 10e3, 1.0, 1.0)
        np.testing.assert_allclose(patches[0, :3], origin)

    def test_strike_direction(self):
        """Patches should advance along the strike direction."""
        origin = np.array([0.0, 0.0, 0.0])
        patches = _seg_to_patches(origin, 60e3, 30e3, 90.0, 30.0, 20e3, 10e3, 1.0, 1.0)
        # Strike=90 means strike_vec = [cos(90), sin(90), 0] = [0, 1, 0]
        # So patches should advance in the East direction
        first_row = patches[:3]  # first 3 patches (first dip row)
        np.testing.assert_allclose(first_row[:, 0], 0.0, atol=1e-10)  # North stays 0
        # East should increase
        assert first_row[1, 1] > first_row[0, 1]
        assert first_row[2, 1] > first_row[1, 1]


# ======================================================================
# 14. Seg format I/O
# ======================================================================


requires_stress_shadows = pytest.mark.skipif(
    not Path("related/stress-shadows").exists(),
    reason="related/stress-shadows reference data not available",
)


class TestSegIO:
    """Test loading and saving .seg files."""

    @requires_stress_shadows
    def test_load_3d_ramp(self):
        """Load the 3D ramp test file."""
        fname = "related/stress-shadows/3d_models/test_data_3d/ramp_3d.seg"
        f = Fault.load(fname, format="seg")
        # 700km / 20km = 35, 250km / 10km = 25 => 875 patches
        assert f.n_patches == 875
        assert f.grid_shape == (35, 25)
        assert f.engine == "okada"
        np.testing.assert_allclose(f.strike, 0.0)
        np.testing.assert_allclose(f.dip, 10.0)

    @requires_stress_shadows
    def test_load_2d_ramp(self):
        """Load the 2D ramp test file."""
        fname = "related/stress-shadows/2d_models/test_data_2d/ramp_2d.seg"
        f = Fault.load(fname, format="seg")
        assert f.n_patches > 0
        np.testing.assert_allclose(f.dip, 10.0)
        # All patches should have the same width (qW=1)
        np.testing.assert_allclose(f.width, f.width[0])

    @requires_stress_shadows
    def test_load_seg_with_ref_coords(self):
        """Loading with a frame should offset the geographic coords."""
        fname = "related/stress-shadows/3d_models/test_data_3d/ramp_3d.seg"
        f0 = Fault.load(fname, format="seg", frame=LocalFrame(lon=0.0, lat=0.0))
        f1 = Fault.load(fname, format="seg", frame=LocalFrame(lon=100.0, lat=10.0))
        # Centers should differ by roughly the ref offset
        assert abs(np.mean(f1.lat) - np.mean(f0.lat)) > 5.0
        assert abs(np.mean(f1.lon) - np.mean(f0.lon)) > 50.0

    def test_seg_save_load_roundtrip(self):
        """Save a uniform fault as .seg and reload it."""
        fault = Fault.planar(
            lat=0.0,
            lon=100.0,
            depth=15e3,
            strike=30.0,
            dip=20.0,
            length=100e3,
            width=50e3,
            n_length=5,
            n_width=5,
        )
        with tempfile.NamedTemporaryFile(suffix=".seg", delete=False) as f:
            fname = f.name

        fault.save(fname, format="seg", frame=LocalFrame(lon=100.0, lat=0.0))
        loaded = Fault.load(fname, format="seg", frame=LocalFrame(lon=100.0, lat=0.0))

        assert loaded.n_patches == fault.n_patches
        np.testing.assert_allclose(loaded.strike, fault.strike, atol=1e-4)
        np.testing.assert_allclose(loaded.dip, fault.dip, atol=1e-4)
        np.testing.assert_allclose(loaded.length, fault.length, atol=1.0)
        np.testing.assert_allclose(loaded.width, fault.width, atol=1.0)
        # Depths match (flt2flt orders shallow-to-deep, planar() deep-to-shallow,
        # so compare sorted values)
        np.testing.assert_allclose(
            np.sort(loaded.depth),
            np.sort(fault.depth),
            rtol=0.01,
        )

        Path(fname).unlink()

    @requires_stress_shadows
    def test_load_seg_depth_increases_downdip(self):
        """Deeper patches should have greater depth values."""
        fname = "related/stress-shadows/3d_models/test_data_3d/ramp_3d.seg"
        f = Fault.load(fname, format="seg")
        nL = f.grid_shape[0]
        # First row (j=0) vs last row (j=nW-1)
        first_row_depth = np.mean(f.depth[:nL])
        last_row_depth = np.mean(f.depth[-nL:])
        assert first_row_depth != last_row_depth

    def test_seg_geometric_growth_load(self):
        """Create a seg file with geometric growth and verify loading."""
        with tempfile.NamedTemporaryFile(suffix=".seg", delete=False, mode="w") as f:
            fname = f.name
            f.write("# test seg file with geometric growth\n")
            f.write(
                "# n  Vpl  x1  x2  x3  Length  Width  Strike  Dip  Rake  "
                "L0  W0  qL  qW\n"
            )
            f.write(
                "1 1.0 0.0 0.0 0.0 60000.0 30000.0 0.0 30.0 90.0 "
                "10000.0 5000.0 1.0 1.5\n"
            )

        f = Fault.load(fname, format="seg")
        assert f.n_patches > 0
        # Should have non-uniform widths
        unique_widths = np.unique(f.width)
        assert len(unique_widths) > 1
        # Widths should sum to total
        np.testing.assert_allclose(np.sum(unique_widths), 30e3)
        # No grid_shape for non-uniform grids
        assert f.grid_shape is None

        Path(fname).unlink()


# ======================================================================
# 14. Triangular fault ned save
# ======================================================================


def _make_tri_fault() -> Fault:
    """Small triangular fault from a simple planar mesh."""
    from geodef.mesh import Mesh

    lon = np.array([100.0, 100.1, 100.0, 100.1])
    lat = np.array([0.0, 0.0, 0.1, 0.1])
    depth = np.array([5e3, 5e3, 15e3, 15e3])
    triangles = np.array([[0, 1, 2], [1, 3, 2]])
    mesh = Mesh(lon=lon, lat=lat, depth=depth, triangles=triangles)
    return Fault.from_mesh(mesh)


class TestFaultSaveNed:
    """Test Fault.save(format='ned') for triangular faults."""

    def test_tri_save_creates_files(self, tmp_path):
        fault = _make_tri_fault()
        fname = str(tmp_path / "tri_fault")
        fault.save(fname, format="ned")
        assert (tmp_path / "tri_fault.ned").exists()
        assert (tmp_path / "tri_fault.tri").exists()

    def test_tri_save_roundtrip(self, tmp_path):
        fault = _make_tri_fault()
        fname = str(tmp_path / "tri_fault")
        fault.save(fname, format="ned")
        loaded = Fault.load(fname, format="ned")
        assert loaded.engine == "tri"
        assert loaded.n_patches == fault.n_patches
        # Round-trip uses flat-earth approximation; ~300 m tolerance (~0.003 deg)
        np.testing.assert_allclose(loaded.lat, fault.lat, atol=3e-4)
        np.testing.assert_allclose(loaded.lon, fault.lon, atol=3e-4)
        np.testing.assert_allclose(loaded.depth, fault.depth, atol=10.0)

    def test_auto_format_tri_picks_ned(self, tmp_path):
        """format=None on a tri fault should default to ned."""
        fault = _make_tri_fault()
        fname = str(tmp_path / "auto_tri")
        fault.save(fname)
        assert (tmp_path / "auto_tri.ned").exists()

    def test_rect_fault_ned_raises(self, simple_fault, tmp_path):
        with pytest.raises(ValueError, match="ned"):
            simple_fault.save(str(tmp_path / "rect"), format="ned")

    def test_tri_fault_center_raises(self, tmp_path):
        fault = _make_tri_fault()
        with pytest.raises(ValueError, match="center"):
            fault.save(str(tmp_path / "tri"), format="center")


# ======================================================================
# 15. Fault.to_gmt()
# ======================================================================


class TestFaultToGmt:
    """Test Fault.to_gmt() polygon export."""

    def test_rect_creates_file(self, simple_fault, tmp_path):
        fpath = tmp_path / "fault.gmt"
        simple_fault.to_gmt(str(fpath), values=np.ones(simple_fault.n_patches))
        assert fpath.exists()

    def test_segment_count_equals_n_patches(self, simple_fault, tmp_path):
        fpath = tmp_path / "fault.gmt"
        simple_fault.to_gmt(str(fpath), values=np.ones(simple_fault.n_patches))
        lines = fpath.read_text().splitlines()
        segment_headers = [ln for ln in lines if ln.startswith(">")]
        assert len(segment_headers) == simple_fault.n_patches

    def test_z_values_in_header(self, tmp_path):
        fault = Fault.planar(
            lat=0.0,
            lon=100.0,
            depth=10e3,
            strike=0.0,
            dip=45.0,
            length=20e3,
            width=10e3,
            n_length=2,
            n_width=1,
        )
        values = np.array([1.5, 3.0])
        fpath = tmp_path / "fault.gmt"
        fault.to_gmt(str(fpath), values=values)
        text = fpath.read_text()
        assert "1.5" in text
        assert "3.0" in text

    def test_default_values_zeros(self, tmp_path):
        fault = Fault.planar(
            lat=0.0,
            lon=100.0,
            depth=10e3,
            strike=0.0,
            dip=45.0,
            length=10e3,
            width=10e3,
        )
        fpath = tmp_path / "fault.gmt"
        fault.to_gmt(str(fpath))  # no values
        text = fpath.read_text()
        assert "0.0" in text or "-Z0" in text

    def test_polygon_vertices_are_lon_lat(self, tmp_path):
        fault = Fault.planar(
            lat=0.0,
            lon=100.0,
            depth=10e3,
            strike=0.0,
            dip=45.0,
            length=10e3,
            width=10e3,
        )
        fpath = tmp_path / "fault.gmt"
        fault.to_gmt(str(fpath))
        data_lines = [
            ln
            for ln in fpath.read_text().splitlines()
            if ln and not ln.startswith(">") and not ln.startswith("#")
        ]
        assert len(data_lines) > 0
        for line in data_lines:
            parts = line.split()
            assert len(parts) == 2
            lon_val = float(parts[0])
            lat_val = float(parts[1])
            # Should be geographically sensible for a fault near 0N, 100E
            assert 95.0 < lon_val < 105.0
            assert -5.0 < lat_val < 5.0

    def test_tri_fault_to_gmt(self, tmp_path):
        fault = _make_tri_fault()
        fpath = tmp_path / "tri.gmt"
        fault.to_gmt(str(fpath), values=np.ones(fault.n_patches))
        lines = fpath.read_text().splitlines()
        segment_headers = [ln for ln in lines if ln.startswith(">")]
        assert len(segment_headers) == fault.n_patches

    def test_values_wrong_length_raises(self, simple_fault, tmp_path):
        with pytest.raises(ValueError, match="values"):
            simple_fault.to_gmt(
                str(tmp_path / "fault.gmt"),
                values=np.ones(simple_fault.n_patches + 1),
            )
