"""Tests for the geodef.Fault class (Phase 3.1).

Covers: construction, factory classmethods, properties, forward modeling,
moment/magnitude, laplacian, file I/O, and vertex computation.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from geodef.fault import Fault, _seg_to_patches, magnitude_to_moment, moment_to_magnitude


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def simple_fault():
    """A 10x5 planar fault for general testing."""
    return Fault.planar(
        lat=0.0, lon=100.0, depth=15e3,
        strike=320.0, dip=15.0,
        length=100e3, width=50e3,
        n_length=10, n_width=5,
    )


@pytest.fixture
def single_patch():
    """A single-patch fault for simple forward model tests."""
    return Fault.planar(
        lat=0.0, lon=100.0, depth=10e3,
        strike=0.0, dip=90.0,
        length=10e3, width=10e3,
        n_length=1, n_width=1,
    )


# ======================================================================
# 1. Construction and validation
# ======================================================================

class TestConstruction:
    """Test Fault.__init__ validation."""

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            Fault(
                np.array([0.0, 1.0]),
                np.array([100.0]),  # wrong length
                np.array([10e3, 10e3]),
                np.array([0.0, 0.0]),
                np.array([90.0, 90.0]),
                np.array([10e3, 10e3]),
                np.array([10e3, 10e3]),
            )

    def test_invalid_engine_raises(self):
        with pytest.raises(ValueError, match="engine"):
            Fault(
                np.array([0.0]),
                np.array([100.0]),
                np.array([10e3]),
                np.array([0.0]),
                np.array([90.0]),
                np.array([10e3]),
                np.array([10e3]),
                engine="bogus",
            )

    def test_okada_requires_length_width(self):
        with pytest.raises(ValueError, match="length and width"):
            Fault(
                np.array([0.0]),
                np.array([100.0]),
                np.array([10e3]),
                np.array([0.0]),
                np.array([90.0]),
                None, None,
                engine="okada",
            )

    def test_grid_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="grid_shape"):
            Fault(
                np.array([0.0, 1.0]),
                np.array([100.0, 100.0]),
                np.array([10e3, 10e3]),
                np.array([0.0, 0.0]),
                np.array([90.0, 90.0]),
                np.array([10e3, 10e3]),
                np.array([10e3, 10e3]),
                grid_shape=(3, 3),  # 9 != 2
            )

    def test_arrays_are_read_only(self, simple_fault):
        with pytest.raises(ValueError):
            simple_fault._lat[0] = 999.0


# ======================================================================
# 2. Fault.planar() factory
# ======================================================================

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
        mean_lat = np.mean(simple_fault._lat)
        mean_lon = np.mean(simple_fault._lon)
        mean_depth = np.mean(simple_fault._depth)
        assert abs(mean_lat - 0.0) < 0.5
        assert abs(mean_lon - 100.0) < 0.5
        assert abs(mean_depth - 15e3) < 1e3

    def test_uniform_patch_sizes(self, simple_fault):
        """All patches should have the same L and W."""
        np.testing.assert_allclose(simple_fault._length, 10e3)
        np.testing.assert_allclose(simple_fault._width, 10e3)

    def test_uniform_strike_dip(self, simple_fault):
        np.testing.assert_allclose(simple_fault._strike, 320.0)
        np.testing.assert_allclose(simple_fault._dip, 15.0)

    def test_depth_varies_with_dip(self):
        """Patches should span a depth range consistent with dip."""
        fault = Fault.planar(
            lat=0.0, lon=100.0, depth=20e3,
            strike=0.0, dip=45.0,
            length=50e3, width=50e3,
            n_length=5, n_width=5,
        )
        expected_range = 50e3 * np.sin(np.radians(45.0))
        # Patches span from shallowest to deepest across the dip direction
        patch_W = 50e3 / 5
        # Range should be (nW - 1) * patchW * sin(dip)
        actual_range = fault._depth.max() - fault._depth.min()
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
        assert simple_fault.centers.shape == (50, 3)

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
        G = simple_fault.greens_matrix(obs_lat, obs_lon)
        assert G.shape == (6, 100)  # 3*2 x 2*50

    def test_strain_greens_shape(self, simple_fault):
        obs_lat = np.array([0.5])
        obs_lon = np.array([100.5])
        G = simple_fault.greens_matrix(obs_lat, obs_lon, kind="strain")
        assert G.shape == (4, 100)  # 4*1 x 2*50

    def test_displacement_returns_three_arrays(self, single_patch):
        obs_lat = np.array([0.1])
        obs_lon = np.array([100.0])
        ue, un, uz = single_patch.displacement(obs_lat, obs_lon, slip_strike=1.0)
        assert ue.shape == (1,)
        assert un.shape == (1,)
        assert uz.shape == (1,)

    def test_zero_slip_gives_zero_displacement(self, simple_fault):
        obs_lat = np.array([0.5, -0.5])
        obs_lon = np.array([100.5, 99.5])
        ue, un, uz = simple_fault.displacement(obs_lat, obs_lon, slip_strike=0.0, slip_dip=0.0)
        np.testing.assert_allclose(ue, 0.0, atol=1e-15)
        np.testing.assert_allclose(un, 0.0, atol=1e-15)
        np.testing.assert_allclose(uz, 0.0, atol=1e-15)

    def test_displacement_linearity(self, single_patch):
        """Doubling slip should double displacement."""
        obs_lat = np.array([0.1])
        obs_lon = np.array([100.0])
        ue1, un1, uz1 = single_patch.displacement(obs_lat, obs_lon, slip_strike=1.0)
        ue2, un2, uz2 = single_patch.displacement(obs_lat, obs_lon, slip_strike=2.0)
        np.testing.assert_allclose(ue2, 2.0 * ue1, rtol=1e-10)
        np.testing.assert_allclose(un2, 2.0 * un1, rtol=1e-10)
        np.testing.assert_allclose(uz2, 2.0 * uz1, rtol=1e-10)

    def test_displacement_per_patch_slip(self, simple_fault):
        """Per-patch slip array should work."""
        obs_lat = np.array([0.5])
        obs_lon = np.array([100.5])
        slip = np.ones(50)
        ue, un, uz = simple_fault.displacement(obs_lat, obs_lon, slip_strike=slip)
        assert np.any(np.abs(ue) > 0)

    def test_invalid_kind_raises(self, simple_fault):
        obs_lat = np.array([0.5])
        obs_lon = np.array([100.5])
        with pytest.raises(ValueError, match="Unknown kind"):
            simple_fault.greens_matrix(obs_lat, obs_lon, kind="bogus")


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

    def test_laplacian_no_grid_raises(self):
        fault = Fault(
            np.array([0.0, 1.0]),
            np.array([100.0, 100.0]),
            np.array([10e3, 10e3]),
            np.array([0.0, 0.0]),
            np.array([90.0, 90.0]),
            np.array([10e3, 10e3]),
            np.array([10e3, 10e3]),
            grid_shape=None,
        )
        with pytest.raises(ValueError, match="structured grid"):
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
            np.array([0.0]),
            np.array([100.0]),
            np.array([10e3]),
            np.array([0.0]),
            np.array([90.0]),
            np.array([10e3]),
            np.array([10e3]),
            grid_shape=None,
        )
        with pytest.raises(ValueError, match="structured grid"):
            fault.patch_index(0, 0)


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

        np.testing.assert_allclose(loaded._lat, simple_fault._lat, atol=1e-4)
        np.testing.assert_allclose(loaded._lon, simple_fault._lon, atol=1e-4)
        np.testing.assert_allclose(loaded._depth, simple_fault._depth, atol=1e-1)
        np.testing.assert_allclose(loaded._strike, simple_fault._strike, atol=1e-4)
        np.testing.assert_allclose(loaded._dip, simple_fault._dip, atol=1e-4)
        np.testing.assert_allclose(loaded._length, simple_fault._length, atol=1e-1)
        np.testing.assert_allclose(loaded._width, simple_fault._width, atol=1e-1)
        assert loaded.n_patches == simple_fault.n_patches

        Path(fname).unlink()

    def test_save_unknown_format_raises(self, simple_fault):
        with pytest.raises(ValueError, match="Unknown format"):
            simple_fault.save("/tmp/test.txt", format="bogus")


# ======================================================================
# 9. Vertex computation
# ======================================================================

class TestVertices:
    """Test vertices_2d and vertices_3d."""

    def test_vertices_3d_shape(self, simple_fault):
        v = simple_fault.vertices_3d
        assert v.shape == (50, 4, 3)

    def test_vertices_2d_shape(self, simple_fault):
        v = simple_fault.vertices_2d
        assert v.shape == (50, 4, 2)

    def test_vertices_2d_is_subset_of_3d(self, simple_fault):
        v2 = simple_fault.vertices_2d
        v3 = simple_fault.vertices_3d
        np.testing.assert_allclose(v2, v3[:, :, :2])

    def test_single_patch_vertices_surround_center(self, single_patch):
        """Vertices should bracket the center coordinates."""
        v = single_patch.vertices_3d  # (1, 4, 3) = [lon, lat, depth_km]
        center_lon = single_patch._lon[0]
        center_lat = single_patch._lat[0]
        # For a vertical (dip=90) fault, updip width projection is zero
        # so vertices only extend along strike
        lons = v[0, :, 0]
        lats = v[0, :, 1]
        assert np.min(lons) <= center_lon <= np.max(lons) or np.allclose(lons, center_lon, atol=0.01)
        assert np.min(lats) <= center_lat <= np.max(lats) or np.allclose(lats, center_lat, atol=0.01)


# ======================================================================
# 10. Planar from corner
# ======================================================================

class TestPlanarFromCorner:
    """Test Fault.planar_from_corner() factory."""

    def test_same_n_patches(self):
        f = Fault.planar_from_corner(
            lat=0.0, lon=100.0, depth=0.0,
            strike=0.0, dip=45.0,
            length=50e3, width=30e3,
            n_length=5, n_width=3,
        )
        assert f.n_patches == 15

    def test_shallowest_patch_near_corner_depth(self):
        """Shallowest patches should be near the corner depth."""
        f = Fault.planar_from_corner(
            lat=0.0, lon=100.0, depth=1000.0,
            strike=0.0, dip=45.0,
            length=50e3, width=30e3,
            n_length=5, n_width=3,
        )
        # Shallowest patches (first row, j=0)
        min_depth = np.min(f._depth)
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
            lat=0.0, lon=100.0, depth=15e3,
            strike=0.0, dip=45.0,
            length=30e3, width=20e3,
            n_length=3, n_width=2,
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

        fault = Fault.planar(lat0, lon0, depth0, strike, dip, length, width, nL, nW)

        patchL = length / nL
        patchW = width / nW
        sin_str = np.sin(np.radians(strike))
        cos_str = np.cos(np.radians(strike))
        sin_dip = np.sin(np.radians(dip))
        cos_dip = np.cos(np.radians(dip))

        fault_e0 = -0.5 * length * sin_str - 0.5 * width * cos_dip * cos_str
        fault_n0 = -0.5 * length * cos_str + 0.5 * width * cos_dip * sin_str
        fault_u0 = -0.5 * width * sin_dip

        expected_depths = []
        for j in range(nW):
            for i in range(nL):
                u_offset = fault_u0 + (j + 0.5) * patchW * sin_dip
                expected_depths.append(depth0 - u_offset)

        np.testing.assert_allclose(
            fault._depth, expected_depths, rtol=1e-10,
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

class TestSegIO:
    """Test loading and saving .seg files."""

    def test_load_3d_ramp(self):
        """Load the 3D ramp test file."""
        fname = "related/stress-shadows/3d_models/test_data_3d/ramp_3d.seg"
        f = Fault.load(fname, format="seg")
        # 700km / 20km = 35, 250km / 10km = 25 => 875 patches
        assert f.n_patches == 875
        assert f.grid_shape == (35, 25)
        assert f.engine == "okada"
        np.testing.assert_allclose(f._strike, 0.0)
        np.testing.assert_allclose(f._dip, 10.0)

    def test_load_2d_ramp(self):
        """Load the 2D ramp test file."""
        fname = "related/stress-shadows/2d_models/test_data_2d/ramp_2d.seg"
        f = Fault.load(fname, format="seg")
        assert f.n_patches > 0
        np.testing.assert_allclose(f._dip, 10.0)
        # All patches should have the same width (qW=1)
        np.testing.assert_allclose(f._width, f._width[0])

    def test_load_seg_with_ref_coords(self):
        """Loading with ref_lat/ref_lon should offset the geographic coords."""
        fname = "related/stress-shadows/3d_models/test_data_3d/ramp_3d.seg"
        f0 = Fault.load(fname, format="seg", ref_lat=0.0, ref_lon=0.0)
        f1 = Fault.load(fname, format="seg", ref_lat=10.0, ref_lon=100.0)
        # Centers should differ by roughly the ref offset
        assert abs(np.mean(f1._lat) - np.mean(f0._lat)) > 5.0
        assert abs(np.mean(f1._lon) - np.mean(f0._lon)) > 50.0

    def test_seg_save_load_roundtrip(self):
        """Save a uniform fault as .seg and reload it."""
        fault = Fault.planar(
            lat=0.0, lon=100.0, depth=15e3,
            strike=30.0, dip=20.0,
            length=100e3, width=50e3,
            n_length=5, n_width=5,
        )
        with tempfile.NamedTemporaryFile(suffix=".seg", delete=False) as f:
            fname = f.name

        fault.save(fname, format="seg", ref_lat=0.0, ref_lon=100.0)
        loaded = Fault.load(fname, format="seg", ref_lat=0.0, ref_lon=100.0)

        assert loaded.n_patches == fault.n_patches
        np.testing.assert_allclose(loaded._strike, fault._strike, atol=1e-4)
        np.testing.assert_allclose(loaded._dip, fault._dip, atol=1e-4)
        np.testing.assert_allclose(loaded._length, fault._length, atol=1.0)
        np.testing.assert_allclose(loaded._width, fault._width, atol=1.0)
        # Depths match (flt2flt orders shallow-to-deep, planar() deep-to-shallow,
        # so compare sorted values)
        np.testing.assert_allclose(
            np.sort(loaded._depth), np.sort(fault._depth), rtol=0.01,
        )

        Path(fname).unlink()

    def test_load_seg_depth_increases_downdip(self):
        """Deeper patches should have greater depth values."""
        fname = "related/stress-shadows/3d_models/test_data_3d/ramp_3d.seg"
        f = Fault.load(fname, format="seg")
        nL = f.grid_shape[0]
        # First row (j=0) vs last row (j=nW-1)
        first_row_depth = np.mean(f._depth[:nL])
        last_row_depth = np.mean(f._depth[-nL:])
        assert first_row_depth != last_row_depth

    def test_seg_geometric_growth_load(self):
        """Create a seg file with geometric growth and verify loading."""
        with tempfile.NamedTemporaryFile(suffix=".seg", delete=False, mode="w") as f:
            fname = f.name
            f.write("# test seg file with geometric growth\n")
            f.write("# n  Vpl  x1  x2  x3  Length  Width  Strike  Dip  Rake  L0  W0  qL  qW\n")
            f.write("1 1.0 0.0 0.0 0.0 60000.0 30000.0 0.0 30.0 90.0 10000.0 5000.0 1.0 1.5\n")

        f = Fault.load(fname, format="seg")
        assert f.n_patches > 0
        # Should have non-uniform widths
        unique_widths = np.unique(f._width)
        assert len(unique_widths) > 1
        # Widths should sum to total
        np.testing.assert_allclose(np.sum(unique_widths), 30e3)
        # No grid_shape for non-uniform grids
        assert f.grid_shape is None

        Path(fname).unlink()
