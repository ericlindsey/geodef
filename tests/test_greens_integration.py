"""Tests for geodef.greens() polymorphic Green's matrix assembly (Phase 3.3).

Covers: greens() with single and joint datasets, all data types (GNSS,
InSAR, Vertical), consistency with fault.displacement(), stack_obs(),
stack_weights(), _project_greens(), and tri engine support.
"""

import numpy as np
import pytest

import geodef
from geodef.data import GNSS, InSAR, Vertical
from geodef.fault import Fault
from geodef.greens import greens, stack_obs, stack_weights, _project_greens


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def fault_4x3():
    """A 4x3 planar fault (12 patches)."""
    return Fault.planar(
        lat=0.0, lon=100.0, depth=15e3,
        strike=320.0, dip=15.0,
        length=80e3, width=40e3,
        n_length=4, n_width=3,
    )


@pytest.fixture
def single_patch():
    """A single-patch fault for simple tests."""
    return Fault.planar(
        lat=0.0, lon=100.0, depth=10e3,
        strike=0.0, dip=90.0,
        length=10e3, width=10e3,
        n_length=1, n_width=1,
    )


@pytest.fixture
def obs_points():
    """4 observation points at ~50km from the fault."""
    lat = np.array([0.5, -0.5, 0.0, 0.0])
    lon = np.array([100.0, 100.0, 100.5, 99.5])
    return lat, lon


@pytest.fixture
def gnss_4station(obs_points):
    """4-station GNSS dataset (3-component)."""
    lat, lon = obs_points
    n = len(lat)
    return GNSS(
        lat, lon,
        ve=np.zeros(n), vn=np.zeros(n), vu=np.zeros(n),
        se=np.ones(n), sn=np.ones(n), su=np.ones(n),
    )


@pytest.fixture
def gnss_horizontal(obs_points):
    """4-station horizontal-only GNSS dataset."""
    lat, lon = obs_points
    n = len(lat)
    return GNSS(
        lat, lon,
        ve=np.zeros(n), vn=np.zeros(n), vu=None,
        se=np.ones(n), sn=np.ones(n), su=None,
    )


@pytest.fixture
def insar_4pixel(obs_points):
    """4-pixel InSAR dataset with ascending look vectors."""
    lat, lon = obs_points
    n = len(lat)
    return InSAR(
        lat, lon,
        los=np.zeros(n), sigma=np.ones(n),
        look_e=np.full(n, 0.38),
        look_n=np.full(n, -0.09),
        look_u=np.full(n, 0.92),
    )


@pytest.fixture
def vertical_4pt(obs_points):
    """4-point Vertical dataset."""
    lat, lon = obs_points
    n = len(lat)
    return Vertical(lat, lon, displacement=np.zeros(n), sigma=np.ones(n))


# ======================================================================
# 1. greens() shape tests — single dataset
# ======================================================================

class TestGreensShape:
    """Verify output shapes for all data types."""

    def test_gnss_3comp_shape(self, fault_4x3, gnss_4station):
        G = greens(fault_4x3, gnss_4station)
        assert G.shape == (12, 24)  # 3*4 obs, 2*12 patches

    def test_gnss_horizontal_shape(self, fault_4x3, gnss_horizontal):
        G = greens(fault_4x3, gnss_horizontal)
        assert G.shape == (8, 24)  # 2*4 obs, 2*12 patches

    def test_insar_shape(self, fault_4x3, insar_4pixel):
        G = greens(fault_4x3, insar_4pixel)
        assert G.shape == (4, 24)  # 1*4 obs, 2*12 patches

    def test_vertical_shape(self, fault_4x3, vertical_4pt):
        G = greens(fault_4x3, vertical_4pt)
        assert G.shape == (4, 24)  # 1*4 obs, 2*12 patches


# ======================================================================
# 2. greens() shape tests — joint datasets
# ======================================================================

class TestGreensJoint:
    """Verify stacking behavior with multiple datasets."""

    def test_gnss_plus_insar_shape(self, fault_4x3, gnss_4station, insar_4pixel):
        G = greens(fault_4x3, [gnss_4station, insar_4pixel])
        assert G.shape == (16, 24)  # (12 + 4) rows

    def test_all_three_types_shape(self, fault_4x3, gnss_4station, insar_4pixel, vertical_4pt):
        G = greens(fault_4x3, [gnss_4station, insar_4pixel, vertical_4pt])
        assert G.shape == (20, 24)  # (12 + 4 + 4) rows

    def test_joint_equals_individual_vstack(self, fault_4x3, gnss_4station, insar_4pixel):
        G_joint = greens(fault_4x3, [gnss_4station, insar_4pixel])
        G_gnss = greens(fault_4x3, gnss_4station)
        G_insar = greens(fault_4x3, insar_4pixel)
        np.testing.assert_array_equal(G_joint, np.vstack([G_gnss, G_insar]))


# ======================================================================
# 3. Consistency with fault.displacement()
# ======================================================================

class TestConsistencyWithDisplacement:
    """Verify that greens() @ slip == fault.displacement() results."""

    def test_gnss_forward_model(self, single_patch, obs_points):
        lat, lon = obs_points
        gnss = GNSS(
            lat, lon,
            ve=np.zeros(4), vn=np.zeros(4), vu=np.zeros(4),
            se=np.ones(4), sn=np.ones(4), su=np.ones(4),
        )

        G = greens(single_patch, gnss)
        slip_s, slip_d = 1.5, 0.3
        m = np.array([slip_s, slip_d])  # single patch, 2 slip components
        pred = G @ m

        ue, un, uz = single_patch.displacement(lat, lon, slip_strike=slip_s, slip_dip=slip_d)
        expected = gnss.project(ue, un, uz)
        np.testing.assert_allclose(pred, expected, rtol=1e-10)

    def test_insar_forward_model(self, single_patch, obs_points):
        lat, lon = obs_points
        n = len(lat)
        insar = InSAR(
            lat, lon,
            los=np.zeros(n), sigma=np.ones(n),
            look_e=np.full(n, 0.38),
            look_n=np.full(n, -0.09),
            look_u=np.full(n, 0.92),
        )

        G = greens(single_patch, insar)
        slip_s, slip_d = 1.0, 0.0
        m = np.array([slip_s, slip_d])
        pred = G @ m

        ue, un, uz = single_patch.displacement(lat, lon, slip_strike=slip_s, slip_dip=slip_d)
        expected = insar.project(ue, un, uz)
        np.testing.assert_allclose(pred, expected, rtol=1e-10)

    def test_vertical_forward_model(self, single_patch, obs_points):
        lat, lon = obs_points
        n = len(lat)
        vert = Vertical(lat, lon, displacement=np.zeros(n), sigma=np.ones(n))

        G = greens(single_patch, vert)
        slip_s, slip_d = 0.0, 2.0
        m = np.array([slip_s, slip_d])
        pred = G @ m

        ue, un, uz = single_patch.displacement(lat, lon, slip_strike=slip_s, slip_dip=slip_d)
        expected = vert.project(ue, un, uz)
        np.testing.assert_allclose(pred, expected, rtol=1e-10)

    def test_multi_patch_forward_model(self, fault_4x3, obs_points):
        """Multi-patch fault: G @ m should reproduce displacement()."""
        lat, lon = obs_points
        n = len(lat)
        gnss = GNSS(
            lat, lon,
            ve=np.zeros(n), vn=np.zeros(n), vu=np.zeros(n),
            se=np.ones(n), sn=np.ones(n), su=np.ones(n),
        )

        rng = np.random.default_rng(42)
        slip_s = rng.uniform(0, 2, fault_4x3.n_patches)
        slip_d = rng.uniform(-1, 1, fault_4x3.n_patches)

        G = greens(fault_4x3, gnss)
        n = fault_4x3.n_patches
        m = np.empty(2 * n)
        m[:n] = slip_s
        m[n:] = slip_d
        pred = G @ m

        ue, un, uz = fault_4x3.displacement(lat, lon, slip_strike=slip_s, slip_dip=slip_d)
        expected = gnss.project(ue, un, uz)
        np.testing.assert_allclose(pred, expected, rtol=1e-10)


# ======================================================================
# 4. Zero slip gives zero prediction
# ======================================================================

class TestZeroSlip:
    """Zero slip columns should give zero predictions."""

    def test_zero_slip_gnss(self, fault_4x3, gnss_4station):
        G = greens(fault_4x3, gnss_4station)
        m = np.zeros(2 * fault_4x3.n_patches)
        pred = G @ m
        np.testing.assert_allclose(pred, 0.0, atol=1e-15)

    def test_zero_slip_insar(self, fault_4x3, insar_4pixel):
        G = greens(fault_4x3, insar_4pixel)
        m = np.zeros(2 * fault_4x3.n_patches)
        pred = G @ m
        np.testing.assert_allclose(pred, 0.0, atol=1e-15)


# ======================================================================
# 5. Linearity
# ======================================================================

class TestLinearity:
    """Scaled slip should produce proportionally scaled predictions."""

    def test_linearity_gnss(self, single_patch, gnss_4station):
        G = greens(single_patch, gnss_4station)
        m1 = np.array([1.0, 0.5])
        m2 = 3.0 * m1
        np.testing.assert_allclose(G @ m2, 3.0 * (G @ m1), rtol=1e-10)


# ======================================================================
# 6. stack_obs and stack_weights
# ======================================================================

class TestStackUtilities:
    """Tests for stack_obs() and stack_weights()."""

    def test_stack_obs_single(self, gnss_4station):
        obs = stack_obs(gnss_4station)
        np.testing.assert_array_equal(obs, gnss_4station.obs)

    def test_stack_obs_joint(self, gnss_4station, insar_4pixel):
        obs = stack_obs([gnss_4station, insar_4pixel])
        expected = np.concatenate([gnss_4station.obs, insar_4pixel.obs])
        np.testing.assert_array_equal(obs, expected)

    def test_stack_obs_three(self, gnss_4station, insar_4pixel, vertical_4pt):
        obs = stack_obs([gnss_4station, insar_4pixel, vertical_4pt])
        assert obs.shape == (gnss_4station.n_obs + insar_4pixel.n_obs + vertical_4pt.n_obs,)

    def test_stack_weights_single_diagonal(self, gnss_4station):
        W = stack_weights(gnss_4station)
        assert W.shape == (gnss_4station.n_obs, gnss_4station.n_obs)
        expected = np.diag(1.0 / gnss_4station.sigma ** 2)
        np.testing.assert_allclose(W, expected, rtol=1e-10)

    def test_stack_weights_joint_block_diagonal(self, gnss_4station, insar_4pixel):
        W = stack_weights([gnss_4station, insar_4pixel])
        n_gnss = gnss_4station.n_obs
        n_insar = insar_4pixel.n_obs
        assert W.shape == (n_gnss + n_insar, n_gnss + n_insar)
        # Off-diagonal blocks should be zero
        np.testing.assert_allclose(W[:n_gnss, n_gnss:], 0.0, atol=1e-15)
        np.testing.assert_allclose(W[n_gnss:, :n_gnss], 0.0, atol=1e-15)

    def test_stack_weights_positive_definite(self, gnss_4station, insar_4pixel):
        W = stack_weights([gnss_4station, insar_4pixel])
        eigvals = np.linalg.eigvalsh(W)
        assert np.all(eigvals > 0)


# ======================================================================
# 7. _project_greens internal helper
# ======================================================================

class TestProjectGreens:
    """Tests for the _project_greens helper function."""

    def test_insar_projection_reduces_rows(self, fault_4x3, insar_4pixel):
        G_raw = fault_4x3.greens_matrix(insar_4pixel.lat, insar_4pixel.lon)
        G_proj = _project_greens(insar_4pixel, G_raw)
        assert G_proj.shape == (4, 24)  # from (12, 24) to (4, 24)

    def test_gnss_3comp_preserves_rows(self, fault_4x3, gnss_4station):
        G_raw = fault_4x3.greens_matrix(gnss_4station.lat, gnss_4station.lon)
        G_proj = _project_greens(gnss_4station, G_raw)
        assert G_proj.shape == (12, 24)  # 3*4 stays 3*4

    def test_vertical_extracts_uz(self, single_patch, obs_points):
        lat, lon = obs_points
        n = len(lat)
        vert = Vertical(lat, lon, displacement=np.zeros(n), sigma=np.ones(n))
        G_raw = single_patch.greens_matrix(lat, lon)
        G_proj = _project_greens(vert, G_raw)

        # Vertical projection should extract rows 2, 5, 8, 11 from G_raw
        G_uz = G_raw[2::3, :]
        np.testing.assert_array_equal(G_proj, G_uz)


# ======================================================================
# 8. Top-level API access
# ======================================================================

class TestTopLevelAPI:
    """Verify greens() is accessible from geodef namespace."""

    def test_greens_accessible(self):
        assert hasattr(geodef.greens, 'greens')

    def test_stack_obs_accessible(self):
        assert hasattr(geodef, 'stack_obs')

    def test_stack_weights_accessible(self):
        assert hasattr(geodef, 'stack_weights')


# ======================================================================
# 9. Tri engine support in fault.greens_matrix()
# ======================================================================

class TestTriEngine:
    """Test that greens_matrix works with engine='tri'."""

    @pytest.fixture
    def tri_fault(self):
        """Build a simple tri fault from two triangles forming a rectangle.

        Creates a vertical, N-S striking rectangle at 10km depth,
        split into two triangles.
        """
        # Rectangle: 10km x 10km, vertical, NS strike, centered at depth 10km
        # In local ENU: corners at (±5000, 0, -5000) to (±5000, 0, -15000)
        v1 = np.array([
            [-5000.0, 0.0, -5000.0],
            [5000.0, 0.0, -5000.0],
            [5000.0, 0.0, -15000.0],
        ])
        v2 = np.array([
            [-5000.0, 0.0, -5000.0],
            [5000.0, 0.0, -15000.0],
            [-5000.0, 0.0, -15000.0],
        ])
        vertices = np.array([v1, v2])

        lat = np.array([0.0, 0.0])
        lon = np.array([100.0, 100.0])
        depth = np.array([10000.0, 10000.0])
        strike = np.array([0.0, 0.0])
        dip = np.array([90.0, 90.0])

        return Fault(lat, lon, depth, strike, dip, None, None,
                     vertices=vertices, engine="tri")

    def test_tri_greens_matrix_shape(self, tri_fault):
        obs_lat = np.array([0.1, -0.1])
        obs_lon = np.array([100.1, 99.9])
        G = tri_fault.greens_matrix(obs_lat, obs_lon, kind="displacement")
        assert G.shape == (6, 4)  # 3*2 obs, 2*2 patches

    def test_tri_strain_greens_shape(self, tri_fault):
        obs_lat = np.array([0.1])
        obs_lon = np.array([100.1])
        G = tri_fault.greens_matrix(obs_lat, obs_lon, kind="strain")
        assert G.shape == (6, 4)  # 6*1 obs, 2*2 patches

    def test_tri_greens_nonzero(self, tri_fault):
        obs_lat = np.array([0.1])
        obs_lon = np.array([100.1])
        G = tri_fault.greens_matrix(obs_lat, obs_lon, kind="displacement")
        assert np.any(G != 0.0)

    def test_tri_with_greens_function(self, tri_fault):
        """greens() works with tri engine faults."""
        obs_lat = np.array([0.1, -0.1])
        obs_lon = np.array([100.1, 99.9])
        n = len(obs_lat)
        gnss = GNSS(
            obs_lat, obs_lon,
            ve=np.zeros(n), vn=np.zeros(n), vu=np.zeros(n),
            se=np.ones(n), sn=np.ones(n), su=np.ones(n),
        )
        G = greens(tri_fault, gnss)
        assert G.shape == (6, 4)

    def test_tri_zero_slip(self, tri_fault):
        obs_lat = np.array([0.1])
        obs_lon = np.array([100.1])
        G = tri_fault.greens_matrix(obs_lat, obs_lon, kind="displacement")
        m = np.zeros(4)
        np.testing.assert_allclose(G @ m, 0.0, atol=1e-15)

    def test_tri_invalid_kind_raises(self, tri_fault):
        obs_lat = np.array([0.1])
        obs_lon = np.array([100.1])
        with pytest.raises(ValueError, match="Unknown kind"):
            tri_fault.greens_matrix(obs_lat, obs_lon, kind="tilt")
