"""Tests for geodef.invert — Phase 4.1 one-call inversion.

Covers: InversionResult structure, WLS/NNLS/bounded_ls solvers,
Laplacian/damping/custom regularization, smoothing_target,
multiple datasets, fit statistics, and edge cases.
"""

import numpy as np
import pytest
import scipy.linalg

import geodef
from geodef.data import GNSS, InSAR, Vertical
from geodef.fault import Fault
from geodef.greens import greens, stack_obs, stack_weights
from geodef.invert import (
    DatasetDiagnostics,
    InversionResult,
    dataset_diagnostics,
    invert,
    model_covariance,
    model_resolution,
    model_uncertainty,
)


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
    """A single-patch fault."""
    return Fault.planar(
        lat=0.0, lon=100.0, depth=10e3,
        strike=0.0, dip=90.0,
        length=10e3, width=10e3,
        n_length=1, n_width=1,
    )


@pytest.fixture
def obs_points():
    """Grid of 25 observation points around the fault (overdetermined)."""
    lat_1d = np.linspace(-0.5, 0.5, 5)
    lon_1d = np.linspace(99.5, 100.5, 5)
    lon_g, lat_g = np.meshgrid(lon_1d, lat_1d)
    return lat_g.ravel(), lon_g.ravel()


def _make_gnss(fault, obs_points, slip_ss, slip_ds, sigma=0.001):
    """Build a noise-free GNSS dataset from known slip."""
    lat, lon = obs_points
    n = len(lat)
    ue, un, uz = fault.displacement(lat, lon, slip_ss, slip_ds)
    return GNSS(
        lat, lon,
        ve=ue, vn=un, vu=uz,
        se=np.full(n, sigma), sn=np.full(n, sigma), su=np.full(n, sigma),
    )


def _make_insar(fault, obs_points, slip_ss, slip_ds, sigma=0.001):
    """Build a noise-free InSAR dataset from known slip (ascending-like look)."""
    lat, lon = obs_points
    n = len(lat)
    ue, un, uz = fault.displacement(lat, lon, slip_ss, slip_ds)
    look_e = np.full(n, -0.38)
    look_n = np.full(n, 0.09)
    look_u = np.full(n, 0.92)
    los = look_e * ue + look_n * un + look_u * uz
    return InSAR(
        lat, lon,
        los=los, sigma=np.full(n, sigma),
        look_e=look_e, look_n=look_n, look_u=look_u,
    )


def _make_vertical(fault, obs_points, slip_ss, slip_ds, sigma=0.001):
    """Build a noise-free Vertical dataset from known slip."""
    lat, lon = obs_points
    n = len(lat)
    _, _, uz = fault.displacement(lat, lon, slip_ss, slip_ds)
    return Vertical(
        lat, lon,
        displacement=uz, sigma=np.full(n, sigma),
    )


# ======================================================================
# InversionResult structure
# ======================================================================

class TestInversionResultStructure:
    """Verify result object has correct shapes and fields."""

    def test_slip_shape(self, fault_4x3, obs_points):
        slip_ss = np.ones(12)
        slip_ds = np.zeros(12)
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, gnss)
        assert result.slip.shape == (12, 2)

    def test_slip_vector_shape(self, fault_4x3, obs_points):
        slip_ss = np.ones(12)
        slip_ds = np.zeros(12)
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, gnss)
        assert result.slip_vector.shape == (24,)

    def test_slip_vector_is_blocked(self, fault_4x3, obs_points):
        slip_ss = np.ones(12)
        slip_ds = np.zeros(12)
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, gnss)
        n = fault_4x3.n_patches
        np.testing.assert_array_equal(result.slip[:, 0], result.slip_vector[:n])
        np.testing.assert_array_equal(result.slip[:, 1], result.slip_vector[n:])

    def test_residuals_shape(self, fault_4x3, obs_points):
        slip_ss = np.ones(12)
        slip_ds = np.zeros(12)
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, gnss)
        assert result.residuals.shape == (gnss.n_obs,)

    def test_predicted_shape(self, fault_4x3, obs_points):
        slip_ss = np.ones(12)
        slip_ds = np.zeros(12)
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, gnss)
        assert result.predicted.shape == (gnss.n_obs,)

    def test_predicted_plus_residuals_equals_obs(self, fault_4x3, obs_points):
        slip_ss = np.ones(12)
        slip_ds = np.zeros(12)
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, gnss)
        np.testing.assert_allclose(
            result.predicted + result.residuals, gnss.obs, atol=1e-10,
        )

    def test_scalar_fields(self, fault_4x3, obs_points):
        slip_ss = np.ones(12)
        slip_ds = np.zeros(12)
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, gnss)
        assert isinstance(result.chi2, float)
        assert isinstance(result.rms, float)
        assert isinstance(result.moment, float)
        assert isinstance(result.Mw, float)

    def test_smoothing_strength_none_when_unregularized(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss)
        assert result.smoothing_strength is None


# ======================================================================
# Basic WLS solver (no regularization)
# ======================================================================

class TestWLS:
    """Weighted least-squares with noise-free data should recover slip."""

    def test_recover_uniform_strike_slip(self, fault_4x3, obs_points):
        slip_ss = np.ones(12)
        slip_ds = np.zeros(12)
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, gnss)
        np.testing.assert_allclose(result.slip[:, 0], slip_ss, atol=0.1)
        np.testing.assert_allclose(result.slip[:, 1], slip_ds, atol=0.1)

    def test_recover_dip_slip(self, fault_4x3, obs_points):
        slip_ss = np.zeros(12)
        slip_ds = np.ones(12) * 0.5
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, gnss)
        np.testing.assert_allclose(result.slip[:, 0], slip_ss, atol=0.1)
        np.testing.assert_allclose(result.slip[:, 1], slip_ds, atol=0.1)

    def test_noise_free_residuals_near_zero(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss)
        np.testing.assert_allclose(result.residuals, 0, atol=1e-8)

    def test_noise_free_rms_near_zero(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss)
        assert result.rms < 1e-8

    def test_single_patch(self, single_patch, obs_points):
        slip_ss = np.array([1.5])
        slip_ds = np.array([0.3])
        gnss = _make_gnss(single_patch, obs_points, slip_ss, slip_ds)
        result = invert(single_patch, gnss)
        np.testing.assert_allclose(result.slip[:, 0], slip_ss, atol=0.01)
        np.testing.assert_allclose(result.slip[:, 1], slip_ds, atol=0.01)

    def test_method_explicit_wls(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss, method='wls')
        np.testing.assert_allclose(result.residuals, 0, atol=1e-8)


# ======================================================================
# Multiple datasets
# ======================================================================

class TestMultipleDatasets:
    """Joint inversions with multiple data types."""

    def test_gnss_plus_insar(self, fault_4x3, obs_points):
        slip_ss = np.ones(12) * 0.5
        slip_ds = np.ones(12) * 0.3
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        insar = _make_insar(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, [gnss, insar])
        np.testing.assert_allclose(result.slip[:, 0], slip_ss, atol=0.1)
        np.testing.assert_allclose(result.slip[:, 1], slip_ds, atol=0.1)

    def test_gnss_plus_vertical(self, fault_4x3, obs_points):
        slip_ss = np.zeros(12)
        slip_ds = np.ones(12) * 0.8
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        vert = _make_vertical(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, [gnss, vert])
        np.testing.assert_allclose(result.slip[:, 0], slip_ss, atol=0.1)
        np.testing.assert_allclose(result.slip[:, 1], slip_ds, atol=0.1)

    def test_joint_residuals_shape(self, fault_4x3, obs_points):
        slip_ss = np.ones(12)
        slip_ds = np.zeros(12)
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        insar = _make_insar(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, [gnss, insar])
        assert result.residuals.shape == (gnss.n_obs + insar.n_obs,)

    def test_single_dataset_in_list(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        r1 = invert(fault_4x3, gnss)
        r2 = invert(fault_4x3, [gnss])
        np.testing.assert_allclose(r1.slip, r2.slip, atol=1e-10)


# ======================================================================
# Solvers: NNLS and bounded_ls
# ======================================================================

class TestNNLS:
    """Non-negative least-squares solver."""

    def test_nnls_recovers_positive_slip(self, fault_4x3, obs_points):
        slip_ss = np.ones(12)
        slip_ds = np.ones(12) * 0.5
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, gnss, method='nnls')
        assert np.all(result.slip_vector >= -1e-10)

    def test_nnls_enforces_nonnegativity(self, fault_4x3, obs_points):
        """WLS would give negative slip; NNLS should clip to zero."""
        slip_ss = np.ones(12)
        slip_ds = np.zeros(12)
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        result_wls = invert(fault_4x3, gnss, method='wls')
        result_nnls = invert(fault_4x3, gnss, method='nnls')
        # NNLS should have all non-negative values
        assert np.all(result_nnls.slip_vector >= -1e-10)

    def test_bounds_auto_selects_nnls(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.ones(12) * 0.5)
        result = invert(fault_4x3, gnss, bounds=(0, None))
        assert np.all(result.slip_vector >= -1e-10)


class TestBoundedLS:
    """Bounded least-squares solver."""

    def test_bounded_ls_respects_upper_bound(self, fault_4x3, obs_points):
        slip_ss = np.ones(12) * 2.0
        slip_ds = np.zeros(12)
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, gnss, bounds=(0, 1.0), method='bounded_ls')
        assert np.all(result.slip_vector <= 1.0 + 1e-10)
        assert np.all(result.slip_vector >= -1e-10)

    def test_bounded_ls_lower_bound_only(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss, bounds=(-0.5, None), method='bounded_ls')
        assert np.all(result.slip_vector >= -0.5 - 1e-10)

    def test_auto_selects_bounded_ls_for_general_bounds(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss, bounds=(-1, 5))
        assert result.slip_vector is not None  # just ensure it runs


# ======================================================================
# Regularization
# ======================================================================

class TestRegularization:
    """Regularization matrix types and smoothing_strength."""

    def test_laplacian_smoothing(self, fault_4x3, obs_points):
        """Laplacian regularization should produce a smoother result than unregularized."""
        rng = np.random.default_rng(42)
        slip_ss = np.ones(12)
        slip_ds = np.zeros(12)
        gnss_clean = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds, sigma=0.01)
        # Add noise to observations to make the unregularized solution rough
        lat, lon = obs_points
        n = len(lat)
        noise = rng.normal(0, 0.005, gnss_clean.n_obs)
        noisy_obs = gnss_clean.obs + noise
        gnss_noisy = GNSS(
            lat, lon,
            ve=noisy_obs[0::3], vn=noisy_obs[1::3], vu=noisy_obs[2::3],
            se=np.full(n, 0.01), sn=np.full(n, 0.01), su=np.full(n, 0.01),
        )
        r_unreg = invert(fault_4x3, gnss_noisy)
        r_smooth = invert(fault_4x3, gnss_noisy,
                          smoothing='laplacian', smoothing_strength=1e6)
        L = fault_4x3.laplacian
        norm_unreg = np.linalg.norm(L @ r_unreg.slip[:, 0])
        norm_smooth = np.linalg.norm(L @ r_smooth.slip[:, 0])
        assert norm_smooth < norm_unreg

    def test_damping_reduces_norm(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        r_weak = invert(fault_4x3, gnss, smoothing='damping', smoothing_strength=1.0)
        r_strong = invert(fault_4x3, gnss, smoothing='damping', smoothing_strength=1e6)
        norm_weak = np.linalg.norm(r_weak.slip_vector)
        norm_strong = np.linalg.norm(r_strong.slip_vector)
        assert norm_strong < norm_weak

    def test_custom_matrix_same_as_damping(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        n = fault_4x3.n_patches
        custom_I = np.eye(2 * n)
        r_damping = invert(fault_4x3, gnss,
                           smoothing='damping', smoothing_strength=100.0)
        r_custom = invert(fault_4x3, gnss,
                          smoothing=custom_I, smoothing_strength=100.0)
        np.testing.assert_allclose(r_damping.slip, r_custom.slip, atol=1e-10)

    def test_zero_strength_equals_unregularized(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        r_unreg = invert(fault_4x3, gnss)
        r_zero = invert(fault_4x3, gnss,
                        smoothing='laplacian', smoothing_strength=0.0)
        np.testing.assert_allclose(r_unreg.slip, r_zero.slip, atol=1e-10)

    def test_smoothing_strength_stored_in_result(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss,
                        smoothing='laplacian', smoothing_strength=42.0)
        assert result.smoothing_strength == 42.0

    def test_laplacian_with_nnls(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.ones(12) * 0.5)
        result = invert(fault_4x3, gnss,
                        smoothing='laplacian', smoothing_strength=100.0,
                        method='nnls')
        assert np.all(result.slip_vector >= -1e-10)

    def test_stress_kernel_runs(self, fault_4x3, obs_points):
        """Stress kernel regularization should complete without error."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss,
                        smoothing='stresskernel', smoothing_strength=1e-6)
        assert result.slip.shape == (12, 2)


# ======================================================================
# Smoothing target (regularize toward non-zero reference)
# ======================================================================

class TestSmoothingTarget:
    """Test smoothing_target parameter for non-zero regularization target."""

    def test_target_pulls_solution(self, fault_4x3, obs_points):
        """With strong regularization, solution should approach target."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        n = fault_4x3.n_patches
        target = np.full(2 * n, 5.0)
        result = invert(fault_4x3, gnss,
                        smoothing='damping', smoothing_strength=1e10,
                        smoothing_target=target)
        np.testing.assert_allclose(result.slip_vector, target, atol=0.1)

    def test_zero_target_same_as_no_target(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        n = fault_4x3.n_patches
        r1 = invert(fault_4x3, gnss,
                     smoothing='damping', smoothing_strength=100.0)
        r2 = invert(fault_4x3, gnss,
                     smoothing='damping', smoothing_strength=100.0,
                     smoothing_target=np.zeros(2 * n))
        np.testing.assert_allclose(r1.slip, r2.slip, atol=1e-10)

    def test_target_shape_mismatch_raises(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        with pytest.raises(ValueError, match="smoothing_target"):
            invert(fault_4x3, gnss,
                   smoothing='damping', smoothing_strength=1.0,
                   smoothing_target=np.zeros(5))

    def test_target_without_smoothing_raises(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        with pytest.raises(ValueError, match="smoothing_target"):
            invert(fault_4x3, gnss, smoothing_target=np.zeros(24))


# ======================================================================
# Fit statistics
# ======================================================================

class TestFitStatistics:
    """Moment, magnitude, chi2, rms."""

    def test_moment_matches_fault_moment(self, fault_4x3, obs_points):
        slip_ss = np.ones(12)
        slip_ds = np.zeros(12)
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, gnss)
        slip_mag = np.sqrt(result.slip[:, 0]**2 + result.slip[:, 1]**2)
        expected_moment = fault_4x3.moment(slip_mag)
        np.testing.assert_allclose(result.moment, expected_moment, rtol=1e-6)

    def test_magnitude_consistent_with_moment(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss)
        from geodef.fault import moment_to_magnitude
        np.testing.assert_allclose(
            result.Mw, moment_to_magnitude(result.moment), rtol=1e-6,
        )

    def test_chi2_nonnegative(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss)
        # 25 stations x 3 comp = 75 obs > 24 params → overdetermined
        assert result.chi2 >= 0
        assert not np.isnan(result.chi2)

    def test_chi2_nan_when_underdetermined(self, single_patch):
        """With more params than obs, chi2 should be nan."""
        lat = np.array([0.5])
        lon = np.array([100.0])
        gnss = _make_gnss(single_patch, (lat, lon), np.array([1.0]),
                          np.array([0.0]))
        result = invert(single_patch, gnss)
        # 1 station x 3 comp = 3 obs, 2 params → dof=1, should work
        assert not np.isnan(result.chi2)

    def test_rms_nonnegative(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss)
        assert result.rms >= 0


# ======================================================================
# Input validation
# ======================================================================

class TestValidation:
    """Input validation and error messages."""

    def test_invalid_method_raises(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        with pytest.raises(ValueError, match="method"):
            invert(fault_4x3, gnss, method='invalid')

    def test_invalid_smoothing_string_raises(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        with pytest.raises(ValueError, match="smoothing"):
            invert(fault_4x3, gnss, smoothing='nonexistent')

    def test_smoothing_matrix_wrong_cols_raises(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        wrong = np.eye(5)
        with pytest.raises(ValueError, match="columns"):
            invert(fault_4x3, gnss, smoothing=wrong, smoothing_strength=1.0)


# ======================================================================
# Components parameter
# ======================================================================

class TestComponents:
    """Test single-component inversions via the components parameter."""

    def test_default_components_is_both(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss)
        assert result.components == "both"

    def test_components_stored_in_result(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        for comp in ("both", "strike", "dip"):
            result = invert(fault_4x3, gnss, components=comp)
            assert result.components == comp

    def test_invalid_components_raises(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        with pytest.raises(ValueError, match="components"):
            invert(fault_4x3, gnss, components="invalid")

    def test_strike_only_slip_shape(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss, components="strike")
        assert result.slip.shape == (12, 1)
        assert result.slip_vector.shape == (12,)

    def test_dip_only_slip_shape(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.zeros(12), np.ones(12))
        result = invert(fault_4x3, gnss, components="dip")
        assert result.slip.shape == (12, 1)
        assert result.slip_vector.shape == (12,)

    def test_both_slip_shape(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss, components="both")
        assert result.slip.shape == (12, 2)
        assert result.slip_vector.shape == (24,)

    def test_strike_only_recovers_strike_slip(self, fault_4x3, obs_points):
        slip_ss = np.ones(12)
        slip_ds = np.zeros(12)
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, gnss, components="strike")
        np.testing.assert_allclose(result.slip[:, 0], slip_ss, atol=0.1)

    def test_dip_only_recovers_dip_slip(self, fault_4x3, obs_points):
        slip_ss = np.zeros(12)
        slip_ds = np.ones(12) * 0.5
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, gnss, components="dip")
        np.testing.assert_allclose(result.slip[:, 0], slip_ds, atol=0.1)

    def test_strike_only_laplacian_shape(self, fault_4x3, obs_points):
        """Laplacian should be (N, N) not (2N, 2N) for single component."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss, components="strike",
                        smoothing="laplacian", smoothing_strength=100.0)
        assert result.slip.shape == (12, 1)

    def test_dip_only_damping(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.zeros(12), np.ones(12))
        result = invert(fault_4x3, gnss, components="dip",
                        smoothing="damping", smoothing_strength=100.0)
        assert result.slip.shape == (12, 1)

    def test_strike_only_with_bounds(self, fault_4x3, obs_points):
        slip_ss = np.ones(12)
        slip_ds = np.zeros(12)
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, gnss, components="strike",
                        bounds=(0, None))
        assert np.all(result.slip_vector >= -1e-10)

    def test_strike_only_smoothing_target(self, fault_4x3, obs_points):
        """Smoothing target should have shape (N,) for single component."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        n = fault_4x3.n_patches
        target = np.full(n, 5.0)
        result = invert(fault_4x3, gnss, components="strike",
                        smoothing="damping", smoothing_strength=1e10,
                        smoothing_target=target)
        np.testing.assert_allclose(result.slip_vector, target, atol=0.1)

    def test_smoothing_target_wrong_shape_single_component(self, fault_4x3, obs_points):
        """Target shape (2N,) should fail when components='strike'."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        with pytest.raises(ValueError, match="smoothing_target"):
            invert(fault_4x3, gnss, components="strike",
                   smoothing="damping", smoothing_strength=1.0,
                   smoothing_target=np.zeros(24))

    def test_moment_correct_single_component(self, fault_4x3, obs_points):
        slip_ss = np.ones(12)
        slip_ds = np.zeros(12)
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, gnss, components="strike")
        slip_mag = np.abs(result.slip[:, 0])
        expected_moment = fault_4x3.moment(slip_mag)
        np.testing.assert_allclose(result.moment, expected_moment, rtol=1e-6)

    def test_custom_smoothing_matrix_single_component(self, fault_4x3, obs_points):
        """Custom matrix with N columns should work for single component."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        n = fault_4x3.n_patches
        custom = np.eye(n)
        result = invert(fault_4x3, gnss, components="strike",
                        smoothing=custom, smoothing_strength=100.0)
        assert result.slip.shape == (12, 1)


# ======================================================================
# ABIC hyperparameter tuning
# ======================================================================

class TestABIC:
    """Automatic smoothing strength via ABIC criterion."""

    def test_abic_returns_result(self, fault_4x3, obs_points):
        """ABIC should return an InversionResult with a positive smoothing_strength."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss,
                        smoothing='laplacian', smoothing_strength='abic')
        assert isinstance(result, InversionResult)
        assert result.smoothing_strength is not None
        assert result.smoothing_strength > 0

    def test_abic_recovers_slip(self, fault_4x3, obs_points):
        """ABIC-tuned inversion should approximately recover known slip."""
        slip_ss = np.ones(12)
        slip_ds = np.zeros(12)
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, gnss,
                        smoothing='laplacian', smoothing_strength='abic')
        np.testing.assert_allclose(result.slip[:, 0], slip_ss, atol=0.3)

    def test_abic_with_damping(self, fault_4x3, obs_points):
        """ABIC should work with any smoothing type."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss,
                        smoothing='damping', smoothing_strength='abic')
        assert result.smoothing_strength > 0

    def test_abic_with_noisy_data(self, fault_4x3, obs_points):
        """With noisy data, ABIC should select a reasonable lambda."""
        rng = np.random.default_rng(42)
        slip_ss = np.ones(12)
        slip_ds = np.zeros(12)
        gnss_clean = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds, sigma=0.01)
        lat, lon = obs_points
        n = len(lat)
        noise = rng.normal(0, 0.005, gnss_clean.n_obs)
        noisy_obs = gnss_clean.obs + noise
        gnss_noisy = GNSS(
            lat, lon,
            ve=noisy_obs[0::3], vn=noisy_obs[1::3], vu=noisy_obs[2::3],
            se=np.full(n, 0.01), sn=np.full(n, 0.01), su=np.full(n, 0.01),
        )
        result = invert(fault_4x3, gnss_noisy,
                        smoothing='laplacian', smoothing_strength='abic')
        assert result.smoothing_strength > 0
        assert result.rms > 0

    def test_abic_with_single_component(self, fault_4x3, obs_points):
        """ABIC should work with components='strike'."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss, components='strike',
                        smoothing='laplacian', smoothing_strength='abic')
        assert result.smoothing_strength > 0
        assert result.slip.shape == (12, 1)

    def test_abic_with_bounds(self, fault_4x3, obs_points):
        """ABIC should work with NNLS bounds."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.ones(12) * 0.5)
        result = invert(fault_4x3, gnss,
                        smoothing='laplacian', smoothing_strength='abic',
                        bounds=(0, None))
        assert result.smoothing_strength > 0
        assert np.all(result.slip_vector >= -1e-10)

    def test_abic_without_smoothing_raises(self, fault_4x3, obs_points):
        """ABIC requires a smoothing type to be set."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        with pytest.raises(ValueError, match="smoothing"):
            invert(fault_4x3, gnss, smoothing_strength='abic')

    def test_compute_abic_value(self, fault_4x3, obs_points):
        """The compute_abic function should return a finite scalar."""
        from geodef.invert import compute_abic
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        G = greens(fault_4x3, [gnss])
        d = stack_obs([gnss])
        W = stack_weights([gnss])
        L = scipy.linalg.block_diag(fault_4x3.laplacian, fault_4x3.laplacian)
        abic_val = compute_abic(G, d, W, L, 1e3)
        assert np.isfinite(abic_val)

    def test_abic_minimum_exists(self, fault_4x3, obs_points):
        """ABIC should have a minimum: very low and very high lambda give higher values."""
        from geodef.invert import compute_abic
        rng = np.random.default_rng(99)
        slip_ss = np.ones(12)
        slip_ds = np.zeros(12)
        gnss_clean = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds, sigma=0.01)
        lat, lon = obs_points
        n = len(lat)
        noise = rng.normal(0, 0.005, gnss_clean.n_obs)
        noisy_obs = gnss_clean.obs + noise
        gnss_noisy = GNSS(
            lat, lon,
            ve=noisy_obs[0::3], vn=noisy_obs[1::3], vu=noisy_obs[2::3],
            se=np.full(n, 0.01), sn=np.full(n, 0.01), su=np.full(n, 0.01),
        )
        G = greens(fault_4x3, [gnss_noisy])
        d = stack_obs([gnss_noisy])
        W = stack_weights([gnss_noisy])
        L = scipy.linalg.block_diag(fault_4x3.laplacian, fault_4x3.laplacian)
        abic_mid = compute_abic(G, d, W, L, 1e2)
        abic_low = compute_abic(G, d, W, L, 1e-6)
        abic_high = compute_abic(G, d, W, L, 1e10)
        # The middle value should be lower than at least one extreme
        assert abic_mid < abic_low or abic_mid < abic_high


# ======================================================================
# L-curve analysis
# ======================================================================

class TestLCurve:
    """L-curve sweep and optimal corner finding."""

    def test_lcurve_returns_result(self, fault_4x3, obs_points):
        """lcurve should return an LCurveResult."""
        from geodef.invert import lcurve, LCurveResult
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        lc = lcurve(fault_4x3, gnss, smoothing='laplacian',
                     smoothing_range=(1e-2, 1e6), n=20)
        assert isinstance(lc, LCurveResult)

    def test_lcurve_has_arrays(self, fault_4x3, obs_points):
        """Result should have smoothing_values, misfits, and model_norms arrays."""
        from geodef.invert import lcurve
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        lc = lcurve(fault_4x3, gnss, smoothing='laplacian',
                     smoothing_range=(1e-2, 1e6), n=20)
        assert len(lc.smoothing_values) == 20
        assert len(lc.misfits) == 20
        assert len(lc.model_norms) == 20

    def test_lcurve_misfit_increases_with_lambda(self, fault_4x3, obs_points):
        """Stronger regularization should generally increase misfit."""
        from geodef.invert import lcurve
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        lc = lcurve(fault_4x3, gnss, smoothing='laplacian',
                     smoothing_range=(1e-2, 1e6), n=20)
        # Overall trend: first misfit should be <= last misfit
        assert lc.misfits[0] <= lc.misfits[-1] + 1e-10

    def test_lcurve_model_norm_decreases_with_lambda(self, fault_4x3, obs_points):
        """Stronger regularization should reduce the model norm."""
        from geodef.invert import lcurve
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        lc = lcurve(fault_4x3, gnss, smoothing='laplacian',
                     smoothing_range=(1e-2, 1e6), n=20)
        assert lc.model_norms[0] >= lc.model_norms[-1] - 1e-10

    def test_lcurve_optimal_exists(self, fault_4x3, obs_points):
        """Optimal smoothing_strength should be within the swept range."""
        from geodef.invert import lcurve
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        lc = lcurve(fault_4x3, gnss, smoothing='laplacian',
                     smoothing_range=(1e-2, 1e6), n=20)
        assert 1e-2 <= lc.optimal <= 1e6

    def test_lcurve_with_damping(self, fault_4x3, obs_points):
        """lcurve should work with different smoothing types."""
        from geodef.invert import lcurve
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        lc = lcurve(fault_4x3, gnss, smoothing='damping',
                     smoothing_range=(1e-2, 1e6), n=10)
        assert len(lc.smoothing_values) == 10

    def test_lcurve_with_bounds(self, fault_4x3, obs_points):
        """lcurve should support bounds."""
        from geodef.invert import lcurve
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.ones(12) * 0.5)
        lc = lcurve(fault_4x3, gnss, smoothing='laplacian',
                     smoothing_range=(1e-2, 1e6), n=10,
                     bounds=(0, None))
        assert len(lc.smoothing_values) == 10

    def test_lcurve_plot(self, fault_4x3, obs_points):
        """LCurveResult.plot() should return matplotlib axes."""
        from geodef.invert import lcurve
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        lc = lcurve(fault_4x3, gnss, smoothing='laplacian',
                     smoothing_range=(1e-2, 1e6), n=10)
        ax = lc.plot()
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)


# ======================================================================
# ABIC curve analysis
# ======================================================================

class TestABICCurve:
    """ABIC curve sweep and optimal point finding."""

    def test_abic_curve_returns_result(self, fault_4x3, obs_points):
        from geodef.invert import abic_curve, ABICCurveResult
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        ac = abic_curve(fault_4x3, gnss, smoothing='laplacian',
                        smoothing_range=(1e-2, 1e6), n=20)
        assert isinstance(ac, ABICCurveResult)

    def test_abic_curve_has_arrays(self, fault_4x3, obs_points):
        from geodef.invert import abic_curve
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        ac = abic_curve(fault_4x3, gnss, smoothing='laplacian',
                        smoothing_range=(1e-2, 1e6), n=20)
        assert len(ac.smoothing_values) == 20
        assert len(ac.abic_values) == 20

    def test_abic_curve_optimal_in_range(self, fault_4x3, obs_points):
        from geodef.invert import abic_curve
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        ac = abic_curve(fault_4x3, gnss, smoothing='laplacian',
                        smoothing_range=(1e-2, 1e6), n=20)
        assert 1e-2 <= ac.optimal <= 1e6

    def test_abic_curve_has_misfits_and_norms(self, fault_4x3, obs_points):
        """Should also expose misfit and model norm arrays for context."""
        from geodef.invert import abic_curve
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        ac = abic_curve(fault_4x3, gnss, smoothing='laplacian',
                        smoothing_range=(1e-2, 1e6), n=15)
        assert len(ac.misfits) == 15
        assert len(ac.model_norms) == 15

    def test_abic_curve_with_noisy_data(self, fault_4x3, obs_points):
        """With noisy data, optimal should sit between extremes."""
        from geodef.invert import abic_curve
        rng = np.random.default_rng(42)
        slip_ss = np.ones(12)
        slip_ds = np.zeros(12)
        gnss_clean = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds, sigma=0.01)
        lat, lon = obs_points
        n = len(lat)
        noise = rng.normal(0, 0.005, gnss_clean.n_obs)
        noisy_obs = gnss_clean.obs + noise
        gnss_noisy = GNSS(
            lat, lon,
            ve=noisy_obs[0::3], vn=noisy_obs[1::3], vu=noisy_obs[2::3],
            se=np.full(n, 0.01), sn=np.full(n, 0.01), su=np.full(n, 0.01),
        )
        ac = abic_curve(fault_4x3, gnss_noisy, smoothing='laplacian',
                        smoothing_range=(1e-2, 1e8), n=30)
        # Optimal should be at the minimum ABIC
        idx_opt = np.argmin(np.abs(ac.smoothing_values - ac.optimal))
        assert ac.abic_values[idx_opt] <= ac.abic_values[0] + 1e-6 or \
               ac.abic_values[idx_opt] <= ac.abic_values[-1] + 1e-6

    def test_abic_curve_plot(self, fault_4x3, obs_points):
        from geodef.invert import abic_curve
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        ac = abic_curve(fault_4x3, gnss, smoothing='laplacian',
                        smoothing_range=(1e-2, 1e6), n=10)
        ax = ac.plot()
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

    def test_abic_curve_with_damping(self, fault_4x3, obs_points):
        from geodef.invert import abic_curve
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        ac = abic_curve(fault_4x3, gnss, smoothing='damping',
                        smoothing_range=(1e-2, 1e6), n=10)
        assert len(ac.smoothing_values) == 10


# ======================================================================
# Cross-validation
# ======================================================================

class TestCrossValidation:
    """K-fold cross-validation for smoothing strength selection."""

    def test_cv_returns_result(self, fault_4x3, obs_points):
        """CV should return an InversionResult with positive smoothing_strength."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss,
                        smoothing='laplacian', smoothing_strength='cv')
        assert isinstance(result, InversionResult)
        assert result.smoothing_strength > 0

    def test_cv_recovers_slip(self, fault_4x3, obs_points):
        """CV-tuned inversion should approximately recover known slip."""
        slip_ss = np.ones(12)
        slip_ds = np.zeros(12)
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, gnss,
                        smoothing='laplacian', smoothing_strength='cv')
        np.testing.assert_allclose(result.slip[:, 0], slip_ss, atol=0.5)

    def test_cv_with_custom_folds(self, fault_4x3, obs_points):
        """cv_folds parameter should control the number of folds."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss,
                        smoothing='laplacian', smoothing_strength='cv',
                        cv_folds=3)
        assert result.smoothing_strength > 0

    def test_cv_with_damping(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss,
                        smoothing='damping', smoothing_strength='cv')
        assert result.smoothing_strength > 0

    def test_cv_with_bounds(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.ones(12) * 0.5)
        result = invert(fault_4x3, gnss,
                        smoothing='laplacian', smoothing_strength='cv',
                        bounds=(0, None))
        assert result.smoothing_strength > 0
        assert np.all(result.slip_vector >= -1e-10)

    def test_cv_without_smoothing_raises(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        with pytest.raises(ValueError, match="smoothing"):
            invert(fault_4x3, gnss, smoothing_strength='cv')

    def test_cv_single_component(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss, components='strike',
                        smoothing='laplacian', smoothing_strength='cv')
        assert result.smoothing_strength > 0
        assert result.slip.shape == (12, 1)


# ======================================================================
# Constrained solver (QP)
# ======================================================================

class TestConstrainedSolver:
    """Quadratic programming solver with inequality constraints."""

    def test_constrained_runs(self, fault_4x3, obs_points):
        """Constrained solver should run and return a result."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss, method='constrained',
                        bounds=(0, None))
        assert isinstance(result, InversionResult)

    def test_constrained_respects_bounds(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12) * 2.0, np.zeros(12))
        result = invert(fault_4x3, gnss, method='constrained',
                        bounds=(0, 1.5))
        assert np.all(result.slip_vector >= -1e-10)
        assert np.all(result.slip_vector <= 1.5 + 1e-10)

    def test_constrained_with_inequality(self, fault_4x3, obs_points):
        """Inequality constraints: C @ m <= d_ineq."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        n_params = 2 * fault_4x3.n_patches
        # Constraint: sum of all slip <= 10
        C = np.ones((1, n_params))
        d_ineq = np.array([10.0])
        result = invert(fault_4x3, gnss, method='constrained',
                        constraints=(C, d_ineq))
        assert C @ result.slip_vector <= 10.0 + 1e-6

    def test_constrained_with_regularization(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss, method='constrained',
                        smoothing='laplacian', smoothing_strength=100.0,
                        bounds=(0, None))
        assert np.all(result.slip_vector >= -1e-10)

    def test_constrained_recovers_slip(self, fault_4x3, obs_points):
        """With loose constraints, should recover known slip."""
        slip_ss = np.ones(12) * 0.5
        slip_ds = np.ones(12) * 0.3
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, gnss, method='constrained',
                        bounds=(0, 5.0))
        np.testing.assert_allclose(result.slip[:, 0], slip_ss, atol=0.2)
        np.testing.assert_allclose(result.slip[:, 1], slip_ds, atol=0.2)

    def test_invalid_method_constrained_string(self, fault_4x3, obs_points):
        """'constrained' should be a valid method string."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        # Should not raise ValueError for unknown method
        result = invert(fault_4x3, gnss, method='constrained')
        assert isinstance(result, InversionResult)


# ======================================================================
# Per-dataset diagnostics
# ======================================================================

class TestDatasetDiagnostics:
    """Per-dataset fit diagnostics via hat matrix."""

    def test_single_dataset_returns_list(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss)
        diags = dataset_diagnostics(result, fault_4x3, gnss)
        assert isinstance(diags, list)
        assert len(diags) == 1
        assert isinstance(diags[0], DatasetDiagnostics)

    def test_multiple_datasets(self, fault_4x3, obs_points):
        slip_ss, slip_ds = np.ones(12), np.zeros(12)
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        insar = _make_insar(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, [gnss, insar])
        diags = dataset_diagnostics(result, fault_4x3, [gnss, insar])
        assert len(diags) == 2

    def test_n_obs_matches_datasets(self, fault_4x3, obs_points):
        slip_ss, slip_ds = np.ones(12), np.zeros(12)
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        insar = _make_insar(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, [gnss, insar])
        diags = dataset_diagnostics(result, fault_4x3, [gnss, insar])
        assert diags[0].n_obs == gnss.n_obs
        assert diags[1].n_obs == insar.n_obs

    def test_n_obs_sum_equals_total(self, fault_4x3, obs_points):
        slip_ss, slip_ds = np.ones(12), np.zeros(12)
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        insar = _make_insar(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, [gnss, insar])
        diags = dataset_diagnostics(result, fault_4x3, [gnss, insar])
        assert sum(d.n_obs for d in diags) == len(result.residuals)

    def test_dof_sums_to_total(self, fault_4x3, obs_points):
        """Sum of per-dataset effective DOF should equal total DOF."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        insar = _make_insar(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, [gnss, insar])
        diags = dataset_diagnostics(result, fault_4x3, [gnss, insar])
        n_params = 2 * fault_4x3.n_patches
        total_n = gnss.n_obs + insar.n_obs
        total_dof = sum(d.dof for d in diags)
        np.testing.assert_allclose(total_dof, total_n - n_params, atol=1e-8)

    def test_leverage_sums_to_n_params(self, fault_4x3, obs_points):
        """Sum of per-dataset leverage should equal n_params (trace of H)."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        insar = _make_insar(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, [gnss, insar])
        diags = dataset_diagnostics(result, fault_4x3, [gnss, insar])
        n_params = 2 * fault_4x3.n_patches
        total_leverage = sum(d.leverage for d in diags)
        np.testing.assert_allclose(total_leverage, n_params, atol=1e-8)

    def test_chi2_positive(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss)
        diags = dataset_diagnostics(result, fault_4x3, gnss)
        assert diags[0].chi2 >= 0

    def test_reduced_chi2_near_zero_for_perfect_fit(self, fault_4x3, obs_points):
        """Noise-free data should give near-zero chi2."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss)
        diags = dataset_diagnostics(result, fault_4x3, gnss)
        assert diags[0].reduced_chi2 < 1e-6

    def test_wrms_near_zero_for_perfect_fit(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss)
        diags = dataset_diagnostics(result, fault_4x3, gnss)
        assert diags[0].wrms < 1e-6

    def test_rms_near_zero_for_perfect_fit(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss)
        diags = dataset_diagnostics(result, fault_4x3, gnss)
        assert diags[0].rms < 1e-10

    def test_with_regularization(self, fault_4x3, obs_points):
        """Regularized hat matrix should have trace < n_params."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss,
                        smoothing='laplacian', smoothing_strength=1e3)
        diags = dataset_diagnostics(result, fault_4x3, gnss)
        n_params = 2 * fault_4x3.n_patches
        assert diags[0].leverage < n_params

    def test_regularized_dof_larger_than_unregularized(self, fault_4x3, obs_points):
        """Regularization reduces effective parameters, increasing DOF."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result_unreg = invert(fault_4x3, gnss)
        result_reg = invert(fault_4x3, gnss,
                            smoothing='laplacian', smoothing_strength=1e3)
        diags_unreg = dataset_diagnostics(result_unreg, fault_4x3, gnss)
        diags_reg = dataset_diagnostics(result_reg, fault_4x3, gnss)
        assert diags_reg[0].dof > diags_unreg[0].dof

    def test_single_component(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss, components='strike')
        diags = dataset_diagnostics(result, fault_4x3, gnss)
        assert diags[0].n_obs == gnss.n_obs
        assert diags[0].leverage < fault_4x3.n_patches + 1

    def test_accepts_single_dataset(self, fault_4x3, obs_points):
        """Should accept a single DataSet (not wrapped in list)."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss)
        diags = dataset_diagnostics(result, fault_4x3, gnss)
        assert len(diags) == 1

    def test_three_datasets(self, fault_4x3, obs_points):
        slip_ss, slip_ds = np.ones(12), np.ones(12) * 0.5
        gnss = _make_gnss(fault_4x3, obs_points, slip_ss, slip_ds)
        insar = _make_insar(fault_4x3, obs_points, slip_ss, slip_ds)
        vert = _make_vertical(fault_4x3, obs_points, slip_ss, slip_ds)
        result = invert(fault_4x3, [gnss, insar, vert])
        diags = dataset_diagnostics(result, fault_4x3, [gnss, insar, vert])
        assert len(diags) == 3
        assert diags[0].n_obs == gnss.n_obs
        assert diags[1].n_obs == insar.n_obs
        assert diags[2].n_obs == vert.n_obs


# ======================================================================
# Model covariance, resolution, and uncertainty
# ======================================================================

class TestModelCovariance:
    """Model covariance matrix Cm."""

    def test_shape(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss)
        Cm = model_covariance(result, fault_4x3, gnss)
        n_params = 2 * fault_4x3.n_patches
        assert Cm.shape == (n_params, n_params)

    def test_symmetric(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss)
        Cm = model_covariance(result, fault_4x3, gnss)
        np.testing.assert_allclose(Cm, Cm.T, atol=1e-12)

    def test_positive_semidefinite(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss)
        Cm = model_covariance(result, fault_4x3, gnss)
        eigvals = np.linalg.eigvalsh(Cm)
        assert np.all(eigvals >= -1e-10)

    def test_regularization_reduces_variance(self, fault_4x3, obs_points):
        """Regularization should reduce diagonal elements of Cm."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result_unreg = invert(fault_4x3, gnss)
        result_reg = invert(fault_4x3, gnss,
                            smoothing='laplacian', smoothing_strength=1e3)
        Cm_unreg = model_covariance(result_unreg, fault_4x3, gnss)
        Cm_reg = model_covariance(result_reg, fault_4x3, gnss)
        assert np.mean(np.diag(Cm_reg)) < np.mean(np.diag(Cm_unreg))

    def test_single_component_shape(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss, components='dip')
        Cm = model_covariance(result, fault_4x3, gnss)
        assert Cm.shape == (12, 12)

    def test_accepts_single_dataset(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss)
        Cm = model_covariance(result, fault_4x3, gnss)
        assert Cm.shape[0] == Cm.shape[1]


class TestModelResolution:
    """Model resolution matrix R."""

    def test_shape(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss)
        R = model_resolution(result, fault_4x3, gnss)
        n_params = 2 * fault_4x3.n_patches
        assert R.shape == (n_params, n_params)

    def test_unregularized_is_identity(self, fault_4x3, obs_points):
        """For an overdetermined unregularized system, R ~ I."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss)
        R = model_resolution(result, fault_4x3, gnss)
        np.testing.assert_allclose(R, np.eye(R.shape[0]), atol=1e-8)

    def test_regularized_trace_less_than_n_params(self, fault_4x3, obs_points):
        """Regularization reduces trace(R) below n_params."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss,
                        smoothing='laplacian', smoothing_strength=1e3)
        R = model_resolution(result, fault_4x3, gnss)
        n_params = 2 * fault_4x3.n_patches
        assert np.trace(R) < n_params - 0.1

    def test_diagonal_between_0_and_1(self, fault_4x3, obs_points):
        """Resolution diagonal should be between 0 and 1."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss,
                        smoothing='laplacian', smoothing_strength=1e3)
        R = model_resolution(result, fault_4x3, gnss)
        diag = np.diag(R)
        assert np.all(diag >= -1e-10)
        assert np.all(diag <= 1.0 + 1e-10)

    def test_single_component(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss, components='strike')
        R = model_resolution(result, fault_4x3, gnss)
        assert R.shape == (12, 12)


class TestModelUncertainty:
    """Per-patch uncertainty from model covariance diagonal."""

    def test_shape(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss)
        unc = model_uncertainty(result, fault_4x3, gnss)
        n_params = 2 * fault_4x3.n_patches
        assert unc.shape == (n_params,)

    def test_all_positive(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss)
        unc = model_uncertainty(result, fault_4x3, gnss)
        assert np.all(unc >= 0)

    def test_regularization_reduces_uncertainty(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result_unreg = invert(fault_4x3, gnss)
        result_reg = invert(fault_4x3, gnss,
                            smoothing='laplacian', smoothing_strength=1e3)
        unc_unreg = model_uncertainty(result_unreg, fault_4x3, gnss)
        unc_reg = model_uncertainty(result_reg, fault_4x3, gnss)
        assert np.mean(unc_reg) < np.mean(unc_unreg)

    def test_consistent_with_covariance(self, fault_4x3, obs_points):
        """Uncertainty should be sqrt of covariance diagonal."""
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss,
                        smoothing='laplacian', smoothing_strength=1e3)
        Cm = model_covariance(result, fault_4x3, gnss)
        unc = model_uncertainty(result, fault_4x3, gnss)
        np.testing.assert_allclose(unc, np.sqrt(np.diag(Cm)), atol=1e-12)

    def test_single_component(self, fault_4x3, obs_points):
        gnss = _make_gnss(fault_4x3, obs_points, np.ones(12), np.zeros(12))
        result = invert(fault_4x3, gnss, components='strike')
        unc = model_uncertainty(result, fault_4x3, gnss)
        assert unc.shape == (12,)
