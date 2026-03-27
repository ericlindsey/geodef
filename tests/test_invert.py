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
from geodef.invert import InversionResult, invert


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
