"""Tests for LinearSystem — cached matrix products and method parity.

Verifies that:
- LinearSystem methods produce identical results to module-level functions.
- Cached properties (GtWG, LtL, Gtwd) are computed lazily and return
  the correct values.
- lcurve wls fast-path matches the augmented-matrix fallback.
- abic_curve uses cached eig_LtL (checked via internal cache entry).
- dataset_diagnostics / model_covariance / model_resolution /
  model_uncertainty agree with the module-level wrappers.
- Validation errors are raised correctly from __init__ and invert().
"""

import numpy as np
import pytest

from geodef.data import GNSS, InSAR
from geodef.fault import Fault
from geodef.greens import stack_weights
from geodef.invert import (
    LinearSystem,
    abic_curve,
    dataset_diagnostics,
    invert,
    lcurve,
    model_covariance,
    model_resolution,
    model_uncertainty,
)


# ======================================================================
# Fixtures (shared with test_invert.py pattern)
# ======================================================================

@pytest.fixture
def fault_4x3():
    return Fault.planar(
        lat=0.0, lon=100.0, depth=15e3,
        strike=320.0, dip=15.0,
        length=80e3, width=40e3,
        n_length=4, n_width=3,
    )


@pytest.fixture
def obs_points():
    lat_1d = np.linspace(-0.5, 0.5, 5)
    lon_1d = np.linspace(99.5, 100.5, 5)
    lon_g, lat_g = np.meshgrid(lon_1d, lat_1d)
    return lat_g.ravel(), lon_g.ravel()


@pytest.fixture
def gnss(fault_4x3, obs_points):
    lat, lon = obs_points
    n = len(lat)
    slip_ss = np.zeros(fault_4x3.n_patches)
    slip_ds = np.ones(fault_4x3.n_patches) * 2.0
    ue, un, uz = fault_4x3.displacement(lat, lon, slip_ss, slip_ds)
    return GNSS(
        lon, lat,
        ve=ue, vn=un, vu=uz,
        se=np.full(n, 0.001), sn=np.full(n, 0.001), su=np.full(n, 0.001),
    )


@pytest.fixture
def insar(fault_4x3, obs_points):
    lat, lon = obs_points
    n = len(lat)
    slip_ss = np.zeros(fault_4x3.n_patches)
    slip_ds = np.ones(fault_4x3.n_patches) * 2.0
    ue, un, uz = fault_4x3.displacement(lat, lon, slip_ss, slip_ds)
    look_e = np.full(n, -0.38)
    look_n = np.full(n, 0.09)
    look_u = np.full(n, 0.92)
    los = look_e * ue + look_n * un + look_u * uz
    return InSAR(
        lon, lat,
        los=los, sigma=np.full(n, 0.001),
        look_e=look_e, look_n=look_n, look_u=look_u,
    )


# ======================================================================
# Construction and cached properties
# ======================================================================

class TestLinearSystemConstruction:
    def test_basic_construction(self, fault_4x3, gnss):
        sys = LinearSystem(fault_4x3, gnss)
        assert sys.G.shape[1] == 2 * fault_4x3.n_patches
        assert len(sys.d) == gnss.n_obs

    def test_single_dataset_normalized(self, fault_4x3, gnss):
        sys = LinearSystem(fault_4x3, gnss)
        assert isinstance(sys.datasets, list)
        assert len(sys.datasets) == 1

    def test_smoothing_builds_L(self, fault_4x3, gnss):
        sys = LinearSystem(fault_4x3, gnss, smoothing="laplacian")
        assert sys.L is not None
        assert sys.L.shape[1] == 2 * fault_4x3.n_patches

    def test_no_smoothing_L_is_none(self, fault_4x3, gnss):
        sys = LinearSystem(fault_4x3, gnss)
        assert sys.L is None

    def test_invalid_components_raises(self, fault_4x3, gnss):
        with pytest.raises(ValueError, match="components"):
            LinearSystem(fault_4x3, gnss, components="sideways")

    def test_invalid_dataset_type_raises(self, fault_4x3):
        with pytest.raises(TypeError, match="DataSet"):
            LinearSystem(fault_4x3, ["not_a_dataset"])


class TestCachedProperties:
    def test_GtWG_shape(self, fault_4x3, gnss):
        sys = LinearSystem(fault_4x3, gnss)
        n = 2 * fault_4x3.n_patches
        assert sys.GtWG.shape == (n, n)

    def test_GtWG_equals_manual(self, fault_4x3, gnss):
        sys = LinearSystem(fault_4x3, gnss)
        expected = sys.G_w.T @ sys.G_w
        np.testing.assert_allclose(sys.GtWG, expected)

    def test_GtWG_cached(self, fault_4x3, gnss):
        sys = LinearSystem(fault_4x3, gnss)
        a = sys.GtWG
        b = sys.GtWG
        assert a is b  # same object — no recomputation

    def test_LtL_shape(self, fault_4x3, gnss):
        sys = LinearSystem(fault_4x3, gnss, smoothing="laplacian")
        n = 2 * fault_4x3.n_patches
        assert sys.LtL.shape == (n, n)

    def test_LtL_equals_manual(self, fault_4x3, gnss):
        sys = LinearSystem(fault_4x3, gnss, smoothing="laplacian")
        np.testing.assert_allclose(sys.LtL, sys.L.T @ sys.L)

    def test_LtL_raises_without_smoothing(self, fault_4x3, gnss):
        sys = LinearSystem(fault_4x3, gnss)
        with pytest.raises(AttributeError):
            _ = sys.LtL

    def test_Gtwd_shape(self, fault_4x3, gnss):
        sys = LinearSystem(fault_4x3, gnss)
        assert sys.Gtwd.shape == (2 * fault_4x3.n_patches,)

    def test_Gtwd_equals_manual(self, fault_4x3, gnss):
        sys = LinearSystem(fault_4x3, gnss)
        expected = sys.G_w.T @ sys.d_w
        np.testing.assert_allclose(sys.Gtwd, expected)


# ======================================================================
# invert() method parity with module-level function
# ======================================================================

class TestInvertMethodParity:
    def test_unregularized_matches_wrapper(self, fault_4x3, gnss):
        sys = LinearSystem(fault_4x3, gnss)
        r_method = sys.invert()
        r_func = invert(fault_4x3, gnss)
        np.testing.assert_allclose(r_method.slip_vector, r_func.slip_vector)

    def test_regularized_matches_wrapper(self, fault_4x3, gnss):
        sys = LinearSystem(fault_4x3, gnss, smoothing="laplacian")
        r_method = sys.invert(smoothing_strength=10.0)
        r_func = invert(fault_4x3, gnss, smoothing="laplacian",
                        smoothing_strength=10.0)
        np.testing.assert_allclose(r_method.slip_vector, r_func.slip_vector)

    def test_multi_dataset_matches_wrapper(self, fault_4x3, gnss, insar):
        sys = LinearSystem(fault_4x3, [gnss, insar], smoothing="damping")
        r_method = sys.invert(smoothing_strength=1.0)
        r_func = invert(fault_4x3, [gnss, insar], smoothing="damping",
                        smoothing_strength=1.0)
        np.testing.assert_allclose(r_method.slip_vector, r_func.slip_vector)

    def test_strike_only_matches_wrapper(self, fault_4x3, gnss):
        sys = LinearSystem(fault_4x3, gnss, components="strike")
        r_method = sys.invert()
        r_func = invert(fault_4x3, gnss, components="strike")
        np.testing.assert_allclose(r_method.slip_vector, r_func.slip_vector)

    def test_nnls_matches_wrapper(self, fault_4x3, gnss):
        sys = LinearSystem(fault_4x3, gnss, smoothing="damping")
        r_method = sys.invert(smoothing_strength=1.0, bounds=(0, None))
        r_func = invert(fault_4x3, gnss, smoothing="damping",
                        smoothing_strength=1.0, bounds=(0, None))
        np.testing.assert_allclose(r_method.slip_vector, r_func.slip_vector,
                                   rtol=1e-5)

    def test_invalid_smoothing_strength_raises(self, fault_4x3, gnss):
        sys = LinearSystem(fault_4x3, gnss)
        with pytest.raises(ValueError):
            sys.invert(smoothing_strength="abic")


# ======================================================================
# lcurve() method parity and fast-path correctness
# ======================================================================

class TestLcurveMethod:
    def test_matches_wrapper(self, fault_4x3, gnss):
        sys = LinearSystem(fault_4x3, gnss, smoothing="laplacian")
        lc_method = sys.lcurve(smoothing_range=(1e-1, 1e3), n=5)
        lc_func = lcurve(fault_4x3, gnss, smoothing="laplacian",
                         smoothing_range=(1e-1, 1e3), n=5)
        np.testing.assert_allclose(lc_method.misfits, lc_func.misfits,
                                   rtol=1e-10)
        np.testing.assert_allclose(lc_method.model_norms, lc_func.model_norms,
                                   rtol=1e-10)

    def test_wls_fast_path_matches_augmented(self, fault_4x3, gnss):
        """WLS fast path (GtWG + lam*LtL solve) must equal augmented lstsq."""
        sys = LinearSystem(fault_4x3, gnss, smoothing="laplacian")
        lambdas = [0.1, 1.0, 10.0]
        for lam in lambdas:
            H = sys.GtWG + lam * sys.LtL
            m_fast = np.linalg.solve(H, sys.Gtwd)
            G_aug = np.vstack([sys.G_w, np.sqrt(lam) * sys.L])
            d_aug = np.concatenate([sys.d_w, np.zeros(sys.L.shape[0])])
            GtG = G_aug.T @ G_aug
            m_aug = np.linalg.solve(GtG, G_aug.T @ d_aug)
            np.testing.assert_allclose(m_fast, m_aug, atol=1e-10,
                                       err_msg=f"Mismatch at lam={lam}")

    def test_no_smoothing_raises(self, fault_4x3, gnss):
        sys = LinearSystem(fault_4x3, gnss)
        with pytest.raises(ValueError, match="smoothing"):
            sys.lcurve()

    def test_returns_lcurve_result(self, fault_4x3, gnss):
        from geodef.invert import LCurveResult
        sys = LinearSystem(fault_4x3, gnss, smoothing="damping")
        lc = sys.lcurve(n=5)
        assert isinstance(lc, LCurveResult)
        assert len(lc.misfits) == 5


# ======================================================================
# abic_curve() method parity
# ======================================================================

class TestAbicCurveMethod:
    def test_matches_wrapper(self, fault_4x3, gnss):
        sys = LinearSystem(fault_4x3, gnss, smoothing="laplacian")
        ac_method = sys.abic_curve(smoothing_range=(1e-1, 1e3), n=5)
        ac_func = abic_curve(fault_4x3, gnss, smoothing="laplacian",
                             smoothing_range=(1e-1, 1e3), n=5)
        np.testing.assert_allclose(ac_method.abic_values, ac_func.abic_values,
                                   rtol=1e-10)
        np.testing.assert_allclose(ac_method.misfits, ac_func.misfits,
                                   rtol=1e-10)

    def test_eig_LtL_cached_after_abic_curve(self, fault_4x3, gnss):
        sys = LinearSystem(fault_4x3, gnss, smoothing="laplacian")
        sys.abic_curve(n=3)
        assert "_eig_LtL" in sys.__dict__

    def test_no_smoothing_raises(self, fault_4x3, gnss):
        sys = LinearSystem(fault_4x3, gnss)
        with pytest.raises(ValueError, match="smoothing"):
            sys.abic_curve()

    def test_optimal_is_minimum_abic(self, fault_4x3, gnss):
        sys = LinearSystem(fault_4x3, gnss, smoothing="damping")
        ac = sys.abic_curve(n=10)
        assert ac.optimal == ac.smoothing_values[np.argmin(ac.abic_values)]


# ======================================================================
# Post-inversion methods parity
# ======================================================================

class TestPostInversionParity:
    @pytest.fixture
    def result_and_sys(self, fault_4x3, gnss):
        sys = LinearSystem(fault_4x3, gnss, smoothing="laplacian")
        result = sys.invert(smoothing_strength=5.0)
        return result, sys

    def test_dataset_diagnostics_matches_wrapper(
        self, fault_4x3, gnss, result_and_sys,
    ):
        result, sys = result_and_sys
        diag_method = sys.dataset_diagnostics(result)
        diag_func = dataset_diagnostics(result, fault_4x3, gnss)
        assert len(diag_method) == len(diag_func)
        assert diag_method[0].chi2 == pytest.approx(diag_func[0].chi2)
        assert diag_method[0].leverage == pytest.approx(diag_func[0].leverage)

    def test_model_covariance_matches_wrapper(
        self, fault_4x3, gnss, result_and_sys,
    ):
        result, sys = result_and_sys
        Cm_method = sys.model_covariance(result)
        Cm_func = model_covariance(result, fault_4x3, gnss)
        np.testing.assert_allclose(Cm_method, Cm_func, rtol=1e-10)

    def test_model_resolution_matches_wrapper(
        self, fault_4x3, gnss, result_and_sys,
    ):
        result, sys = result_and_sys
        R_method = sys.model_resolution(result)
        R_func = model_resolution(result, fault_4x3, gnss)
        np.testing.assert_allclose(R_method, R_func, rtol=1e-10)

    def test_model_uncertainty_matches_wrapper(
        self, fault_4x3, gnss, result_and_sys,
    ):
        result, sys = result_and_sys
        unc_method = sys.model_uncertainty(result)
        unc_func = model_uncertainty(result, fault_4x3, gnss)
        np.testing.assert_allclose(unc_method, unc_func, rtol=1e-10)

    def test_GtWG_reused_across_calls(self, fault_4x3, gnss):
        """GtWG cached_property is computed once and shared by all methods."""
        sys = LinearSystem(fault_4x3, gnss, smoothing="laplacian")
        result = sys.invert(smoothing_strength=5.0)
        # Trigger several methods that all use GtWG
        _ = sys.model_resolution(result)
        _ = sys.model_covariance(result)
        _ = sys.dataset_diagnostics(result)
        # The cached value should be the same object throughout
        assert "GtWG" in sys.__dict__

    def test_unregularized_covariance(self, fault_4x3, gnss):
        sys = LinearSystem(fault_4x3, gnss)
        result = sys.invert()
        Cm = sys.model_covariance(result)
        # Unregularized: Cm = (GtWG)^{-1}, so GtWG @ Cm ≈ I
        product = sys.GtWG @ Cm
        np.testing.assert_allclose(product, np.eye(product.shape[0]), atol=1e-8)
