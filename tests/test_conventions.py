"""Regularization-convention tests spanning every solver path.

docs/conventions.md declares one convention:

    Phi(m) = (d - G m)^T W (d - G m) + lambda * ||L (m - m_ref)||^2

with augmented rows sqrt(lambda) * L. These tests pin that convention in the
direct solver, LinearSystem, the augmented system, model covariance, ABIC,
gradient-based geometry search, and the Bayesian fixed-lambda (profiled)
posterior.

The key instrument is scaling invariance: replacing (L, lambda) by
(L / sqrt(c), c * lambda) leaves Phi unchanged under the declared convention,
but shifts it under a lambda^2 convention. Any path that disagrees with the
others is using a different convention.
"""

import importlib.util

import numpy as np
import numpy.testing as npt
import pytest
import scipy.linalg

import geodef
from geodef import GNSS, Fault
from geodef.greens import matrix as greens
from geodef.greens import stack_obs, stack_weights

LAM = 3.0e-2
SCALE = 7.5  # arbitrary c != 1

has_jax = importlib.util.find_spec("jax") is not None
has_blackjax = importlib.util.find_spec("blackjax") is not None


@pytest.fixture(scope="module")
def problem():
    """Small synthetic slip inversion with a known answer."""
    fault = Fault.planar(
        lat=0.0,
        lon=100.0,
        depth=12_000.0,
        strike=90.0,
        dip=30.0,
        length=40_000.0,
        width=24_000.0,
        n_length=4,
        n_width=3,
    )
    rng = np.random.default_rng(42)
    n_sta = 30
    obs_lat = rng.uniform(-0.3, 0.3, n_sta)
    obs_lon = rng.uniform(99.7, 100.3, n_sta)

    slip_true = np.zeros(2 * fault.n_patches)
    slip_true[fault.n_patches :] = 1.0  # 1 m dip slip

    G_true = greens(fault, _gnss(obs_lat, obs_lon, np.zeros(3 * n_sta)))
    d_clean = G_true @ slip_true
    noise = rng.normal(0.0, 0.002, d_clean.size)
    gnss = _gnss(obs_lat, obs_lon, d_clean + noise)

    lap = fault.laplacian
    L = scipy.linalg.block_diag(lap, lap)
    return fault, gnss, L


def _gnss(lat: np.ndarray, lon: np.ndarray, d: np.ndarray) -> GNSS:
    n = lat.size
    d = d.reshape(n, 3)
    s = np.full(n, 0.002)
    return GNSS(
        lon=lon,
        lat=lat,
        ve=d[:, 0],
        vn=d[:, 1],
        vu=d[:, 2],
        se=s,
        sn=s,
        su=s,
    )


class TestLinearPaths:
    def test_direct_solve_matches_normal_equations(self, problem) -> None:
        """invert() solves (GtWG + lambda LtL) m = GtWd — lambda, not lambda^2."""
        fault, gnss, L = problem
        result = geodef.invert.solve(fault, gnss, smoothing=L, smoothing_strength=LAM)
        G = greens(fault, gnss)
        W = stack_weights(gnss)
        d = stack_obs(gnss)
        m_manual = np.linalg.solve(G.T @ W @ G + LAM * (L.T @ L), G.T @ W @ d)
        npt.assert_allclose(result.slip_vector, m_manual, rtol=1e-8)

    def test_linear_system_matches_invert(self, problem) -> None:
        fault, gnss, L = problem
        a = geodef.invert.solve(fault, gnss, smoothing=L, smoothing_strength=LAM)
        b = geodef.LinearSystem(fault, gnss, L).invert(smoothing_strength=LAM)
        npt.assert_allclose(a.slip_vector, b.slip_vector, rtol=1e-12)

    def test_augmented_system_equivalence(self, problem) -> None:
        """Stacking sqrt(lambda) L rows reproduces the direct solution."""
        fault, gnss, L = problem
        G = greens(fault, gnss)
        W = stack_weights(gnss)
        d = stack_obs(gnss)
        sqrtW = np.diag(np.sqrt(np.diag(W)))
        G_aug = np.vstack([sqrtW @ G, np.sqrt(LAM) * L])
        d_aug = np.concatenate([sqrtW @ d, np.zeros(L.shape[0])])
        m_aug, *_ = np.linalg.lstsq(G_aug, d_aug, rcond=None)
        result = geodef.invert.solve(fault, gnss, smoothing=L, smoothing_strength=LAM)
        npt.assert_allclose(result.slip_vector, m_aug, rtol=1e-6, atol=1e-10)

    def test_invert_scaling_invariance(self, problem) -> None:
        """(L, lam) -> (L/sqrt(c), c lam) must not change the solution."""
        fault, gnss, L = problem
        a = geodef.invert.solve(fault, gnss, smoothing=L, smoothing_strength=LAM)
        b = geodef.invert.solve(
            fault,
            gnss,
            smoothing=L / np.sqrt(SCALE),
            smoothing_strength=SCALE * LAM,
        )
        npt.assert_allclose(a.slip_vector, b.slip_vector, rtol=1e-9)

    def test_model_covariance_convention(self, problem) -> None:
        """C_m = (GtWG + lambda LtL)^-1 with lambda to the first power."""
        fault, gnss, L = problem
        result = geodef.invert.solve(fault, gnss, smoothing=L, smoothing_strength=LAM)
        C = geodef.model_covariance(result, fault, gnss)
        G = greens(fault, gnss)
        W = stack_weights(gnss)
        C_manual = np.linalg.inv(G.T @ W @ G + LAM * (L.T @ L))
        npt.assert_allclose(C, C_manual, rtol=1e-8)


class TestABICConvention:
    def test_abic_scaling_invariance(self, problem) -> None:
        fault, gnss, L = problem
        G = greens(fault, gnss)
        W = stack_weights(gnss)
        d = stack_obs(gnss)
        a = geodef.compute_abic(G, d, W, L, LAM)
        b = geodef.compute_abic(G, d, W, L / np.sqrt(SCALE), SCALE * LAM)
        assert a == pytest.approx(b, rel=1e-9)

    def test_abic_distinguishes_lambdas(self, problem) -> None:
        """Sanity: ABIC is not constant in lambda (the invariance above
        is not vacuous)."""
        fault, gnss, L = problem
        G = greens(fault, gnss)
        W = stack_weights(gnss)
        d = stack_obs(gnss)
        a = geodef.compute_abic(G, d, W, L, LAM)
        b = geodef.compute_abic(G, d, W, L, 100 * LAM)
        assert a != pytest.approx(b, rel=1e-6)


@pytest.mark.skipif(not has_jax, reason="requires jax")
class TestGeometrySearchConvention:
    def test_geometry_search_scaling_invariance(self, problem) -> None:
        from geodef import backend

        fault, gnss, L = problem
        theta0 = np.array([0.0, 0.0, 12_000.0, 90.0, 25.0, 40_000.0, 24_000.0])
        backend.set_backend("jax")
        try:
            kwargs = dict(
                ref_lat=0.0,
                ref_lon=100.0,
                free=["dip"],
                n_length=4,
                n_width=3,
            )
            a = geodef.geometry_search(
                theta0, gnss, smoothing=L, smoothing_strength=LAM, **kwargs
            )
            b = geodef.geometry_search(
                theta0,
                gnss,
                smoothing=L / np.sqrt(SCALE),
                smoothing_strength=SCALE * LAM,
                **kwargs,
            )
            npt.assert_allclose(a.theta, b.theta, rtol=1e-5)
        finally:
            backend.set_backend("numpy")


@pytest.mark.skipif(not (has_jax and has_blackjax), reason="requires jax and blackjax")
class TestBayesFixedLambdaConvention:
    def test_profiled_logpdf_scaling_invariance(self, problem) -> None:
        from geodef import backend, bayes

        fault, gnss, L = problem
        theta0 = np.array([0.0, 0.0, 12_000.0, 90.0, 30.0, 40_000.0, 24_000.0])
        backend.set_backend("jax")
        try:
            kwargs = dict(
                ref_lat=0.0,
                ref_lon=100.0,
                free=["dip", "depth"],
                theta_prior={"dip": (5.0, 60.0), "depth": (2e3, 30e3)},
                n_length=4,
                n_width=3,
                mode="profiled",
            )
            post_a = bayes.RectPosterior(
                theta0, gnss, smoothing=L, smoothing_strength=LAM, **kwargs
            )
            post_b = bayes.RectPosterior(
                theta0,
                gnss,
                smoothing=L / np.sqrt(SCALE),
                smoothing_strength=SCALE * LAM,
                **kwargs,
            )
            # param_names is ['dip', 'depth', 'log10_sigma'] in profiled mode
            assert post_a.param_names == ["dip", "depth", "log10_sigma"]
            x = np.array([28.0, 11_000.0, 0.1])
            npt.assert_allclose(
                float(post_a.logpdf(x)), float(post_b.logpdf(x)), rtol=1e-8
            )
        finally:
            backend.set_backend("numpy")
