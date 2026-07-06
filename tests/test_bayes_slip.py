"""Tests for geodef.bayes.SlipPosterior — joint slip sampling with positivity.

The density is validated exactly against an independent NumPy
reimplementation of the spec formulas, then against the analytic
Gaussian conditional used by RectPosterior (the "collapse consistency"
check that catches sigma/lambda convention mismatches), then end to end
against RectPosterior's collapsed sampler and against emcee. Skipped
entirely when JAX is not installed.
"""

import numpy as np
import pytest
import scipy.linalg
import scipy.stats

from geodef import backend, bayes
from geodef.data import GNSS
from geodef.fault import Fault
from geodef.invert import LinearSystem

jax = pytest.importorskip("jax")


@pytest.fixture(autouse=True)
def jax_backend():
    """Run every test in this module on the JAX backend."""
    backend.set_backend("jax")
    yield
    backend.set_backend("numpy")


_REF_LAT, _REF_LON = -2.0, 100.0
_TRUE = {
    "depth": 25e3,
    "strike": 315.0,
    "dip": 15.0,
    "length": 120e3,
    "width": 60e3,
}
_NL, _NW = 3, 3


@pytest.fixture(scope="module")
def fault3x3():
    """A 3x3-patch planar fault at a fixed, known geometry."""
    backend.set_backend("numpy")
    fault = Fault.planar(
        lat=_REF_LAT,
        lon=_REF_LON,
        depth=_TRUE["depth"],
        strike=_TRUE["strike"],
        dip=_TRUE["dip"],
        length=_TRUE["length"],
        width=_TRUE["width"],
        n_length=_NL,
        n_width=_NW,
    )
    backend.set_backend("jax")
    return fault


@pytest.fixture(scope="module")
def gnss_signed(fault3x3):
    """GNSS velocities from a dip-slip field with a genuine negative patch.

    The bump is positive at the center; one corner, spatially isolated
    from the bump (different along-strike column *and* down-dip row),
    is negative, so the unconstrained (wls) solution is expected to go
    negative there and only there — the scenario positivity is meant
    to fix, without confounding it with the resolution trade-offs that
    a *correlated* negative patch would introduce.
    """
    backend.set_backend("numpy")
    fault = fault3x3
    n_patches = fault.n_patches
    i = np.arange(n_patches) % _NL
    j = np.arange(n_patches) // _NL
    bump = 3.0 * np.exp(-(((i - 1.0) / 1.0) ** 2 + (j - 1.0) ** 2))
    dip_lobe = -1.5 * np.exp(-(((i - 2.0) / 0.6) ** 2 + ((j - 0.0) / 0.6) ** 2))
    slip_ds = bump + dip_lobe

    glon, glat = np.meshgrid(np.linspace(99.3, 100.7, 7), np.linspace(-2.7, -1.3, 7))
    glon, glat = glon.ravel(), glat.ravel()
    ue, un, uz = fault.displacement(glat, glon, np.zeros(n_patches), slip_ds)
    rng = np.random.default_rng(7)
    sigma = 0.001
    n = len(glat)
    data = GNSS(
        glon,
        glat,
        ve=ue + rng.normal(0, sigma, n),
        vn=un + rng.normal(0, sigma, n),
        vu=uz + rng.normal(0, sigma, n),
        se=np.full(n, sigma),
        sn=np.full(n, sigma),
        su=np.full(n, sigma),
    )
    backend.set_backend("jax")
    return data


@pytest.fixture(scope="module")
def fault1x2():
    """A tiny 1x2-patch planar fault for the emcee cross-check."""
    backend.set_backend("numpy")
    fault = Fault.planar(
        lat=_REF_LAT,
        lon=_REF_LON,
        depth=10e3,
        strike=0.0,
        dip=40.0,
        length=20e3,
        width=10e3,
        n_length=2,
        n_width=1,
    )
    backend.set_backend("jax")
    return fault


@pytest.fixture(scope="module")
def gnss_1x2_signed(fault1x2):
    """GNSS data from the 1x2 fault, dip-slip with one negative patch."""
    backend.set_backend("numpy")
    fault = fault1x2
    slip_ds = np.array([2.0, -0.8])
    glon, glat = np.meshgrid(
        np.linspace(_REF_LON - 0.5, _REF_LON + 0.5, 4),
        np.linspace(_REF_LAT - 0.5, _REF_LAT + 0.5, 4),
    )
    glon, glat = glon.ravel(), glat.ravel()
    ue, un, uz = fault.displacement(glat, glon, np.zeros(2), slip_ds)
    rng = np.random.default_rng(13)
    sigma = 0.003
    n = len(glat)
    data = GNSS(
        glon,
        glat,
        ve=ue + rng.normal(0, sigma, n),
        vn=un + rng.normal(0, sigma, n),
        vu=uz + rng.normal(0, sigma, n),
        se=np.full(n, sigma),
        sn=np.full(n, sigma),
        su=np.full(n, sigma),
    )
    backend.set_backend("jax")
    return data


def _posterior(fault, datasets, **overrides):
    kwargs = dict(
        fault=fault,
        datasets=datasets,
        components="dip",
        mode="hierarchical",
        smoothing="laplacian",
    )
    kwargs.update(overrides)
    return bayes.SlipPosterior(**kwargs)


# ======================================================================
# Independent NumPy reference implementation of the spec formulas
# ======================================================================


def _softplus_np(v):
    return np.log1p(np.exp(-np.abs(v))) + np.maximum(v, 0.0)


def _log_sigmoid_np(v):
    return -np.log1p(np.exp(-v))


def _reference_logpdf(post, x):
    """Independent NumPy evaluation of SlipPosterior.logpdf(x)."""
    n_z = post._n_slip
    z = np.asarray(x[:n_z], dtype=float)
    log10_sigma = float(x[n_z])
    sigma2 = 10.0 ** (2.0 * log10_sigma)
    if post._lambda_fixed is not None:
        lam = float(post._lambda_fixed)
        log10_lambda = None
    else:
        log10_lambda = float(x[n_z + 1])
        lam = 10.0**log10_lambda

    L0 = post._L0
    v = post._mu0 + post._sigma_ref * scipy.linalg.solve_triangular(
        L0.T, z, lower=False
    )
    mask = post._mask
    m = np.where(mask, _softplus_np(v), v)
    logJ = np.sum(np.where(mask, _log_sigmoid_np(v), 0.0)) + post._logJ_affine

    lo_s, hi_s = post._log10_sigma_prior
    lp = -np.log(hi_s - lo_s) if lo_s <= log10_sigma <= hi_s else -np.inf
    if log10_lambda is not None:
        lo_l, hi_l = post._log10_lambda_prior
        lp += -np.log(hi_l - lo_l) if lo_l <= log10_lambda <= hi_l else -np.inf

    p = n_z
    K = post._K
    quad = m @ K @ m
    log_prior_m = (
        -0.5 * p * np.log(2.0 * np.pi * sigma2)
        + 0.5 * (post._logdet_rank * np.log(lam) + post._logdet_sum)
        - lam * quad / (2.0 * sigma2)
    )
    log_prior = lp + log_prior_m + logJ

    r = post._d_w - post._G_w @ m
    n = post.n_data
    log_lik = -0.5 * n * np.log(2.0 * np.pi * sigma2) - (r @ r) / (2.0 * sigma2)
    return log_prior + log_lik


# ======================================================================
# Construction and validation
# ======================================================================


class TestConstruction:
    def test_param_names_hierarchical_dip(self, fault3x3, gnss_signed):
        post = _posterior(fault3x3, gnss_signed)
        n_slip = _NL * _NW
        assert post.param_names == [f"z{i}" for i in range(n_slip)] + [
            "log10_sigma",
            "log10_lambda",
        ]
        assert post.x0.shape == (n_slip + 2,)
        np.testing.assert_allclose(post.x0[:n_slip], 0.0)
        assert np.all(np.isfinite(post.x0))
        assert post.n_params == n_slip + 2

    def test_param_names_fixed(self, fault3x3, gnss_signed):
        post = _posterior(fault3x3, gnss_signed, mode="fixed", smoothing_strength=1.0)
        n_slip = _NL * _NW
        assert post.param_names == [f"z{i}" for i in range(n_slip)] + ["log10_sigma"]

    def test_param_names_weak(self, fault3x3, gnss_signed):
        post = _posterior(
            fault3x3,
            gnss_signed,
            mode="weak",
            smoothing=None,
            slip_scale=5.0,
        )
        n_slip = _NL * _NW
        assert post.param_names == [f"z{i}" for i in range(n_slip)] + ["log10_sigma"]

    def test_param_names_both_components(self, fault3x3, gnss_signed):
        post = _posterior(fault3x3, gnss_signed, components="both")
        n_slip = 2 * _NL * _NW
        assert len(post.param_names) == n_slip + 2
        assert post._n_slip == n_slip

    def test_x0_in_bounds(self, fault3x3, gnss_signed):
        post = _posterior(fault3x3, gnss_signed)
        assert np.all(post.x0 >= post._lo)
        assert np.all(post.x0 <= post._hi)

    def test_lo_hi_z_are_infinite(self, fault3x3, gnss_signed):
        post = _posterior(fault3x3, gnss_signed)
        n_slip = _NL * _NW
        assert np.all(np.isneginf(post._lo[:n_slip]))
        assert np.all(np.isposinf(post._hi[:n_slip]))
        assert np.isfinite(post._lo[n_slip])
        assert np.isfinite(post._hi[n_slip])

    def test_positive_none_mask_all_false(self, fault3x3, gnss_signed):
        post = _posterior(fault3x3, gnss_signed, positive=None)
        assert not np.any(post._mask)

    def test_positive_dip_with_components_dip(self, fault3x3, gnss_signed):
        post = _posterior(fault3x3, gnss_signed, components="dip", positive="dip")
        assert np.all(post._mask)

    def test_positive_both_valid_for_dip_components(self, fault3x3, gnss_signed):
        post = _posterior(fault3x3, gnss_signed, components="dip", positive="both")
        assert np.all(post._mask)

    def test_positive_strike_with_both_components(self, fault3x3, gnss_signed):
        post = _posterior(fault3x3, gnss_signed, components="both", positive="strike")
        n_patches = _NL * _NW
        assert np.all(post._mask[:n_patches])
        assert not np.any(post._mask[n_patches:])

    def test_positive_dip_with_both_components(self, fault3x3, gnss_signed):
        post = _posterior(fault3x3, gnss_signed, components="both", positive="dip")
        n_patches = _NL * _NW
        assert not np.any(post._mask[:n_patches])
        assert np.all(post._mask[n_patches:])

    def test_positive_bool_array(self, fault3x3, gnss_signed):
        n_slip = _NL * _NW
        mask = np.zeros(n_slip, dtype=bool)
        mask[0] = True
        post = _posterior(fault3x3, gnss_signed, positive=mask)
        np.testing.assert_array_equal(post._mask, mask)

    def test_positive_wrong_length_array_raises(self, fault3x3, gnss_signed):
        with pytest.raises(ValueError, match="positive"):
            _posterior(fault3x3, gnss_signed, positive=np.zeros(2, dtype=bool))

    def test_positive_strike_with_dip_components_raises(self, fault3x3, gnss_signed):
        with pytest.raises(ValueError, match="strike"):
            _posterior(fault3x3, gnss_signed, components="dip", positive="strike")

    def test_positive_dip_with_strike_components_raises(self, fault3x3, gnss_signed):
        with pytest.raises(ValueError, match="dip"):
            _posterior(fault3x3, gnss_signed, components="strike", positive="dip")

    def test_positive_unknown_string_raises(self, fault3x3, gnss_signed):
        with pytest.raises(ValueError, match="positive"):
            _posterior(fault3x3, gnss_signed, positive="bogus")

    def test_bad_mode_raises(self, fault3x3, gnss_signed):
        with pytest.raises(ValueError, match="mode"):
            _posterior(fault3x3, gnss_signed, mode="profiled")

    def test_weak_requires_slip_scale(self, fault3x3, gnss_signed):
        with pytest.raises(ValueError, match="slip_scale"):
            _posterior(fault3x3, gnss_signed, mode="weak", smoothing=None)

    def test_weak_requires_no_smoothing(self, fault3x3, gnss_signed):
        with pytest.raises(ValueError, match="smoothing"):
            _posterior(fault3x3, gnss_signed, mode="weak", slip_scale=5.0)

    def test_hierarchical_requires_smoothing(self, fault3x3, gnss_signed):
        with pytest.raises(ValueError, match="smoothing"):
            _posterior(fault3x3, gnss_signed, smoothing=None)

    def test_fixed_requires_smoothing_strength(self, fault3x3, gnss_signed):
        with pytest.raises(ValueError, match="smoothing_strength"):
            _posterior(fault3x3, gnss_signed, mode="fixed")

    def test_fixed_requires_smoothing(self, fault3x3, gnss_signed):
        with pytest.raises(ValueError, match="smoothing"):
            _posterior(
                fault3x3,
                gnss_signed,
                mode="fixed",
                smoothing=None,
                smoothing_strength=1.0,
            )

    def test_requires_jax_backend(self, fault3x3, gnss_signed):
        backend.set_backend("numpy")
        with pytest.raises(RuntimeError, match="JAX backend"):
            _posterior(fault3x3, gnss_signed)


# ======================================================================
# Density identity vs an independent NumPy reference
# ======================================================================


class TestDensityIdentity:
    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(
                mode="fixed",
                positive="dip",
                components="both",
                smoothing_strength=2.0,
            ),
            dict(mode="hierarchical", positive=None, components="dip"),
        ],
    )
    def test_logpdf_matches_numpy_reference(self, fault3x3, gnss_signed, kwargs):
        post = _posterior(fault3x3, gnss_signed, **kwargs)
        rng = np.random.default_rng(0)
        for _ in range(3):
            z = rng.normal(size=post._n_slip) * 0.3
            log10_sigma = rng.uniform(*post._log10_sigma_prior) * 0.3
            x = np.concatenate([z, [log10_sigma]])
            if post._log10_lambda_prior is not None:
                log10_lambda = rng.uniform(*post._log10_lambda_prior) * 0.1
                x = np.concatenate([x, [log10_lambda]])

            got = float(post.logpdf(x))
            ref = _reference_logpdf(post, x)
            np.testing.assert_allclose(got, ref, rtol=1e-10)


# ======================================================================
# Collapse consistency: joint density = collapsed density + Gaussian
# conditional correction. Validates the sigma/lambda power convention.
# ======================================================================


def _rect_style_collapsed_logpdf(post, log10_sigma, log10_lambda):
    """RectPosterior's collapsed log_prior + log_likelihood formula.

    Reimplemented directly from ``post``'s own ``_G_w``/``_d_w``/``_K``
    (rather than instantiating an actual ``RectPosterior``) so the
    comparison isolates the sigma/lambda power convention from an
    unrelated discrepancy: ``RectPosterior`` assembles G with the
    JAX-autodiff-friendly flat-Cartesian ``rect_greens``, while
    ``SlipPosterior`` uses the full geodetic ``greens()`` pipeline via
    ``LinearSystem`` — the two differ at the ~1e-4 relative level (test
    ``test_bayes.py`` sidesteps the same discrepancy by injecting a
    shared G into a ``LinearSystem`` before comparing to ABIC), enough
    to blow a 1e-8 identity even though both correctly implement the
    same formula on their own G.
    """
    sigma2 = 10.0 ** (2.0 * log10_sigma)
    lam = 10.0**log10_lambda
    G_w, d_w, K = post._G_w, post._d_w, post._K
    n = post.n_data

    H = G_w.T @ G_w + lam * K
    m_hat = np.linalg.solve(H, G_w.T @ d_w)
    r = d_w - G_w @ m_hat
    s_val = r @ r + lam * (m_hat @ K @ m_hat)
    _, logdet_H = np.linalg.slogdet(H)

    log_lik = -0.5 * n * np.log(2.0 * np.pi * sigma2) - s_val / (2.0 * sigma2)
    log_lik += 0.5 * (post._logdet_rank * np.log(lam) + post._logdet_sum - logdet_H)

    lo_s, hi_s = post._log10_sigma_prior
    lo_l, hi_l = post._log10_lambda_prior
    log_prior = -np.log(hi_s - lo_s) - np.log(hi_l - lo_l)
    return log_prior + log_lik, H, m_hat


class TestCollapseConsistency:
    def test_joint_equals_collapsed_times_conditional(self, fault3x3, gnss_signed):
        post = _posterior(
            fault3x3,
            gnss_signed,
            positive=None,
            mode="hierarchical",
            components="dip",
        )

        rng = np.random.default_rng(1)
        n_z = post._n_slip
        for _ in range(4):
            z = rng.normal(size=n_z) * 0.5
            log10_sigma = rng.uniform(*post._log10_sigma_prior) * 0.2
            log10_lambda = rng.uniform(*post._log10_lambda_prior) * 0.2
            x = np.concatenate([z, [log10_sigma], [log10_lambda]])

            joint = float(post.logpdf(x))
            coll, H, m_hat = _rect_style_collapsed_logpdf(
                post, log10_sigma, log10_lambda
            )

            # Conditional Gaussian m | hyp, d ~ N(m_hat, sigma^2 H^-1).
            sigma2 = 10.0 ** (2.0 * log10_sigma)
            L0 = post._L0
            v = post._mu0 + post._sigma_ref * scipy.linalg.solve_triangular(
                L0.T, z, lower=False
            )
            m = v  # positive=None: identity map
            logJ = post._logJ_affine

            cov = sigma2 * np.linalg.inv(H)
            log_cond = scipy.stats.multivariate_normal.logpdf(m, mean=m_hat, cov=cov)

            np.testing.assert_allclose(joint, coll + log_cond + logJ, atol=1e-8)


# ======================================================================
# Sampler agreement with the collapsed posterior
# ======================================================================


class TestSamplerAgreement:
    def test_matches_rect_posterior_hierarchical(self, fault3x3, gnss_signed):
        pytest.importorskip("blackjax")
        slip_post = _posterior(
            fault3x3,
            gnss_signed,
            positive=None,
            mode="hierarchical",
            components="dip",
        )
        rect_post = bayes.RectPosterior(
            theta0=np.array(
                [
                    0.0,
                    0.0,
                    _TRUE["depth"],
                    _TRUE["strike"],
                    _TRUE["dip"],
                    _TRUE["length"],
                    _TRUE["width"],
                ]
            ),
            datasets=gnss_signed,
            ref_lat=_REF_LAT,
            ref_lon=_REF_LON,
            free=[],
            theta_prior=None,
            n_length=_NL,
            n_width=_NW,
            components="dip",
            mode="hierarchical",
            smoothing="laplacian",
        )

        slip_result = bayes.sample(
            slip_post, n_samples=800, n_warmup=800, n_chains=2, seed=0
        )
        rect_result = bayes.sample(
            rect_post, n_samples=800, n_warmup=800, n_chains=2, seed=1
        )

        for i, name in enumerate(["log10_sigma", "log10_lambda"]):
            a = slip_result.flat[:, slip_post._n_slip + i]
            b = rect_result.flat[:, i]
            pooled_sd = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
            assert abs(a.mean() - b.mean()) < 0.25 * pooled_sd, name
            ratio = a.std(ddof=1) / b.std(ddof=1)
            assert 0.6 < ratio < 1.6, name

        slip_draws = slip_post.slip_draws(slip_result.flat)
        rect_draws = rect_post.slip_draws(rect_result.flat, seed=2)
        mean_a = slip_draws.mean(axis=0)
        mean_b = rect_draws.mean(axis=0)
        pooled_sd = np.sqrt(
            0.5 * (slip_draws.var(axis=0, ddof=1) + rect_draws.var(axis=0, ddof=1))
        )
        assert np.all(np.abs(mean_a - mean_b) < 0.3 * pooled_sd)


# ======================================================================
# Positivity end to end
# ======================================================================


class TestPositivity:
    def test_positive_dip_matches_bounded_inversion(self, fault3x3, gnss_signed):
        pytest.importorskip("blackjax")
        sys = LinearSystem(
            fault3x3, [gnss_signed], smoothing="laplacian", components="dip"
        )
        lam = 1.0

        wls = sys.invert(smoothing_strength=lam, method="wls")
        assert np.any(wls.slip_vector < 0), "fixture must produce a negative wls slip"

        bounded = sys.invert(smoothing_strength=lam, bounds=(0, None))

        post = _posterior(
            fault3x3,
            gnss_signed,
            mode="fixed",
            positive="dip",
            components="dip",
            smoothing_strength=lam,
        )
        result = bayes.sample(post, n_samples=800, n_warmup=800, n_chains=2, seed=2)
        draws = post.slip_draws(result.flat)

        assert np.all(draws >= 0.0)

        mean = draws.mean(axis=0)
        sd = draws.std(axis=0, ddof=1)
        tol = np.maximum(0.35 * np.max(np.abs(bounded.slip_vector)) * 0.15, 3 * sd)
        assert np.all(np.abs(mean - bounded.slip_vector) <= tol)

        rms_bounded = np.sqrt(np.mean((mean - bounded.slip_vector) ** 2))
        rms_wls = np.sqrt(np.mean((mean - wls.slip_vector) ** 2))
        assert rms_bounded < rms_wls


# ======================================================================
# emcee cross-check with positivity
# ======================================================================


class TestEmceeCrossCheck:
    def test_emcee_agrees_with_nuts(self, fault1x2, gnss_1x2_signed):
        pytest.importorskip("blackjax")
        emcee = pytest.importorskip("emcee")

        post = bayes.SlipPosterior(
            fault=fault1x2,
            datasets=gnss_1x2_signed,
            components="dip",
            mode="fixed",
            smoothing="damping",  # laplacian needs >= 3 patches per dimension
            smoothing_strength=1.0,
            positive="dip",
        )
        assert post.n_params == 3

        result = bayes.sample(post, n_samples=800, n_warmup=800, n_chains=2, seed=0)
        nuts_mean = result.flat.mean(axis=0)
        nuts_sd = result.flat.std(axis=0, ddof=1)

        logpdf = jax.jit(post.logpdf)
        ndim = post.n_params
        nwalkers = 16
        rng = np.random.default_rng(3)
        p0 = nuts_mean + 3.0 * nuts_sd * rng.standard_normal((nwalkers, ndim))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lambda x: float(logpdf(x)))
        sampler.run_mcmc(p0, 800, progress=False)
        chain = sampler.get_chain(discard=300, flat=True)

        for i in range(ndim):
            assert abs(chain[:, i].mean() - nuts_mean[i]) < 0.35 * nuts_sd[i]
            ratio = chain[:, i].std(ddof=1) / nuts_sd[i]
            assert 0.7 < ratio < 1.4


# ======================================================================
# Gradients
# ======================================================================


class TestGradients:
    def test_grad_matches_finite_differences(self, fault3x3, gnss_signed):
        post = _posterior(fault3x3, gnss_signed, positive="dip")
        rng = np.random.default_rng(4)
        n_z = post._n_slip
        x = np.concatenate([rng.normal(size=n_z) * 0.2, [0.1], [0.3]])
        g = backend.to_numpy(jax.grad(post.logpdf)(x))
        assert np.all(np.isfinite(g))

        fd = np.zeros_like(x)
        for i in range(len(x)):
            h = 1e-6 * max(abs(x[i]), 1e-3)
            xp_, xm = x.copy(), x.copy()
            xp_[i] += h
            xm[i] -= h
            fd[i] = (float(post.logpdf(xp_)) - float(post.logpdf(xm))) / (2 * h)
        np.testing.assert_allclose(g, fd, rtol=5e-4, atol=1e-8)


# ======================================================================
# API smoke tests
# ======================================================================


class TestAPISmoke:
    def test_slip_draws_shape_and_row_matches_slip_of(self, fault3x3, gnss_signed):
        post = _posterior(fault3x3, gnss_signed)
        rng = np.random.default_rng(5)
        n_z = post._n_slip
        samples = np.stack(
            [
                np.concatenate([rng.normal(size=n_z) * 0.1, [0.0], [0.5]])
                for _ in range(6)
            ]
        )
        draws = post.slip_draws(samples)
        assert draws.shape == (6, n_z)
        np.testing.assert_allclose(draws[0], post.slip_of(samples[0]), rtol=1e-6)

    def test_predict_shape(self, fault3x3, gnss_signed):
        post = _posterior(fault3x3, gnss_signed)
        rng = np.random.default_rng(6)
        n_z = post._n_slip
        samples = np.stack(
            [
                np.concatenate([rng.normal(size=n_z) * 0.1, [0.0], [0.5]])
                for _ in range(4)
            ]
        )
        pred = post.predict(samples)
        assert pred.shape == (4, post.n_data)

    def test_logpdf_jits(self, fault3x3, gnss_signed):
        post = _posterior(fault3x3, gnss_signed)
        rng = np.random.default_rng(7)
        n_z = post._n_slip
        x = np.concatenate([rng.normal(size=n_z) * 0.1, [0.0], [0.5]])
        jitted = jax.jit(post.logpdf)
        np.testing.assert_allclose(float(jitted(x)), float(post.logpdf(x)), rtol=1e-12)


# ======================================================================
# RectPosterior positivity path: joint geometry + slip sampling
# ======================================================================
#
# The same whitened-softplus positivity machinery, but with free
# geometry: slip rejoins the sampled state as a whitened block appended
# after the hyperparameters, and G(theta) is re-assembled per evaluation
# through a single custom_jvp so the Okada kernel is traced seven times
# per gradient regardless of slip size.

_THETA_TRUE = np.array(
    [
        0.0,
        0.0,
        _TRUE["depth"],
        _TRUE["strike"],
        _TRUE["dip"],
        _TRUE["length"],
        _TRUE["width"],
    ]
)


@pytest.fixture(scope="module")
def gnss_positive(fault3x3):
    """GNSS data from an all-positive dip-slip bump (positivity well specified).

    Distinct from ``gnss_signed``: here the truth respects the
    constraint, so recovering geometry *and* non-negative slip is the
    correct answer and dip should come back near its true value.
    """
    backend.set_backend("numpy")
    fault = fault3x3
    n_patches = fault.n_patches
    i = np.arange(n_patches) % _NL
    j = np.arange(n_patches) // _NL
    slip_ds = 3.0 * np.exp(-(((i - 1.0) / 1.0) ** 2 + (j - 1.0) ** 2))
    glon, glat = np.meshgrid(np.linspace(99.3, 100.7, 7), np.linspace(-2.7, -1.3, 7))
    glon, glat = glon.ravel(), glat.ravel()
    ue, un, uz = fault.displacement(glat, glon, np.zeros(n_patches), slip_ds)
    rng = np.random.default_rng(11)
    sigma = 0.001
    n = len(glat)
    data = GNSS(
        glon,
        glat,
        ve=ue + rng.normal(0, sigma, n),
        vn=un + rng.normal(0, sigma, n),
        vu=uz + rng.normal(0, sigma, n),
        se=np.full(n, sigma),
        sn=np.full(n, sigma),
        su=np.full(n, sigma),
    )
    backend.set_backend("jax")
    return data


def _rect(datasets, **overrides):
    kwargs = dict(
        theta0=_THETA_TRUE,
        datasets=datasets,
        ref_lat=_REF_LAT,
        ref_lon=_REF_LON,
        free=["dip"],
        theta_prior={"dip": (5.0, 45.0)},
        n_length=_NL,
        n_width=_NW,
        components="dip",
        mode="profiled",
        smoothing="laplacian",
        smoothing_strength=1.0,
    )
    kwargs.update(overrides)
    return bayes.RectPosterior(**kwargs)


class TestRectPositiveConstruction:
    def test_layout_appends_z_after_hypers(self, gnss_signed):
        post = _rect(gnss_signed, positive="dip")
        n_slip = _NL * _NW
        assert post._joint
        assert post._n_hyper == 2  # dip, log10_sigma (profiled: no lambda)
        assert post.param_names == ["dip", "log10_sigma"] + [
            f"z{i}" for i in range(n_slip)
        ]
        assert post.n_params == 2 + n_slip
        assert post.x0.shape == (2 + n_slip,)
        assert np.all(np.isneginf(post._lo[2:]))
        assert np.all(np.isposinf(post._hi[2:]))

    def test_hierarchical_layout_has_lambda(self, gnss_signed):
        post = _rect(
            gnss_signed, positive="dip", mode="hierarchical", smoothing_strength=1.0
        )
        n_slip = _NL * _NW
        assert post._n_hyper == 3
        assert post.param_names[:3] == ["dip", "log10_sigma", "log10_lambda"]
        assert post.n_params == 3 + n_slip

    def test_positive_none_stays_collapsed(self, gnss_signed):
        post = _rect(gnss_signed, positive=None)
        assert not post._joint
        assert post.param_names == ["dip", "log10_sigma"]
        assert post.n_params == 2


class TestRectPositiveDensity:
    def test_joint_equals_collapsed_times_conditional(self, gnss_signed):
        """With an all-False mask (identity map) and fixed geometry, the
        joint density must equal the collapsed density plus the exact
        Gaussian-conditional correction — both on the same rect_greens G,
        so the tolerance is tight."""
        from geodef.gradients import rect_greens

        common = dict(
            theta0=_THETA_TRUE,
            datasets=gnss_signed,
            ref_lat=_REF_LAT,
            ref_lon=_REF_LON,
            free=[],
            theta_prior=None,
            n_length=_NL,
            n_width=_NW,
            components="dip",
            mode="hierarchical",
            smoothing="laplacian",
        )
        coll = bayes.RectPosterior(**common)
        n_slip = _NL * _NW
        joint = bayes.RectPosterior(**common, positive=np.zeros(n_slip, dtype=bool))

        theta0 = np.asarray(_THETA_TRUE, dtype=float)
        G3 = backend.to_numpy(
            rect_greens(
                backend.xp.asarray(theta0),
                joint._e_obs,
                joint._n_obs,
                _NL,
                _NW,
                0.25,
            )
        )
        G_w = (joint._W_half_P @ G3)[:, joint._col_start : joint._col_stop]

        rng = np.random.default_rng(2)
        for _ in range(4):
            ls, ll = rng.uniform(-1, 1), rng.uniform(-2, 2)
            z = rng.normal(size=n_slip) * 0.5
            c = float(
                coll.log_prior(np.array([ls, ll]))
                + coll.log_likelihood(np.array([ls, ll]))
            )
            joint_val = float(joint.logpdf(np.concatenate([[ls, ll], z])))

            sigma2, lam = 10.0 ** (2 * ls), 10.0**ll
            v = joint._mu0 + scipy.linalg.solve_triangular(joint._L0.T, z, lower=False)
            H = G_w.T @ G_w + lam * joint._LtL
            m_hat = np.linalg.solve(H, G_w.T @ joint._d_w)
            log_cond = scipy.stats.multivariate_normal.logpdf(
                v, mean=m_hat, cov=sigma2 * np.linalg.inv(H)
            )
            np.testing.assert_allclose(
                joint_val, c + log_cond + joint._logJ_affine, atol=1e-6
            )

    def test_grad_matches_finite_differences(self, gnss_signed):
        post = _rect(gnss_signed, positive="dip")
        rng = np.random.default_rng(4)
        x = post.x0.copy()
        x[0] = 16.0  # dip
        x[post._n_hyper :] = 0.2 * rng.normal(size=post._n_slip)
        g = backend.to_numpy(jax.grad(post.logpdf)(x))
        assert np.all(np.isfinite(g))
        fd = np.zeros_like(x)
        for i in range(len(x)):
            h = 1e-5 * max(abs(x[i]), 1e-3)
            xp_, xm = x.copy(), x.copy()
            xp_[i] += h
            xm[i] -= h
            fd[i] = (float(post.logpdf(xp_)) - float(post.logpdf(xm))) / (2 * h)
        np.testing.assert_allclose(g, fd, rtol=1e-3, atol=1e-6)

    def test_slip_mode_respects_positivity(self, gnss_signed):
        post = _rect(gnss_signed, positive="dip")
        x = post.x0.copy()
        x[post._n_hyper :] = -0.5  # push whitened coords negative
        assert np.all(post.slip_mode(x) >= 0.0)


class TestRectPositiveSampling:
    def test_recovers_dip_and_positive_slip(self, gnss_positive):
        pytest.importorskip("blackjax")
        post = _rect(gnss_positive, free=["dip"], positive="dip")
        result = bayes.sample(post, n_samples=800, n_warmup=800, n_chains=2, seed=0)
        dip = result.flat[:, 0]
        assert abs(dip.mean() - _TRUE["dip"]) < 4.0
        draws = post.slip_draws(result.flat)
        assert np.all(draws >= 0.0)
        assert draws.shape == (result.flat.shape[0], _NL * _NW)

    def test_emcee_agrees_with_nuts(self, fault1x2, gnss_1x2_signed):
        pytest.importorskip("blackjax")
        emcee = pytest.importorskip("emcee")
        theta1x2 = np.array([0.0, 0.0, 10e3, 0.0, 40.0, 20e3, 10e3])
        post = bayes.RectPosterior(
            theta0=theta1x2,
            datasets=gnss_1x2_signed,
            ref_lat=_REF_LAT,
            ref_lon=_REF_LON,
            free=["dip"],
            theta_prior={"dip": (20.0, 60.0)},
            n_length=2,
            n_width=1,
            components="dip",
            mode="profiled",
            smoothing="damping",
            smoothing_strength=1.0,
            positive="dip",
        )
        assert post.n_params == 4  # dip, log10_sigma, z0, z1

        result = bayes.sample(post, n_samples=800, n_warmup=800, n_chains=2, seed=0)
        nuts_mean = result.flat.mean(axis=0)
        nuts_sd = result.flat.std(axis=0, ddof=1)

        logpdf = jax.jit(post.logpdf)
        ndim, nwalkers = post.n_params, 16
        rng = np.random.default_rng(3)
        p0 = nuts_mean + 2.0 * nuts_sd * rng.standard_normal((nwalkers, ndim))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lambda x: float(logpdf(x)))
        sampler.run_mcmc(p0, 1000, progress=False)
        chain = sampler.get_chain(discard=400, flat=True)
        for i in range(ndim):
            assert abs(chain[:, i].mean() - nuts_mean[i]) < 0.4 * nuts_sd[i]
            assert 0.6 < chain[:, i].std(ddof=1) / nuts_sd[i] < 1.5


class TestRectPositiveAPI:
    def test_slip_draws_deterministic(self, gnss_signed):
        post = _rect(gnss_signed, positive="dip")
        rng = np.random.default_rng(5)
        samples = np.stack(
            [
                np.concatenate([[15.0, 0.0], rng.normal(size=post._n_slip) * 0.2])
                for _ in range(4)
            ]
        )
        a = post.slip_draws(samples, seed=1)
        b = post.slip_draws(samples, seed=999)
        np.testing.assert_allclose(a, b)  # seed ignored on the joint path
        np.testing.assert_allclose(a[0], post.slip_mode(samples[0]), rtol=1e-6)

    def test_predict_shape(self, gnss_signed):
        post = _rect(gnss_signed, positive="dip")
        samples = np.stack([post.x0, post.x0])
        assert post.predict(samples).shape == (2, post.n_data)
