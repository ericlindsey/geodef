"""Tests for geodef.bayes — collapsed Bayesian posteriors and sampling.

The marginal likelihood is validated exactly against a dense
multivariate-normal density (matrix-determinant-lemma form) and against
the ABIC machinery in geodef.invert; gradients are validated against
finite differences. Skipped entirely when JAX is not installed.
"""

import numpy as np
import pytest
import scipy.stats

from geodef import backend, bayes, gradients
from geodef.data import GNSS
from geodef.fault import Fault
from geodef.invert import LinearSystem, _projection_matrix

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
def gnss_data():
    """GNSS velocities from a smooth dip-slip distribution, tiny noise."""
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
    n_patches = fault.n_patches
    i = np.arange(n_patches) % _NL
    j = np.arange(n_patches) // _NL
    bump = np.exp(-(((i - 1.0) / 1.0) ** 2 + (j - 1.0) ** 2))
    slip_ds = 3.0 * bump

    glon, glat = np.meshgrid(np.linspace(99.0, 101.0, 5), np.linspace(-3.0, -1.0, 5))
    glon, glat = glon.ravel(), glat.ravel()
    ue, un, uz = fault.displacement(glat, glon, np.zeros(n_patches), slip_ds)
    rng = np.random.default_rng(7)
    sigma = 0.002
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


def _posterior(gnss_data, **overrides):
    kwargs = dict(
        theta0=_THETA_TRUE,
        datasets=gnss_data,
        ref_lat=_REF_LAT,
        ref_lon=_REF_LON,
        free=["dip", "depth"],
        theta_prior={"dip": (5.0, 45.0), "depth": (10e3, 60e3)},
        n_length=_NL,
        n_width=_NW,
        components="dip",
        mode="hierarchical",
        smoothing="laplacian",
        smoothing_strength=1.0,
    )
    kwargs.update(overrides)
    return bayes.RectPosterior(**kwargs)


# ======================================================================
# Construction and parameter layout
# ======================================================================


class TestConstruction:
    def test_param_names_hierarchical(self, gnss_data):
        post = _posterior(gnss_data)
        assert post.param_names == ["dip", "depth", "log10_sigma", "log10_lambda"]
        assert post.x0.shape == (4,)
        np.testing.assert_allclose(post.x0[:2], [_TRUE["dip"], _TRUE["depth"]])
        np.testing.assert_allclose(post.x0[3], 0.0)  # log10(1.0)

    def test_param_names_weak(self, gnss_data):
        post = _posterior(
            gnss_data,
            mode="weak",
            smoothing=None,
            smoothing_strength=None,
            slip_scale=10.0,
        )
        assert post.param_names == ["dip", "depth", "log10_sigma"]

    def test_param_names_profiled(self, gnss_data):
        post = _posterior(gnss_data, mode="profiled")
        assert post.param_names == ["dip", "depth", "log10_sigma"]

    def test_free_can_be_empty(self, gnss_data):
        post = _posterior(gnss_data, free=[], theta_prior=None)
        assert post.param_names == ["log10_sigma", "log10_lambda"]

    def test_requires_jax_backend(self, gnss_data):
        backend.set_backend("numpy")
        with pytest.raises(RuntimeError, match="JAX backend"):
            _posterior(gnss_data)

    def test_unknown_free_raises(self, gnss_data):
        with pytest.raises(ValueError, match="rake"):
            _posterior(gnss_data, free=["rake"], theta_prior={"rake": (0, 1)})

    def test_missing_prior_raises(self, gnss_data):
        with pytest.raises(ValueError, match="theta_prior"):
            _posterior(gnss_data, theta_prior={"dip": (5.0, 45.0)})

    def test_bad_mode_raises(self, gnss_data):
        with pytest.raises(ValueError, match="mode"):
            _posterior(gnss_data, mode="full")

    def test_weak_requires_slip_scale(self, gnss_data):
        with pytest.raises(ValueError, match="slip_scale"):
            _posterior(gnss_data, mode="weak", smoothing=None, smoothing_strength=None)

    def test_hierarchical_requires_smoothing(self, gnss_data):
        with pytest.raises(ValueError, match="smoothing"):
            _posterior(gnss_data, smoothing=None)


# ======================================================================
# Exact marginal-likelihood values
# ======================================================================


def _dense_reference_loglik(post, theta, lam, sigma):
    """Dense multivariate-normal log-density of the weighted data.

    Independent evaluation of the collapsed likelihood: the weighted
    data are Gaussian with covariance sigma^2 (I + G_w (lam LtL)^-1
    G_w^T), computable directly when LtL has full rank.
    """
    G3 = backend.to_numpy(
        gradients.rect_greens(
            theta, post._e_obs, post._n_obs, post._n_length, post._n_width
        )
    )
    G_w = post._W_half_P @ G3
    G_w = G_w[:, post._col_start : post._col_stop]
    cov = sigma**2 * (
        np.eye(len(post._d_w)) + G_w @ np.linalg.solve(lam * post._LtL, G_w.T)
    )
    return scipy.stats.multivariate_normal.logpdf(post._d_w, mean=None, cov=cov)


class TestMarginalLikelihood:
    def test_weak_mode_matches_dense_gaussian(self, gnss_data):
        """mode='weak' (L = I, full-rank prior) admits an exact dense
        reference density via the matrix determinant lemma."""
        post = _posterior(
            gnss_data,
            mode="weak",
            smoothing=None,
            smoothing_strength=None,
            slip_scale=5.0,
        )
        for dip, depth, log10_sigma in [
            (15.0, 25e3, 0.0),
            (20.0, 30e3, 0.3),
            (10.0, 20e3, -0.2),
        ]:
            x = np.array([dip, depth, log10_sigma])
            ll = float(post.log_likelihood(x))
            theta = _THETA_TRUE.copy()
            theta[4], theta[2] = dip, depth
            ref = _dense_reference_loglik(post, theta, 1.0 / 5.0**2, 10.0**log10_sigma)
            np.testing.assert_allclose(ll, ref, rtol=1e-9)

    def test_hierarchical_matches_dense_gaussian_with_damping(self, gnss_data):
        """Hierarchical mode with a full-rank damping operator also
        admits the dense reference; exercises the sampled-lambda path."""
        post = _posterior(gnss_data, smoothing="damping")
        for log10_lam in (-1.0, 0.0, 2.0):
            x = np.array([15.0, 25e3, 0.1, log10_lam])
            ll = float(post.log_likelihood(x))
            ref = _dense_reference_loglik(post, _THETA_TRUE, 10.0**log10_lam, 10.0**0.1)
            np.testing.assert_allclose(ll, ref, rtol=1e-9)

    def test_hierarchical_matches_abic(self, gnss_data):
        """Maximizing the marginal over sigma recovers ABIC up to a
        lambda-independent constant (Fukuda & Johnson lineage)."""
        post = _posterior(gnss_data, free=[], theta_prior=None)

        template = Fault.planar(
            lat=_REF_LAT,
            lon=_REF_LON,
            depth=_THETA_TRUE[2],
            strike=_THETA_TRUE[3],
            dip=_THETA_TRUE[4],
            length=_THETA_TRUE[5],
            width=_THETA_TRUE[6],
            n_length=_NL,
            n_width=_NW,
        )
        sys = LinearSystem(
            template, [gnss_data], smoothing="laplacian", components="dip"
        )
        # Inject the flat-Cartesian G used by the posterior so the two
        # pipelines see identical linear systems.
        G3 = backend.to_numpy(
            gradients.rect_greens(_THETA_TRUE, post._e_obs, post._n_obs, _NL, _NW)
        )
        P = _projection_matrix([gnss_data])
        sys.G = (P @ G3)[:, post._col_start : post._col_stop]
        sys.G_w = post._W_half_P @ G3
        sys.G_w = sys.G_w[:, post._col_start : post._col_stop]

        def neg2_ll_profiled_sigma(log10_lam):
            """-2 max_sigma log-marginal at a given lambda."""
            n = post.n_data
            s_val = float(post._misfit_total(np.array([0.0, log10_lam])))
            sigma2_hat = s_val / n

            x = np.array([0.5 * np.log10(sigma2_hat), log10_lam])
            return -2.0 * float(post.log_likelihood(x))

        vals = []
        for log10_lam in (-1.0, 1.0):
            abic = sys._abic_value(10.0**log10_lam)[0]
            vals.append(neg2_ll_profiled_sigma(log10_lam) - abic)
        np.testing.assert_allclose(vals[0], vals[1], rtol=1e-8)

    def test_profiled_mode_matches_manual_solve(self, gnss_data):
        """Profiled mode: -N/2 log(2 pi sigma^2) - S/(2 sigma^2) with S
        from the regularized normal equations, no Occam terms."""
        post = _posterior(gnss_data, mode="profiled", smoothing_strength=2.0)
        x = np.array([17.0, 27e3, 0.2])
        ll = float(post.log_likelihood(x))

        theta = _THETA_TRUE.copy()
        theta[4], theta[2] = 17.0, 27e3
        G3 = backend.to_numpy(
            gradients.rect_greens(theta, post._e_obs, post._n_obs, _NL, _NW)
        )
        G_w = (post._W_half_P @ G3)[:, post._col_start : post._col_stop]
        H = G_w.T @ G_w + 2.0 * post._LtL
        m = np.linalg.solve(H, G_w.T @ post._d_w)
        r = post._d_w - G_w @ m
        s_val = float(r @ r + 2.0 * m @ post._LtL @ m)
        sigma2 = 10.0 ** (2 * 0.2)
        n = post.n_data
        ref = -0.5 * n * np.log(2 * np.pi * sigma2) - s_val / (2 * sigma2)
        np.testing.assert_allclose(ll, ref, rtol=1e-10)


# ======================================================================
# Priors and the posterior density
# ======================================================================


class TestLogPrior:
    def test_uniform_prior_bounds(self, gnss_data):
        post = _posterior(gnss_data)
        x_in = np.array([15.0, 25e3, 0.0, 0.0])
        x_out = np.array([50.0, 25e3, 0.0, 0.0])  # dip above upper bound
        assert np.isfinite(float(post.logpdf(x_in)))
        assert float(post.logpdf(x_out)) == -np.inf

    def test_gradient_finite_outside_bounds(self, gnss_data):
        """The clip guard must keep gradients NaN-free even at rejected
        points, so the NUTS integrator can recover."""
        post = _posterior(gnss_data)
        g = jax.grad(post.logpdf)(np.array([50.0, 25e3, 0.0, 0.0]))
        assert not np.any(np.isnan(backend.to_numpy(g)))

    def test_normal_prior_quadratic(self, gnss_data):
        post = _posterior(
            gnss_data,
            theta_prior={"dip": ("normal", 15.0, 5.0), "depth": (10e3, 60e3)},
        )
        x1 = np.array([15.0, 25e3, 0.0, 0.0])
        x2 = np.array([20.0, 25e3, 0.0, 0.0])
        lp1 = float(post.log_prior(x1))
        lp2 = float(post.log_prior(x2))
        np.testing.assert_allclose(lp1 - lp2, 0.5 * (5.0 / 5.0) ** 2)

    def test_logpdf_is_prior_plus_likelihood(self, gnss_data):
        post = _posterior(gnss_data)
        x = np.array([16.0, 26e3, 0.1, 0.5])
        np.testing.assert_allclose(
            float(post.logpdf(x)),
            float(post.log_prior(x)) + float(post.log_likelihood(x)),
            rtol=1e-12,
        )


class TestDiagnostics:
    def test_rhat_near_one_for_iid_chains(self):
        rng = np.random.default_rng(0)
        chains = rng.normal(size=(4, 1000))
        assert abs(bayes.split_rhat(chains) - 1.0) < 0.02

    def test_rhat_detects_disagreeing_chains(self):
        rng = np.random.default_rng(1)
        chains = rng.normal(size=(4, 1000))
        chains[0] += 5.0
        assert bayes.split_rhat(chains) > 1.5

    def test_ess_iid_close_to_total(self):
        rng = np.random.default_rng(2)
        chains = rng.normal(size=(4, 1000))
        ess = bayes.effective_sample_size(chains)
        assert 0.5 * 4000 < ess < 1.5 * 4000

    def test_ess_reduced_for_correlated_chains(self):
        rng = np.random.default_rng(3)
        phi = 0.9
        n = 2000
        chains = np.zeros((4, n))
        for c in range(4):
            e = rng.normal(size=n)
            for t in range(1, n):
                chains[c, t] = phi * chains[c, t - 1] + e[t]
        ess = bayes.effective_sample_size(chains)
        # true tau = (1+phi)/(1-phi) = 19 -> ess ~ 8000/19 ~ 420
        assert ess < 0.25 * 4 * n

    def test_constant_chain_guards(self):
        chains = np.ones((2, 100))
        assert bayes.split_rhat(chains) == 1.0
        assert bayes.effective_sample_size(chains) == 200.0


@pytest.fixture(scope="module")
def gnss_single_patch():
    """GNSS data from a single-patch fault with known dip of 40 deg."""
    backend.set_backend("numpy")
    fault = Fault.planar(
        lat=_REF_LAT,
        lon=_REF_LON,
        depth=10e3,
        strike=0.0,
        dip=40.0,
        length=20e3,
        width=10e3,
        n_length=1,
        n_width=1,
    )
    glon, glat = np.meshgrid(
        np.linspace(_REF_LON - 0.7, _REF_LON + 0.7, 5),
        np.linspace(_REF_LAT - 0.7, _REF_LAT + 0.7, 5),
    )
    glon, glat = glon.ravel(), glat.ravel()
    ue, un, uz = fault.displacement(glat, glon, np.array([0.0]), np.array([2.0]))
    rng = np.random.default_rng(11)
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


@pytest.fixture(scope="module")
def single_patch_result(gnss_single_patch):
    """One NUTS run on the single-patch problem, shared across tests."""
    pytest.importorskip("blackjax")
    backend.set_backend("jax")
    post = bayes.RectPosterior(
        theta0=np.array([0.0, 0.0, 10e3, 0.0, 30.0, 20e3, 10e3]),
        datasets=gnss_single_patch,
        ref_lat=_REF_LAT,
        ref_lon=_REF_LON,
        free=["dip"],
        theta_prior={"dip": (15.0, 70.0)},
        mode="weak",
        smoothing=None,
        slip_scale=5.0,
    )
    result = bayes.sample(post, n_samples=400, n_warmup=400, n_chains=2, seed=0)
    return post, result


@pytest.mark.slow
class TestSample:
    def test_recovers_single_patch_dip(self, single_patch_result):
        _, result = single_patch_result
        assert result.samples.shape == (2, 400, 2)
        assert result.param_names == ["dip", "log10_sigma"]
        dip_mean = result.flat[:, 0].mean()
        assert abs(dip_mean - 40.0) < 3.0
        # noise scale factor should be near 1 (data sigma matches)
        assert abs(result.flat[:, 1].mean()) < 0.3
        assert np.all(result.rhat < 1.1)
        assert np.all(result.ess > 40)
        assert result.n_divergent < 0.1 * 2 * 400
        assert 0.5 < result.acceptance_rate <= 1.0
        assert np.all(np.isfinite(result.log_prob))

    def test_hierarchical_smoke(self, gnss_data):
        pytest.importorskip("blackjax")
        post = _posterior(gnss_data, free=["dip"], theta_prior={"dip": (5.0, 45.0)})
        result = bayes.sample(post, n_samples=150, n_warmup=200, n_chains=2, seed=1)
        assert result.samples.shape == (2, 150, 3)
        assert result.param_names == ["dip", "log10_sigma", "log10_lambda"]
        assert np.all(np.isfinite(result.samples))
        assert result.rhat.shape == (3,)
        assert result.ess.shape == (3,)

    def test_summary_and_plot(self, single_patch_result):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _, result = single_patch_result
        stats = result.summary()
        for key in ("mean", "sd", "q05", "q50", "q95", "rhat", "ess"):
            assert key in stats
            assert stats[key].shape == (2,)
        fig, axes = result.plot_pairs(truths=[40.0, 0.0])
        assert axes.shape == (2, 2)
        plt.close(fig)

    def test_emcee_agrees_with_nuts(self, single_patch_result):
        """Gradient-free ensemble MCMC on the same logpdf must find the
        same posterior — an independent check of the NUTS machinery."""
        emcee = pytest.importorskip("emcee")

        post, result = single_patch_result
        logpdf = jax.jit(post.logpdf)

        nuts_mean = result.flat.mean(axis=0)
        nuts_sd = result.flat.std(axis=0, ddof=1)

        ndim = post.n_params
        nwalkers = 16
        rng = np.random.default_rng(3)
        p0 = nuts_mean + 3.0 * nuts_sd * rng.standard_normal((nwalkers, ndim))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lambda x: float(logpdf(x)))
        sampler.run_mcmc(p0, 800, progress=False)
        chain = sampler.get_chain(discard=300, flat=True)

        for i in range(ndim):
            assert abs(chain[:, i].mean() - nuts_mean[i]) < 0.3 * nuts_sd[i]
            ratio = chain[:, i].std(ddof=1) / nuts_sd[i]
            assert 0.75 < ratio < 1.35

    def test_requires_blackjax_message(self, gnss_single_patch, monkeypatch):
        """Without blackjax the error must point at the bayes extra."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "blackjax":
                raise ImportError("No module named 'blackjax'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        post = bayes.RectPosterior(
            theta0=np.array([0.0, 0.0, 10e3, 0.0, 30.0, 20e3, 10e3]),
            datasets=gnss_single_patch,
            ref_lat=_REF_LAT,
            ref_lon=_REF_LON,
            free=["dip"],
            theta_prior={"dip": (15.0, 70.0)},
            mode="weak",
            smoothing=None,
            slip_scale=5.0,
        )
        with pytest.raises(ImportError, match="geodef\\[bayes\\]"):
            bayes.sample(post, n_samples=10, n_warmup=10, n_chains=1)


class TestConditionalSlip:
    def test_slip_mode_matches_manual_ridge_solve(self, gnss_data):
        post = _posterior(gnss_data)
        x = np.array([17.0, 27e3, 0.1, 0.5])
        m_hat = post.slip_mode(x)
        assert m_hat.shape == (_NL * _NW,)

        theta = _THETA_TRUE.copy()
        theta[4], theta[2] = 17.0, 27e3
        G3 = backend.to_numpy(
            gradients.rect_greens(theta, post._e_obs, post._n_obs, _NL, _NW)
        )
        G_w = (post._W_half_P @ G3)[:, post._col_start : post._col_stop]
        lam = 10.0**0.5
        H = G_w.T @ G_w + lam * post._LtL
        ref = np.linalg.solve(H, G_w.T @ post._d_w)
        np.testing.assert_allclose(m_hat, ref, rtol=1e-9)

    def test_slip_draws_match_conditional_gaussian(self, gnss_data):
        """Repeated draws at one fixed x must reproduce the analytic
        conditional mean and covariance sigma^2 H^-1."""
        post = _posterior(gnss_data)
        x = np.array([15.0, 25e3, 0.0, 1.0])
        n = 4000
        draws = post.slip_draws(np.tile(x, (n, 1)), seed=5)
        assert draws.shape == (n, _NL * _NW)

        m_hat = post.slip_mode(x)
        theta = _THETA_TRUE.copy()
        G3 = backend.to_numpy(
            gradients.rect_greens(theta, post._e_obs, post._n_obs, _NL, _NW)
        )
        G_w = (post._W_half_P @ G3)[:, post._col_start : post._col_stop]
        lam = 10.0**1.0
        H = G_w.T @ G_w + lam * post._LtL
        cov = np.linalg.inv(H)  # sigma = 1 at log10_sigma = 0
        sd = np.sqrt(np.diag(cov))

        err_mean = np.abs(draws.mean(axis=0) - m_hat)
        assert np.all(err_mean < 4.0 * sd / np.sqrt(n))
        np.testing.assert_allclose(draws.std(axis=0, ddof=1), sd, rtol=0.1)

    def test_predict_shape_and_mean(self, gnss_data):
        post = _posterior(gnss_data)
        x = np.array([15.0, 25e3, 0.0, 1.0])
        n = 2000
        pred = post.predict(np.tile(x, (n, 1)), seed=6)
        assert pred.shape == (n, post.n_data)

        # average prediction converges to the conditional-mode prediction
        m_hat = post.slip_mode(x)
        G3 = backend.to_numpy(
            gradients.rect_greens(_THETA_TRUE, post._e_obs, post._n_obs, _NL, _NW)
        )
        P = _projection_matrix([gnss_data])
        G_d = (P @ G3)[:, post._col_start : post._col_stop]
        d_mode = G_d @ m_hat
        resid = pred.mean(axis=0) - d_mode
        assert np.max(np.abs(resid)) < 5e-3

    def test_slip_draws_vary_with_geometry(self, gnss_data):
        """Different geometry samples must produce different conditional
        distributions (the whole point of propagating theta)."""
        post = _posterior(gnss_data)
        xs = np.array([[12.0, 20e3, 0.0, 1.0], [18.0, 30e3, 0.0, 1.0]])
        draws = post.slip_draws(xs, seed=7)
        assert draws.shape == (2, _NL * _NW)
        assert not np.allclose(draws[0], draws[1])


class TestGradients:
    def test_grad_matches_finite_differences(self, gnss_data):
        post = _posterior(gnss_data)
        x = np.array([17.0, 28e3, 0.15, 0.4])
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

    def test_logpdf_jits(self, gnss_data):
        post = _posterior(gnss_data)
        x = np.array([16.0, 26e3, 0.1, 0.5])
        jitted = jax.jit(post.logpdf)
        np.testing.assert_allclose(float(jitted(x)), float(post.logpdf(x)), rtol=1e-12)
