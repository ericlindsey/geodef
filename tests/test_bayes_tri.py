"""Tests for geodef.bayes.TriWarp and TriPosterior — triangular-mesh geometry.

TriWarp's linear normal-offset parameterization is validated against its
own closed-form properties (interpolation, linearity, watertightness).
TriPosterior's collapsed density is validated against LinearSystem.G_w
(the obs-frame anchor), an independent NumPy reimplementation of the
collapsed formula, and finite-difference gradients. Skipped entirely when
JAX is not installed.
"""

import numpy as np
import pytest

from geodef import backend, bayes, gradients
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


_REF_LAT, _REF_LON = -1.5, 99.0


def _tri_fault(
    nx: int,
    ny: int,
    ref_lat: float,
    ref_lon: float,
    length: float = 20e3,
    width: float = 10e3,
    depth0: float = 5e3,
    dip_grad: float = 0.3,
) -> Fault:
    """A structured nx-by-ny node grid over a gently dipping plane.

    Each grid cell becomes two triangles. Depths follow
    ``depth0 + dip_grad * north_offset``, matching the shape used to
    validate the Laplacian-smoothing path (>= 7 triangles).
    """
    u = np.linspace(-length / 2, length / 2, nx)
    v = np.linspace(-width / 2, width / 2, ny)
    uu, vv = np.meshgrid(u, v)
    e = uu.ravel()
    n = vv.ravel()
    z = -(depth0 + dip_grad * n)
    nodes = np.column_stack([e, n, z])

    ii, jj = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1))
    i00 = (jj * nx + ii).ravel()
    i10 = (jj * nx + ii + 1).ravel()
    i01 = ((jj + 1) * nx + ii).ravel()
    i11 = ((jj + 1) * nx + ii + 1).ravel()
    triangles = np.concatenate(
        [np.column_stack([i00, i10, i11]), np.column_stack([i00, i11, i01])]
    )
    return Fault.from_triangles(
        nodes, ref_lat=ref_lat, ref_lon=ref_lon, triangles=triangles
    )


def _gnss_from_fault(fault, slip_ds, sigma, seed, n_lon=6, n_lat=5, span=(0.15, 0.1)):
    """Synthetic GNSS velocities from a known fault and dip-slip field."""
    glon, glat = np.meshgrid(
        np.linspace(_REF_LON - span[0], _REF_LON + span[0], n_lon),
        np.linspace(_REF_LAT - span[1], _REF_LAT + span[1], n_lat),
    )
    glon, glat = glon.ravel(), glat.ravel()
    ue, un, uz = fault.displacement(glat, glon, np.zeros(fault.n_patches), slip_ds)
    rng = np.random.default_rng(seed)
    n = len(glat)
    return GNSS(
        lon=glon,
        lat=glat,
        ve=ue + rng.normal(0, sigma, n),
        vn=un + rng.normal(0, sigma, n),
        vu=uz + rng.normal(0, sigma, n),
        se=np.full(n, sigma),
        sn=np.full(n, sigma),
        su=np.full(n, sigma),
    )


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture(scope="module")
def small_mesh_fault():
    """A 3x2-node grid (4 triangles) — kept small for TriWarp-only checks."""
    backend.set_backend("numpy")
    fault = _tri_fault(
        3, 2, _REF_LAT, _REF_LON, length=6e3, width=3e3, depth0=2e3, dip_grad=0.1
    )
    backend.set_backend("jax")
    return fault


@pytest.fixture(scope="module")
def small_warp(small_mesh_fault):
    backend.set_backend("numpy")
    warp = bayes.TriWarp(small_mesh_fault, n_knots=(3, 2))
    backend.set_backend("jax")
    return warp


@pytest.fixture(scope="module")
def mesh_fault():
    """A 4x3-node grid (12 triangles) — the standard TriPosterior fixture.

    Large enough for the knn Laplacian (k=6, requires >= 7 patches).
    """
    backend.set_backend("numpy")
    fault = _tri_fault(4, 3, _REF_LAT, _REF_LON)
    backend.set_backend("jax")
    return fault


@pytest.fixture(scope="module")
def warp4(mesh_fault):
    backend.set_backend("numpy")
    warp = bayes.TriWarp(mesh_fault, n_knots=(2, 2))
    backend.set_backend("jax")
    return warp


@pytest.fixture(scope="module")
def gnss_tri(warp4):
    """GNSS data from a known warped geometry and a smooth dip-slip bump."""
    backend.set_backend("numpy")
    true_theta = np.array([600.0, -400.0, 300.0, -200.0])
    true_fault = warp4.fault(true_theta)
    centers = true_fault.centers_local
    slip_ds = 2.0 * np.exp(-((centers[:, 1] / 4e3) ** 2))
    data = _gnss_from_fault(true_fault, slip_ds, sigma=5e-4, seed=3)
    backend.set_backend("jax")
    return data


@pytest.fixture(scope="module")
def tri_post_hier(warp4, gnss_tri):
    # Module-scoped fixtures are set up before the function-scoped
    # ``jax_backend`` autouse fixture, so the backend must be forced here
    # rather than relied on from that fixture's ordering.
    backend.set_backend("jax")
    return bayes.TriPosterior(
        warp4,
        gnss_tri,
        knot_prior=(-20000.0, 20000.0),
        components="dip",
        mode="hierarchical",
        smoothing="laplacian",
        smoothing_strength=1.0,
    )


@pytest.fixture(scope="module")
def tri_post_profiled(warp4, gnss_tri):
    backend.set_backend("jax")
    return bayes.TriPosterior(
        warp4,
        gnss_tri,
        knot_prior=(-20000.0, 20000.0),
        components="dip",
        mode="profiled",
        smoothing="laplacian",
        smoothing_strength=1.0,
    )


# ======================================================================
# Part B — TriWarp invariants
# ======================================================================


class TestTriWarp:
    def test_preserves_fault_frame(self, small_mesh_fault):
        warp = bayes.TriWarp(small_mesh_fault, n_knots=(2, 2))

        assert warp.frame is small_mesh_fault.frame
        trial = warp.fault(np.zeros(warp.n_knots))
        assert trial.frame is small_mesh_fault.frame

    def test_n_knots_and_shapes(self, small_warp):
        assert small_warp.n_knots == 6
        assert small_warp.knots_uv.shape == (6, 2)
        assert small_warp.knots_xyz.shape == (6, 3)
        assert small_warp.normal.shape == (3,)
        assert small_warp.length_scale > 0.0

    def test_vertices_zero_theta_matches_reference(self, small_warp, small_mesh_fault):
        v = backend.to_numpy(small_warp.vertices(np.zeros(small_warp.n_knots)))
        np.testing.assert_array_equal(v, small_mesh_fault.vertices)

    def test_linearity(self, small_warp):
        rng = np.random.default_rng(0)
        nk = small_warp.n_knots
        a = rng.normal(scale=50.0, size=nk)
        b = rng.normal(scale=50.0, size=nk)
        v0 = backend.to_numpy(small_warp.vertices(np.zeros(nk)))
        va = backend.to_numpy(small_warp.vertices(a)) - v0
        vb = backend.to_numpy(small_warp.vertices(b)) - v0
        vab = backend.to_numpy(small_warp.vertices(a + b)) - v0
        np.testing.assert_allclose(vab, va + vb, atol=1e-8)

    def test_watertight_shared_corners(self, small_warp):
        rng = np.random.default_rng(1)
        theta = rng.normal(scale=100.0, size=small_warp.n_knots)
        off = backend.to_numpy(small_warp.offsets(theta))
        flat_v0 = small_warp._v0_flat
        _, inverse = np.unique(np.round(flat_v0, 6), axis=0, return_inverse=True)
        for group in np.unique(inverse):
            idx = np.where(inverse == group)[0]
            np.testing.assert_allclose(off[idx], off[idx[0]], atol=1e-9)

    def test_interpolation_recovers_theta_at_knots(self, small_warp):
        """phi(knots, knots) @ inv(A) @ theta ~= theta (ridge-limited)."""
        rng = np.random.default_rng(2)
        theta = rng.normal(scale=50.0, size=small_warp.n_knots)
        length_scale = small_warp.length_scale
        diff = small_warp.knots_uv[:, None, :] - small_warp.knots_uv[None, :, :]
        phi_kk = np.exp(-np.sum(diff**2, axis=-1) / (2.0 * length_scale**2))
        a_mat = phi_kk + 1e-8 * np.eye(small_warp.n_knots)
        recovered = phi_kk @ np.linalg.solve(a_mat, theta)
        np.testing.assert_allclose(recovered, theta, atol=1e-4 * np.max(np.abs(theta)))

    def test_check_flags_above_surface(self, small_warp):
        assert small_warp.check(np.zeros(small_warp.n_knots)) is True
        assert small_warp.check(np.full(small_warp.n_knots, 5000.0)) is False

    def test_jit_matches_numpy_backend(self, small_warp):
        theta = np.linspace(10.0, -10.0, small_warp.n_knots)
        backend.set_backend("numpy")
        ref = np.asarray(small_warp.vertices(theta))
        backend.set_backend("jax")
        jitted = jax.jit(small_warp.vertices)
        got = backend.to_numpy(jitted(theta))
        np.testing.assert_allclose(got, ref, rtol=1e-10)

    def test_fault_builds_valid_tri_fault(self, small_warp):
        f = small_warp.fault(np.zeros(small_warp.n_knots))
        assert f.engine == "tri"
        assert f.n_patches == small_warp._n_tri

    def test_plot_smoke(self, small_warp):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = small_warp.plot(theta=np.full(small_warp.n_knots, 50.0))
        assert fig is not None
        assert ax is not None
        plt.close(fig)

        fig2, ax2 = small_warp.plot()
        assert fig2 is not None
        plt.close(fig2)


class TestTriWarpConstruction:
    def test_non_tri_fault_raises(self):
        rect = Fault.planar(
            lat=_REF_LAT,
            lon=_REF_LON,
            depth=10e3,
            strike=0.0,
            dip=30.0,
            length=20e3,
            width=10e3,
        )
        with pytest.raises(ValueError, match="triangular"):
            bayes.TriWarp(rect)

    def test_explicit_knots(self, small_mesh_fault):
        knots = np.array([[-1000.0, -500.0], [0.0, 0.0], [1000.0, 500.0]])
        warp = bayes.TriWarp(small_mesh_fault, knots=knots)
        assert warp.n_knots == 3
        np.testing.assert_allclose(warp.knots_uv, knots)


# ======================================================================
# Part C — TriPosterior
# ======================================================================


class TestFrameAnchor:
    def test_exposes_warp_frame(self, tri_post_hier, warp4):
        assert tri_post_hier.frame is warp4.frame

    def test_matches_linear_system_g_w_at_theta_zero(
        self, tri_post_hier, mesh_fault, gnss_tri
    ):
        sys = LinearSystem(
            mesh_fault, [gnss_tri], smoothing="laplacian", components="dip"
        )
        x = np.zeros(tri_post_hier.n_params)
        g_w = backend.to_numpy(tri_post_hier._assemble(x)[2])
        # The JAX (vmapped, where-selected) and NumPy (gathered) tri
        # kernel code paths accumulate rounding differences up to ~1e-10
        # relative even at identical float64 inputs; rtol=1e-9 keeps this
        # a tight frame-anchor check without chasing kernel-internal noise.
        np.testing.assert_allclose(g_w, sys.G_w, rtol=1e-9)


def _reference_logpdf(post, x):
    """Independent NumPy evaluation of TriPosterior.logpdf(x)."""
    nk = post.warp.n_knots
    theta = np.asarray(x[:nk], dtype=float)
    log10_sigma = float(x[nk])
    sigma2 = 10.0 ** (2.0 * log10_sigma)
    if post._lambda_fixed is not None:
        lam = float(post._lambda_fixed)
    else:
        lam = 10.0 ** float(x[nk + 1])

    backend.set_backend("numpy")
    v = np.asarray(post.warp.vertices(theta), dtype=float)
    v_clipped = v.copy()
    v_clipped[..., 2] = np.minimum(v[..., 2], 0.0)
    g3 = gradients.tri_greens(v_clipped, post._obs, post._nu)
    backend.set_backend("jax")

    g_w = (post._W_half_P @ g3)[:, post._col_start : post._col_stop]
    ltl = post._LtL
    h = g_w.T @ g_w + lam * ltl
    m_hat = np.linalg.solve(h, g_w.T @ post._d_w)
    r = post._d_w - g_w @ m_hat
    s_val = r @ r + lam * (m_hat @ ltl @ m_hat)
    n = post.n_data
    log_lik = -0.5 * n * np.log(2.0 * np.pi * sigma2) - s_val / (2.0 * sigma2)
    if post._include_logdet:
        _, logdet_h = np.linalg.slogdet(h)
        logdet_prior = post._logdet_rank * np.log(lam) + post._logdet_sum
        log_lik += 0.5 * (logdet_prior - logdet_h)

    log_prior = 0.0
    for k in range(len(x)):
        lo, hi = post._lo[k], post._hi[k]
        if post._is_uniform[k]:
            log_prior += -np.log(hi - lo) if lo <= x[k] <= hi else -np.inf
        else:
            zk = (x[k] - post._mu[k]) / post._sd[k]
            log_prior += -0.5 * zk**2 - np.log(post._sd[k]) - 0.5 * np.log(2.0 * np.pi)

    max_z = float(np.max(v[..., 2]))
    guard = -np.inf if max_z > 0.0 else 0.0
    return log_prior + guard + log_lik


class TestDensityIdentity:
    def test_logpdf_matches_numpy_reference_hierarchical(self, tri_post_hier):
        post = tri_post_hier
        nk = post.warp.n_knots
        rng = np.random.default_rng(0)
        for _ in range(3):
            x = np.concatenate(
                [rng.uniform(-200, 200, size=nk), [rng.uniform(-0.3, 0.3)]]
            )
            if post._lambda_fixed is None:
                x = np.concatenate([x, [rng.uniform(-1.0, 1.0)]])
            got = float(post.logpdf(x))
            ref = _reference_logpdf(post, x)
            np.testing.assert_allclose(got, ref, rtol=1e-9)

    def test_logpdf_matches_numpy_reference_profiled(self, tri_post_profiled):
        post = tri_post_profiled
        nk = post.warp.n_knots
        rng = np.random.default_rng(1)
        for _ in range(3):
            x = np.concatenate(
                [rng.uniform(-200, 200, size=nk), [rng.uniform(-0.3, 0.3)]]
            )
            got = float(post.logpdf(x))
            ref = _reference_logpdf(post, x)
            np.testing.assert_allclose(got, ref, rtol=1e-9)


class TestGradients:
    @pytest.mark.slow
    def test_grad_matches_finite_differences(self, tri_post_hier):
        post = tri_post_hier
        nk = post.warp.n_knots
        rng = np.random.default_rng(4)
        x = post.x0.copy()
        x[:nk] += rng.uniform(-50.0, 50.0, size=nk)
        x[nk] = 0.1
        x[nk + 1] = 0.2

        g = backend.to_numpy(jax.grad(post.logpdf)(x))
        assert np.all(np.isfinite(g))

        fd = np.zeros_like(x)
        for i in range(len(x)):
            h = 5.0 if i < nk else 1e-4
            xp_, xm = x.copy(), x.copy()
            xp_[i] += h
            xm[i] -= h
            fd[i] = (float(post.logpdf(xp_)) - float(post.logpdf(xm))) / (2 * h)
        np.testing.assert_allclose(g, fd, rtol=1e-3, atol=1e-6)


class TestHalfSpaceGuard:
    def test_logpdf_is_neg_inf_above_surface(self, tri_post_hier):
        nk = tri_post_hier.warp.n_knots
        x = tri_post_hier.x0.copy()
        x[:nk] = 15000.0
        assert tri_post_hier.warp.check(x[:nk]) is False
        assert float(tri_post_hier.logpdf(x)) == -np.inf

    def test_gradient_finite_near_boundary(self, tri_post_hier):
        nk = tri_post_hier.warp.n_knots
        x = tri_post_hier.x0.copy()
        x[:nk] = 500.0
        assert tri_post_hier.warp.check(x[:nk]) is True
        g = backend.to_numpy(jax.grad(tri_post_hier.logpdf)(x))
        assert np.all(np.isfinite(g))


class TestConstruction:
    def test_param_names_hierarchical(self, warp4, gnss_tri):
        post = bayes.TriPosterior(
            warp4,
            gnss_tri,
            knot_prior=(-500.0, 500.0),
            components="dip",
            mode="hierarchical",
            smoothing="laplacian",
        )
        nk = warp4.n_knots
        assert post.param_names == [f"knot{i}" for i in range(nk)] + [
            "log10_sigma",
            "log10_lambda",
        ]
        assert post.n_params == nk + 2
        assert post.x0.shape == (nk + 2,)
        np.testing.assert_allclose(post.x0[:nk], 0.0)

    def test_param_names_weak(self, warp4, gnss_tri):
        post = bayes.TriPosterior(
            warp4,
            gnss_tri,
            knot_prior=(-500.0, 500.0),
            components="dip",
            mode="weak",
            smoothing=None,
            slip_scale=5.0,
        )
        nk = warp4.n_knots
        assert post.param_names == [f"knot{i}" for i in range(nk)] + ["log10_sigma"]

    def test_param_names_profiled(self, warp4, gnss_tri):
        post = bayes.TriPosterior(
            warp4,
            gnss_tri,
            knot_prior=(-500.0, 500.0),
            components="dip",
            mode="profiled",
            smoothing="laplacian",
            smoothing_strength=1.0,
        )
        nk = warp4.n_knots
        assert post.param_names == [f"knot{i}" for i in range(nk)] + ["log10_sigma"]

    def test_knot_prior_single_spec_broadcasts(self, warp4, gnss_tri):
        post = bayes.TriPosterior(
            warp4,
            gnss_tri,
            knot_prior=("normal", 0.0, 300.0),
            components="dip",
            mode="profiled",
            smoothing="laplacian",
            smoothing_strength=1.0,
        )
        nk = warp4.n_knots
        assert np.all(~post._is_uniform[:nk])
        np.testing.assert_allclose(post._sd[:nk], 300.0)

    def test_knot_prior_per_knot_list(self, warp4, gnss_tri):
        nk = warp4.n_knots
        specs = [
            (-100.0, 100.0),
            (-200.0, 200.0),
            ("normal", 0.0, 50.0),
            (-300.0, 300.0),
        ]
        assert len(specs) == nk
        post = bayes.TriPosterior(
            warp4,
            gnss_tri,
            knot_prior=specs,
            components="dip",
            mode="profiled",
            smoothing="laplacian",
            smoothing_strength=1.0,
        )
        np.testing.assert_allclose(post._hi[:3], [100.0, 200.0, np.inf])
        assert not post._is_uniform[2]
        assert post._is_uniform[3]

    def test_knot_prior_wrong_length_raises(self, warp4, gnss_tri):
        specs = [(-100.0, 100.0), (-200.0, 200.0)]
        with pytest.raises(ValueError, match="knot_prior"):
            bayes.TriPosterior(
                warp4,
                gnss_tri,
                knot_prior=specs,
                components="dip",
                mode="profiled",
                smoothing="laplacian",
                smoothing_strength=1.0,
            )

    def test_knots0_sets_starting_point(self, warp4, gnss_tri):
        nk = warp4.n_knots
        k0 = np.linspace(-200.0, 400.0, nk)
        post = bayes.TriPosterior(
            warp4,
            gnss_tri,
            knot_prior=(-500.0, 500.0),
            knots0=k0,
            components="dip",
            mode="profiled",
            smoothing="laplacian",
            smoothing_strength=1.0,
        )
        np.testing.assert_allclose(post.x0[:nk], k0)

    def test_knots0_clipped_into_bounds(self, warp4, gnss_tri):
        nk = warp4.n_knots
        post = bayes.TriPosterior(
            warp4,
            gnss_tri,
            knot_prior=(-500.0, 500.0),
            knots0=np.full(nk, 900.0),
            components="dip",
            mode="profiled",
            smoothing="laplacian",
            smoothing_strength=1.0,
        )
        np.testing.assert_allclose(post.x0[:nk], 500.0)

    def test_knots0_wrong_shape_raises(self, warp4, gnss_tri):
        with pytest.raises(ValueError, match="knots0"):
            bayes.TriPosterior(
                warp4,
                gnss_tri,
                knot_prior=(-500.0, 500.0),
                knots0=np.zeros(warp4.n_knots + 1),
                components="dip",
                mode="profiled",
                smoothing="laplacian",
                smoothing_strength=1.0,
            )

    def test_requires_jax_backend(self, warp4, gnss_tri):
        backend.set_backend("numpy")
        with pytest.raises(RuntimeError, match="JAX backend"):
            bayes.TriPosterior(
                warp4,
                gnss_tri,
                knot_prior=(-500.0, 500.0),
                components="dip",
                mode="profiled",
                smoothing="laplacian",
                smoothing_strength=1.0,
            )

    def test_bad_mode_raises(self, warp4, gnss_tri):
        with pytest.raises(ValueError, match="mode"):
            bayes.TriPosterior(
                warp4, gnss_tri, knot_prior=(-500.0, 500.0), mode="bogus"
            )

    def test_bad_components_raises(self, warp4, gnss_tri):
        with pytest.raises(ValueError, match="components"):
            bayes.TriPosterior(
                warp4,
                gnss_tri,
                knot_prior=(-500.0, 500.0),
                components="rake",
                mode="profiled",
                smoothing="laplacian",
                smoothing_strength=1.0,
            )

    def test_weak_requires_slip_scale(self, warp4, gnss_tri):
        with pytest.raises(ValueError, match="slip_scale"):
            bayes.TriPosterior(
                warp4,
                gnss_tri,
                knot_prior=(-500.0, 500.0),
                mode="weak",
                smoothing=None,
            )

    def test_hierarchical_requires_smoothing(self, warp4, gnss_tri):
        with pytest.raises(ValueError, match="smoothing"):
            bayes.TriPosterior(
                warp4,
                gnss_tri,
                knot_prior=(-500.0, 500.0),
                mode="hierarchical",
                smoothing=None,
            )

    def test_profiled_requires_smoothing_strength(self, warp4, gnss_tri):
        with pytest.raises(ValueError, match="smoothing_strength"):
            bayes.TriPosterior(
                warp4,
                gnss_tri,
                knot_prior=(-500.0, 500.0),
                mode="profiled",
                smoothing="laplacian",
            )


# ======================================================================
# End-to-end recovery
# ======================================================================


@pytest.mark.slow
class TestRecovery:
    def test_recovers_known_knot_offsets(self, mesh_fault):
        """Posterior over knot offsets covers the truth, starting near it.

        As with the RectPosterior sampling tests (which start at the true
        geometry), warmup begins near the mode via ``knots0``: with tight
        data, a start far from the mode makes the posterior so stiff in
        ``log10_sigma`` that step-size adaptation collapses — a warmup
        pathology, not a property of the posterior (whose surface is
        smooth and peaks at the truth; see the density/gradient tests).
        The explicit ``length_scale`` keeps the two knots' vertex
        footprints distinguishable (the (2,1)-grid default spans the
        whole mesh, making them ~98% colinear).
        """
        pytest.importorskip("blackjax")
        backend.set_backend("numpy")
        warp2 = bayes.TriWarp(mesh_fault, n_knots=(2, 1), length_scale=10e3)
        true_theta = np.array([800.0, -500.0])
        true_fault = warp2.fault(true_theta)
        centers = true_fault.centers_local
        slip_ds = 2.5 * np.exp(-((centers[:, 1] / 4e3) ** 2))
        data = _gnss_from_fault(true_fault, slip_ds, sigma=3e-4, seed=9)
        backend.set_backend("jax")

        post = bayes.TriPosterior(
            warp2,
            data,
            knot_prior=(-3000.0, 3000.0),
            knots0=np.array([600.0, -300.0]),
            components="dip",
            mode="hierarchical",
            smoothing="laplacian",
            smoothing_strength=1.0,
        )
        result = bayes.sample(post, n_samples=500, n_warmup=500, n_chains=2, seed=0)

        assert np.all(result.rhat[:2] < 1.1)
        # The 0.3 mm data noise makes the posterior so tight (sd of a few
        # meters) that the noise realization's shift of the likelihood
        # peak can exceed 3 sd; the absolute floor covers that ordinary
        # frequentist offset (observed ~12 m on the 800 m knot) without
        # weakening the recovery claim on the 20 km mesh.
        for i, truth in enumerate(true_theta):
            mean = result.flat[:, i].mean()
            sd = result.flat[:, i].std(ddof=1)
            assert abs(mean - truth) < max(3.0 * sd, 30.0)
