"""Tests for the differentiable forward models in geodef.gradients.

All Jacobians are validated against central finite differences of the
forward model. Skipped entirely when JAX is not installed.
"""

import numpy as np
import pytest

from geodef import backend, gradients

jax = pytest.importorskip("jax")


@pytest.fixture(autouse=True)
def jax_backend():
    """Run every test in this module on the JAX backend."""
    backend.set_backend("jax")
    yield
    backend.set_backend("numpy")


_THETA = np.array([500.0, -800.0, 12e3, 37.0, 55.0, 15e3, 8e3])
_SLIP = np.array([1.0, 0.6, 0.2])
_E_OBS = np.array([6e3, -5e3, 11e3, 1e3, -9e3])
_N_OBS = np.array([4e3, 9e3, -7e3, 2e3, -3e3])

_VERTICES = np.array([[0.0, 0.0, -1e3], [8e3, 2e3, -4e3], [1e3, 6e3, -6e3]])
_OBS = np.column_stack([_E_OBS, _N_OBS, np.zeros(5)])


def _fd_jacobian(func, x, rel_step=1e-6, scale=1.0):
    """Central finite-difference Jacobian of func w.r.t. array x.

    ``scale`` sets the step floor so that parameters whose value happens
    to be near zero (e.g. a vertex coordinate at the origin) still get a
    step matched to the problem's length scale rather than a roundoff-
    dominated one.
    """
    x = np.asarray(x, dtype=float)
    base = np.asarray(func(x))
    jac = np.zeros(base.shape + x.shape)
    for i in np.ndindex(x.shape):
        h = rel_step * max(abs(x[i]), scale)
        xp_ = x.copy()
        xm = x.copy()
        xp_[i] += h
        xm[i] -= h
        jac[(..., *i)] = (np.asarray(func(xp_)) - np.asarray(func(xm))) / (2 * h)
    return jac


class TestRectForward:
    def test_matches_numpy_forward(self):
        d = backend.to_numpy(gradients.rect_displacement(_THETA, _SLIP, _E_OBS, _N_OBS))
        assert d.shape == (5, 3)

        backend.set_backend("numpy")
        from geodef import okada85

        e0, n0, depth, strike, dip, length, width = _THETA
        expected = np.zeros((5, 3))
        for slip_val, rake, opening in [
            (_SLIP[0], 0.0, 0.0),
            (_SLIP[1], 90.0, 0.0),
        ]:
            ue, un, uz = okada85.displacement(
                _E_OBS - e0,
                _N_OBS - n0,
                depth,
                strike,
                dip,
                length,
                width,
                rake,
                slip_val,
                0.0,
                0.25,
            )
            expected += np.column_stack([ue, un, uz])
        ue, un, uz = okada85.displacement(
            _E_OBS - e0,
            _N_OBS - n0,
            depth,
            strike,
            dip,
            length,
            width,
            0.0,
            0.0,
            _SLIP[2],
            0.25,
        )
        expected += np.column_stack([ue, un, uz])

        np.testing.assert_allclose(d, expected, rtol=1e-10, atol=1e-15)


class TestRectJacobian:
    def test_theta_jacobian_matches_finite_differences(self):
        d, d_dtheta, d_dslip = gradients.rect_displacement_jacobian(
            _THETA, _SLIP, _E_OBS, _N_OBS
        )
        assert d.shape == (5, 3)
        assert d_dtheta.shape == (5, 3, 7)
        assert d_dslip.shape == (5, 3, 3)
        assert np.all(np.isfinite(backend.to_numpy(d_dtheta)))

        fd = _fd_jacobian(
            lambda th: backend.to_numpy(
                gradients.rect_displacement(th, _SLIP, _E_OBS, _N_OBS)
            ),
            _THETA,
        )
        np.testing.assert_allclose(
            backend.to_numpy(d_dtheta), fd, rtol=1e-5, atol=1e-12
        )

    def test_slip_jacobian_matches_finite_differences(self):
        _, _, d_dslip = gradients.rect_displacement_jacobian(
            _THETA, _SLIP, _E_OBS, _N_OBS
        )
        fd = _fd_jacobian(
            lambda m: backend.to_numpy(
                gradients.rect_displacement(_THETA, m, _E_OBS, _N_OBS)
            ),
            _SLIP,
        )
        np.testing.assert_allclose(backend.to_numpy(d_dslip), fd, rtol=1e-6, atol=1e-14)

    def test_slip_jacobian_is_linear_basis(self):
        """d is linear in slip, so ∂d/∂m must reproduce the unit responses."""
        _, _, d_dslip = gradients.rect_displacement_jacobian(
            _THETA, _SLIP, _E_OBS, _N_OBS
        )
        for i in range(3):
            unit = np.zeros(3)
            unit[i] = 1.0
            d_unit = backend.to_numpy(
                gradients.rect_displacement(_THETA, unit, _E_OBS, _N_OBS)
            )
            np.testing.assert_allclose(
                backend.to_numpy(d_dslip)[:, :, i], d_unit, rtol=1e-12, atol=1e-18
            )


class TestTriForward:
    def test_matches_numpy_forward(self):
        d = backend.to_numpy(gradients.tri_displacement(_VERTICES, _SLIP, _OBS))
        backend.set_backend("numpy")
        from geodef import tri

        expected = tri.TDdispHS(_OBS, _VERTICES, _SLIP, 0.25)
        np.testing.assert_allclose(d, expected, rtol=1e-10, atol=1e-15)


class TestTriJacobian:
    @pytest.mark.slow
    def test_vertex_jacobian_matches_finite_differences(self):
        d, d_dv, d_dslip = gradients.tri_displacement_jacobian(_VERTICES, _SLIP, _OBS)
        assert d.shape == (5, 3)
        assert d_dv.shape == (5, 3, 3, 3)
        assert d_dslip.shape == (5, 3, 3)
        assert np.all(np.isfinite(backend.to_numpy(d_dv)))

        fd = _fd_jacobian(
            lambda v: backend.to_numpy(gradients.tri_displacement(v, _SLIP, _OBS)),
            _VERTICES,
            rel_step=1e-6,
            scale=1e3,
        )
        np.testing.assert_allclose(backend.to_numpy(d_dv), fd, rtol=2e-5, atol=1e-12)

    def test_slip_jacobian_matches_finite_differences(self):
        _, _, d_dslip = gradients.tri_displacement_jacobian(_VERTICES, _SLIP, _OBS)
        fd = _fd_jacobian(
            lambda m: backend.to_numpy(gradients.tri_displacement(_VERTICES, m, _OBS)),
            _SLIP,
        )
        np.testing.assert_allclose(backend.to_numpy(d_dslip), fd, rtol=1e-6, atol=1e-14)


class TestBackendGuard:
    def test_requires_jax_backend(self):
        backend.set_backend("numpy")
        with pytest.raises(RuntimeError, match="JAX backend"):
            gradients.rect_displacement_jacobian(_THETA, _SLIP, _E_OBS, _N_OBS)


# ======================================================================
# Full Green's matrix assembly G(theta)
# ======================================================================


class TestRectGreens:
    def test_matches_independent_patch_construction(self):
        """G columns equal unit-slip responses at independently derived
        patch centers (corner + strike/dip unit-vector layout)."""
        backend.set_backend("numpy")
        from geodef import okada85

        n_length, n_width = 3, 2
        e0, n0, depth, strike, dip, length, width = _THETA
        G = gradients.rect_greens(
            _THETA, _E_OBS, _N_OBS, n_length=n_length, n_width=n_width
        )
        npatch = n_length * n_width
        assert G.shape == (3 * len(_E_OBS), 2 * npatch)

        sin_str, cos_str = np.sin(np.radians(strike)), np.cos(np.radians(strike))
        sin_dip, cos_dip = np.sin(np.radians(dip)), np.cos(np.radians(dip))
        pL, pW = length / n_length, width / n_width
        fault_e0 = -0.5 * length * sin_str - 0.5 * width * cos_dip * cos_str
        fault_n0 = -0.5 * length * cos_str + 0.5 * width * cos_dip * sin_str
        fault_u0 = -0.5 * width * sin_dip

        for jj in range(n_width):
            for ii in range(n_length):
                e_c = (
                    e0
                    + fault_e0
                    + (ii + 0.5) * pL * sin_str
                    + (jj + 0.5) * pW * cos_dip * cos_str
                )
                n_c = (
                    n0
                    + fault_n0
                    + (ii + 0.5) * pL * cos_str
                    - (jj + 0.5) * pW * cos_dip * sin_str
                )
                d_c = depth - (fault_u0 + (jj + 0.5) * pW * sin_dip)
                col = jj * n_length + ii
                for rake, block in [(0.0, 0), (90.0, 1)]:
                    ue, un, uz = okada85.displacement(
                        _E_OBS - e_c,
                        _N_OBS - n_c,
                        d_c,
                        strike,
                        dip,
                        pL,
                        pW,
                        rake,
                        1.0,
                        0.0,
                        0.25,
                    )
                    expected = np.column_stack([ue, un, uz]).ravel()
                    np.testing.assert_allclose(
                        G[:, block * npatch + col],
                        expected,
                        rtol=1e-12,
                        atol=1e-18,
                        err_msg=f"patch ({ii},{jj}) block {block}",
                    )

    def test_matches_displacement_greens_layout(self):
        """Row/column layout agrees with greens.displacement_greens on a
        small-extent fault (flat-earth vs geodetic transform differences
        stay below the loose tolerance)."""
        backend.set_backend("numpy")
        from geodef import greens
        from geodef.fault import Fault

        theta = np.array([0.0, 0.0, 4e3, 37.0, 55.0, 3e3, 2e3])
        n_length, n_width = 2, 2
        e_obs = np.array([1.5e3, -1e3, 2e3])
        n_obs = np.array([0.5e3, 2e3, -1.5e3])

        fault = Fault.planar(
            lat=0.0,
            lon=0.0,
            depth=theta[2],
            strike=theta[3],
            dip=theta[4],
            length=theta[5],
            width=theta[6],
            n_length=n_length,
            n_width=n_width,
        )
        # small-angle conversion of local ENU obs to geographic
        lat_obs = n_obs / 111194.9266
        lon_obs = e_obs / 111194.9266
        G_ref = greens.displacement_greens(
            lat_obs,
            lon_obs,
            fault._lat,
            fault._lon,
            fault._depth,
            fault.strike * np.ones(fault.n_patches),
            fault.dip * np.ones(fault.n_patches),
            fault._length,
            fault._width,
        )
        G = gradients.rect_greens(
            theta, e_obs, n_obs, n_length=n_length, n_width=n_width
        )
        # loose tolerances: the reference pipeline places patches with
        # geodetic transforms while rect_greens is flat-Cartesian; the
        # comparison verifies layout and conventions, not kernel accuracy
        np.testing.assert_allclose(G, G_ref, rtol=2e-2, atol=2e-5)

    def test_theta_jacobian_matches_finite_differences(self):
        jax_mod = jax
        G_func = lambda th: gradients.rect_greens(  # noqa: E731
            th, _E_OBS, _N_OBS, n_length=2, n_width=2
        )
        jac = jax_mod.jacfwd(G_func)(_THETA)
        assert np.all(np.isfinite(backend.to_numpy(jac)))
        fd = _fd_jacobian(lambda th: backend.to_numpy(G_func(th)), _THETA)
        np.testing.assert_allclose(backend.to_numpy(jac), fd, rtol=2e-4, atol=1e-12)


class TestTriGreens:
    _MESH = np.array(
        [
            [[0.0, 0.0, -1e3], [8e3, 2e3, -4e3], [1e3, 6e3, -6e3]],
            [[8e3, 2e3, -4e3], [9e3, 8e3, -8e3], [1e3, 6e3, -6e3]],
        ]
    )

    def test_columns_are_unit_slip_responses(self):
        backend.set_backend("numpy")
        from geodef import tri

        G = gradients.tri_greens(self._MESH, _OBS)
        ntri = 2
        assert G.shape == (3 * len(_OBS), 2 * ntri)
        for i in range(ntri):
            for slip_vec, block in [((1.0, 0.0, 0.0), 0), ((0.0, 1.0, 0.0), 1)]:
                expected = tri.TDdispHS(
                    _OBS, self._MESH[i], np.array(slip_vec), 0.25
                ).ravel()
                np.testing.assert_allclose(
                    G[:, block * ntri + i], expected, rtol=1e-12, atol=1e-18
                )

    @pytest.mark.slow
    def test_vertex_jacobian_matches_finite_differences(self):
        G_func = lambda v: gradients.tri_greens(v, _OBS)  # noqa: E731
        jac = jax.jacfwd(G_func)(self._MESH)
        assert np.all(np.isfinite(backend.to_numpy(jac)))
        fd = _fd_jacobian(
            lambda v: backend.to_numpy(G_func(v)),
            self._MESH,
            rel_step=1e-6,
            scale=1e3,
        )
        np.testing.assert_allclose(backend.to_numpy(jac), fd, rtol=2e-5, atol=1e-12)

    def test_jit_matches_eager(self):
        """The vmapped assembly is jit-compatible and unchanged by jit."""
        eager = backend.to_numpy(gradients.tri_greens(self._MESH, _OBS))
        jitted = backend.to_numpy(
            jax.jit(lambda v: gradients.tri_greens(v, _OBS))(
                backend.xp.asarray(self._MESH)
            )
        )
        np.testing.assert_allclose(jitted, eager, rtol=1e-12, atol=1e-18)

    def test_vmap_matches_numpy_loop(self):
        """The JAX vmap over the mesh axis reproduces the NumPy triangle loop."""
        jax_g = backend.to_numpy(gradients.tri_greens(self._MESH, _OBS))
        backend.set_backend("numpy")
        loop_g = backend.to_numpy(gradients.tri_greens(self._MESH, _OBS))
        backend.set_backend("jax")
        np.testing.assert_allclose(jax_g, loop_g, rtol=1e-9, atol=1e-12)

    @pytest.mark.slow
    def test_vertex_jacobian_finite_on_vertical_side(self):
        """A vertical TD side (AngSetupFSC's degenerate branch, common to
        strike-slip meshes) must not NaN-poison the vertex Jacobian.

        At an *exactly* vertical side the free-surface correction is a
        hard special case (zeroed, as in the published algorithm), so the
        map is discontinuous there and no finite-difference match is
        expected — only that the gradient stays finite (no ``0 * inf``
        poison from ``arccos`` at ``+/-1``). FD agreement in the smooth
        regime is covered by ``test_vertex_jacobian_matches_finite_differences``.
        """
        mesh = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, -5e3], [6e3, 0.0, -3e3]],  # P1P2 vertical
                [[0.0, 0.0, -5e3], [6e3, 0.0, -3e3], [6e3, 0.0, -8e3]],
            ]
        )
        obs = np.array([[3e3, 2e3, 0.0], [-2e3, -1e3, -1e3], [8e3, 4e3, -2e3]])
        jac = backend.to_numpy(jax.jacfwd(lambda v: gradients.tri_greens(v, obs))(mesh))
        assert np.all(np.isfinite(jac))
        # primal is finite too, and jit-consistent, at the degenerate config
        eager = backend.to_numpy(gradients.tri_greens(mesh, obs))
        jitted = backend.to_numpy(
            jax.jit(lambda v: gradients.tri_greens(v, obs))(backend.xp.asarray(mesh))
        )
        assert np.all(np.isfinite(eager))
        np.testing.assert_allclose(jitted, eager, rtol=1e-12, atol=1e-15)

    def test_primal_correct_near_vertical(self):
        """A steeply dipping (near-vertical) side exercises the clip/where
        path just short of the degeneracy; the JAX primal must still match
        the NumPy loop exactly. FD gradient checks are unreliable here
        (steep derivatives dominate the truncation error), so correctness
        of the path is validated on the value, and gradient *finiteness*
        by the exactly-vertical test above."""
        mesh = np.array(
            [
                [[0.0, 0.0, 0.0], [50.0, 0.0, -5e3], [6e3, 0.0, -3e3]],  # near-vertical
                [[50.0, 0.0, -5e3], [6e3, 0.0, -3e3], [6e3, 0.0, -8e3]],
            ]
        )
        obs = np.array([[3e3, 2e3, 0.0], [-2e3, -1e3, -1e3], [8e3, 4e3, -2e3]])
        jax_g = backend.to_numpy(gradients.tri_greens(mesh, obs))
        jac = backend.to_numpy(jax.jacfwd(lambda v: gradients.tri_greens(v, obs))(mesh))
        assert np.all(np.isfinite(jac))  # clip keeps arccos' gradient finite
        backend.set_backend("numpy")
        loop_g = backend.to_numpy(gradients.tri_greens(mesh, obs))
        backend.set_backend("jax")
        np.testing.assert_allclose(jax_g, loop_g, rtol=1e-9, atol=1e-12)


class TestReverseMode:
    """Reverse-mode (jax.grad) safety of the rectangular path.

    The Bayesian solver differentiates a scalar log-density with
    ``jax.grad``, so the okada85 kernels must be reverse-mode safe. The
    hazard is Okada's ``arctan(xi*eta/(q*R))`` term: observation points
    that lie exactly on the fault-plane ray (``q == 0``, easily hit by
    symmetric synthetic grids over a dipping fault) make its autodiff
    produce ``0 * inf = nan`` in both modes even though the value is
    finite.
    """

    # 41-point profile across a 45-degree dipping fault; one point lands
    # exactly on the fault-plane ray (q == 0) for this geometry.
    _THETA_DIP = np.array([0.0, 0.0, 8e3, 0.0, 45.0, 15e3, 8e3])
    _E_LINE = np.linspace(-20e3, 20e3, 41)
    _N_LINE = np.zeros(41)

    @staticmethod
    def _loss(theta, slip, e_obs, n_obs):
        import jax.numpy as jnp

        d = gradients.rect_displacement(theta, slip, e_obs, n_obs)
        return jnp.sum(d**2)

    def _q_values(self, theta):
        """Okada q coordinate of each observation for this geometry."""
        from geodef import okada85

        args = okada85.setup_args(
            self._E_LINE,
            self._N_LINE,
            theta[2],
            theta[3],
            theta[4],
            theta[5],
            theta[6],
            0.0,
            1.0,
            0.0,
        )
        return backend.to_numpy(args[4])

    def test_profile_hits_fault_plane_ray(self):
        """The test grid must actually contain a q == 0 lane."""
        q = self._q_values(self._THETA_DIP)
        assert np.any(q == 0.0)

    @pytest.mark.parametrize(
        "slip", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    )
    def test_grad_finite_on_fault_plane_ray(self, slip):
        g = jax.grad(self._loss)(
            self._THETA_DIP, np.asarray(slip), self._E_LINE, self._N_LINE
        )
        assert np.all(np.isfinite(backend.to_numpy(g)))

    def test_grad_matches_fd_on_fault_plane_ray(self):
        """The q == 0 lane is a removable artefact: the displacement is
        smooth there, so the gradient must match finite differences."""
        slip = np.array([1.0, 0.7, 0.0])
        g = backend.to_numpy(
            jax.grad(self._loss)(self._THETA_DIP, slip, self._E_LINE, self._N_LINE)
        )
        fd = _fd_jacobian(
            lambda th: np.asarray(self._loss(th, slip, self._E_LINE, self._N_LINE)),
            self._THETA_DIP,
            rel_step=1e-7,
            scale=1e3,
        )
        np.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-12)

    def test_primal_continuous_across_fault_plane_ray(self):
        """Values on the q == 0 lane agree with immediately adjacent
        points (the fix must not change the physics)."""
        slip = np.array([1.0, 0.7, 0.0])
        q = self._q_values(self._THETA_DIP)
        idx = int(np.argmin(np.abs(q)))
        for de in (-1e-3, 1e-3):
            d0 = backend.to_numpy(
                gradients.rect_displacement(
                    self._THETA_DIP,
                    slip,
                    self._E_LINE[idx : idx + 1],
                    self._N_LINE[idx : idx + 1],
                )
            )
            d1 = backend.to_numpy(
                gradients.rect_displacement(
                    self._THETA_DIP,
                    slip,
                    self._E_LINE[idx : idx + 1] + de,
                    self._N_LINE[idx : idx + 1],
                )
            )
            np.testing.assert_allclose(d0, d1, rtol=1e-4, atol=1e-10)

    def test_primal_unchanged_numpy_backend(self):
        """NumPy-backend values on the q == 0 profile match the JAX path
        and stay warning-free."""
        slip = np.array([1.0, 0.7, 0.0])
        d_jax = backend.to_numpy(
            gradients.rect_displacement(
                self._THETA_DIP, slip, self._E_LINE, self._N_LINE
            )
        )
        backend.set_backend("numpy")
        d_np = backend.to_numpy(
            gradients.rect_displacement(
                self._THETA_DIP, slip, self._E_LINE, self._N_LINE
            )
        )
        np.testing.assert_allclose(d_np, d_jax, rtol=1e-12, atol=1e-16)

    @pytest.mark.parametrize("dip", [30.0, 60.0, 89.0])
    def test_grad_matches_jacfwd_regular_geometry(self, dip):
        theta = np.array([500.0, -800.0, 12e3, 37.0, dip, 15e3, 8e3])
        g_rev = backend.to_numpy(jax.grad(self._loss)(theta, _SLIP, _E_OBS, _N_OBS))
        g_fwd = backend.to_numpy(jax.jacfwd(self._loss)(theta, _SLIP, _E_OBS, _N_OBS))
        assert np.all(np.isfinite(g_rev))
        np.testing.assert_allclose(g_rev, g_fwd, rtol=1e-7, atol=1e-14)

    def test_grad_finite_through_rect_greens(self):
        import jax.numpy as jnp

        def loss(theta):
            G = gradients.rect_greens(
                theta, self._E_LINE, self._N_LINE, n_length=3, n_width=2
            )
            return jnp.sum(G**2)

        g = backend.to_numpy(jax.grad(loss)(self._THETA_DIP))
        assert np.all(np.isfinite(g))

    def test_grad_finite_vertical_fault_on_trace(self):
        """Vertical fault with observations exactly on the trace."""
        theta = np.array([0.0, 0.0, 8e3, 0.0, 90.0, 15e3, 8e3])
        e_obs = np.array([0.0, 0.0, 0.0])
        n_obs = np.array([2e3, 3e3, 30e3])
        slip = np.array([1.0, 0.0, 0.0])
        g = backend.to_numpy(jax.grad(self._loss)(theta, slip, e_obs, n_obs))
        assert np.all(np.isfinite(g))


class TestLosProject:
    def test_matches_manual_dot_product(self):
        backend.set_backend("numpy")
        rng = np.random.default_rng(3)
        nobs = 4
        look = rng.normal(size=(nobs, 3))
        look /= np.linalg.norm(look, axis=1, keepdims=True)
        G = rng.normal(size=(3 * nobs, 6))

        G_los = gradients.los_project(G, look)
        assert G_los.shape == (nobs, 6)
        for i in range(nobs):
            expected = (
                look[i, 0] * G[3 * i]
                + look[i, 1] * G[3 * i + 1]
                + look[i, 2] * G[3 * i + 2]
            )
            np.testing.assert_allclose(G_los[i], expected, rtol=1e-14)

    def test_traceable_through_projection(self):
        look = np.tile([0.6, 0.0, 0.8], (5, 1))

        def los_data(th):
            return gradients.los_project(
                gradients.rect_greens(th, _E_OBS, _N_OBS, 2, 2), look
            )

        jac = jax.jacfwd(los_data)(_THETA)
        assert np.all(np.isfinite(backend.to_numpy(jac)))
        fd = _fd_jacobian(lambda th: backend.to_numpy(los_data(th)), _THETA)
        np.testing.assert_allclose(backend.to_numpy(jac), fd, rtol=1e-5, atol=1e-12)
