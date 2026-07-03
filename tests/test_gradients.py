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
