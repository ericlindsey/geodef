"""Cross-validation of the JAX backend against the Matlab reference data.

Runs the same okada85 and triangular-dislocation reference comparisons as
``test_okada85.py`` and ``test_tdcalc.py``, but with the JAX backend active,
at the same tolerances the NumPy engines meet. Skipped entirely when JAX is
not installed.
"""

import numpy as np
import pytest

from geodef import backend, greens, okada85, okada92
from geodef import tri as tdcalc
from tests.test_okada85 import (
    _REFERENCE_PARAMS,
    _REFERENCE_RESULTS,
    _centroid_to_okada_args,
)
from tests.test_tdcalc import _SUBSET_STRIDE, _load_reference

jax = pytest.importorskip("jax")


@pytest.fixture(autouse=True)
def jax_backend():
    """Run every test in this module on the JAX backend."""
    backend.set_backend("jax")
    yield
    backend.set_backend("numpy")


@pytest.mark.parametrize("case_idx", range(9), ids=[f"case_{i}" for i in range(9)])
class TestOkada85JaxReference:
    """The 9 Matlab reference cases, evaluated on the JAX backend."""

    def _args(self, case_idx: int) -> tuple:
        p = _REFERENCE_PARAMS
        e, n, depth = _centroid_to_okada_args(
            p["x"][case_idx],
            p["y"][case_idx],
            p["d"][case_idx],
            p["strike"],
            p["dip"][case_idx],
            p["L"],
            p["W"],
        )
        return (
            e,
            n,
            depth,
            p["strike"],
            p["dip"][case_idx],
            p["L"],
            p["W"],
            p["rake"][case_idx],
            p["slip"][case_idx],
            p["u3"][case_idx],
            p["nu"],
        )

    def test_displacement(self, case_idx: int) -> None:
        ue, un, uz = okada85.displacement(*self._args(case_idx))
        ref = _REFERENCE_RESULTS[case_idx]
        np.testing.assert_almost_equal(backend.to_numpy(ue), ref[0], decimal=15)
        np.testing.assert_almost_equal(backend.to_numpy(un), ref[1], decimal=15)
        np.testing.assert_almost_equal(backend.to_numpy(uz), ref[2], decimal=15)

    def test_tilt(self, case_idx: int) -> None:
        uze, uzn = okada85.tilt(*self._args(case_idx))
        ref = _REFERENCE_RESULTS[case_idx]
        np.testing.assert_almost_equal(backend.to_numpy(uze), ref[7], decimal=15)
        np.testing.assert_almost_equal(backend.to_numpy(uzn), ref[8], decimal=15)

    def test_strain(self, case_idx: int) -> None:
        unn, une, uen, uee = okada85.strain(*self._args(case_idx))
        ref = _REFERENCE_RESULTS[case_idx]
        np.testing.assert_almost_equal(backend.to_numpy(-uee), ref[3], decimal=15)
        np.testing.assert_almost_equal(backend.to_numpy(-uen), ref[4], decimal=15)
        np.testing.assert_almost_equal(backend.to_numpy(-une), ref[5], decimal=15)
        np.testing.assert_almost_equal(backend.to_numpy(-unn), ref[6], decimal=15)


@pytest.mark.parametrize(
    "config_name", ["FS_simple", "FS_complex", "HS_simple", "HS_complex"]
)
class TestTriJaxReference:
    """Triangular-dislocation Matlab references, evaluated on the JAX backend."""

    def test_displacement(self, config_name: str) -> None:
        ref = _load_reference(config_name)
        obs = ref["obs"][::_SUBSET_STRIDE]
        triangle = ref["tri"].astype(float)
        slip = ref["slip"].astype(float)
        nu = float(ref["nu"])
        disp_func = (
            tdcalc.TDdispFS if config_name.startswith("FS") else tdcalc.TDdispHS
        )

        disp = backend.to_numpy(disp_func(obs, triangle, slip, nu))

        np.testing.assert_almost_equal(disp[:, 0], ref["UEf"][::_SUBSET_STRIDE])
        np.testing.assert_almost_equal(disp[:, 1], ref["UNf"][::_SUBSET_STRIDE])
        np.testing.assert_almost_equal(disp[:, 2], ref["UVf"][::_SUBSET_STRIDE])

    def test_strain(self, config_name: str) -> None:
        ref = _load_reference(config_name)
        obs = ref["obs"][::_SUBSET_STRIDE]
        triangle = ref["tri"].astype(float)
        slip = ref["slip"].astype(float)
        nu = float(ref["nu"])
        strain_func = (
            tdcalc.TDstrainFS if config_name.startswith("FS") else tdcalc.TDstrainHS
        )

        strain = backend.to_numpy(strain_func(obs, triangle, slip, nu)).T
        ref_strain = ref["Strain"][::_SUBSET_STRIDE, :].T

        np.testing.assert_almost_equal(strain, ref_strain)


class TestFloat32Mode:
    """Opt-in float32 precision runs end-to-end with reduced accuracy."""

    @pytest.fixture(autouse=True)
    def float32_precision(self):
        backend.set_precision("float32")
        yield
        backend.set_precision("float64")

    def test_okada85_displacement_close_to_float64(self) -> None:
        rng = np.random.default_rng(2)
        e = rng.uniform(5e3, 3e4, 100) * rng.choice([-1, 1], 100)
        n = rng.uniform(5e3, 3e4, 100) * rng.choice([-1, 1], 100)
        args = (e, n, 8e3, 35.0, 55.0, 2e4, 1e4, 20.0, 1.2, 0.1)

        backend.set_precision("float64")
        ref = okada85.displacement(*args)
        backend.set_precision("float32")
        result = okada85.displacement(*args)

        scale = max(np.max(np.abs(backend.to_numpy(c))) for c in ref)
        for r, x in zip(result, ref):
            assert backend.to_numpy(r).dtype == np.float32
            np.testing.assert_allclose(
                backend.to_numpy(r), backend.to_numpy(x),
                rtol=2e-3, atol=2e-4 * scale,
            )

    def test_tri_displacement_close_to_float64(self) -> None:
        rng = np.random.default_rng(4)
        nobs = 100
        obs = np.column_stack(
            [rng.uniform(-2e4, 2e4, nobs), rng.uniform(-2e4, 2e4, nobs),
             np.zeros(nobs)]
        )
        triangle = np.array(
            [[0.0, 0.0, -1e3], [8e3, 2e3, -4e3], [1e3, 6e3, -6e3]]
        )
        slip = np.array([1.0, 0.5, 0.2])

        backend.set_precision("float64")
        ref = backend.to_numpy(tdcalc.TDdispHS(obs, triangle, slip, 0.25))
        backend.set_precision("float32")
        result = backend.to_numpy(tdcalc.TDdispHS(obs, triangle, slip, 0.25))

        assert result.dtype == np.float32
        scale = np.max(np.abs(ref))
        np.testing.assert_allclose(result, ref, rtol=5e-3, atol=5e-4 * scale)

    def test_displacement_greens_close_to_float64(self) -> None:
        rng = np.random.default_rng(6)
        nobs, npatch = 30, 8
        args = (
            rng.uniform(-0.3, 0.3, nobs),
            rng.uniform(-0.3, 0.3, nobs),
            rng.uniform(-0.15, 0.15, npatch),
            rng.uniform(-0.15, 0.15, npatch),
            rng.uniform(5e3, 2e4, npatch),
            rng.uniform(0.0, 360.0, npatch),
            rng.uniform(10.0, 90.0, npatch),
            np.full(npatch, 8e3),
            np.full(npatch, 5e3),
        )

        backend.set_precision("float64")
        ref = greens.displacement_greens(*args)
        backend.set_precision("float32")
        result = greens.displacement_greens(*args)

        # far-field coefficients of small patches lose relative accuracy
        # in float32 (the Chinnery corner differences nearly cancel), so
        # assert against the matrix scale rather than element-wise
        scale = np.max(np.abs(ref))
        np.testing.assert_allclose(result, ref, rtol=0, atol=2e-2 * scale)


class TestAbicSweepJax:
    """The batched JAX ABIC sweep reproduces the NumPy loop."""

    def test_abic_curve_matches_numpy(self) -> None:
        from geodef.data import GNSS
        from geodef.fault import Fault
        from geodef.invert import abic_curve

        fault = Fault.planar(
            lat=0.0, lon=100.0, depth=15e3, strike=320.0, dip=15.0,
            length=80e3, width=40e3, n_length=4, n_width=3,
        )
        lat_1d = np.linspace(-0.5, 0.5, 5)
        lon_1d = np.linspace(99.5, 100.5, 5)
        lon_g, lat_g = np.meshgrid(lon_1d, lat_1d)
        lat, lon = lat_g.ravel(), lon_g.ravel()

        rng = np.random.default_rng(21)
        slip_ss = rng.uniform(0.0, 1.0, fault.n_patches)
        slip_ds = rng.uniform(0.0, 0.5, fault.n_patches)

        backend.set_backend("numpy")
        ue, un, uz = fault.displacement(lat, lon, slip_ss, slip_ds)
        n = len(lat)
        gnss = GNSS(
            lon, lat, ve=ue, vn=un, vu=uz,
            se=np.full(n, 0.001), sn=np.full(n, 0.001), su=np.full(n, 0.001),
        )

        result_np = abic_curve(fault, gnss, smoothing_range=(1e0, 1e6), n=12)
        backend.set_backend("jax")
        result_jax = abic_curve(fault, gnss, smoothing_range=(1e0, 1e6), n=12)

        np.testing.assert_allclose(
            result_jax.abic_values, result_np.abic_values, rtol=1e-8
        )
        np.testing.assert_allclose(
            result_jax.misfits, result_np.misfits, rtol=1e-8
        )
        np.testing.assert_allclose(
            result_jax.model_norms, result_np.model_norms, rtol=1e-8
        )
        assert result_jax.optimal == result_np.optimal


class TestBackendEquivalence:
    """NumPy and JAX backends agree on a general dipping-fault case."""

    def test_okada92_matches_numpy(self) -> None:
        rng = np.random.default_rng(11)
        n = 100
        X = rng.uniform(-3e4, 3e4, n)
        Y = rng.uniform(-3e4, 3e4, n)
        Z = -rng.uniform(0.0, 2e4, n)
        args = (12e3, 37.0, 55.0, 15e3, 8e3, 1.0, 0.7, 0.3, 30e9, 0.25)

        backend.set_backend("numpy")
        disp_np, strain_np = okada92.okada92(X, Y, Z, *args)
        backend.set_backend("jax")
        disp_jax, strain_jax = okada92.okada92(X, Y, Z, *args)

        np.testing.assert_allclose(
            backend.to_numpy(disp_jax), disp_np, rtol=1e-10, atol=1e-18
        )
        np.testing.assert_allclose(
            backend.to_numpy(strain_jax), strain_np, rtol=1e-10, atol=1e-20
        )

    def test_okada85_displacement_matches_numpy(self) -> None:
        rng = np.random.default_rng(42)
        e = rng.uniform(-3e4, 3e4, 200)
        n = rng.uniform(-3e4, 3e4, 200)
        args = (e, n, 8e3, 35.0, 55.0, 2e4, 1e4, 20.0, 1.2, 0.1)

        backend.set_backend("numpy")
        expected = okada85.displacement(*args)
        backend.set_backend("jax")
        result = okada85.displacement(*args)

        for r, x in zip(result, expected):
            np.testing.assert_allclose(backend.to_numpy(r), x, rtol=1e-12, atol=1e-15)

    def test_displacement_greens_matches_numpy(self) -> None:
        rng = np.random.default_rng(7)
        nobs, npatch = 40, 12
        lat = rng.uniform(-0.3, 0.3, nobs)
        lon = rng.uniform(-0.3, 0.3, nobs)
        lat0 = rng.uniform(-0.15, 0.15, npatch)
        lon0 = rng.uniform(-0.15, 0.15, npatch)
        depth = rng.uniform(5e3, 2e4, npatch)
        strike = rng.uniform(0.0, 360.0, npatch)
        dip = rng.uniform(10.0, 90.0, npatch)
        L = np.full(npatch, 8e3)
        W = np.full(npatch, 5e3)
        args = (lat, lon, lat0, lon0, depth, strike, dip, L, W)

        backend.set_backend("numpy")
        expected = greens.displacement_greens(*args)
        backend.set_backend("jax")
        result = greens.displacement_greens(*args)

        assert isinstance(result, np.ndarray)
        # JIT fusion reassociates the Chinnery differences, so agreement is
        # slightly looser than eager op-by-op evaluation
        np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-12)

    def test_strain_greens_surface_matches_numpy(self) -> None:
        rng = np.random.default_rng(9)
        nobs, npatch = 30, 10
        lat = rng.uniform(-0.3, 0.3, nobs)
        lon = rng.uniform(-0.3, 0.3, nobs)
        lat0 = rng.uniform(-0.15, 0.15, npatch)
        lon0 = rng.uniform(-0.15, 0.15, npatch)
        depth = rng.uniform(5e3, 2e4, npatch)
        strike = rng.uniform(0.0, 360.0, npatch)
        dip = rng.uniform(10.0, 90.0, npatch)
        L = np.full(npatch, 8e3)
        W = np.full(npatch, 5e3)
        args = (lat, lon, lat0, lon0, depth, strike, dip, L, W)

        backend.set_backend("numpy")
        expected = greens.strain_greens(*args)
        backend.set_backend("jax")
        result = greens.strain_greens(*args)

        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-16)

    def test_strain_greens_depth_matches_numpy(self) -> None:
        rng = np.random.default_rng(13)
        npatch = 8
        lat0 = rng.uniform(-0.15, 0.15, npatch)
        lon0 = rng.uniform(-0.15, 0.15, npatch)
        depth = rng.uniform(5e3, 2e4, npatch)
        strike = rng.uniform(0.0, 360.0, npatch)
        dip = rng.uniform(10.0, 90.0, npatch)
        L = np.full(npatch, 8e3)
        W = np.full(npatch, 5e3)
        # self-stress configuration: observe at the patch centroids
        args = (lat0, lon0, lat0, lon0, depth, strike, dip, L, W)

        backend.set_backend("numpy")
        expected = greens.strain_greens(*args, obs_depth=depth)
        backend.set_backend("jax")
        result = greens.strain_greens(*args, obs_depth=depth)

        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-16)

    def test_tri_displacement_matches_numpy(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        obs = np.column_stack(
            [rng.uniform(-2e4, 2e4, n), rng.uniform(-2e4, 2e4, n), np.zeros(n)]
        )
        triangle = np.array([[0.0, 0.0, -1e3], [8e3, 2e3, -4e3], [1e3, 6e3, -6e3]])
        slip = np.array([1.0, 0.5, 0.2])

        backend.set_backend("numpy")
        expected = tdcalc.TDdispHS(obs, triangle, slip, 0.25)
        backend.set_backend("jax")
        result = backend.to_numpy(tdcalc.TDdispHS(obs, triangle, slip, 0.25))

        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-15)
