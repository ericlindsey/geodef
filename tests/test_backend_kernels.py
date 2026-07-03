"""Cross-validation of the JAX backend against the Matlab reference data.

Runs the same okada85 and triangular-dislocation reference comparisons as
``test_okada85.py`` and ``test_tdcalc.py``, but with the JAX backend active,
at the same tolerances the NumPy engines meet. Skipped entirely when JAX is
not installed.
"""

import numpy as np
import pytest

from geodef import backend, greens, okada85
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


class TestBackendEquivalence:
    """NumPy and JAX backends agree on a general dipping-fault case."""

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
