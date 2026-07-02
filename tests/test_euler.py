"""Tests for geodef.euler pole estimation and velocity prediction."""

import numpy as np
import pytest

from geodef import euler


@pytest.fixture
def stations():
    """A spread of stations around a subduction margin."""
    rng = np.random.default_rng(0)
    lat = rng.uniform(-5.0, 15.0, 40)
    lon = rng.uniform(95.0, 110.0, 40)
    return lat, lon


class TestEulerVector:
    def test_location_roundtrip(self):
        pole = (55.0, -120.0, 0.75)
        omega = euler.euler_vector(*pole)
        lat, lon, rate = euler.euler_location(omega)
        assert lat == pytest.approx(pole[0], abs=1e-6)
        assert lon == pytest.approx(pole[1], abs=1e-6)
        assert rate == pytest.approx(pole[2], abs=1e-9)

    def test_vector_shape(self):
        assert euler.euler_vector(10.0, 100.0, 0.5).shape == (3,)


class TestPoleVelocity:
    def test_shapes(self, stations):
        lat, lon = stations
        ve, vn = euler.pole_velocity(lat, lon, 55.0, -120.0, 0.75)
        assert ve.shape == lat.shape
        assert vn.shape == lat.shape

    def test_zero_rate_gives_zero_velocity(self, stations):
        lat, lon = stations
        ve, vn = euler.pole_velocity(lat, lon, 55.0, -120.0, 0.0)
        np.testing.assert_allclose(ve, 0.0, atol=1e-12)
        np.testing.assert_allclose(vn, 0.0, atol=1e-12)

    def test_rot_matrix_shape(self, stations):
        lat, lon = stations
        rx = euler.euler_rot_matrix(lat, lon)
        assert rx.shape == (2 * lat.shape[0], 3)


class TestBestFitPole:
    def test_recovers_known_pole(self, stations):
        lat, lon = stations
        true = (55.0, -120.0, 0.75)
        ve, vn = euler.pole_velocity(lat, lon, *true)
        sig = np.full(lat.shape[0], 0.5)
        pole, cov, chi2 = euler.best_fit_pole(lat, lon, ve, vn, sig, sig, 0.0)
        np.testing.assert_allclose(pole, true, atol=1e-4)
        assert chi2 < 1e-10  # noise-free data fits exactly
        assert cov.shape == (3, 3)

    def test_recovers_pole_with_noise(self, stations):
        lat, lon = stations
        true = (40.0, -100.0, 1.2)
        ve, vn = euler.pole_velocity(lat, lon, *true)
        rng = np.random.default_rng(1)
        sig = np.full(lat.shape[0], 0.3)
        ve_n = ve + rng.normal(0, 0.3, lat.shape[0])
        vn_n = vn + rng.normal(0, 0.3, lat.shape[0])
        pole, _, chi2 = euler.best_fit_pole(lat, lon, ve_n, vn_n, sig, sig, 0.0)
        np.testing.assert_allclose(pole[:2], true[:2], atol=2.0)
        assert pole[2] == pytest.approx(true[2], abs=0.1)
        assert 0.3 < chi2 < 3.0  # reduced chi2 near 1 for correct sigmas

    def test_too_few_stations_raises(self):
        with pytest.raises(ValueError, match="two stations"):
            euler.best_fit_pole(
                np.array([0.0]),
                np.array([100.0]),
                np.array([1.0]),
                np.array([1.0]),
                np.array([0.5]),
                np.array([0.5]),
            )


class TestRemovePole:
    def test_removes_rigid_motion(self, stations):
        lat, lon = stations
        pole = (55.0, -120.0, 0.75)
        ve, vn = euler.pole_velocity(lat, lon, *pole)
        ve_res, vn_res = euler.remove_pole(lat, lon, ve, vn, *pole)
        np.testing.assert_allclose(ve_res, 0.0, atol=1e-10)
        np.testing.assert_allclose(vn_res, 0.0, atol=1e-10)
