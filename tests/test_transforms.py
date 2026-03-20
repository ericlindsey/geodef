"""Tests for geodef.transforms (coordinate transformations).

Migrated from related/shakeout_v2/test_geod_transform.py with updated imports.
"""

import pytest
import numpy as np

from geodef import transforms


# ---------------------------------------------------------------------------
# Test Constants
# ---------------------------------------------------------------------------
a = 6378137.0
finv = 298.257223563
f = 1.0 / finv
e2 = 2.0 * f - f * f
b = a * (1.0 - f)

ATOL = 1e-6
RTOL = 1e-6


# ---------------------------------------------------------------------------
# 1. Round-Tripping (Identity) Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lat,lon,alt", [
    (35.0, -106.0, 1500.0),
    (0.0, 0.0, 0.0),
    (90.0, 0.0, 0.0),
    (-90.0, 180.0, 10000.0),
    (-30.5, 179.99, -500.0),
])
def test_geod2ecef2geod_scalar(lat, lon, alt):
    """Test round trip: geod -> ecef -> geod using scalar inputs."""
    x, y, z = transforms.geod2ecef(lat, lon, alt)
    lat_out, lon_out, alt_out = transforms.ecef2geod(x, y, z)

    np.testing.assert_allclose(lat_out, lat, atol=ATOL)
    np.testing.assert_allclose(lon_out, lon, atol=ATOL)
    np.testing.assert_allclose(alt_out, alt, atol=ATOL)


@pytest.mark.parametrize("lat,lon,alt,lat0,lon0,alt0", [
    (35.1, -106.1, 1500.0, 35.0, -106.0, 1500.0),
    (0.0, 90.0, 0.0, 0.0, 0.0, 0.0),
])
def test_geod2enu2geod_scalar(lat, lon, alt, lat0, lon0, alt0):
    """Test round trip: geod -> enu -> geod using scalar inputs."""
    e, n, u = transforms.geod2enu(lat, lon, alt, lat0, lon0, alt0)
    lat_out, lon_out, alt_out = transforms.enu2geod(e, n, u, lat0, lon0, alt0)

    np.testing.assert_allclose(lat_out, lat, atol=ATOL)
    np.testing.assert_allclose(lon_out, lon, atol=ATOL)
    np.testing.assert_allclose(alt_out, alt, atol=ATOL)


@pytest.mark.parametrize("e,n,u,lat0,lon0,alt0", [
    (100.0, -200.0, 50.0, 35.0, -106.0, 1500.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
])
def test_ecef2enu2ecef_scalar(e, n, u, lat0, lon0, alt0):
    """Test round trip: enu -> ecef -> enu using scalar inputs."""
    x_ecef, y_ecef, z_ecef = transforms.enu2ecef(e, n, u, lat0, lon0, alt0)
    e_out, n_out, u_out = transforms.ecef2enu(x_ecef, y_ecef, z_ecef, lat0, lon0, alt0)

    np.testing.assert_allclose(e_out, e, atol=ATOL)
    np.testing.assert_allclose(n_out, n, atol=ATOL)
    np.testing.assert_allclose(u_out, u, atol=ATOL)


# ---------------------------------------------------------------------------
# 2. Known Reference Values
# ---------------------------------------------------------------------------
def test_geod2ecef_reference_equator():
    """Test against known WGS84 dimensions at the equator."""
    x, y, z = transforms.geod2ecef(0.0, 0.0, 0.0)
    np.testing.assert_allclose(x, a, atol=ATOL)
    np.testing.assert_allclose(y, 0.0, atol=ATOL)
    np.testing.assert_allclose(z, 0.0, atol=ATOL)


def test_geod2ecef_reference_pole():
    """Test against known WGS84 dimensions at the North Pole."""
    x, y, z = transforms.geod2ecef(90.0, 0.0, 0.0)
    np.testing.assert_allclose(x, 0.0, atol=ATOL)
    np.testing.assert_allclose(y, 0.0, atol=ATOL)
    np.testing.assert_allclose(z, b, atol=ATOL)


def test_spher2geod_vs_geod2spher():
    """Test the conversions between spherical and geodetic latitude."""
    lat_geod = np.array([0.0, 45.0, 90.0, -45.0, -90.0])
    lat_spher = transforms.geod2spher(lat_geod)

    np.testing.assert_allclose(lat_spher[[0, 2, 4]], lat_geod[[0, 2, 4]], atol=ATOL)

    lat_geod_out = transforms.spher2geod(lat_spher)
    np.testing.assert_allclose(lat_geod_out, lat_geod, atol=ATOL)


# ---------------------------------------------------------------------------
# 3. Edge Cases
# ---------------------------------------------------------------------------
def test_coincident_points_distance():
    """Distance between a point and itself should be exactly 0."""
    lat, lon = 35.0, -106.0
    dist_vincenty, az0, az1 = transforms.vincenty(lat, lon, lat, lon)
    dist_haver = transforms.haversine(lat, lon, lat, lon)

    assert dist_vincenty == 0.0
    assert dist_haver == 0.0


def test_antipodal_distance():
    """Tests for points 180 degrees apart."""
    dist, az0, az1 = transforms.vincenty(90.0, 0.0, -90.0, 0.0)

    assert not np.isnan(dist)
    assert 19.9e6 < dist < 20.1e6

    with pytest.warns(RuntimeWarning, match="Vincenty formula failed to converge"):
        dist_eq, az0_eq, az1_eq = transforms.vincenty(0.0, 0.0, 0.0, 180.0)

    assert np.isnan(dist_eq)
    assert np.isnan(az0_eq)
    assert np.isnan(az1_eq)

    dist_haver = transforms.haversine(0.0, 0.0, 0.0, 180.0)
    np.testing.assert_allclose(dist_haver, np.pi * 6371000.0, rtol=1e-3)


# ---------------------------------------------------------------------------
# 4. Data Types and Vectorization
# ---------------------------------------------------------------------------
def test_geod2ecef2geod_vector():
    """Test geod2ecef and ecef2geod with numpy arrays."""
    lat = np.array([0.0, 45.0, 90.0, -30.0])
    lon = np.array([0.0, -106.0, 180.0, 45.0])
    alt = np.array([0.0, 1500.0, -100.0, 10000.0])

    x, y, z = transforms.geod2ecef(lat, lon, alt)

    assert isinstance(x, np.ndarray)
    assert np.shape(x) == lat.shape

    lat_out, lon_out, alt_out = transforms.ecef2geod(x, y, z)

    np.testing.assert_allclose(lat_out, lat, atol=ATOL)
    np.testing.assert_allclose(lon_out, lon, atol=ATOL)
    np.testing.assert_allclose(alt_out, alt, atol=ATOL)


def test_ecef2enu_vel_vector():
    """Test ecef2enu_vel with vectorized inputs."""
    lat0 = np.array([0.0, 45.0])
    lon0 = np.array([0.0, -105.0])

    xvel = np.array([1.0, 0.0])
    yvel = np.array([0.0, 1.0])
    zvel = np.array([0.0, 0.0])

    evel, nvel, uvel = transforms.ecef2enu_vel(xvel, yvel, zvel, lat0, lon0)

    assert len(evel) == 2
    assert len(nvel) == 2
    assert len(uvel) == 2


def test_enu2ecef_vel_vector():
    """Test enu2ecef_vel with vectorized inputs."""
    lat0 = np.array([0.0, 45.0])
    lon0 = np.array([0.0, -105.0])

    evel = np.array([1.0, 0.0])
    nvel = np.array([0.0, 1.0])
    uvel = np.array([0.0, 0.0])

    xvel, yvel, zvel = transforms.enu2ecef_vel(evel, nvel, uvel, lat0, lon0)

    assert len(xvel) == 2
    assert len(yvel) == 2
    assert len(zvel) == 2


def test_enu2ecef_sigma_vector():
    """Test enu2ecef_sigma with vectorized inputs."""
    esigma = np.array([0.1, 0.2])
    nsigma = np.array([0.1, 0.2])
    usigma = np.array([0.3, 0.4])
    rhoen = np.array([0.0, 0.5])
    lat0 = np.array([35.0, 45.0])
    lon0 = np.array([-106.0, 120.0])

    cov_out = transforms.enu2ecef_sigma(esigma, nsigma, usigma, rhoen, lat0, lon0)

    assert cov_out.shape == (6, 6)


# ---------------------------------------------------------------------------
# 5. Extension Features
# ---------------------------------------------------------------------------
def test_custom_ellipsoid():
    """Test custom ellipsoid inputs."""
    sphere = transforms.Ellipsoid(a=6371000.0, f=0.0)

    x, y, z = transforms.geod2ecef(0.0, 0.0, 0.0, ellps=sphere)
    np.testing.assert_allclose(x, 6371000.0, atol=ATOL)
    np.testing.assert_allclose(y, 0.0, atol=ATOL)
    np.testing.assert_allclose(z, 0.0, atol=ATOL)

    x, y, z = transforms.geod2ecef(90.0, 0.0, 0.0, ellps=sphere)
    np.testing.assert_allclose(x, 0.0, atol=ATOL)
    np.testing.assert_allclose(y, 0.0, atol=ATOL)
    np.testing.assert_allclose(z, 6371000.0, atol=ATOL)


def test_pyproj_integration():
    """Test dynamic pyproj integration using crs kwarg."""
    import importlib.util
    if importlib.util.find_spec('pyproj') is None:
        pytest.skip("pyproj not installed, skipping pyproj dynamic adapter tests.")

    lat, lon, alt = 35.0, -106.0, 1500.0

    x1, y1, z1 = transforms.geod2ecef(lat, lon, alt)
    x2, y2, z2 = transforms.geod2ecef(lat, lon, alt, crs="EPSG:4326")

    np.testing.assert_allclose(x1, x2, atol=1e-2)
    np.testing.assert_allclose(y1, y2, atol=1e-2)
    np.testing.assert_allclose(z1, z2, atol=1e-2)

    lat_out, lon_out, alt_out = transforms.ecef2geod(x2, y2, z2, crs="EPSG:4326")
    np.testing.assert_allclose(lat_out, lat, atol=ATOL)
    np.testing.assert_allclose(lon_out, lon, atol=ATOL)
    np.testing.assert_allclose(alt_out, alt, atol=ATOL)
