"""Tests for geodef.DataSet classes (Phase 3.2).

Covers: construction, validation, properties, project() methods,
covariance, component filtering (GNSS), and file I/O.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from geodef.data import GNSS, InSAR, Vertical, DataSet


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def gnss_3station():
    """Three-station GNSS dataset with all components."""
    lat = np.array([0.0, 1.0, 2.0])
    lon = np.array([100.0, 101.0, 102.0])
    ve = np.array([1.0, 2.0, 3.0])
    vn = np.array([0.5, 1.5, 2.5])
    vu = np.array([-0.1, -0.2, -0.3])
    se = np.array([0.1, 0.2, 0.3])
    sn = np.array([0.1, 0.2, 0.3])
    su = np.array([0.5, 0.5, 0.5])
    return GNSS(lat, lon, ve, vn, vu, se, sn, su)


@pytest.fixture
def gnss_horizontal():
    """Two-station GNSS dataset with horizontal only (vu=None)."""
    lat = np.array([0.0, 1.0])
    lon = np.array([100.0, 101.0])
    ve = np.array([1.0, 2.0])
    vn = np.array([0.5, 1.5])
    se = np.array([0.1, 0.2])
    sn = np.array([0.1, 0.2])
    return GNSS(lat, lon, ve, vn, None, se, sn, None)


@pytest.fixture
def insar_5pixel():
    """Five-pixel InSAR dataset with ascending-like look vectors."""
    lat = np.arange(5, dtype=float)
    lon = np.full(5, 100.0)
    los = np.array([0.01, 0.02, 0.03, 0.02, 0.01])
    sigma = np.full(5, 0.005)
    look_e = np.full(5, 0.38)
    look_n = np.full(5, -0.09)
    look_u = np.full(5, 0.92)
    return InSAR(lat, lon, los, sigma, look_e, look_n, look_u)


@pytest.fixture
def vertical_4pt():
    """Four-point vertical displacement dataset."""
    lat = np.array([0.0, 0.5, 1.0, 1.5])
    lon = np.full(4, 100.0)
    disp = np.array([0.01, 0.02, 0.015, 0.005])
    sigma = np.full(4, 0.003)
    return Vertical(lat, lon, disp, sigma)


# ======================================================================
# 1. DataSet base class
# ======================================================================

class TestDataSetBase:
    """Test that DataSet is an abstract base."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            DataSet(np.array([0.0]), np.array([100.0]))

    def test_gnss_is_dataset(self, gnss_3station):
        assert isinstance(gnss_3station, DataSet)

    def test_insar_is_dataset(self, insar_5pixel):
        assert isinstance(insar_5pixel, DataSet)

    def test_vertical_is_dataset(self, vertical_4pt):
        assert isinstance(vertical_4pt, DataSet)


# ======================================================================
# 2. GNSS construction and properties
# ======================================================================

class TestGNSSConstruction:
    """Test GNSS.__init__ validation and basic properties."""

    def test_basic_properties(self, gnss_3station):
        assert gnss_3station.n_stations == 3
        assert gnss_3station.n_obs == 9  # 3 components x 3 stations
        assert gnss_3station.greens_type == "displacement"

    def test_obs_interleaved(self, gnss_3station):
        obs = gnss_3station.obs
        assert obs.shape == (9,)
        # [e1, n1, u1, e2, n2, u2, e3, n3, u3]
        np.testing.assert_array_equal(
            obs, [1.0, 0.5, -0.1, 2.0, 1.5, -0.2, 3.0, 2.5, -0.3]
        )

    def test_sigma_interleaved(self, gnss_3station):
        sigma = gnss_3station.sigma
        assert sigma.shape == (9,)
        np.testing.assert_array_equal(
            sigma, [0.1, 0.1, 0.5, 0.2, 0.2, 0.5, 0.3, 0.3, 0.5]
        )

    def test_lat_lon(self, gnss_3station):
        np.testing.assert_array_equal(gnss_3station.lat, [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(gnss_3station.lon, [100.0, 101.0, 102.0])

    def test_immutable_arrays(self, gnss_3station):
        with pytest.raises(ValueError):
            gnss_3station.lat[0] = 999.0
        with pytest.raises(ValueError):
            gnss_3station.obs[0] = 999.0

    def test_horizontal_only(self, gnss_horizontal):
        assert gnss_horizontal.n_stations == 2
        assert gnss_horizontal.n_obs == 4  # 2 components x 2 stations
        np.testing.assert_array_equal(
            gnss_horizontal.obs, [1.0, 0.5, 2.0, 1.5]
        )
        np.testing.assert_array_equal(
            gnss_horizontal.sigma, [0.1, 0.1, 0.2, 0.2]
        )

    def test_components_property(self, gnss_3station, gnss_horizontal):
        assert gnss_3station.components == "enu"
        assert gnss_horizontal.components == "en"

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            GNSS(
                np.array([0.0, 1.0]),
                np.array([100.0]),  # wrong length
                np.array([1.0, 2.0]),
                np.array([0.5, 1.5]),
                np.array([-0.1, -0.2]),
                np.array([0.1, 0.2]),
                np.array([0.1, 0.2]),
                np.array([0.5, 0.5]),
            )

    def test_ve_vn_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            GNSS(
                np.array([0.0]),
                np.array([100.0]),
                np.array([1.0, 2.0]),  # wrong length
                np.array([0.5]),
                np.array([-0.1]),
                np.array([0.1]),
                np.array([0.1]),
                np.array([0.5]),
            )

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="positive"):
            GNSS(
                np.array([0.0]),
                np.array([100.0]),
                np.array([1.0]),
                np.array([0.5]),
                np.array([-0.1]),
                np.array([-0.1]),  # negative sigma
                np.array([0.1]),
                np.array([0.5]),
            )

    def test_vu_none_su_none_required(self):
        """If vu is None, su must also be None."""
        with pytest.raises(ValueError, match="both.*None"):
            GNSS(
                np.array([0.0]),
                np.array([100.0]),
                np.array([1.0]),
                np.array([0.5]),
                None,  # vu=None
                np.array([0.1]),
                np.array([0.1]),
                np.array([0.5]),  # su provided
            )

    def test_scalar_inputs_broadcast(self):
        """Single-station GNSS from scalar values."""
        g = GNSS(
            np.array([0.0]),
            np.array([100.0]),
            np.array([1.0]),
            np.array([0.5]),
            np.array([-0.1]),
            np.array([0.1]),
            np.array([0.2]),
            np.array([0.5]),
        )
        assert g.n_stations == 1
        assert g.n_obs == 3


# ======================================================================
# 3. GNSS project()
# ======================================================================

class TestGNSSProject:
    """Test GNSS.project() maps displacement components correctly."""

    def test_project_3component(self, gnss_3station):
        ue = np.array([10.0, 20.0, 30.0])
        un = np.array([1.0, 2.0, 3.0])
        uz = np.array([0.1, 0.2, 0.3])
        result = gnss_3station.project(ue, un, uz)
        expected = np.array([10.0, 1.0, 0.1, 20.0, 2.0, 0.2, 30.0, 3.0, 0.3])
        np.testing.assert_array_equal(result, expected)

    def test_project_horizontal_only(self, gnss_horizontal):
        ue = np.array([10.0, 20.0])
        un = np.array([1.0, 2.0])
        uz = np.array([0.1, 0.2])
        result = gnss_horizontal.project(ue, un, uz)
        expected = np.array([10.0, 1.0, 20.0, 2.0])
        np.testing.assert_array_equal(result, expected)

    def test_project_output_length(self, gnss_3station):
        n = gnss_3station.n_stations
        ue = np.ones(n)
        un = np.ones(n)
        uz = np.ones(n)
        assert gnss_3station.project(ue, un, uz).shape == (gnss_3station.n_obs,)


# ======================================================================
# 4. InSAR construction and properties
# ======================================================================

class TestInSARConstruction:
    """Test InSAR.__init__ validation and basic properties."""

    def test_basic_properties(self, insar_5pixel):
        assert insar_5pixel.n_stations == 5
        assert insar_5pixel.n_obs == 5  # 1 component per pixel
        assert insar_5pixel.greens_type == "displacement"

    def test_obs(self, insar_5pixel):
        np.testing.assert_array_equal(
            insar_5pixel.obs, [0.01, 0.02, 0.03, 0.02, 0.01]
        )

    def test_sigma(self, insar_5pixel):
        np.testing.assert_array_equal(
            insar_5pixel.sigma, np.full(5, 0.005)
        )

    def test_immutable_arrays(self, insar_5pixel):
        with pytest.raises(ValueError):
            insar_5pixel.lat[0] = 999.0

    def test_mismatched_look_vector_raises(self):
        with pytest.raises(ValueError, match="same length"):
            InSAR(
                np.array([0.0]),
                np.array([100.0]),
                np.array([0.01]),
                np.array([0.005]),
                np.array([0.38, 0.38]),  # wrong length
                np.array([-0.09]),
                np.array([0.92]),
            )

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="positive"):
            InSAR(
                np.array([0.0]),
                np.array([100.0]),
                np.array([0.01]),
                np.array([-0.005]),  # negative
                np.array([0.38]),
                np.array([-0.09]),
                np.array([0.92]),
            )


# ======================================================================
# 5. InSAR project()
# ======================================================================

class TestInSARProject:
    """Test InSAR.project() applies LOS projection."""

    def test_project_los(self, insar_5pixel):
        ue = np.ones(5)
        un = np.ones(5)
        uz = np.ones(5)
        result = insar_5pixel.project(ue, un, uz)
        # look_e * ue + look_n * un + look_u * uz
        expected = 0.38 * 1.0 + (-0.09) * 1.0 + 0.92 * 1.0
        np.testing.assert_allclose(result, np.full(5, expected))

    def test_project_pure_vertical(self):
        """Pure vertical look vector should return uz."""
        insar = InSAR(
            np.array([0.0]),
            np.array([100.0]),
            np.array([0.01]),
            np.array([0.005]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
        )
        result = insar.project(np.array([5.0]), np.array([3.0]), np.array([7.0]))
        np.testing.assert_allclose(result, [7.0])

    def test_project_output_length(self, insar_5pixel):
        n = insar_5pixel.n_stations
        result = insar_5pixel.project(np.ones(n), np.ones(n), np.ones(n))
        assert result.shape == (insar_5pixel.n_obs,)


# ======================================================================
# 6. Vertical construction and properties
# ======================================================================

class TestVerticalConstruction:
    """Test Vertical.__init__ validation and basic properties."""

    def test_basic_properties(self, vertical_4pt):
        assert vertical_4pt.n_stations == 4
        assert vertical_4pt.n_obs == 4
        assert vertical_4pt.greens_type == "displacement"

    def test_obs(self, vertical_4pt):
        np.testing.assert_array_equal(
            vertical_4pt.obs, [0.01, 0.02, 0.015, 0.005]
        )

    def test_sigma(self, vertical_4pt):
        np.testing.assert_array_equal(
            vertical_4pt.sigma, np.full(4, 0.003)
        )

    def test_immutable_arrays(self, vertical_4pt):
        with pytest.raises(ValueError):
            vertical_4pt.lat[0] = 999.0

    def test_mismatched_raises(self):
        with pytest.raises(ValueError, match="same length"):
            Vertical(
                np.array([0.0, 1.0]),
                np.array([100.0]),  # wrong length
                np.array([0.01, 0.02]),
                np.array([0.003, 0.003]),
            )


# ======================================================================
# 7. Vertical project()
# ======================================================================

class TestVerticalProject:
    """Test Vertical.project() extracts vertical component."""

    def test_project_returns_uz(self, vertical_4pt):
        ue = np.array([10.0, 20.0, 30.0, 40.0])
        un = np.array([1.0, 2.0, 3.0, 4.0])
        uz = np.array([0.1, 0.2, 0.3, 0.4])
        result = vertical_4pt.project(ue, un, uz)
        np.testing.assert_array_equal(result, uz)

    def test_project_output_length(self, vertical_4pt):
        n = vertical_4pt.n_stations
        result = vertical_4pt.project(np.ones(n), np.ones(n), np.ones(n))
        assert result.shape == (vertical_4pt.n_obs,)


# ======================================================================
# 8. Covariance matrix
# ======================================================================

class TestCovariance:
    """Test default and explicit covariance matrices."""

    def test_default_diagonal_gnss(self, gnss_3station):
        cov = gnss_3station.covariance
        assert cov.shape == (9, 9)
        expected_diag = gnss_3station.sigma ** 2
        np.testing.assert_allclose(np.diag(cov), expected_diag)
        # Off-diagonal should be zero
        np.testing.assert_array_equal(cov - np.diag(np.diag(cov)), 0.0)

    def test_default_diagonal_insar(self, insar_5pixel):
        cov = insar_5pixel.covariance
        assert cov.shape == (5, 5)
        np.testing.assert_allclose(np.diag(cov), insar_5pixel.sigma ** 2)

    def test_explicit_covariance(self):
        """User-provided covariance matrix overrides sigma-based default."""
        lat = np.array([0.0, 1.0])
        lon = np.array([100.0, 101.0])
        disp = np.array([0.01, 0.02])
        sigma = np.array([0.003, 0.003])
        cov = np.array([[9e-6, 1e-6], [1e-6, 9e-6]])
        v = Vertical(lat, lon, disp, sigma, covariance=cov)
        np.testing.assert_array_equal(v.covariance, cov)

    def test_covariance_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            Vertical(
                np.array([0.0]),
                np.array([100.0]),
                np.array([0.01]),
                np.array([0.003]),
                covariance=np.eye(5),  # wrong shape for 1 obs
            )

    def test_covariance_cached(self, vertical_4pt):
        """Covariance should be computed once and cached."""
        cov1 = vertical_4pt.covariance
        cov2 = vertical_4pt.covariance
        assert cov1 is cov2


# ======================================================================
# 9. GNSS file I/O
# ======================================================================

class TestGNSSLoad:
    """Test GNSS.load() for .dat format."""

    def test_load_dat_format(self, tmp_path):
        """Load a .dat file with standard columns."""
        dat_file = tmp_path / "test.dat"
        dat_file.write_text(
            "# lon lat hgt uE uN uZ sigE sigN sigZ\n"
            "100.0 0.0 0.0 1.0 0.5 -0.1 0.1 0.1 0.5\n"
            "101.0 1.0 0.0 2.0 1.5 -0.2 0.2 0.2 0.5\n"
        )
        g = GNSS.load(dat_file)
        assert g.n_stations == 2
        assert g.n_obs == 6
        np.testing.assert_allclose(g.lat, [0.0, 1.0])
        np.testing.assert_allclose(g.lon, [100.0, 101.0])

    def test_load_dat_horizontal_only(self, tmp_path):
        """Load .dat file and select only horizontal components."""
        dat_file = tmp_path / "test.dat"
        dat_file.write_text(
            "# lon lat hgt uE uN uZ sigE sigN sigZ\n"
            "100.0 0.0 0.0 1.0 0.5 -0.1 0.1 0.1 0.5\n"
        )
        g = GNSS.load(dat_file, components="en")
        assert g.n_obs == 2
        assert g.components == "en"

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            GNSS.load("/nonexistent/file.dat")


# ======================================================================
# 10. InSAR file I/O
# ======================================================================

class TestInSARLoad:
    """Test InSAR.load() for .dat format."""

    def test_load_dat_format(self, tmp_path):
        dat_file = tmp_path / "test.dat"
        dat_file.write_text(
            "# lon lat hgt uLOS sigLOS losE losN losU\n"
            "100.0 0.0 0.0 0.01 0.005 0.38 -0.09 0.92\n"
            "100.0 1.0 0.0 0.02 0.005 0.38 -0.09 0.92\n"
        )
        d = InSAR.load(dat_file)
        assert d.n_stations == 2
        assert d.n_obs == 2
        np.testing.assert_allclose(d.lat, [0.0, 1.0])

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            InSAR.load("/nonexistent/file.dat")


# ======================================================================
# 11. Vertical file I/O
# ======================================================================

class TestVerticalLoad:
    """Test Vertical.load() for .dat format."""

    def test_load_dat_format(self, tmp_path):
        dat_file = tmp_path / "test.dat"
        dat_file.write_text(
            "# lon lat hgt uZ sigZ\n"
            "100.0 0.0 0.0 0.01 0.003\n"
            "100.0 1.0 0.0 0.02 0.003\n"
        )
        d = Vertical.load(dat_file)
        assert d.n_stations == 2
        assert d.n_obs == 2
        np.testing.assert_allclose(d.obs, [0.01, 0.02])

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            Vertical.load("/nonexistent/file.dat")
