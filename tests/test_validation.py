"""Tests for geodef.validation — physical input validation and reports."""

import numpy as np
import pytest

from geodef.data import GNSS, InSAR, Vertical
from geodef.fault import Fault
from geodef.validation import (
    ValidationIssue,
    ValidationReport,
    as_1d_floats,
    check_covariance,
    check_positive,
    check_range,
)

# ====================================================================
# Low-level helpers
# ====================================================================


class TestAs1dFloats:
    def test_passes_through(self) -> None:
        arr = as_1d_floats("x", [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_rejects_2d(self) -> None:
        with pytest.raises(ValueError, match=r"x.*1-D.*\(2, 2\)"):
            as_1d_floats("x", np.zeros((2, 2)))

    def test_rejects_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="x.*length 3.*got 2"):
            as_1d_floats("x", [1.0, 2.0], n=3)

    def test_rejects_nan_naming_argument(self) -> None:
        with pytest.raises(ValueError, match="x.*non-finite"):
            as_1d_floats("x", [1.0, np.nan])

    def test_rejects_inf(self) -> None:
        with pytest.raises(ValueError, match="x.*non-finite"):
            as_1d_floats("x", [np.inf, 1.0])

    def test_unit_in_message(self) -> None:
        with pytest.raises(ValueError, match="meters"):
            as_1d_floats("depth", [np.nan], unit="meters")


class TestCheckRange:
    def test_in_range_ok(self) -> None:
        check_range("dip", np.array([0.0, 45.0, 90.0]), 0.0, 90.0, unit="degrees")

    def test_out_of_range_names_argument_and_unit(self) -> None:
        with pytest.raises(ValueError, match="dip.*degrees"):
            check_range("dip", np.array([120.0]), 0.0, 90.0, unit="degrees")

    def test_scalar_ok(self) -> None:
        check_range("lat", 45.0, -90.0, 90.0, unit="degrees")


class TestCheckPositive:
    def test_positive_ok(self) -> None:
        check_positive("sigma", np.array([0.1, 2.0]), unit="meters")

    def test_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="sigma.*positive"):
            check_positive("sigma", np.array([0.0, 1.0]), unit="meters")

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="length"):
            check_positive("length", -5.0, unit="meters")


class TestCheckCovariance:
    def _spd(self, n: int) -> np.ndarray:
        rng = np.random.default_rng(0)
        A = rng.normal(size=(n, n))
        return A @ A.T + n * np.eye(n)

    def test_spd_passes(self) -> None:
        check_covariance(self._spd(4), 4)

    def test_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match=r"covariance.*\(3, 3\)"):
            check_covariance(self._spd(4), 3)

    def test_asymmetric_rejected_with_magnitude(self) -> None:
        cov = self._spd(4)
        cov[0, 1] += 1.0
        with pytest.raises(ValueError, match="symmetric"):
            check_covariance(cov, 4)

    def test_not_positive_definite_rejected_with_remedy(self) -> None:
        cov = -np.eye(3)
        with pytest.raises(ValueError, match="positive.*definite"):
            check_covariance(cov, 3)

    def test_semidefinite_escape_hatch(self) -> None:
        cov = np.zeros((3, 3))  # PSD but not PD
        check_covariance(cov, 3, require_positive_definite=False)


# ====================================================================
# Report objects
# ====================================================================


class TestValidationReport:
    def test_ok_when_no_errors(self) -> None:
        rep = ValidationReport(
            issues=(ValidationIssue("warning", "sigma", "large spread"),)
        )
        assert rep.ok
        assert rep.n_warnings == 1
        assert rep.n_errors == 0

    def test_not_ok_with_error(self) -> None:
        rep = ValidationReport(
            issues=(ValidationIssue("error", "depth", "above surface"),)
        )
        assert not rep.ok

    def test_str_contains_fields(self) -> None:
        rep = ValidationReport(
            issues=(ValidationIssue("error", "depth", "above surface"),)
        )
        text = str(rep)
        assert "depth" in text and "above surface" in text

    def test_raise_if_errors(self) -> None:
        rep = ValidationReport(
            issues=(ValidationIssue("error", "depth", "above surface"),)
        )
        with pytest.raises(ValueError, match="above surface"):
            rep.raise_if_errors()


# ====================================================================
# Dataset constructor validation
# ====================================================================


def _gnss_kwargs(n: int = 3) -> dict:
    return dict(
        lon=np.linspace(100.0, 100.2, n),
        lat=np.linspace(0.0, 0.2, n),
        ve=np.zeros(n),
        vn=np.zeros(n),
        se=np.full(n, 0.001),
        sn=np.full(n, 0.001),
    )


class TestDatasetConstructorValidation:
    def test_gnss_rejects_nan_component(self) -> None:
        kw = _gnss_kwargs()
        kw["ve"] = np.array([0.0, np.nan, 0.0])
        with pytest.raises(ValueError, match="ve.*non-finite"):
            GNSS(**kw)

    def test_gnss_rejects_bad_latitude(self) -> None:
        kw = _gnss_kwargs()
        kw["lat"] = np.array([0.0, 95.0, 0.0])
        with pytest.raises(ValueError, match="lat.*degrees"):
            GNSS(**kw)

    def test_gnss_rejects_empty(self) -> None:
        kw = {k: np.array([]) for k in _gnss_kwargs()}
        with pytest.raises(ValueError, match="at least one"):
            GNSS(**kw)

    def test_vertical_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="displacement.*non-finite"):
            Vertical(
                lon=np.array([100.0]),
                lat=np.array([0.0]),
                displacement=np.array([np.nan]),
                sigma=np.array([0.01]),
            )

    def test_covariance_must_be_symmetric(self) -> None:
        n = 3
        cov = np.eye(3 * n)
        cov[0, 1] = 0.5  # asymmetric
        with pytest.raises(ValueError, match="symmetric"):
            GNSS(
                **_gnss_kwargs(n),
                vu=np.zeros(n),
                su=np.full(n, 0.001),
                covariance=cov,
            )

    def test_covariance_escape_hatch(self) -> None:
        n = 3
        cov = np.zeros((2 * n, 2 * n))  # PSD only
        gnss = GNSS(**_gnss_kwargs(n), covariance=cov, validate_covariance=False)
        assert gnss.n_obs == 2 * n


class TestInSARLookVectors:
    def _kwargs(self, look_e, look_n, look_u, **extra) -> dict:
        n = 2
        return dict(
            lon=np.array([100.0, 100.1]),
            lat=np.array([0.0, 0.1]),
            los=np.zeros(n),
            sigma=np.full(n, 0.005),
            look_e=np.full(n, look_e),
            look_n=np.full(n, look_n),
            look_u=np.full(n, look_u),
            **extra,
        )

    def test_unit_vector_accepted(self) -> None:
        InSAR(**self._kwargs(-0.38, -0.09, 0.92))

    def test_non_unit_rejected_with_norm_in_message(self) -> None:
        with pytest.raises(ValueError, match="look.*unit"):
            InSAR(**self._kwargs(-3.8, -0.9, 9.2))

    def test_normalize_option(self) -> None:
        insar = InSAR(**self._kwargs(-3.8, -0.9, 9.2, normalize_look=True))
        norms = np.sqrt(insar.look_e**2 + insar.look_n**2 + insar.look_u**2)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-12)

    def test_downward_look_flagged_in_validate(self) -> None:
        insar = InSAR(**self._kwargs(-0.38, -0.09, -0.92))
        report = insar.validate()
        assert any("look_u" in i.field for i in report.issues)


# ====================================================================
# Fault constructor validation
# ====================================================================


def _planar_kwargs(**over) -> dict:
    kw = dict(
        lat=0.0,
        lon=100.0,
        depth=15_000.0,
        strike=90.0,
        dip=30.0,
        length=40_000.0,
        width=20_000.0,
        n_length=2,
        n_width=2,
    )
    kw.update(over)
    return kw


class TestFaultConstructorValidation:
    def test_rejects_dip_over_90(self) -> None:
        with pytest.raises(ValueError, match="dip.*degrees"):
            Fault.planar(**_planar_kwargs(dip=120.0))

    def test_rejects_negative_length(self) -> None:
        with pytest.raises(ValueError, match="length.*positive"):
            Fault.planar(**_planar_kwargs(length=-1.0))

    def test_rejects_negative_depth(self) -> None:
        with pytest.raises(ValueError, match="depth"):
            Fault.planar(**_planar_kwargs(depth=-5_000.0))

    def test_rejects_nan_strike(self) -> None:
        with pytest.raises(ValueError, match="strike.*finite"):
            Fault.planar(**_planar_kwargs(strike=np.nan))


# ====================================================================
# validate() reports
# ====================================================================


class TestValidateReports:
    def test_healthy_fault_ok(self) -> None:
        fault = Fault.planar(**_planar_kwargs())
        report = fault.validate()
        assert report.ok

    def test_above_surface_patch_is_error(self) -> None:
        # centroid depth 100 m, width 20 km at dip 30 -> top edge above surface
        fault = Fault.planar(**_planar_kwargs(depth=100.0, n_width=1))
        report = fault.validate()
        assert not report.ok
        assert any("surface" in i.message for i in report.issues)

    def test_extreme_aspect_ratio_warns(self) -> None:
        fault = Fault.planar(
            **_planar_kwargs(length=500_000.0, width=1_000.0, depth=30_000.0)
        )
        report = fault.validate()
        assert any(i.severity == "warning" for i in report.issues)

    def test_healthy_gnss_ok(self) -> None:
        gnss = GNSS(**_gnss_kwargs())
        assert gnss.validate().ok

    def test_duplicate_stations_warn(self) -> None:
        kw = _gnss_kwargs()
        kw["lon"] = np.array([100.0, 100.0, 100.2])
        kw["lat"] = np.array([0.0, 0.0, 0.2])
        report = GNSS(**kw).validate()
        assert any("duplicate" in i.message.lower() for i in report.issues)

    def test_mesh_degenerate_triangle_error(self) -> None:
        from geodef.mesh import Mesh

        lon = np.array([100.0, 100.1, 100.05, 100.2])
        lat = np.array([0.0, 0.0, 0.0, 0.1])
        depth = np.array([0.0, 0.0, 0.0, 5000.0])
        # first triangle references node 1 twice -> exactly zero area
        triangles = np.array([[0, 1, 1], [1, 3, 2]])
        mesh = Mesh(lon=lon, lat=lat, depth=depth, triangles=triangles)
        report = mesh.validate()
        assert not report.ok
        assert any("degenerate" in i.message.lower() for i in report.issues)


class TestTransformsTerminateOnNaN:
    def test_ecef2geod_nan_returns_not_hangs(self) -> None:
        """NaN inputs must propagate as NaN, not spin the fixed-point loop."""
        from geodef.transforms import ecef2geod

        lat, lon, alt = ecef2geod(np.nan, np.nan, np.nan)
        assert np.isnan(float(np.asarray(lat)))
