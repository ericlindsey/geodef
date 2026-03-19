"""Tests for Okada (1985) surface displacement, tilt, and strain functions.

Reference values are from the original Matlab implementation by Beauducel (2012),
ported to Python by Lindsey (2014). The 9 test cases cover various dip angles
(70, 90), rake angles (0, 90, 180), and slip types (shear vs. tensile).
"""

import numpy as np
import pytest

import okada85

# -----------------------------------------------------------------------
# Original 9 reference test cases from the Matlab implementation.
# Each row is (ue, un, uz, -uee, -uen, -une, -unn, uze, uzn).
# These test displacement, strain, and tilt simultaneously.
# -----------------------------------------------------------------------
_REFERENCE_PARAMS = {
    "nu": 0.25,
    "L": 3.0,
    "W": 2.0,
    "strike": 90.0,
    "x": [2., 2., 2., 0., 0., 0., 0., 0., 0.],
    "y": [3., 3., 3., 0., 0., 0., 0., 0., 0.],
    "d": [4., 4., 4., 4., 4., 4., 6., 6., 6.],
    "dip": [70., 70., 70., 90., 90., 90., 90., 90., 90.],
    "rake": [0., 90., 0., 0., 90., 0., 180., 90., 180.],
    "slip": [1., 1., 0., 1., 1., 0., 1., 1., 0.],
    "u3": [0., 0., 1., 0., 0., 1., 0., 0., 1.],
}

_REFERENCE_RESULTS = [
    (-8.689165004256261e-03, -4.297582189741731e-03, -2.747405827638823e-03,
     -1.220438675268007e-03, 2.469697394431684e-04, -8.191372879334214e-03,
     -5.813975227325929e-04, -5.174968695670765e-03, 2.945389615109786e-04),
    (-4.682348762835457e-03, -3.526726796871769e-02, -3.563855767326866e-02,
     -8.867245527911540e-03, -1.518582321831328e-04, 4.056585617604535e-03,
     -1.035487654241981e-02, 4.088128489997486e-03, 2.626254787455854e-03),
    (-2.659960096441058e-04, 1.056407487698295e-02, 3.214193114221133e-03,
     -5.654954762127719e-04, 1.992743608041701e-03, -1.066213796012181e-03,
     1.229710985665510e-02, -3.730219351653117e-04, 1.040095554700254e-02),
    (0.000000000000000e+00, 5.253097376910021e-03, 0.000000000000000e+00,
     0.000000000000000e+00, -1.863722792616869e-02, -2.325128307637018e-03,
     0.000000000000000e+00, 0.000000000000000e+00, 2.288515018789224e-02),
    (0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00,
     0.000000000000000e+00, 2.747808530970814e-02, 0.000000000000000e+00,
     0.000000000000000e+00, 0.000000000000000e+00, -7.166487554729019e-02),
    (1.222848229982107e-02, 0.000000000000000e+00, -1.606274646428264e-02,
     -4.181651077400775e-03, 0.000000000000000e+00, 0.000000000000000e+00,
     -2.325128307637018e-03, -9.146107533038163e-03, 0.000000000000000e+00),
    (0.000000000000000e+00, -1.303097451419518e-03, 0.000000000000000e+00,
     0.000000000000000e+00, 2.726036397973775e-03, 7.345401310885204e-04,
     0.000000000000000e+00, 0.000000000000000e+00, -4.421657134187061e-03),
    (0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00,
     0.000000000000000e+00, 5.157341419851447e-03, 0.000000000000000e+00,
     0.000000000000000e+00, 0.000000000000000e+00, -1.901154674368966e-02),
    (3.506717844685072e-03, 0.000000000000000e+00, -7.740131086250291e-03,
     -1.770218903898004e-03, 0.000000000000000e+00, 0.000000000000000e+00,
     -7.345401310885204e-04, -1.842986424261335e-03, 0.000000000000000e+00),
]


def _centroid_to_okada_args(
    x: float, y: float, d: float, strike: float, dip: float,
    L: float, W: float,
) -> tuple[float, float, float]:
    """Convert centroid coordinates to Okada reference-point coordinates.

    The original test uses fault centroid coords; okada85 expects the
    reference point at the center of the top edge.

    Args:
        x: Easting of observation point.
        y: Northing of observation point.
        d: Depth of fault centroid.
        strike: Strike angle in degrees.
        dip: Dip angle in degrees.
        L: Along-strike length.
        W: Down-dip width.

    Returns:
        Tuple of (e, n, depth) for okada85 functions.
    """
    dip_rad = dip * np.pi / 180
    e = x - L / 2
    n = y - np.cos(dip_rad) * W / 2
    depth = d - np.sin(dip_rad) * W / 2
    return e, n, depth


@pytest.mark.parametrize("case_idx", range(9), ids=[f"case_{i}" for i in range(9)])
class TestOkada85Reference:
    """Test against the 9 original Matlab/Python reference cases."""

    def test_displacement(self, case_idx: int) -> None:
        """Verify displacement components match reference to 15 decimal places."""
        p = _REFERENCE_PARAMS
        e, n, depth = _centroid_to_okada_args(
            p["x"][case_idx], p["y"][case_idx], p["d"][case_idx],
            p["strike"], p["dip"][case_idx], p["L"], p["W"],
        )
        ue, un, uz = okada85.displacement(
            e, n, depth, p["strike"], p["dip"][case_idx],
            p["L"], p["W"], p["rake"][case_idx],
            p["slip"][case_idx], p["u3"][case_idx], p["nu"],
        )
        ref = _REFERENCE_RESULTS[case_idx]
        np.testing.assert_almost_equal(ue, ref[0], decimal=15)
        np.testing.assert_almost_equal(un, ref[1], decimal=15)
        np.testing.assert_almost_equal(uz, ref[2], decimal=15)

    def test_tilt(self, case_idx: int) -> None:
        """Verify tilt components match reference to 15 decimal places."""
        p = _REFERENCE_PARAMS
        e, n, depth = _centroid_to_okada_args(
            p["x"][case_idx], p["y"][case_idx], p["d"][case_idx],
            p["strike"], p["dip"][case_idx], p["L"], p["W"],
        )
        uze, uzn = okada85.tilt(
            e, n, depth, p["strike"], p["dip"][case_idx],
            p["L"], p["W"], p["rake"][case_idx],
            p["slip"][case_idx], p["u3"][case_idx], p["nu"],
        )
        ref = _REFERENCE_RESULTS[case_idx]
        np.testing.assert_almost_equal(uze, ref[7], decimal=15)
        np.testing.assert_almost_equal(uzn, ref[8], decimal=15)

    def test_strain(self, case_idx: int) -> None:
        """Verify strain components match reference to 15 decimal places."""
        p = _REFERENCE_PARAMS
        e, n, depth = _centroid_to_okada_args(
            p["x"][case_idx], p["y"][case_idx], p["d"][case_idx],
            p["strike"], p["dip"][case_idx], p["L"], p["W"],
        )
        unn, une, uen, uee = okada85.strain(
            e, n, depth, p["strike"], p["dip"][case_idx],
            p["L"], p["W"], p["rake"][case_idx],
            p["slip"][case_idx], p["u3"][case_idx], p["nu"],
        )
        ref = _REFERENCE_RESULTS[case_idx]
        np.testing.assert_almost_equal(-uee, ref[3], decimal=15)
        np.testing.assert_almost_equal(-uen, ref[4], decimal=15)
        np.testing.assert_almost_equal(-une, ref[5], decimal=15)
        np.testing.assert_almost_equal(-unn, ref[6], decimal=15)


class TestOkada85Geometry:
    """Test various fault geometries and slip types."""

    @pytest.mark.parametrize("dip", [15.0, 45.0, 70.0, 90.0])
    def test_pure_strike_slip(self, dip: float) -> None:
        """Pure strike-slip (rake=0) produces finite displacements."""
        ue, un, uz = okada85.displacement(
            5.0, 5.0, 10.0, 0.0, dip, 10.0, 5.0, 0.0, 1.0, 0.0,
        )
        assert np.isfinite(ue) and np.isfinite(un) and np.isfinite(uz)

    @pytest.mark.parametrize("dip", [15.0, 45.0, 70.0, 90.0])
    def test_pure_dip_slip(self, dip: float) -> None:
        """Pure dip-slip (rake=90) produces finite displacements."""
        ue, un, uz = okada85.displacement(
            5.0, 5.0, 10.0, 0.0, dip, 10.0, 5.0, 90.0, 1.0, 0.0,
        )
        assert np.isfinite(ue) and np.isfinite(un) and np.isfinite(uz)

    @pytest.mark.parametrize("dip", [15.0, 45.0, 70.0, 90.0])
    def test_pure_tensile(self, dip: float) -> None:
        """Pure tensile opening produces finite displacements."""
        ue, un, uz = okada85.displacement(
            5.0, 5.0, 10.0, 0.0, dip, 10.0, 5.0, 0.0, 0.0, 1.0,
        )
        assert np.isfinite(ue) and np.isfinite(un) and np.isfinite(uz)


class TestOkada85Symmetry:
    """Test physical symmetry properties of fault solutions."""

    def test_vertical_strike_slip_antisymmetry(self) -> None:
        """Vertical strike-slip: ue is antisymmetric across the fault trace.

        For a N-S striking vertical fault at x=0, strike-slip motion
        should produce ue(+y) = -ue(-y) along the perpendicular axis.
        """
        y_pos = 5.0
        y_neg = -5.0
        ue_pos, _, _ = okada85.displacement(
            y_pos, 0.0, 10.0, 0.0, 90.0, 20.0, 10.0, 0.0, 1.0, 0.0,
        )
        ue_neg, _, _ = okada85.displacement(
            y_neg, 0.0, 10.0, 0.0, 90.0, 20.0, 10.0, 0.0, 1.0, 0.0,
        )
        np.testing.assert_almost_equal(ue_pos, -ue_neg, decimal=14)

    def test_zero_slip_gives_zero_displacement(self) -> None:
        """Zero slip and zero opening should give zero displacements."""
        ue, un, uz = okada85.displacement(
            5.0, 5.0, 10.0, 45.0, 30.0, 10.0, 5.0, 0.0, 0.0, 0.0,
        )
        np.testing.assert_almost_equal(ue, 0.0, decimal=15)
        np.testing.assert_almost_equal(un, 0.0, decimal=15)
        np.testing.assert_almost_equal(uz, 0.0, decimal=15)

    def test_linearity(self) -> None:
        """Doubling slip should double displacements (linear elasticity)."""
        args = (5.0, 5.0, 10.0, 45.0, 30.0, 10.0, 5.0, 0.0)
        ue1, un1, uz1 = okada85.displacement(*args, 1.0, 0.0)
        ue2, un2, uz2 = okada85.displacement(*args, 2.0, 0.0)
        np.testing.assert_almost_equal(ue2, 2 * ue1, decimal=14)
        np.testing.assert_almost_equal(un2, 2 * un1, decimal=14)
        np.testing.assert_almost_equal(uz2, 2 * uz1, decimal=14)


class TestOkada85FarField:
    """Test far-field behavior of fault solutions."""

    def test_displacement_decays_with_distance(self) -> None:
        """Displacement magnitude decreases with distance from fault."""
        args = (10.0, 45.0, 30.0, 10.0, 5.0, 0.0, 1.0, 0.0)
        dists = [10.0, 50.0, 100.0]
        magnitudes = []
        for r in dists:
            ue, un, uz = okada85.displacement(r, r, *args)
            magnitudes.append(np.sqrt(ue**2 + un**2 + uz**2))
        assert magnitudes[0] > magnitudes[1] > magnitudes[2]


class TestOkada85Vectorized:
    """Test that okada85 handles array inputs correctly."""

    def test_array_input(self) -> None:
        """Passing arrays of observation points should work."""
        e = np.array([1.0, 2.0, 3.0])
        n = np.array([1.0, 2.0, 3.0])
        ue, un, uz = okada85.displacement(
            e, n, 10.0, 0.0, 45.0, 10.0, 5.0, 0.0, 1.0, 0.0,
        )
        assert ue.shape == (3,)
        assert un.shape == (3,)
        assert uz.shape == (3,)

    def test_array_matches_scalar(self) -> None:
        """Array results should match individual scalar calls."""
        e = np.array([1.0, 5.0, 10.0])
        n = np.array([2.0, 6.0, 11.0])
        ue_arr, un_arr, uz_arr = okada85.displacement(
            e, n, 10.0, 0.0, 45.0, 10.0, 5.0, 0.0, 1.0, 0.0,
        )
        for i in range(3):
            ue_s, un_s, uz_s = okada85.displacement(
                e[i], n[i], 10.0, 0.0, 45.0, 10.0, 5.0, 0.0, 1.0, 0.0,
            )
            np.testing.assert_almost_equal(ue_arr[i], ue_s, decimal=15)
            np.testing.assert_almost_equal(un_arr[i], un_s, decimal=15)
            np.testing.assert_almost_equal(uz_arr[i], uz_s, decimal=15)
