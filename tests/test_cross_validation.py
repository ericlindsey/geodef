"""Cross-validation tests between okada85, okada92 (DC3D), and tdcalc.

These tests verify mutual consistency of the three Green's function
implementations by comparing results across different methods for
equivalent geometries:
- okada85 vs DC3D at the surface (Z=0) -- validates the core Okada92 engine
- okada85 vs okada92 wrapper at the surface -- known to fail (wrapper bug)
- tdcalc vs okada85 at the surface (rectangle = two triangles)
- tdcalc vs DC3D at depth (rectangle = two triangles)

SIGN CONVENTION NOTE:
    The dip-slip component has opposite sign convention between Okada and
    Nikkhoo/tdcalc. Positive dip-slip in tdcalc produces the opposite
    displacement to positive dip-slip (DISL2) in Okada. The magnitude is
    identical. This is accounted for in cross-validation by negating the
    tdcalc dip-slip component when comparing to Okada. The geodef unified
    library will standardize this convention.
"""

import numpy as np
import pytest

import okada85
from okada92 import DC3D, DCCON0, okada92
import tdcalc

_G = 30.0
_NU = 0.25
_ALPHA = 2 / 3  # alpha for nu=0.25


def _okada_slip_to_tdcalc(disl: tuple[float, float, float]) -> np.ndarray:
    """Convert Okada (DISL1, DISL2, DISL3) to tdcalc slip convention.

    Negates the dip-slip component to account for the sign convention
    difference between the two codes.

    Args:
        disl: Okada slip components (strike-slip, dip-slip, tensile).

    Returns:
        tdcalc slip array [strike-slip, -dip-slip, tensile].
    """
    return np.array([disl[0], -disl[1], disl[2]])


def _rect_to_triangles(
    e_center: float, n_center: float, depth: float,
    strike: float, dip: float, L: float, W: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a rectangular fault (centered at given point) into two triangles.

    Creates two right triangles that together form the rectangle, with
    vertices in the Earth-fixed coordinate system (East, North, Up).

    Args:
        e_center: Easting of fault centroid.
        n_center: Northing of fault centroid.
        depth: Depth of fault centroid (positive down).
        strike: Strike angle in degrees from North.
        dip: Dip angle in degrees from horizontal.
        L: Along-strike length.
        W: Down-dip width.

    Returns:
        Tuple of (tri1, tri2), each shape (3,3) with rows [E, N, Z].
    """
    strike_rad = np.radians(strike)
    dip_rad = np.radians(dip)

    s_e = np.sin(strike_rad)
    s_n = np.cos(strike_rad)
    d_e = np.cos(strike_rad) * np.cos(dip_rad)
    d_n = -np.sin(strike_rad) * np.cos(dip_rad)
    d_z = np.sin(dip_rad)

    corners = np.zeros((4, 3))
    for i, (along_strike, down_dip) in enumerate([
        (-L / 2, -W / 2),
        (L / 2, -W / 2),
        (-L / 2, W / 2),
        (L / 2, W / 2),
    ]):
        corners[i, 0] = e_center + along_strike * s_e + down_dip * d_e
        corners[i, 1] = n_center + along_strike * s_n + down_dip * d_n
        corners[i, 2] = -(depth + down_dip * d_z)

    tri1 = corners[[0, 1, 2]]
    tri2 = corners[[1, 3, 2]]
    return tri1, tri2


def _okada85_to_dc3d_coords(
    e: float, n: float, depth: float, obs_z: float,
    strike: float, dip: float, L: float, W: float,
) -> tuple[float, float, float, float]:
    """Convert okada85 centroid-relative coords to DC3D internal coordinates.

    Uses the same transform as okada85.setup_args to get the Okada
    fault-system coordinates, then maps them to DC3D's parameterization.

    Args:
        e: Easting relative to fault centroid.
        n: Northing relative to fault centroid.
        depth: Depth of fault centroid (positive down).
        obs_z: Observation depth (Z <= 0, with 0 at surface).
        strike: Strike in degrees.
        dip: Dip in degrees.
        L: Along-strike length.
        W: Down-dip width.

    Returns:
        Tuple of (X, Y, Z, DEPTH) for DC3D.
    """
    strike_rad = np.radians(strike)
    dip_rad = np.radians(dip)
    cs = np.cos(strike_rad)
    ss = np.sin(strike_rad)
    cd = np.cos(dip_rad)
    sd = np.sin(dip_rad)

    d = depth + sd * W / 2

    ec = e + cs * cd * W / 2
    nc = n - ss * cd * W / 2
    x = cs * nc + ss * ec + L / 2
    y = ss * nc - cs * ec + cd * W

    return float(x), float(y), obs_z, float(d)


def _dc3d_to_geographic(
    ux: float, uy: float, uz: float, strike: float,
) -> tuple[float, float, float]:
    """Rotate DC3D fault-system displacements to geographic (E, N, Up).

    Args:
        ux: Displacement along strike (DC3D x-axis).
        uy: Displacement perpendicular to strike (DC3D y-axis).
        uz: Vertical displacement.
        strike: Strike angle in degrees.

    Returns:
        Tuple of (ue, un, uz) in geographic coordinates.
    """
    strike_rad = np.radians(strike)
    cs = np.cos(strike_rad)
    ss = np.sin(strike_rad)
    ue = ss * ux - cs * uy
    un = cs * ux + ss * uy
    return ue, un, uz


class TestOkada85VsDC3DSurface:
    """DC3D at Z=0 should reproduce Okada85 exactly.

    This tests the core Okada92 engine (DC3D) against Okada85 by
    converting coordinates manually, bypassing the okada92() wrapper.
    """

    @pytest.mark.parametrize("geometry", [
        {"strike": 0.0, "dip": 45.0, "depth": 10.0, "L": 20.0, "W": 10.0},
        {"strike": 90.0, "dip": 70.0, "depth": 4.0, "L": 3.0, "W": 2.0},
        {"strike": 45.0, "dip": 15.0, "depth": 20.0, "L": 50.0, "W": 30.0},
        {"strike": 0.0, "dip": 90.0, "depth": 5.0, "L": 10.0, "W": 5.0},
    ], ids=["moderate_dip", "steep_dip", "shallow_dip", "vertical"])
    @pytest.mark.parametrize("slip_type,disl", [
        ("strike_slip", (1.0, 0.0, 0.0)),
        ("dip_slip", (0.0, 1.0, 0.0)),
        ("tensile", (0.0, 0.0, 1.0)),
    ])
    def test_dc3d_matches_okada85_at_surface(
        self,
        geometry: dict,
        slip_type: str,
        disl: tuple[float, float, float],
    ) -> None:
        """DC3D with correct coordinate mapping matches okada85 at Z=0."""
        s, d, depth, L, W = (
            geometry["strike"], geometry["dip"], geometry["depth"],
            geometry["L"], geometry["W"],
        )

        obs_points = [(5.0, 8.0), (-3.0, -5.0), (10.0, 0.0), (0.5, 15.0)]

        for e_obs, n_obs in obs_points:
            # okada85 with rake/slip convention
            rake = 0.0 if disl[0] != 0 else 90.0
            slip = disl[0] if disl[0] != 0 else disl[1]
            opening = disl[2]
            ue85, un85, uz85 = okada85.displacement(
                e_obs, n_obs, depth, s, d, L, W, rake, slip, opening, _NU,
            )

            # DC3D with internal coordinates
            X, Y, Z, DEPTH = _okada85_to_dc3d_coords(
                e_obs, n_obs, depth, 0.0, s, d, L, W,
            )
            disp, _, iret = DC3D(
                _ALPHA, X, Y, Z, DEPTH, d,
                0.0, L, 0.0, W, *disl,
            )
            assert iret == 0
            ue92, un92, uz92 = _dc3d_to_geographic(
                disp[0, 0], disp[1, 0], disp[2, 0], s,
            )

            np.testing.assert_allclose(
                [ue92, un92, uz92], [ue85, un85, uz85],
                rtol=1e-12, atol=1e-15,
                err_msg=f"DC3D/okada85 mismatch at ({e_obs}, {n_obs})",
            )


class TestOkada85VsOkada92Wrapper:
    """okada92() wrapper at Z=0 should reproduce okada85 exactly."""

    @pytest.mark.parametrize("geometry", [
        {"strike": 0.0, "dip": 45.0, "depth": 10.0, "L": 20.0, "W": 10.0},
        {"strike": 90.0, "dip": 70.0, "depth": 4.0, "L": 3.0, "W": 2.0},
        {"strike": 45.0, "dip": 15.0, "depth": 20.0, "L": 50.0, "W": 30.0},
        {"strike": 0.0, "dip": 90.0, "depth": 5.0, "L": 10.0, "W": 5.0},
    ], ids=["moderate_dip", "steep_dip", "shallow_dip", "vertical"])
    @pytest.mark.parametrize("slip_type,disl", [
        ("strike_slip", (1.0, 0.0, 0.0)),
        ("dip_slip", (0.0, 1.0, 0.0)),
        ("tensile", (0.0, 0.0, 1.0)),
    ])
    def test_surface_displacement_matches(
        self,
        geometry: dict,
        slip_type: str,
        disl: tuple[float, float, float],
    ) -> None:
        """okada92() wrapper at Z=0 should match okada85."""
        s, d, depth, L, W = (
            geometry["strike"], geometry["dip"], geometry["depth"],
            geometry["L"], geometry["W"],
        )

        obs_points = [(5.0, 8.0), (-3.0, -5.0), (10.0, 0.0), (0.5, 15.0)]

        for e_obs, n_obs in obs_points:
            rake = 0.0 if disl[0] != 0 else 90.0
            slip = disl[0] if disl[0] != 0 else disl[1]
            opening = disl[2]
            ue85, un85, uz85 = okada85.displacement(
                e_obs, n_obs, depth, s, d, L, W, rake, slip, opening, _NU,
            )
            disp92, _ = okada92(
                e_obs, n_obs, 0.0, depth, s, d, L, W,
                *disl, _G, _NU,
            )
            np.testing.assert_allclose(
                [disp92[0, 0], disp92[1, 0], disp92[2, 0]],
                [ue85, un85, uz85],
                rtol=1e-10, atol=1e-14,
                err_msg=f"okada92/okada85 mismatch at ({e_obs}, {n_obs})",
            )


class TestTDcalcVsOkada85Surface:
    """Two coplanar triangles forming a rectangle should match okada85."""

    @pytest.mark.parametrize("geometry", [
        {"strike": 0.0, "dip": 45.0, "depth": 10.0, "L": 20.0, "W": 10.0},
        {"strike": 0.0, "dip": 90.0, "depth": 8.0, "L": 10.0, "W": 5.0},
        {"strike": 90.0, "dip": 30.0, "depth": 15.0, "L": 30.0, "W": 20.0},
    ], ids=["moderate_dip", "vertical", "shallow_oblique"])
    @pytest.mark.parametrize("okada_disl", [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ], ids=["strike_slip", "dip_slip", "tensile"])
    def test_rectangle_equivalence_surface(
        self, geometry: dict, okada_disl: tuple[float, float, float],
    ) -> None:
        """Sum of two triangle displacements at surface should match okada85."""
        s, d, depth, L, W = (
            geometry["strike"], geometry["dip"], geometry["depth"],
            geometry["L"], geometry["W"],
        )

        tri1, tri2 = _rect_to_triangles(0.0, 0.0, depth, s, d, L, W)
        slip_td = _okada_slip_to_tdcalc(okada_disl)

        obs = np.array([
            [5.0, 10.0, 0.0],
            [-8.0, -3.0, 0.0],
            [15.0, 2.0, 0.0],
            [0.5, 20.0, 0.0],
        ])

        disp1 = tdcalc.TDdispHS(obs, tri1, slip_td, _NU)
        disp2 = tdcalc.TDdispHS(obs, tri2, slip_td, _NU)
        disp_td_total = disp1 + disp2

        # Map Okada disl to okada85 rake/slip/opening convention
        if okada_disl[0] != 0:
            rake, slip, opening = 0.0, okada_disl[0], 0.0
        elif okada_disl[1] != 0:
            rake, slip, opening = 90.0, okada_disl[1], 0.0
        else:
            rake, slip, opening = 0.0, 0.0, okada_disl[2]

        for i in range(len(obs)):
            ue85, un85, uz85 = okada85.displacement(
                obs[i, 0], obs[i, 1], depth, s, d, L, W,
                rake, slip, opening, _NU,
            )
            np.testing.assert_allclose(
                disp_td_total[i], [ue85, un85, uz85],
                rtol=1e-6, atol=1e-10,
                err_msg=f"Triangle/rectangle mismatch at obs {obs[i]}",
            )


class TestTDcalcVsDC3DDepth:
    """Two coplanar triangles should match DC3D at subsurface observation points."""

    @pytest.mark.parametrize("geometry", [
        {"strike": 0.0, "dip": 45.0, "depth": 15.0, "L": 20.0, "W": 10.0},
        {"strike": 0.0, "dip": 90.0, "depth": 10.0, "L": 10.0, "W": 5.0},
        {"strike": 90.0, "dip": 30.0, "depth": 20.0, "L": 30.0, "W": 15.0},
    ], ids=["moderate_dip", "vertical", "shallow_oblique"])
    @pytest.mark.parametrize("slip_type,disl", [
        ("strike_slip", (1.0, 0.0, 0.0)),
        ("dip_slip", (0.0, 1.0, 0.0)),
        ("tensile", (0.0, 0.0, 1.0)),
    ])
    def test_displacement_at_depth(
        self,
        geometry: dict,
        slip_type: str,
        disl: tuple[float, float, float],
    ) -> None:
        """Sum of two triangles at depth should match DC3D."""
        s, d, depth, L, W = (
            geometry["strike"], geometry["dip"], geometry["depth"],
            geometry["L"], geometry["W"],
        )

        tri1, tri2 = _rect_to_triangles(0.0, 0.0, depth, s, d, L, W)
        slip_td = _okada_slip_to_tdcalc(disl)

        obs_points = [
            (5.0, 10.0, -2.0),
            (-8.0, -3.0, -5.0),
            (15.0, 2.0, -1.0),
        ]

        for e_obs, n_obs, z_obs in obs_points:
            obs = np.array([[e_obs, n_obs, z_obs]])
            disp1 = tdcalc.TDdispHS(obs, tri1, slip_td, _NU)
            disp2 = tdcalc.TDdispHS(obs, tri2, slip_td, _NU)
            disp_td_total = (disp1 + disp2).flatten()

            X, Y, Z, DEPTH = _okada85_to_dc3d_coords(
                e_obs, n_obs, depth, z_obs, s, d, L, W,
            )
            disp92, _, iret = DC3D(
                _ALPHA, X, Y, Z, DEPTH, d,
                0.0, L, 0.0, W, *disl,
            )
            assert iret == 0
            ue92, un92, uz92 = _dc3d_to_geographic(
                disp92[0, 0], disp92[1, 0], disp92[2, 0], s,
            )

            np.testing.assert_allclose(
                disp_td_total, [ue92, un92, uz92],
                rtol=1e-6, atol=1e-10,
                err_msg=(
                    f"Triangle/DC3D mismatch at ({e_obs}, {n_obs}, {z_obs})"
                ),
            )

    @pytest.mark.parametrize("geometry", [
        {"strike": 0.0, "dip": 45.0, "depth": 15.0, "L": 20.0, "W": 10.0},
    ], ids=["moderate_dip"])
    def test_strain_at_depth(self, geometry: dict) -> None:
        """Sum of two triangle strains at depth should match DC3D strain."""
        s, d, depth, L, W = (
            geometry["strike"], geometry["dip"], geometry["depth"],
            geometry["L"], geometry["W"],
        )

        tri1, tri2 = _rect_to_triangles(0.0, 0.0, depth, s, d, L, W)

        e_obs, n_obs, z_obs = 5.0, 10.0, -2.0
        obs = np.array([[e_obs, n_obs, z_obs]])
        slip_td = np.array([1.0, 0.0, 0.0])

        strain1 = tdcalc.TDstrainHS(obs, tri1, slip_td, _NU)
        strain2 = tdcalc.TDstrainHS(obs, tri2, slip_td, _NU)
        strain_td = (strain1 + strain2).flatten()

        X, Y, Z, DEPTH = _okada85_to_dc3d_coords(
            e_obs, n_obs, depth, z_obs, s, d, L, W,
        )
        _, strain92_raw, iret = DC3D(
            _ALPHA, X, Y, Z, DEPTH, d,
            0.0, L, 0.0, W, 1.0, 0.0, 0.0,
        )
        assert iret == 0

        # DC3D strain is displacement gradient in fault coords (3x3)
        # Rotate to geographic before symmetrizing
        strike_rad = np.radians(s)
        cs = np.cos(strike_rad)
        ss = np.sin(strike_rad)
        R = np.array([[ss, -cs, 0], [cs, ss, 0], [0, 0, 1]])
        strain92_geo = R @ strain92_raw @ R.T

        strain92_sym = np.array([
            strain92_geo[0, 0],
            strain92_geo[1, 1],
            strain92_geo[2, 2],
            0.5 * (strain92_geo[0, 1] + strain92_geo[1, 0]),
            0.5 * (strain92_geo[0, 2] + strain92_geo[2, 0]),
            0.5 * (strain92_geo[1, 2] + strain92_geo[2, 1]),
        ])

        np.testing.assert_allclose(
            strain_td, strain92_sym,
            rtol=1e-5, atol=1e-10,
            err_msg="Triangle/DC3D strain mismatch at depth",
        )
