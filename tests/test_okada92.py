"""Tests for Okada (1992) internal deformation functions.

The okada92 implementation computes displacements and strains at arbitrary
depth (Z <= 0) in an elastic half-space. Key tests include:
- Surface (Z=0) results must match okada85
- Depth variation should be smooth and finite
- Known reference values from the original Fortran dc3d code
"""

from pathlib import Path

import numpy as np
import pytest

from geodef.okada92 import okada92

# Standard elastic parameters
_G = 30.0
_NU = 0.25


def _alpha_from_G_nu(G: float, nu: float) -> float:
    """Compute Okada's alpha parameter from shear modulus and Poisson's ratio."""
    lam = 2 * G * nu / (1 - 2 * nu)
    return (lam + G) / (lam + 2 * G)


class TestOkada92Basic:
    """Basic functionality tests for okada92."""

    def test_returns_displacement_and_strain(self) -> None:
        """okada92 should return displacement array and strain matrix."""
        disp, strain = okada92(
            10.0,
            10.0,
            -5.0,
            50.0,
            0.0,
            45.0,
            100.0,
            50.0,
            1.0,
            0.0,
            0.0,
            _G,
            _NU,
        )
        assert disp.shape == (3, 1)
        assert strain.shape == (3, 3)
        assert np.all(np.isfinite(disp))
        assert np.all(np.isfinite(strain))

    def test_positive_z_raises(self) -> None:
        """Positive Z should raise ValueError."""
        with pytest.raises(ValueError, match="z-coordinate is positive"):
            okada92(
                10.0,
                0.0,
                1.0,
                50.0,
                0.0,
                45.0,
                100.0,
                50.0,
                1.0,
                0.0,
                0.0,
                _G,
                _NU,
            )

    @pytest.mark.parametrize("dip", [15.0, 45.0, 70.0, 90.0])
    def test_various_dips_finite(self, dip: float) -> None:
        """Displacements should be finite for various dip angles."""
        disp, strain = okada92(
            20.0,
            20.0,
            -5.0,
            30.0,
            0.0,
            dip,
            50.0,
            25.0,
            1.0,
            0.0,
            0.0,
            _G,
            _NU,
        )
        assert np.all(np.isfinite(disp))

    @pytest.mark.parametrize(
        "slip_type",
        [
            (1.0, 0.0, 0.0),  # strike-slip
            (0.0, 1.0, 0.0),  # dip-slip
            (0.0, 0.0, 1.0),  # tensile
        ],
    )
    def test_individual_slip_components(
        self,
        slip_type: tuple[float, float, float],
    ) -> None:
        """Each slip component should produce finite results independently."""
        disp, strain = okada92(
            15.0,
            15.0,
            -3.0,
            20.0,
            0.0,
            45.0,
            40.0,
            20.0,
            *slip_type,
            _G,
            _NU,
        )
        assert np.all(np.isfinite(disp))
        assert np.all(np.isfinite(strain))


class TestOkada92DepthVariation:
    """Test that deformation varies smoothly with depth."""

    def test_displacement_varies_with_depth(self) -> None:
        """Displacements at different depths should differ."""
        depths = [-1.0, -5.0, -10.0, -20.0]
        disps = []
        for z in depths:
            disp, _ = okada92(
                20.0,
                0.0,
                z,
                30.0,
                0.0,
                45.0,
                50.0,
                25.0,
                1.0,
                0.0,
                0.0,
                _G,
                _NU,
            )
            disps.append(disp.flatten())
        for i in range(len(depths) - 1):
            assert not np.allclose(disps[i], disps[i + 1])

    def test_deep_displacement_decays(self) -> None:
        """Far from the fault, displacements should be small."""
        disp_near, _ = okada92(
            10.0,
            0.0,
            -5.0,
            30.0,
            0.0,
            45.0,
            50.0,
            25.0,
            1.0,
            0.0,
            0.0,
            _G,
            _NU,
        )
        disp_far, _ = okada92(
            200.0,
            200.0,
            -5.0,
            30.0,
            0.0,
            45.0,
            50.0,
            25.0,
            1.0,
            0.0,
            0.0,
            _G,
            _NU,
        )
        mag_near = np.linalg.norm(disp_near)
        mag_far = np.linalg.norm(disp_far)
        assert mag_near > mag_far


class TestOkada92Linearity:
    """Test linearity of the elastic solution."""

    def test_doubling_slip_doubles_displacement(self) -> None:
        """Linear elasticity: 2x slip = 2x displacement."""
        args = (20.0, 10.0, -5.0, 30.0, 0.0, 45.0, 50.0, 25.0)
        disp1, _ = okada92(*args, 1.0, 0.0, 0.0, _G, _NU)
        disp2, _ = okada92(*args, 2.0, 0.0, 0.0, _G, _NU)
        np.testing.assert_allclose(
            disp2.flatten(),
            2 * disp1.flatten(),
            rtol=1e-12,
        )

    def test_zero_slip_zero_displacement(self) -> None:
        """Zero slip should produce zero displacement."""
        disp, strain = okada92(
            20.0,
            10.0,
            -5.0,
            30.0,
            0.0,
            45.0,
            50.0,
            25.0,
            0.0,
            0.0,
            0.0,
            _G,
            _NU,
        )
        np.testing.assert_allclose(disp.flatten(), 0.0, atol=1e-15)
        np.testing.assert_allclose(strain, 0.0, atol=1e-15)

    def test_slip_components_superpose(self) -> None:
        """Combined slip must equal the sum of the individual components.

        Guards against cross-contamination between the strike, dip, and
        tensile DU blocks (the pre-vectorization port reused a stale DU
        buffer across blocks with the tensile entries truncated).
        """
        args = (20.0, 10.0, -5.0, 30.0, 0.0, 45.0, 50.0, 25.0)
        disp_ss, strain_ss = okada92(*args, 1.0, 0.0, 0.0, _G, _NU)
        disp_ds, strain_ds = okada92(*args, 0.0, 1.0, 0.0, _G, _NU)
        disp_tf, strain_tf = okada92(*args, 0.0, 0.0, 1.0, _G, _NU)
        disp_all, strain_all = okada92(*args, 1.0, 1.0, 1.0, _G, _NU)
        np.testing.assert_allclose(
            disp_all, disp_ss + disp_ds + disp_tf, rtol=1e-12, atol=1e-18
        )
        np.testing.assert_allclose(
            strain_all, strain_ss + strain_ds + strain_tf, rtol=1e-12, atol=1e-18
        )


# ======================================================================
# Vectorized evaluation
# ======================================================================


_GOLDEN_PATH = Path(__file__).parent / "reference_data" / "okada92_scalar_golden.npz"


class TestOkada92Vectorized:
    """Array-input evaluation matches the scalar path and legacy results."""

    def test_array_input_shapes(self) -> None:
        n = 7
        X = np.linspace(10.0, 50.0, n)
        Y = np.linspace(-20.0, 20.0, n)
        Z = np.linspace(-1.0, -15.0, n)
        disp, strain = okada92(
            X, Y, Z, 30.0, 10.0, 45.0, 50.0, 25.0, 1.0, 0.5, 0.2, _G, _NU
        )
        assert disp.shape == (n, 3)
        assert strain.shape == (n, 3, 3)
        assert np.all(np.isfinite(disp))
        assert np.all(np.isfinite(strain))

    def test_array_matches_scalar_calls(self) -> None:
        rng = np.random.default_rng(5)
        n = 25
        X = rng.uniform(-100.0, 100.0, n)
        Y = rng.uniform(-100.0, 100.0, n)
        Z = -rng.uniform(0.0, 40.0, n)
        params = (30.0, 25.0, 60.0, 50.0, 25.0, 1.0, -0.5, 0.3, _G, _NU)

        disp, strain = okada92(X, Y, Z, *params)
        for i in range(n):
            d_i, s_i = okada92(float(X[i]), float(Y[i]), float(Z[i]), *params)
            np.testing.assert_allclose(disp[i], d_i[:, 0], rtol=1e-13, atol=1e-20)
            np.testing.assert_allclose(strain[i], s_i, rtol=1e-13, atol=1e-20)

    @pytest.mark.parametrize("geometry", ["dipping", "vertical", "shallow_dip"])
    @pytest.mark.parametrize("slip", ["ss", "ds"])
    def test_matches_scalar_port_golden(self, geometry: str, slip: str) -> None:
        """Shear results match golden data from the pre-vectorization port."""
        ref = dict(np.load(str(_GOLDEN_PATH)))
        slip_args = {"ss": (1.0, 0.0), "ds": (0.0, 1.0)}[slip]
        disp, strain = okada92(
            ref["X"],
            ref["Y"],
            ref["Z"],
            float(ref[f"{geometry}_depth"]),
            float(ref[f"{geometry}_strike"]),
            float(ref[f"{geometry}_dip"]),
            float(ref[f"{geometry}_length"]),
            float(ref[f"{geometry}_width"]),
            *slip_args,
            0.0,
            30e9,
            0.25,
            allow_singular=True,
        )
        np.testing.assert_allclose(
            disp, ref[f"{geometry}_{slip}_disp"], rtol=1e-12, atol=1e-20
        )
        np.testing.assert_allclose(
            strain, ref[f"{geometry}_{slip}_strain"], rtol=1e-12, atol=1e-22
        )


class TestOkada92StrainConsistency:
    """The strain output must be the gradient of the displacement field."""

    @pytest.mark.parametrize(
        "slip",
        [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        ids=["strike_slip", "dip_slip", "tensile"],
    )
    def test_strain_matches_finite_differences(
        self, slip: tuple[float, float, float]
    ) -> None:
        """Central differences of displacement reproduce the strain tensor.

        This validates every displacement-gradient component, including the
        tensile z-derivative entries that were truncated in the scalar port.
        """
        params = (12e3, 37.0, 55.0, 15e3, 8e3, *slip, 30e9, 0.25)
        pts = np.array(
            [
                [6e3, 4e3, -2e3],
                [-5e3, 9e3, -9e3],
                [11e3, -7e3, -16e3],
                [1e3, 2e3, -25e3],
            ]
        )
        h = 0.5

        _, strain = okada92(pts[:, 0], pts[:, 1], pts[:, 2], *params)

        for axis in range(3):
            step = np.zeros(3)
            step[axis] = h
            plus = pts + step
            minus = pts - step
            disp_p, _ = okada92(plus[:, 0], plus[:, 1], plus[:, 2], *params)
            disp_m, _ = okada92(minus[:, 0], minus[:, 1], minus[:, 2], *params)
            fd = (disp_p - disp_m) / (2 * h)
            np.testing.assert_allclose(
                strain[:, axis, :],
                fd,
                rtol=2e-5,
                atol=1e-13,
                err_msg=f"gradient axis {axis}",
            )
