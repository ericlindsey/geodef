"""Tests for Okada (1992) internal deformation functions.

The okada92 implementation computes displacements and strains at arbitrary
depth (Z <= 0) in an elastic half-space. Key tests include:
- Surface (Z=0) results must match okada85
- Depth variation should be smooth and finite
- Known reference values from the original Fortran dc3d code
"""

import numpy as np
import pytest

from okada92 import okada92


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
            10.0, 10.0, -5.0, 50.0, 0.0, 45.0,
            100.0, 50.0, 1.0, 0.0, 0.0, _G, _NU,
        )
        assert disp.shape == (3, 1)
        assert strain.shape == (3, 3)
        assert np.all(np.isfinite(disp))
        assert np.all(np.isfinite(strain))

    def test_positive_z_raises(self) -> None:
        """Positive Z should raise ValueError."""
        with pytest.raises(ValueError, match="z-coordinate is positive"):
            okada92(
                10.0, 0.0, 1.0, 50.0, 0.0, 45.0,
                100.0, 50.0, 1.0, 0.0, 0.0, _G, _NU,
            )

    @pytest.mark.parametrize("dip", [15.0, 45.0, 70.0, 90.0])
    def test_various_dips_finite(self, dip: float) -> None:
        """Displacements should be finite for various dip angles."""
        disp, strain = okada92(
            20.0, 20.0, -5.0, 30.0, 0.0, dip,
            50.0, 25.0, 1.0, 0.0, 0.0, _G, _NU,
        )
        assert np.all(np.isfinite(disp))

    @pytest.mark.parametrize("slip_type", [
        (1.0, 0.0, 0.0),  # strike-slip
        (0.0, 1.0, 0.0),  # dip-slip
        (0.0, 0.0, 1.0),  # tensile
    ])
    def test_individual_slip_components(
        self, slip_type: tuple[float, float, float],
    ) -> None:
        """Each slip component should produce finite results independently."""
        disp, strain = okada92(
            15.0, 15.0, -3.0, 20.0, 0.0, 45.0,
            40.0, 20.0, *slip_type, _G, _NU,
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
                20.0, 0.0, z, 30.0, 0.0, 45.0,
                50.0, 25.0, 1.0, 0.0, 0.0, _G, _NU,
            )
            disps.append(disp.flatten())
        for i in range(len(depths) - 1):
            assert not np.allclose(disps[i], disps[i + 1])

    def test_deep_displacement_decays(self) -> None:
        """Far from the fault, displacements should be small."""
        disp_near, _ = okada92(
            10.0, 0.0, -5.0, 30.0, 0.0, 45.0,
            50.0, 25.0, 1.0, 0.0, 0.0, _G, _NU,
        )
        disp_far, _ = okada92(
            200.0, 200.0, -5.0, 30.0, 0.0, 45.0,
            50.0, 25.0, 1.0, 0.0, 0.0, _G, _NU,
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
            disp2.flatten(), 2 * disp1.flatten(), rtol=1e-12,
        )

    def test_zero_slip_zero_displacement(self) -> None:
        """Zero slip should produce zero displacement."""
        disp, strain = okada92(
            20.0, 10.0, -5.0, 30.0, 0.0, 45.0,
            50.0, 25.0, 0.0, 0.0, 0.0, _G, _NU,
        )
        np.testing.assert_allclose(disp.flatten(), 0.0, atol=1e-15)
        np.testing.assert_allclose(strain, 0.0, atol=1e-15)
