"""Tests for geodef.greens (Laplacian operators and Green's matrix utilities).

Laplacian tests migrated from related/shakeout_v2/test_laplacian.py.
"""

import numpy as np
import pytest

from geodef.greens import build_laplacian_2d, build_laplacian_2d_simple


# ---------------------------------------------------------------------------
# Laplacian (forward/backward difference boundaries)
# ---------------------------------------------------------------------------

class TestLaplacian2D:
    """Tests for build_laplacian_2d."""

    def test_matrix_shape(self):
        nL, nW = 5, 4
        L = build_laplacian_2d(nL, nW)
        assert L.shape == (nL * nW, nL * nW)

    def test_nullspace(self):
        """Laplacian of a constant field should be zero."""
        nL, nW = 5, 5
        L = build_laplacian_2d(nL, nW)
        constant_field = np.ones(nL * nW)
        result = L @ constant_field
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_interior_stencil(self):
        """Interior point should have stencil [1, 1, -4, 1, 1]."""
        nL, nW = 5, 5
        L = build_laplacian_2d(nL, nW)
        i, j = 2, 2
        k = j * nL + i
        assert L[k, k] == -4.0
        assert L[k, k - 1] == 1.0
        assert L[k, k + 1] == 1.0
        assert L[k, k - nL] == 1.0
        assert L[k, k + nL] == 1.0
        assert np.sum(L[k, :]) == 0.0

    def test_top_left_corner_stencil(self):
        nL, nW = 5, 5
        L = build_laplacian_2d(nL, nW)
        k = 0
        assert L[k, k] == 2.0
        assert L[k, k + 1] == -2.0
        assert L[k, k + 2] == 1.0
        assert L[k, k + nL] == -2.0
        assert L[k, k + 2 * nL] == 1.0
        assert np.sum(L[k, :]) == 0.0

    def test_top_edge_stencil(self):
        nL, nW = 5, 5
        L = build_laplacian_2d(nL, nW)
        i, j = 2, 0
        k = j * nL + i
        assert L[k, k] == -1.0
        assert L[k, k - 1] == 1.0
        assert L[k, k + 1] == 1.0
        assert L[k, k + nL] == -2.0
        assert L[k, k + 2 * nL] == 1.0
        assert np.sum(L[k, :]) == 0.0

    def test_bottom_right_corner_stencil(self):
        nL, nW = 5, 5
        L = build_laplacian_2d(nL, nW)
        i, j = nL - 1, nW - 1
        k = j * nL + i
        assert L[k, k] == 2.0
        assert L[k, k - 1] == -2.0
        assert L[k, k - 2] == 1.0
        assert L[k, k - nL] == -2.0
        assert L[k, k - 2 * nL] == 1.0
        assert np.sum(L[k, :]) == 0.0

    def test_small_grid_error(self):
        with pytest.raises(ValueError, match="at least 3 patches"):
            build_laplacian_2d(2, 5)


# ---------------------------------------------------------------------------
# Simple Laplacian (free boundary conditions)
# ---------------------------------------------------------------------------

class TestLaplacian2DSimple:
    """Tests for build_laplacian_2d_simple."""

    def test_matrix_shape(self):
        nL, nW = 5, 4
        L = build_laplacian_2d_simple(nL, nW)
        assert L.shape == (nL * nW, nL * nW)

    def test_row_sums_zero(self):
        """Every row should sum to zero (free boundary conditions)."""
        nL, nW = 5, 5
        L = build_laplacian_2d_simple(nL, nW)
        row_sums = np.sum(L, axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-10)

    def test_nullspace(self):
        """Laplacian of a constant field should be zero."""
        nL, nW = 4, 4
        L = build_laplacian_2d_simple(nL, nW)
        constant_field = np.ones(nL * nW)
        np.testing.assert_allclose(L @ constant_field, 0.0, atol=1e-10)

    def test_corner_has_two_neighbors(self):
        """Corner patch should have diagonal = -2."""
        nL, nW = 5, 5
        L = build_laplacian_2d_simple(nL, nW)
        assert L[0, 0] == -2.0

    def test_edge_has_three_neighbors(self):
        """Edge (non-corner) patch should have diagonal = -3."""
        nL, nW = 5, 5
        L = build_laplacian_2d_simple(nL, nW)
        assert L[1, 1] == -3.0

    def test_interior_has_four_neighbors(self):
        """Interior patch should have diagonal = -4."""
        nL, nW = 5, 5
        L = build_laplacian_2d_simple(nL, nW)
        k = 2 * nL + 2  # (2,2) interior point
        assert L[k, k] == -4.0
