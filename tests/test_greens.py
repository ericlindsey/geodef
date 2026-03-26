"""Tests for geodef.greens (Laplacian operators and Green's matrix utilities).

Laplacian tests migrated from related/shakeout_v2/test_laplacian.py.
"""

import numpy as np
import pytest

from geodef.greens import (
    build_laplacian_2d,
    build_laplacian_2d_simple,
    build_laplacian_knn,
)


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


# ---------------------------------------------------------------------------
# KNN Laplacian (unstructured meshes, distance-weighted)
# ---------------------------------------------------------------------------

class TestLaplacianKNN:
    """Tests for build_laplacian_knn."""

    @pytest.fixture()
    def regular_grid(self):
        """6x6 regular grid of 3D points (flat, z=0)."""
        x = np.arange(6, dtype=float)
        xx, yy = np.meshgrid(x, x)
        coords = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(36)])
        return coords

    @pytest.fixture()
    def tilted_grid(self):
        """6x6 grid on a tilted plane (z = 0.1*x + 0.2*y)."""
        x = np.arange(6, dtype=float)
        xx, yy = np.meshgrid(x, x)
        zz = 0.1 * xx + 0.2 * yy
        return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    def test_matrix_shape(self, regular_grid):
        L = build_laplacian_knn(regular_grid, k=4)
        n = regular_grid.shape[0]
        assert L.shape == (n, n)

    def test_rows_sum_to_zero(self, regular_grid):
        """Every row must sum to zero (constant field in nullspace)."""
        L = build_laplacian_knn(regular_grid, k=4)
        np.testing.assert_allclose(L.sum(axis=1), 0.0, atol=1e-10)

    def test_nullspace_constant(self, regular_grid):
        """Laplacian of a constant field should be zero."""
        L = build_laplacian_knn(regular_grid, k=4)
        np.testing.assert_allclose(L @ np.ones(36), 0.0, atol=1e-10)

    def test_sparse_output(self, regular_grid):
        """Should return a scipy sparse matrix."""
        import scipy.sparse
        L = build_laplacian_knn(regular_grid, k=4)
        assert scipy.sparse.issparse(L)

    def test_diagonal_negative(self, regular_grid):
        """Diagonal entries should be negative (or zero for degenerate cases)."""
        L = build_laplacian_knn(regular_grid, k=4)
        diag = L.diagonal()
        assert np.all(diag <= 0)

    def test_off_diagonal_nonnegative(self, regular_grid):
        """Off-diagonal entries should be non-negative."""
        L = build_laplacian_knn(regular_grid, k=4)
        Ld = L.toarray()
        np.fill_diagonal(Ld, 0.0)
        assert np.all(Ld >= -1e-15)

    def test_symmetry(self, regular_grid):
        """KNN Laplacian should be symmetric after symmetrization."""
        L = build_laplacian_knn(regular_grid, k=4)
        Ld = L.toarray()
        np.testing.assert_allclose(Ld, Ld.T, atol=1e-12)

    def test_different_k_values(self, regular_grid):
        """Should work with various k values."""
        for k in [3, 4, 6, 8]:
            L = build_laplacian_knn(regular_grid, k=k)
            np.testing.assert_allclose(L.sum(axis=1), 0.0, atol=1e-10)

    def test_tilted_plane_nullspace(self, tilted_grid):
        """Should still annihilate constants on a non-flat surface."""
        L = build_laplacian_knn(tilted_grid, k=4)
        np.testing.assert_allclose(L @ np.ones(len(tilted_grid)), 0.0, atol=1e-10)

    def test_regular_grid_recovers_simple_laplacian(self, regular_grid):
        """On a regular grid with k=4, KNN should match the simple Laplacian structure.

        Interior points should couple to exactly their 4 grid neighbors.
        """
        L = build_laplacian_knn(regular_grid, k=4)
        Ld = L.toarray()
        # Interior point (2,2) on a 6x6 grid: index = 2*6 + 2 = 14
        k_idx = 14
        row = Ld[k_idx]
        # Should have exactly 4 nonzero off-diagonal entries (the 4 grid neighbors)
        neighbors = np.where(row > 0)[0]
        assert len(neighbors) == 4
        # Neighbors should be the 4 adjacent grid points
        expected = {14 - 1, 14 + 1, 14 - 6, 14 + 6}
        assert set(neighbors) == expected

    def test_irregular_mesh(self):
        """Should work on scattered points (not on a grid)."""
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 10, size=(50, 3))
        L = build_laplacian_knn(coords, k=6)
        np.testing.assert_allclose(L.sum(axis=1), 0.0, atol=1e-10)

    def test_k_too_large_raises(self):
        """k >= n_points should raise ValueError."""
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        with pytest.raises(ValueError, match="k"):
            build_laplacian_knn(coords, k=3)

    def test_k_less_than_one_raises(self):
        """k < 1 should raise ValueError."""
        coords = np.zeros((10, 3))
        with pytest.raises(ValueError, match="k"):
            build_laplacian_knn(coords, k=0)

    def test_weights_inversely_proportional_to_distance(self, regular_grid):
        """Closer neighbors should get larger weights than farther ones.

        On a regular grid with k=8, the 4 cardinal neighbors (distance=1)
        should have larger weights than the 4 diagonal neighbors (distance=sqrt(2)).
        """
        L = build_laplacian_knn(regular_grid, k=8)
        Ld = L.toarray()
        # Interior point (2,2) on 6x6 grid: index 14
        k_idx = 14
        cardinal = [13, 15, 8, 20]  # left, right, up, down
        diagonal = [7, 9, 19, 21]  # corners
        cardinal_weights = [Ld[k_idx, j] for j in cardinal]
        diagonal_weights = [Ld[k_idx, j] for j in diagonal]
        # Cardinal (dist=1) should have larger weight than diagonal (dist=sqrt(2))
        assert min(cardinal_weights) > max(diagonal_weights)
