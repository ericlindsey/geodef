import pytest
import numpy as np
import sys
import os

# Add the notebooks directory to the path so we can import okada_utils
sys.path.append(os.path.join(os.getcwd(), 'notebooks'))
from okada_utils import build_laplacian_2d

def test_laplacian_matrix_shape():
    nL, nW = 5, 4
    L = build_laplacian_2d(nL, nW)
    assert L.shape == (nL * nW, nL * nW)

def test_laplacian_nullspace():
    """Laplacian of a constant field should be zero."""
    nL, nW = 5, 5
    L = build_laplacian_2d(nL, nW)
    constant_field = np.ones(nL * nW)
    result = L @ constant_field
    np.testing.assert_allclose(result, 0.0, atol=1e-10)

def test_laplacian_interior_stencil():
    """Test interior stencil: [1, 1, -4, 1, 1]."""
    nL, nW = 5, 5
    L = build_laplacian_2d(nL, nW)
    i, j = 2, 2
    k = j * nL + i
    # Neighbors: (i-1,j), (i+1,j), (i,j-1), (i,j+1)
    # nL=5, k=12. Neighbors: 11, 13, 7, 17
    assert L[k, k] == -4.0
    assert L[k, k-1] == 1.0
    assert L[k, k+1] == 1.0
    assert L[k, k-nL] == 1.0
    assert L[k, k+nL] == 1.0
    # Sum of row is zero
    assert np.sum(L[k, :]) == 0.0

def test_laplacian_top_left_corner_stencil():
    """Test top-left corner (0,0) stencil."""
    nL, nW = 5, 5
    L = build_laplacian_2d(nL, nW)
    k = 0
    # Expected: 2 at (0,0), -2 at (1,0), 1 at (2,0), -2 at (0,1), 1 at (0,2)
    assert L[k, k] == 2.0
    assert L[k, k+1] == -2.0
    assert L[k, k+2] == 1.0
    assert L[k, k+nL] == -2.0
    assert L[k, k+2*nL] == 1.0
    assert np.sum(L[k, :]) == 0.0

def test_laplacian_top_edge_stencil():
    """Test top edge (2,0) stencil."""
    nL, nW = 5, 5
    L = build_laplacian_2d(nL, nW)
    i, j = 2, 0
    k = j * nL + i # k = 2
    # Along strike: [1, -2, 1] at k-1, k, k+1
    # Down dip: [1, -2, 1] at k, k+nL, k+2nL
    # Sum at k: -2 + 1 = -1
    assert L[k, k] == -1.0
    assert L[k, k-1] == 1.0
    assert L[k, k+1] == 1.0
    assert L[k, k+nL] == -2.0
    assert L[k, k+2*nL] == 1.0
    assert np.sum(L[k, :]) == 0.0

def test_laplacian_bottom_right_corner_stencil():
    """Test bottom-right corner stencil."""
    nL, nW = 5, 5
    L = build_laplacian_2d(nL, nW)
    i, j = nL - 1, nW - 1
    k = j * nL + i
    # [1, -2, 1] backward in i: L[k,k]=1, L[k,k-1]=-2, L[k,k-2]=1
    # [1, -2, 1] backward in j: L[k,k]+=1, L[k,k-nL]=-2, L[k,k-2nL]=1
    assert L[k, k] == 2.0
    assert L[k, k-1] == -2.0
    assert L[k, k-2] == 1.0
    assert L[k, k-nL] == -2.0
    assert L[k, k-2*nL] == 1.0
    assert np.sum(L[k, :]) == 0.0

def test_laplacian_small_grid_error():
    with pytest.raises(ValueError, match="at least 3 patches"):
        build_laplacian_2d(2, 5)
