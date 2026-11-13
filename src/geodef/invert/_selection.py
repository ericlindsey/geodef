"""Standalone ABIC reference implementation (Fukuda & Johnson, 2008, 2010).

Private submodule of :mod:`geodef.invert`. The L-curve and ABIC sweeps
live on :class:`LinearSystem`, which caches the matrix products they
share; ``compute_abic`` evaluates one lambda from raw matrices.
"""

import numpy as np

from geodef.invert._solvers import _rank_positive_eigs


def compute_abic(
    G: np.ndarray,
    d: np.ndarray,
    W: np.ndarray,
    L: np.ndarray,
    regularization_strength: float,
) -> float:
    """Compute the ABIC value for a given regularization strength.

    Implements the Akaike Bayesian Information Criterion following
    Fukuda & Johnson (2008, 2010).

    Args:
        G: Green's matrix, shape (M, P).
        d: Data vector, shape (M,).
        W: Weight matrix, shape (M, M).
        L: Regularization matrix, shape (K, P).
        regularization_strength: Regularization weight (lambda = alpha^2).

    Returns:
        ABIC scalar value (lower is better).
    """
    alpha2 = regularization_strength
    n_data = len(d)

    GtWG = G.T @ W @ G
    LtL = L.T @ L
    H = GtWG + alpha2 * LtL
    m = np.linalg.solve(H, G.T @ W @ d)

    residuals = d - G @ m
    misfit = float(residuals @ W @ residuals)
    penalty = alpha2 * float(m @ LtL @ m)
    total = max(misfit + penalty, 1e-300)
    abic1 = n_data * np.log(total)

    eig_prior = alpha2 * _rank_positive_eigs(np.linalg.eigvalsh(LtL))
    abic2 = np.sum(np.log(eig_prior))

    eig_post = _rank_positive_eigs(np.linalg.eigvalsh(H))
    abic3 = np.sum(np.log(eig_post))

    return float(abic1 - abic2 + abic3)
