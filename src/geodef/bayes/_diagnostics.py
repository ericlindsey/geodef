"""Chain convergence diagnostics: split R-hat and effective sample size.

Private submodule of :mod:`geodef.bayes`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _split_chains(chains: npt.ArrayLike) -> np.ndarray:
    """Split each chain in half, doubling the chain count."""
    c = np.atleast_2d(np.asarray(chains, dtype=float))
    half = c.shape[1] // 2
    return np.concatenate([c[:, :half], c[:, half : 2 * half]], axis=0)


def split_rhat(chains: npt.ArrayLike) -> float:
    """Split-chain potential scale reduction factor (Gelman-Rubin R-hat).

    Each chain is split in half so within-chain non-stationarity also
    inflates the statistic. Values near 1 indicate convergence; above
    ~1.01-1.05, run longer.

    Args:
        chains: Draws of one parameter, shape (n_chains, n_samples).

    Returns:
        The split R-hat statistic.
    """
    c = _split_chains(chains)
    n = c.shape[1]
    w = float(c.var(axis=1, ddof=1).mean())
    if w == 0.0:
        return 1.0
    b_over_n = float(c.mean(axis=1).var(ddof=1))
    var_plus = (n - 1) / n * w + b_over_n
    return float(np.sqrt(var_plus / w))


def effective_sample_size(chains: npt.ArrayLike) -> float:
    """Effective sample size from split chains (Geyer/Stan estimator).

    Uses the multi-chain autocorrelation estimate with Geyer's initial
    monotone positive-pair truncation.

    Args:
        chains: Draws of one parameter, shape (n_chains, n_samples).

    Returns:
        Estimated number of independent draws.
    """
    c = _split_chains(chains)
    m, n = c.shape
    w = float(c.var(axis=1, ddof=1).mean())
    if w == 0.0:
        return float(m * n)
    b_over_n = float(c.mean(axis=1).var(ddof=1))
    var_plus = (n - 1) / n * w + b_over_n

    dev = c - c.mean(axis=1, keepdims=True)
    acov = np.stack([np.correlate(d, d, "full")[n - 1 :] / n for d in dev])
    rho = 1.0 - (w - acov.mean(axis=0)) / var_plus

    tau = -rho[0]
    prev = np.inf
    for k in range(n // 2):
        pair = rho[2 * k] + rho[2 * k + 1]
        if pair <= 0.0:
            break
        pair = min(pair, prev)
        tau += 2.0 * pair
        prev = pair
    tau = max(tau, 1.0 / (m * n))
    return float(m * n / tau)
