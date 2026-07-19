"""Hyperparameter selection: ABIC and L-curve sweeps.

Private submodule of :mod:`geodef.invert`. The free functions here are
thin wrappers that prepare a ``LinearSystem`` and run its cached sweeps;
``compute_abic`` is the standalone single-lambda reference
implementation (Fukuda & Johnson, 2008, 2010).
"""

import numpy as np

from geodef.data import DataSet
from geodef.fault import Fault
from geodef.invert._results import ABICCurveResult, LCurveResult
from geodef.invert._solvers import BoundsSpec, _rank_positive_eigs
from geodef.invert._system import LinearSystem


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


def lcurve(
    fault: Fault,
    datasets: DataSet | list[DataSet],
    regularization: str | np.ndarray = "laplacian",
    regularization_range: tuple[float, float] = (1e-2, 1e6),
    n: int = 50,
    bounds: BoundsSpec = None,
    method: str | None = None,
    components: str = "both",
    rake: float | None = None,
    slip_azimuth: float | None = None,
    plate_rake: float | np.ndarray | None = None,
) -> LCurveResult:
    """Sweep regularization strength and compute the L-curve.

    Args:
        fault: Fault geometry.
        datasets: One or more geodetic datasets.
        regularization: Regularization type.
        regularization_range: ``(min_lambda, max_lambda)`` range to sweep.
        n: Number of lambda values to evaluate.
        bounds: Per-component slip bounds.
        method: Solver method.
        components: Which slip components to solve for.
        rake: Fixed rake angle in degrees, required when
            ``components='rake'``.
        slip_azimuth: Geographic slip azimuth in degrees, required when
            ``components='azimuth'``.
        plate_rake: Local plate-rake direction, required when
            ``components='plate'``.

    Returns:
        LCurveResult with sweep arrays and optimal lambda.
    """
    sys = LinearSystem(
        fault, datasets, regularization, components, rake, slip_azimuth, plate_rake
    )
    return sys.lcurve(regularization_range, n, bounds, method)


def abic_curve(
    fault: Fault,
    datasets: DataSet | list[DataSet],
    regularization: str | np.ndarray = "laplacian",
    regularization_range: tuple[float, float] = (1e-2, 1e6),
    n: int = 50,
    components: str = "both",
    rake: float | None = None,
    slip_azimuth: float | None = None,
    plate_rake: float | np.ndarray | None = None,
) -> ABICCurveResult:
    """Sweep regularization strength and compute the ABIC at each value.

    Also records misfit and model norm for context. The optimal lambda
    is the one that minimizes ABIC.

    Args:
        fault: Fault geometry.
        datasets: One or more geodetic datasets.
        regularization: Regularization type.
        regularization_range: ``(min_lambda, max_lambda)`` range to sweep.
        n: Number of lambda values to evaluate.
        components: Which slip components to solve for.
        rake: Fixed rake angle in degrees, required when
            ``components='rake'``.
        slip_azimuth: Geographic slip azimuth in degrees, required when
            ``components='azimuth'``.
        plate_rake: Local plate-rake direction, required when
            ``components='plate'``.

    Returns:
        ABICCurveResult with sweep arrays and optimal lambda.
    """
    sys = LinearSystem(
        fault, datasets, regularization, components, rake, slip_azimuth, plate_rake
    )
    return sys.abic_curve(regularization_range, n)
