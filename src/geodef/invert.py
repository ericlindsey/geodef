"""One-call inversion for fault slip from geodetic data.

Solves d = Gm for slip m with optional regularization and bounds.
Supports weighted least-squares, non-negative least-squares, and
bounded least-squares solvers.
"""

import dataclasses

import numpy as np
import scipy.linalg
import scipy.optimize

from geodef.data import DataSet
from geodef.fault import Fault, moment_to_magnitude
from geodef.greens import greens, stack_obs, stack_weights

_VALID_METHODS = {"wls", "nnls", "bounded_ls"}
_VALID_SMOOTHING_STRINGS = {"laplacian", "damping", "stresskernel"}


@dataclasses.dataclass(frozen=True)
class InversionResult:
    """Result of a fault slip inversion.

    Attributes:
        slip: Strike-slip and dip-slip per patch, shape (N, 2).
        slip_vector: Interleaved solution [ss0, ds0, ss1, ds1, ...], shape (2N,).
        residuals: Observation minus prediction, shape (M,).
        predicted: Forward-modeled observations, shape (M,).
        chi2: Reduced chi-squared misfit.
        rms: Root-mean-square of residuals.
        moment: Scalar seismic moment in N-m.
        Mw: Moment magnitude.
        smoothing_strength: Regularization weight used, or None if unregularized.
    """

    slip: np.ndarray
    slip_vector: np.ndarray
    residuals: np.ndarray
    predicted: np.ndarray
    chi2: float
    rms: float
    moment: float
    Mw: float
    smoothing_strength: float | None


def invert(
    fault: Fault,
    datasets: DataSet | list[DataSet],
    smoothing: str | np.ndarray | None = None,
    smoothing_strength: float = 0.0,
    bounds: tuple[float | None, float | None] | None = None,
    method: str | None = None,
    smoothing_target: np.ndarray | None = None,
) -> InversionResult:
    """Invert geodetic data for fault slip.

    Args:
        fault: Fault geometry.
        datasets: One or more geodetic datasets.
        smoothing: Regularization type. One of ``'laplacian'``,
            ``'damping'``, ``'stresskernel'``, a custom matrix, or None.
        smoothing_strength: Scalar weight on the regularization term.
        bounds: Per-component slip bounds ``(lower, upper)``.
            Use None for unbounded side, e.g. ``(0, None)``.
        method: Solver — ``'wls'``, ``'nnls'``, or ``'bounded_ls'``.
            Auto-selected from bounds if None.
        smoothing_target: Reference model vector, shape (2N,). Regularizes
            toward this target instead of zero: minimizes
            ``||L(m - m_ref)||^2``. Only valid when smoothing is set.

    Returns:
        InversionResult with slip, residuals, and fit statistics.

    Raises:
        ValueError: For invalid arguments.
    """
    if isinstance(datasets, DataSet):
        datasets = [datasets]

    n_params = 2 * fault.n_patches

    _validate_args(
        smoothing, smoothing_strength, bounds, method,
        smoothing_target, n_params,
    )

    # Assemble data system
    G = greens(fault, datasets)
    d = stack_obs(datasets)
    W = stack_weights(datasets)

    # Weight the system: G_w, d_w such that ||G_w m - d_w||^2 = (Gm-d)^T W (Gm-d)
    G_w, d_w = _apply_weights(G, d, W)

    # Build and append regularization
    if smoothing is not None and smoothing_strength > 0:
        L = _build_smoothing_matrix(fault, smoothing, n_params)
        d_reg = _build_reg_rhs(L, smoothing_strength, smoothing_target)
        G_aug = np.vstack([G_w, np.sqrt(smoothing_strength) * L])
        d_aug = np.concatenate([d_w, d_reg])
        reg_strength: float | None = smoothing_strength
    else:
        G_aug = G_w
        d_aug = d_w
        reg_strength = None

    # Auto-select solver
    if method is None:
        method = _auto_select_method(bounds)

    # Solve
    m = _solve(G_aug, d_aug, bounds, method)

    # Compute fit statistics on unweighted data
    predicted = G @ m
    residuals = d - predicted
    chi2 = _compute_chi2(residuals, W, n_params)
    rms = float(np.sqrt(np.mean(residuals ** 2)))

    n_patches = fault.n_patches
    slip = np.column_stack([m[:n_patches], m[n_patches:]])
    slip_mag = np.sqrt(slip[:, 0] ** 2 + slip[:, 1] ** 2)
    moment = fault.moment(slip_mag)
    mw = moment_to_magnitude(moment)

    return InversionResult(
        slip=slip,
        slip_vector=m,
        residuals=residuals,
        predicted=predicted,
        chi2=chi2,
        rms=rms,
        moment=moment,
        Mw=mw,
        smoothing_strength=reg_strength,
    )


def _validate_args(
    smoothing: str | np.ndarray | None,
    smoothing_strength: float,
    bounds: tuple[float | None, float | None] | None,
    method: str | None,
    smoothing_target: np.ndarray | None,
    n_params: int,
) -> None:
    """Validate invert() arguments."""
    if method is not None and method not in _VALID_METHODS:
        raise ValueError(
            f"method must be one of {_VALID_METHODS}, got {method!r}"
        )

    if isinstance(smoothing, str) and smoothing not in _VALID_SMOOTHING_STRINGS:
        raise ValueError(
            f"smoothing must be one of {_VALID_SMOOTHING_STRINGS} "
            f"or a numpy array, got {smoothing!r}"
        )

    if isinstance(smoothing, np.ndarray) and smoothing.shape[1] != n_params:
        raise ValueError(
            f"smoothing matrix must have {n_params} columns, "
            f"got {smoothing.shape[1]}"
        )

    if smoothing_target is not None:
        if smoothing is None and smoothing_strength == 0.0:
            raise ValueError(
                "smoothing_target requires smoothing to be set"
            )
        target = np.asarray(smoothing_target)
        if target.shape != (n_params,):
            raise ValueError(
                f"smoothing_target must have shape ({n_params},), "
                f"got {target.shape}"
            )


def _apply_weights(
    G: np.ndarray, d: np.ndarray, W: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply data weights via W^(1/2).

    For diagonal W, uses efficient element-wise scaling.
    For full W, uses Cholesky decomposition.

    Returns:
        (G_weighted, d_weighted).
    """
    off_diag = W - np.diag(np.diag(W))
    if np.allclose(off_diag, 0):
        w_half = np.sqrt(np.diag(W))
        return w_half[:, np.newaxis] * G, w_half * d

    W_half = scipy.linalg.cholesky(W, lower=False)
    return W_half @ G, W_half @ d


def _build_smoothing_matrix(
    fault: Fault,
    smoothing: str | np.ndarray,
    n_params: int,
) -> np.ndarray:
    """Build the regularization matrix L.

    Args:
        fault: Fault geometry.
        smoothing: Smoothing type or custom matrix.
        n_params: Number of model parameters (2 * n_patches).

    Returns:
        Regularization matrix with n_params columns.
    """
    if isinstance(smoothing, np.ndarray):
        return smoothing

    if smoothing == "damping":
        return np.eye(n_params)

    if smoothing == "laplacian":
        L_patch = fault.laplacian
        return scipy.linalg.block_diag(L_patch, L_patch)

    if smoothing == "stresskernel":
        return fault.stress_kernel()

    raise ValueError(f"Unknown smoothing type: {smoothing!r}")


def _build_reg_rhs(
    L: np.ndarray,
    smoothing_strength: float,
    smoothing_target: np.ndarray | None,
) -> np.ndarray:
    """Build the right-hand side for the regularization rows.

    For standard regularization (target=None): zeros.
    For target regularization: sqrt(lambda) * L @ m_ref.
    """
    if smoothing_target is None:
        return np.zeros(L.shape[0])
    return np.sqrt(smoothing_strength) * (L @ smoothing_target)


def _auto_select_method(
    bounds: tuple[float | None, float | None] | None,
) -> str:
    """Choose solver based on bounds."""
    if bounds is None:
        return "wls"
    lower, upper = bounds
    if lower == 0 and upper is None:
        return "nnls"
    return "bounded_ls"


def _solve(
    G: np.ndarray,
    d: np.ndarray,
    bounds: tuple[float | None, float | None] | None,
    method: str,
) -> np.ndarray:
    """Dispatch to the appropriate solver.

    Returns:
        Solution vector m, shape (n_params,).
    """
    if method == "wls":
        m, _, _, _ = np.linalg.lstsq(G, d, rcond=None)
        return m

    if method == "nnls":
        m, _ = scipy.optimize.nnls(G, d)
        return m

    if method == "bounded_ls":
        lower = -np.inf if bounds is None or bounds[0] is None else bounds[0]
        upper = np.inf if bounds is None or bounds[1] is None else bounds[1]
        result = scipy.optimize.lsq_linear(G, d, bounds=(lower, upper))
        return result.x

    raise ValueError(f"Unknown method: {method!r}")


def _compute_chi2(
    residuals: np.ndarray,
    W: np.ndarray,
    n_params: int,
) -> float:
    """Compute reduced chi-squared: r^T W r / (M - n_params)."""
    n_obs = len(residuals)
    dof = n_obs - n_params
    if dof <= 0:
        return float("nan")
    weighted_ssr = residuals @ W @ residuals
    return float(weighted_ssr / dof)
