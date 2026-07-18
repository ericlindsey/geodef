"""Least-squares solver dispatch: WLS, NNLS, bounded, and constrained.

Private submodule of :mod:`geodef.invert`. Operates on the augmented
(whitened, regularization-row-stacked) system assembled by
``LinearSystem``; bounds arrive pre-expanded to per-parameter arrays.
"""

import numpy as np
import scipy.optimize

_VALID_METHODS = {"wls", "nnls", "bounded_ls", "constrained"}


# A bound may be a scalar (all parameters), an array of length n_components
# (one value per slip component, broadcast over patches), or an array of
# length n_params (one value per parameter). ``None`` means unbounded.
_BoundValue = float | np.ndarray | None


BoundsSpec = tuple[_BoundValue, _BoundValue] | None


# Internal fully-expanded form: per-parameter lower/upper arrays.
_ExpandedBounds = tuple[np.ndarray, np.ndarray] | None


def _expand_bounds(
    bounds: BoundsSpec,
    n_patches: int,
    n_components: int,
) -> _ExpandedBounds:
    """Expand bounds to per-parameter lower/upper arrays.

    Each of ``(lower, upper)`` may be ``None`` (unbounded), a scalar (applied
    to every parameter), an array of length ``n_components`` (one value per
    slip component, broadcast across all patches), or an array of length
    ``n_params = n_patches * n_components`` (one value per parameter).

    Args:
        bounds: The user bounds specification, or None.
        n_patches: Number of patches N.
        n_components: Number of slip components solved for (1 or 2).

    Returns:
        ``(lower, upper)`` per-parameter arrays with ``-inf``/``+inf`` for
        unbounded entries, or None if ``bounds`` is None.

    Raises:
        ValueError: If an array bound has an unsupported length.
    """
    if bounds is None:
        return None
    n_params = n_patches * n_components

    def _expand(val: _BoundValue, fill: float) -> np.ndarray:
        if val is None:
            return np.full(n_params, fill)
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 0:
            return np.full(n_params, float(arr))
        if arr.shape == (n_params,):
            return arr
        if n_components > 1 and arr.shape == (n_components,):
            return np.repeat(arr, n_patches)
        raise ValueError(
            "bounds array must be a scalar, length n_components "
            f"({n_components}), or length n_params ({n_params}); "
            f"got shape {arr.shape}"
        )

    lower_raw, upper_raw = bounds
    return _expand(lower_raw, -np.inf), _expand(upper_raw, np.inf)


def _auto_select_method(bounds: _ExpandedBounds) -> str:
    """Choose solver based on expanded per-parameter bounds."""
    if bounds is None:
        return "wls"
    lower, upper = bounds
    if np.all(lower == 0.0) and np.all(np.isposinf(upper)):
        return "nnls"
    return "bounded_ls"


def _solve(
    G: np.ndarray,
    d: np.ndarray,
    bounds: _ExpandedBounds,
    method: str,
    constraints: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    """Dispatch to the appropriate solver.

    Returns:
        Solution vector m, shape (n_params,).
    """
    if method == "wls":
        m_rows, n_cols = G.shape
        if m_rows > n_cols:
            # Overdetermined: normal equations are faster than lstsq (SVD).
            return np.linalg.solve(G.T @ G, G.T @ d)
        # Underdetermined or square: lstsq gives the minimum-norm solution.
        m, _, _, _ = np.linalg.lstsq(G, d, rcond=None)
        return m

    if method == "nnls":
        m, _ = scipy.optimize.nnls(G, d)
        return m

    if method == "bounded_ls":
        lower, upper = (-np.inf, np.inf) if bounds is None else bounds
        result = scipy.optimize.lsq_linear(G, d, bounds=(lower, upper))
        return result.x

    if method == "constrained":
        return _solve_constrained(G, d, bounds, constraints)

    raise ValueError(f"Unknown method: {method!r}")


def _solve_constrained(
    G: np.ndarray,
    d: np.ndarray,
    bounds: _ExpandedBounds,
    constraints: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    """Solve via quadratic programming (minimize ||Gm - d||^2 subject to constraints).

    Uses scipy.optimize.minimize with SLSQP, which supports both
    bounds and linear inequality constraints.

    Args:
        G: Design matrix (possibly augmented with regularization).
        d: Data vector (possibly augmented).
        bounds: Per-component (lower, upper) bounds, or None.
        constraints: ``(C, d_ineq)`` such that ``C @ m <= d_ineq``, or None.

    Returns:
        Solution vector m.

    Raises:
        RuntimeError: If SLSQP fails to converge to a feasible solution.
    """
    objective_scale = max(float(np.linalg.norm(G)), float(np.linalg.norm(d)), 1.0)
    G_scaled = G / objective_scale
    d_scaled = d / objective_scale
    GtG = G_scaled.T @ G_scaled
    Gtd = G_scaled.T @ d_scaled

    def objective(m: np.ndarray) -> float:
        r = G_scaled @ m - d_scaled
        return 0.5 * float(r @ r)

    def gradient(m: np.ndarray) -> np.ndarray:
        return GtG @ m - Gtd

    if bounds is not None:
        lower, upper = bounds
        scipy_bounds = list(zip(lower, upper))
    else:
        scipy_bounds = None

    scipy_constraints = []
    if constraints is not None:
        C, d_ineq = constraints
        scipy_constraints.append(
            {
                "type": "ineq",
                "fun": lambda m, C=C, d_ineq=d_ineq: d_ineq - C @ m,
                "jac": lambda m, C=C: -C,
            }
        )

    m0, _, _, _ = np.linalg.lstsq(G, d, rcond=None)
    if bounds is not None:
        m0 = np.clip(m0, bounds[0], bounds[1])

    result = scipy.optimize.minimize(
        objective,
        m0,
        jac=gradient,
        method="SLSQP",
        bounds=scipy_bounds,
        constraints=scipy_constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Constrained solver failed: {result.message}")
    if constraints is not None:
        C, d_ineq = constraints
        feasibility_tolerance = 1e-8 * max(float(np.max(np.abs(d_ineq))), 1.0)
        max_violation = float(np.max(C @ result.x - d_ineq))
        if max_violation > feasibility_tolerance:
            raise RuntimeError(
                "Constrained solver returned an infeasible solution: "
                f"maximum inequality violation is {max_violation:.3g}"
            )
    return result.x


def _rank_positive_eigs(eigs: np.ndarray) -> np.ndarray:
    """Eigenvalues above the numerical-rank cutoff (as in ``matrix_rank``).

    A graph Laplacian's zero modes come back from ``eigvalsh`` as values of
    order 1e-15 with either sign; a plain ``> 0`` filter keeps them, which
    injects a spurious ``n0 * log(lambda)`` term into ABIC and biases the
    selected regularization strength.
    """
    eigs = np.abs(np.asarray(eigs, dtype=float))
    if eigs.size == 0:
        return eigs
    tol = eigs.max() * eigs.size * np.finfo(float).eps
    return eigs[eigs > tol]


def _compute_reduced_chi2(
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
