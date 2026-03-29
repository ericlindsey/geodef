"""One-call inversion for fault slip from geodetic data.

Solves d = Gm for slip m with optional regularization and bounds.
Supports weighted least-squares, non-negative least-squares,
bounded least-squares, and constrained (QP) solvers.
Automatic hyperparameter tuning via ABIC or cross-validation.
"""

import dataclasses

import numpy as np
import scipy.linalg
import scipy.optimize

from geodef.data import DataSet
from geodef.fault import Fault, moment_to_magnitude
from geodef.greens import greens, stack_obs, stack_weights

_VALID_METHODS = {"wls", "nnls", "bounded_ls", "constrained"}
_VALID_SMOOTHING_STRINGS = {"laplacian", "damping", "stresskernel"}
_VALID_STRENGTH_STRINGS = {"abic", "cv"}
_VALID_COMPONENTS = {"both", "strike", "dip"}


@dataclasses.dataclass(frozen=True)
class InversionResult:
    """Result of a fault slip inversion.

    Attributes:
        slip: Slip per patch, shape (N, n_components). Columns ordered
            as [strike-slip, dip-slip] for ``components='both'``, or
            a single column for ``'strike'`` or ``'dip'``.
        slip_vector: Blocked solution vector, shape (n_components * N,).
        residuals: Observation minus prediction, shape (M,).
        predicted: Forward-modeled observations, shape (M,).
        chi2: Reduced chi-squared misfit.
        rms: Root-mean-square of residuals.
        moment: Scalar seismic moment in N-m.
        Mw: Moment magnitude.
        smoothing: Regularization type used, or None if unregularized.
        smoothing_strength: Regularization weight used, or None if unregularized.
        components: Which slip components were solved for.
    """

    slip: np.ndarray
    slip_vector: np.ndarray
    residuals: np.ndarray
    predicted: np.ndarray
    chi2: float
    rms: float
    moment: float
    Mw: float
    smoothing: str | np.ndarray | None
    smoothing_strength: float | None
    components: str


@dataclasses.dataclass(frozen=True)
class DatasetDiagnostics:
    """Per-dataset fit diagnostics.

    Attributes:
        chi2: Weighted sum of squared residuals for this dataset.
        reduced_chi2: chi2 / effective DOF.
        wrms: Weighted root-mean-square residual.
        rms: Unweighted root-mean-square residual.
        n_obs: Number of observations in this dataset.
        dof: Effective degrees of freedom (n_obs - leverage).
        leverage: Sum of hat-matrix diagonal entries for this dataset
            (effective number of parameters consumed).
    """

    chi2: float
    reduced_chi2: float
    wrms: float
    rms: float
    n_obs: int
    dof: float
    leverage: float


@dataclasses.dataclass(frozen=True)
class LCurveResult:
    """Result of an L-curve analysis.

    Attributes:
        smoothing_values: Array of lambda values swept.
        misfits: Data misfit norm at each lambda.
        model_norms: Regularized model norm at each lambda.
        optimal: Lambda at the maximum-curvature corner.
    """

    smoothing_values: np.ndarray
    misfits: np.ndarray
    model_norms: np.ndarray
    optimal: float

    def plot(
        self,
        *,
        ax: "matplotlib.axes.Axes | None" = None,
        line_kwargs: dict | None = None,
        marker_kwargs: dict | None = None,
        annotate: bool = True,
    ) -> "matplotlib.axes.Axes":
        """Plot the L-curve with the optimal point marked.

        Args:
            ax: Axes to plot on. Creates a new figure if ``None``.
            line_kwargs: Extra kwargs for the curve line.
            marker_kwargs: Extra kwargs for the optimal-point marker.
            annotate: Whether to label the optimal point with its
                smoothing-strength value (default ``True``).

        Returns:
            The axes used for plotting.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        lkw = {"color": "b", "marker": ".", "linestyle": "-"}
        if line_kwargs:
            lkw.update(line_kwargs)
        ax.loglog(self.misfits, self.model_norms, **lkw)

        mkw: dict = {"color": "r", "marker": "o", "markersize": 10,
                      "linestyle": "none"}
        if marker_kwargs:
            mkw.update(marker_kwargs)
        idx = np.argmin(np.abs(self.smoothing_values - self.optimal))
        ax.loglog(self.misfits[idx], self.model_norms[idx], **mkw)

        if annotate:
            ax.annotate(
                f"λ = {self.optimal:.3g}",
                xy=(self.misfits[idx], self.model_norms[idx]),
                xytext=(10, 10), textcoords="offset points",
                fontsize=9, color=mkw.get("color", "r"),
                arrowprops={"arrowstyle": "->",
                             "color": mkw.get("color", "r")},
            )

        ax.set_xlabel("Data misfit ||Gm - d||")
        ax.set_ylabel("Model norm ||Lm||")
        ax.set_title("L-curve")
        return ax


@dataclasses.dataclass(frozen=True)
class ABICCurveResult:
    """Result of an ABIC curve analysis.

    Attributes:
        smoothing_values: Array of lambda values swept.
        abic_values: ABIC value at each lambda (lower is better).
        misfits: Data misfit norm at each lambda.
        model_norms: Regularized model norm at each lambda.
        optimal: Lambda at the minimum ABIC.
    """

    smoothing_values: np.ndarray
    abic_values: np.ndarray
    misfits: np.ndarray
    model_norms: np.ndarray
    optimal: float

    def plot(
        self,
        *,
        ax: "matplotlib.axes.Axes | None" = None,
        line_kwargs: dict | None = None,
        marker_kwargs: dict | None = None,
        annotate: bool = True,
    ) -> "matplotlib.axes.Axes":
        """Plot ABIC vs smoothing strength with the optimal point marked.

        Args:
            ax: Axes to plot on. Creates a new figure if ``None``.
            line_kwargs: Extra kwargs for the curve line.
            marker_kwargs: Extra kwargs for the optimal-point marker.
            annotate: Whether to label the optimal point with its
                smoothing-strength value (default ``True``).

        Returns:
            The axes used for plotting.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        lkw = {"color": "b", "marker": ".", "linestyle": "-"}
        if line_kwargs:
            lkw.update(line_kwargs)
        ax.semilogx(self.smoothing_values, self.abic_values, **lkw)

        mkw: dict = {"color": "r", "marker": "o", "markersize": 10,
                      "linestyle": "none"}
        if marker_kwargs:
            mkw.update(marker_kwargs)
        idx = np.argmin(np.abs(self.smoothing_values - self.optimal))
        ax.semilogx(self.smoothing_values[idx], self.abic_values[idx], **mkw)

        if annotate:
            ax.annotate(
                f"λ = {self.optimal:.3g}",
                xy=(self.smoothing_values[idx], self.abic_values[idx]),
                xytext=(0, 20), textcoords="offset points",
                fontsize=9, color=mkw.get("color", "r"),
                arrowprops={"arrowstyle": "->",
                             "color": mkw.get("color", "r")},
            )

        ax.set_xlabel("Smoothing strength (lambda)")
        ax.set_ylabel("ABIC")
        ax.set_title("ABIC curve")
        return ax


def invert(
    fault: Fault,
    datasets: DataSet | list[DataSet],
    smoothing: str | np.ndarray | None = None,
    smoothing_strength: float | str = 0.0,
    bounds: tuple[float | None, float | None] | None = None,
    method: str | None = None,
    smoothing_target: np.ndarray | None = None,
    components: str = "both",
    constraints: tuple[np.ndarray, np.ndarray] | None = None,
    cv_folds: int = 5,
) -> InversionResult:
    """Invert geodetic data for fault slip.

    Args:
        fault: Fault geometry.
        datasets: One or more geodetic datasets.
        smoothing: Regularization type. One of ``'laplacian'``,
            ``'damping'``, ``'stresskernel'``, a custom matrix, or None.
        smoothing_strength: Scalar weight on the regularization term,
            or ``'abic'`` / ``'cv'`` for automatic tuning.
        bounds: Per-component slip bounds ``(lower, upper)``.
            Use None for unbounded side, e.g. ``(0, None)``.
        method: Solver — ``'wls'``, ``'nnls'``, ``'bounded_ls'``, or
            ``'constrained'``. Auto-selected from bounds if None.
        smoothing_target: Reference model vector, shape
            (n_components * N,). Regularizes toward this target instead
            of zero: minimizes ``||L(m - m_ref)||^2``. Only valid when
            smoothing is set.
        components: Which slip components to solve for. One of
            ``'both'`` (strike + dip, default), ``'strike'``, or
            ``'dip'``.
        constraints: Inequality constraints ``(C, d_ineq)`` such that
            ``C @ m <= d_ineq``. Only used with ``method='constrained'``.
        cv_folds: Number of folds for cross-validation (default 5).

    Returns:
        InversionResult with slip, residuals, and fit statistics.

    Raises:
        ValueError: For invalid arguments.
    """
    if isinstance(datasets, DataSet):
        datasets = [datasets]

    n_patches = fault.n_patches
    n_components = 2 if components == "both" else 1
    n_params = n_components * n_patches

    _validate_args(
        datasets, components, smoothing, smoothing_strength, bounds, method,
        smoothing_target, n_params,
    )

    # Assemble full data system (always 2N columns from greens)
    G_full = greens(fault, datasets)
    d = stack_obs(datasets)
    W = stack_weights(datasets)

    # Select columns for requested component(s)
    G = _select_columns(G_full, n_patches, components)

    # Weight the system: G_w, d_w such that ||G_w m - d_w||^2 = (Gm-d)^T W (Gm-d)
    G_w, d_w = _apply_weights(G, d, W)

    # Build smoothing matrix (needed for auto-tuning and fixed regularization)
    L: np.ndarray | None = None
    if smoothing is not None:
        L = _build_smoothing_matrix(fault, smoothing, n_params, n_components)

    # Resolve automatic smoothing_strength
    if isinstance(smoothing_strength, str):
        if smoothing_strength == "abic":
            smoothing_strength = _find_abic_optimal(G, d, W, L, n_params)
        elif smoothing_strength == "cv":
            smoothing_strength = _find_cv_optimal(
                G_w, d_w, L, bounds, method, cv_folds, n_params,
            )

    # Build and append regularization
    if L is not None and smoothing_strength > 0:
        d_reg = _build_reg_rhs(L, smoothing_strength, smoothing_target)
        G_aug = np.vstack([G_w, np.sqrt(smoothing_strength) * L])
        d_aug = np.concatenate([d_w, d_reg])
        reg_strength: float | None = smoothing_strength
    else:
        G_aug = G_w
        d_aug = d_w
        reg_strength = None if smoothing_strength == 0.0 else smoothing_strength

    # Auto-select solver
    if method is None:
        method = _auto_select_method(bounds)

    # Solve
    m = _solve(G_aug, d_aug, bounds, method, constraints)

    # Compute fit statistics on unweighted data
    predicted = G @ m
    residuals = d - predicted
    chi2 = _compute_chi2(residuals, W, n_params)
    rms = float(np.sqrt(np.mean(residuals ** 2)))

    # Reshape slip and compute moment
    if components == "both":
        slip = np.column_stack([m[:n_patches], m[n_patches:]])
        slip_mag = np.sqrt(slip[:, 0] ** 2 + slip[:, 1] ** 2)
    else:
        slip = m.reshape(-1, 1)
        slip_mag = np.abs(m)
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
        smoothing=smoothing if reg_strength is not None else None,
        smoothing_strength=reg_strength,
        components=components,
    )


def compute_abic(
    G: np.ndarray,
    d: np.ndarray,
    W: np.ndarray,
    L: np.ndarray,
    smoothing_strength: float,
) -> float:
    """Compute the ABIC value for a given smoothing strength.

    Implements the Akaike Bayesian Information Criterion following
    Fukuda & Johnson (2008, 2010).

    Args:
        G: Green's matrix, shape (M, P).
        d: Data vector, shape (M,).
        W: Weight matrix, shape (M, M).
        L: Regularization matrix, shape (K, P).
        smoothing_strength: Regularization weight (lambda = alpha^2).

    Returns:
        ABIC scalar value (lower is better).
    """
    alpha2 = smoothing_strength
    n_data = len(d)

    # Solve the regularized system for the best-fit model
    GtWG = G.T @ W @ G
    LtL = L.T @ L
    H = GtWG + alpha2 * LtL
    m = np.linalg.solve(H, G.T @ W @ d)

    # Term 1: N * log(misfit + regularization penalty)
    residuals = d - G @ m
    misfit = float(residuals @ W @ residuals)
    penalty = alpha2 * float(m @ LtL @ m)
    total = max(misfit + penalty, 1e-300)
    abic1 = n_data * np.log(total)

    # Term 2: sum of log eigenvalues of alpha^2 * L^T L (prior)
    eig_prior = alpha2 * np.abs(np.linalg.eigvalsh(LtL))
    eig_prior = eig_prior[eig_prior > 0]
    abic2 = np.sum(np.log(eig_prior))

    # Term 3: sum of log eigenvalues of G^T W G + alpha^2 L^T L (posterior)
    eig_post = np.abs(np.linalg.eigvalsh(H))
    eig_post = eig_post[eig_post > 0]
    abic3 = np.sum(np.log(eig_post))

    return float(abic1 - abic2 + abic3)


def lcurve(
    fault: Fault,
    datasets: DataSet | list[DataSet],
    smoothing: str | np.ndarray = "laplacian",
    smoothing_range: tuple[float, float] = (1e-2, 1e6),
    n: int = 50,
    bounds: tuple[float | None, float | None] | None = None,
    method: str | None = None,
    components: str = "both",
) -> LCurveResult:
    """Sweep smoothing strength and compute the L-curve.

    Args:
        fault: Fault geometry.
        datasets: One or more geodetic datasets.
        smoothing: Regularization type.
        smoothing_range: ``(min_lambda, max_lambda)`` range to sweep.
        n: Number of lambda values to evaluate.
        bounds: Per-component slip bounds.
        method: Solver method.
        components: Which slip components to solve for.

    Returns:
        LCurveResult with sweep arrays and optimal lambda.
    """
    if isinstance(datasets, DataSet):
        datasets = [datasets]

    n_patches = fault.n_patches
    n_components = 2 if components == "both" else 1
    n_params = n_components * n_patches

    G_full = greens(fault, datasets)
    d = stack_obs(datasets)
    W = stack_weights(datasets)
    G = _select_columns(G_full, n_patches, components)
    G_w, d_w = _apply_weights(G, d, W)
    L = _build_smoothing_matrix(fault, smoothing, n_params, n_components)

    lambdas = np.geomspace(smoothing_range[0], smoothing_range[1], n)
    misfits = np.empty(n)
    model_norms = np.empty(n)

    solve_method = method if method is not None else _auto_select_method(bounds)

    for i, lam in enumerate(lambdas):
        G_aug = np.vstack([G_w, np.sqrt(lam) * L])
        d_aug = np.concatenate([d_w, np.zeros(L.shape[0])])
        m = _solve(G_aug, d_aug, bounds, solve_method, None)
        residuals = d - G @ m
        misfits[i] = np.sqrt(float(residuals @ residuals))
        model_norms[i] = np.sqrt(float((L @ m) @ (L @ m)))

    optimal = _lcurve_corner(lambdas, misfits, model_norms)

    return LCurveResult(
        smoothing_values=lambdas,
        misfits=misfits,
        model_norms=model_norms,
        optimal=optimal,
    )


def abic_curve(
    fault: Fault,
    datasets: DataSet | list[DataSet],
    smoothing: str | np.ndarray = "laplacian",
    smoothing_range: tuple[float, float] = (1e-2, 1e6),
    n: int = 50,
    components: str = "both",
) -> ABICCurveResult:
    """Sweep smoothing strength and compute the ABIC at each value.

    Also records misfit and model norm for context. The optimal lambda
    is the one that minimizes ABIC.

    Args:
        fault: Fault geometry.
        datasets: One or more geodetic datasets.
        smoothing: Regularization type.
        smoothing_range: ``(min_lambda, max_lambda)`` range to sweep.
        n: Number of lambda values to evaluate.
        components: Which slip components to solve for.

    Returns:
        ABICCurveResult with sweep arrays and optimal lambda.
    """
    if isinstance(datasets, DataSet):
        datasets = [datasets]

    n_patches = fault.n_patches
    n_components = 2 if components == "both" else 1
    n_params = n_components * n_patches

    G_full = greens(fault, datasets)
    d = stack_obs(datasets)
    W = stack_weights(datasets)
    G = _select_columns(G_full, n_patches, components)
    L = _build_smoothing_matrix(fault, smoothing, n_params, n_components)

    lambdas = np.geomspace(smoothing_range[0], smoothing_range[1], n)
    abic_values = np.empty(n)
    misfits = np.empty(n)
    model_norms = np.empty(n)

    # Precompute for efficiency
    GtWG = G.T @ W @ G
    LtL = L.T @ L
    Gtwd = G.T @ W @ d

    for i, lam in enumerate(lambdas):
        H = GtWG + lam * LtL
        m = np.linalg.solve(H, Gtwd)
        residuals = d - G @ m
        misfits[i] = np.sqrt(float(residuals @ residuals))
        model_norms[i] = np.sqrt(float((L @ m) @ (L @ m)))
        abic_values[i] = compute_abic(G, d, W, L, lam)

    optimal = float(lambdas[np.argmin(abic_values)])

    return ABICCurveResult(
        smoothing_values=lambdas,
        abic_values=abic_values,
        misfits=misfits,
        model_norms=model_norms,
        optimal=optimal,
    )


# ======================================================================
# Per-dataset diagnostics and model assessment (Phase 5)
# ======================================================================

def dataset_diagnostics(
    result: InversionResult,
    fault: Fault,
    datasets: DataSet | list[DataSet],
) -> list[DatasetDiagnostics]:
    """Compute per-dataset fit diagnostics using the hat matrix.

    For each dataset, computes chi-squared, reduced chi-squared, WRMS,
    RMS, effective DOF, and leverage using the (regularized) hat matrix
    ``H = G_w (G_w^T G_w + lambda L^T L)^{-1} G_w^T``.

    Args:
        result: Output from ``invert()``.
        fault: Fault geometry (same as passed to ``invert()``).
        datasets: Dataset(s) used in the inversion.

    Returns:
        List of ``DatasetDiagnostics``, one per dataset.
    """
    if isinstance(datasets, DataSet):
        datasets = [datasets]

    G, W, L = _rebuild_system(result, fault, datasets)
    lev = _hat_diagonal(G, W, L, result.smoothing_strength)
    residuals = result.residuals

    diags = []
    offset = 0
    for ds in datasets:
        n = ds.n_obs
        idx = slice(offset, offset + n)
        r_k = residuals[idx]
        W_k = W[idx, idx]

        chi2_k = float(r_k @ W_k @ r_k)
        lev_k = float(np.sum(lev[idx]))
        dof_k = n - lev_k
        reduced_chi2_k = chi2_k / dof_k if dof_k > 0 else float("nan")
        wrms_k = float(np.sqrt(chi2_k / n))
        rms_k = float(np.sqrt(np.mean(r_k ** 2)))

        diags.append(DatasetDiagnostics(
            chi2=chi2_k,
            reduced_chi2=reduced_chi2_k,
            wrms=wrms_k,
            rms=rms_k,
            n_obs=n,
            dof=dof_k,
            leverage=lev_k,
        ))
        offset += n

    return diags


def model_covariance(
    result: InversionResult,
    fault: Fault,
    datasets: DataSet | list[DataSet],
) -> np.ndarray:
    """Compute the model covariance matrix.

    For the unregularized case::

        Cm = (G^T W G)^{-1}

    For the regularized case (Tarantola, 2005)::

        H_inv = (G^T W G + lambda L^T L)^{-1}
        Cm = H_inv @ G^T W G @ H_inv

    Args:
        result: Output from ``invert()``.
        fault: Fault geometry.
        datasets: Dataset(s) used in the inversion.

    Returns:
        Model covariance matrix, shape (n_params, n_params).
    """
    if isinstance(datasets, DataSet):
        datasets = [datasets]

    G, W, L = _rebuild_system(result, fault, datasets)
    GtWG = G.T @ W @ G

    if L is not None and result.smoothing_strength is not None:
        LtL = L.T @ L
        H = GtWG + result.smoothing_strength * LtL
        H_inv = np.linalg.inv(H)
        return H_inv @ GtWG @ H_inv

    return np.linalg.inv(GtWG)


def model_resolution(
    result: InversionResult,
    fault: Fault,
    datasets: DataSet | list[DataSet],
) -> np.ndarray:
    """Compute the model resolution matrix.

    ``R = (G^T W G + lambda L^T L)^{-1} G^T W G``

    For perfect resolution (overdetermined, no regularization), R = I.
    With regularization, diagonal values < 1 indicate smoothed/damped
    parameters.

    Args:
        result: Output from ``invert()``.
        fault: Fault geometry.
        datasets: Dataset(s) used in the inversion.

    Returns:
        Resolution matrix, shape (n_params, n_params).
    """
    if isinstance(datasets, DataSet):
        datasets = [datasets]

    G, W, L = _rebuild_system(result, fault, datasets)
    GtWG = G.T @ W @ G

    if L is not None and result.smoothing_strength is not None:
        H = GtWG + result.smoothing_strength * (L.T @ L)
        return np.linalg.solve(H, GtWG)

    return np.linalg.solve(GtWG, GtWG)


def model_uncertainty(
    result: InversionResult,
    fault: Fault,
    datasets: DataSet | list[DataSet],
) -> np.ndarray:
    """Compute per-parameter 1-sigma uncertainty from model covariance.

    Equivalent to ``np.sqrt(np.diag(model_covariance(...)))``.

    Args:
        result: Output from ``invert()``.
        fault: Fault geometry.
        datasets: Dataset(s) used in the inversion.

    Returns:
        Uncertainty array, shape (n_params,).
    """
    Cm = model_covariance(result, fault, datasets)
    return np.sqrt(np.maximum(np.diag(Cm), 0.0))


def _rebuild_system(
    result: InversionResult,
    fault: Fault,
    datasets: list[DataSet],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Reconstruct G, W, L from a result and its inputs.

    Returns:
        (G, W, L) where G is the component-selected Green's matrix,
        W is the weight matrix, and L is the smoothing matrix (or None).
    """
    n_patches = fault.n_patches
    n_components = 2 if result.components == "both" else 1
    n_params = n_components * n_patches

    G_full = greens(fault, datasets)
    W = stack_weights(datasets)
    G = _select_columns(G_full, n_patches, result.components)

    L: np.ndarray | None = None
    if result.smoothing is not None:
        L = _build_smoothing_matrix(fault, result.smoothing, n_params, n_components)

    return G, W, L


def _hat_diagonal(
    G: np.ndarray,
    W: np.ndarray,
    L: np.ndarray | None,
    smoothing_strength: float | None,
) -> np.ndarray:
    """Compute the diagonal of the (regularized) hat matrix.

    ``H = G_w (G_w^T G_w + lambda L^T L)^{-1} G_w^T``

    where G_w = W^{1/2} G is the whitened design matrix.

    Returns:
        Leverage vector, shape (M,).
    """
    G_w, _ = _apply_weights(G, np.zeros(G.shape[0]), W)

    GwGw = G_w.T @ G_w
    if L is not None and smoothing_strength is not None and smoothing_strength > 0:
        GwGw = GwGw + smoothing_strength * (L.T @ L)

    # H = G_w @ inv(GwGw) @ G_w^T, but we only need diag(H)
    # diag(H) = row-wise sum of (G_w @ inv(GwGw)) * G_w
    A = np.linalg.solve(GwGw.T, G_w.T).T  # = G_w @ inv(GwGw)
    return np.sum(A * G_w, axis=1)


# ======================================================================
# Internal helpers
# ======================================================================

def _validate_args(
    datasets: list[DataSet],
    components: str,
    smoothing: str | np.ndarray | None,
    smoothing_strength: float | str,
    bounds: tuple[float | None, float | None] | None,
    method: str | None,
    smoothing_target: np.ndarray | None,
    n_params: int,
) -> None:
    """Validate invert() arguments."""
    for ds in datasets:
        if not isinstance(ds, DataSet):
            raise TypeError(
                f"datasets must contain DataSet instances, got {type(ds).__name__}"
            )

    if components not in _VALID_COMPONENTS:
        raise ValueError(
            f"components must be one of {_VALID_COMPONENTS}, "
            f"got {components!r}"
        )

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

    # Validate auto-tuning requires smoothing
    if isinstance(smoothing_strength, str):
        if smoothing_strength not in _VALID_STRENGTH_STRINGS:
            raise ValueError(
                f"smoothing_strength must be a float or one of "
                f"{_VALID_STRENGTH_STRINGS}, got {smoothing_strength!r}"
            )
        if smoothing is None:
            raise ValueError(
                f"smoothing_strength='{smoothing_strength}' requires "
                f"smoothing to be set"
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


def _select_columns(
    G_full: np.ndarray, n_patches: int, components: str,
) -> np.ndarray:
    """Select G matrix columns for the requested slip component(s).

    Args:
        G_full: Full Green's matrix, shape (M, 2*N).
        n_patches: Number of fault patches N.
        components: ``'both'``, ``'strike'``, or ``'dip'``.

    Returns:
        G matrix with columns for the requested components.
    """
    if components == "both":
        return G_full
    if components == "strike":
        return G_full[:, :n_patches]
    return G_full[:, n_patches:]


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
    n_components: int,
) -> np.ndarray:
    """Build the regularization matrix L.

    Args:
        fault: Fault geometry.
        smoothing: Smoothing type or custom matrix.
        n_params: Number of model parameters (n_components * n_patches).
        n_components: Number of slip components (1 or 2).

    Returns:
        Regularization matrix with n_params columns.
    """
    if isinstance(smoothing, np.ndarray):
        return smoothing

    if smoothing == "damping":
        return np.eye(n_params)

    if smoothing == "laplacian":
        L_patch = fault.laplacian
        if n_components == 1:
            return L_patch
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
    constraints: tuple[np.ndarray, np.ndarray] | None,
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

    if method == "constrained":
        return _solve_constrained(G, d, bounds, constraints)

    raise ValueError(f"Unknown method: {method!r}")


def _solve_constrained(
    G: np.ndarray,
    d: np.ndarray,
    bounds: tuple[float | None, float | None] | None,
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
    """
    n_params = G.shape[1]
    GtG = G.T @ G
    Gtd = G.T @ d

    def objective(m: np.ndarray) -> float:
        r = G @ m - d
        return 0.5 * float(r @ r)

    def gradient(m: np.ndarray) -> np.ndarray:
        return GtG @ m - Gtd

    # Build scipy bounds
    if bounds is not None:
        lower = -np.inf if bounds[0] is None else bounds[0]
        upper = np.inf if bounds[1] is None else bounds[1]
        scipy_bounds = [(lower, upper)] * n_params
    else:
        scipy_bounds = None

    # Build scipy constraints
    scipy_constraints = []
    if constraints is not None:
        C, d_ineq = constraints
        scipy_constraints.append({
            "type": "ineq",
            "fun": lambda m, C=C, d_ineq=d_ineq: d_ineq - C @ m,
            "jac": lambda m, C=C: -C,
        })

    # Initial guess: unconstrained least-squares (clipped to bounds)
    m0, _, _, _ = np.linalg.lstsq(G, d, rcond=None)
    if bounds is not None:
        lower_val = -np.inf if bounds[0] is None else bounds[0]
        upper_val = np.inf if bounds[1] is None else bounds[1]
        m0 = np.clip(m0, lower_val, upper_val)

    result = scipy.optimize.minimize(
        objective, m0, jac=gradient, method="SLSQP",
        bounds=scipy_bounds, constraints=scipy_constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return result.x


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


def _find_abic_optimal(
    G: np.ndarray,
    d: np.ndarray,
    W: np.ndarray,
    L: np.ndarray | None,
    n_params: int,
) -> float:
    """Find optimal smoothing_strength by minimizing ABIC.

    Searches in log10 space using bounded scalar optimization.

    Returns:
        Optimal smoothing_strength (lambda).
    """
    if L is None:
        raise ValueError("ABIC requires a smoothing matrix")

    def objective(log10_lam: float) -> float:
        lam = 10.0 ** log10_lam
        return compute_abic(G, d, W, L, lam)

    result = scipy.optimize.minimize_scalar(
        objective, bounds=(-6, 10), method="bounded",
    )
    return 10.0 ** result.x


def _find_cv_optimal(
    G_w: np.ndarray,
    d_w: np.ndarray,
    L: np.ndarray | None,
    bounds: tuple[float | None, float | None] | None,
    method: str | None,
    cv_folds: int,
    n_params: int,
) -> float:
    """Find optimal smoothing_strength by K-fold cross-validation.

    For each candidate lambda, partitions data rows into K folds,
    trains on K-1 folds, evaluates prediction error on the held-out
    fold, and selects the lambda with minimum mean prediction error.

    Returns:
        Optimal smoothing_strength (lambda).
    """
    if L is None:
        raise ValueError("Cross-validation requires a smoothing matrix")

    n_obs = G_w.shape[0]
    solve_method = method if method is not None else _auto_select_method(bounds)

    # Create K random disjoint folds
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_obs)
    fold_sizes = np.full(cv_folds, n_obs // cv_folds)
    fold_sizes[:n_obs % cv_folds] += 1
    folds = np.split(perm, np.cumsum(fold_sizes[:-1]))

    lambdas = np.geomspace(1e-4, 1e8, 50)
    cv_errors = np.zeros(len(lambdas))

    for i, lam in enumerate(lambdas):
        fold_errors = 0.0
        for fold in folds:
            mask = np.ones(n_obs, dtype=bool)
            mask[fold] = False
            G_train = G_w[mask]
            d_train = d_w[mask]
            G_test = G_w[fold]
            d_test = d_w[fold]

            G_aug = np.vstack([G_train, np.sqrt(lam) * L])
            d_aug = np.concatenate([d_train, np.zeros(L.shape[0])])
            m = _solve(G_aug, d_aug, bounds, solve_method, None)
            pred_test = G_test @ m
            fold_errors += float(np.sum((d_test - pred_test) ** 2))

        cv_errors[i] = fold_errors / n_obs

    return float(lambdas[np.argmin(cv_errors)])


def _lcurve_corner(
    lambdas: np.ndarray,
    misfits: np.ndarray,
    model_norms: np.ndarray,
) -> float:
    """Find the L-curve corner (maximum curvature point).

    Computes curvature of the parametric curve (log misfit, log model_norm)
    and returns the lambda at maximum curvature.

    Returns:
        Optimal lambda at the corner.
    """
    x = np.log(np.maximum(misfits, 1e-300))
    y = np.log(np.maximum(model_norms, 1e-300))

    # Parametric curvature: kappa = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2) ** 1.5

    # Exclude endpoints (unreliable gradient estimates)
    curvature[0] = -np.inf
    curvature[-1] = -np.inf

    return float(lambdas[np.argmax(curvature)])
