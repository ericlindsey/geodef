"""One-call inversion for fault slip from geodetic data.

Solves d = Gm for slip m with optional regularization and bounds.
Supports weighted least-squares, non-negative least-squares,
bounded least-squares, and constrained (QP) solvers.
Automatic hyperparameter tuning via ABIC or cross-validation.
"""

import dataclasses
import functools
from pathlib import Path

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

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save(self, fname: str | Path) -> None:
        """Save inversion result to a NumPy ``.npz`` archive.

        All numeric arrays and scalar fields are preserved.  String fields
        (``smoothing``, ``components``) are stored as object arrays.
        Custom ``smoothing`` matrices are saved as arrays; named strings are
        saved as-is.

        Args:
            fname: Output file path.  ``.npz`` extension is recommended.
        """
        smoothing_str: str
        smoothing_arr: np.ndarray | None
        if self.smoothing is None:
            smoothing_str = "__none__"
            smoothing_arr = None
        elif isinstance(self.smoothing, str):
            smoothing_str = self.smoothing
            smoothing_arr = None
        else:
            smoothing_str = "__array__"
            smoothing_arr = np.asarray(self.smoothing)

        strength = (
            np.array([float("nan")])
            if self.smoothing_strength is None
            else np.array([self.smoothing_strength])
        )

        arrays: dict = {
            "slip": self.slip,
            "slip_vector": self.slip_vector,
            "residuals": self.residuals,
            "predicted": self.predicted,
            "chi2": np.array([self.chi2]),
            "rms": np.array([self.rms]),
            "moment": np.array([self.moment]),
            "Mw": np.array([self.Mw]),
            "smoothing_str": np.array([smoothing_str]),
            "smoothing_strength": strength,
            "components": np.array([self.components]),
        }
        if smoothing_arr is not None:
            arrays["smoothing_arr"] = smoothing_arr

        np.savez_compressed(fname, **arrays)

    @classmethod
    def load(cls, fname: str | Path) -> "InversionResult":
        """Load an inversion result from a ``.npz`` archive.

        Args:
            fname: Path to a ``.npz`` file previously written by ``save()``.

        Returns:
            Reconstructed ``InversionResult`` instance.
        """
        data = np.load(fname, allow_pickle=False)

        smoothing_str = str(data["smoothing_str"][0])
        if smoothing_str == "__none__":
            smoothing: str | np.ndarray | None = None
        elif smoothing_str == "__array__":
            smoothing = data["smoothing_arr"]
        else:
            smoothing = smoothing_str

        raw_strength = float(data["smoothing_strength"][0])
        strength: float | None = None if np.isnan(raw_strength) else raw_strength

        return cls(
            slip=data["slip"],
            slip_vector=data["slip_vector"],
            residuals=data["residuals"],
            predicted=data["predicted"],
            chi2=float(data["chi2"][0]),
            rms=float(data["rms"][0]),
            moment=float(data["moment"][0]),
            Mw=float(data["Mw"][0]),
            smoothing=smoothing,
            smoothing_strength=strength,
            components=str(data["components"][0]),
        )

    def save_table(self, fname: str | Path, fault: "Fault") -> None:
        """Save slip distribution as a human-readable text table.

        Writes a ``#``-prefixed header with summary statistics followed by
        one data row per fault patch.  For rectangular faults the columns
        are ``lon lat depth_m strike dip length_m width_m slip_strike_m
        slip_dip_m``; for triangular faults ``length_m`` and ``width_m`` are
        replaced by ``area_m2``.

        Args:
            fault: Fault geometry matching this result.
            fname: Output file path.
        """
        slip_2d = self.slip if self.slip.ndim == 2 else self.slip[:, np.newaxis]
        n_comp = slip_2d.shape[1]

        smoothing_desc = (
            "none"
            if self.smoothing is None
            else (self.smoothing if isinstance(self.smoothing, str) else "custom")
        )
        strength_desc = (
            "N/A" if self.smoothing_strength is None
            else f"{self.smoothing_strength:.6g}"
        )

        header_lines = [
            "geodef InversionResult",
            f"components: {self.components}",
            f"smoothing: {smoothing_desc}, strength: {strength_desc}",
            f"chi2_reduced: {self.chi2:.6g}",
            f"rms: {self.rms:.6g} m",
            f"moment: {self.moment:.6g} N-m",
            f"Mw: {self.Mw:.4f}",
        ]

        if fault.engine == "okada":
            col_names = "lon lat depth_m strike dip length_m width_m"
            geom = np.column_stack([
                fault._lon, fault._lat, fault._depth,
                fault._strike, fault._dip,
                fault._length, fault._width,
            ])
        else:
            col_names = "lon lat depth_m strike dip area_m2"
            geom = np.column_stack([
                fault._lon, fault._lat, fault._depth,
                fault._strike, fault._dip,
                fault.areas,
            ])

        slip_cols = "  ".join(
            ["slip_strike_m", "slip_dip_m"][:n_comp]
        )
        header_lines.append(f"{col_names}  {slip_cols}")

        data = np.column_stack([geom, slip_2d])
        header = "\n".join(header_lines)
        np.savetxt(Path(fname), data, header=header, fmt="%.6f")


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
        misfits: Data misfit norm ||Gm - d|| at each lambda.
        model_norms: Regularized model norm ||Lm|| at each lambda.
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
        misfits: Data misfit norm ||Gm - d|| at each lambda.
        model_norms: Regularized model norm ||Lm|| at each lambda.
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


# ======================================================================
# LinearSystem: persistent prepared system with cached matrix products
# ======================================================================

class LinearSystem:
    """Prepared linear system for fault slip inversion.

    Encapsulates the Green's matrix, data vector, weight matrix, and
    smoothing matrix for a given fault-dataset pair.  Expensive derived
    products (G^T W G, L^T L, G^T W d) are computed on first access and
    cached, so they are shared across ``invert``, ``lcurve``,
    ``abic_curve``, and the post-inversion analysis methods.

    Use this class directly when performing multiple analyses on the same
    fault and datasets (e.g. comparing L-curve and ABIC, then running
    diagnostics after inversion).  The module-level convenience functions
    (``invert``, ``lcurve``, etc.) create a ``LinearSystem`` internally
    and are fully backward-compatible.

    Args:
        fault: Fault geometry.
        datasets: One or more geodetic datasets.
        smoothing: Regularization type — ``'laplacian'``, ``'damping'``,
            ``'stresskernel'``, a custom matrix, or ``None``.
        components: Slip components to solve for: ``'both'`` (default),
            ``'strike'``, or ``'dip'``.

    Examples:
        >>> sys = LinearSystem(fault, [gnss, insar], smoothing='laplacian')
        >>> lc = sys.lcurve()
        >>> result = sys.invert(smoothing_strength=lc.optimal)
        >>> diag = sys.dataset_diagnostics(result)
    """

    def __init__(
        self,
        fault: Fault,
        datasets: DataSet | list[DataSet],
        smoothing: str | np.ndarray | None = None,
        components: str = "both",
    ) -> None:
        if isinstance(datasets, DataSet):
            datasets = [datasets]
        for ds in datasets:
            if not isinstance(ds, DataSet):
                raise TypeError(
                    f"datasets must contain DataSet instances, got {type(ds).__name__}"
                )
        if components not in _VALID_COMPONENTS:
            raise ValueError(
                f"components must be one of {_VALID_COMPONENTS}, got {components!r}"
            )

        self.fault = fault
        self.datasets = datasets
        self.smoothing = smoothing
        self.components = components

        n_patches = fault.n_patches
        n_components = 2 if components == "both" else 1
        self._n_patches = n_patches
        self._n_params = n_components * n_patches

        G_full = greens(fault, datasets)
        self.d = stack_obs(datasets)
        self.W = stack_weights(datasets)
        self.G = _select_columns(G_full, n_patches, components)
        self.G_w, self.d_w = _apply_weights(self.G, self.d, self.W)
        self.L: np.ndarray | None = (
            _build_smoothing_matrix(fault, smoothing, self._n_params, n_components)
            if smoothing is not None else None
        )

    @functools.cached_property
    def GtWG(self) -> np.ndarray:
        """G^T W G — normal equations matrix (without regularization)."""
        return self.G_w.T @ self.G_w

    @functools.cached_property
    def LtL(self) -> np.ndarray:
        """L^T L — regularization normal equations matrix.

        Raises:
            AttributeError: If the system was constructed without smoothing.
        """
        if self.L is None:
            raise AttributeError(
                "LtL is not available: LinearSystem has no smoothing matrix"
            )
        return self.L.T @ self.L

    @functools.cached_property
    def Gtwd(self) -> np.ndarray:
        """G^T W d — normal equations right-hand side."""
        return self.G_w.T @ self.d_w

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _abic_value(
        self, smoothing_strength: float,
    ) -> tuple[float, float, float]:
        """ABIC, misfit norm, and model norm at a given smoothing strength.

        Uses cached GtWG, LtL, and Gtwd.  ``eig_LtL`` (lambda-independent)
        is computed on the first call and cached in ``self.__dict__``.

        The ABIC formula (Fukuda & Johnson 2008, 2010) requires the weighted
        misfit ``r^T W r`` internally.  The returned ``misfit_norm`` is the
        unweighted ``||Gm - d||`` for consistent plotting against lcurve.

        Args:
            smoothing_strength: Regularization weight lambda.

        Returns:
            (abic, misfit_norm, model_norm) where misfit_norm = ||Gm - d||
            and model_norm = ||Lm||.
        """
        alpha2 = smoothing_strength
        n_data = len(self.d)

        H = self.GtWG + alpha2 * self.LtL
        m = np.linalg.solve(H, self.Gtwd)

        residuals = self.d - self.G @ m
        misfit_weighted = float(residuals @ self.W @ residuals)
        penalty = alpha2 * float(m @ self.LtL @ m)
        total = max(misfit_weighted + penalty, 1e-300)
        abic1 = n_data * np.log(total)

        # eig_LtL is lambda-independent — compute once and cache
        eig_LtL: np.ndarray | None = self.__dict__.get("_eig_LtL")
        if eig_LtL is None:
            eig_LtL = np.linalg.eigvalsh(self.LtL)
            self.__dict__["_eig_LtL"] = eig_LtL

        eig_prior = alpha2 * np.abs(eig_LtL)
        eig_prior = eig_prior[eig_prior > 0]
        abic2 = float(np.sum(np.log(eig_prior)))

        eig_post = np.abs(np.linalg.eigvalsh(H))
        eig_post = eig_post[eig_post > 0]
        abic3 = float(np.sum(np.log(eig_post)))

        abic = abic1 - abic2 + abic3
        misfit_norm = float(np.sqrt(residuals @ residuals))
        model_norm = float(np.sqrt((self.L @ m) @ (self.L @ m)))
        return abic, misfit_norm, model_norm

    def _optimal_abic(self) -> float:
        """Find optimal smoothing strength by minimizing ABIC.

        Returns:
            Optimal lambda.
        """
        if self.L is None:
            raise ValueError("ABIC requires a smoothing matrix")

        def objective(log10_lam: float) -> float:
            return self._abic_value(10.0 ** log10_lam)[0]

        result = scipy.optimize.minimize_scalar(
            objective, bounds=(-6, 10), method="bounded",
        )
        return 10.0 ** result.x

    def _optimal_cv(
        self,
        bounds: tuple[float | None, float | None] | None,
        method: str | None,
        cv_folds: int,
    ) -> float:
        """Find optimal smoothing strength by K-fold cross-validation.

        Args:
            bounds: Per-component slip bounds.
            method: Solver method.
            cv_folds: Number of folds.

        Returns:
            Optimal lambda.
        """
        if self.L is None:
            raise ValueError("Cross-validation requires a smoothing matrix")

        n_obs = self.G_w.shape[0]
        solve_method = method if method is not None else _auto_select_method(bounds)

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
                G_aug = np.vstack([self.G_w[mask], np.sqrt(lam) * self.L])
                d_aug = np.concatenate([self.d_w[mask], np.zeros(self.L.shape[0])])
                m = _solve(G_aug, d_aug, bounds, solve_method, None)
                pred_test = self.G_w[fold] @ m
                fold_errors += float(np.sum((self.d_w[fold] - pred_test) ** 2))
            cv_errors[i] = fold_errors / n_obs

        return float(lambdas[np.argmin(cv_errors)])

    def _hat_diagonal(self, smoothing_strength: float | None) -> np.ndarray:
        """Diagonal of the hat matrix H = G_w (G_w^T G_w + λ L^T L)^{-1} G_w^T.

        Args:
            smoothing_strength: Regularization weight, or None.

        Returns:
            Leverage vector, shape (M,).
        """
        H = self.GtWG.copy()
        if self.L is not None and smoothing_strength is not None and smoothing_strength > 0:
            H += smoothing_strength * self.LtL
        A = np.linalg.solve(H.T, self.G_w.T).T
        return np.sum(A * self.G_w, axis=1)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def invert(
        self,
        smoothing_strength: float | str = 0.0,
        bounds: tuple[float | None, float | None] | None = None,
        method: str | None = None,
        smoothing_target: np.ndarray | None = None,
        constraints: tuple[np.ndarray, np.ndarray] | None = None,
        cv_folds: int = 5,
    ) -> InversionResult:
        """Invert for fault slip using this prepared system.

        Args:
            smoothing_strength: Scalar regularization weight, or
                ``'abic'`` / ``'cv'`` for automatic tuning.
            bounds: Per-component slip bounds ``(lower, upper)``.
            method: Solver — ``'wls'``, ``'nnls'``, ``'bounded_ls'``,
                or ``'constrained'``. Auto-selected from bounds if None.
            smoothing_target: Reference model, shape (n_params,).
                Regularizes toward this target instead of zero.
            constraints: Inequality constraints ``(C, d_ineq)`` such
                that ``C @ m <= d_ineq``.
            cv_folds: Number of folds for cross-validation (default 5).

        Returns:
            InversionResult with slip, residuals, and fit statistics.

        Raises:
            ValueError: For invalid arguments.
        """
        _validate_args(
            self.datasets, self.components, self.smoothing, smoothing_strength,
            bounds, method, smoothing_target, self._n_params,
        )

        if isinstance(smoothing_strength, str):
            if smoothing_strength == "abic":
                smoothing_strength = self._optimal_abic()
            elif smoothing_strength == "cv":
                smoothing_strength = self._optimal_cv(bounds, method, cv_folds)

        if self.L is not None and smoothing_strength > 0:
            d_reg = _build_reg_rhs(self.L, smoothing_strength, smoothing_target)
            G_aug = np.vstack([self.G_w, np.sqrt(smoothing_strength) * self.L])
            d_aug = np.concatenate([self.d_w, d_reg])
            reg_strength: float | None = smoothing_strength
        else:
            G_aug = self.G_w
            d_aug = self.d_w
            reg_strength = None if smoothing_strength == 0.0 else smoothing_strength

        if method is None:
            method = _auto_select_method(bounds)

        m = _solve(G_aug, d_aug, bounds, method, constraints)

        predicted = self.G @ m
        residuals = self.d - predicted
        chi2 = _compute_chi2(residuals, self.W, self._n_params)
        rms = float(np.sqrt(np.mean(residuals ** 2)))

        if self.components == "both":
            slip = np.column_stack([m[:self._n_patches], m[self._n_patches:]])
            slip_mag = np.sqrt(slip[:, 0] ** 2 + slip[:, 1] ** 2)
        else:
            slip = m.reshape(-1, 1)
            slip_mag = np.abs(m)
        moment = self.fault.moment(slip_mag)
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
            smoothing=self.smoothing if reg_strength is not None else None,
            smoothing_strength=reg_strength,
            components=self.components,
        )

    def lcurve(
        self,
        smoothing_range: tuple[float, float] = (1e-2, 1e6),
        n: int = 50,
        bounds: tuple[float | None, float | None] | None = None,
        method: str | None = None,
    ) -> LCurveResult:
        """Sweep smoothing strength and compute the L-curve.

        For unconstrained (``wls``) solves, GtWG, LtL, and Gtwd are used
        directly so each iteration is a single linear solve with no matrix
        assembly.  For constrained solves the augmented system is used.

        Misfits are the unweighted norm ``||Gm - d||``.

        Args:
            smoothing_range: ``(min_lambda, max_lambda)`` range to sweep.
            n: Number of lambda values to evaluate.
            bounds: Per-component slip bounds.
            method: Solver method.

        Returns:
            LCurveResult with sweep arrays and optimal lambda.

        Raises:
            ValueError: If the system has no smoothing matrix.
        """
        if self.L is None:
            raise ValueError("lcurve requires a smoothing matrix")

        lambdas = np.geomspace(smoothing_range[0], smoothing_range[1], n)
        misfits = np.empty(n)
        model_norms = np.empty(n)

        solve_method = method if method is not None else _auto_select_method(bounds)

        if solve_method == "wls":
            for i, lam in enumerate(lambdas):
                H = self.GtWG + lam * self.LtL
                m = np.linalg.solve(H, self.Gtwd)
                residuals = self.d - self.G @ m
                misfits[i] = float(np.sqrt(residuals @ residuals))
                model_norms[i] = float(np.sqrt((self.L @ m) @ (self.L @ m)))
        else:
            for i, lam in enumerate(lambdas):
                G_aug = np.vstack([self.G_w, np.sqrt(lam) * self.L])
                d_aug = np.concatenate([self.d_w, np.zeros(self.L.shape[0])])
                m = _solve(G_aug, d_aug, bounds, solve_method, None)
                residuals = self.d - self.G @ m
                misfits[i] = float(np.sqrt(residuals @ residuals))
                model_norms[i] = float(np.sqrt((self.L @ m) @ (self.L @ m)))

        optimal = _lcurve_corner(lambdas, misfits, model_norms)
        return LCurveResult(
            smoothing_values=lambdas,
            misfits=misfits,
            model_norms=model_norms,
            optimal=optimal,
        )

    def abic_curve(
        self,
        smoothing_range: tuple[float, float] = (1e-2, 1e6),
        n: int = 50,
    ) -> ABICCurveResult:
        """Sweep smoothing strength and compute the ABIC at each value.

        GtWG, LtL, Gtwd, and eig_LtL are all computed once and reused
        across all iterations.  Misfits are the unweighted norm ``||Gm - d||``,
        consistent with ``lcurve``.

        Args:
            smoothing_range: ``(min_lambda, max_lambda)`` range to sweep.
            n: Number of lambda values to evaluate.

        Returns:
            ABICCurveResult with sweep arrays and optimal lambda.

        Raises:
            ValueError: If the system has no smoothing matrix.
        """
        if self.L is None:
            raise ValueError("abic_curve requires a smoothing matrix")

        lambdas = np.geomspace(smoothing_range[0], smoothing_range[1], n)
        abic_values = np.empty(n)
        misfits = np.empty(n)
        model_norms = np.empty(n)

        for i, lam in enumerate(lambdas):
            abic_values[i], misfits[i], model_norms[i] = self._abic_value(lam)

        optimal = float(lambdas[np.argmin(abic_values)])
        return ABICCurveResult(
            smoothing_values=lambdas,
            abic_values=abic_values,
            misfits=misfits,
            model_norms=model_norms,
            optimal=optimal,
        )

    def dataset_diagnostics(
        self, result: InversionResult,
    ) -> list[DatasetDiagnostics]:
        """Compute per-dataset fit diagnostics using the hat matrix.

        Args:
            result: Output from ``invert()``.

        Returns:
            List of ``DatasetDiagnostics``, one per dataset.
        """
        lev = self._hat_diagonal(result.smoothing_strength)
        residuals = result.residuals

        diags = []
        offset = 0
        for ds in self.datasets:
            n = ds.n_obs
            idx = slice(offset, offset + n)
            r_k = residuals[idx]
            W_k = self.W[idx, idx]

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

    def model_covariance(self, result: InversionResult) -> np.ndarray:
        """Compute the model covariance matrix.

        For the unregularized case::

            Cm = (G^T W G)^{-1}

        For the regularized case (Tarantola, 2005)::

            H_inv = (G^T W G + lambda L^T L)^{-1}
            Cm = H_inv @ G^T W G @ H_inv

        Args:
            result: Output from ``invert()``.

        Returns:
            Model covariance matrix, shape (n_params, n_params).
        """
        if self.L is not None and result.smoothing_strength is not None:
            H = self.GtWG + result.smoothing_strength * self.LtL
            H_inv = np.linalg.inv(H)
            return H_inv @ self.GtWG @ H_inv
        return np.linalg.inv(self.GtWG)

    def model_resolution(self, result: InversionResult) -> np.ndarray:
        """Compute the model resolution matrix.

        ``R = (G^T W G + lambda L^T L)^{-1} G^T W G``

        Args:
            result: Output from ``invert()``.

        Returns:
            Resolution matrix, shape (n_params, n_params).
        """
        if self.L is not None and result.smoothing_strength is not None:
            H = self.GtWG + result.smoothing_strength * self.LtL
            return np.linalg.solve(H, self.GtWG)
        return np.linalg.solve(self.GtWG, self.GtWG)

    def model_uncertainty(self, result: InversionResult) -> np.ndarray:
        """Compute per-parameter 1-sigma uncertainty from model covariance.

        Args:
            result: Output from ``invert()``.

        Returns:
            Uncertainty array, shape (n_params,).
        """
        Cm = self.model_covariance(result)
        return np.sqrt(np.maximum(np.diag(Cm), 0.0))


# ======================================================================
# Module-level convenience functions (backward-compatible wrappers)
# ======================================================================

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
    sys = LinearSystem(fault, datasets, smoothing, components)
    return sys.invert(smoothing_strength, bounds, method, smoothing_target,
                      constraints, cv_folds)


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

    GtWG = G.T @ W @ G
    LtL = L.T @ L
    H = GtWG + alpha2 * LtL
    m = np.linalg.solve(H, G.T @ W @ d)

    residuals = d - G @ m
    misfit = float(residuals @ W @ residuals)
    penalty = alpha2 * float(m @ LtL @ m)
    total = max(misfit + penalty, 1e-300)
    abic1 = n_data * np.log(total)

    eig_prior = alpha2 * np.abs(np.linalg.eigvalsh(LtL))
    eig_prior = eig_prior[eig_prior > 0]
    abic2 = np.sum(np.log(eig_prior))

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
    sys = LinearSystem(fault, datasets, smoothing, components)
    return sys.lcurve(smoothing_range, n, bounds, method)


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
    sys = LinearSystem(fault, datasets, smoothing, components)
    return sys.abic_curve(smoothing_range, n)


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
    sys = LinearSystem(fault, datasets, result.smoothing, result.components)
    return sys.dataset_diagnostics(result)


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
    sys = LinearSystem(fault, datasets, result.smoothing, result.components)
    return sys.model_covariance(result)


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
    sys = LinearSystem(fault, datasets, result.smoothing, result.components)
    return sys.model_resolution(result)


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
    sys = LinearSystem(fault, datasets, result.smoothing, result.components)
    return sys.model_uncertainty(result)


# ======================================================================
# Private helpers
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

    if bounds is not None:
        lower = -np.inf if bounds[0] is None else bounds[0]
        upper = np.inf if bounds[1] is None else bounds[1]
        scipy_bounds = [(lower, upper)] * n_params
    else:
        scipy_bounds = None

    scipy_constraints = []
    if constraints is not None:
        C, d_ineq = constraints
        scipy_constraints.append({
            "type": "ineq",
            "fun": lambda m, C=C, d_ineq=d_ineq: d_ineq - C @ m,
            "jac": lambda m, C=C: -C,
        })

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

    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2) ** 1.5

    curvature[0] = -np.inf
    curvature[-1] = -np.inf

    return float(lambdas[np.argmax(curvature)])
