"""Prepared linear system: assembly, weighting, validation, and caching.

Private submodule of :mod:`geodef.invert`. ``LinearSystem`` assembles
G, d, W, and L for a fault-dataset pair, caches the expensive derived
products, and hosts the invert/lcurve/abic sweeps that reuse them.
"""

import functools
import hashlib

import numpy as np
import scipy.linalg

from geodef import backend
from geodef.data import DataSet
from geodef.fault import Fault, moment_to_magnitude
from geodef.greens import matrix, select_slip_columns, stack_obs, stack_weights
from geodef.invert._regularization import (
    _VALID_REGULARIZATION_STRINGS,
    _build_reg_rhs,
    _build_regularization_matrix,
)
from geodef.invert._results import (
    ABICCurveResult,
    DatasetDiagnostics,
    InversionResult,
    LCurveResult,
    _physical_components,
)
from geodef.invert._solvers import (
    _VALID_METHODS,
    BoundsSpec,
    _auto_select_method,
    _compute_reduced_chi2,
    _expand_bounds,
    _ExpandedBounds,
    _rank_positive_eigs,
    _solve,
)
from geodef.slip import magnitude

_VALID_STRENGTH_STRINGS = {"abic", "cv"}
_VALID_COMPONENTS = {"both", "strike", "dip", "rake", "azimuth", "plate"}


class LinearSystem:
    """Prepared linear system for fault slip inversion.

    Encapsulates the Green's matrix, data vector, weight matrix, and
    regularization matrix for a given fault-dataset pair.  Expensive derived
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
        regularization: Regularization type — ``'laplacian'``, ``'damping'``,
            ``'stresskernel'``, a custom matrix, or ``None``.
        components: Slip components to solve for: ``'both'`` (default),
            ``'strike'``, ``'dip'``, ``'rake'``, ``'azimuth'``, or ``'plate'``.
        rake: Constant local rake for ``components='rake'``.
        slip_azimuth: Constant geographic direction for
            ``components='azimuth'``.
        plate_rake: Scalar or per-patch large-scale direction in local rake
            coordinates for ``components='plate'``.

    Examples:
        >>> sys = LinearSystem(fault, [gnss, insar], regularization='laplacian')
        >>> lc = sys.lcurve()
        >>> result = sys.invert(regularization_strength=lc.optimal)
        >>> diag = sys.dataset_diagnostics(result)
    """

    def __init__(
        self,
        fault: Fault,
        datasets: DataSet | list[DataSet],
        regularization: str | np.ndarray | None = None,
        components: str = "both",
        rake: float | None = None,
        slip_azimuth: float | None = None,
        plate_rake: float | np.ndarray | None = None,
    ) -> None:
        if isinstance(datasets, DataSet):
            datasets = [datasets]
        for ds in datasets:
            if not isinstance(ds, DataSet):
                raise TypeError(
                    f"datasets must contain DataSet instances, got {type(ds).__name__}"
                )
        if not datasets:
            raise ValueError("datasets must contain at least one DataSet")
        semantics = {(dataset.quantity, dataset.units) for dataset in datasets}
        if len(semantics) != 1:
            raise ValueError(
                "joint datasets must use the same quantity and units; "
                f"received {sorted(semantics)}"
            )

        explicit_names = [
            dataset.dataset_name
            for dataset in datasets
            if dataset.dataset_name is not None
        ]
        if len(explicit_names) != len(set(explicit_names)):
            raise ValueError("explicit dataset names must be unique")
        used_names = set(explicit_names)
        generated_counts: dict[str, int] = {}
        dataset_names: list[str] = []
        for dataset in datasets:
            if dataset.dataset_name is not None:
                dataset_names.append(dataset.dataset_name)
                continue
            base = type(dataset).__name__.lower()
            count = generated_counts.get(base, 0) + 1
            candidate = base if count == 1 else f"{base}_{count}"
            while candidate in used_names:
                count += 1
                candidate = f"{base}_{count}"
            generated_counts[base] = count
            used_names.add(candidate)
            dataset_names.append(candidate)

        offset = 0
        dataset_slices = []
        for dataset in datasets:
            dataset_slices.append(slice(offset, offset + dataset.n_obs))
            offset += dataset.n_obs
        if components not in _VALID_COMPONENTS:
            raise ValueError(
                f"components must be one of {_VALID_COMPONENTS}, got {components!r}"
            )
        if components == "rake" and rake is None:
            raise ValueError("components='rake' requires a rake angle in degrees")
        if rake is not None and components != "rake":
            raise ValueError(
                f"rake angle is only used with components='rake', "
                f"got components={components!r}"
            )
        if components == "azimuth" and slip_azimuth is None:
            raise ValueError("components='azimuth' requires a slip_azimuth in degrees")
        if slip_azimuth is not None and components != "azimuth":
            raise ValueError(
                f"slip_azimuth is only used with components='azimuth', "
                f"got components={components!r}"
            )
        if components == "plate" and plate_rake is None:
            raise ValueError("components='plate' requires plate_rake")
        if plate_rake is not None and components != "plate":
            raise ValueError(
                "plate_rake is only used with components='plate', "
                f"got components={components!r}"
            )

        self.fault = fault
        self.datasets = datasets
        self.dataset_names = tuple(dataset_names)
        self.dataset_slices = tuple(dataset_slices)
        self.quantity, self.units = next(iter(semantics))
        self.regularization = regularization
        self.components = components
        self.rake = rake
        self.slip_azimuth = slip_azimuth
        self.plate_rake = (
            None
            if plate_rake is None
            else np.broadcast_to(
                np.asarray(plate_rake, dtype=float), (fault.n_patches,)
            ).copy()
        )

        n_patches = fault.n_patches
        n_components = 2 if components in {"both", "plate"} else 1
        self._n_patches = n_patches
        self._n_params = n_components * n_patches

        G_full = matrix(fault, datasets)
        self.d = stack_obs(datasets)
        self.W = stack_weights(datasets)
        self.G = select_slip_columns(
            G_full,
            n_patches,
            components,
            rake,
            fault_strike=fault.strike,
            slip_azimuth=slip_azimuth,
            plate_rake=self.plate_rake,
        )
        self.G_w, self.d_w = _apply_weights(self.G, self.d, self.W)
        self.L: np.ndarray | None = (
            _build_regularization_matrix(
                fault,
                regularization,
                self._n_params,
                n_components,
                components,
                rake,
                slip_azimuth,
                self.plate_rake,
            )
            if regularization is not None
            else None
        )

    @functools.cached_property
    def GtWG(self) -> np.ndarray:
        """G^T W G — normal equations matrix (without regularization)."""
        return self.G_w.T @ self.G_w

    @functools.cached_property
    def LtL(self) -> np.ndarray:
        """L^T L — regularization normal equations matrix.

        Raises:
            AttributeError: If the system was constructed without regularization.
        """
        if self.L is None:
            raise AttributeError(
                "LtL is not available: LinearSystem has no regularization matrix"
            )
        return self.L.T @ self.L

    @functools.cached_property
    def Gtwd(self) -> np.ndarray:
        """G^T W d — normal equations right-hand side."""
        return self.G_w.T @ self.d_w

    def condition_report(
        self, regularization_strength: float | None = None
    ) -> dict[str, float]:
        """Conditioning diagnostics for the prepared (whitened) system.

        Reports how close the least-squares problem is to exhausting the
        active floating-point precision. ``cond_normal_equations`` is
        the square of ``cond_G`` — the conditioning a normal-equations
        solve actually experiences — so values approaching ``1/eps``
        (about 4.5e15 in float64) mean the unregularized solution is
        dominated by roundoff.

        Args:
            regularization_strength: If given (and the system has a
                regularization operator), also report ``cond_H`` for
                ``H = G^T W G + lambda L^T L`` at that lambda — the
                matrix the regularized solve factorizes.

        Returns:
            Dict with ``cond_G`` (whitened Green's matrix),
            ``cond_normal_equations``, ``rank_G`` (numerical rank),
            ``n_params``, and optionally ``cond_H``.
        """
        singular_values = np.linalg.svd(self.G_w, compute_uv=False)
        largest = float(singular_values[0])
        smallest = float(singular_values[-1])
        eps = float(np.finfo(self.G_w.dtype).eps)
        tolerance = max(self.G_w.shape) * eps * largest
        rank = int(np.sum(singular_values > tolerance))
        cond = float("inf") if smallest == 0.0 else largest / smallest
        report: dict[str, float] = {
            "cond_G": cond,
            "cond_normal_equations": cond**2,
            "rank_G": rank,
            "n_params": int(self.G.shape[1]),
        }
        if regularization_strength is not None and self.L is not None:
            H = self.GtWG + regularization_strength * self.LtL
            report["cond_H"] = float(np.linalg.cond(H))
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _abic_value(
        self,
        regularization_strength: float,
    ) -> tuple[float, float, float]:
        """ABIC, misfit norm, and model norm at a given regularization strength.

        Uses cached GtWG, LtL, and Gtwd.  ``eig_LtL`` (lambda-independent)
        is computed on the first call and cached in ``self.__dict__``.

        The ABIC formula (Fukuda & Johnson 2008, 2010) requires the weighted
        misfit ``r^T W r`` internally.  The returned ``misfit_norm`` is the
        unweighted ``||Gm - d||`` for consistent plotting against lcurve.

        Args:
            regularization_strength: Regularization weight lambda.

        Returns:
            (abic, misfit_norm, model_norm) where misfit_norm = ||Gm - d||
            and model_norm = ||Lm||.
        """
        alpha2 = regularization_strength
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

        eig_prior = alpha2 * _rank_positive_eigs(eig_LtL)
        abic2 = float(np.sum(np.log(eig_prior)))

        eig_post = _rank_positive_eigs(np.linalg.eigvalsh(H))
        abic3 = float(np.sum(np.log(eig_post)))

        abic = abic1 - abic2 + abic3
        misfit_norm = float(np.sqrt(residuals @ residuals))
        model_norm = float(np.sqrt((self.L @ m) @ (self.L @ m)))
        return abic, misfit_norm, model_norm

    def _abic_sweep_jax(
        self,
        lambdas: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the ABIC sweep as one batched JAX computation.

        Computes the same quantities as ``_abic_value`` at every lambda,
        but the linear solves, log-determinants, and norms are batched
        across the lambda axis so XLA fuses the whole sweep. The
        posterior log-determinant uses ``slogdet(H)`` instead of filtered
        eigenvalues; for the positive-definite systems swept here the two
        agree.

        Args:
            lambdas: Regularization weights to sweep, shape (n,).

        Returns:
            Arrays ``(abic_values, misfits, model_norms)``, each (n,).
        """
        import jax.numpy as jnp

        assert self.L is not None
        n_data = len(self.d)
        lam = jnp.asarray(lambdas)
        GtWG = jnp.asarray(self.GtWG)
        LtL = jnp.asarray(self.LtL)
        Gtwd = jnp.asarray(self.Gtwd)
        G = jnp.asarray(self.G)
        d = jnp.asarray(self.d)
        W = jnp.asarray(self.W)
        L = jnp.asarray(self.L)

        H = GtWG[None, :, :] + lam[:, None, None] * LtL[None, :, :]
        rhs = jnp.broadcast_to(Gtwd, (len(lambdas), len(Gtwd)))
        m = jnp.linalg.solve(H, rhs[..., None]).squeeze(-1)

        r = d[None, :] - m @ G.T
        misfit_weighted = jnp.einsum("nm,mk,nk->n", r, W, r)
        penalty = lam * jnp.einsum("np,pq,nq->n", m, LtL, m)
        total = misfit_weighted + penalty
        total = jnp.maximum(total, jnp.finfo(total.dtype).tiny)
        abic1 = n_data * backend.to_numpy(jnp.log(total))

        # eig_LtL is lambda-independent — compute once and cache, and
        # split sum(log(lam*|e|)) into k*log(lam) + sum(log|e|)
        eig_LtL: np.ndarray | None = self.__dict__.get("_eig_LtL")
        if eig_LtL is None:
            eig_LtL = np.linalg.eigvalsh(self.LtL)
            self.__dict__["_eig_LtL"] = eig_LtL
        eig_pos = _rank_positive_eigs(eig_LtL)
        abic2 = len(eig_pos) * np.log(lambdas) + np.sum(np.log(eig_pos))

        _, abic3 = jnp.linalg.slogdet(H)

        abic = abic1 - abic2 + backend.to_numpy(abic3)
        misfits = np.sqrt(np.sum(backend.to_numpy(r) ** 2, axis=1))
        Lm = backend.to_numpy(m @ L.T)
        model_norms = np.sqrt(np.sum(Lm**2, axis=1))
        return abic, misfits, model_norms

    def _optimal_abic(self) -> float:
        """Find optimal regularization strength by minimizing ABIC.

        Returns:
            Optimal lambda.
        """
        if self.L is None:
            raise ValueError("ABIC requires a regularization matrix")

        def objective(log10_lam: float) -> float:
            return self._abic_value(10.0**log10_lam)[0]

        result = scipy.optimize.minimize_scalar(
            objective,
            bounds=(-6, 10),
            method="bounded",
        )
        return 10.0**result.x

    def _optimal_cv(
        self,
        bounds: _ExpandedBounds,
        method: str | None,
        cv_folds: int,
    ) -> float:
        """Find optimal regularization strength by K-fold cross-validation.

        Args:
            bounds: Expanded per-parameter slip bounds.
            method: Solver method.
            cv_folds: Number of folds.

        Returns:
            Optimal lambda.
        """
        if self.L is None:
            raise ValueError("Cross-validation requires a regularization matrix")

        n_obs = self.G_w.shape[0]
        solve_method = method if method is not None else _auto_select_method(bounds)

        rng = np.random.default_rng(0)
        perm = rng.permutation(n_obs)
        fold_sizes = np.full(cv_folds, n_obs // cv_folds)
        fold_sizes[: n_obs % cv_folds] += 1
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

    def _hat_diagonal(self, regularization_strength: float | None) -> np.ndarray:
        """Diagonal of the hat matrix H = G_w (G_w^T G_w + λ L^T L)^{-1} G_w^T.

        Args:
            regularization_strength: Regularization weight, or None.

        Returns:
            Leverage vector, shape (M,).
        """
        H = self.GtWG.copy()
        if (
            self.L is not None
            and regularization_strength is not None
            and regularization_strength > 0
        ):
            H += regularization_strength * self.LtL
        A = np.linalg.solve(H.T, self.G_w.T).T
        return np.sum(A * self.G_w, axis=1)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def invert(
        self,
        regularization_strength: float | str = 0.0,
        bounds: BoundsSpec = None,
        method: str | None = None,
        regularization_target: np.ndarray | None = None,
        constraints: tuple[np.ndarray, np.ndarray] | None = None,
        cv_folds: int = 5,
    ) -> InversionResult:
        """Invert for fault slip using this prepared system.

        Args:
            regularization_strength: Scalar regularization weight, or
                ``'abic'`` / ``'cv'`` for automatic tuning.
            bounds: Per-component slip bounds ``(lower, upper)``.
            method: Solver — ``'wls'``, ``'nnls'``, ``'bounded_ls'``,
                or ``'constrained'``. Auto-selected from bounds if None.
            regularization_target: Reference vector, shape ``(n_params,)``.
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
            self.datasets,
            self.components,
            self.regularization,
            regularization_strength,
            bounds,
            method,
            regularization_target,
            self._n_params,
            self.rake,
            self.slip_azimuth,
            self.plate_rake,
        )

        exp_bounds = _expand_bounds(
            bounds, self._n_patches, self._n_params // self._n_patches
        )

        regularization_selection = (
            regularization_strength
            if isinstance(regularization_strength, str)
            else None
        )
        if isinstance(regularization_strength, str):
            if regularization_strength == "abic":
                strength = self._optimal_abic()
            elif regularization_strength == "cv":
                strength = self._optimal_cv(exp_bounds, method, cv_folds)
            else:
                raise ValueError(
                    "regularization_strength string must be 'abic' or 'cv', "
                    f"got {regularization_strength!r}"
                )
        else:
            strength = float(regularization_strength)

        if self.L is not None and strength > 0:
            d_reg = _build_reg_rhs(self.L, strength, regularization_target)
            G_aug = np.vstack([self.G_w, np.sqrt(strength) * self.L])
            d_aug = np.concatenate([self.d_w, d_reg])
            reg_strength: float | None = strength
        else:
            G_aug = self.G_w
            d_aug = self.d_w
            reg_strength = None if strength == 0.0 else strength

        if method is None:
            method = _auto_select_method(exp_bounds)

        m = _solve(G_aug, d_aug, exp_bounds, method, constraints)

        predicted = self.G @ m
        residuals = self.d - predicted
        reduced_chi2 = _compute_reduced_chi2(residuals, self.W, self._n_params)
        rms = float(np.sqrt(np.mean(residuals**2)))

        if self.components in {"both", "plate"}:
            slip = np.column_stack([m[: self._n_patches], m[self._n_patches :]])
        else:
            slip = m.reshape(-1, 1)
        basis_angle: float | np.ndarray | None
        if self.components == "rake":
            basis_angle = self.rake
        elif self.components == "azimuth":
            assert self.slip_azimuth is not None
            basis_angle = self.slip_azimuth - self.fault.strike
        elif self.components == "plate":
            basis_angle = self.plate_rake
        else:
            basis_angle = None
        strike_slip, dip_slip = _physical_components(m, self.components, basis_angle)
        result_warnings: list[str] = []
        if self.d.size <= self._n_params:
            result_warnings.append(
                "the inversion has no positive nominal degrees of freedom"
            )
        if self.quantity == "velocity":
            moment = float("nan")
            mw = float("nan")
            result_warnings.append(
                "moment and Mw are undefined for velocity data; slip is a slip rate"
            )
        else:
            moment = self.fault.moment(magnitude(strike_slip, dip_slip))
            mw = moment_to_magnitude(moment)

        constraint_matrix: np.ndarray | None = None
        constraint_bounds: np.ndarray | None = None
        if constraints is not None:
            constraint_matrix = np.asarray(constraints[0], dtype=float).copy()
            constraint_bounds = np.asarray(constraints[1], dtype=float).copy()
        lower_bounds: np.ndarray | None = None
        upper_bounds: np.ndarray | None = None
        if exp_bounds is not None:
            lower_bounds = exp_bounds[0].copy()
            upper_bounds = exp_bounds[1].copy()
        system_hash = _system_hash(self.G, self.d, self.W, self.L)
        fit_diagnostics = tuple(
            self._compute_dataset_diagnostics(residuals, reg_strength)
        )

        return InversionResult(
            slip=slip,
            slip_vector=m,
            residuals=residuals,
            predicted=predicted,
            reduced_chi2=reduced_chi2,
            rms=rms,
            moment=moment,
            Mw=mw,
            regularization=self.regularization if reg_strength is not None else None,
            regularization_strength=reg_strength,
            components=self.components,
            rake=self.rake,
            slip_azimuth=self.slip_azimuth,
            plate_rake=self.plate_rake,
            local_rake=(
                self.slip_azimuth - self.fault.strike
                if self.slip_azimuth is not None
                else None
            ),
            dataset_names=self.dataset_names,
            dataset_slices=self.dataset_slices,
            solver=method,
            success=True,
            message=f"{method} completed",
            regularization_selection=regularization_selection,
            backend=backend.get_backend(),
            precision=backend.get_precision(),
            warnings=tuple(result_warnings),
            quantity=self.quantity,
            units=self.units,
            system_hash=system_hash,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            regularization_target=(
                None
                if regularization_target is None
                else np.asarray(regularization_target, dtype=float).copy()
            ),
            constraint_matrix=constraint_matrix,
            constraint_bounds=constraint_bounds,
            dataset_diagnostics=fit_diagnostics,
        )

    def lcurve(
        self,
        regularization_range: tuple[float, float] = (1e-2, 1e6),
        n: int = 50,
        bounds: BoundsSpec = None,
        method: str | None = None,
    ) -> LCurveResult:
        """Sweep regularization strength and compute the L-curve.

        For unconstrained (``wls``) solves, GtWG, LtL, and Gtwd are used
        directly so each iteration is a single linear solve with no matrix
        assembly.  For constrained solves the augmented system is used.

        Misfits are the unweighted norm ``||Gm - d||``.

        Args:
            regularization_range: ``(min_lambda, max_lambda)`` range to sweep.
            n: Number of lambda values to evaluate.
            bounds: Per-component slip bounds.
            method: Solver method.

        Returns:
            LCurveResult with sweep arrays and optimal lambda.

        Raises:
            ValueError: If the system has no regularization matrix.
        """
        if self.L is None:
            raise ValueError("lcurve requires a regularization matrix")

        lambdas = np.geomspace(regularization_range[0], regularization_range[1], n)
        misfits = np.empty(n)
        model_norms = np.empty(n)

        exp_bounds = _expand_bounds(
            bounds, self._n_patches, self._n_params // self._n_patches
        )
        solve_method = method if method is not None else _auto_select_method(exp_bounds)

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
                m = _solve(G_aug, d_aug, exp_bounds, solve_method, None)
                residuals = self.d - self.G @ m
                misfits[i] = float(np.sqrt(residuals @ residuals))
                model_norms[i] = float(np.sqrt((self.L @ m) @ (self.L @ m)))

        optimal = _lcurve_corner(lambdas, misfits, model_norms)
        return LCurveResult(
            regularization_values=lambdas,
            misfits=misfits,
            model_norms=model_norms,
            optimal=optimal,
        )

    def abic_curve(
        self,
        regularization_range: tuple[float, float] = (1e-2, 1e6),
        n: int = 50,
    ) -> ABICCurveResult:
        """Sweep regularization strength and compute the ABIC at each value.

        GtWG, LtL, Gtwd, and eig_LtL are all computed once and reused
        across all iterations.  Misfits are the unweighted norm ``||Gm - d||``,
        consistent with ``lcurve``.

        Args:
            regularization_range: ``(min_lambda, max_lambda)`` range to sweep.
            n: Number of lambda values to evaluate.

        Returns:
            ABICCurveResult with sweep arrays and optimal lambda.

        Raises:
            ValueError: If the system has no regularization matrix.
        """
        if self.L is None:
            raise ValueError("abic_curve requires a regularization matrix")

        lambdas = np.geomspace(regularization_range[0], regularization_range[1], n)

        if backend.get_backend() == "jax":
            abic_values, misfits, model_norms = self._abic_sweep_jax(lambdas)
        else:
            abic_values = np.empty(n)
            misfits = np.empty(n)
            model_norms = np.empty(n)
            for i, lam in enumerate(lambdas):
                abic_values[i], misfits[i], model_norms[i] = self._abic_value(lam)

        optimal = float(lambdas[np.argmin(abic_values)])
        return ABICCurveResult(
            regularization_values=lambdas,
            abic_values=abic_values,
            misfits=misfits,
            model_norms=model_norms,
            optimal=optimal,
        )

    def dataset_diagnostics(
        self,
        result: InversionResult,
    ) -> list[DatasetDiagnostics]:
        """Compute per-dataset fit diagnostics using the hat matrix.

        Args:
            result: Output from ``invert()``.

        Returns:
            List of ``DatasetDiagnostics``, one per dataset.
        """
        return self._compute_dataset_diagnostics(
            result.residuals, result.regularization_strength
        )

    def _compute_dataset_diagnostics(
        self,
        residuals: np.ndarray,
        regularization_strength: float | None,
    ) -> list[DatasetDiagnostics]:
        """Compute named-fit statistics from solve-time arrays."""
        lev = self._hat_diagonal(regularization_strength)

        diags = []
        for ds, idx in zip(self.datasets, self.dataset_slices):
            n = ds.n_obs
            r_k = residuals[idx]
            W_k = self.W[idx, idx]

            chi2_k = float(r_k @ W_k @ r_k)
            lev_k = float(np.sum(lev[idx]))
            dof_k = n - lev_k
            reduced_chi2_k = chi2_k / dof_k if dof_k > 0 else float("nan")
            wrms_k = float(np.sqrt(chi2_k / n))
            rms_k = float(np.sqrt(np.mean(r_k**2)))

            diags.append(
                DatasetDiagnostics(
                    chi2=chi2_k,
                    reduced_chi2=reduced_chi2_k,
                    wrms=wrms_k,
                    rms=rms_k,
                    n_obs=n,
                    dof=dof_k,
                    leverage=lev_k,
                )
            )

        return diags

    def model_covariance(
        self, result: InversionResult, kind: str = "posterior"
    ) -> np.ndarray:
        """Compute the model covariance matrix.

        For the unregularized case both kinds reduce to::

            Cm = (G^T W G)^{-1}

        For the regularized case, with ``H = G^T W G + lambda L^T L``
        (see docs/conventions.md):

        - ``kind='posterior'`` (default) — the linear-Gaussian posterior
          covariance ``Cm = H^{-1}``, treating ``lambda L^T L`` as a prior
          precision. This is the quantity taught in Tutorial 09 and the one
          consistent with the Bayesian slip draws in ``geodef.bayes``.
        - ``kind='estimator'`` — the frequentist covariance of the penalized
          estimator under data noise alone (Tarantola, 2005)::

              Cm = H^{-1} @ G^T W G @ H^{-1}

          It excludes the bias the regularization introduces, so it shrinks
          to zero as ``lambda`` grows; interpret it together with the
          resolution matrix.

        Args:
            result: Output from ``invert()``.
            kind: ``'posterior'`` (default) or ``'estimator'``.

        Returns:
            Model covariance matrix, shape (n_params, n_params).
        """
        if kind not in ("posterior", "estimator"):
            raise ValueError(f"kind must be 'posterior' or 'estimator', got {kind!r}")
        if self.L is not None and result.regularization_strength is not None:
            H = self.GtWG + result.regularization_strength * self.LtL
            if kind == "posterior":
                return np.linalg.inv(H)
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
        if self.L is not None and result.regularization_strength is not None:
            H = self.GtWG + result.regularization_strength * self.LtL
            return np.linalg.solve(H, self.GtWG)
        return np.linalg.solve(self.GtWG, self.GtWG)

    def model_uncertainty(
        self, result: InversionResult, kind: str = "posterior"
    ) -> np.ndarray:
        """Compute per-parameter 1-sigma uncertainty from model covariance.

        Args:
            result: Output from ``invert()``.
            kind: Covariance kind, ``'posterior'`` (default) or
                ``'estimator'``; see :meth:`model_covariance`.

        Returns:
            Uncertainty array, shape (n_params,).
        """
        Cm = self.model_covariance(result, kind=kind)
        return np.sqrt(np.maximum(np.diag(Cm), 0.0))


def _validate_args(
    datasets: list[DataSet],
    components: str,
    regularization: str | np.ndarray | None,
    regularization_strength: float | str,
    bounds: BoundsSpec,
    method: str | None,
    regularization_target: np.ndarray | None,
    n_params: int,
    rake: float | None = None,
    slip_azimuth: float | None = None,
    plate_rake: np.ndarray | None = None,
) -> None:
    """Validate invert() arguments."""
    for ds in datasets:
        if not isinstance(ds, DataSet):
            raise TypeError(
                f"datasets must contain DataSet instances, got {type(ds).__name__}"
            )

    if components not in _VALID_COMPONENTS:
        raise ValueError(
            f"components must be one of {_VALID_COMPONENTS}, got {components!r}"
        )

    if components == "rake" and rake is None:
        raise ValueError("components='rake' requires a rake angle in degrees")
    if rake is not None and components != "rake":
        raise ValueError(
            f"rake angle is only used with components='rake', "
            f"got components={components!r}"
        )
    if components == "azimuth" and slip_azimuth is None:
        raise ValueError("components='azimuth' requires a slip_azimuth in degrees")
    if slip_azimuth is not None and components != "azimuth":
        raise ValueError(
            f"slip_azimuth is only used with components='azimuth', "
            f"got components={components!r}"
        )
    if components == "plate" and plate_rake is None:
        raise ValueError("components='plate' requires plate_rake")
    if plate_rake is not None and components != "plate":
        raise ValueError(
            "plate_rake is only used with components='plate', "
            f"got components={components!r}"
        )

    if method is not None and method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")

    if (
        isinstance(regularization, str)
        and regularization not in _VALID_REGULARIZATION_STRINGS
    ):
        raise ValueError(
            f"regularization must be one of {_VALID_REGULARIZATION_STRINGS} "
            f"or a numpy array, got {regularization!r}"
        )

    if isinstance(regularization, np.ndarray) and regularization.shape[1] != n_params:
        raise ValueError(
            f"regularization matrix must have {n_params} columns, "
            f"got {regularization.shape[1]}"
        )

    if isinstance(regularization_strength, str):
        if regularization_strength not in _VALID_STRENGTH_STRINGS:
            raise ValueError(
                f"regularization_strength must be a float or one of "
                f"{_VALID_STRENGTH_STRINGS}, got {regularization_strength!r}"
            )
        if regularization is None:
            raise ValueError(
                f"regularization_strength='{regularization_strength}' requires "
                f"regularization to be set"
            )

    if regularization_target is not None:
        if regularization is None and regularization_strength == 0.0:
            raise ValueError("regularization_target requires regularization to be set")
        target = np.asarray(regularization_target)
        if target.shape != (n_params,):
            raise ValueError(
                f"regularization_target must have shape ({n_params},), "
                f"got {target.shape}"
            )


def _apply_weights(
    G: np.ndarray,
    d: np.ndarray,
    W: np.ndarray,
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


def _system_hash(
    greens_matrix: np.ndarray,
    observations: np.ndarray,
    weights: np.ndarray,
    regularizer: np.ndarray | None,
) -> str:
    """Fingerprint the numerical system needed to verify a reproduced solve."""
    digest = hashlib.sha256()
    for label, array in (
        ("G", greens_matrix),
        ("d", observations),
        ("W", weights),
        ("L", regularizer),
    ):
        digest.update(label.encode())
        if array is None:
            digest.update(b"none")
            continue
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.dtype.str.encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


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
