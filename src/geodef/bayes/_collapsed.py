"""Collapsed geometry posteriors: the shared base and the planar case.

Private submodule of :mod:`geodef.bayes`. ``_CollapsedPosterior`` builds
the analytically slip-marginalized log-density; ``RectPosterior``
specializes it to planar-fault geometry parameters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import scipy.linalg

from geodef import backend
from geodef.bayes._util import (
    _VALID_MODES,
    _parse_positive,
    _require_jax,
    _slip_transform,
)
from geodef.data import DataSet
from geodef.fault import Fault
from geodef.geometry import (
    LocalFrame,
    _resolve_frame,
    as_planar_vector,
    planar_parameter_dict,
)
from geodef.gradients import rect_greens
from geodef.invert import LinearSystem
from geodef.invert._geometry import (
    _THETA_NAMES,
    _fault_from_planar_vector,
    _projection_matrix,
)
from geodef.invert._solvers import _rank_positive_eigs


class _CollapsedPosterior:
    """Base class for collapsed (slip-marginalized) geometry posteriors.

    Holds everything that is generic over the weighted-Green's-matrix
    assembly ``self._assemble(x)``: the whole-``logpdf`` forward-mode
    ``custom_jvp`` wrapper, prior parsing/clipping, the collapsed slip
    linear algebra (:meth:`_collapse_terms`), the traceable density
    pieces (``log_likelihood``, ``log_prior``, ``logpdf``), and the
    conditional slip posterior / posterior predictive (``slip_mode``,
    ``slip_draws``, ``predict``). A subclass implements ``_assemble`` to
    plug in its own forward model (e.g. a rectangular Okada patch grid
    or a warped triangular mesh); everything else here is inherited
    unchanged.

    A subclass whose slip prior stops being conjugate in some parameter
    regime (:class:`RectPosterior`'s positivity path) may set
    ``self._joint = True`` and override ``_log_likelihood_joint``,
    ``_log_prior_joint``, and ``_joint_reconstruct``; ``log_likelihood``,
    ``log_prior``, ``slip_mode``, and ``_vmapped_draws`` dispatch to
    those instead of the collapsed path whenever ``_joint`` is set. The
    base implementations of the three raise ``NotImplementedError`` —
    they only exist to satisfy mypy and are never reached while
    ``_joint`` stays False.

    Attributes a subclass's ``__init__`` must set:
        _lo, _hi: Per-parameter lower/upper bounds, ``+-inf`` for
            normal-prior entries.
        _is_uniform: Bool array, True where the prior is uniform (vs.
            normal).
        _mu, _sd: Normal-prior mean/sd, unused where ``_is_uniform``.
        n_data: Number of (weighted, stacked) observations.
        _include_logdet: Whether ``log_likelihood`` includes the Occam
            (log-determinant) terms (False in ``'profiled'`` mode).
        _logdet_rank, _logdet_sum: Precomputed pseudo-determinant pieces
            of the slip-prior precision (rank and sum of log positive
            eigenvalues).
        _LtL: Slip-prior regularization matrix ``L^T L``.
        _d_w: Weighted, stacked observation vector.
        _W_half: Upper-Cholesky factor of the data weight matrix (used
            to undo weighting in predictions).
        _lambda_fixed: Fixed ``lambda`` (``'weak'``/``'profiled'``
            modes) or None when ``lambda`` is sampled
            (``'hierarchical'``).
        free: Names of the sampled free geometry/warp parameters.
        param_names: Full sampled-parameter layout.
        n_params: ``len(param_names)``.
        x0: Initial sampled-parameter vector.
        _logpdf_fn: The callable ``logpdf`` dispatches to — typically
            ``self._build_logpdf()`` (collapsed path) or a joint-path
            equivalent.
    """

    _joint: bool = False

    # Attribute contract (see class docstring); declared here, unset,
    # purely so mypy can check the base-class methods below. Subclasses
    # assign real values in their own ``__init__``.
    _lo: np.ndarray
    _hi: np.ndarray
    _is_uniform: np.ndarray
    _mu: np.ndarray
    _sd: np.ndarray
    n_data: int
    _include_logdet: bool
    _logdet_rank: int
    _logdet_sum: float
    _LtL: np.ndarray
    _d_w: np.ndarray
    _W_half: np.ndarray
    _lambda_fixed: float | None
    free: list[str]
    param_names: list[str]
    n_params: int
    x0: np.ndarray
    _logpdf_fn: Any

    def _build_logpdf(self) -> Any:
        """Wrap the log-posterior with a forward-mode differentiation rule.

        XLA compilation of the reverse-mode gradient through the nested
        okada85 subfunctions is pathologically slow (minutes, vs seconds
        for the forward pass), while forward-mode compiles quickly and —
        with only a handful of sampled parameters — evaluates just as
        fast. A ``custom_jvp`` rule built on ``jax.jacfwd`` makes every
        downstream transform (``jax.grad`` in NUTS included) use forward
        mode; the rule is linear in the tangents, so reverse mode
        transposes through it exactly.
        """
        import jax
        import jax.numpy as jnp

        def raw(x: Any) -> Any:
            return self.log_prior(x) + self.log_likelihood(x)

        wrapped = jax.custom_jvp(raw)

        @wrapped.defjvp
        def _jvp(primals: tuple, tangents: tuple) -> tuple:
            (x,), (t,) = primals, tangents
            return raw(x), jnp.dot(jax.jacfwd(raw)(x), t)

        return wrapped

    @staticmethod
    def _parse_prior(name: str, spec: tuple) -> tuple:
        """Normalize a prior spec to ('uniform', lo, hi) or ('normal', mu, sd)."""
        if len(spec) == 2:
            lo, hi = map(float, spec)
            if not lo < hi:
                raise ValueError(f"theta_prior[{name!r}]: bounds must have lo < hi")
            return ("uniform", lo, hi)
        if len(spec) == 3 and spec[0] == "normal":
            mu, sd = map(float, spec[1:])
            if sd <= 0:
                raise ValueError(f"theta_prior[{name!r}]: sd must be positive")
            return ("normal", mu, sd)
        raise ValueError(
            f"theta_prior[{name!r}] must be (lo, hi) or ('normal', mu, sd), "
            f"got {spec!r}"
        )

    # ------------------------------------------------------------------
    # Traceable density pieces
    # ------------------------------------------------------------------

    def _clip(self, x: Any) -> Any:
        """Clip uniform-prior parameters into bounds (traceable)."""
        import jax.numpy as jnp

        return jnp.clip(jnp.asarray(x), jnp.asarray(self._lo), jnp.asarray(self._hi))

    def _assemble(self, x: Any) -> tuple:
        """Marginalization ingredients at sampled parameters x (traceable).

        A subclass assembles the weighted Green's matrix ``G_w`` for its
        own forward model at (bound-clipped) ``x`` and calls
        :meth:`_collapse_terms` to complete the collapsed slip linear
        algebra.

        Returns:
            Tuple ``(sigma2, lam, G_w, chol_H, m_hat, S)``: noise
            variance factor, prior strength, weighted Green's matrix,
            Cholesky factor of ``H = G_w^T G_w + lam LtL``, conditional
            slip mode, and the total misfit
            ``S = ||d_w - G_w m||^2 + lam ||L m||^2``.

        Raises:
            NotImplementedError: Always in the base class; subclasses
                must override.
        """
        raise NotImplementedError("Subclasses must implement _assemble")

    def _collapse_terms(self, G_w: Any, lam: Any) -> tuple:
        """Collapsed slip linear algebra at a weighted Green's matrix (traceable).

        Shared by every ``_assemble`` implementation: given ``G_w`` and
        the prior strength ``lam``, forms ``H = G_w^T G_w + lam LtL``,
        its Cholesky factor, the conditional slip mode ``m_hat`` (the
        ridge solution), and the total misfit that feeds the collapsed
        log-likelihood.

        Args:
            G_w: Weighted Green's matrix, shape (n_data, n_slip).
            lam: Slip-prior strength (scalar).

        Returns:
            Tuple ``(chol_H, m_hat, S)``: Cholesky factor of
            ``H = G_w^T G_w + lam LtL``, conditional slip mode, and the
            total misfit ``S = ||d_w - G_w m_hat||^2 + lam ||L m_hat||^2``.
        """
        import jax.numpy as jnp
        from jax.scipy.linalg import cho_solve

        LtL = jnp.asarray(self._LtL)
        d_w = jnp.asarray(self._d_w)

        H = G_w.T @ G_w + lam * LtL
        chol_H = jnp.linalg.cholesky(H)
        m_hat = cho_solve((chol_H, True), G_w.T @ d_w)
        r = d_w - G_w @ m_hat
        S = r @ r + lam * (m_hat @ LtL @ m_hat)
        return chol_H, m_hat, S

    def _misfit_total(self, x: npt.ArrayLike) -> np.ndarray:
        """Total misfit S = ||d_w - G_w m||^2 + lam ||L m||^2 at x."""
        return self._assemble(x)[5]

    def log_likelihood(self, x: npt.ArrayLike) -> np.ndarray:
        """Collapsed log-likelihood log p(d_w | x) (traceable).

        The exact Gaussian marginal over slip in the hierarchical and
        weak modes (up to the constant ``log|W|/2`` from weighting the
        data, and using the pseudo-determinant convention when the
        regularization operator is rank-deficient); the profiled objective
        without the Occam log-determinant terms in profiled mode.

        Args:
            x: Sampled parameter vector, ordered as ``param_names``.

        Returns:
            Scalar log-likelihood.
        """
        import jax.numpy as jnp

        if self._joint:
            return cast(np.ndarray, self._log_likelihood_joint(x))
        sigma2, lam, _, chol_H, _, S = self._assemble(x)
        n = self.n_data
        ll = -0.5 * n * jnp.log(2.0 * jnp.pi * sigma2) - S / (2.0 * sigma2)
        if self._include_logdet:
            logdet_H = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol_H)))
            logdet_prior = self._logdet_rank * jnp.log(lam) + self._logdet_sum
            ll = ll + 0.5 * (logdet_prior - logdet_H)
        return ll

    def log_prior(self, x: npt.ArrayLike) -> np.ndarray:
        """Log-prior over the sampled parameters (traceable).

        Args:
            x: Sampled parameter vector, ordered as ``param_names``.

        Returns:
            Scalar log-prior; ``-inf`` outside uniform bounds.
        """
        import jax.numpy as jnp

        if self._joint:
            return cast(np.ndarray, self._log_prior_joint(x))
        x = jnp.asarray(x)
        lo = jnp.asarray(self._lo)
        hi = jnp.asarray(self._hi)
        in_bounds = (x >= lo) & (x <= hi)
        lp_uniform = jnp.where(in_bounds, -jnp.log(hi - lo), -jnp.inf)
        z = (x - jnp.asarray(self._mu)) / jnp.asarray(self._sd)
        lp_normal = (
            -0.5 * z**2 - jnp.log(jnp.asarray(self._sd)) - 0.5 * jnp.log(2.0 * jnp.pi)
        )
        terms = jnp.where(jnp.asarray(self._is_uniform), lp_uniform, lp_normal)
        return cast(np.ndarray, jnp.sum(terms))

    def logpdf(self, x: npt.ArrayLike) -> np.ndarray:
        """Log-posterior density (traceable, differentiable).

        Equals ``log_prior(x) + log_likelihood(x)``; the likelihood is
        evaluated at bound-clipped parameters so its gradient stays
        finite at rejected points. On the collapsed path (``positive is
        None``) differentiation goes through a whole-``logpdf``
        forward-mode ``custom_jvp`` rule (see ``_build_logpdf``); on the
        positivity path it is plain reverse-mode with the ``custom_jvp``
        placed around ``G(theta)`` alone (see ``_build_logpdf_positive``).
        Either way both ``jax.grad`` and ``jax.jacfwd`` work and compile
        quickly.

        Args:
            x: Sampled parameter vector, ordered as ``param_names``.

        Returns:
            Scalar log-posterior; ``-inf`` outside uniform prior bounds.
        """
        return cast(np.ndarray, self._logpdf_fn(x))

    def _log_likelihood_joint(self, x: Any) -> Any:
        """Half-collapsed marginal log-likelihood (slip integrated/profiled).

        Base stub — only overridden by a subclass whose slip prior stops
        being conjugate in some parameter regime and sets
        ``self._joint = True`` (e.g. :class:`RectPosterior`'s positivity
        path).

        Raises:
            NotImplementedError: Always in the base class.
        """
        raise NotImplementedError(
            "_log_likelihood_joint must be implemented by subclasses with _joint=True"
        )

    def _log_prior_joint(self, x: Any) -> Any:
        """Hyperparameter priors + whitening Jacobian (sampled-space prior).

        Base stub — only overridden by a subclass whose slip prior stops
        being conjugate in some parameter regime and sets
        ``self._joint = True`` (e.g. :class:`RectPosterior`'s positivity
        path).

        Raises:
            NotImplementedError: Always in the base class.
        """
        raise NotImplementedError(
            "_log_prior_joint must be implemented by subclasses with _joint=True"
        )

    def _joint_reconstruct(self, x: Any, key: Any) -> tuple:
        """Full slip and unweighted prediction at x on the joint-slip path.

        Base stub — only overridden by a subclass whose slip prior stops
        being conjugate in some parameter regime and sets
        ``self._joint = True`` (e.g. :class:`RectPosterior`'s positivity
        path).

        Raises:
            NotImplementedError: Always in the base class.
        """
        raise NotImplementedError(
            "_joint_reconstruct must be implemented by subclasses with _joint=True"
        )

    # ------------------------------------------------------------------
    # Conditional slip posterior and posterior predictive
    # ------------------------------------------------------------------

    def slip_mode(self, x: npt.ArrayLike) -> np.ndarray:
        """Conditional slip mode (= mean) at sampled parameters x.

        The slip conditional on ``x`` is Gaussian,
        ``m | x, d ~ N(m_hat, sigma^2 H^-1)`` with
        ``H = G_w^T G_w + lam L^T L``; this returns ``m_hat``.

        Args:
            x: Sampled parameter vector, ordered as ``param_names``.

        Returns:
            Conditional slip mode, shape (n_slip,). On the positivity
            (joint-slip) path this is the full slip vector at ``x``: the
            constrained block mapped through the softplus, and the
            marginalized block set to its conditional mean.
        """
        if self._joint:
            m, _ = self._joint_reconstruct(np.asarray(x, dtype=float), None)
            return backend.to_numpy(m)
        return backend.to_numpy(self._assemble(np.asarray(x, dtype=float))[4])

    def _draw_one(self, x: Any, key: Any) -> tuple:
        """One conditional slip draw and its data-space prediction.

        Draws ``m = m_hat + sigma * L_H^-T z`` (exact conditional
        Gaussian via the Cholesky factor of H) and maps it to
        unweighted data space. Traceable; vmapped over posterior
        samples by :meth:`slip_draws` and :meth:`predict`.
        """
        import jax
        import jax.numpy as jnp
        from jax.scipy.linalg import solve_triangular

        sigma2, _, G_w, chol_H, m_hat, _ = self._assemble(x)
        z = jax.random.normal(key, m_hat.shape, dtype=m_hat.dtype)
        m = m_hat + jnp.sqrt(sigma2) * solve_triangular(chol_H.T, z, lower=False)
        # unweighted prediction: W_half is upper triangular, so undo it
        d_pred = solve_triangular(jnp.asarray(self._W_half), G_w @ m, lower=False)
        return m, d_pred

    def _vmapped_draws(self, samples: npt.ArrayLike, seed: int) -> tuple:
        """Conditional draws and predictions for each sample row."""
        jax = _require_jax()
        import jax.numpy as jnp

        samples = np.atleast_2d(np.asarray(samples, dtype=float))
        if self._joint:
            keys = jax.random.split(jax.random.PRNGKey(seed), samples.shape[0])
            m, d_pred = jax.jit(jax.vmap(self._joint_reconstruct))(
                jnp.asarray(samples), keys
            )
            return backend.to_numpy(m), backend.to_numpy(d_pred)
        keys = jax.random.split(jax.random.PRNGKey(seed), samples.shape[0])
        m, d_pred = jax.jit(jax.vmap(self._draw_one))(jnp.asarray(samples), keys)
        return backend.to_numpy(m), backend.to_numpy(d_pred)

    def slip_draws(self, samples: npt.ArrayLike, seed: int = 0) -> np.ndarray:
        """Exact conditional slip draws, one per posterior sample.

        Completes the collapsed sampler: drawing one slip vector from
        the Gaussian conditional ``p(m | x, d)`` per posterior sample
        of ``x`` yields draws from the joint posterior ``p(m, x | d)``
        (Rao-Blackwellization), so per-patch statistics of the result
        include geometry and hyperparameter uncertainty.

        On the positivity path the constrained block is already part of
        ``x`` (a deterministic softplus map, ``seed``-independent, and
        non-negative exactly), while the marginalized unconstrained block
        is completed from its Gaussian conditional — one draw per sample,
        so it does use ``seed``. When every component is constrained
        nothing is marginalized and the whole result is deterministic.

        Args:
            samples: Sampled parameter vectors, shape (n, n_params) —
                e.g. ``PosteriorResult.flat`` (optionally thinned).
            seed: PRNG seed for the conditional draws — of the full slip
                on the collapsed path, of the marginalized block on the
                positivity path.

        Returns:
            Slip draws, shape (n, n_slip). Columns follow the
            components layout (``[:N]`` strike-slip then ``[N:]``
            dip-slip when components='both').
        """
        return self._vmapped_draws(samples, seed)[0]

    def predict(self, samples: npt.ArrayLike, seed: int = 0) -> np.ndarray:
        """Posterior-predictive mean field at each posterior sample.

        Maps one conditional slip draw per sample to unweighted data
        space (``G(theta) m``, projected like the observations), so
        row statistics give credible intervals for the noise-free
        predicted data.

        Args:
            samples: Sampled parameter vectors, shape (n, n_params).
            seed: PRNG seed for the conditional slip draws.

        Returns:
            Predictions, shape (n, n_data), rows ordered like the
            stacked observation vector.
        """
        return self._vmapped_draws(samples, seed)[1]


class RectPosterior(_CollapsedPosterior):
    """Collapsed log-posterior for planar-fault geometry and scales.

    The sampled parameter vector ``x`` stacks the free geometry
    parameters (subset of ``e0, n0, depth, strike, dip, length,
    width``), then ``log10_sigma`` (data noise scale factor; 1 means
    the dataset covariances are exact), then ``log10_lambda`` when
    ``mode='hierarchical'``. ``logpdf``, ``log_likelihood``, and
    ``log_prior`` are pure functions of ``x``, traceable and
    differentiable with JAX, so they can be handed directly to
    gradient-based samplers.

    Uniform-prior parameters are clipped to their bounds before they
    enter the elastic kernels, so gradients stay finite even at
    rejected (out-of-bounds) points; the prior term carries the
    ``-inf``.

    Args:
        theta0: Named parameter mapping, or expert array
            ``[e0, n0, depth, strike, dip, length, width]``. Fixed parameters
            keep these values and free ones are initialized from them. Array
            input requires ``frame`` or ``ref_lat``/``ref_lon``.
        datasets: One or more displacement datasets (GNSS, InSAR,
            Vertical).
        ref_lat: Latitude anchoring the local Cartesian frame.
        ref_lon: Longitude anchoring the local Cartesian frame.
        frame: Explicit local frame for array ``theta0``. Mutually exclusive
            with an incompatible legacy ``ref_lat``/``ref_lon`` origin.
        free: Names of geometry parameters to sample. May be empty for
            pure hyperparameter inference.
        theta_prior: Prior for each free geometry parameter, keyed by
            name: ``(lo, hi)`` for uniform or ``('normal', mu, sd)``.
        n_length: Number of patches along strike.
        n_width: Number of patches down dip.
        components: Slip components for the marginalized linear solve:
            ``'both'``, ``'strike'``, or ``'dip'``.
        mode: Slip-prior mode: ``'hierarchical'``, ``'weak'``, or
            ``'profiled'`` (see module docstring).
        regularization: Regularization operator (as in ``invert()``) for the
            hierarchical and profiled modes; must be None for
            ``'weak'``.
        regularization_strength: Fixed lambda for ``'profiled'``; initial
            lambda (sampler starting point) for ``'hierarchical'``.
        slip_scale: Prior slip scale in meters for ``'weak'`` — the
            prior is ``m ~ N(0, (sigma * slip_scale)^2 I)``.
        positive: Positivity constraint on slip. ``None`` (default)
            keeps the fully collapsed sampler (all slip marginalized
            analytically). Setting it — ``'strike'``, ``'dip'``,
            ``'both'``, or a bool array of length ``n_slip`` — makes the
            slip prior a **truncated** Gaussian on the selected
            components, which is no longer conjugate there. Only those
            constrained components rejoin the sampled state (a whitened
            block appended after the hyperparameters, passed through a
            softplus so ``m >= 0`` holds exactly, with the map's
            log-Jacobian carried in the density); the **unconstrained**
            slip is still marginalized analytically (a half-collapse), so
            the sampled dimension grows only by the number of constrained
            components. This reduces to the fully collapsed sampler when
            no component is constrained, and to a fully joint slip
            sampler when all are. Each gradient traces the Okada kernel
            only seven times (a ``custom_jvp`` around ``G(theta)``),
            regardless of how many components are sampled.
        log10_sigma_prior: Uniform prior bounds on ``log10_sigma``.
        log10_lambda_prior: Uniform prior bounds on ``log10_lambda``
            (hierarchical mode only).
        nu: Poisson's ratio.

    Raises:
        RuntimeError: If the JAX backend is not active.
        ValueError: If a free-parameter name, prior, or mode option is
            invalid or inconsistent.
    """

    def __init__(
        self,
        theta0: npt.ArrayLike | Mapping[str, float],
        datasets: DataSet | list[DataSet],
        *,
        ref_lat: float | None = None,
        ref_lon: float | None = None,
        frame: LocalFrame | None = None,
        free: Sequence[str] = ("depth", "dip"),
        theta_prior: dict[str, tuple] | None = None,
        n_length: int = 1,
        n_width: int = 1,
        components: str = "both",
        mode: str = "hierarchical",
        regularization: str | np.ndarray | None = "laplacian",
        regularization_strength: float | None = None,
        slip_scale: float | None = None,
        positive: str | npt.ArrayLike | None = None,
        log10_sigma_prior: tuple[float, float] = (-2.0, 2.0),
        log10_lambda_prior: tuple[float, float] = (-8.0, 8.0),
        nu: float = 0.25,
    ) -> None:
        _require_jax()
        if isinstance(datasets, DataSet):
            datasets = [datasets]
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        unknown = [name for name in free if name not in _THETA_NAMES]
        if unknown:
            raise ValueError(
                f"Unknown free parameter(s) {unknown}; expected names from "
                f"{_THETA_NAMES}."
            )
        if components not in ("both", "strike", "dip"):
            raise ValueError(
                f"components must be 'both', 'strike', or 'dip', got {components!r}"
            )
        theta_prior = dict(theta_prior or {})
        missing = [name for name in free if name not in theta_prior]
        if missing:
            raise ValueError(f"theta_prior is missing entries for {missing}")
        if mode == "weak":
            if slip_scale is None:
                raise ValueError("mode='weak' requires slip_scale (meters)")
            if regularization is not None:
                raise ValueError(
                    "mode='weak' uses an identity slip prior; "
                    "regularization must be None"
                )
        if mode == "hierarchical" and regularization is None:
            raise ValueError("mode='hierarchical' requires a regularization operator")
        if mode == "profiled" and regularization_strength is None:
            raise ValueError(
                "mode='profiled' requires a fixed regularization_strength (lambda)"
            )

        frame = _resolve_frame(frame, ref_lat, ref_lon)
        theta0 = as_planar_vector(theta0)
        self.mode = mode
        self.free = list(free)
        self.components = components
        self._components = components
        self.datasets = datasets
        self.theta0 = np.array(theta0, copy=True)
        self.frame = frame
        self._theta0 = theta0
        self._free_idx = np.array(
            [_THETA_NAMES.index(name) for name in free], dtype=int
        )
        self._n_length = int(n_length)
        self._n_width = int(n_width)
        self._nu = float(nu)

        # Template system provides the stacked data, weights, and
        # regularization operator; its Green's matrix is not used.
        template = _fault_from_planar_vector(theta0, frame, n_length, n_width)
        sys = LinearSystem(template, datasets, regularization, components)
        n_patches = n_length * n_width
        self._col_start, self._col_stop = {
            "both": (0, 2 * n_patches),
            "strike": (0, n_patches),
            "dip": (n_patches, 2 * n_patches),
        }[components]
        n_params = self._col_stop - self._col_start
        self._n_slip = n_params
        self._n_patches = n_patches

        e_parts, n_parts = [], []
        for ds in datasets:
            enu = frame.to_enu(
                lon=ds.lon,
                lat=ds.lat,
                alt=np.full(ds.n_stations, frame.origin_alt),
            )
            e_parts.append(enu[:, 0])
            n_parts.append(enu[:, 1])
        self._e_obs = np.concatenate(e_parts)
        self._n_obs = np.concatenate(n_parts)

        self._W_half = scipy.linalg.cholesky(sys.W, lower=False)
        self._W_half_P = self._W_half @ _projection_matrix(datasets)
        self._d_w = self._W_half @ sys.d
        self.n_data = len(self._d_w)

        # Slip-prior precision structure: lambda * LtL, with the
        # lambda-independent log-determinant pieces precomputed. The
        # pseudo-determinant convention (positive eigenvalues only)
        # matches LinearSystem._abic_value for rank-deficient operators.
        if mode == "weak":
            assert slip_scale is not None
            self._LtL = np.eye(n_params)
            self._lambda_fixed: float | None = 1.0 / slip_scale**2
            self._logdet_rank = n_params
            self._logdet_sum = 0.0
        else:
            if regularization is not None:
                self._LtL = sys.LtL
                eig = np.abs(np.linalg.eigvalsh(self._LtL))
            else:
                self._LtL = np.zeros((n_params, n_params))
                eig = np.zeros(n_params)
            if mode == "profiled":
                assert regularization_strength is not None
                self._lambda_fixed = float(regularization_strength)
            else:
                self._lambda_fixed = None
            pos = _rank_positive_eigs(eig)
            self._logdet_rank = len(pos)
            self._logdet_sum = float(np.sum(np.log(pos)))

        # Reference lambda for the whitening precision on the positivity
        # path (only used when lambda is sampled, i.e. hierarchical).
        if self._lambda_fixed is not None:
            self._lam_ref = float(self._lambda_fixed)
        elif regularization_strength:
            self._lam_ref = float(regularization_strength)
        else:
            self._lam_ref = 10.0 ** (
                0.5 * (log10_lambda_prior[0] + log10_lambda_prior[1])
            )
        self._include_logdet = mode != "profiled"

        # Sampled-parameter layout and priors
        self.param_names = list(free) + ["log10_sigma"]
        specs = [self._parse_prior(name, theta_prior[name]) for name in free]
        specs.append(("uniform",) + tuple(map(float, log10_sigma_prior)))
        x0 = list(theta0[self._free_idx]) + [float(np.clip(0.0, *log10_sigma_prior))]
        if mode == "hierarchical":
            self.param_names.append("log10_lambda")
            specs.append(("uniform",) + tuple(map(float, log10_lambda_prior)))
            lam0 = (
                float(np.log10(regularization_strength))
                if regularization_strength
                else 0.5 * (log10_lambda_prior[0] + log10_lambda_prior[1])
            )
            x0.append(float(np.clip(lam0, *log10_lambda_prior)))
        self.x0 = np.array(x0)
        self.n_params = len(self.param_names)

        self._is_uniform = np.array([s[0] == "uniform" for s in specs])
        self._lo = np.array([s[1] if s[0] == "uniform" else -np.inf for s in specs])
        self._hi = np.array([s[2] if s[0] == "uniform" else np.inf for s in specs])
        self._mu = np.array([s[1] if s[0] == "normal" else 0.0 for s in specs])
        self._sd = np.array([s[2] if s[0] == "normal" else 1.0 for s in specs])

        # Positivity path: when `positive` is set, slip cannot be
        # marginalized (the truncated-Gaussian prior is not conjugate),
        # so the whole slip vector rejoins the sampled state as a
        # whitened `z` block appended after the hyperparameters. The
        # collapsed path (positive=None) is left completely untouched.
        self._joint = positive is not None
        self._n_hyper = self.n_params
        if positive is not None:
            self._setup_positive(positive, log10_sigma_prior, log10_lambda_prior)
            self._logpdf_fn = self._build_logpdf_positive()
        else:
            self._mask = np.zeros(n_params, dtype=bool)
            self._logpdf_fn = self._build_logpdf()

    def geometry(self, x: npt.ArrayLike) -> dict[str, float]:
        """Return named planar parameters represented by one state.

        This is a user-facing, non-JAX view. The likelihood methods continue to
        consume array states directly for tracing and vectorization.

        Args:
            x: One posterior parameter state, shape ``(n_params,)``.

        Returns:
            Parameter dictionary in the posterior's :attr:`frame`.

        Raises:
            ValueError: If ``x`` is not one complete parameter state.
        """
        state = np.asarray(x, dtype=float)
        if state.shape != (self.n_params,):
            raise ValueError(f"x must have shape ({self.n_params},), got {state.shape}")
        theta = np.array(self._theta0, copy=True)
        theta[self._free_idx] = state[: len(self.free)]
        return planar_parameter_dict(theta)

    def fault(self, x: npt.ArrayLike) -> Fault:
        """Return the planar fault represented by one parameter state.

        Args:
            x: One posterior parameter state, shape ``(n_params,)``.

        Returns:
            A fault discretized with this posterior's grid shape.
        """
        theta = as_planar_vector(self.geometry(x))
        return _fault_from_planar_vector(
            theta, self.frame, self._n_length, self._n_width
        )

    def _setup_positive(
        self,
        positive: str | npt.ArrayLike,
        log10_sigma_prior: tuple[float, float],
        log10_lambda_prior: tuple[float, float],
    ) -> None:
        """Build the extra state for the positivity (joint-slip) path.

        Parses the positivity mask, builds the whitening reference at
        ``theta0`` (via the same ``rect_greens`` assembly the likelihood
        uses, so the reference is consistent with the sampled forward
        model), and appends the whitened slip block ``z`` to the sampled
        layout after the hyperparameters.
        """
        n_slip = self._n_slip
        self._mask = _parse_positive(
            positive, self._components, self._n_patches, n_slip
        )

        # Reference geometry for whitening: G_w(theta0) through the JAX
        # assembly, evaluated once and frozen. Only sets the affine map's
        # conditioning — correctness comes from the log-Jacobian, not the
        # reference values.
        theta0 = np.asarray(self._theta0, dtype=float)
        G3_0 = backend.to_numpy(
            rect_greens(
                backend.xp.asarray(theta0),
                self._e_obs,
                self._n_obs,
                self._n_length,
                self._n_width,
                self._nu,
            )
        )
        G_w0 = (self._W_half_P @ G3_0)[:, self._col_start : self._col_stop]
        lam_ref = self._lam_ref
        sigma_ref = 1.0
        H0 = G_w0.T @ G_w0 + lam_ref * self._LtL
        mu0_full = scipy.linalg.solve(H0, G_w0.T @ self._d_w, assume_a="pos")

        # Half-collapse: only the positivity-constrained components stay
        # in the sampled state; the unconstrained block is marginalized
        # analytically (its Gaussian conditional is integrated in the
        # density, and completed post-hoc in slip draws). The sampled
        # block is whitened by the Schur complement of the reference H0 —
        # its exact marginal precision — which reduces to H0 itself when
        # every component is constrained (nothing to marginalize).
        c_idx = np.flatnonzero(self._mask)
        f_idx = np.flatnonzero(~self._mask)
        self._c_idx = c_idx
        self._f_idx = f_idx
        p_c = len(c_idx)

        if len(f_idx) > 0:
            H_cc = H0[np.ix_(c_idx, c_idx)]
            H_cf = H0[np.ix_(c_idx, f_idx)]
            H_ff = H0[np.ix_(f_idx, f_idx)]
            schur = H_cc - H_cf @ scipy.linalg.solve(H_ff, H_cf.T, assume_a="pos")
        else:
            schur = H0[np.ix_(c_idx, c_idx)]
        L0 = scipy.linalg.cholesky(schur, lower=True)
        self._sigma_ref = sigma_ref
        self._L0 = L0
        self._mu0 = mu0_full[c_idx]
        # All sampled components are constrained, so the transform's
        # softplus applies to every one of them.
        self._mask_c = np.ones(p_c, dtype=bool)
        self._logJ_affine = p_c * np.log(sigma_ref) - float(
            np.sum(np.log(np.diagonal(L0)))
        )

        # Fixed regularization sub-blocks for the marginalization.
        K = self._LtL
        self._K_cc = K[np.ix_(c_idx, c_idx)]
        self._K_cf = K[np.ix_(c_idx, f_idx)]
        self._K_ff = K[np.ix_(f_idx, f_idx)]

        self.param_names = list(self.param_names) + [f"z{i}" for i in c_idx]
        self.x0 = np.concatenate([self.x0, np.zeros(p_c)])
        self._lo = np.concatenate([self._lo, np.full(p_c, -np.inf)])
        self._hi = np.concatenate([self._hi, np.full(p_c, np.inf)])
        self.n_params = len(self.param_names)

    # ------------------------------------------------------------------
    # Positivity (joint slip) path
    # ------------------------------------------------------------------

    def _build_logpdf_positive(self) -> Any:
        """Build the joint-slip log-posterior with a G-level fwd-mode rule.

        Only the Green's assembly ``G(theta)`` sits inside the slow
        okada trace, so — unlike the collapsed path's whole-``logpdf``
        forward-mode wrapper — the ``custom_jvp`` is placed around
        ``rect_greens`` alone: its tangent materializes ``jacfwd`` over
        the seven geometry parameters and contracts with the geometry
        tangent. Everything downstream (the whitening solve, softplus,
        and misfit) is ordinary reverse-mode-friendly linear algebra, so
        the (potentially large) slip block ``z`` differentiates by plain
        ``jax.grad`` at one matvec of cost, and the expensive kernel is
        traced only seven times per gradient regardless of slip size.
        """
        import jax
        import jax.numpy as jnp

        e_obs, n_obs = self._e_obs, self._n_obs
        nl, nw, nu = self._n_length, self._n_width, self._nu

        def raw_G(theta: Any) -> Any:
            return rect_greens(theta, e_obs, n_obs, nl, nw, nu)

        G_of_theta = jax.custom_jvp(raw_G)

        @G_of_theta.defjvp
        def _g_jvp(primals: tuple, tangents: tuple) -> tuple:
            (theta,), (t,) = primals, tangents
            jac = jax.jacfwd(raw_G)(theta)
            return raw_G(theta), jnp.tensordot(jac, t, axes=([2], [0]))

        self._G_of_theta = G_of_theta
        return self._logpdf_joint

    def _assemble_joint(self, x: Any) -> tuple:
        """Half-collapsed density ingredients at sampled x (traceable).

        Assembles ``G(theta)``, maps the whitened block to the
        constrained slip ``m_c``, and analytically marginalizes the
        unconstrained slip block: ``H_f = G_f^T G_f + lam K_ff``,
        ``b = G_f^T r_c - lam K_cf^T m_c`` with ``r_c = d_w - G_c m_c``,
        and ``S_c = ||r_c||^2 + lam m_c^T K_cc m_c - b^T H_f^-1 b``.
        Reduces to the fully joint density (no marginalization) when
        every component is constrained.

        Returns:
            Tuple ``(sigma2, lam, m_c, logJ, S_c, logdet_Hf)``.
        """
        import jax.numpy as jnp
        from jax.scipy.linalg import cho_solve

        x = jnp.clip(jnp.asarray(x), jnp.asarray(self._lo), jnp.asarray(self._hi))
        n_free = len(self.free)
        theta = jnp.asarray(self._theta0)
        if n_free:
            theta = theta.at[jnp.asarray(self._free_idx)].set(x[:n_free])
        sigma2 = 10.0 ** (2.0 * x[n_free])
        if self._lambda_fixed is not None:
            lam = jnp.asarray(self._lambda_fixed)
        else:
            lam = 10.0 ** x[n_free + 1]

        G3 = self._G_of_theta(theta)
        G_w = (jnp.asarray(self._W_half_P) @ G3)[:, self._col_start : self._col_stop]
        z = x[self._n_hyper :]
        m_c, logJ = _slip_transform(
            z, self._mu0, self._L0, self._mask_c, self._sigma_ref, self._logJ_affine
        )

        G_c = G_w[:, jnp.asarray(self._c_idx)]
        r_c = jnp.asarray(self._d_w) - G_c @ m_c
        quad_cc = m_c @ jnp.asarray(self._K_cc) @ m_c
        if len(self._f_idx) > 0:
            G_f = G_w[:, jnp.asarray(self._f_idx)]
            H_f = G_f.T @ G_f + lam * jnp.asarray(self._K_ff)
            chol_Hf = jnp.linalg.cholesky(H_f)
            b = G_f.T @ r_c - lam * (jnp.asarray(self._K_cf).T @ m_c)
            y = cho_solve((chol_Hf, True), b)
            S_c = r_c @ r_c + lam * quad_cc - b @ y
            logdet_Hf = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol_Hf)))
        else:
            S_c = r_c @ r_c + lam * quad_cc
            logdet_Hf = jnp.asarray(0.0)
        return sigma2, lam, m_c, logJ, S_c, logdet_Hf

    def _hyper_logprior(self, x: Any) -> Any:
        """Uniform/normal priors on the sampled hyperparameters (traceable)."""
        import jax.numpy as jnp

        xh = jnp.asarray(x)[: self._n_hyper]
        lo, hi = (
            jnp.asarray(self._lo[: self._n_hyper]),
            jnp.asarray(self._hi[: self._n_hyper]),
        )
        in_bounds = (xh >= lo) & (xh <= hi)
        lp_uniform = jnp.where(in_bounds, -jnp.log(hi - lo), -jnp.inf)
        zc = (xh - jnp.asarray(self._mu)) / jnp.asarray(self._sd)
        lp_normal = (
            -0.5 * zc**2 - jnp.log(jnp.asarray(self._sd)) - 0.5 * jnp.log(2.0 * jnp.pi)
        )
        return jnp.sum(jnp.where(jnp.asarray(self._is_uniform), lp_uniform, lp_normal))

    def _logpdf_joint(self, x: Any) -> Any:
        """Half-collapsed joint log-posterior (single assembly, traceable)."""
        import jax.numpy as jnp

        sigma2, lam, _, logJ, S_c, logdet_Hf = self._assemble_joint(x)
        n_marg = self.n_data + len(self._c_idx)
        marg = (
            -0.5 * n_marg * jnp.log(2.0 * jnp.pi * sigma2)
            + 0.5 * (self._logdet_rank * jnp.log(lam) + self._logdet_sum)
            - 0.5 * logdet_Hf
            - S_c / (2.0 * sigma2)
        )
        return self._hyper_logprior(x) + logJ + marg

    def _log_likelihood_joint(self, x: Any) -> Any:
        """Half-collapsed marginal log-likelihood (slip integrated/profiled)."""
        import jax.numpy as jnp

        sigma2, lam, _, _, S_c, logdet_Hf = self._assemble_joint(x)
        n_marg = self.n_data + len(self._c_idx)
        return (
            -0.5 * n_marg * jnp.log(2.0 * jnp.pi * sigma2)
            + 0.5 * (self._logdet_rank * jnp.log(lam) + self._logdet_sum)
            - 0.5 * logdet_Hf
            - S_c / (2.0 * sigma2)
        )

    def _log_prior_joint(self, x: Any) -> Any:
        """Hyperparameter priors + whitening Jacobian (sampled-space prior)."""
        _, _, _, logJ, _, _ = self._assemble_joint(x)
        return self._hyper_logprior(x) + logJ

    def _assemble(self, x: Any) -> tuple:
        """Marginalization ingredients at sampled parameters x (traceable).

        Returns:
            Tuple ``(sigma2, lam, G_w, chol_H, m_hat, S)``: noise
            variance factor, prior strength, weighted Green's matrix,
            Cholesky factor of ``H = G_w^T G_w + lam LtL``, conditional
            slip mode, and the total misfit
            ``S = ||d_w - G_w m||^2 + lam ||L m||^2``.
        """
        import jax.numpy as jnp

        x = self._clip(x)
        n_free = len(self.free)
        theta = jnp.asarray(self._theta0)
        if n_free:
            theta = theta.at[jnp.asarray(self._free_idx)].set(x[:n_free])
        sigma2 = 10.0 ** (2.0 * x[n_free])
        if self._lambda_fixed is not None:
            lam = jnp.asarray(self._lambda_fixed)
        else:
            lam = 10.0 ** x[n_free + 1]

        G3 = rect_greens(
            theta, self._e_obs, self._n_obs, self._n_length, self._n_width, self._nu
        )
        G_w = (jnp.asarray(self._W_half_P) @ G3)[:, self._col_start : self._col_stop]
        chol_H, m_hat, S = self._collapse_terms(G_w, lam)
        return sigma2, lam, G_w, chol_H, m_hat, S

    def _joint_reconstruct(self, x: Any, key: Any) -> tuple:
        """Full slip and unweighted prediction at x on the positivity path.

        The constrained block is the deterministic softplus map of the
        sampled ``z``; the marginalized block is completed from its exact
        Gaussian conditional ``m_f | m_c, x, d ~ N(H_f^-1 b, sigma^2
        H_f^-1)`` — drawn when ``key`` is a PRNG key, or set to the
        conditional mean when ``key`` is None (used by :meth:`slip_mode`).
        """
        import jax
        import jax.numpy as jnp
        from jax.scipy.linalg import cho_solve, solve_triangular

        x = jnp.clip(jnp.asarray(x), jnp.asarray(self._lo), jnp.asarray(self._hi))
        n_free = len(self.free)
        theta = jnp.asarray(self._theta0)
        if n_free:
            theta = theta.at[jnp.asarray(self._free_idx)].set(x[:n_free])
        sigma2 = 10.0 ** (2.0 * x[n_free])
        if self._lambda_fixed is not None:
            lam = jnp.asarray(self._lambda_fixed)
        else:
            lam = 10.0 ** x[n_free + 1]

        G3 = self._G_of_theta(theta)
        G_w = (jnp.asarray(self._W_half_P) @ G3)[:, self._col_start : self._col_stop]
        z = x[self._n_hyper :]
        m_c, _ = _slip_transform(
            z, self._mu0, self._L0, self._mask_c, self._sigma_ref, self._logJ_affine
        )
        c_idx = jnp.asarray(self._c_idx)
        m = jnp.zeros(self._n_slip, dtype=m_c.dtype).at[c_idx].set(m_c)
        if len(self._f_idx) > 0:
            f_idx = jnp.asarray(self._f_idx)
            G_c = G_w[:, c_idx]
            r_c = jnp.asarray(self._d_w) - G_c @ m_c
            G_f = G_w[:, f_idx]
            H_f = G_f.T @ G_f + lam * jnp.asarray(self._K_ff)
            chol_Hf = jnp.linalg.cholesky(H_f)
            b = G_f.T @ r_c - lam * (jnp.asarray(self._K_cf).T @ m_c)
            m_f_mean = cho_solve((chol_Hf, True), b)
            if key is None:
                m_f = m_f_mean
            else:
                eps = jax.random.normal(key, m_f_mean.shape, dtype=m_f_mean.dtype)
                m_f = m_f_mean + jnp.sqrt(sigma2) * solve_triangular(
                    chol_Hf.T, eps, lower=False
                )
            m = m.at[f_idx].set(m_f)
        d_pred = solve_triangular(jnp.asarray(self._W_half), G_w @ m, lower=False)
        return m, d_pred
