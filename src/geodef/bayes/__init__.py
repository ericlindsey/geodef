"""Collapsed Bayesian posteriors for nonlinear fault-geometry inference.

Builds log-posterior densities over planar-fault geometry and
noise/regularization hyperparameters in which the (potentially hundreds
of) linear slip parameters are marginalized **analytically** — a
Rao-Blackwellized, "collapsed" formulation. Because slip enters the
forward model linearly, its Gaussian integral has a closed form built
from the same Cholesky/log-determinant quantities the ABIC machinery in
:mod:`geodef.invert` uses (ABIC is -2 log marginal likelihood up to
constants; Yabuki & Matsu'ura 1992, Fukuda & Johnson 2008). Samplers
therefore explore only the ~5-10 dimensional space of geometry plus
scales, with per-evaluation cost of one small Cholesky factorization.

The data model is ``d = G(theta) m + e`` with ``e ~ N(0, sigma^2 W^-1)``
and the conjugate slip prior ``m ~ N(0, sigma^2 (lambda L^T L)^-1)``.
Three prior modes are supported:

- ``'hierarchical'``: ``L`` is a regularization operator (e.g. Laplacian) and
  ``log10(lambda)`` is **sampled**, so posteriors average over all
  regularization strengths weighted by the evidence.
- ``'weak'``: ``L = I`` with a fixed, user-chosen slip scale — the
  collapsed analog of "unsmoothed" MCMC: resolved patches tighten,
  unresolved patches show honestly wide uncertainty.
- ``'profiled'``: fixed lambda and no Occam (log-determinant) terms;
  the slip is profiled rather than marginalized, mirroring the
  ``geometry_search`` objective. Useful for comparison, not a proper
  marginal posterior.

Requires the JAX backend::

    import geodef
    geodef.backend.set_backend("jax")
    frame = geodef.LocalFrame(-2.0, 100.0)
    post = geodef.bayes.RectPosterior(
        theta0, datasets, frame=frame,
        free=["dip", "depth"],
        theta_prior={"dip": (5.0, 60.0), "depth": (5e3, 40e3)},
        n_length=8, n_width=4, regularization="laplacian",
    )
    log_density = post.logpdf(post.x0)
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import scipy.linalg

from geodef import backend
from geodef.bayes._util import (
    _VALID_MODES,
    _VALID_SLIP_MODES,
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
from geodef.gradients import rect_greens, tri_greens
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


class TriWarp:
    """Low-dimensional normal-offset parameterization of a triangular mesh.

    Warps a fixed-connectivity triangular mesh along its best-fit-plane
    normal by smoothly interpolating offsets (meters) from a small set of
    control knots with a Gaussian RBF. Connectivity and the interpolation
    matrix are frozen at construction, so ``vertices(theta)`` is an exact
    linear map of ``theta`` — no remeshing, hence no posterior
    discontinuities from connectivity flips, and every warped mesh is
    automatically watertight (vertices that coincide before warping share
    the same interpolated offset, since they share the same (u, v)).

    Args:
        fault: Reference triangular :class:`Fault`; kept for :meth:`fault` and
            by :class:`TriPosterior`.
        knots: Explicit knot locations in the mesh's best-fit-plane (u, v)
            coordinates, shape (nk, 2). Takes precedence over ``n_knots``.
        n_knots: ``(n_u, n_v)`` grid shape spanning the mesh's (u, v)
            bounding box (corners included) when ``knots`` is None. Flat
            knot index ``k = iu + n_u * iv`` (v varies slowest).
        length_scale: Gaussian RBF length scale in meters. Defaults to
            the larger of the two knot-grid spacings for a grid, or the
            median nearest-neighbor knot distance for explicit knots.
        ridge: Diagonal ridge added to the knot kernel matrix before
            solving for the interpolation weights (numerical stability).

    Raises:
        ValueError: If ``fault`` is not triangular, ``knots`` has the wrong
            shape, or no default ``length_scale`` can be inferred.
    """

    def __init__(
        self,
        fault: Fault,
        *,
        knots: npt.ArrayLike | None = None,
        n_knots: tuple[int, int] = (3, 2),
        length_scale: float | None = None,
        ridge: float = 1e-8,
    ) -> None:
        if fault.vertices is None:
            raise ValueError(
                "TriWarp requires a triangular Fault (fault.vertices is None)"
            )
        self._ref_fault = fault
        self.frame = fault.frame
        self._ref_lat = self.frame.origin_lat
        self._ref_lon = self.frame.origin_lon

        v0 = np.asarray(fault.vertices, dtype=float)
        self._shape = v0.shape
        self._n_tri = v0.shape[0]
        p = v0.reshape(-1, 3)
        self._v0_flat = p

        # Best-fit plane by SVD: e_u, n_hat from the first/last right
        # singular vectors; n_hat flipped to point up, e_v recomputed so
        # (e_u, e_v, n_hat) is right-handed.
        center = p.mean(axis=0)
        _, _, vt = np.linalg.svd(p - center)
        e_u, n_hat = vt[0], vt[2]
        if n_hat[2] < 0.0:
            n_hat = -n_hat
        e_v = np.cross(n_hat, e_u)
        self._center = center
        self._e_u = e_u
        self._e_v = e_v
        self._n_hat = n_hat

        uv = (p - center) @ np.column_stack([e_u, e_v])

        if knots is not None:
            knots_uv = np.asarray(knots, dtype=float)
            if knots_uv.ndim != 2 or knots_uv.shape[1] != 2:
                raise ValueError("knots must have shape (nk, 2)")
            diff = knots_uv[:, None, :] - knots_uv[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dist, np.inf)
            default_length_scale = float(np.median(dist.min(axis=1)))
        else:
            n_u, n_v = n_knots
            u_vals = np.linspace(uv[:, 0].min(), uv[:, 0].max(), n_u)
            v_vals = np.linspace(uv[:, 1].min(), uv[:, 1].max(), n_v)
            uu, vv = np.meshgrid(u_vals, v_vals)
            knots_uv = np.column_stack([uu.ravel(), vv.ravel()])
            du = (u_vals[-1] - u_vals[0]) / (n_u - 1) if n_u > 1 else 0.0
            dv = (v_vals[-1] - v_vals[0]) / (n_v - 1) if n_v > 1 else 0.0
            default_length_scale = max(du, dv)

        if length_scale is not None:
            self._length_scale = float(length_scale)
        elif default_length_scale > 0.0:
            self._length_scale = default_length_scale
        else:
            raise ValueError(
                "Could not infer a default length_scale from the knot "
                "layout; pass length_scale explicitly."
            )

        self.knots_uv = knots_uv
        self.knots_xyz = (
            center + np.outer(knots_uv[:, 0], e_u) + np.outer(knots_uv[:, 1], e_v)
        )
        self._n_knots = knots_uv.shape[0]

        two_l2 = 2.0 * self._length_scale**2
        kk_diff = knots_uv[:, None, :] - knots_uv[None, :, :]
        phi_kk = np.exp(-np.sum(kk_diff**2, axis=-1) / two_l2)
        a_mat = phi_kk + ridge * np.eye(self._n_knots)
        pk_diff = uv[:, None, :] - knots_uv[None, :, :]
        phi_pk = np.exp(-np.sum(pk_diff**2, axis=-1) / two_l2)
        self._b = scipy.linalg.solve(a_mat, phi_pk.T, assume_a="pos").T

    @property
    def n_knots(self) -> int:
        """Number of control knots."""
        return self._n_knots

    @property
    def length_scale(self) -> float:
        """Gaussian RBF length scale, in meters."""
        return self._length_scale

    @property
    def normal(self) -> np.ndarray:
        """Best-fit-plane unit normal, shape (3,), oriented upward."""
        return self._n_hat

    def offsets(self, theta: npt.ArrayLike) -> Any:
        """Per-flat-vertex normal offsets in meters, ``B @ theta``.

        Traceable — works under ``jit``/``vmap``/``jax.jacfwd`` on the
        JAX backend, and plain NumPy otherwise.

        Args:
            theta: Knot offsets, shape (n_knots,).

        Returns:
            Offsets, shape (3 * n_triangles,).
        """
        return backend.xp.asarray(self._b) @ backend.xp.asarray(theta)

    def vertices(self, theta: npt.ArrayLike) -> Any:
        """Warped mesh vertices, ``V0 + normal * offsets(theta)``.

        Traceable — works under ``jit``/``vmap``/``jax.jacfwd`` on the
        JAX backend, and plain NumPy otherwise.

        Args:
            theta: Knot offsets, shape (n_knots,).

        Returns:
            Vertex coordinates, shape (n_triangles, 3, 3).
        """
        off = self.offsets(theta)
        v_flat = backend.xp.asarray(self._v0_flat) + off[:, None] * backend.xp.asarray(
            self._n_hat
        )
        return backend.xp.reshape(v_flat, self._shape)

    def check(self, theta: npt.ArrayLike) -> bool:
        """True iff every warped vertex satisfies the half-space ``z <= 0``.

        Args:
            theta: Knot offsets, shape (n_knots,).

        Returns:
            Whether the warped mesh stays entirely underground.
        """
        off = backend.to_numpy(self.offsets(np.asarray(theta, dtype=float)))
        v_flat = self._v0_flat + off[:, None] * self._n_hat
        return bool(np.all(v_flat[:, 2] <= 0.0))

    def fault(self, theta: npt.ArrayLike) -> Fault:
        """Build a concrete ``Fault`` at warp parameters ``theta``.

        Uses the reference fault's (mean-centroid) frame, so the result
        is a drop-in replacement for the reference fault in any existing
        forward-modeling or plotting tool.

        Args:
            theta: Knot offsets, shape (n_knots,).

        Returns:
            A triangular ``Fault`` with the warped geometry.
        """
        verts = backend.to_numpy(self.vertices(np.asarray(theta, dtype=float)))
        return Fault.from_triangles(
            verts.astype(float),
            frame=self.frame,
            medium=self._ref_fault.medium,
        )

    def plot(self, theta: npt.ArrayLike | None = None, ax: Any = None) -> tuple:
        """3D preview of the reference mesh, an optional warp, and knots.

        A quick way to sanity-check a warp's geometry — and set sensible
        ``knot_prior`` bounds via :meth:`check` — before handing it to
        :class:`TriPosterior`.

        Args:
            theta: Warp parameters to preview, or None to show only the
                reference mesh.
            ax: Existing 3D matplotlib axes to draw on; a new figure and
                axes are created when None.

        Returns:
            Tuple ``(fig, ax)``.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
        else:
            fig = ax.figure

        ref_tris = self._v0_flat.reshape(self._shape)
        wire = Poly3DCollection(
            ref_tris, facecolors="none", edgecolors="0.6", linewidths=0.6
        )
        ax.add_collection3d(wire)
        all_pts = ref_tris.reshape(-1, 3)

        if theta is not None:
            theta_arr = np.asarray(theta, dtype=float)
            verts = backend.to_numpy(self.vertices(theta_arr))
            per_vertex = backend.to_numpy(self.offsets(theta_arr)).reshape(
                self._n_tri, 3
            )
            tri_offset = per_vertex.mean(axis=1)
            vmax = float(np.max(np.abs(tri_offset))) or 1.0
            cmap = plt.get_cmap("coolwarm")
            colors = cmap(0.5 + 0.5 * tri_offset / vmax)
            warped = Poly3DCollection(
                verts, facecolors=colors, edgecolors="k", linewidths=0.3, alpha=0.9
            )
            ax.add_collection3d(warped)
            all_pts = np.concatenate([all_pts, verts.reshape(-1, 3)])

        kx = self.knots_xyz
        ax.scatter(kx[:, 0], kx[:, 1], kx[:, 2], color="k", s=25, depthshade=False)
        for i in range(self._n_knots):
            ax.text(kx[i, 0], kx[i, 1], kx[i, 2], f"k{i}")

        pts = np.concatenate([all_pts, kx])
        center = pts.mean(axis=0)
        radius = float(np.max(np.linalg.norm(pts - center, axis=1))) or 1.0
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_zlabel("Up (m)")
        return fig, ax


def _parse_knot_prior(
    knot_prior: tuple | Sequence[tuple], n_knots: int, parse: Any
) -> list[tuple]:
    """Normalize a ``TriPosterior`` ``knot_prior`` to ``n_knots`` specs.

    Accepts a single ``(lo, hi)`` / ``('normal', mu, sd)`` spec applied to
    every knot, or a sequence of exactly ``n_knots`` per-knot specs.

    Args:
        knot_prior: The user-supplied prior spec(s).
        n_knots: Number of knots (``warp.n_knots``).
        parse: ``_CollapsedPosterior._parse_prior``, injected to avoid an
            import cycle.

    Returns:
        List of ``n_knots`` parsed specs.

    Raises:
        ValueError: If a sequence of per-knot specs has the wrong length.
    """
    is_single = (
        len(knot_prior) == 2 and all(isinstance(v, (int, float)) for v in knot_prior)
    ) or (len(knot_prior) == 3 and knot_prior[0] == "normal")
    if is_single:
        return [parse(f"knot{i}", knot_prior) for i in range(n_knots)]
    if len(knot_prior) != n_knots:
        raise ValueError(
            f"knot_prior sequence must have length warp.n_knots ({n_knots}), "
            f"got {len(knot_prior)}"
        )
    return [parse(f"knot{i}", spec) for i, spec in enumerate(knot_prior)]


class TriPosterior(_CollapsedPosterior):
    """Collapsed log-posterior for a warped triangular-mesh geometry.

    Uses the same collapsed slip machinery as :class:`RectPosterior` —
    the slip prior stays Gaussian, so slip is marginalized analytically
    — but the forward model is a linear normal-offset warp
    (:class:`TriWarp`) of a fixed-connectivity triangular mesh instead of
    a rectangular patch grid. There is no positivity path here (future
    work).

    The sampled parameter vector ``x`` stacks the warp's knot offsets
    (meters, ``warp.n_knots`` of them), then ``log10_sigma``, then
    ``log10_lambda`` when ``mode='hierarchical'``.

    Args:
        warp: A :class:`TriWarp` built from the reference triangular
            mesh; ``warp.vertices(theta)`` supplies the forward geometry.
        datasets: One or more displacement datasets (GNSS, InSAR,
            Vertical).
        knot_prior: Prior for every knot offset: a single ``(lo, hi)``
            or ``('normal', mu, sd)`` applied to all knots, or a sequence
            of ``warp.n_knots`` per-knot specs.
        knots0: Initial knot offsets (meters), shape (n_knots,) — the
            sampler's starting point, clipped into uniform prior bounds.
            Defaults to zeros (the reference mesh). Start from your best
            estimate when you have one: with tight data and a start far
            from the mode, the huge initial misfit makes the posterior
            extremely stiff in ``log10_sigma`` and warmup can fail to
            adapt (the same reason ``RectPosterior`` starts at
            ``theta0``).
        components: Slip components for the marginalized linear solve:
            ``'both'``, ``'strike'``, or ``'dip'``.
        mode: Slip-prior mode: ``'hierarchical'``, ``'weak'``, or
            ``'profiled'`` (see the module docstring).
        regularization: Regularization operator for the hierarchical and
            profiled modes; must be None for ``'weak'``.
        regularization_strength: Fixed lambda for ``'profiled'``; initial
            lambda (sampler starting point) for ``'hierarchical'``.
        slip_scale: Prior slip scale in meters for ``'weak'``.
        log10_sigma_prior: Uniform prior bounds on ``log10_sigma``.
        log10_lambda_prior: Uniform prior bounds on ``log10_lambda``
            (hierarchical mode only).
        nu: Poisson's ratio.

    Raises:
        RuntimeError: If the JAX backend is not active.
        ValueError: If ``knot_prior``, ``mode``, or ``components`` is
            invalid or inconsistent.
    """

    def __init__(
        self,
        warp: TriWarp,
        datasets: DataSet | list[DataSet],
        *,
        knot_prior: tuple | Sequence[tuple],
        knots0: npt.ArrayLike | None = None,
        components: str = "both",
        mode: str = "hierarchical",
        regularization: str | np.ndarray | None = "laplacian",
        regularization_strength: float | None = None,
        slip_scale: float | None = None,
        log10_sigma_prior: tuple[float, float] = (-2.0, 2.0),
        log10_lambda_prior: tuple[float, float] = (-8.0, 8.0),
        nu: float = 0.25,
    ) -> None:
        _require_jax()
        if isinstance(datasets, DataSet):
            datasets = [datasets]
        if knots0 is not None:
            knots0 = np.asarray(knots0, dtype=float)
            if knots0.shape != (warp.n_knots,):
                raise ValueError(
                    f"knots0 must have shape ({warp.n_knots},), got {knots0.shape}"
                )
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        if components not in ("both", "strike", "dip"):
            raise ValueError(
                f"components must be 'both', 'strike', or 'dip', got {components!r}"
            )
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

        self.mode = mode
        self.warp = warp
        self.components = components
        self.datasets = datasets
        self._nu = float(nu)
        nk = warp.n_knots

        ref_fault = warp._ref_fault
        sys = LinearSystem(ref_fault, datasets, regularization, components)
        n_patches = ref_fault.n_patches
        self._col_start, self._col_stop = {
            "both": (0, 2 * n_patches),
            "strike": (0, n_patches),
            "dip": (n_patches, 2 * n_patches),
        }[components]
        n_slip = self._col_stop - self._col_start
        self._n_slip = n_slip
        self._n_patches = n_patches

        frame = warp.frame
        self.frame = frame
        e_parts, n_parts = [], []
        for ds in datasets:
            enu = frame.to_enu(
                lon=ds.lon,
                lat=ds.lat,
                alt=np.full(ds.n_stations, frame.origin_alt),
            )
            e_parts.append(enu[:, 0])
            n_parts.append(enu[:, 1])
        e_obs = np.concatenate(e_parts)
        n_obs = np.concatenate(n_parts)
        self._obs = np.column_stack([e_obs, n_obs, np.zeros_like(e_obs)])

        self._W_half = scipy.linalg.cholesky(sys.W, lower=False)
        self._W_half_P = self._W_half @ _projection_matrix(datasets)
        self._d_w = self._W_half @ sys.d
        self.n_data = len(self._d_w)

        # Slip-prior precision structure: identical to RectPosterior's.
        if mode == "weak":
            assert slip_scale is not None
            self._LtL = np.eye(n_slip)
            self._lambda_fixed: float | None = 1.0 / slip_scale**2
            self._logdet_rank = n_slip
            self._logdet_sum = 0.0
        else:
            if regularization is not None:
                self._LtL = sys.LtL
                eig = np.abs(np.linalg.eigvalsh(self._LtL))
            else:
                self._LtL = np.zeros((n_slip, n_slip))
                eig = np.zeros(n_slip)
            if mode == "profiled":
                assert regularization_strength is not None
                self._lambda_fixed = float(regularization_strength)
            else:
                self._lambda_fixed = None
            pos = _rank_positive_eigs(eig)
            self._logdet_rank = len(pos)
            self._logdet_sum = float(np.sum(np.log(pos)))
        self._include_logdet = mode != "profiled"

        # Sampled-parameter layout and priors: knot offsets, then hypers.
        specs = _parse_knot_prior(knot_prior, nk, self._parse_prior)
        k0 = np.zeros(nk) if knots0 is None else knots0
        x0 = [
            float(np.clip(k0[i], spec[1], spec[2]))
            if spec[0] == "uniform"
            else float(k0[i])
            for i, spec in enumerate(specs)
        ]
        self.param_names = [f"knot{i}" for i in range(nk)] + ["log10_sigma"]
        specs.append(("uniform",) + tuple(map(float, log10_sigma_prior)))
        x0.append(float(np.clip(0.0, *log10_sigma_prior)))
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
        self.free = [f"knot{i}" for i in range(nk)]

        self._is_uniform = np.array([s[0] == "uniform" for s in specs])
        self._lo = np.array([s[1] if s[0] == "uniform" else -np.inf for s in specs])
        self._hi = np.array([s[2] if s[0] == "uniform" else np.inf for s in specs])
        self._mu = np.array([s[1] if s[0] == "normal" else 0.0 for s in specs])
        self._sd = np.array([s[2] if s[0] == "normal" else 1.0 for s in specs])

        self._logpdf_fn = self._build_logpdf()

    def _assemble(self, x: Any) -> tuple:
        """Marginalization ingredients at sampled parameters x (traceable).

        Warps the reference mesh via ``warp.vertices(theta)``, clipping
        vertex depths to the half-space (``z <= 0``) before the tri
        kernel sees them so leapfrog excursions outside the prior bounds
        never hit an undefined geometry — the actual half-space
        violation is instead penalized in :meth:`log_prior`.

        Returns:
            Tuple ``(sigma2, lam, G_w, chol_H, m_hat, S)``, see
            :meth:`_CollapsedPosterior._assemble`.
        """
        import jax.numpy as jnp

        x = self._clip(x)
        nk = self.warp.n_knots
        theta = x[:nk]
        sigma2 = 10.0 ** (2.0 * x[nk])
        if self._lambda_fixed is not None:
            lam = jnp.asarray(self._lambda_fixed)
        else:
            lam = 10.0 ** x[nk + 1]

        v = self.warp.vertices(theta)
        v = v.at[..., 2].set(jnp.minimum(v[..., 2], 0.0))
        g3 = tri_greens(v, self._obs, self._nu)
        g_w = (jnp.asarray(self._W_half_P) @ g3)[:, self._col_start : self._col_stop]
        chol_h, m_hat, s = self._collapse_terms(g_w, lam)
        return sigma2, lam, g_w, chol_h, m_hat, s

    def log_prior(self, x: npt.ArrayLike) -> np.ndarray:
        """Base prior plus a traceable half-space guard on the warp.

        Adds ``-inf`` when the (bound-clipped) sampled knots would push
        any vertex of the *unclipped* warp above ``z = 0`` — the guard
        checks the true geometry, unlike :meth:`_assemble`'s numerical
        z-clamp, which only keeps the kernel's inputs well-defined.

        Args:
            x: Sampled parameter vector, ordered as ``param_names``.

        Returns:
            Scalar log-prior; ``-inf`` outside uniform bounds or when the
            warp violates the half-space.
        """
        import jax.numpy as jnp

        base = super().log_prior(x)
        x_c = self._clip(x)
        theta = x_c[: self.warp.n_knots]
        max_z = jnp.max(self.warp.vertices(theta)[..., 2])
        guard = jnp.where(max_z > 0.0, -jnp.inf, 0.0)
        return cast(np.ndarray, base + guard)


class SlipPosterior:
    """Joint log-posterior over slip and scales at fixed fault geometry.

    :class:`RectPosterior` marginalizes slip analytically, which is only
    possible because its slip prior is Gaussian. Enforcing positivity
    (a truncated-Gaussian slip prior) breaks that conjugacy, so
    ``SlipPosterior`` samples slip jointly with ``log10_sigma`` (and
    optionally ``log10_lambda``) instead of collapsing it — geometry is
    fixed, but ``fault`` may be *any* :class:`~geodef.fault.Fault`
    (rectangular or triangular-mesh), since the Green's matrix is
    assembled once at construction and never re-touched.

    Positivity is enforced exactly by a softplus reparameterization of a
    whitened Gaussian: the sampled vector stacks ``z`` (one entry per
    slip component, in the same whitened space regardless of whether
    that component is constrained) with the hyperparameters. A fixed
    reference linear system ``H0 = G_w^T G_w + lambda_ref LtL`` (Cholesky
    factor ``L0``) defines an affine map ``z -> v`` centered at the
    reference ridge solution ``mu0``; constrained components then pass
    through ``softplus`` (``jax.nn.softplus``) to become non-negative
    slip, unconstrained components pass through unchanged. Because the
    reference system is fixed, every ``log_prob`` gradient costs exactly
    one matrix-vector product — plain reverse-mode ``jax.grad`` is fast
    here, unlike :class:`RectPosterior` (whose ``custom_jvp`` forward-mode
    wrapper exists only because okada kernels sit inside its trace; here
    they do not).

    Mode names deliberately differ from :class:`RectPosterior`: there is
    no ``'profiled'`` mode, because nothing here is ever profiled out —
    slip is sampled, not point-estimated. ``'fixed'`` is the analog of
    ``RectPosterior``'s ``'profiled'`` mode (a single, user-chosen
    lambda) but is a proper joint posterior, including the Occam
    (log-determinant) terms of the slip prior.

    The slip prior is a truncated Gaussian, zero-mean with covariance
    proportional to ``(sigma^2/lambda) K^+`` (``K = LtL``, ``K^+`` its
    pseudoinverse), restricted to the orthant selected by ``positive``.
    Its truncation normalizer ``Z`` is dropped from the density: for a
    zero-mean Gaussian, the probability mass falling in an orthant (a
    cone) does not depend on the overall covariance scale, so ``Z`` is
    the same constant for every ``(sigma, lambda)`` and cancels out of
    any posterior that samples them — including the hierarchical
    ``log10_lambda`` marginal, whose exactness under positivity relies
    on precisely this scale invariance.

    Args:
        fault: Fixed fault geometry (any ``Fault``).
        datasets: One or more displacement datasets (GNSS, InSAR,
            Vertical).
        components: Slip components to sample: ``'both'``, ``'strike'``,
            or ``'dip'``.
        mode: Slip-prior mode:

            - ``'hierarchical'``: regularization operator, ``log10_lambda``
              sampled (requires ``regularization``).
            - ``'fixed'``: regularization operator at a fixed
              ``regularization_strength`` — a proper posterior at one
              lambda, not a profile (requires ``regularization`` and
              ``regularization_strength``).
            - ``'weak'``: identity prior with a fixed ``slip_scale``
              (requires ``slip_scale``; ``regularization`` must be None).
        regularization: Regularization operator (as in ``LinearSystem``) for
            the hierarchical and fixed modes; must be None for
            ``'weak'``.
        regularization_strength: Fixed lambda for ``'fixed'``; reference
            lambda (whitening only, not sampled) for ``'hierarchical'``
            when given, else the midpoint of ``log10_lambda_prior``.
        slip_scale: Prior slip scale in meters for ``'weak'`` — the
            prior is ``m ~ N(0, (sigma * slip_scale)^2 I)`` truncated to
            the constrained orthant.
        positive: Which slip components are non-negative:

            - ``None`` — no positivity constraint.
            - ``'both'`` — every sampled slip component (valid for any
              ``components``).
            - ``'strike'`` — the strike-slip block (requires
              ``components in ('both', 'strike')``).
            - ``'dip'`` — the dip-slip block (requires
              ``components in ('both', 'dip')``).
            - A bool array of length ``n_slip``, True where positive.
        log10_sigma_prior: Uniform prior bounds on ``log10_sigma``.
        log10_lambda_prior: Uniform prior bounds on ``log10_lambda``
            (hierarchical mode only).

    Raises:
        RuntimeError: If the JAX backend is not active.
        ValueError: If ``mode``, ``components``, or ``positive`` is
            invalid, or a mode-required argument is missing.
    """

    def __init__(
        self,
        fault: Fault,
        datasets: DataSet | list[DataSet],
        *,
        components: str = "both",
        mode: str = "hierarchical",
        regularization: str | np.ndarray | None = "laplacian",
        regularization_strength: float | None = None,
        slip_scale: float | None = None,
        positive: str | npt.ArrayLike | None = None,
        log10_sigma_prior: tuple[float, float] = (-2.0, 2.0),
        log10_lambda_prior: tuple[float, float] = (-8.0, 8.0),
    ) -> None:
        _require_jax()
        if isinstance(datasets, DataSet):
            datasets = [datasets]
        if mode not in _VALID_SLIP_MODES:
            raise ValueError(f"mode must be one of {_VALID_SLIP_MODES}, got {mode!r}")
        if components not in ("both", "strike", "dip"):
            raise ValueError(
                f"components must be 'both', 'strike', or 'dip', got {components!r}"
            )
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
        if mode == "fixed":
            if regularization is None:
                raise ValueError("mode='fixed' requires a regularization operator")
            if regularization_strength is None:
                raise ValueError(
                    "mode='fixed' requires a fixed regularization_strength (lambda)"
                )

        sys = LinearSystem(fault, datasets, regularization, components)
        self.mode = mode
        self.components = components
        self._G_w = np.asarray(sys.G_w, dtype=np.float64)
        self._d_w = np.asarray(sys.d_w, dtype=np.float64)
        self._G = np.asarray(sys.G, dtype=np.float64)
        self.n_data = len(self._d_w)
        n_params = self._G_w.shape[1]
        self._n_slip = n_params

        # Slip-prior precision structure: lambda * K, with the
        # lambda-independent pseudo-determinant pieces (rank and sum of
        # log positive eigenvalues) precomputed, matching RectPosterior.
        if mode == "weak":
            assert slip_scale is not None
            self._K = np.eye(n_params)
            lam_ref = 1.0 / slip_scale**2
            self._logdet_rank = n_params
            self._logdet_sum = 0.0
        else:
            self._K = np.asarray(sys.LtL, dtype=np.float64)
            eig = np.abs(np.linalg.eigvalsh(self._K))
            pos = _rank_positive_eigs(eig)
            self._logdet_rank = len(pos)
            self._logdet_sum = float(np.sum(np.log(pos)))
            if mode == "fixed":
                assert regularization_strength is not None
                lam_ref = float(regularization_strength)
            else:
                lam_ref = (
                    float(regularization_strength)
                    if regularization_strength
                    else 10.0 ** (0.5 * (log10_lambda_prior[0] + log10_lambda_prior[1]))
                )
        self._lambda_fixed: float | None = None if mode == "hierarchical" else lam_ref

        self._mask = _parse_positive(positive, components, fault.n_patches, n_params)

        # Whitening reference: a fixed ridge solution and its Cholesky
        # factor define the affine z -> v map. Using the reference
        # (lam_ref, sigma_ref=1) rather than the sampled (lambda, sigma)
        # keeps the map a pure reparameterization — correctness comes
        # from the Jacobian below, conditioning from a reasonable
        # reference.
        sigma_ref = 1.0
        H0 = self._G_w.T @ self._G_w + lam_ref * self._K
        L0 = scipy.linalg.cholesky(H0, lower=True)
        mu0 = scipy.linalg.cho_solve((L0, True), self._G_w.T @ self._d_w)
        self._sigma_ref = sigma_ref
        self._L0 = L0
        self._mu0 = mu0
        self._logJ_affine = n_params * np.log(sigma_ref) - float(
            np.sum(np.log(np.diagonal(L0)))
        )

        # Sampled-parameter layout and priors.
        self.param_names = [f"z{i}" for i in range(n_params)] + ["log10_sigma"]
        lo = [-np.inf] * n_params + [float(log10_sigma_prior[0])]
        hi = [np.inf] * n_params + [float(log10_sigma_prior[1])]
        x0 = [0.0] * n_params + [float(np.clip(0.0, *log10_sigma_prior))]
        self._log10_sigma_prior = (
            float(log10_sigma_prior[0]),
            float(log10_sigma_prior[1]),
        )
        self._log10_lambda_prior: tuple[float, float] | None = None
        if mode == "hierarchical":
            self.param_names.append("log10_lambda")
            self._log10_lambda_prior = (
                float(log10_lambda_prior[0]),
                float(log10_lambda_prior[1]),
            )
            lo.append(float(log10_lambda_prior[0]))
            hi.append(float(log10_lambda_prior[1]))
            x0.append(float(np.clip(np.log10(lam_ref), *log10_lambda_prior)))
        self.x0 = np.array(x0)
        self.n_params = len(self.param_names)
        self._lo = np.array(lo)
        self._hi = np.array(hi)

    # ------------------------------------------------------------------
    # Traceable density pieces
    # ------------------------------------------------------------------

    def _clip(self, x: Any) -> Any:
        """Clip uniform-prior hyperparameters into bounds (traceable).

        Slip entries (``z``) carry ``+-inf`` bounds, so they pass
        through unchanged; only ``log10_sigma`` (and ``log10_lambda``)
        are actually clipped.
        """
        import jax.numpy as jnp

        return jnp.clip(jnp.asarray(x), jnp.asarray(self._lo), jnp.asarray(self._hi))

    def _transform(self, x: Any) -> tuple:
        """Sigma^2, lambda, slip, and log-Jacobian at sampled x (traceable).

        Maps the whitened ``z`` block of (bound-clipped) ``x`` through
        the fixed-reference affine whitening and a per-component
        softplus (where ``positive`` constrains that component) to
        produce the slip vector ``m``, plus the log-Jacobian of that
        map.

        Returns:
            Tuple ``(sigma2, lam, m, logJ)``.
        """
        import jax.numpy as jnp

        x = self._clip(x)
        n_z = self._n_slip
        z = x[:n_z]
        sigma2 = 10.0 ** (2.0 * x[n_z])
        if self._lambda_fixed is not None:
            lam = jnp.asarray(self._lambda_fixed)
        else:
            lam = 10.0 ** x[n_z + 1]

        m, logJ = _slip_transform(
            z, self._mu0, self._L0, self._mask, self._sigma_ref, self._logJ_affine
        )
        return sigma2, lam, m, logJ

    def log_likelihood(self, x: npt.ArrayLike) -> np.ndarray:
        """Log-likelihood ``log p(d_w | m, sigma)`` (traceable).

        Args:
            x: Sampled parameter vector, ordered as ``param_names``.

        Returns:
            Scalar log-likelihood.
        """
        import jax.numpy as jnp

        sigma2, _, m, _ = self._transform(x)
        r = jnp.asarray(self._d_w) - jnp.asarray(self._G_w) @ m
        n = self.n_data
        ll = -0.5 * n * jnp.log(2.0 * jnp.pi * sigma2) - (r @ r) / (2.0 * sigma2)
        return cast(np.ndarray, ll)

    def log_prior(self, x: npt.ArrayLike) -> np.ndarray:
        """Log-prior over slip and scales (traceable).

        Combines the uniform priors on ``log10_sigma`` (and
        ``log10_lambda``), the (unnormalized-by-a-constant) truncated
        Gaussian slip prior ``log p(m | sigma, lambda)``, and the
        whitening log-Jacobian.

        Args:
            x: Sampled parameter vector, ordered as ``param_names``.

        Returns:
            Scalar log-prior; ``-inf`` outside uniform bounds.
        """
        import jax.numpy as jnp

        x_raw = jnp.asarray(x)
        sigma2, lam, m, logJ = self._transform(x)
        p = self._n_slip

        lo_s, hi_s = self._log10_sigma_prior
        log10_sigma = x_raw[p]
        in_bounds = (log10_sigma >= lo_s) & (log10_sigma <= hi_s)
        lp = jnp.where(in_bounds, -jnp.log(hi_s - lo_s), -jnp.inf)

        if self._log10_lambda_prior is not None:
            lo_l, hi_l = self._log10_lambda_prior
            log10_lambda = x_raw[p + 1]
            in_bounds_l = (log10_lambda >= lo_l) & (log10_lambda <= hi_l)
            lp = lp + jnp.where(in_bounds_l, -jnp.log(hi_l - lo_l), -jnp.inf)

        K = jnp.asarray(self._K)
        quad = m @ K @ m
        log_prior_m = (
            -0.5 * p * jnp.log(2.0 * jnp.pi * sigma2)
            + 0.5 * (self._logdet_rank * jnp.log(lam) + self._logdet_sum)
            - lam * quad / (2.0 * sigma2)
        )
        return cast(np.ndarray, lp + log_prior_m + logJ)

    def logpdf(self, x: npt.ArrayLike) -> np.ndarray:
        """Log-posterior density (traceable, differentiable).

        Equals ``log_prior(x) + log_likelihood(x)``; a plain traceable
        function (no ``custom_jvp``) since each evaluation costs only
        one matvec through the fixed Green's matrix, so reverse-mode
        ``jax.grad`` is already fast.

        Args:
            x: Sampled parameter vector, ordered as ``param_names``.

        Returns:
            Scalar log-posterior; ``-inf`` outside uniform prior bounds.
        """
        return cast(np.ndarray, self.log_prior(x) + self.log_likelihood(x))

    # ------------------------------------------------------------------
    # Slip and posterior predictive
    # ------------------------------------------------------------------

    def slip_of(self, x: npt.ArrayLike) -> np.ndarray:
        """Transform one sampled vector to its slip vector.

        Args:
            x: Sampled parameter vector, ordered as ``param_names``.

        Returns:
            Slip vector, shape (n_slip,).
        """
        m = self._transform(np.asarray(x, dtype=float))[2]
        return backend.to_numpy(m)

    def _transform_and_predict(self, x: Any) -> tuple:
        """Slip and unweighted predicted data at x (traceable)."""
        import jax.numpy as jnp

        m = self._transform(x)[2]
        d_pred = jnp.asarray(self._G) @ m
        return m, d_pred

    def _vmapped_transform(self, samples: npt.ArrayLike) -> tuple:
        """Slip and predictions for each sample row (vmapped, jitted)."""
        jax = _require_jax()

        samples = np.atleast_2d(np.asarray(samples, dtype=float))
        m, d_pred = jax.jit(jax.vmap(self._transform_and_predict))(samples)
        return backend.to_numpy(m), backend.to_numpy(d_pred)

    def slip_draws(self, samples: npt.ArrayLike) -> np.ndarray:
        """Slip vector at each posterior sample.

        Unlike :meth:`RectPosterior.slip_draws`, this is a deterministic
        transform, not a random draw: slip is part of the sampled state
        ``x``, so each row of ``samples`` already determines one slip
        vector exactly (no seed).

        Args:
            samples: Sampled parameter vectors, shape (n, n_params).

        Returns:
            Slip draws, shape (n, n_slip). Columns follow the
            components layout (``[:N]`` strike-slip then ``[N:]``
            dip-slip when components='both').
        """
        return self._vmapped_transform(samples)[0]

    def predict(self, samples: npt.ArrayLike) -> np.ndarray:
        """Predicted data at each posterior sample.

        Args:
            samples: Sampled parameter vectors, shape (n, n_params).

        Returns:
            Predictions ``sys.G @ m``, shape (n, n_data), unweighted
            and projected like the stacked observation vector.
        """
        return self._vmapped_transform(samples)[1]


def _parse_positive(
    positive: str | npt.ArrayLike | None,
    components: str,
    n_patches: int,
    n_params: int,
) -> np.ndarray:
    """Resolve the ``positive`` argument of :class:`SlipPosterior` to a mask.

    Args:
        positive: None, 'strike', 'dip', 'both', or a bool array.
        components: The posterior's sampled slip components.
        n_patches: Number of fault patches (one block's width).
        n_params: Total number of sampled slip components.

    Returns:
        Bool array, shape (n_params,), True where positivity-constrained.

    Raises:
        ValueError: If ``positive`` names a block absent from
            ``components``, is an unrecognized string, or is an array
            of the wrong length.
    """
    if positive is None:
        return np.zeros(n_params, dtype=bool)
    if isinstance(positive, str):
        if positive == "both":
            return np.ones(n_params, dtype=bool)
        if positive == "strike":
            if components not in ("both", "strike"):
                raise ValueError(
                    "positive='strike' requires a sampled strike-slip block "
                    f"(components={components!r})"
                )
            mask = np.zeros(n_params, dtype=bool)
            if components == "both":
                mask[:n_patches] = True
            else:
                mask[:] = True
            return mask
        if positive == "dip":
            if components not in ("both", "dip"):
                raise ValueError(
                    "positive='dip' requires a sampled dip-slip block "
                    f"(components={components!r})"
                )
            mask = np.zeros(n_params, dtype=bool)
            if components == "both":
                mask[n_patches:] = True
            else:
                mask[:] = True
            return mask
        raise ValueError(
            "positive must be None, 'strike', 'dip', 'both', or a bool array, "
            f"got {positive!r}"
        )
    positive_mask = np.asarray(positive, dtype=bool)
    if positive_mask.shape != (n_params,):
        raise ValueError(
            f"positive array must have shape ({n_params},), got {positive_mask.shape}"
        )
    return positive_mask


# ======================================================================
# Convergence diagnostics
# ======================================================================


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


# ======================================================================
# NUTS sampling
# ======================================================================


@dataclasses.dataclass(frozen=True)
class PosteriorResult:
    """Posterior draws and diagnostics from :func:`sample`.

    Attributes:
        param_names: Sampled parameter names, matching the last axis of
            ``samples``.
        samples: Posterior draws, shape (n_chains, n_samples, n_params).
        log_prob: Log-posterior at each draw, shape
            (n_chains, n_samples).
        acceptance_rate: Mean NUTS acceptance probability.
        n_divergent: Total number of divergent transitions (should be
            ~0; many divergences mean the results are unreliable).
        rhat: Split R-hat per parameter, shape (n_params,).
        ess: Effective sample size per parameter, shape (n_params,).
    """

    param_names: list[str]
    samples: np.ndarray
    log_prob: np.ndarray
    acceptance_rate: float
    n_divergent: int
    rhat: np.ndarray
    ess: np.ndarray

    @property
    def flat(self) -> np.ndarray:
        """All chains concatenated, shape (n_chains*n_samples, n_params)."""
        return self.samples.reshape(-1, len(self.param_names))

    def summary(self) -> dict[str, np.ndarray]:
        """Per-parameter posterior summary statistics.

        Returns:
            Dict with arrays of shape (n_params,) under the keys
            ``'mean'``, ``'sd'``, ``'q05'``, ``'q50'``, ``'q95'``,
            ``'rhat'``, and ``'ess'``.
        """
        flat = self.flat
        q05, q50, q95 = np.percentile(flat, [5, 50, 95], axis=0)
        return {
            "mean": flat.mean(axis=0),
            "sd": flat.std(axis=0, ddof=1),
            "q05": q05,
            "q50": q50,
            "q95": q95,
            "rhat": self.rhat,
            "ess": self.ess,
        }

    def plot_pairs(
        self,
        truths: Sequence[float] | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> tuple:
        """Corner-style pair plot: histograms on the diagonal, scatter below.

        Args:
            truths: Optional true parameter values to mark, one per
                sampled parameter.
            figsize: Optional figure size; defaults to 2 inches per
                parameter.

        Returns:
            Tuple ``(fig, axes)`` with axes of shape
            (n_params, n_params).
        """
        import matplotlib.pyplot as plt

        k = len(self.param_names)
        flat = self.flat
        if figsize is None:
            figsize = (2.0 * k, 2.0 * k)
        fig, axes = plt.subplots(k, k, figsize=figsize, squeeze=False)
        for i in range(k):
            for j in range(k):
                ax = axes[i, j]
                if j > i:
                    ax.set_axis_off()
                    continue
                if i == j:
                    ax.hist(flat[:, i], bins=40, color="C0", alpha=0.8)
                    if truths is not None:
                        ax.axvline(truths[i], color="C3", lw=1.2)
                    ax.set_yticks([])
                else:
                    ax.scatter(flat[:, j], flat[:, i], s=2, alpha=0.25, color="C0")
                    if truths is not None:
                        ax.axvline(truths[j], color="C3", lw=1.0)
                        ax.axhline(truths[i], color="C3", lw=1.0)
                if i == k - 1:
                    ax.set_xlabel(self.param_names[j])
                else:
                    ax.set_xticklabels([])
                if j == 0 and i > 0:
                    ax.set_ylabel(self.param_names[i])
        fig.tight_layout()
        return fig, axes


def sample(
    post: RectPosterior | SlipPosterior,
    *,
    n_samples: int = 1000,
    n_warmup: int = 1000,
    n_chains: int = 4,
    seed: int = 0,
    target_acceptance: float = 0.8,
    inits: npt.ArrayLike | None = None,
) -> PosteriorResult:
    """Sample a posterior with NUTS (blackjax) and report diagnostics.

    Runs blackjax window adaptation (step size + diagonal mass matrix)
    once from ``post.x0``, then draws all chains with the adapted
    kernel inside a single jitted, chain-vectorized (``vmap``)
    computation, so multi-core CPUs sample chains in parallel.

    By default the chains start from the warmup's end position,
    overdispersed by twice the adapted posterior scale — starting every
    chain back at ``x0`` would force each one to re-walk the approach
    to the mode with the small adapted step size. Pass explicit
    ``inits`` (e.g. spread across prior modes) to diagnose multimodal
    posteriors instead.

    Args:
        post: The posterior density to sample — :class:`RectPosterior`
            (collapsed geometry posterior) or :class:`SlipPosterior`
            (fixed-geometry joint slip posterior); only the shared
            ``x0``, ``logpdf``, ``n_params``, and bounds attributes are
            used.
        n_samples: Post-warmup draws per chain.
        n_warmup: Window-adaptation steps.
        n_chains: Number of chains.
        seed: PRNG seed for jittered starts and sampling.
        target_acceptance: NUTS target acceptance probability.
        inits: Optional explicit starting points, shape
            (n_chains, n_params).

    Returns:
        PosteriorResult with draws, log-probabilities, acceptance rate,
        divergence count, and split R-hat / ESS per parameter.

    Raises:
        RuntimeError: If the JAX backend is not active.
        ImportError: If blackjax is not installed.
    """
    jax = _require_jax()
    try:
        import blackjax
    except ImportError as err:
        raise ImportError(
            "Sampling requires blackjax. Install it with: pip install geodef[bayes]"
        ) from err
    import jax.numpy as jnp

    if inits is not None:
        inits_a = np.atleast_2d(np.asarray(inits, dtype=float))
        if inits_a.shape != (n_chains, post.n_params):
            raise ValueError(
                f"inits must have shape ({n_chains}, {post.n_params}), "
                f"got {inits_a.shape}"
            )
    else:
        inits_a = None

    key = jax.random.PRNGKey(seed)
    warm_key, run_key = jax.random.split(key)

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        post.logpdf,
        target_acceptance_rate=target_acceptance,
        progress_bar=False,
    )
    (adapt_state, nuts_params), _ = warmup.run(
        warm_key, jnp.asarray(post.x0), num_steps=n_warmup
    )
    nuts = blackjax.nuts(post.logpdf, **nuts_params)

    if inits_a is None:
        center = backend.to_numpy(adapt_state.position)
        scale = 2.0 * np.sqrt(backend.to_numpy(nuts_params["inverse_mass_matrix"]))
        rng = np.random.default_rng(seed)
        inits_a = center + scale * rng.standard_normal((n_chains, post.n_params))
        span = np.where(np.isfinite(post._hi - post._lo), post._hi - post._lo, 1.0)
        margin = 1e-6 * span
        inits_a = np.clip(inits_a, post._lo + margin, post._hi - margin)

    def run_chain(chain_key: Any, position: Any) -> tuple:
        state = nuts.init(position)
        keys = jax.random.split(chain_key, n_samples)

        def one_step(state: Any, step_key: Any) -> tuple:
            state, info = nuts.step(step_key, state)
            return state, (
                state.position,
                state.logdensity,
                info.acceptance_rate,
                info.is_divergent,
            )

        _, out = jax.lax.scan(one_step, state, keys)
        return out

    chain_keys = jax.random.split(run_key, n_chains)
    positions, logdens, accept, divergent = jax.jit(jax.vmap(run_chain))(
        chain_keys, jnp.asarray(inits_a)
    )

    samples = backend.to_numpy(positions)
    log_prob = backend.to_numpy(logdens)
    rhat = np.array([split_rhat(samples[:, :, i]) for i in range(post.n_params)])
    ess = np.array(
        [effective_sample_size(samples[:, :, i]) for i in range(post.n_params)]
    )
    return PosteriorResult(
        param_names=list(post.param_names),
        samples=samples,
        log_prob=log_prob,
        acceptance_rate=float(np.mean(backend.to_numpy(accept))),
        n_divergent=int(np.sum(backend.to_numpy(divergent))),
        rhat=rhat,
        ess=ess,
    )
