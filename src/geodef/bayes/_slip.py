"""Joint slip-and-scales posterior at fixed geometry, with positivity.

Private submodule of :mod:`geodef.bayes`. ``SlipPosterior`` samples the
full slip vector jointly with noise/regularization scales, supporting
positivity through the log-transform in ``_util._slip_transform``.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import scipy.linalg

from geodef import backend
from geodef.bayes._util import (
    _VALID_SLIP_MODES,
    _parse_positive,
    _require_jax,
    _slip_transform,
)
from geodef.data import DataSet
from geodef.fault import Fault
from geodef.invert import LinearSystem
from geodef.invert._solvers import _rank_positive_eigs


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
