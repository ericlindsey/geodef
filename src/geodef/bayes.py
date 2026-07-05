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

- ``'hierarchical'``: ``L`` is a smoothing operator (e.g. Laplacian) and
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
    post = geodef.bayes.RectPosterior(
        theta0, datasets, ref_lat=..., ref_lon=...,
        free=["dip", "depth"],
        theta_prior={"dip": (5.0, 60.0), "depth": (5e3, 40e3)},
        n_length=8, n_width=4, smoothing="laplacian",
    )
    log_density = post.logpdf(post.x0)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import scipy.linalg

from geodef import backend, transforms
from geodef.data import DataSet
from geodef.fault import Fault
from geodef.gradients import rect_greens
from geodef.invert import _THETA_NAMES, LinearSystem, _projection_matrix

_VALID_MODES = ("hierarchical", "weak", "profiled")
_LN10 = float(np.log(10.0))


def _require_jax() -> Any:
    """Return the jax module, or raise if the JAX backend is not active."""
    if backend.get_backend() != "jax":
        raise RuntimeError(
            "geodef.bayes requires the JAX backend; "
            "call geodef.backend.set_backend('jax') first."
        )
    import jax

    return jax


class RectPosterior:
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
        theta0: Template geometry ``[e0, n0, depth, strike, dip,
            length, width]``; fixed parameters keep these values and
            free ones are initialized from them. ``e0``/``n0`` are
            centroid offsets in meters from (ref_lat, ref_lon).
        datasets: One or more displacement datasets (GNSS, InSAR,
            Vertical).
        ref_lat: Latitude anchoring the local Cartesian frame.
        ref_lon: Longitude anchoring the local Cartesian frame.
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
        smoothing: Regularization operator (as in ``invert()``) for the
            hierarchical and profiled modes; must be None for
            ``'weak'``.
        smoothing_strength: Fixed lambda for ``'profiled'``; initial
            lambda (sampler starting point) for ``'hierarchical'``.
        slip_scale: Prior slip scale in meters for ``'weak'`` — the
            prior is ``m ~ N(0, (sigma * slip_scale)^2 I)``.
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
        theta0: npt.ArrayLike,
        datasets: DataSet | list[DataSet],
        *,
        ref_lat: float,
        ref_lon: float,
        free: Sequence[str] = ("depth", "dip"),
        theta_prior: dict[str, tuple] | None = None,
        n_length: int = 1,
        n_width: int = 1,
        components: str = "both",
        mode: str = "hierarchical",
        smoothing: str | np.ndarray | None = "laplacian",
        smoothing_strength: float | None = None,
        slip_scale: float | None = None,
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
            if smoothing is not None:
                raise ValueError(
                    "mode='weak' uses an identity slip prior; smoothing must be None"
                )
        if mode == "hierarchical" and smoothing is None:
            raise ValueError("mode='hierarchical' requires a smoothing operator")
        if mode == "profiled" and smoothing_strength is None:
            raise ValueError(
                "mode='profiled' requires a fixed smoothing_strength (lambda)"
            )

        theta0 = np.asarray(theta0, dtype=float)
        self.mode = mode
        self.free = list(free)
        self.datasets = datasets
        self._theta0 = theta0
        self._free_idx = np.array(
            [_THETA_NAMES.index(name) for name in free], dtype=int
        )
        self._n_length = int(n_length)
        self._n_width = int(n_width)
        self._nu = float(nu)

        # Template system provides the stacked data, weights, and
        # regularization operator; its Green's matrix is not used.
        template = Fault.planar(
            lat=ref_lat,
            lon=ref_lon,
            depth=theta0[2],
            strike=theta0[3],
            dip=theta0[4],
            length=theta0[5],
            width=theta0[6],
            n_length=n_length,
            n_width=n_width,
        )
        sys = LinearSystem(template, datasets, smoothing, components)
        n_patches = n_length * n_width
        self._col_start, self._col_stop = {
            "both": (0, 2 * n_patches),
            "strike": (0, n_patches),
            "dip": (n_patches, 2 * n_patches),
        }[components]
        n_params = self._col_stop - self._col_start

        e_parts, n_parts = [], []
        for ds in datasets:
            e_ds, n_ds, _ = transforms.geod2enu(
                ds.lat, ds.lon, np.zeros(ds.n_stations), ref_lat, ref_lon, 0.0
            )
            e_parts.append(e_ds)
            n_parts.append(n_ds)
        self._e_obs = np.concatenate(e_parts)
        self._n_obs = np.concatenate(n_parts)

        W_half = scipy.linalg.cholesky(sys.W, lower=False)
        self._W_half_P = W_half @ _projection_matrix(datasets)
        self._d_w = W_half @ sys.d
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
            if smoothing is not None:
                self._LtL = sys.LtL
                eig = np.abs(np.linalg.eigvalsh(self._LtL))
            else:
                self._LtL = np.zeros((n_params, n_params))
                eig = np.zeros(n_params)
            if mode == "profiled":
                assert smoothing_strength is not None
                self._lambda_fixed = float(smoothing_strength)
            else:
                self._lambda_fixed = None
            pos = eig[eig > 0]
            self._logdet_rank = len(pos)
            self._logdet_sum = float(np.sum(np.log(pos)))
        self._include_logdet = mode != "profiled"

        # Sampled-parameter layout and priors
        self.param_names = list(free) + ["log10_sigma"]
        specs = [self._parse_prior(name, theta_prior[name]) for name in free]
        specs.append(("uniform",) + tuple(map(float, log10_sigma_prior)))
        x0 = list(theta0[self._free_idx]) + [
            float(np.clip(0.0, *log10_sigma_prior))
        ]
        if mode == "hierarchical":
            self.param_names.append("log10_lambda")
            specs.append(("uniform",) + tuple(map(float, log10_lambda_prior)))
            lam0 = (
                float(np.log10(smoothing_strength))
                if smoothing_strength
                else 0.5 * (log10_lambda_prior[0] + log10_lambda_prior[1])
            )
            x0.append(float(np.clip(lam0, *log10_lambda_prior)))
        self.x0 = np.array(x0)
        self.n_params = len(self.param_names)

        self._is_uniform = np.array([s[0] == "uniform" for s in specs])
        self._lo = np.array(
            [s[1] if s[0] == "uniform" else -np.inf for s in specs]
        )
        self._hi = np.array([s[2] if s[0] == "uniform" else np.inf for s in specs])
        self._mu = np.array([s[1] if s[0] == "normal" else 0.0 for s in specs])
        self._sd = np.array([s[2] if s[0] == "normal" else 1.0 for s in specs])

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

        Returns:
            Tuple ``(sigma2, lam, G_w, chol_H, m_hat, S)``: noise
            variance factor, prior strength, weighted Green's matrix,
            Cholesky factor of ``H = G_w^T G_w + lam LtL``, conditional
            slip mode, and the total misfit
            ``S = ||d_w - G_w m||^2 + lam ||L m||^2``.
        """
        import jax.numpy as jnp
        from jax.scipy.linalg import cho_solve

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
        LtL = jnp.asarray(self._LtL)
        d_w = jnp.asarray(self._d_w)

        H = G_w.T @ G_w + lam * LtL
        chol_H = jnp.linalg.cholesky(H)
        m_hat = cho_solve((chol_H, True), G_w.T @ d_w)
        r = d_w - G_w @ m_hat
        S = r @ r + lam * (m_hat @ LtL @ m_hat)
        return sigma2, lam, G_w, chol_H, m_hat, S

    def _misfit_total(self, x: npt.ArrayLike) -> np.ndarray:
        """Total misfit S = ||d_w - G_w m||^2 + lam ||L m||^2 at x."""
        return self._assemble(x)[5]

    def log_likelihood(self, x: npt.ArrayLike) -> np.ndarray:
        """Collapsed log-likelihood log p(d_w | x) (traceable).

        The exact Gaussian marginal over slip in the hierarchical and
        weak modes (up to the constant ``log|W|/2`` from weighting the
        data, and using the pseudo-determinant convention when the
        smoothing operator is rank-deficient); the profiled objective
        without the Occam log-determinant terms in profiled mode.

        Args:
            x: Sampled parameter vector, ordered as ``param_names``.

        Returns:
            Scalar log-likelihood.
        """
        import jax.numpy as jnp

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

        x = jnp.asarray(x)
        lo = jnp.asarray(self._lo)
        hi = jnp.asarray(self._hi)
        in_bounds = (x >= lo) & (x <= hi)
        lp_uniform = jnp.where(in_bounds, -jnp.log(hi - lo), -jnp.inf)
        z = (x - jnp.asarray(self._mu)) / jnp.asarray(self._sd)
        lp_normal = (
            -0.5 * z**2
            - jnp.log(jnp.asarray(self._sd))
            - 0.5 * jnp.log(2.0 * jnp.pi)
        )
        terms = jnp.where(jnp.asarray(self._is_uniform), lp_uniform, lp_normal)
        return cast(np.ndarray, jnp.sum(terms))

    def logpdf(self, x: npt.ArrayLike) -> np.ndarray:
        """Log-posterior density (traceable, differentiable).

        Equals ``log_prior(x) + log_likelihood(x)``; the likelihood is
        evaluated at bound-clipped parameters so its gradient stays
        finite at rejected points.

        Args:
            x: Sampled parameter vector, ordered as ``param_names``.

        Returns:
            Scalar log-posterior; ``-inf`` outside uniform prior bounds.
        """
        return self.log_prior(x) + self.log_likelihood(x)
