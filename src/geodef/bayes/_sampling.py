"""NUTS sampling entry point and its result record.

Private submodule of :mod:`geodef.bayes`. ``sample`` wraps blackjax
window adaptation plus vmapped multi-chain NUTS; ``PosteriorResult``
carries draws and diagnostics.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from geodef import backend
from geodef.bayes._diagnostics import effective_sample_size, split_rhat
from geodef.bayes._util import _require_jax

if TYPE_CHECKING:
    from geodef.bayes import RectPosterior, SlipPosterior


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
