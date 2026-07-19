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

from geodef.bayes._collapsed import (
    RectPosterior as RectPosterior,
)
from geodef.bayes._diagnostics import (
    effective_sample_size as effective_sample_size,
)
from geodef.bayes._diagnostics import (
    split_rhat as split_rhat,
)
from geodef.bayes._sampling import (
    PosteriorResult as PosteriorResult,
)
from geodef.bayes._sampling import (
    sample as sample,
)
from geodef.bayes._slip import (
    SlipPosterior as SlipPosterior,
)
from geodef.bayes._triwarp import (
    TriPosterior as TriPosterior,
)
from geodef.bayes._triwarp import (
    TriWarp as TriWarp,
)
