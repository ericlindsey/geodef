"""GeoDef: forward and inverse modeling of fault slip in elastic half-spaces."""

__version__ = "0.1.0"

from geodef import (
    backend,
    bayes,
    cache,
    data,
    euler,
    geomap,
    geometry,
    gradients,
    greens,
    invert,
    medium,
    mesh,
    okada,
    okada85,
    okada92,
    plot,
    slip,
    transforms,
    tri,
    validation,
)
from geodef.data import GNSS, DataSet, InSAR, Vertical

# Only beginner-public names (Fault, GNSS/InSAR/Vertical, DataSet, LocalFrame,
# ElasticMedium, DEFAULT_MEDIUM, solve, InversionResult) are imported at the top
# level and listed in ``__all__`` below. Expert-public names are reached through
# their module (``geodef.invert.lcurve``, ``geodef.greens.stack_obs``,
# ``geodef.data.spatial_covariance``, ``geodef.fault.moment_to_magnitude``, ...);
# the transitional top-level aliases were removed in the roadmap 2.2 export trim.
from geodef.fault import Fault
from geodef.geometry import LocalFrame
from geodef.invert import InversionResult, solve
from geodef.medium import DEFAULT_MEDIUM, ElasticMedium

# ----------------------------------------------------------------------
# Public API tiers
#
# ``__all__`` is the beginner-public vocabulary: the small, stable set a
# novice needs for the everyday forward-modeling and inversion path, plus
# the submodules that are the discovery surface for everything else.
#
# The expert-public API lives under those submodule paths -- e.g.
# ``geodef.invert.lcurve``, ``geodef.greens.matrix``, ``geodef.slip.pack``.
# The transitional top-level aliases for those names were removed in the
# roadmap 2.2 export trim; reach every expert name through its module.
#
# Private names are underscore-prefixed and are not part of any tier.
# ----------------------------------------------------------------------
__all__ = [
    # Submodules (discovery surface for the expert-public API)
    "backend",
    "bayes",
    "cache",
    "data",
    "euler",
    "geomap",
    "geometry",
    "gradients",
    "greens",
    "invert",
    "medium",
    "mesh",
    "okada",
    "okada85",
    "okada92",
    "plot",
    "slip",
    "transforms",
    "tri",
    "validation",
    # Beginner-public: domain objects
    "Fault",
    "GNSS",
    "InSAR",
    "Vertical",
    "DataSet",
    "LocalFrame",
    "ElasticMedium",
    "DEFAULT_MEDIUM",
    # Beginner-public: the one-shot inversion and its result record
    "solve",
    "InversionResult",
]
