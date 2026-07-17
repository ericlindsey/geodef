"""Contract for the top-level public API tiers (roadmap 1.6).

``geodef.__all__`` is the beginner-public vocabulary. Expert-public names
live under their submodule path and, while several are still re-exported at
the top level for backward compatibility, they are deliberately excluded
from ``__all__``.
"""

import geodef

# The beginner-public vocabulary: domain objects, the one-shot solve, its
# result record, and the submodules that are the discovery surface.
BEGINNER_NAMES = frozenset(
    {
        "Fault",
        "GNSS",
        "InSAR",
        "Vertical",
        "DataSet",
        "LocalFrame",
        "ElasticMedium",
        "DEFAULT_MEDIUM",
        "solve",
        "InversionResult",
    }
)

# Expert-public names still re-exported at the top level, but reached through
# their module in new code and kept out of ``__all__``.
EXPERT_TOP_LEVEL_NAMES = frozenset(
    {
        "LinearSystem",
        "lcurve",
        "abic_curve",
        "compute_abic",
        "geometry_search",
        "model_covariance",
        "model_resolution",
        "model_uncertainty",
        "ABICCurveResult",
        "LCurveResult",
        "GeometrySearchResult",
        "DatasetDiagnostics",
        "select_slip_columns",
        "stack_obs",
        "stack_weights",
        "spatial_covariance",
        "magnitude_to_moment",
        "moment_to_magnitude",
    }
)


def test_beginner_names_are_public():
    assert BEGINNER_NAMES <= set(geodef.__all__)


def test_expert_names_excluded_from_all():
    assert EXPERT_TOP_LEVEL_NAMES.isdisjoint(geodef.__all__)


def test_expert_names_remain_importable_during_transition():
    for name in EXPERT_TOP_LEVEL_NAMES:
        assert hasattr(geodef, name), name


def test_all_entries_are_resolvable_attributes():
    for name in geodef.__all__:
        assert hasattr(geodef, name), name
