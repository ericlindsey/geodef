"""Contract for the public API tiers (roadmap 1.6, 2.2, and 3.1).

``geodef.__all__`` is the beginner-public vocabulary. Expert-public names
live under their submodule path only; the roadmap 2.2 export trim removed
the transitional top-level aliases, so those names are no longer reachable
as ``geodef.<name>``.

``docs/api_stability.md`` is the published stability map; the tests below
parse it and fail when the map and the code disagree, in either direction.
"""

import inspect
import json
import re
from pathlib import Path

import pytest

import geodef

STABILITY_MAP = Path(__file__).parent.parent / "docs" / "api_stability.md"

# Reference ports whose interior names are public-spelled but private-tier;
# only their entry points are listed in the stability map.
KERNEL_MODULES = frozenset({"okada85", "okada92", "tri"})

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

# Expert-public names that live under their module path only. The roadmap 2.2
# export trim removed their transitional top-level aliases, so they must be kept
# out of ``__all__`` and must no longer be reachable as ``geodef.<name>``.
EXPERT_MODULE_ONLY_NAMES = frozenset(
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
    assert EXPERT_MODULE_ONLY_NAMES.isdisjoint(geodef.__all__)


def test_expert_names_not_reachable_at_top_level():
    """The 2.2 export trim removed the transitional top-level aliases."""
    for name in EXPERT_MODULE_ONLY_NAMES:
        assert not hasattr(geodef, name), (
            f"geodef.{name} is still top-level; reach it through its module"
        )


def test_all_entries_are_resolvable_attributes():
    for name in geodef.__all__:
        assert hasattr(geodef, name), name


# ----------------------------------------------------------------------
# Stability-map consistency (docs/api_stability.md, roadmap 3.1)
# ----------------------------------------------------------------------


def _map_text() -> str:
    return STABILITY_MAP.read_text()


def _section(text: str, heading: str) -> str:
    """Return the map text from ``heading`` to the next same-level heading."""
    depth = len(heading.split(" ")[0])  # 2 for "##", 3 for "###"
    start = text.index(f"{heading}\n")
    rest = text[start + len(heading) :]
    nxt = re.search(rf"^#{{1,{depth}}} ", rest, re.MULTILINE)
    return rest[: nxt.start()] if nxt else rest


def _table_names(section: str) -> list[str]:
    """First-column backticked names of every table row in ``section``."""
    return re.findall(r"^\| `([\w.]+)` \|", section, re.MULTILINE)


def _module_sections(text: str) -> dict[str, list[str]]:
    """Map module name -> listed names for every ``### `geodef.X``` section."""
    sections: dict[str, list[str]] = {}
    for match in re.finditer(r"^### `geodef\.(\w+)`$", text, re.MULTILINE):
        sections[match.group(1)] = _table_names(
            _section(text, f"### `geodef.{match.group(1)}`")
        )
    return sections


def _public_members(module) -> set[str]:
    """Public functions/classes defined in (not imported into) module."""
    prefix = module.__name__ + "."
    members = set()
    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        defined_in = getattr(obj, "__module__", "") or ""
        if defined_in != module.__name__ and not defined_in.startswith(prefix):
            continue
        members.add(name)
    return members


def test_map_exists():
    assert STABILITY_MAP.exists()


def test_map_beginner_table_matches_all():
    """The beginner table plus the submodule list is exactly ``__all__``."""
    section = _section(_map_text(), "## Beginner-public tier (top level)")
    table = set(_table_names(section))
    submodule_para = section[section.index("The submodules") :]
    submodules = set(re.findall(r"`(\w+)`", submodule_para))
    all_names = set(geodef.__all__)
    modules_in_all = {n for n in all_names if inspect.ismodule(getattr(geodef, n))}
    assert table == all_names - modules_in_all
    assert submodules == modules_in_all


MODULE_SECTIONS = _module_sections(_map_text()) if STABILITY_MAP.exists() else {}


@pytest.mark.parametrize("mod_name", sorted(MODULE_SECTIONS))
def test_map_names_exist(mod_name: str):
    """Every name the map lists for a module is a real attribute."""
    module = getattr(geodef, mod_name)
    for name in MODULE_SECTIONS[mod_name]:
        assert hasattr(module, name), f"geodef.{mod_name}.{name} listed but missing"


@pytest.mark.parametrize("mod_name", sorted(set(MODULE_SECTIONS) - KERNEL_MODULES))
def test_map_is_complete(mod_name: str):
    """Every public function/class defined in a module is in the map."""
    module = getattr(geodef, mod_name)
    missing = _public_members(module) - set(MODULE_SECTIONS[mod_name])
    assert not missing, (
        f"public names in geodef.{mod_name} missing from docs/api_stability.md: "
        f"{sorted(missing)}"
    )


def test_map_covers_every_submodule():
    """Every non-kernel discovery-surface submodule has a map section."""
    modules_in_all = {n for n in geodef.__all__ if inspect.ismodule(getattr(geodef, n))}
    expected = modules_in_all - KERNEL_MODULES
    missing = expected - set(MODULE_SECTIONS)
    assert not missing, f"modules without a stability-map section: {sorted(missing)}"


def test_map_kernel_entry_points_exist():
    """The kernel entry-point table names resolve, module-qualified."""
    section = _section(_map_text(), "## Kernel modules and the reference-port interior")
    entries = _table_names(section)
    assert entries, "kernel entry-point table not found"
    for entry in entries:
        mod_name, name = entry.split(".")
        assert mod_name in KERNEL_MODULES
        assert hasattr(getattr(geodef, mod_name), name), entry


def test_surface_snapshot_matches():
    """Transient guard for the 3.2 module splits (remove after Phase 3).

    ``tests/reference_data/public_surface.json`` records the public
    functions/classes defined in the modules being split; each extraction
    commit must leave the surface exactly unchanged.
    """
    snapshot_path = Path(__file__).parent / "reference_data" / "public_surface.json"
    snapshot = json.loads(snapshot_path.read_text())
    for mod_name, expected in snapshot.items():
        module = getattr(geodef, mod_name)
        current = {}
        prefix = module.__name__ + "."
        for name, obj in vars(module).items():
            if name.startswith("_"):
                continue
            if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            defined_in = getattr(obj, "__module__", "") or ""
            if defined_in != module.__name__ and not defined_in.startswith(prefix):
                continue
            current[name] = "class" if inspect.isclass(obj) else "function"
        assert current == expected, f"geodef.{mod_name} surface changed"
