"""Package layering, import-cycle, and base-install contracts (roadmap 3.1).

The declared layer table lives in ``plans/PHASE3_INTERNALS.md`` (and will
move to the module docs when Phase 3 ships): dependency direction points
down, module-level imports must never point up, and the two known upward
*deferred* (function-level) imports are pinned in an explicit allowlist so
new ones cannot appear silently. Lower layers may reference higher-layer
types under ``TYPE_CHECKING`` only.
"""

import ast
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent.parent / "src" / "geodef"

# Layer table: dependency direction points down (imports allowed only to
# the same or a lower layer at module level).
LAYERS = {
    # 1. Foundation
    "backend": 1,
    "validation": 1,
    "medium": 1,
    "transforms": 1,
    "cache": 1,
    # 2. Kernels
    "okada85": 2,
    "okada92": 2,
    "tri": 2,
    "okada": 2,
    # 3. Operators
    "greens": 3,
    "_engines": 3,
    "geometry": 3,
    "slip": 3,
    "gradients": 3,
    # 4. Domain
    "fault": 4,
    "_fault_io": 4,
    "data": 4,
    "mesh": 4,
    "euler": 4,
    # 5. Workflows
    "invert": 5,
    "bayes": 5,
    # 6. Edges
    "plot": 6,
    "geomap": 6,
}

# Known upward function-level imports. Deferred imports run at call time,
# not import time, so they cannot create import cycles; each entry here is
# a deliberate, documented exception, not a precedent.
ALLOWED_DEFERRED_UP = {
    # stack_obs/stack_weights/project accept DataSet instances.
    ("greens", "data"),
    # plate_rake_from_euler evaluates an Euler pole per patch.
    ("slip", "euler"),
}


def _top_module(path: Path) -> str:
    """Map a source file to its top-level geodef module name."""
    rel = path.relative_to(SRC)
    return rel.parts[0].removesuffix(".py")


def _type_checking_nodes(tree: ast.AST) -> set[int]:
    hidden: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            name = getattr(test, "id", getattr(test, "attr", None))
            if name == "TYPE_CHECKING":
                for sub in ast.walk(node):
                    hidden.add(id(sub))
    return hidden


def _targets(node: ast.stmt) -> set[str]:
    """Top-level geodef modules imported by one import statement."""
    out: set[str] = set()
    if isinstance(node, ast.Import):
        for alias in node.names:
            parts = alias.name.split(".")
            if parts[0] == "geodef":
                out.add(parts[1] if len(parts) > 1 else "__init__")
    elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
        parts = node.module.split(".")
        if parts[0] == "geodef":
            if len(parts) > 1:
                out.add(parts[1])
            else:
                # ``from geodef import X`` -- X must be submodules, which
                # test_from_geodef_imports_only_submodules enforces.
                out.update(alias.name for alias in node.names)
    return out


def _import_edges() -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Collect (source, target) runtime import edges across the package.

    Returns ``(module_level, deferred)`` edge sets, skipping
    ``TYPE_CHECKING`` blocks, intra-package imports, and ``__init__``.
    """
    module_level: set[tuple[str, str]] = set()
    deferred: set[tuple[str, str]] = set()
    for path in sorted(SRC.rglob("*.py")):
        source = _top_module(path)
        if source == "__init__":
            continue
        tree = ast.parse(path.read_text())
        hidden = _type_checking_nodes(tree)

        def visit(node: ast.AST, in_func: bool, source: str = source) -> None:
            for child in ast.iter_child_nodes(node):
                inner = in_func or isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef)
                )
                if (
                    isinstance(child, (ast.Import, ast.ImportFrom))
                    and id(child) not in hidden
                ):
                    for target in _targets(child):
                        if target != source:
                            edge = (source, target)
                            (deferred if in_func else module_level).add(edge)
                visit(child, inner, source)

        visit(tree, False)
    return module_level, deferred


MODULE_LEVEL_EDGES, DEFERRED_EDGES = _import_edges()


def test_every_module_has_a_layer():
    modules = {_top_module(p) for p in SRC.rglob("*.py")} - {"__init__"}
    unassigned = {m for m in modules if not m.startswith("_")} - set(LAYERS)
    assert not unassigned, f"modules missing from the layer table: {sorted(unassigned)}"


def test_no_runtime_import_of_package_init():
    """No internal module may ``import geodef`` at runtime."""
    offenders = {
        edge for edge in MODULE_LEVEL_EDGES | DEFERRED_EDGES if edge[1] == "__init__"
    }
    assert not offenders, sorted(offenders)


def test_from_geodef_imports_only_submodules():
    """``from geodef import X`` must name submodules, not re-exports."""
    submodules = {_top_module(p) for p in SRC.rglob("*.py")} - {"__init__"}
    offenders = {
        edge
        for edge in MODULE_LEVEL_EDGES | DEFERRED_EDGES
        if edge[1] not in submodules and edge[1] != "__init__"
    }
    assert not offenders, sorted(offenders)


def test_module_level_imports_point_down():
    offenders = {
        (src, dst) for src, dst in MODULE_LEVEL_EDGES if LAYERS[dst] > LAYERS[src]
    }
    assert not offenders, (
        f"module-level imports pointing up the layer table: {sorted(offenders)}"
    )


def test_deferred_upward_imports_are_pinned():
    upward = {(src, dst) for src, dst in DEFERRED_EDGES if LAYERS[dst] > LAYERS[src]}
    new = upward - ALLOWED_DEFERRED_UP
    stale = ALLOWED_DEFERRED_UP - upward
    assert not new, (
        f"new upward deferred imports (add a reviewed exception): {sorted(new)}"
    )
    assert not stale, f"stale allowlist entries: {sorted(stale)}"


def test_module_level_graph_is_acyclic():
    graph: dict[str, set[str]] = {}
    for src, dst in MODULE_LEVEL_EDGES:
        graph.setdefault(src, set()).add(dst)

    visiting: set[str] = set()
    done: set[str] = set()

    def dfs(node: str, stack: list[str]) -> None:
        visiting.add(node)
        for nxt in graph.get(node, ()):
            if nxt in done:
                continue
            assert nxt not in visiting, f"import cycle: {stack + [node, nxt]}"
            dfs(nxt, stack + [node])
        visiting.discard(node)
        done.add(node)

    for node in list(graph):
        if node not in done:
            dfs(node, [])


def test_base_import_pulls_no_optional_stacks():
    """``import geodef`` must not initialize JAX or other optional stacks.

    Run in a subprocess so this session's imports cannot mask a
    regression. Meaningful when the optional stacks are installed (CI's
    full tier); trivially green when they are absent.
    """
    banned = ["jax", "blackjax", "cartopy", "meshpy", "pyproj", "netCDF4", "pandas"]
    code = (
        "import sys\n"
        "import geodef\n"
        f"loaded = [m for m in {banned!r} if m in sys.modules]\n"
        "assert not loaded, f'import geodef pulled optional stacks: {loaded}'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
