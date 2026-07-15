"""Documentation consistency checks.

Guards against the drift called out in PLAN.md 0.3: public members that the
per-module reference never mentions, code examples with syntax errors, and
documented names that no longer exist in the package.
"""

import ast
import inspect
import re
from pathlib import Path

import pytest

import geodef

DOCS_DIR = Path(__file__).parent.parent / "docs"

# Modules with a docs/<name>.md reference page.
DOCUMENTED_MODULES = [
    "backend",
    "bayes",
    "cache",
    "data",
    "euler",
    "fault",
    "geomap",
    "geometry",
    "gradients",
    "greens",
    "invert",
    "medium",
    "mesh",
    "okada",
    "plot",
    "transforms",
    "validation",
]


def _public_members(module) -> list[str]:
    """Names of public functions/classes defined in (not imported into) module."""
    members = []
    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        members.append(name)
    return members


def _python_blocks(text: str) -> list[str]:
    """Extract ```python fenced code blocks from Markdown text."""
    return re.findall(r"```python\n(.*?)```", text, re.DOTALL)


@pytest.mark.parametrize("mod_name", DOCUMENTED_MODULES)
def test_docs_page_exists(mod_name: str) -> None:
    assert (DOCS_DIR / f"{mod_name}.md").exists()


@pytest.mark.parametrize("mod_name", DOCUMENTED_MODULES)
def test_docs_link_conventions(mod_name: str) -> None:
    """Every module reference page links the conventions page."""
    text = (DOCS_DIR / f"{mod_name}.md").read_text()
    assert "conventions.md" in text


@pytest.mark.parametrize("mod_name", DOCUMENTED_MODULES)
def test_public_members_documented(mod_name: str) -> None:
    """Every public function/class must be mentioned in its reference page."""
    module = getattr(geodef, mod_name)
    text = (DOCS_DIR / f"{mod_name}.md").read_text()
    missing = [name for name in _public_members(module) if name not in text]
    assert not missing, f"docs/{mod_name}.md does not mention public members: {missing}"


@pytest.mark.parametrize(
    "doc_path",
    sorted(DOCS_DIR.glob("*.md")) + [Path(__file__).parent.parent / "README.md"],
    ids=lambda p: p.name,
)
def test_doc_examples_are_valid_python(doc_path: Path) -> None:
    """Every fenced python example must at least parse."""
    for i, block in enumerate(_python_blocks(doc_path.read_text())):
        try:
            ast.parse(block)
        except SyntaxError as err:
            pytest.fail(
                f"{doc_path.name} python block {i + 1} has a syntax error: {err}"
            )


def test_documented_top_level_names_exist() -> None:
    """Names written as geodef.<name> in the docs must exist in the package."""
    referenced: set[str] = set()
    for doc_path in DOCS_DIR.glob("*.md"):
        referenced.update(
            re.findall(r"geodef\.([A-Za-z_][A-Za-z0-9_]*)", doc_path.read_text())
        )
    missing = sorted(name for name in referenced if not hasattr(geodef, name))
    assert not missing, f"docs reference nonexistent geodef.<name>: {missing}"
