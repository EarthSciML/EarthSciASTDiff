"""Test setup: make `earthsci_ast` importable from a sibling EarthSciAST
checkout when it is not installed (the cross-repo conformance layout), and
locate the shared fixture/golden directories."""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_SRC = HERE.parent / "src"
REPO_ROOT = HERE.parent.parent.parent  # EarthSciASTDiff/

sys.path.insert(0, str(PKG_SRC))

try:
    import earthsci_ast
except ImportError:  # sibling checkout fallback
    cand = os.environ.get("EARTHSCI_AST_PY")
    if cand is None:
        cand = str(REPO_ROOT.parent / "EarthSciAST" / "pkg" / "earthsci-ast-py" / "src")
    sys.path.insert(0, cand)
    import earthsci_ast

# conformance.yml pins one earthsci-ast release per job and sets this; check
# the tests import that release (not, say, a sibling checkout).
_want = os.environ.get("EARTHSCI_AST_VERSION")
if _want:
    from importlib.metadata import version

    _got = version("earthsci-ast")
    if _got != _want:
        raise RuntimeError(f"the tests import earthsci-ast {_got}, but CI pinned {_want}")

import pytest  # noqa: E402  (after the sys.path setup above)


@pytest.fixture(scope="session")
def valid_dir() -> Path:
    return REPO_ROOT / "tests" / "valid"


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    return REPO_ROOT / "tests" / "goldens"


# Operator references (esm-spec §9.6) are kept, never expanded, in the
# reference's goldens: a coefficient reads a definition through
# `apply_operator`, and a stencil's bands are composed at each use. The Python
# binding differentiates the tree `earthsci_ast` loads, and every Python
# EarthSciAST release so far (0.15 included) expands references at load: its
# `ExprNode` has no `bindings`. Until one keeps them (as the Julia and Rust
# loaders do from 0.15), the goldens that depend on kept references cannot be
# matched, and their tests skip. The check is the loader's behavior, not a
# version, so the tests run again as soon as a release keeps references.

# Goldens without an `apply_operator` node that still depend on kept
# references: the structure-only bands of a document whose stencil bands are
# composed at their uses (their boxes split where the definition's do).
_REF_STRUCTURE_GOLDENS = frozenset({"coupled_chem_advection.structure.bands.json"})


@functools.cache
def loader_keeps_operator_refs() -> bool:
    """Whether the installed ``earthsci_ast`` keeps an operator reference in a
    loaded model's equations (two_shape_stencil's ``sq`` uses)."""
    from earthsci_ast.esm_types import ExprNode

    file = earthsci_ast.load_path(str(REPO_ROOT / "tests" / "valid" / "two_shape_stencil.esm"))

    def has_ref(e) -> bool:
        if not isinstance(e, ExprNode):
            return False
        if e.op in ("apply_operator", "apply_expression_template"):
            return True
        kids = [*(e.args or []), *(e.values or []), e.expr, e.filter]
        return any(has_ref(c) for c in kids)

    return any(has_ref(eq.rhs) for m in file.models.values() for eq in m.equations)


def ref_golden(path: Path) -> str:
    """The text of golden ``path``; the test skips when the golden depends on
    kept operator references and the installed loader expands them."""
    text = path.read_text()
    needs = "apply_operator" in text or path.name in _REF_STRUCTURE_GOLDENS
    if needs and not loader_keeps_operator_refs():
        pytest.skip(
            f"{path.name} depends on kept operator references, which this "
            "earthsci_ast's loader expands (no Python EarthSciAST release keeps them yet)"
        )
    return text
