"""Test setup: make `earthsci_ast` importable from a sibling EarthSciAST
checkout when it is not installed (the cross-repo conformance layout), and
locate the shared fixture/golden directories."""

from __future__ import annotations

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
    import earthsci_ast  # noqa: F401

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
