"""Differentiate a model, read its sparsity and structure, and evaluate the Jacobian."""

import json
from pathlib import Path

import earthsci_ast
import earthsci_astdiff

FIXTURES = Path(__file__).resolve().parents[4] / "tests" / "valid"


# region: bands ops=jacobian_bands,sysview
file = earthsci_ast.load_path(FIXTURES / "bd_chem.esm")
entries = earthsci_astdiff.jacobian_bands(file)
# One entry per (row variable, column variable) pair that occurs in the model.
assert sorted({(en.u, en.v) for en in entries}) == [
    ("A", "A"), ("A", "B"), ("A", "C"), ("B", "A"), ("B", "B"), ("C", "B"),
]
view = earthsci_astdiff.sysview(file)
# endregion


# region: pattern ops=jacobian_pattern
pairs, _ = earthsci_astdiff.jacobian_pattern(file)
# 4 cells x 6 variable pairs, one stored entry each, even where a value is 0.
assert len(pairs) == 24
assert ("A[1]", "B[1]") in pairs
assert ("A[1]", "A[2]") not in pairs
# endregion


# region: structure ops=detect_structure,plan_factorization
assert earthsci_astdiff.detect_structure(entries, view) == "block_diagonal"
plan = earthsci_astdiff.plan_factorization(entries, view)
assert sorted(plan.block_vars) == ["A", "B", "C"]
# endregion


# region: assemble ops=assemble_jacobian
values = earthsci_astdiff.assemble_jacobian(file, states={"A": [2.0] * 4})
assert values[("A[1]", "A[1]")] == -0.04  # -k1
assert values[("B[1]", "B[1]")] == -3.0   # -2 k2 B, with B = 0.5 by default
# endregion


# region: block ops=jacobian_block
block = earthsci_astdiff.jacobian_block(file)
assert block["structure"] == "block_diagonal"
assert block["factorization"]["type"] == "block_diagonal_lu"
json.dumps(block)  # a plain, serializable dict
# endregion
