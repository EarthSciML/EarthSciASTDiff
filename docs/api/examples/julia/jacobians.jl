# Differentiate a model, read its sparsity and structure, and evaluate the Jacobian.

using EarthSciAST
using EarthSciASTDiff
using SparseArrays: nnz

const FIXTURES = normpath(joinpath(@__DIR__, "..", "..", "..", "..", "tests", "valid"))

# region: bands ops=jacobian_bands,sysview
file = EarthSciAST.load_path(joinpath(FIXTURES, "bd_chem.esm"))
view = sysview(file, "Chem")
entries = jacobian_bands(view)
# One entry per (row variable, column variable) pair that occurs in the model.
@assert sort(unique([(en.u, en.v) for en in entries])) ==
    [("A", "A"), ("A", "B"), ("A", "C"), ("B", "A"), ("B", "B"), ("C", "B")]
# endregion

# region: pattern ops=jacobian_pattern
pat = jacobian_pattern(file)
# 4 cells x 6 variable pairs, one stored entry each, even where a value is 0.
@assert nnz(pat.pattern) == 24
@assert pat.pattern[1, 1]                    # A[1] depends on A[1]
@assert !pat.pattern[1, 2]                   # but not on A[2]
@assert pat.rownames[1] == "A[1]"
# endregion

# region: structure ops=detect_structure,plan_factorization
@assert detect_structure(entries, view) == :block_diagonal
plan = plan_factorization(entries, view)
@assert sort(plan.block_vars) == ["A", "B", "C"]
# endregion

# region: assemble ops=assemble_jacobian
res = assemble_jacobian(file)                # at the document's initial state
@assert res.structure == :block_diagonal
@assert res.J[1, 1] == -0.04                 # -k1
@assert res.J[5, 5] == -3.0                  # -2 k2 B, with B = 0.5 by default
# endregion

# region: prepare ops=prepare_jacobian,jac_prototype
jac = prepare_jacobian(file)                 # differentiate and compile once
J = jac_prototype(jac)                       # sparse matrix on the fixed pattern
u = fill(2.0, 12)
p = jac.p_default                            # the document's parameter values
jac(J, u, p, 0.0)                            # fill J in place, at any (u, p, t)
@assert size(J) == (12, 12)
@assert nnz(J) == 24
@assert J[1, 1] == -0.04
# endregion

# region: block ops=jacobian_document
doc = jacobian_document(file, "Chem")
block = doc["jacobians"]["Chem"]
@assert block["structure"] == "block_diagonal"
@assert block["factorization"]["type"] == "block_diagonal_lu"
# endregion
