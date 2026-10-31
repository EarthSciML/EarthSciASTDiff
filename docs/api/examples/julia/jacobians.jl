# Differentiate a model, read its sparsity and structure, and emit the Jacobian
# as documents EarthSciAST evaluates.

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

# region: evaluate ops=evaluation_document,jvp_document,jac_prototype
# EarthSciASTDiff evaluates nothing: it emits ESM documents that EarthSciAST
# evaluates. The band fields J_k, as a derived model:
doc, fields = evaluation_document(file)
@assert haskey(doc["models"], "JacobianEval")
@assert length(fields) == length(entries)
# J·v as a derived model: inputs `_v_<C>`, outputs `_jvp_<R>`.
jdoc, (model, v, jvp) = jvp_document(file)
@assert model == "JvpEval"
J = jac_prototype(file)                      # sparse matrix on the fixed pattern
@assert size(J) == (12, 12) && nnz(J) == 24
# endregion

# region: block ops=jacobian_document
doc = jacobian_document(file, "Chem")
block = doc["jacobians"]["Chem"]
@assert block["structure"] == "block_diagonal"
@assert block["factorization"]["type"] == "block_diagonal_lu"
# endregion
