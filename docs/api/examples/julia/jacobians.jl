# Differentiate a model, read its sparsity and structure, and emit the documents that evaluate the Jacobian.

using EarthSciAST
using EarthSciASTDiff
using JSON3
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

# region: assemble ops=evaluation_document
# The Julia package computes no values: it emits a document whose right-hand
# side, run once by an unmodified EarthSciAST evaluator, leaves every band
# value in `du`. `fields[k]` names entry k's band field.
doc, fields = evaluation_document(file)
prob = EarthSciAST.esm_problem(
    JSON3.read(stable_json(doc)), (0.0, 1.0)
)
du = zero(prob.u0)
prob.f!(du, prob.u0, prob.p, 0.0)            # at the document's initial state
k = findfirst(en -> (en.u, en.v) == ("A", "A"), entries)
@assert du[prob.var_map["JacobianEval.$(fields[k])[1]"]] == -0.04   # -k1
# endregion

# region: prepare ops=jac_prototype,scatter_plan,bind_plan
J = jac_prototype(file)                      # sparse matrix on the fixed pattern
@assert size(J) == (12, 12)
@assert nnz(J) == 24
# A scatter plan places the band fields in your own storage (here CSR),
# checked against the source before it is used.
bound = bind_plan(file, scatter_plan(file, Csr()))
@assert bound.len == 24
@assert all(1 <= k <= length(fields) for (k, _, _) in bound.pairs)
# endregion

# region: block ops=jacobian_document
doc = jacobian_document(file, "Chem")
block = doc["jacobians"]["Chem"]
@assert block["structure"] == "block_diagonal"
@assert block["factorization"]["type"] == "block_diagonal_lu"
# endregion
