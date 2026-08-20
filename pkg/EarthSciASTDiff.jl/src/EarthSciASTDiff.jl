"""
    EarthSciASTDiff

Symbolic differentiation of EarthSciAST (`.esm`) models by AST substitution.

Computes the analytical Jacobian of an ESM model's ODE right-hand side with
respect to its state variables (`wrt = :states`) or its parameters
(`wrt = :parameters`), as a list of structured **bands** ([`Band`](@ref)):
each band is a rectangular region of the equation's index space, symbolic
column-index expressions, and a scalar ESM coefficient expression. The band
list is itself serializable as an ESM-document block (the draft `"jacobians"`
top-level block, see `esm-jacobian-spec.md` at the repository root), so any
language binding that can evaluate ESM expressions can evaluate the Jacobian.

Layered API, lowest to highest:

  - [`dscalar`](@ref)              — derivative of one scalar expression w.r.t. a site
  - [`simplify_branches`](@ref)    — `ifelse`-aware simplification (swell control);
                                     [`cse_templates`](@ref) (emitted block) and
                                     [`hoist_observed`](@ref) (evaluation model)
                                     are the two CSE halves
  - [`jacobian_bands`](@ref)       — all bands of a model / flattened system
  - [`jacobian_document`](@ref)    — original document + serialized `"jacobians"` block
  - [`jacobian_pattern`](@ref)     — structural sparsity pattern (`SparseMatrixCSC{Bool}`)
  - [`detect_structure`](@ref)     — `:block_diagonal` / `:banded` / `:general`
  - [`plan_factorization`](@ref)   — symbolic factorization info (general interface;
                                     per-cell block LU is the first implementation)
  - [`prepare_jacobian`](@ref)     — reusable evaluator: `jac!(J, u, p, t)`
  - [`assemble_jacobian`](@ref)    — one-shot: values + pattern at a point
  - [`odeproblem`](@ref)           — `SciMLBase.ODEProblem` with the analytical
                                     sparse Jacobian attached (needs SciMLBase
                                     loaded; see also [`ode_components`](@ref))

Differentiation runs on the *expanded, lowered* expression tree — the same
tree `EarthSciAST.build_evaluator` compiles — so the derivative and the RHS
can never come from different forms of the model. Sparsity is **structural**
(variable occurrence), never a numeric probe: `ifelse`/`min`/`max` branches
are unioned, so the pattern is point-independent and solver preparations can
be reused.
"""
module EarthSciASTDiff

using EarthSciAST
using EarthSciAST: ASTExpr, NumExpr, IntExpr, VarExpr, OpExpr, IndexSetRef,
                   canonical_json, simplify, map_children, parse_expression
using SparseArrays
using OrderedCollections: OrderedDict
using JSON3

export Site, dscalar, simplify_branches,
       Band, JacEntry, bands, jacobian_bands,
       SysView, sysview, expanded_model,
       jacobian_document, parse_jacobian_block, stable_json,
       cse_templates, expand_templates, hoist_observed,
       jacobian_pattern, assemble_jacobian, prepare_jacobian, JacobianEvaluator,
       ode_components, odefunction, odeproblem,
       detect_structure, jac_prototype,
       FactorizationPlan, BlockDiagonalPlan, plan_factorization, lu_fill

include("expr_helpers.jl")   # literal/constructor helpers, Site, occurrence
include("simplify_branches.jl") # ifelse-aware simplification (swell control)
include("cse.jl")            # hash-consed template extraction (swell control)
include("scalar_rules.jl")   # dscalar + closed-function derivative table
include("inline.jl")         # observed inlining, index-of-makearray lowering
include("bands.jl")          # Band + the array-level band calculus
include("clip.jl")           # static region clipping + band re-merging
include("system.jl")         # SysView over Model / FlattenedSystem, expansion
include("jacobian.jl")       # jacobian_bands driver
include("eval_cse.jl")       # evaluation-model CSE (hoisted observed buffers)
include("emit.jl")           # "jacobians" document block: emit / parse / goldens
include("assemble.jl")       # numeric assembly + prepared evaluator
include("structure.jl")      # structure detection + jac_prototype
include("factorization.jl")  # factorization-info interface + block-diagonal plan
include("ode.jl")            # solver-free ODE pieces + SciMLBase ext dispatchers

end # module
