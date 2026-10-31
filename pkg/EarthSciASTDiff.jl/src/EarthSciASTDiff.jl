"""
    EarthSciASTDiff

Symbolic differentiation of EarthSciAST (`.esm`) models by AST substitution.

Computes the analytical Jacobian of an ESM model's ODE right-hand side with
respect to its state variables (`wrt = :states`), its parameters
(`wrt = :parameters`, array-valued ones cell by cell), its forcing buffers
(`wrt = :forcing`, the cells of the live data buffers, record axis included)
or time (`wrt = :time`), as a list of structured **bands** ([`Band`](@ref)):
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
                                     (`:column_block` given a `column_axis`)
  - [`plan_factorization`](@ref)   — symbolic factorization info (general interface;
                                     per-cell block LU is the first implementation)
  - [`prepare_jacobian`](@ref)     — reusable evaluator: `jac!(J, u, p, t)`;
                                     [`vjp!`](@ref) / [`vjp_forcing!`](@ref) apply
                                     the transposed bands without forming `J`
  - [`jacobian_bands2`](@ref)      — second-order bands for an axis pair;
                                     [`prepare_hvp`](@ref) + [`hvp!`](@ref) contract
                                     them (Rosenbrock adjoints) without a tensor
  - [`scatter_plan`](@ref)         — checked band-field → storage map (lane blocks,
                                     sparse lane blocks, column bands, CSR);
                                     [`bind_plan`](@ref) + [`jac_into!`](@ref) fill it
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
    canonical_json, simplify, map_children, expression_from_json
using SparseArrays
using OrderedCollections: OrderedDict
using JSON3

export Site, dscalar, simplify_branches,
    Band, JacEntry, bands, jacobian_bands, forcing_names, jacobian_columns,
    SysView, sysview, expanded_model, inlinable_observed,
    jacobian_document, parse_jacobian_block, parse_jacobian_blocks, stable_json,
    cse_templates, expand_templates, hoist_observed,
    jacobian_pattern, assemble_jacobian, prepare_jacobian, JacobianEvaluator,
    vjp!, vjp_forcing!,
    JacEntry2, jacobian_bands2, col_idx, col2_idx, prepare_hvp, HVPEvaluator, hvp!,
    ode_components, odefunction, odeproblem,
    detect_structure, column_lines, jac_prototype,
    FactorizationPlan, BlockDiagonalPlan, ColumnBlockPlan, plan_factorization,
    serialize_plan, lu_fill,
    StorageLayout, LaneBlocks, LaneSparseBlocks, ColumnBanded, Csr,
    ScatterPlan, scatter_plan, serialize_scatter_plan,
    BoundScatter, bind_plan, jac_into!, scatter_targets

include("expr_helpers.jl")   # literal/constructor helpers, Site, occurrence
include("hashcons.jl")       # structural interning (size-independent equality)
include("simplify_branches.jl") # ifelse-aware simplification (swell control)
include("cse.jl")            # hash-consed template extraction (swell control)
include("scalar_rules.jl")   # dscalar + closed-function derivative table
include("inline.jl")         # observed inlining, index-of-makearray lowering
include("bands.jl")          # Band + the array-level band calculus
include("reductions.jl")     # filters, max/min/* reductions, one-hot index reads
include("clip.jl")           # static region clipping + band re-merging
include("system.jl")         # SysView over Model / FlattenedSystem, expansion
include("jacobian.jl")       # jacobian_bands driver
include("eval_cse.jl")       # evaluation-model CSE (hoisted observed buffers)
include("emit.jl")           # "jacobians" document block: emit / parse / goldens
include("assemble.jl")       # numeric assembly + prepared evaluator
include("structure.jl")      # structure detection + jac_prototype
include("factorization.jl")  # factorization-info interface + block-diagonal plan
include("second_order.jl")   # second-order bands + the hvp! contraction (§4.4)
include("scatter.jl")        # scatter plans: band fields → consumer storage layouts
include("ode.jl")            # solver-free ODE pieces + SciMLBase ext dispatchers

end # module
