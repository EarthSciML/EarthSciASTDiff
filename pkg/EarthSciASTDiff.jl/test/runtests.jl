using Test
using EarthSciASTDiff
using EarthSciAST
using EarthSciAST: ASTExpr, NumExpr, IntExpr, VarExpr, OpExpr
using ForwardDiff
using SparseArrays
using LinearAlgebra

# conformance.yml pins one EarthSciAST release per job and sets this; the
# test sandbox Pkg.test builds is a separate resolve, so check it kept the pin.
if !isempty(get(ENV, "EARTHSCI_AST_VERSION", ""))
    want = VersionNumber(ENV["EARTHSCI_AST_VERSION"])
    got = pkgversion(EarthSciAST)
    got == want || error("the tests loaded EarthSciAST $got, but CI pinned $want")
end

const FIXTURES = joinpath(@__DIR__, "fixtures")
fixture(name) = EarthSciAST.load_path(joinpath(FIXTURES, name))

include("docvals.jl")

# The native compiler declines a model whose coefficients carry a filtered or
# non-`+` faq node (`compiler_refused_rule`); the interpreter evaluates them.
_hkw(name) = name in ("filtered_scan", "semiring_reduce") ?
    (compiler = :interpreter,) : NamedTuple()

# ForwardDiff oracle: dense Jacobian of the tree-walk RHS w.r.t. states.
function ad_jacobian(f!, u, p, t)
    du = similar(u)
    return sparse(ForwardDiff.jacobian((du, u) -> f!(du, u, p, t), du, u))
end

@testset "EarthSciASTDiff" begin
    include("faq_tag_test.jl")
    include("scalar_rules_test.jl")
    include("cse_test.jl")
    include("clip_test.jl")
    include("inline_test.jl")
    include("eval_cse_test.jl")
    include("evaldoc_test.jl")
    include("contracted_test.jl")
    include("oracle_test.jl")
    include("prune_test.jl")
    include("observed_test.jl")
    include("contracting_observed_test.jl")
    include("reductions_test.jl")
    include("roundtrip_test.jl")
    include("time_test.jl")
    include("forcing_test.jl")
    include("second_order_test.jl")
    include("products_test.jl")
    include("structure_test.jl")
    include("factorization_test.jl")
    include("scatter_test.jl")
    include("source_layout_test.jl")
end
