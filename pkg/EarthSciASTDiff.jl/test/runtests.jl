using Test
using EarthSciASTDiff
using EarthSciAST
using EarthSciAST: ASTExpr, NumExpr, IntExpr, VarExpr, OpExpr
using ForwardDiff
using SparseArrays
using LinearAlgebra

const FIXTURES = joinpath(@__DIR__, "fixtures")
fixture(name) = EarthSciAST.load_path(joinpath(FIXTURES, name))

# ForwardDiff oracle: dense Jacobian of the tree-walk RHS w.r.t. states.
function ad_jacobian(f!, u, p, t)
    du = similar(u)
    sparse(ForwardDiff.jacobian((du, u) -> f!(du, u, p, t), du, u))
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
    include("roundtrip_test.jl")
    include("time_test.jl")
    include("structure_test.jl")
    include("factorization_test.jl")
    include("scatter_test.jl")
    include("ode_test.jl")
    include("source_layout_test.jl")
end
