# Structure-only entries (esm-jacobian-spec.md §2.1): no coefficient, the
# same clipping and sites, merged on structure. Their points include the full
# mode's, and equal them unless a full band's coefficient simplifies to zero.

using EarthSciASTDiff: _scatter_pairs, serialize_band

const _SO_GOLDENS = joinpath(@__DIR__, "..", "..", "..", "tests", "goldens")

# The fixtures whose state structure-only goldens this reference writes
# (scripts/regenerate-goldens.jl): every model fixture with a state band
# golden, and the flattened coupled document.
const STRUCTURE_FIXTURES = (
    "bd_chem", "adv_interior", "contracted_ops", "flux_form_adv", "minmod_adv",
    "contracting_observed", "tgrad_photolysis", "forcing_bracket",
    "filtered_scan", "semiring_reduce", "coupled_chem_advection",
)

_so_view(name) = name == "coupled_chem_advection" ?
    sysview(EarthSciAST.flatten(fixture("$name.esm"))) :
    (f = fixture("$name.esm"); sysview(f, first(keys(f.models))))

@testset "structure-only entries: goldens and points" begin
    for name in STRUCTURE_FIXTURES
        sv = _so_view(name)
        full = jacobian_bands(sv)
        so = jacobian_bands(sv; coefficients = false)
        @test all(en -> en.band.coef === nothing, so)
        @test all(en -> !haskey(serialize_band(en), "coef"), so)
        pf = Set(_scatter_pairs(sv, full)); ps = Set(_scatter_pairs(sv, so))
        @test issubset(pf, ps)
        # no band of these fixtures has a coefficient that simplifies to zero
        @test pf == ps
        # merging on structure never yields more bands than merging on values
        @test length(so) <= length(full)
        text = stable_json(Any[serialize_band(en) for en in so])
        @test text == read(joinpath(_SO_GOLDENS, "$name.structure.bands.json"), String)
    end
end

@testset "structure-only entries: other axes" begin
    for (name, wrt) in (
            ("tgrad_photolysis", :time), ("forcing_bracket", :forcing),
            ("forcing_bracket", :parameters),
        )
        sv = _so_view(name)
        full = jacobian_bands(sv; wrt = wrt)
        so = jacobian_bands(sv; wrt = wrt, coefficients = false)
        @test issubset(Set(_scatter_pairs(sv, full)), Set(_scatter_pairs(sv, so)))
    end
end

@testset "structure-only entries keep a band whose coefficient is zero" begin
    # d/dx[i] of x[i] - x[i] simplifies to 0: the full mode drops the band,
    # the structure-only mode keeps it (a strict superset).
    ctx = EarthSciASTDiff.Ctx(Dict("i" => 4), Dict("x" => [4]), Set{String}())
    xi = OpExpr("index", ASTExpr[VarExpr("x"), VarExpr("i")])
    rhs = OpExpr(
        "faq", ASTExpr[]; output_idx = Any["i"],
        expr_body = OpExpr("-", ASTExpr[xi, xi]),
        ranges = Dict{String, Any}("i" => Any[1, 4])
    )
    @test isempty(bands(rhs, "x", ctx; shape_u = [4]))
    so = bands(rhs, "x", ctx; shape_u = [4], coefficients = false)
    @test length(so) == 1 && so[1].coef === nothing && so[1].rows == [(1, 4)]
end

@testset "structure-only pattern" begin
    file = fixture("adv_interior.esm")
    p = jacobian_pattern(file)
    q = jacobian_pattern(file; coefficients = false)
    @test q.pattern == p.pattern
    @test all(en -> en.band.coef === nothing, q.entries)
end
