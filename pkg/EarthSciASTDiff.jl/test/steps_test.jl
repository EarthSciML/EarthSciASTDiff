# Named steps (esm-jacobian-spec.md §4.6): bands composed through the
# observed graph. Structure-only passes compose; their points equal the
# inlining path's, and a chain with many paths stays linear.

using EarthSciASTDiff: serialize_band

# (row cell, column cell) of every band point
function _step_points(sv, entries)
    out = Set{Tuple{String, String}}()
    for en in entries
        EarthSciASTDiff._foreach_point(en.band) do cell, _, ci
            push!(out, (EarthSciASTDiff._cellname(en.u, cell), EarthSciASTDiff._cellname(en.v, collect(ci))))
        end
    end
    return out
end

@testset "named steps: a diamond chain composes, linear in its levels" begin
    file = fixture("diamond_chain.esm")
    sv = sysview(file, "Chain")
    entries = jacobian_bands(sv; coefficients = false)
    @test stable_json(Any[serialize_band(en) for en in entries]) ==
        read(joinpath(FIXTURES, "..", "..", "..", "..", "tests", "goldens", "diamond_chain.structure.bands.json"), String)
    # the inlining path (2^12 paths) gives the same points
    old = EarthSciASTDiff._extended_pass(
        sv, EarthSciASTDiff._ctx(sv), EarthSciASTDiff._observed_defs(sv), :states, nothing, String[];
        coefficients = false, steps = false
    )
    @test _step_points(sv, entries) == _step_points(sv, old.rows)
    # u reads a12 pointwise (diagonal in u and v); v reads it through a sum
    # over k as well (dense in u and v)
    pts = _step_points(sv, entries)
    @test ("u[1]", "v[2]") ∉ pts && ("u[1]", "v[1]") in pts && ("v[1]", "u[3]") in pts
    # every structure-only fixture: the composed points are the inlining path's
    for name in (STRUCTURE_FIXTURES..., "pbl_rank1", "tgrad_named_steps", "elementwise_aligned")
        src = EarthSciAST.load_path(joinpath(FIXTURES, "..", "..", "..", "..", "tests", "valid", "$name.esm"))
        v = name == "coupled_chem_advection" ? sysview(EarthSciAST.flatten(src)) : sysview(src, first(keys(src.models)))
        for wrt in (:states, :parameters, :forcing)
            args = (v, EarthSciASTDiff._ctx(v), EarthSciASTDiff._observed_defs(v), wrt, nothing, String[])
            o = try
                EarthSciASTDiff._extended_pass(args...; coefficients = false, steps = false)
            catch
                nothing
            end
            o === nothing && continue
            n = EarthSciASTDiff._extended_pass(args...; coefficients = false)
            @test _step_points(v, n.rows) == _step_points(v, o.rows)
        end
    end
end

@testset "named steps: time companions align elementwise operands by index set" begin
    # w = c·q + k·t + r over [x, y], q over [y], k over [x], r over [y, x]:
    # its companion `_dt_w` = k read at the x index, aligned by name (#84)
    d = EarthSciASTDiff.JSON3.read(read(joinpath(FIXTURES, "elementwise_aligned.esm"), String), Dict{String, Any})
    eq = only(e for e in d["models"]["Box"]["equations"] if e["lhs"] == "w")
    eq["rhs"]["args"][2] = Dict("op" => "*", "args" => Any["k", "t"])
    file = EarthSciAST.load_string(EarthSciASTDiff.JSON3.write(d))
    blk = jacobian_document(file, "Box"; wrt = :time)["jacobians"]["Box"]
    comps = blk["intermediates"]
    @test [c["name"] for c in comps] == ["_dt_w"]
    rhs = EarthSciASTDiff.expression_from_json(comps[1]["rhs"])
    @test EarthSciASTDiff.free_variables(rhs) ⊆ Set(["k", "i", "j", "_r1", "_r2"])
    @test "k" in EarthSciASTDiff.free_variables(rhs)
end
