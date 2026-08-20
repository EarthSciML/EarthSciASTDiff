# Inliner coverage for `index(makearray …)` value shapes (inline.jl
# `_index_makearray`).
#
# A makearray region's VALUE may be a full-rank aggregate — one naming every
# region dim, singleton dims included — or a reduced-rank one naming only the
# non-singleton dims. `_bands_makearray!` has always distinguished the two
# (`val_fullrank`); the inliner accepted only the reduced-rank shape, so a
# full-rank value over a region with a PINNED dim (a 3-D field's boundary face,
# and the shape ReSEACT's coupled document trips on) was rejected outright with
# "makearray value rank 3 ≠ region free rank 2".
#
# `F` here is deliberately STATE-dependent, so inline pruning cannot make the
# test vacuous by declining to inline it.

using JSON3

@testset "inline: full-rank aggregate value over a pinned makearray region" begin
    O(kv...) = Dict{String,Any}(kv...)
    _u(i, j) = O("op" => "index", "args" => Any["u", i, j])
    # F = makearray( 0 over [1,3]x[1,2] ; 2*u[i,j] over the PINNED row i=2 )
    face = O("op" => "aggregate", "args" => Any[], "output_idx" => Any["i", "j"],
             "ranges" => O("i" => Any[2, 2], "j" => Any[1, 2]),
             "expr" => O("op" => "*", "args" => Any[2.0, _u("i", "j")]))
    Fdef = O("op" => "makearray", "args" => Any[],
             "regions" => Any[Any[Any[1, 3], Any[1, 2]], Any[Any[2, 2], Any[1, 2]]],
             "values" => Any[0.0, face])
    rhs = O("op" => "aggregate", "args" => Any[], "output_idx" => Any["i", "j"],
            "ranges" => O("i" => Any[1, 3], "j" => Any[1, 2]),
            "expr" => O("op" => "+", "args" => Any[
                O("op" => "-", "args" => Any[_u("i", "j")]),
                O("op" => "index", "args" => Any["F", "i", "j"])]))
    doc = O("esm" => "0.8.0",
            "metadata" => O("name" => "PinnedFace"),
            "index_sets" => O("x" => O("kind" => "interval", "size" => 3),
                              "y" => O("kind" => "interval", "size" => 2)),
            "models" => O("PinnedFace" => O(
                "variables" => O(
                    "u" => O("type" => "state", "shape" => Any["x", "y"], "default" => 0.5),
                    "F" => O("type" => "observed", "shape" => Any["x", "y"],
                             "expression" => Fdef)),
                "equations" => Any[
                    O("lhs" => O("op" => "D", "args" => Any["u"], "wrt" => "t"),
                      "rhs" => rhs)])))

    mktempdir() do d
        path = joinpath(d, "pinned_face.esm")
        open(io -> JSON3.write(io, doc), path, "w")
        file = EarthSciAST.load(path)
        f!, u0, p, _, _ = build_evaluator(file)
        u = [0.3, 0.7, 1.1, 1.9, 2.3, 2.9][1:length(u0)]

        res = assemble_jacobian(file; u = u, p = p)
        Jad = ad_jacobian(f!, u, p, 0.0)
        @test maximum(abs.(Matrix(res.J) .- Matrix(Jad))) ≤ 1e-14
        @test all((Jad .!= 0) .<= res.pattern)
        # The pinned row really carries the +2, and no other row does: a
        # diagonal of -1 with +2 added only where the region's singleton dim
        # selects it. Diagonal-only, so `res.J` is its own witness.
        vm = Dict(n => i for (i, n) in enumerate(res.rownames))
        for j in 1:2, i in 1:3
            want = i == 2 ? 1.0 : -1.0            # -1 + 2 on the pinned row
            @test res.J[vm["u[$i,$j]"], vm["u[$i,$j]"]] ≈ want
        end
    end
end
