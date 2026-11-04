# Inliner coverage for `index(makearray …)` value shapes (inline.jl
# `_index_makearray`).
#
# A makearray region's VALUE may be a full-rank faq node — one naming every
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

@testset "inline: full-rank faq value over a pinned makearray region" begin
    O(kv...) = Dict{String, Any}(kv...)
    _u(i, j) = O("op" => "index", "args" => Any["u", i, j])
    # F = makearray( 0 over [1,3]x[1,2] ; 2*u[i,j] over the PINNED row i=2 )
    face = O(
        "op" => "faq", "args" => Any[], "output_idx" => Any["i", "j"],
        "ranges" => O("i" => Any[2, 2], "j" => Any[1, 2]),
        "expr" => O("op" => "*", "args" => Any[2.0, _u("i", "j")])
    )
    Fdef = O(
        "op" => "makearray", "args" => Any[],
        "regions" => Any[Any[Any[1, 3], Any[1, 2]], Any[Any[2, 2], Any[1, 2]]],
        "values" => Any[0.0, face]
    )
    rhs = O(
        "op" => "faq", "args" => Any[], "output_idx" => Any["i", "j"],
        "ranges" => O("i" => Any[1, 3], "j" => Any[1, 2]),
        "expr" => O(
            "op" => "+", "args" => Any[
                O("op" => "-", "args" => Any[_u("i", "j")]),
                O("op" => "index", "args" => Any["F", "i", "j"]),
            ]
        )
    )
    doc = O(
        "esm" => "1.1.0",
        "metadata" => O("name" => "PinnedFace"),
        "index_sets" => O(
            "x" => O("kind" => "interval", "size" => 3),
            "y" => O("kind" => "interval", "size" => 2)
        ),
        "models" => O(
            "PinnedFace" => O(
                "variables" => O(
                    "u" => O("type" => "unknown", "shape" => Any["x", "y"], "default" => 0.5),
                    # esm 1.0.0: an observed is an unknown DEFINED by a
                    # bare-variable-LHS equation; `variable.expression` is gone.
                    "F" => O("type" => "unknown", "shape" => Any["x", "y"])
                ),
                "equations" => Any[
                    O("lhs" => "F", "rhs" => Fdef),
                    O(
                        "lhs" => O("op" => "D", "args" => Any["u"], "wrt" => "t"),
                        "rhs" => rhs
                    ),
                ]
            )
        )
    )

    mktempdir() do d
        path = joinpath(d, "pinned_face.esm")
        open(io -> JSON3.write(io, doc), path, "w")
        file = EarthSciAST.load_path(path)
        f!, u0, p, _, _ = build_evaluator(file)
        u = [0.3, 0.7, 1.1, 1.9, 2.3, 2.9][1:length(u0)]

        res = doc_jacobian(file; u = u, p = p)
        Jad = ad_jacobian(f!, u, p, 0.0)
        @test maximum(abs.(Matrix(res.J) .- Matrix(Jad))) ≤ 1.0e-14
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

# `makearray` region values read by POSITION (esm-spec §4.3.2; the Rust
# binding's tests/makearray_values.rs checks the same cases against central
# differences of earthsci-ast's own right-hand side): a value's element at
# its own index `origin + p` lands on region index `lo + p`, where `origin`
# is a faq value's range start (1 for any other array). A value may be
# full-rank or reduced-rank, and a faq node, a makearray or an array
# variable (the last not here; see below). Each case is read through an
# observed (the inliner's `_index_makearray`) and as the right-hand side
# itself (`_bands_makearray!`), against the Jacobian worked out by hand.
@testset "inline: makearray values are read by position" begin
    O(kv...) = Dict{String, Any}(kv...)
    _u(i, j) = O("op" => "index", "args" => Any["u", i, j])
    twice(oidx, ranges, a, b) = O(
        "op" => "faq", "args" => Any[], "output_idx" => Any[oidx...],
        "ranges" => ranges,
        "expr" => O("op" => "*", "args" => Any[2.0, _u(a, b)])
    )
    U = [0.3 0.7; 1.1 1.9; 2.3 2.9]

    # `reads`: (row, column) cells with ∂ = 2 from the value.
    function check(label, region, value, reads)
        for observed in (true, false)
            Fdef = O(
                "op" => "makearray", "args" => Any[],
                "regions" => Any[Any[Any[1, 3], Any[1, 2]], region],
                "values" => Any[0.0, value]
            )
            variables = O("u" => O("type" => "unknown", "shape" => Any["x", "y"], "default" => 0.5))
            equations = Any[]
            if observed
                variables["F"] = O("type" => "unknown", "shape" => Any["x", "y"])
                push!(equations, O("lhs" => "F", "rhs" => Fdef))
                push!(
                    equations, O(
                        "lhs" => O("op" => "D", "args" => Any["u"], "wrt" => "t"),
                        "rhs" => O(
                            "op" => "faq", "args" => Any[], "output_idx" => Any["i", "j"],
                            "ranges" => O("i" => Any[1, 3], "j" => Any[1, 2]),
                            "expr" => O(
                                "op" => "+", "args" => Any[
                                    O("op" => "-", "args" => Any[_u("i", "j")]),
                                    O("op" => "index", "args" => Any["F", "i", "j"]),
                                ]
                            )
                        )
                    )
                )
            else
                push!(
                    equations,
                    O("lhs" => O("op" => "D", "args" => Any["u"], "wrt" => "t"), "rhs" => Fdef)
                )
            end
            doc = O(
                "esm" => "1.1.0", "metadata" => O("name" => "S"),
                "index_sets" => O(
                    "x" => O("kind" => "interval", "size" => 3),
                    "y" => O("kind" => "interval", "size" => 2)
                ),
                "models" => O("S" => O("variables" => variables, "equations" => equations))
            )
            mktempdir() do d
                path = joinpath(d, "s.esm")
                open(io -> JSON3.write(io, doc), path, "w")
                file = EarthSciAST.load_path(path)
                lay = EarthSciASTDiff.jacobian_pattern(file).rownames
                vm = Dict(n => i for (i, n) in enumerate(lay))
                cell(n) = (m = match(r"^u\[(\d+),(\d+)\]$", n); U[parse(Int, m[1]), parse(Int, m[2])])
                u = [cell(n) for n in lay]
                res = doc_jacobian(file; u = u)
                want = zeros(6, 6)
                if observed
                    for k in 1:6
                        want[k, k] = -1.0
                    end
                end
                for (r, c) in reads
                    want[vm[r], vm[c]] += 2.0
                end
                @test res.rownames == lay
                got = Matrix(res.J)
                ok = maximum(abs.(got .- want)) ≤ 1.0e-12
                ok || @info "makearray value mismatch" label observed got want
                @test ok
            end
        end
    end

    for r in 1:3   # region row 2, value range i ∈ [r,r]: row 2 reads u[r,j]
        check(
            "full-rank faq, i ∈ [$r,$r]", Any[Any[2, 2], Any[1, 2]],
            twice(["i", "j"], O("i" => Any[r, r], "j" => Any[1, 2]), "i", "j"),
            [("u[2,$j]", "u[$r,$j]") for j in 1:2]
        )
    end
    # rows [2,3] of column 2, value k ∈ [1,2]: row i reads u[i−1,1]
    check(
        "reduced-rank faq", Any[Any[2, 3], Any[2, 2]],
        twice(["k"], O("k" => Any[1, 2]), "k", 1),
        [("u[2,2]", "u[1,1]"), ("u[3,2]", "u[2,1]")]
    )
    # An array-VARIABLE value (full-rank `G`, reduced-rank `H`) is covered by
    # the Rust and Python suites only: EarthSciAST.jl's tree-walk cannot
    # build a source whose makearray value is a bare variable
    # (`E_TREEWALK_UNBOUND_VARIABLE`), and `doc_jacobian` builds the source
    # for its state layout.
    inner = O(
        "op" => "makearray", "args" => Any[], "regions" => Any[Any[Any[1, 1], Any[1, 2]]],
        "values" => Any[twice(["j"], O("j" => Any[1, 2]), 1, "j")]
    )
    check(
        "full-rank makearray", Any[Any[2, 2], Any[1, 2]], inner,
        [("u[2,1]", "u[1,1]"), ("u[2,2]", "u[1,2]")]
    )
end
