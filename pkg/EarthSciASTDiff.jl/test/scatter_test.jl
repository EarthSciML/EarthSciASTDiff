# Scatter plans (scatter.jl, spec §8): band fields → a consumer's storage.
# Pinned byte-for-byte by tests/goldens/<fixture>.<layout>.plan.json, which
# the Rust binding must reproduce. Executing a bound plan is the consumer's
# work: here the test does it, over band fields EarthSciAST evaluates from the
# evaluation document, and checks the result against `J`.

using EarthSciASTDiff: _nlanes

# Which layouts each golden fixture gets a plan golden for (the rest refuse;
# see the refusal testset). Kept in step with scripts/regenerate-goldens.jl
# and the Rust tests/scatter.rs.
const PLAN_LAYOUTS = [
    ("bd_chem", "lane_blocks", LaneBlocks()),
    ("bd_chem", "lane_sparse_blocks", LaneSparseBlocks()),
    ("bd_chem", "column_banded", ColumnBanded("z")),
    ("bd_chem", "csr", Csr()),
    ("adv_interior", "column_banded", ColumnBanded("x")),
    ("adv_interior", "csr", Csr()),
    ("flux_form_adv", "csr", Csr()),
    ("minmod_adv", "csr", Csr()),
    ("contracted_ops", "csr", Csr()),
    ("contracting_observed", "csr", Csr()),
    ("filtered_scan", "csr", Csr()),
    ("semiring_reduce", "csr", Csr()),
]

@testset "scatter plans: goldens" begin
    golden_dir = joinpath(@__DIR__, "..", "..", "..", "tests", "goldens")
    for (name, tag, layout) in PLAN_LAYOUTS
        plan = scatter_plan(fixture("$name.esm"), layout)
        @test stable_json(serialize_scatter_plan(plan)) ==
            read(joinpath(golden_dir, "$name.$tag.plan.json"), String)
    end
end

# A consumer's execution of a bound plan: resolve each (band, field cell) to
# its evaluator name once, as a consumer binds its slots, then zero `out` and
# `+=` each cell into its position, in plan order.
function _run_plan(b::BoundScatter, field, fields)
    shapes = [Tuple(s) for (_, _, s) in b.plan.bands]
    src = [
        EarthSciASTDiff._cellname(fields[k], Tuple(CartesianIndices(shapes[k])[c]))
            for (k, c, _) in b.pairs
    ]
    out = zeros(b.len)
    for (i, (_, _, pos)) in enumerate(b.pairs)
        out[pos] += field(src[i])
    end
    return out
end

@testset "scatter plans: a bound plan places every band value of J" begin
    cases = collect(PLAN_LAYOUTS)
    push!(cases, ("adv_react_2d", "csr", Csr()))
    for (name, tag, layout) in cases
        file = fixture("$name.esm")
        _, u0, p0, _, _ = build_evaluator(file; _hkw(name)...)
        u = 0.2 .+ 0.6 .* abs.(sin.((1:length(u0)) .* 1.7))
        res = doc_jacobian(file; u = u, p = p0, build_kwargs = _hkw(name))
        J = res.J
        slot = Dict(n => i for (i, n) in enumerate(res.rownames))
        doc, fields = evaluation_document(file)
        ev = _doc_evaluator(doc, "JacobianEval"; build_kwargs = _hkw(name))
        field = _doc_call(ev, _doc_source(file, nothing, _hkw(name)), u, p0, 0.0, Dict())
        plan = scatter_plan(file, layout)
        nl = _nlanes(plan)
        for L in unique([nl, nl + 3])
            b = bind_plan(file, plan; lanes = L)
            out = _run_plan(b, field, fields)
            tg = scatter_targets(plan; lanes = L)
            @test length(tg) == length(out) == b.len
            ok = true; covered = Set{Tuple{Int, Int}}()
            for (pos, t) in enumerate(tg)
                if t === nothing
                    ok &= out[pos] == 0.0          # padding / out-of-range slot
                else
                    r, c = slot[t[1]], slot[t[2]]
                    push!(covered, (r, c))
                    ok &= isapprox(out[pos], J[r, c]; rtol = 1.0e-14, atol = 1.0e-300)
                end
            end
            @test ok
            # every stored entry of the pattern has a home in the layout
            I, Jc, _ = findnz(res.pattern)
            @test issubset(Set(zip(I, Jc)), covered)
            # padding lanes never change a real lane's value
            if L > nl
                o0 = _run_plan(bind_plan(file, plan; lanes = nl), field, fields)
                t0 = scatter_targets(plan; lanes = nl)
                v0 = Dict(t => o0[i] for (i, t) in enumerate(t0) if t !== nothing)
                @test all(out[i] === v0[t] for (i, t) in enumerate(tg) if t !== nothing)
            end
        end
    end
end

@testset "scatter plans: refusals" begin
    bd = fixture("bd_chem.esm")
    adv = fixture("adv_interior.esm")
    refuses(f) = try
        f(); false
    catch e
        e isa ArgumentError
    end
    # layout does not fit the structure
    @test refuses(() -> scatter_plan(adv, LaneBlocks()))
    @test refuses(() -> scatter_plan(adv, LaneSparseBlocks()))
    @test refuses(() -> scatter_plan(fixture("minmod_adv.esm"), ColumnBanded("x")))  # periodic wrap
    @test refuses(() -> scatter_plan(fixture("adv_react_2d.esm"), ColumnBanded("lon")))  # scalar state
    @test refuses(() -> scatter_plan(bd, ColumnBanded("nope")))
    # an empty band list on a model with array states is never a valid plan
    sv = sysview(bd, "Chem")
    @test refuses(() -> scatter_plan(JacEntry[], sv, Csr()))
    # columns that are not states
    @test refuses(() -> scatter_plan(jacobian_bands(sv; wrt = :parameters), sv, Csr()))
    # binding: a plan from another document, a short capacity
    @test refuses(() -> bind_plan(bd, scatter_plan(adv, Csr())))
    @test refuses(() -> bind_plan(bd, scatter_plan(bd, LaneBlocks()); lanes = 2))
    # a contracted band whose column is its row cell is :block_diagonal, but a
    # lane template reads one field cell per lane — it would drop k = 2..hi
    bde = jacobian_bands(sv; wrt = :states)
    b1 = bde[1].band
    con = [
        JacEntry(
            bde[1].u, bde[1].v,
            Band(b1.rows, b1.ridx, b1.cidx, b1.coef, [("kk", 1, 3)])
        );
        bde[2:end]
    ]
    @test refuses(() -> scatter_plan(con, sv, LaneBlocks()))
    @test refuses(() -> scatter_plan(con, sv, LaneSparseBlocks()))
    # same variables and field shapes, mirrored stencil (u[i-1] ↔ u[i+1]):
    # the band list check alone passes; decoding the contributions does not
    asv = sysview(adv, "Adv")
    ae = jacobian_bands(asv; wrt = :states)
    sw(en, o) = JacEntry(
        en.u, en.v,
        Band(en.band.rows, en.band.ridx, o.band.cidx, en.band.coef)
    )
    mirrored = [sw(ae[1], ae[2]), sw(ae[2], ae[1])]
    for layout in (Csr(), ColumnBanded("x"))
        bind_plan(adv, scatter_plan(ae, asv, layout))            # the real one binds
        @test refuses(() -> bind_plan(adv, scatter_plan(mirrored, asv, layout)))
    end
    # a plan whose state cells repeat: refused with the intended message
    p0 = scatter_plan(bd, Csr())
    dup = EarthSciASTDiff.CsrPlan(
        [p0.states; p0.states], p0.bands, p0.rowptr,
        p0.colind, p0.items
    )
    msg = try
        bind_plan(bd, dup); ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("not this source's (order or duplicate state cells)", msg)
end

@testset "scatter plans: lane templates are O(nnz), not O(nnz·cells)" begin
    plan = scatter_plan(fixture("bd_chem.esm"), LaneBlocks())
    @test length(plan.items) == 6            # one per band, whatever the grid
    @test plan.block_vars == ["C", "A", "B"] # the factorization plan's Markowitz order
    sp = scatter_plan(fixture("bd_chem.esm"), LaneSparseBlocks())
    fp = plan_factorization(
        jacobian_bands(fixture("bd_chem.esm"), "Chem"),
        sysview(fixture("bd_chem.esm"), "Chem")
    )
    @test length(sp.lu_pattern) == count(fp.lu)
end
