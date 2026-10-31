# Scatter plans (scatter.jl, spec §8): band fields → a consumer's storage.
# Pinned byte-for-byte by tests/goldens/<fixture>.<layout>.plan.json, which
# the Rust binding must reproduce; the values a bound plan writes must be the
# prepared CSC `jac!`'s, element for element.

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

@testset "scatter plans: bound values equal the CSC jac!" begin
    # The contracted-band fixtures are excluded only because their derived
    # documents do not compile on the current EarthSciAST (EarthSciAST#547);
    # their plan goldens are still pinned above.
    cases = [
        (n, t, l) for (n, t, l) in PLAN_LAYOUTS
            if !(n in ("contracted_ops", "contracting_observed", "filtered_scan", "semiring_reduce"))
    ]
    push!(cases, ("adv_react_2d", "csr", Csr()))
    for (name, tag, layout) in cases
        file = fixture("$name.esm")
        _, u0, p0, _, _ = build_evaluator(file)
        jac = prepare_jacobian(file)
        u = 0.2 .+ 0.6 .* abs.(sin.((1:length(u0)) .* 1.7))
        J = copy(jac.prototype)
        jac(J, u, p0, 0.0)
        slot = Dict(n => i for (i, n) in enumerate(jac.rownames))
        plan = scatter_plan(file, layout)
        nl = _nlanes(plan)
        for L in unique([nl, nl + 3])
            b = bind_plan(jac, plan; lanes = L)
            out = jac_into!(zeros(b.len), jac, b, u, p0, 0.0)
            tg = scatter_targets(plan; lanes = L)
            @test length(tg) == length(out)
            ok = true; covered = Set{Tuple{Int, Int}}()
            for (pos, t) in enumerate(tg)
                if t === nothing
                    ok &= out[pos] == 0.0          # padding / out-of-range slot
                else
                    r, c = slot[t[1]], slot[t[2]]
                    push!(covered, (r, c))
                    ok &= out[pos] == J[r, c]      # same summation order: bitwise
                end
            end
            @test ok
            # every stored entry of the CSC pattern has a home in the layout
            I, Jc, _ = findnz(jac.pattern)
            @test issubset(Set(zip(I, Jc)), covered)
            # padding lanes never change a real lane's value
            if L > nl
                b0 = bind_plan(jac, plan; lanes = nl)
                o0 = jac_into!(zeros(b0.len), jac, b0, u, p0, 0.0)
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
    # binding: a plan from another document, a short capacity, a non-state evaluator
    jac = prepare_jacobian(bd)
    @test refuses(() -> bind_plan(jac, scatter_plan(adv, Csr())))
    @test refuses(() -> bind_plan(jac, scatter_plan(bd, LaneBlocks()); lanes = 2))
    @test refuses(
        () -> bind_plan(
            prepare_jacobian(bd; wrt = :parameters),
            scatter_plan(bd, Csr())
        )
    )
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
    ajac = prepare_jacobian(adv)
    for layout in (Csr(), ColumnBanded("x"))
        bind_plan(ajac, scatter_plan(ae, asv, layout))           # the real one binds
        @test refuses(() -> bind_plan(ajac, scatter_plan(mirrored, asv, layout)))
    end
    # a plan whose state cells repeat: refused with the intended message
    p0 = scatter_plan(bd, Csr())
    dup = EarthSciASTDiff.CsrPlan(
        [p0.states; p0.states], p0.bands, p0.rowptr,
        p0.colind, p0.items
    )
    msg = try
        bind_plan(jac, dup); ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("not this evaluator's (duplicate state cells)", msg)
    b = bind_plan(jac, scatter_plan(bd, Csr()))
    @test_throws DimensionMismatch jac_into!(
        zeros(b.len + 1), jac, b,
        ones(12), nothing, 0.0
    )
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
