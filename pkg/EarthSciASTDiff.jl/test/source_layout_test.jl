# `source_layout` deletes `prepare_jacobian`'s redundant rebuild of the SOURCE
# model (roughly half its wall time on a large grid). The bar is that it is a
# pure saving: the prepared evaluator must be IDENTICAL, field for field, to the
# one the rebuild produces -- and a layout that does NOT come from a matching
# build must fail loudly rather than silently mis-place every band.

# Everything about a JacobianEvaluator that a caller can observe, minus the
# compiled band RHS itself (which `source_layout` never touches).
_jac_fingerprint(j) = (
    umap = j.umap, scatter = j.scatter,
    colnames = j.colnames, rownames = j.rownames,
    structure = j.structure, wrt = j.wrt,
    nentries = length(j.entries),
    colptr = j.prototype.colptr, rowval = j.prototype.rowval,
    nzval = j.prototype.nzval,
    pattern = j.pattern, nuj = length(j.uj),
)

@testset "source_layout: identical evaluator, no rebuild" begin
    file = fixture("adv_react_2d.esm")
    f!, u0, p, _, vm = build_evaluator(file)
    layout = (u0 = u0, p = p, var_map = vm)

    ja = prepare_jacobian(file)
    jb = prepare_jacobian(file; source_layout = layout)
    @test _jac_fingerprint(ja) == _jac_fingerprint(jb)

    # ... and they agree numerically, at points, to the last bit.
    Ja = jac_prototype(ja); Jb = jac_prototype(jb)
    for _ in 1:3
        u = 0.3 .+ rand(length(u0))
        ja(Ja, u, p, 0.0); jb(Jb, u, p, 0.0)
        @test Ja.nzval == Jb.nzval
    end

    # jacobian_pattern has the same redundant build and the same kwarg.
    ra = jacobian_pattern(file)
    rb = jacobian_pattern(file; source_layout = layout)
    @test ra.pattern == rb.pattern
    @test ra.rownames == rb.rownames && ra.colnames == rb.colnames

    # and the one-shot convenience, which had a THIRD build.
    u = 0.3 .+ rand(length(u0))
    @test assemble_jacobian(file; u = u, p = p).J ==
        assemble_jacobian(file; u = u, p = p, source_layout = layout).J
end

@testset "source_layout: parameter columns do not need `p`" begin
    file = fixture("adv_react_2d.esm")
    _, u0, p, _, vm = build_evaluator(file)
    ja = prepare_jacobian(file; wrt = :parameters)
    jb = prepare_jacobian(
        file; wrt = :parameters,
        source_layout = (u0 = u0, p = p, var_map = vm)
    )
    @test _jac_fingerprint(ja) == _jac_fingerprint(jb)
    # the canonical column space comes from the declared shapes, not `p`
    @test _jac_fingerprint(
        prepare_jacobian(
            file; wrt = :parameters, source_layout = (u0 = u0, var_map = vm)
        )
    ) == _jac_fingerprint(ja)
    # state columns never look at `p`, so there it stays optional.
    @test _jac_fingerprint(
        prepare_jacobian(
            file; source_layout = (u0 = u0, var_map = vm)
        )
    ) ==
        _jac_fingerprint(prepare_jacobian(file))
end

@testset "source_layout: a FlattenedSystem source" begin
    flat = flatten(fixture("coupled_chem_advection.esm"))
    _, u0, p, _, vm = build_evaluator(flat)
    @test _jac_fingerprint(prepare_jacobian(flat)) ==
        _jac_fingerprint(
        prepare_jacobian(
            flat; source_layout = (u0 = u0, p = p, var_map = vm)
        )
    )
end

@testset "source_layout: a mismatched layout is an error, not a wrong answer" begin
    file = fixture("adv_react_2d.esm")
    _, u0, p, _, vm = build_evaluator(file)

    @test_throws ArgumentError prepare_jacobian(file; source_layout = (u0 = u0,))
    @test_throws ArgumentError prepare_jacobian(file; source_layout = (var_map = vm,))
    @test_throws ArgumentError prepare_jacobian(
        file; source_layout = (u0 = u0[1:(end - 1)], p = p, var_map = vm)
    )

    # right length, but the slots are not a state layout of anything.
    bad_slots = Dict(k => 1 for k in keys(vm))
    @test_throws ArgumentError prepare_jacobian(
        file; source_layout = (u0 = u0, p = p, var_map = bad_slots)
    )

    # right shape, wrong NAMES -- the failure class the contract exists for.
    # It must be caught, not scattered into the wrong cells.
    bad_names = Dict{String, Int}()
    for (k, v) in vm
        bad_names["zz_" * k] = v
    end
    @test_throws ArgumentError prepare_jacobian(
        file; source_layout = (u0 = u0, p = p, var_map = bad_names)
    )
    @test_throws ArgumentError jacobian_pattern(
        file; source_layout = (u0 = u0, p = p, var_map = bad_names)
    )
end
