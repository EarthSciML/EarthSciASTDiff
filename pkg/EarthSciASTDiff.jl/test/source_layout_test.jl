# `source_layout` deletes `jacobian_pattern`'s redundant rebuild of the SOURCE
# model. The bar is that it is a pure saving: the pattern must be IDENTICAL to
# the one the rebuild produces -- and a layout that does NOT come from a
# matching build must fail loudly rather than silently mis-place every entry.

_pat_fingerprint(r) = (pattern = r.pattern, rownames = r.rownames, colnames = r.colnames)

@testset "source_layout: identical pattern, no rebuild" begin
    file = fixture("adv_react_2d.esm")
    _, u0, p, _, vm = build_evaluator(file)
    layout = (u0 = u0, p = p, var_map = vm)
    for wrt in (:states, :parameters)
        @test _pat_fingerprint(jacobian_pattern(file; wrt = wrt)) ==
            _pat_fingerprint(jacobian_pattern(file; wrt = wrt, source_layout = layout))
        # neither axis looks at `p`: the parameter columns come from the
        # declared shapes
        @test _pat_fingerprint(jacobian_pattern(file; wrt = wrt)) == _pat_fingerprint(
            jacobian_pattern(file; wrt = wrt, source_layout = (u0 = u0, var_map = vm))
        )
    end
end

@testset "source_layout: a FlattenedSystem source" begin
    flat = flatten(fixture("coupled_chem_advection.esm"))
    _, u0, p, _, vm = build_evaluator(flat)
    @test _pat_fingerprint(jacobian_pattern(flat)) ==
        _pat_fingerprint(jacobian_pattern(flat; source_layout = (u0 = u0, p = p, var_map = vm)))
end

@testset "source_layout: a mismatched layout is an error, not a wrong answer" begin
    file = fixture("adv_react_2d.esm")
    _, u0, p, _, vm = build_evaluator(file)

    @test_throws ArgumentError jacobian_pattern(file; source_layout = (u0 = u0,))
    @test_throws ArgumentError jacobian_pattern(file; source_layout = (var_map = vm,))
    @test_throws ArgumentError jacobian_pattern(
        file; source_layout = (u0 = u0[1:(end - 1)], p = p, var_map = vm)
    )

    # right length, but the slots are not a state layout of anything.
    bad_slots = Dict(k => 1 for k in keys(vm))
    @test_throws ArgumentError jacobian_pattern(
        file; source_layout = (u0 = u0, p = p, var_map = bad_slots)
    )

    # right shape, wrong NAMES -- the failure class the contract exists for.
    # It must be caught, not placed into the wrong cells.
    bad_names = Dict{String, Int}()
    for (k, v) in vm
        bad_names["zz_" * k] = v
    end
    @test_throws ArgumentError jacobian_pattern(
        file; source_layout = (u0 = u0, p = p, var_map = bad_names)
    )
end
