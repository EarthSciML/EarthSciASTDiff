# Per-cell blocks over caller-named cell axes (`cell_blocks`, `:cell_block`,
# `CellBlockPlan`; spec §4, §6), and the rank rule: a band whose column and
# row have different ranks is never diagonal. The shared golden is
# byte-compared here and in the Rust and Python bindings.

const _CB_GOLDENS = joinpath(@__DIR__, "..", "..", "..", "tests", "goldens")

_cb_load(name) = (f = EarthSciAST.load_path(joinpath(FIXTURES, "$name.esm")); sv = sysview(f, first(keys(f.models))); (f, sv, jacobian_bands(sv)))

@testset "cell blocks: the shared golden" begin
    _, sv, entries = _cb_load("cell_block_mixed_rank")
    cases = map((nothing, ["lev", "bin"], ["lev"], ["salt"])) do axes
        st = detect_structure(entries, sv; cell_axes = axes)
        plan = st in (:block_diagonal, :cell_block) ?
            serialize_plan(plan_factorization(entries, sv; cell_axes = axes, structure = st)) : nothing
        Dict{String, Any}("cell_axes" => axes, "structure" => String(st), "factorization" => plan)
    end
    @test stable_json(Any[cases...]) ==
        read(joinpath(_CB_GOLDENS, "cell_block_mixed_rank.cell_blocks.json"), String)
end

@testset "cell blocks: a column of another rank is not diagonal" begin
    _, sv, entries = _cb_load("cell_block_mixed_rank")
    # w over [bin, lev] reads s over [lev, bin, salt]: the shared dimensions
    # alone would read as diagonal (the old classification)
    @test any(en -> length(en.band.cidx) != length(en.band.ridx), entries)
    @test detect_structure(entries, sv) == :general
    @test plan_factorization(entries, sv) === nothing
end

@testset "cell blocks: what the plan is" begin
    _, sv, entries = _cb_load("cell_block_mixed_rank")
    cb = cell_blocks(entries, sv, ["bin", "lev"])
    @test cb.block_vars == ["s", "w"]
    @test cb.elements == [("s", [1]), ("s", [2]), ("s", [3]), ("w", Int[])]
    @test size(cb.pattern) == (4, 4)
    @test cb.pattern[4, 1:3] == trues(3) && cb.pattern[1:3, 4] == trues(3)
    p = plan_factorization(entries, sv; cell_axes = ["lev", "bin"])
    @test p isa CellBlockPlan
    @test all(p.lu .>= p.pattern)
    # an axis no variable is shaped over, an axis named twice, none at all
    @test cell_blocks(entries, sv, ["salt"]) === nothing
    @test cell_blocks(entries, sv, ["lev", "lev"]) === nothing
    @test cell_blocks(entries, sv, String[]) === nothing
    @test_throws ArgumentError plan_factorization(entries, sv; structure = :cell_block)
    @test_throws ArgumentError plan_factorization(entries, sv; structure = :cell_block, cell_axes = ["salt"])
end

@testset "cell blocks: other classes are kept" begin
    # a stencil leaves its cell: still banded
    _, sv, entries = _cb_load("adv_interior")
    ax = String(first(sv.variables[entries[1].u].shape))
    @test detect_structure(entries, sv) == :banded
    @test detect_structure(entries, sv; cell_axes = [ax]) == :banded
    # a block-diagonal Jacobian stays block diagonal
    _, sv, entries = _cb_load("bd_chem")
    @test detect_structure(entries, sv) == :block_diagonal
end

@testset "structure: dimensions pair by index-set name (the shared golden)" begin
    file = EarthSciAST.load_path(joinpath(FIXTURES, "named_dims.esm"))
    out = Dict{String, Any}()
    for mname in sort!(collect(keys(file.models)))
        sv = sysview(file, mname)
        entries = jacobian_bands(sv)
        st = detect_structure(entries, sv)
        plan = plan_factorization(entries, sv; structure = st)
        out[mname] = Dict{String, Any}(
            "structure" => String(st), "factorization" => plan === nothing ? nothing : serialize_plan(plan)
        )
    end
    @test stable_json(out) == read(joinpath(_CB_GOLDENS, "named_dims.structure.json"), String)
    # a over [cell, salt] reading b over [cell, el] at the same indices is
    # diagonal by position only
    @test out["Mismatch"]["structure"] == "general"
    # p over [x, y] reading q over [y, x] at q[j, i] is diagonal by name only
    @test out["Transposed"]["structure"] == "block_diagonal"
end

@testset "cell blocks: the emitted block takes the cell axes" begin
    file = EarthSciAST.load_path(joinpath(FIXTURES, "cell_block_mixed_rank.esm"))
    doc = jacobian_document(file, "Phase"; cell_axes = ["lev", "bin"])
    block = doc["jacobians"]["Phase"]
    @test stable_json(block) == read(joinpath(_CB_GOLDENS, "cell_block_mixed_rank.cell-lev-bin.block.json"), String)
    @test block["structure"] == "cell_block"
    sv = sysview(file, "Phase")
    plan = plan_factorization(jacobian_bands(sv), sv; cell_axes = ["lev", "bin"])
    @test stable_json(block["factorization"]) == stable_json(serialize_plan(plan))
    # without cell axes: general, and no plan
    b0 = jacobian_document(file, "Phase")["jacobians"]["Phase"]
    @test b0["structure"] == "general" && !haskey(b0, "factorization")
end
