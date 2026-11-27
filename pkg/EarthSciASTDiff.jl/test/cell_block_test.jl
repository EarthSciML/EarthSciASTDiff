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

@testset "structure: the positional classes (the shared golden)" begin
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
    # p over [x, y] reading q over [y, x] at q[j, i] is diagonal by name
    # only, and the classes are positional: q[j, i] is not stored at p's
    # position (i, j), though x and y have one size
    @test out["Transposed"]["structure"] == "general"
    @test out["Transposed"]["factorization"] === nothing
end

@testset "structure: block_diagonal is positional; cell_block pairs by name" begin
    file = EarthSciAST.load_path(joinpath(FIXTURES, "named_dims.esm"))
    sv = sysview(file, "Transposed")
    entries = jacobian_bands(sv)
    @test detect_structure(entries, sv) == :general
    # the lane-block scatter plan would place ∂p[i, j]/∂q[j, i] in lane
    # (i, j)'s slot for q[i, j]: refused
    err = try
        scatter_plan(file, LaneBlocks(); model_name = "Transposed"); nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("block_diagonal", sprint(showerror, err))
    # over the cell axes [x, y] the axes pair by name: one 2-element block
    # per cell, p and q of that cell
    @test detect_structure(entries, sv; cell_axes = ["x", "y"]) == :cell_block
    cb = cell_blocks(entries, sv, ["x", "y"])
    @test cb.block_vars == ["p", "q"]
    @test cb.elements == [("p", Int[]), ("q", Int[])]
    @test cb.pattern == Bool[1 1; 0 1]
end

@testset "second-order structure: positional, as first order" begin
    # Mismatch: ∂²a/∂a∂b has a second column on b over [cell, el], the
    # row over [cell, salt]: by position the column is the row's, by index
    # set it is not. Even with the first-order class taken as block
    # diagonal, the pair block is not.
    file = EarthSciAST.load_path(joinpath(FIXTURES, "named_dims.esm"))
    sv = sysview(file, "Mismatch")
    e1 = jacobian_bands(sv)
    e2 = jacobian_bands2(sv)
    @test any(en -> en.w == "b", e2)
    memo = EarthSciASTDiff._FirstOrder()
    memo.structure[e1] = :block_diagonal
    @test EarthSciASTDiff._structure2(e1, e2, sv, :states, memo) === nothing
    # a positional pair block keeps the class
    _, sv0, e10 = _cb_load("bd_chem")
    memo0 = EarthSciASTDiff._FirstOrder()
    @test EarthSciASTDiff._structure2(e10, jacobian_bands2(sv0), sv0, :states, memo0) == :block_diagonal
end

@testset "cell axes: a lone name is one axis" begin
    file, sv, entries = _cb_load("cell_block_mixed_rank")
    for lone in ("lev", :lev)
        # the axis `lev`, not the axes `l`, `e`, `v`
        @test cell_blocks(entries, sv, lone) == cell_blocks(entries, sv, ["lev"])
        @test detect_structure(entries, sv; cell_axes = lone) ==
            detect_structure(entries, sv; cell_axes = ["lev"])
        @test stable_json(jacobian_document(file, "Phase"; cell_axes = lone)) ==
            stable_json(jacobian_document(file, "Phase"; cell_axes = ["lev"]))
    end
    # a one-axis cell_block Jacobian, planned from a lone name
    f = EarthSciAST.load_path(joinpath(FIXTURES, "named_dims.esm"))
    tsv = sysview(f, "Transposed")
    te = jacobian_bands(tsv)
    for lone in ("x", :x)
        @test detect_structure(te, tsv; cell_axes = lone) == :cell_block
        @test serialize_plan(plan_factorization(te, tsv; cell_axes = lone)) ==
            serialize_plan(plan_factorization(te, tsv; cell_axes = ["x"]))
    end
end

@testset "cell blocks: a shape over an undeclared index set" begin
    _, sv, entries = _cb_load("cell_block_mixed_rank")
    var = sv.variables["w"]
    sv.variables["w"] = EarthSciAST.ModelVariable(var.type; shape = vcat(var.shape, ["foo"]))
    err = try
        cell_blocks(entries, sv, ["lev"]); nothing
    catch e
        e
    end
    @test err isa EarthSciASTDiff.BandError && err.msg == "unknown index set `foo` in shape"
end
