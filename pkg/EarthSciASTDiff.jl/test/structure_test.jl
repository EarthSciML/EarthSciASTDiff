# Structure classification from the bands' column-index expressions.

@testset "detect_structure" begin
    # Pure per-cell chemistry → block-diagonal.
    file = fixture("bd_chem.esm")
    sv = sysview(file, "Chem")
    entries = jacobian_bands(sv)
    @test detect_structure(entries, sv) == :block_diagonal

    # Interior stencil, constant faces → every band affine → banded.
    file = fixture("adv_interior.esm")
    sv = sysview(file, "Adv")
    entries = jacobian_bands(sv)
    @test detect_structure(entries, sv) == :banded
    @test all(!isempty(en.band.cidx) for en in entries)

    # Scalar ↔ array coupling and zero-gradient faces → general.
    file = fixture("adv_react_2d.esm")
    sv = sysview(file, "AdvReact")
    entries = jacobian_bands(sv)
    @test detect_structure(entries, sv) == :general
end

@testset "jac_prototype matches pattern" begin
    file = fixture("bd_chem.esm")
    proto = jac_prototype(file)
    res = jacobian_pattern(file)
    @test size(proto) == size(res.pattern)
    @test all(proto.nzval .== 0.0)                # a prototype is all stored zeros
    @test proto.rowval == res.pattern.rowval
    @test proto.colptr == res.pattern.colptr
    # Block-diagonal check by permutation: cell-major reordering makes the
    # pattern block-diagonal with 3×3 blocks.
    n = 12; ns = 3; nc = 4
    names = res.rownames
    # species-major layout: A[1..4], B[1..4], C[1..4] → cell-major permutation
    cellmajor = [findfirst(==(sp * "[" * string(c) * "]"), names)
                 for c in 1:nc for sp in ("A", "B", "C")]
    P = res.pattern[cellmajor, cellmajor]
    for i in 1:n, j in 1:n
        blk_i = (i - 1) ÷ ns; blk_j = (j - 1) ÷ ns
        blk_i != blk_j && @test !P[i, j]
    end
end
