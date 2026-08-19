# The factorization-info interface and its first (block-diagonal) plan.

# Brute-force oracle for lu_fill: simulate structural Gaussian elimination.
function brute_fill(P0::AbstractMatrix{Bool})
    P = Matrix{Bool}(P0); n = size(P, 1)
    for k in 1:n, i in (k+1):n
        P[i, k] || continue
        for j in (k+1):n
            P[i, j] |= P[k, j]
        end
    end
    P
end

@testset "lu_fill" begin
    for _ in 1:20
        n = rand(3:8)
        P = rand(n, n) .< 0.35
        for i in 1:n; P[i, i] = true; end        # nonzero diagonal (no pivoting)
        F = lu_fill(P)
        @test Matrix(F) == brute_fill(P)
        @test all(F .>= P)                        # fill only adds entries
    end
end

@testset "markowitz ordering beats natural order on an arrow matrix" begin
    # Arrow pointing the wrong way: dense first row/col. Natural order fills
    # completely; eliminating the hub last keeps the pattern sparse.
    n = 8
    P = falses(n, n)
    for i in 1:n
        P[i, i] = true; P[1, i] = true; P[i, 1] = true
    end
    fill_natural = count(lu_fill(P))
    ord = EarthSciASTDiff.markowitz_ordering(P)
    fill_ordered = count(lu_fill(P[ord, ord]))
    @test fill_ordered < fill_natural
    @test fill_ordered == count(P)               # arrow eliminated last: no fill
    @test sort(ord) == 1:n
end

@testset "plan_factorization" begin
    file = fixture("bd_chem.esm")
    sv = sysview(file, "Chem")
    entries = jacobian_bands(sv)
    plan = plan_factorization(entries, sv)
    @test plan isa BlockDiagonalPlan
    @test sort(plan.block_vars) == ["A", "B", "C"]
    @test size(plan.pattern) == (3, 3)
    @test all(plan.lu .>= plan.pattern)
    ser = EarthSciASTDiff.serialize_plan(plan)
    @test ser["type"] == "block_diagonal_lu"
    @test length(ser["lu_pattern"]) == count(plan.lu)

    # The general interface returns nothing for classes without a plan yet.
    file2 = fixture("adv_interior.esm")
    sv2 = sysview(file2, "Adv")
    entries2 = jacobian_bands(sv2)
    @test plan_factorization(entries2, sv2) === nothing
end
