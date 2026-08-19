# Symbolic factorization information — a GENERAL interface with the per-cell
# block (chemistry-shaped) LU as the first implementation.
#
# The interface is structure-agnostic on purpose: `plan_factorization`
# dispatches on the DETECTED structure class, and each plan type carries
# whatever a solver needs to factorize matrices of that class without a
# runtime symbolic analysis (KPP precedent: the LU pattern *including
# fill-in* and the elimination ordering are decided at generation time).
# Future implementations slot in per structure class: banded (bandwidth +
# block strides), general CSC (AMD/Markowitz ordering + fill), …

"""
    FactorizationPlan

Abstract supertype of symbolic factorization information derived from a band
list. Concrete plans are produced by [`plan_factorization`](@ref) and
serialized into the `"jacobians"` document block by `serialize_plan`.
"""
abstract type FactorizationPlan end

"""
    plan_factorization(entries, sv::SysView;
                       structure = detect_structure(entries, sv))
        -> Union{FactorizationPlan, Nothing}

Symbolic factorization info for the detected structure class, or `nothing`
when no plan is implemented for that class (the Jacobian itself is still
fully usable — a plan is an optimization, never a requirement).

Currently implemented: `:block_diagonal` → [`BlockDiagonalPlan`](@ref).
"""
function plan_factorization(entries::Vector{JacEntry}, sv::SysView;
                            structure::Symbol = detect_structure(entries, sv))
    structure == :block_diagonal && return _block_diagonal_plan(entries, sv)
    return nothing
end

"Serialize a plan into the `\"factorization\"` field of the document block."
function serialize_plan end

# ── first implementation: per-cell block LU ─────────────────────────────────

"""
    BlockDiagonalPlan <: FactorizationPlan

Per-cell block factorization info for a `:block_diagonal` Jacobian (the
chemistry shape: every coupling is within-cell, so the global matrix is one
identical `ns × ns`-patterned block per cell):

  - `block_vars`   — the state variables forming the block, in block order.
  - `pattern`      — the block's structural pattern (`ns × ns`, block order).
  - `ordering`     — elimination ordering (a permutation of `1:ns`) chosen by
    a greedy Markowitz heuristic to reduce fill-in.
  - `lu`           — the LU pattern **including fill-in** under `ordering`
    (of the permuted block), so a solver can allocate and generate
    straight-line factorization code without symbolic analysis at runtime.
"""
struct BlockDiagonalPlan <: FactorizationPlan
    block_vars::Vector{String}
    pattern::BitMatrix
    ordering::Vector{Int}
    lu::BitMatrix
end

function _block_diagonal_plan(entries::Vector{JacEntry}, sv::SysView)
    vars = sort!(unique(vcat([en.u for en in entries], [en.v for en in entries])))
    slot = Dict(v => i for (i, v) in enumerate(vars))
    ns = length(vars)
    P = falses(ns, ns)
    for i in 1:ns
        P[i, i] = true                       # W = I − γJ always has the diagonal
    end
    for en in entries
        P[slot[en.u], slot[en.v]] = true
    end
    ordering = markowitz_ordering(P)
    Pp = P[ordering, ordering]
    return BlockDiagonalPlan(vars[ordering], Pp, ordering, lu_fill(Pp))
end

"""
    lu_fill(pattern::AbstractMatrix{Bool}) -> BitMatrix

Symbolic (no-pivoting) LU of a structural pattern: the input pattern plus
every fill-in entry Gaussian elimination creates, in natural order.
"""
function lu_fill(pattern::AbstractMatrix{Bool})::BitMatrix
    P = BitMatrix(pattern)
    n = size(P, 1)
    @assert size(P, 2) == n
    for k in 1:n, i in (k+1):n
        if P[i, k]
            for j in (k+1):n
                P[i, j] |= P[k, j]
            end
        end
    end
    return P
end

"""
    markowitz_ordering(pattern) -> Vector{Int}

Greedy Markowitz elimination ordering: at each step eliminate the remaining
row/column minimizing `(nnz_row − 1)·(nnz_col − 1)` on the symbolically
updated pattern (the KPP `#REORDER` idea: push the most-connected species to
the end so the LU pattern stays small).
"""
function markowitz_ordering(pattern::AbstractMatrix{Bool})::Vector{Int}
    P = BitMatrix(pattern)
    n = size(P, 1)
    remaining = collect(1:n)
    order = Int[]
    while !isempty(remaining)
        best = remaining[1]; bestcost = typemax(Int)
        for i in remaining
            r = count(j -> P[i, j], remaining) - 1
            c = count(j -> P[j, i], remaining) - 1
            cost = r * c
            cost < bestcost && ((best, bestcost) = (i, cost))
        end
        push!(order, best)
        rest = filter(!=(best), remaining)
        for i in rest
            if P[i, best]
                for j in rest
                    P[i, j] |= P[best, j]
                end
            end
        end
        remaining = rest
    end
    return order
end

function serialize_plan(p::BlockDiagonalPlan)
    nzpairs(M) = Any[Any[ij[1], ij[2]] for ij in
                     sort!([(i, j) for i in 1:size(M,1), j in 1:size(M,2) if M[i, j]])]
    OrderedDict{String,Any}(
        "type"       => "block_diagonal_lu",
        "block_vars" => p.block_vars,
        "ordering"   => p.ordering,
        "pattern"    => nzpairs(p.pattern),
        "lu_pattern" => nzpairs(p.lu),
    )
end
