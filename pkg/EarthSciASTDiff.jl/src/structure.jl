# Matrix-structure detection — read off the bands' column-index expressions,
# never probed numerically.

# Classify one column index expression against its row index:
#   :diag   — col == row                       (per-cell coupling)
#   :affine — col == row ± integer literal     (bounded stencil offset)
#   :fixed  — col is a literal                 (boundary face / periodic wrap)
#   :other  — anything else                    (conn tables, contractions, …)
function _offset_class(c::ASTExpr, r)::Symbol
    if r isa String
        c isa VarExpr && c.name == r && return :diag
        if c isa OpExpr && c.op in ("+", "-") && length(c.args) == 2
            a, b = c.args
            islit(x) = x isa IntExpr || x isa NumExpr
            if a isa VarExpr && a.name == r && islit(b)
                return :affine
            elseif c.op == "+" && b isa VarExpr && b.name == r && islit(a)
                return :affine
            end
        end
        (c isa IntExpr || c isa NumExpr) && return :fixed
        return :other
    else # constant row index (singleton dim)
        if c isa IntExpr || c isa NumExpr
            v = c isa IntExpr ? Int(c.value) : Int(round(c.value))
            return v == r ? :diag : :fixed
        end
        return :other
    end
end

"""
    detect_structure(entries, sv::SysView) -> Symbol

Classify the Jacobian's structure from the band list:

  - `:empty`          — no bands.
  - `:block_diagonal` — every array-state band is diagonal (`col == row` in
    every dimension) and no band couples a scalar state to an array state:
    after the species-major → cell-major permutation the matrix is
    block-diagonal with one dense-patterned block per cell.
  - `:banded`         — off-diagonal couplings exist but every column index
    is `row ± literal` (bounded stencil offsets): banded/block-banded once
    strides are fixed by the layout.
  - `:general`        — anything else (fixed boundary columns, periodic wrap
    corners, indirect gathers): use a general sparse type.

This is a CLASSIFICATION of the exact pattern, not an approximation — the
pattern itself is always available via [`jacobian_pattern`](@ref).
"""
function detect_structure(entries::Vector{JacEntry}, sv::SysView)::Symbol
    isempty(entries) && return :empty
    ctx = _ctx(sv)
    seen_affine = false
    for en in entries
        rowarr = haskey(ctx.shapes, en.u)
        colarr = haskey(ctx.shapes, en.v)
        rowarr != colarr && return :general       # scalar ↔ array coupling
        b = en.band
        isempty(b.cidx) && continue               # scalar–scalar: its own block
        per = [_offset_class(c, r) for (c, r) in zip(b.cidx, b.ridx)]
        any(==(:other), per) && return :general
        any(==(:fixed), per) && return :general
        any(==(:affine), per) && (seen_affine = true)
    end
    return seen_affine ? :banded : :block_diagonal
end

"""
    jac_prototype(jac::JacobianEvaluator) -> SparseMatrixCSC{Float64,Int}
    jac_prototype(input; wrt = :states, model_name = nothing)

A freshly zeroed sparse prototype on the structural pattern — the object to
pass as `ODEFunction(f!; jac_prototype = …)`. Stored (structural) zeros are
kept so the pattern is point-independent.
"""
jac_prototype(jac::JacobianEvaluator) = copy(jac.prototype)

function jac_prototype(input; wrt::Symbol = :states, model_name = nothing)
    pat = jacobian_pattern(input; wrt = wrt, model_name = model_name).pattern
    return SparseMatrixCSC(pat.m, pat.n, copy(pat.colptr), copy(pat.rowval),
                           zeros(length(pat.rowval)))
end
