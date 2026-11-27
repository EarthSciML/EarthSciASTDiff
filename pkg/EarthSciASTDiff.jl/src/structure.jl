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
  - `:column_block`    — only when `column_axis` (an index-set name) is given,
    in place of `:banded` or `:general`: every coupling stays within one
    LINE along that axis (see [`column_lines`](@ref)) — a column scan, a
    contraction over the axis, a stencil along it — so a solver can
    factorize line by line.
  - `:cell_block`      — only when `cell_axes` (index-set names) is given, in
    place of `:banded` or `:general` and before `:column_block`: every
    coupling stays inside one CELL of those axes (see [`cell_blocks`](@ref)),
    so the matrix is one block per cell holding every element of every
    variable along the other axes (a species or salt axis summed inside the
    cell, variables of different ranks), factorized cell by cell.

`:block_diagonal` and `:banded` are POSITIONAL: a band's column is compared
with its row dimension by dimension at the same position, because their
consumers (the lane-block scatter plan, the block-diagonal LU plan) pair a
row cell with the column cell at the same storage position. So a band whose
column and row have different ranks (a variable over `cell` reading one
over `cell × salt`), whose column is over an index set its row is not (a
variable over `cell × salt` reading one over `cell × el` with matching
indices), or whose column is over the row's index sets in another order (a
variable over `x × y` reading one over `y × x` at `[j, i]`) is never
`:block_diagonal` or `:banded`: it is `:general` unless `cell_axes` makes it
`:cell_block`, whose plan pairs the cell axes by index-set name. When either
variable's index-set names are not known, the dimensions pair by position.

`column_axis` and `cell_axes` come from the caller (an integrator's locality
pass): an index-set name carries no vertical or cell meaning by itself.

This is a CLASSIFICATION of the exact pattern, not an approximation — the
pattern itself is always available via [`jacobian_pattern`](@ref).
"""
function detect_structure(
        entries::Vector{JacEntry}, sv::SysView; column_axis = nothing, cell_axes = nothing
    )::Symbol
    st = _detect_structure(entries, sv)
    st in (:empty, :block_diagonal) && return st
    cell_axes = _cell_axes_list(cell_axes)
    cell_axes === nothing || cell_blocks(entries, sv, cell_axes) === nothing || return :cell_block
    column_axis === nothing && return st
    return column_lines(entries, sv, String(column_axis)) === nothing ? st : :column_block
end

"""
    _positional(ctx, u, v) -> Bool

Whether column variable `v` lies over row variable `u`'s index sets in the
same order, so that pairing their dimensions by position (what
`:block_diagonal` and `:banded`, and the lane-block scatter plan and the
block-diagonal LU plan that read them, do) is pairing them by index-set name
(esm-spec §4.3.4). A column over an index set its row is not, or over the
row's sets in another order, is not positional. When either variable's
index-set names are not known (`ctx.axes`), it is.
"""
function _positional(ctx, u::String, v::String)
    axu = get(ctx.axes, u, nothing)
    axv = get(ctx.axes, v, nothing)
    return axu === nothing || axv === nothing || axu == axv
end

function _detect_structure(entries::Vector{JacEntry}, sv::SysView)::Symbol
    isempty(entries) && return :empty
    ctx = _ctx(sv)
    seen_affine = false
    for en in entries
        rowarr = haskey(ctx.shapes, en.u)
        colarr = haskey(ctx.shapes, en.v)
        rowarr != colarr && return :general       # scalar ↔ array coupling
        b = en.band
        isempty(b.cidx) && continue               # scalar–scalar: its own block
        # a column of another rank than its row is not the row in every
        # dimension, whatever the shared dimensions say
        length(b.cidx) == length(b.ridx) || return :general
        # the classes are positional (their consumers pair row and column
        # cells by storage position): a column over other index sets than
        # its row, or over them in another order, is not the row
        _positional(ctx, en.u, en.v) || return :general
        per = [_offset_class(b.cidx[d], b.ridx[d]) for d in eachindex(b.cidx)]
        any(==(:other), per) && return :general
        any(==(:fixed), per) && return :general
        any(==(:affine), per) && (seen_affine = true)
    end
    return seen_affine ? :banded : :block_diagonal
end

"""
    column_lines(entries, sv, axis::String) -> Union{NamedTuple, Nothing}

The per-line structure of a state Jacobian whose couplings all stay within
one line along index set `axis`, or `nothing` when one does not: every band
couples two array states of one common shape that holds `axis` once, and
at every band point the column equals the row in every other dimension. A
point reading past the axis is a ghost read and does not count. Returns
`(axis, dim, line_length, block_vars, kl, ku, pattern, rank1, rank1_undecided)`: `kl` / `ku` the
largest distances below / above the diagonal any point reaches along the
axis, and `pattern` their name — `"dense"` (both span the line), `"lower"`,
`"upper"`, `"banded"` or `"diag_plus_rank1"` — read off the exact band
points, never probed, with `rank1` the rank-one terms of a
`"diag_plus_rank1"` line block (esm-jacobian-spec.md §6; empty otherwise).

Structure-only entries (§2.1) carry no coefficient to factor. When every
entry off distance 0 has the shape of a rank-one term and one of them has no
coefficient, whether the line block is `"diag_plus_rank1"` depends on the
coefficients: `rank1_undecided` is then `true`, `pattern` names the
bandwidth alone and `rank1` is empty, and [`plan_factorization`](@ref)
refuses to plan from it.
"""
function column_lines(entries::Vector{JacEntry}, sv::SysView, axis::String)
    isempty(entries) && return nothing
    shape = nothing
    for en in entries, n in (en.u, en.v)
        s = get(sv.variables, n, nothing)
        (s === nothing || s.shape === nothing || isempty(s.shape)) && return nothing
        sh = String[string(x) for x in s.shape]
        shape === nothing && (shape = sh)
        sh == shape || return nothing
    end
    dim = findall(==(axis), shape)
    length(dim) == 1 || return nothing
    dim = only(dim)
    N = _ctx(sv).index_sets[axis]
    kl = 0; ku = 0
    # per entry: whether every point sits at distance 0 on the axis
    diag = fill(true, length(entries))
    for (k, en) in enumerate(entries)
        ok = _all_points(en.band) do cell, _, cidx
            for d in eachindex(cidx)
                d == dim || cidx[d] == cell[d] || return false
            end
            1 <= cidx[dim] <= N || return true            # ghost read
            off = cidx[dim] - cell[dim]
            kl = max(kl, -off); ku = max(ku, off)
            diag[k] &= off == 0
            return true
        end
        ok || return nothing
    end
    # every entry at distance 0 or a rank-one term, and at least one of
    # those: D + Σ u vᵀ per line, for a Sherman–Morrison solve
    rank1 = Any[]
    lowrank = (kl, ku) != (0, 0)
    undecided = false
    for (k, en) in enumerate(entries)
        diag[k] && continue
        if en.band.coef === nothing && _rank1_shape(en.band, dim)
            undecided = true
            continue
        end
        t = _rank1_term(en.band, dim)
        if t === nothing
            lowrank = false
        else
            push!(
                rank1, OrderedDict{String, Any}(
                    "entry" => k,
                    "u" => _ser_expr(_canon_lits(t[1])), "v" => _ser_expr(_canon_lits(t[2])),
                )
            )
        end
    end
    undecided &= lowrank
    (lowrank && !undecided) || empty!(rank1)
    pattern = !isempty(rank1) ? "diag_plus_rank1" :
        kl == N - 1 && ku == N - 1 ? "dense" :
        ku == 0 && kl == N - 1 ? "lower" :
        kl == 0 && ku == N - 1 ? "upper" : "banded"
    vars = sort!(unique(vcat([en.u for en in entries], [en.v for en in entries])))
    return (
        axis = axis, dim = dim, line_length = N, block_vars = vars,
        kl = kl, ku = ku, pattern = pattern, rank1 = rank1, rank1_undecided = undecided,
    )
end

# The caller's cell axes as index-set names: a lone name (a String or a
# Symbol) is one axis, not a sequence of one-character names.
_cell_axes_list(::Nothing) = nothing
_cell_axes_list(a::Union{AbstractString, Symbol}) = String[String(a)]
_cell_axes_list(a) = String[string(x) for x in a]

"""
    cell_blocks(entries, sv, cell_axes) -> Union{NamedTuple, Nothing}

The per-cell block structure of a state Jacobian whose couplings all stay
inside one cell of the index sets `cell_axes`, or `nothing` when one does
not: every row and column variable is an array whose shape holds each cell
axis exactly once, and at every band point the column's index on each cell
axis equals the row's (the axes are paired by index-set name, so variables
of different ranks and orders pair). A point whose column falls outside its
variable's extent is a ghost read and does not count. The block of a cell
holds, for each variable in name order, its elements along its other axes,
column-major (first index fastest): `elements` lists them as
`(variable, indices along the other axes)`, 1-based, and `pattern` is the
union over cells of the block's structural pattern in that order, its
diagonal included (`W = I − γJ` always has it). Returns
`(cell_axes, block_vars, elements, pattern)`.
"""
function cell_blocks(entries::Vector{JacEntry}, sv::SysView, cell_axes)
    axes = _cell_axes_list(cell_axes)
    (isempty(entries) || isempty(axes) || !allunique(axes)) && return nothing
    sizes = _ctx(sv).index_sets
    vars = sort!(unique(vcat([en.u for en in entries], [en.v for en in entries])))
    info = Dict{String, Any}()
    elements = Tuple{String, Vector{Int}}[]
    for v in vars
        s = get(sv.variables, v, nothing)
        (s === nothing || s.shape === nothing || isempty(s.shape)) && return nothing
        sh = String[string(x) for x in s.shape]
        pos = Int[]
        for a in axes
            k = findall(==(a), sh)
            length(k) == 1 || return nothing
            push!(pos, only(k))
        end
        other = [d for d in eachindex(sh) if !(d in pos)]
        ext = Int[sizes[sh[d]] for d in other]
        info[v] = (pos = pos, other = other, ext = ext, full = Int[sizes[x] for x in sh], off = length(elements))
        if isempty(ext)
            push!(elements, (v, Int[]))
        else
            for ci in CartesianIndices(Tuple(ext))
                push!(elements, (v, collect(Int, Tuple(ci))))
            end
        end
    end
    ns = length(elements)
    P = falses(ns, ns)
    for i in 1:ns
        P[i, i] = true
    end
    _local(I, idx) = begin
        k = 1; stride = 1
        for (d, x) in zip(I.other, I.ext)
            k += (idx[d] - 1) * stride
            stride *= x
        end
        I.off + k
    end
    for en in entries
        iu, iv = info[en.u], info[en.v]
        ok = _all_points(en.band) do cell, _, cidx
            length(cidx) == length(iv.full) || return false
            for d in eachindex(cidx)
                1 <= cidx[d] <= iv.full[d] || return true    # ghost read
            end
            for k in eachindex(axes)
                cidx[iv.pos[k]] == cell[iu.pos[k]] || return false
            end
            P[_local(iu, cell), _local(iv, cidx)] = true
            return true
        end
        ok || return nothing
    end
    return (cell_axes = axes, block_vars = vars, elements = elements, pattern = P)
end

# The factors of a product: numerator and denominator factors, `*` and `/`
# flattened and a unary minus taken out as -1.
function _factors!(e::ASTExpr, num::Vector{ASTExpr}, den::Vector{ASTExpr})
    if e isa OpExpr
        if e.op == "*"
            foreach(a -> _factors!(a, num, den), e.args)
            return
        elseif e.op == "/" && length(e.args) == 2
            _factors!(e.args[1], num, den)
            _factors!(e.args[2], den, num)
            return
        elseif e.op == "-" && length(e.args) == 1
            push!(num, lit(-1))
            _factors!(e.args[1], num, den)
            return
        end
    end
    push!(num, e)
    return
end

# `coef = u · v` with `u` free of the contracted name `kap` and `v` free of
# every row name, or `nothing` when a factor reads both.
function _split_rank1(coef::ASTExpr, kap::String, rows::Set{String})
    num = ASTExpr[]; den = ASTExpr[]
    _factors!(coef, num, den)
    un = ASTExpr[]; ud = ASTExpr[]; vn = ASTExpr[]; vd = ASTExpr[]
    for (fs, is_num) in ((num, true), (den, false)), f in fs
        fv = free_variables(f)
        k = kap in fv
        r = !isdisjoint(fv, rows)
        k && r && return nothing
        push!(k ? (is_num ? vn : vd) : (is_num ? un : ud), f)
    end
    build(n, d) = isempty(d) ? mul(n...) : op("/", mul(n...), mul(d...))
    return build(un, ud), build(vn, vd)
end

# A band whose every point reads one other cell of its row's line — the line
# position being its one contracted name, under no `where` — with a
# coefficient that splits as `u(row) · v(κ)`: a rank-one term. A
# structure-only band (no coefficient, spec §2.1) has none to split.
function _rank1_term(b::Band, dim::Int)
    (b.coef === nothing || !_rank1_shape(b, dim)) && return nothing
    kap = b.contracted[1][1]
    return _split_rank1(b.coef, kap, Set{String}(r for r in b.ridx if r isa String))
end

# The shape of a rank-one term, coefficient aside: one contracted name, the
# line position, under no `where`.
function _rank1_shape(b::Band, dim::Int)
    (b.guard === nothing && length(b.contracted) == 1) || return false
    kap = b.contracted[1][1]
    return dim <= length(b.cidx) && b.cidx[dim] isa VarExpr && b.cidx[dim].name == kap
end

"""
    jac_prototype(input; wrt = :states, model_name = nothing)

A freshly zeroed sparse prototype on the structural pattern — the object to
pass as `ODEFunction(f!; jac_prototype = …)`. Stored (structural) zeros are
kept so the pattern is point-independent.
"""
function jac_prototype(input; wrt::Symbol = :states, model_name = nothing)
    pat = jacobian_pattern(input; wrt = wrt, model_name = model_name).pattern
    return SparseMatrixCSC(
        pat.m, pat.n, copy(pat.colptr), copy(pat.rowval),
        zeros(length(pat.rowval))
    )
end
