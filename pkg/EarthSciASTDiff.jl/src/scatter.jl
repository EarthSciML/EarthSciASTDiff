# Scatter plans — the precomputed, checked map from band-field values to a
# consumer's own Jacobian storage (spec §8; EarthSciASTDiff#4 W3).
#
# The derived evaluation document (emit.jl) computes one band field `J_k` per
# entry; `prepare_jacobian` scatters those fields into a CSC `nzval`. A
# consumer with its own storage (lane-SoA per-cell blocks, a KPP-style sparse
# block, per-column bands, CSR) would otherwise have to invert that scatter
# and re-derive the state ordering — the reseact class of bug, where a layout
# from another build puts every band in the wrong cell with no visible
# symptom. A plan is that map, built once from the band list, serialized
# byte-identically by every binding (`tests/goldens/*.plan.json`), and bound
# to an evaluator only after its band list and state names have been checked.
#
# Indices in a plan are 0-based throughout (it is a storage map, and its main
# consumers are Rust and C-layout kernels):
#
#   - band field `k`, cell `f`: field `J_{k+1}`, column-major flat cell `f` of
#     its shape (the row variable's shape, then each contracted extent).
#   - canonical state order: `view_states` (sorted names), species-major,
#     each state's cells column-major — the rows and columns of a `Csr` plan.
#
# Lane layouts are per-cell TEMPLATES (O(nnz), not O(nnz·cells)): lane `ℓ` is
# cell `ℓ` of the lane domain, and the lane capacity `L` is chosen at bind
# time, so one plan serves every leaf whatever its capacity.

"""
    StorageLayout

Target storage for a [`ScatterPlan`](@ref): [`LaneBlocks`](@ref),
[`LaneSparseBlocks`](@ref), [`ColumnBanded`](@ref) or [`Csr`](@ref).
"""
abstract type StorageLayout end

"""
    LaneBlocks()

Dense `n × n` block per cell, lane-SoA: `J[r, c]` of lane `ℓ` at
`(r + n·c)·L + ℓ` (0-based). Block order is the factorization plan's
Markowitz `block_vars`, then any state no band touches (sorted). Needs a
`:block_diagonal` Jacobian with every state on one cell shape.
"""
struct LaneBlocks <: StorageLayout end

"""
    LaneSparseBlocks()

KPP-style sparse block per cell: `J[r, c]` of lane `ℓ` at `q·L + ℓ`, `q` the
position of `(r, c)` in the plan's `lu_pattern` (fill-in included; fill
slots are zeroed on every call). Same preconditions as [`LaneBlocks`](@ref).
"""
struct LaneSparseBlocks <: StorageLayout end

"""
    ColumnBanded(axis)

Block-banded per column along index set `axis`; lanes are the columns (every
other dimension, column-major). `J[r, c]` at row level `k`, column level
`k + d`, lane `ℓ`: `((((c·nvar + r)·(kl+ku+1) + (d+kl))·N + k)·L + ℓ`, `r`
and `c` in sorted state order. Needs every coupling within a column, at a
literal offset along `axis`.
"""
struct ColumnBanded <: StorageLayout
    axis::String
end

"""
    Csr()

General sparse rows over the canonical state order (sorted names,
species-major, cells column-major): `rowptr`, sorted `colind`, duplicates
merged. Ghost reads (a column outside its variable) have no entry.
"""
struct Csr <: StorageLayout end

"""
    ScatterPlan

Band fields → storage positions for one (band list, layout). Built by
[`scatter_plan`](@ref), serialized by [`serialize_scatter_plan`](@ref),
bound to an evaluator by [`bind_plan`](@ref). Every plan carries the
canonical `states` (name, shape) and the `bands` (row var, column var, field
shape) it was built from, which is what binding checks.
"""
abstract type ScatterPlan end

const _States = Vector{Tuple{String,Vector{Int}}}
const _Bands = Vector{Tuple{String,String,Vector{Int}}}

struct LaneBlocksPlan <: ScatterPlan
    states::_States
    bands::_Bands
    cells::Vector{Int}                 # the lane domain (common state shape)
    block_vars::Vector{String}
    items::Vector{NTuple{2,Int}}       # (band k, block offset r + n·c)
end

struct LaneSparseBlocksPlan <: ScatterPlan
    states::_States
    bands::_Bands
    cells::Vector{Int}
    block_vars::Vector{String}
    lu_pattern::Vector{NTuple{2,Int}}  # (r, c), sorted; position = q
    items::Vector{NTuple{2,Int}}       # (band k, q)
end

struct ColumnBandedPlan <: ScatterPlan
    states::_States
    bands::_Bands
    cells::Vector{Int}
    axis::String
    axis_dim::Int                      # 0-based dimension of `axis`
    vars::Vector{String}
    kl::Int
    ku::Int
    items::Vector{NTuple{3,Int}}       # (band k, row level, offset without lane)
end

struct CsrPlan <: ScatterPlan
    states::_States
    bands::_Bands
    rowptr::Vector{Int}
    colind::Vector{Int}
    items::Vector{NTuple{3,Int}}       # (band k, field cell, nz position)
end

_field_shape(en::JacEntry, ctx::Ctx) =
    vcat(get(ctx.shapes, en.u, Int[]), Int[hi for (_, _, hi) in en.band.contracted])

# Column-major 0-based flat index of a 1-based cell.
function _flat(cell, shape)
    f = 0; stride = 1
    for (c, n) in zip(cell, shape)
        f += (c - 1) * stride; stride *= n
    end
    return f
end

_plan_error(layout, msg) = throw(ArgumentError("$(nameof(typeof(layout))): $msg"))

"""
    scatter_plan(input, layout::StorageLayout; model_name = nothing) -> ScatterPlan
    scatter_plan(entries, sv::SysView, layout) -> ScatterPlan

Build the scatter plan of a model's state Jacobian for `layout`. Throws an
`ArgumentError` naming the reason when the layout does not fit the Jacobian
(a stencil under `LaneBlocks`, a periodic wrap under `ColumnBanded`, …), and
always for an empty band list on a model with array states — that is the
silent all-zero Jacobian, never a valid plan.
"""
function scatter_plan(input, layout::StorageLayout; model_name = nothing)
    sv, _ = _sv_src(input, model_name)
    return scatter_plan(jacobian_bands(sv; wrt = :states), sv, layout)
end

function scatter_plan(entries::Vector{JacEntry}, sv::SysView, layout::StorageLayout)
    ctx = _ctx(sv)
    names = view_states(sv)
    isempty(names) && _plan_error(layout, "the model has no states")
    states = _States([(u, get(ctx.shapes, u, Int[])) for u in names])
    isempty(entries) && any(!isempty(s) for (_, s) in states) && _plan_error(layout,
        "empty band list for a model with array states (an all-zero Jacobian " *
        "is never a valid plan — check the band calculus against this document)")
    known = Set(names)
    for en in entries
        en.v in known || _plan_error(layout,
            "band column `$(en.v)` is not a state; scatter plans cover wrt = :states")
    end
    bands = _Bands([(en.u, en.v, _field_shape(en, ctx)) for en in entries])
    return _build_plan(layout, entries, sv, ctx, states, bands)
end

function _lane_cells(layout, states::_States)
    shapes = unique(s for (_, s) in states)
    length(shapes) == 1 || _plan_error(layout,
        "every state must share one cell shape; got $(join(sort(string.(shapes)), ", "))")
    return only(shapes)
end

function _block_order(layout, entries, sv, states)
    # A lane template reads one field cell per lane; a contracted band's
    # field has a cell per contracted point too, which it would drop.
    for (k, en) in enumerate(entries)
        isempty(en.band.contracted) || _plan_error(layout,
            "band $k ($(en.u) ← $(en.v)) has contracted dimensions")
    end
    st = detect_structure(entries, sv)
    st == :block_diagonal || _plan_error(layout,
        "needs a :block_diagonal Jacobian (per-cell coupling only); this one is :$st")
    fp = _block_diagonal_plan(entries, sv)
    extra = sort!([u for (u, _) in states if !(u in fp.block_vars)])
    return fp, vcat(fp.block_vars, extra)
end

function _build_plan(layout::LaneBlocks, entries, sv, ctx, states, bands)
    cells = _lane_cells(layout, states)
    _, bv = _block_order(layout, entries, sv, states)
    pos = Dict(u => i - 1 for (i, u) in enumerate(bv))
    n = length(bv)
    items = sort!([(k - 1, pos[en.u] + n * pos[en.v]) for (k, en) in enumerate(entries)];
                  by = it -> (it[2], it[1]))
    return LaneBlocksPlan(states, bands, cells, bv, items)
end

function _build_plan(layout::LaneSparseBlocks, entries, sv, ctx, states, bands)
    cells = _lane_cells(layout, states)
    fp, bv = _block_order(layout, entries, sv, states)
    n0 = length(fp.block_vars)
    lu = sort!(vcat([(i - 1, j - 1) for i in 1:n0, j in 1:n0 if fp.lu[i, j]],
                    [(i - 1, i - 1) for i in (n0+1):length(bv)]))
    q = Dict(rc => i - 1 for (i, rc) in enumerate(lu))
    pos = Dict(u => i - 1 for (i, u) in enumerate(bv))
    items = sort!([(k - 1, q[(pos[en.u], pos[en.v])]) for (k, en) in enumerate(entries)];
                  by = it -> (it[2], it[1]))
    return LaneSparseBlocksPlan(states, bands, cells, bv, lu, items)
end

function _build_plan(layout::ColumnBanded, entries, sv, ctx, states, bands)
    any(isempty(s) for (_, s) in states) && _plan_error(layout,
        "every state must be an array over `$(layout.axis)`; " *
        "$(join([u for (u, s) in states if isempty(s)], ", ")) is scalar")
    dimnames = unique([String.(sv.variables[u].shape) for (u, _) in states])
    length(dimnames) == 1 || _plan_error(layout,
        "every state must share one shape; got $(join(sort(string.(dimnames)), ", "))")
    ax = findfirst(==(layout.axis), only(dimnames))
    ax === nothing && _plan_error(layout,
        "`$(layout.axis)` is not a dimension of the states $(only(dimnames))")
    cells = states[1][2]
    N = cells[ax]
    vars = [u for (u, _) in states]
    vidx = Dict(u => i - 1 for (i, u) in enumerate(vars))
    nvar = length(vars)
    pts = Tuple{Int,Int,Int,Int,Int}[]       # (band k, r, c, row level, offset d)
    for (k, en) in enumerate(entries)
        b = en.band
        isempty(b.contracted) || _plan_error(layout,
            "band $k ($(en.u) ← $(en.v)) has contracted dimensions")
        for (m, (c, r)) in enumerate(zip(b.cidx, b.ridx))
            cls = _offset_class(c, r)
            ok = m == ax ? cls in (:diag, :affine) : cls == :diag
            ok || _plan_error(layout, "band $k ($(en.u) ← $(en.v)) couples cells " *
                (m == ax ? "at a non-literal offset along `$(layout.axis)`" :
                           "across columns (dimension $m is not the axis)"))
        end
        (lo, hi) = b.rows[ax]
        for lev in lo:hi
            bind = b.ridx[ax] isa String ?
                Dict{String,Float64}(b.ridx[ax] => Float64(lev)) : Dict{String,Float64}()
            col = _eval_cidx(b.cidx[ax], bind)
            1 <= col <= N || continue        # ghost read: no column
            push!(pts, (k - 1, vidx[en.u], vidx[en.v], lev - 1, col - lev))
        end
    end
    kl = maximum((-p[5] for p in pts); init = 0)
    ku = maximum((p[5] for p in pts); init = 0)
    W = kl + ku + 1
    items = sort!([(k, lev, ((c * nvar + r) * W + (d + kl)) * N + lev)
                   for (k, r, c, lev, d) in pts]; by = it -> (it[3], it[1]))
    return ColumnBandedPlan(states, bands, cells, layout.axis, ax - 1, vars, kl, ku, items)
end

function _build_plan(layout::Csr, entries, sv, ctx, states, bands)
    base = Dict{String,Int}(); n = 0
    for (u, s) in states
        base[u] = n; n += prod(s; init = 1)
    end
    shape = Dict(states)
    pts = NTuple{4,Int}[]                    # (row, col, band k, field cell)
    for (k, en) in enumerate(entries)
        vshape = shape[en.v]
        _foreach_point(en.band) do cell, cc, cidx
            all(1 <= c <= s for (c, s) in zip(cidx, vshape)) || return  # ghost read
            push!(pts, (base[en.u] + _flat(cell, shape[en.u]),
                        base[en.v] + _flat(cidx, vshape),
                        k - 1, _flat(vcat(cell, cc), bands[k][3])))
        end
    end
    nz = sort!(unique([(p[1], p[2]) for p in pts]))
    pos = Dict(rc => i - 1 for (i, rc) in enumerate(nz))
    rowptr = zeros(Int, n + 1)
    for (r, _) in nz
        rowptr[r+2] += 1
    end
    cumsum!(rowptr, rowptr)
    items = sort!([(k, f, pos[(r, c)]) for (r, c, k, f) in pts];
                  by = it -> (it[3], it[1], it[2]))
    return CsrPlan(states, bands, rowptr, [c for (_, c) in nz], items)
end

# ── serialization (the golden format, spec §8) ──────────────────────────────

_layout_tag(::LaneBlocksPlan) = "lane_blocks"
_layout_tag(::LaneSparseBlocksPlan) = "lane_sparse_blocks"
_layout_tag(::ColumnBandedPlan) = "column_banded"
_layout_tag(::CsrPlan) = "csr"

_ser_items(items) = Any[Any[it...] for it in items]

"""
    serialize_scatter_plan(plan::ScatterPlan) -> OrderedDict

The plan as a JSON-ready dict — `stable_json` of it is the
`tests/goldens/<fixture>.<layout>.plan.json` golden every binding reproduces.
"""
function serialize_scatter_plan(p::ScatterPlan)
    d = OrderedDict{String,Any}(
        "layout" => _layout_tag(p), "wrt" => "states",
        "states" => Any[OrderedDict{String,Any}("name" => u, "shape" => Any[s...])
                        for (u, s) in p.states],
        "bands" => Any[OrderedDict{String,Any}("u" => u, "v" => v, "shape" => Any[s...])
                       for (u, v, s) in p.bands],
        "items" => _ser_items(p.items))
    _ser_layout!(d, p)
    return d
end

function _ser_layout!(d, p::LaneBlocksPlan)
    d["cells"] = Any[p.cells...]; d["block_vars"] = Any[p.block_vars...]
end
function _ser_layout!(d, p::LaneSparseBlocksPlan)
    d["cells"] = Any[p.cells...]; d["block_vars"] = Any[p.block_vars...]
    d["lu_pattern"] = _ser_items(p.lu_pattern)
end
function _ser_layout!(d, p::ColumnBandedPlan)
    d["cells"] = Any[p.cells...]; d["axis"] = p.axis; d["axis_dim"] = p.axis_dim
    d["vars"] = Any[p.vars...]; d["kl"] = p.kl; d["ku"] = p.ku
end
function _ser_layout!(d, p::CsrPlan)
    d["rowptr"] = Any[p.rowptr...]; d["colind"] = Any[p.colind...]
end

# ── binding to an evaluator, and the scatter itself ─────────────────────────

_ncells(shape) = prod(shape; init = 1)

# Lanes of a lane plan (cells), or of a column plan (cells off the axis).
_nlanes(p::Union{LaneBlocksPlan,LaneSparseBlocksPlan}) = _ncells(p.cells)
_nlanes(p::ColumnBandedPlan) = _ncells(p.cells) ÷ p.cells[p.axis_dim+1]
_nlanes(::CsrPlan) = 1

_outlen(p::LaneBlocksPlan, L) = length(p.block_vars)^2 * L
_outlen(p::LaneSparseBlocksPlan, L) = length(p.lu_pattern) * L
_outlen(p::ColumnBandedPlan, L) =
    length(p.vars)^2 * (p.kl + p.ku + 1) * p.cells[p.axis_dim+1] * L
_outlen(p::CsrPlan, _) = length(p.colind)

# Field cell (0-based flat) of column `ℓ` at row level `lev`, both 0-based.
function _column_cell(p::ColumnBandedPlan, lev::Int, ℓ::Int)
    f = 0; stride = 1; rest = ℓ
    for (m, n) in enumerate(p.cells)
        c = m == p.axis_dim + 1 ? lev : (rest % n)
        m == p.axis_dim + 1 || (rest ÷= n)
        f += c * stride; stride *= n
    end
    return f
end

# (band k, field cell, output position) per contribution, in accumulation
# order: plan order, lanes innermost — so every output element sums its
# contributions in the same order whatever the lane capacity.
_contributions(p::Union{LaneBlocksPlan,LaneSparseBlocksPlan}, L) =
    [(k, ℓ, off * L + ℓ) for (k, off) in p.items for ℓ in 0:(_nlanes(p)-1)]
_contributions(p::ColumnBandedPlan, L) =
    [(k, _column_cell(p, lev, ℓ), off * L + ℓ)
     for (k, lev, off) in p.items for ℓ in 0:(_nlanes(p)-1)]
_contributions(p::CsrPlan, _) = p.items

"""
    BoundScatter

A [`ScatterPlan`](@ref) checked against one [`JacobianEvaluator`](@ref) at
a lane capacity: fill with [`jac_into!`](@ref).
"""
struct BoundScatter{P<:ScatterPlan}
    plan::P
    lanes::Int
    len::Int
    pairs::Vector{Tuple{Int,Int}}      # (derived-model du slot, 1-based output position)
end

# Every state cell name, in canonical order (the rows of a `Csr` plan).
function _state_cells(states::_States)
    names = String[]
    for (u, s) in states
        isempty(s) ? push!(names, u) :
            append!(names, [_cellname(u, Tuple(c)) for c in CartesianIndices(Tuple(s))])
    end
    return names
end

_bind_error(msg) = throw(ArgumentError("bind_plan: " * msg))

"""
    bind_plan(jac::JacobianEvaluator, plan::ScatterPlan; lanes = nothing) -> BoundScatter

Check `plan` against `jac` and precompute its gather. Binding is strict: the
plan's band list (row var, column var, field shape per band) and canonical
state cells must equal the evaluator's, and every contribution is decoded
against the evaluator's own bands — each must land on the (row, column) cell
its band point differentiates, and every band point with a column must be
covered exactly once. A plan from another build, another document or another
stencil is refused, never silently misapplied. `lanes` is the lane capacity
`L` of a lane or column layout (default: exactly the lane count; padding
lanes stay zero).
"""
function bind_plan(jac::JacobianEvaluator, plan::ScatterPlan; lanes = nothing)
    jac.wrt == :states || throw(ArgumentError(
        "scatter plans cover wrt = :states; this evaluator is wrt = :$(jac.wrt)"))
    (length(plan.bands) == length(jac.entries) &&
     all(en.u == u && en.v == v && jac.field_shapes[k] == s
         for (k, (en, (u, v, s))) in enumerate(zip(jac.entries, plan.bands)))) ||
        _bind_error("the plan was built from a different band list than this " *
            "evaluator ($(length(plan.bands)) vs $(length(jac.entries)) bands, or a " *
            "band's variables or field shape differ)")
    names = _state_cells(plan.states)
    if !(length(names) == length(jac.rownames) && allunique(names) &&
         Set(names) == Set(jac.rownames))
        extra = setdiff(union(names, jac.rownames), intersect(names, jac.rownames))
        _bind_error("the plan's states are not this evaluator's" *
            (isempty(extra) ? " (duplicate state cells)" : " (e.g. `$(first(extra))`)"))
    end
    nl = _nlanes(plan)
    L = lanes === nothing ? nl : Int(lanes)
    L >= nl || _bind_error("lane capacity $L < $nl lanes")
    # The evaluator's own points: (band k, field cell) → (row, col) cell
    # names, or `nothing` for a ghost read (a value with no storage position).
    vshape = Dict(plan.states)
    want = Dict{Tuple{Int,Int},Union{Nothing,Tuple{String,String}}}()
    for (k, en) in enumerate(jac.entries)
        sh = jac.field_shapes[k]; vs = vshape[en.v]
        _foreach_point(en.band) do cell, cc, cidx
            want[(k - 1, _flat(vcat(cell, cc), sh))] =
                all(1 <= c <= n for (c, n) in zip(cidx, vs)) ?
                    (_cellname(en.u, cell), _cellname(en.v, cidx)) : nothing
        end
    end
    targets = scatter_targets(plan; lanes = L)
    seen = Set{Tuple{Int,Int}}()
    pairs = Tuple{Int,Int}[]           # (derived-model du slot, 1-based output position)
    for (k, f, pos) in _contributions(plan, L)
        if haskey(want, (k, f))
            t = want[(k, f)]
            t === nothing && _bind_error("band $(k + 1) field cell $f is a ghost " *
                "read; the plan gives it a storage position")
            t == targets[pos+1] || _bind_error("band $(k + 1) field cell $f is " *
                "$(t[1]) ← $(t[2]) here; the plan puts it at $(something(targets[pos+1], "padding"))")
            (k, f) in seen && _bind_error("band $(k + 1) field cell $f is scattered twice")
            push!(seen, (k, f))
        end                            # else: outside the band's region, so zero
        c = Tuple(CartesianIndices(Tuple(plan.bands[k+1][3]))[f+1])
        push!(pairs, (_slot(jac.slots, _cellname("J_$(k + 1)", c), "band"), pos + 1))
    end
    for (kf, t) in want
        t === nothing || kf in seen || _bind_error("the plan drops band $(kf[1] + 1) " *
            "field cell $(kf[2]) ($(t[1]) ← $(t[2]))")
    end
    return BoundScatter(plan, L, _outlen(plan, L), pairs)
end

"""
    jac_into!(out, jac::JacobianEvaluator, b::BoundScatter, u, p, t) -> out

Evaluate the band fields at `(u, p, t)` (`u` in the evaluator's state order)
and scatter them into `out` per the bound plan: zeroed, then `+=` in plan
order (several bands may land on one element).
"""
function jac_into!(out::AbstractVector, jac::JacobianEvaluator, b::BoundScatter,
                   u::AbstractVector, p, t::Real)
    length(out) == b.len || throw(DimensionMismatch(
        "output has length $(length(out)); the bound plan needs $(b.len)"))
    duj = _eval_fields!(jac, u, p, t)
    fill!(out, zero(eltype(out)))
    for (s, pos) in b.pairs
        out[pos] += duj[s]
    end
    return out
end

"""
    scatter_targets(plan::ScatterPlan; lanes = nothing)
        -> Vector{Union{Nothing,Tuple{String,String}}}

The (row, column) state-cell names each output position of `plan` holds at
lane capacity `lanes` (`nothing` for padding lanes and out-of-range band
slots). For checking a consumer's decode, and for tests.
"""
function scatter_targets(p::ScatterPlan; lanes = nothing)
    L = lanes === nothing ? _nlanes(p) : Int(lanes)
    out = Vector{Union{Nothing,Tuple{String,String}}}(nothing, _outlen(p, L))
    _targets!(out, p, L)
    return out
end

_lanecell(shape, ℓ) = isempty(shape) ? Int[] :
    collect(Tuple(CartesianIndices(Tuple(shape))[ℓ+1]))

function _targets!(out, p::LaneBlocksPlan, L)
    n = length(p.block_vars)
    for c in 0:(n-1), r in 0:(n-1), ℓ in 0:(_nlanes(p)-1)
        cell = _lanecell(p.cells, ℓ)
        out[(r + n * c) * L + ℓ + 1] =
            (_cellname(p.block_vars[r+1], cell), _cellname(p.block_vars[c+1], cell))
    end
end
function _targets!(out, p::LaneSparseBlocksPlan, L)
    for (q, (r, c)) in enumerate(p.lu_pattern), ℓ in 0:(_nlanes(p)-1)
        cell = _lanecell(p.cells, ℓ)
        out[(q - 1) * L + ℓ + 1] =
            (_cellname(p.block_vars[r+1], cell), _cellname(p.block_vars[c+1], cell))
    end
end
function _targets!(out, p::ColumnBandedPlan, L)
    nvar = length(p.vars); N = p.cells[p.axis_dim+1]; W = p.kl + p.ku + 1
    for c in 0:(nvar-1), r in 0:(nvar-1), j in 0:(W-1), lev in 0:(N-1),
        ℓ in 0:(_nlanes(p)-1)
        col = lev + j - p.kl
        0 <= col < N || continue
        rc = _lanecell(p.cells, _column_cell(p, lev, ℓ))
        cc = copy(rc); cc[p.axis_dim+1] = col + 1
        out[(((c * nvar + r) * W + j) * N + lev) * L + ℓ + 1] =
            (_cellname(p.vars[r+1], rc), _cellname(p.vars[c+1], cc))
    end
end
function _targets!(out, p::CsrPlan, _)
    names = _state_cells(p.states)
    for r in 1:(length(p.rowptr)-1), q in (p.rowptr[r]+1):p.rowptr[r+1]
        out[q] = (names[r], names[p.colind[q]+1])
    end
end
