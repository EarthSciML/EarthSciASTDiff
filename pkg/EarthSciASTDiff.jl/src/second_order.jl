# Second-order bands (esm-jacobian-spec.md §4.4): the band calculus applied
# once more, to the coefficients of the first-order bands. A Rosenbrock step
# contains J(uₙ) explicitly, so its discrete adjoint needs ∇(μᵀ J(u) w) once
# per step; that is a contraction of these entries, never a Hessian tensor.

"""
    JacEntry2

One second-order band: `∂²(equation for u)/∂v ∂w`. `band` is the first-order
entry's band with its coefficient replaced by `∂coef/∂(site of w)`: the same
`rows`, `ridx` and `contracted` (plus any dimensions the second site adds
under a nested sum), and `cidx` holding `v`'s column indices FOLLOWED by
`w`'s. The first `ncol` of them are `v`'s ([`col_idx`](@ref) /
[`col2_idx`](@ref) split them). Keeping both in one `Band` lets clipping,
merging and normalization run unchanged.
"""
struct JacEntry2
    u::String
    v::String
    w::String
    band::Band
    ncol::Int
end

col_idx(en::JacEntry2) = en.band.cidx[1:(en.ncol)]
col2_idx(en::JacEntry2) = en.band.cidx[(en.ncol + 1):end]

# A second-order `wrt` is a pair: the first derivative's axis, then the
# second's (esm-jacobian-spec.md §4.4).
_axis_pair(wrt::Pair) = _axis_pair((wrt.first, wrt.second))
function _axis_pair(wrt)::Tuple{Symbol, Symbol}
    (wrt isa Union{Tuple, AbstractVector} && length(wrt) == 2) ||
        throw(ArgumentError("a second-order wrt is a pair (a1, a2), got $wrt"))
    a1, a2 = Symbol(wrt[1]), Symbol(wrt[2])
    a1 in (:states, :time) ||
        throw(ArgumentError("the first axis of a pair must be :states or :time, got $a1"))
    a2 in (:states, :parameters, :forcing) || throw(
        ArgumentError("the second axis of a pair must be :states, :parameters or :forcing, got $a2")
    )
    return (a1, a2)
end

"""
    jacobian_bands2(file::EsmFile, model_name; wrt = (:states, :states), forcing = nothing) -> Vector{JacEntry2}
    jacobian_bands2(flat::FlattenedSystem;    wrt = (:states, :states), forcing = nothing) -> Vector{JacEntry2}
    jacobian_bands2(sv::SysView;              wrt = (:states, :states), forcing = nothing) -> Vector{JacEntry2}

Second-order bands for the axis pair `wrt = (a1, a2)`, `a1 ∈ (:states, :time)`
and `a2 ∈ (:states, :parameters, :forcing)`: every first-order entry of the
`a1` axis ([`jacobian_bands`](@ref)) with its coefficient differentiated by
the same calculus with respect to every site of every `a2` target, broadcast
over the entry's rows with region clipping and re-merging. No new rules: the
non-smooth policy (spec §3.2) applies again, and the pattern is the global
structural union over branches.

`(r, c, s)` and `(r, s, c)` come out independently, from distinct first-order
entries; consumers must not assume either is deduplicated. On the
`:parameters` and `:forcing` second axes every `col2_idx` must be closed at
build time, as on the first-order axes (esm-jacobian-spec.md §4.3).
"""
jacobian_bands2(file::EarthSciAST.EsmFile, model_name; kw...) =
    jacobian_bands2(sysview(file, model_name); kw...)
jacobian_bands2(flat::EarthSciAST.FlattenedSystem; kw...) =
    jacobian_bands2(sysview(flat); kw...)
jacobian_bands2(sv::SysView; wrt = (:states, :states), forcing = nothing) =
    last(_bands2(sv, _axis_pair(wrt)..., forcing))

# (first-order entries, second-order entries). The first-order pass inlines
# the observed that move with EITHER axis: one that reads only parameters
# (an Arrhenius rate) is opaque on the state axis, and would hide its
# parameter sites from the second derivative.
function _bands2(sv::SysView, a1::Symbol, a2::Symbol, forcing, memo::_FirstOrder = _FirstOrder())
    ctx = _ctx(sv)
    t2 = _targets(sv, a2, forcing, a2 == :states ? _observed_defs(sv) : nothing)
    e1 = _first_order!(memo, sv, a1, a2, forcing, t2)
    tset = Set{String}(t2)
    out = JacEntry2[]
    for en in e1
        for w in sort!(String[x for x in free_variables(en.band.coef) if x in tset])
            for b in merge_bands(_coef_bands(en.band, w, ctx))
                push!(out, JacEntry2(en.u, en.v, w, normalize_band(b), length(en.band.cidx)))
            end
        end
    end
    a2 in (:parameters, :forcing) && foreach(_check_closed_col2, out)
    return e1, out
end

# One band per site of `w` in a first-order coefficient, broadcast over the
# entry's rows: `_scalar_bands!` with the entry's contracted names bound too.
function _coef_bands(b::Band, w::String, ctx::Ctx)
    out = Band[]
    outer = String[[r for r in b.ridx if r isa String]; [n for (n, _, _) in b.contracted]]
    for (rows2, body) in clip_regions(b.coef, b.ridx, b.rows)
        nested = _nested_sums(body, w, outer)
        taken = Ref{Union{Nothing, Set{String}}}(nothing)
        for s in sorted_sites(body, w)
            s.expr isa VarExpr && is_array(ctx, w) && throw(
                BandError("whole-array reference to `$w` in a band coefficient")
            )
            c2 = s.expr isa VarExpr ? ASTExpr[] : ASTExpr[s.expr.args[2:end]...]
            c2, d, extra, g = _lift_nested(nested, s, body, c2, dscalar(body, s), ctx, outer, taken)
            coef = _simp(d)
            iszero_lit(coef) && continue
            c2, coef, contracted = _one_hot(c2, coef, vcat(b.contracted, extra), b.ridx, w, ctx)
            push!(
                out, Band(
                    rows2, b.ridx, vcat(b.cidx, c2), coef, contracted, _conj(b.guard, g)
                )
            )
        end
    end
    return out
end

_check_closed_col2(en::JacEntry2) = _check_closed(
    en.band, col2_idx(en),
    c -> "second column index $(_json(c)) of `$(en.w)` in ∂²($(en.u))/∂$(en.v)∂$(en.w)"
)

# `:block_diagonal` when the first-order state block is and every second
# column sits on the row's own cell, so the contraction is lane-local;
# `nothing` (field omitted) otherwise, and always on a time pair.
function _structure2(
        e1::Vector{JacEntry}, entries::Vector{JacEntry2}, sv::SysView, a1::Symbol,
        memo::_FirstOrder = _FirstOrder()
    )
    a1 == :states || return nothing
    _structure!(memo, e1, sv) == :block_diagonal || return nothing
    ctx = _ctx(sv)
    for en in entries
        haskey(ctx.shapes, en.u) == haskey(ctx.shapes, en.w) || return nothing
        c2 = col2_idx(en)
        isempty(c2) && continue
        length(c2) == length(en.band.ridx) || return nothing
        all(((c, r),) -> _offset_class(c, r) == :diag, zip(c2, en.band.ridx)) ||
            return nothing
    end
    return :block_diagonal
end

# ── emission ─────────────────────────────────────────────────────────────────

function serialize_band(en::JacEntry2)
    b = en.band
    d = OrderedDict{String, Any}(
        "row" => en.u,
        "col" => en.v,
        "col2" => en.w,
        "rows" => Any[Any[lo, hi] for (lo, hi) in b.rows],
        "row_idx" => Any[b.ridx...],
        "col_idx" => Any[_ser_expr(c) for c in col_idx(en)],
        "col2_idx" => Any[_ser_expr(c) for c in col2_idx(en)],
        "coef" => _ser_expr(b.coef),
    )
    isempty(b.contracted) ||
        (d["contracted"] = Any[Any[n, lo, hi] for (n, lo, hi) in b.contracted])
    b.guard === nothing || (d["where"] = _ser_expr(b.guard))
    return d
end

function parse_band2(d::AbstractDict)::JacEntry2
    en = parse_band(d)
    c2 = ASTExpr[expression_from_json(c) for c in d["col2_idx"]]
    b = en.band
    return JacEntry2(
        en.u, en.v, String(d["col2"]),
        _rebuild(b; cidx = vcat(b.cidx, c2)), length(b.cidx)
    )
end

_with_band(en::JacEntry2, b::Band) = JacEntry2(en.u, en.v, en.w, b, en.ncol)

# The `"jacobians"` block of one axis pair (esm-jacobian-spec.md §4.4): no
# factorization, `structure` only when the contraction is lane-local, and the
# second axis's `columns` when it is a parameter or forcing axis.
function _jacobian_block(
        sv::SysView, wrt::Tuple{Symbol, Symbol}, memo::_FirstOrder = _FirstOrder();
        forcing = nothing, cse::Bool = true, cse_min_nodes::Int = 12,
        column_axis = nothing       # a pair block has no structure class to refine
    )
    a1, a2 = wrt
    e1, entries = _bands2(sv, a1, a2, forcing, memo)
    return _block(
        Any[String(a1), String(a2)], entries,
        a2 in (:parameters, :forcing) ? _columns_field(sv, a2, forcing) : nothing,
        _structure2(e1, entries, sv, a1, memo), cse, cse_min_nodes
    )
end

# ── evaluation: the contraction kernel ──────────────────────────────────────

"""
    HVPEvaluator

Prepared second-order contraction for one axis pair: call
[`hvp!`](@ref)`(g, hv, pairs, u, p, t)`. Fields of note: `entries`,
`rownames` (μ's index space, the states), `colnames` (w's: the states, or
`["t"]` on a time pair), `col2names` (g's: the second axis, in its canonical
order), `columns` (the second axis's column space on `:parameters` /
`:forcing`), `wrt`.
"""
struct HVPEvaluator{F, P}
    fJ!::F
    oop::Bool
    uj::Vector{Float64}
    duj::Vector{Float64}
    umap::Vector{Int}
    p_default::P
    items::Vector{NTuple{4, Int}}  # (slot in duj, row, col, col2), one per point
    rc::Vector{Tuple{Int, Int}}    # the distinct (row, col) the items read M at
    rcidx::Vector{Int}             # item k reads M[rcidx[k]]
    M::Vector{Float64}             # scratch: M[q] = Σᵢ μᵢ[row] wᵢ[col]
    entries::Vector{JacEntry2}
    rownames::Vector{String}
    colnames::Vector{String}
    col2names::Vector{String}
    columns::Union{Nothing, Vector{Tuple{String, Vector{Int}}}}
    structure::Union{Nothing, Symbol}
    wrt::Tuple{Symbol, Symbol}
end

"""
    prepare_hvp(input; wrt = (:states, :states), model_name = nothing, forcing = nothing,
                build_kwargs = NamedTuple(), source_layout = nothing) -> HVPEvaluator

Build the second-order contraction once ([`jacobian_bands2`](@ref)): each
second-order coefficient field becomes a state of the derived evaluation model
(the same mechanism as [`prepare_jacobian`](@ref), whose keywords and
`source_layout` contract this shares), and a precomputed
`(field slot, row, col, col2)` list drives the contraction. A second column
outside its variable's extent is a ghost read on the state and parameter
axes and an error on the forcing axis, as for the first-order columns.
"""
function prepare_hvp(
        input; wrt = (:states, :states), model_name = nothing, forcing = nothing,
        build_kwargs = NamedTuple(), source_layout = nothing,
        eval_cse::Bool = true, eval_cse_min_nodes::Int = 12
    )
    a1, a2 = _axis_pair(wrt)
    sv, src = _sv_src(input, model_name)
    e1, entries = _bands2(sv, a1, a2, forcing)
    # The derived model only reads rows, row/contracted indices and the
    # coefficient, so a second-order entry rides through as a first-order one.
    doc, _ = _evaluation_document(
        sv, JacEntry[JacEntry(en.u, en.v, en.band) for en in entries];
        cse = eval_cse, cse_min_nodes = eval_cse_min_nodes
    )
    fJ!, u0j, pj, _, vmj = build_evaluator(doc; model_name = "JacobianEval", build_kwargs...)
    n, _, vm0 = _source_layout(src, model_name, build_kwargs, source_layout, a1)
    columns = _axis_columns(sv, a2, forcing)
    colslot, _, colnames = _colmap(vm0, nothing, a1)
    col2slot, _, col2names = _colmap(vm0, columns, a2)
    umap = zeros(Int, n)
    for (name, i) in vm0
        umap[i] = _slot(vmj, name, "band")
    end
    strict = source_layout !== nothing
    items = NTuple{4, Int}[]
    for (k, en) in enumerate(entries)
        _foreach_point(en.band) do cell, cc, cidx
            rowname = _cellname(en.u, cell)
            n2 = _cellname(en.w, cidx[(en.ncol + 1):end])
            n1 = _cellname(en.v, cidx[1:(en.ncol)])
            r = _rowslot(vm0, rowname, colslot(n1) * col2slot(n2), strict)  # a wrong layout first
            c2 = _colslot(col2slot, n2, rowname, a2)
            c1 = _colslot(colslot, n1, rowname, a1)
            (c1 == 0 || c2 == 0) && return
            push!(items, (_slot(vmj, _cellname("J_$k", vcat(cell, cc)), "band"), r, c1, c2))
        end
    end
    lut = Dict{Tuple{Int, Int}, Int}()
    rc = Tuple{Int, Int}[]
    rcidx = Int[get!(() -> (push!(rc, (it[2], it[3])); length(rc)), lut, (it[2], it[3])) for it in items]
    oop = !applicable(fJ!, zeros(1), zeros(1), pj, 0.0) &&
        applicable(fJ!, zeros(1), pj, 0.0)
    return HVPEvaluator(
        fJ!, oop, zeros(length(u0j)), zeros(length(u0j)), umap, pj,
        items, rc, rcidx, zeros(length(rc)), entries,
        first.(sort(collect(vm0), by = last)), colnames, col2names, columns,
        _structure2(e1, entries, sv, a1), (a1, a2)
    )
end

# One `(μ, w)` pair is a Tuple whose first element is a vector; anything else
# is a collection of pairs.
_stage_pairs(x::Tuple{AbstractVector, Any}) = (x,)
_stage_pairs(x) = x
_wat(w::Number, c::Int) = (c == 1 || throw(BoundsError(w, c)); w)
_wat(w::AbstractVector, c::Int) = w[c]

"""
    hvp!(g, hv::HVPEvaluator, pairs, u, p, t; form = :contract) -> g

The second-order contraction at `(u, p, t)`:

    g += ∇_{a2} Σᵢ μᵢᵀ J(u) wᵢ          (pair (:states, a2))
    g += ∇_{a2} (Σᵢ wᵢ μᵢ)ᵀ f_t         (pair (:time, a2); each wᵢ a scalar)

with `J = ∂f/∂(a1)`. `pairs` is one `(μ, w)` tuple or a collection of them
(one per Rosenbrock stage); `μ` is indexed like `hv.rownames`, `w` like
`hv.colnames` and `g` like `hv.col2names`. `g` ACCUMULATES — zero it first
for a single product.

`form = :contract` (default) sums the stages into `M(row, col)` once and runs
`g[col2] += coef2 · M(row, col)` over the points. On a `(:states, :states)`
pair `form = :symmetric` uses the symmetry of `∇²(μᵀf)` instead,
`g[col] += μ[row] · coef2 · w[col2]` per stage, which needs no `M`.
"""
function hvp!(
        g::AbstractVector, hv::HVPEvaluator, pairs, u::AbstractVector, p, t::Real;
        form::Symbol = :contract
    )
    length(g) == length(hv.col2names) || throw(
        DimensionMismatch("g has length $(length(g)), the second axis $(length(hv.col2names))")
    )
    ps = _stage_pairs(pairs)
    for (μ, w) in ps
        length(μ) == length(hv.rownames) || throw(
            DimensionMismatch("μ has length $(length(μ)), the row space $(length(hv.rownames))")
        )
        length(w) == length(hv.colnames) || throw(
            DimensionMismatch("w has length $(length(w)), the column space $(length(hv.colnames))")
        )
    end
    duj = _eval_fields!(hv, u, p, t)
    if form == :contract
        M = hv.M
        fill!(M, 0.0)
        for (μ, w) in ps, (q, (r, c)) in enumerate(hv.rc)
            M[q] += μ[r] * _wat(w, c)
        end
        for (k, (slot, _, _, c2)) in enumerate(hv.items)
            g[c2] += duj[slot] * M[hv.rcidx[k]]
        end
    elseif form == :symmetric
        hv.wrt == (:states, :states) || throw(
            ArgumentError("form = :symmetric needs a (:states, :states) pair, got $(hv.wrt)")
        )
        for (μ, w) in ps, (slot, r, c, c2) in hv.items
            g[c] += μ[r] * duj[slot] * w[c2]
        end
    else
        throw(ArgumentError("form must be :contract or :symmetric, got :$form"))
    end
    return g
end
