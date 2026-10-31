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
