# Named steps on every axis (esm-jacobian-spec.md §4.6): the bands of a row
# are composed through the observed graph rather than read off the row with
# every observed inlined. Each observed that moves with the axis is a column
# of the band calculus; its own definition is differentiated once, and its
# bands onto the axis's targets are composed once and memoized, so the cost
# grows with the number of observed, not with the number of paths through
# them. Nothing is inlined and then deduplicated, and there is no threshold.
# (The time companions, companions.jl, are this mechanism's time case.)

"""
    _Steps

One pass's named-step layout: `targets` the axis's columns; `cols` the
observed that move with them and are columns too (those a column index does
not need inlined); `inline` the pass's inlining map (the observed a column
index reads); `ctx` the band context with every column's extents; `composed[o]`
the bands of observed `o` onto the targets once composed (`JacEntry(o, v,
band)`, `v` a target).
"""
struct _Steps
    targets::Set{String}
    cols::Set{String}
    inline::Dict{String, ASTExpr}
    obs::Dict{String, ASTExpr}
    ctx::Ctx
    scalars::Set{String}
    co::Bool
    composed::Dict{String, Vector{JacEntry}}
    busy::Set{String}
end

# The layout of a pass over `targets`. `obs` is every observed definition
# with the recurrences left out; `scalars` the declared scalars that are not
# observed.
function _steps_setup(sv::SysView, ctx::Ctx, obs::Dict{String, ASTExpr}, targets, scalars, co::Bool)
    tset = Set{String}(targets)
    moving = _moving_observed(obs, tset)
    sub = _subscript_observed!(Set{String}(), obs, sv.equations, moving)
    cols = setdiff(moving, sub)
    shapes = copy(ctx.shapes)
    memo = Dict{String, Vector{Int}}()
    for o in sort!(collect(cols))
        ext = _obs_extents(o, sv, obs, ctx, memo, Set{String}())
        isempty(ext) || (shapes[o] = ext)
    end
    ctx2 = Ctx(ctx.index_sets, shapes, ctx.params, ctx.refs, ctx.origins, ctx.axes)
    return _Steps(
        tset, cols, Dict{String, ASTExpr}(n => obs[n] for n in sub), obs, ctx2,
        Set{String}(scalars), co, Dict{String, Vector{JacEntry}}(), Set{String}()
    )
end

# The bands of one row (`rhs`, rows `u` of shape `shape_u`) onto the
# targets: its direct bands onto each target, and each direct band onto an
# observed column composed with that observed's composed bands. Columns in
# code-point order; within a column the direct bands, then the composites in
# the order of the observed they pass through; merged and normalized.
function _steps_row(st::_Steps, u::String, rhs::ASTExpr, inl, shape_u::Vector{Int})
    ctx = st.ctx
    rhs = _with_layer(ctx.refs) do
        inline_observed(
            rhs, inl; shapes = ctx.shapes, axes = ctx.axes, rank_obs = st.obs, scalars = st.scalars
        )
    end
    bycol = Dict{String, Vector{Band}}()
    for v in sort!(String[x for x in free_variables(rhs) if x in st.targets || x in st.cols])
        bs = Band[normalize_band(b) for b in merge_bands(_ref_bands(rhs, v, ctx, shape_u, st.co))]
        if v in st.targets
            append!(get!(bycol, v, Band[]), bs)
            continue
        end
        inner = _steps_composed!(st, v)
        for w in sort!(unique(String[e.v for e in inner]))
            ib = Band[e.band for e in inner if e.v == w]
            append!(get!(bycol, w, Band[]), _compose(bs, ib, w, ctx, st.co))
        end
    end
    out = JacEntry[]
    for w in sort!(collect(keys(bycol)))
        for b in merge_bands(_steps_combine(bycol[w], st.co))
            push!(out, JacEntry(u, w, normalize_band(b)))
        end
    end
    return out
end

# Bands of one column with the same structure (rows, row names, column
# indices, contracted dimensions and `where`), reached by different paths
# through the observed graph, combined into the first of them: their
# coefficients summed (an n-ary `+`, in order), or, structure-only, kept
# once. This is what keeps a row's bands linear in the observed rather than
# in the paths.
function _steps_combine(bs::Vector{Band}, co::Bool)
    keys_ = Any[]
    groups = Dict{Any, Vector{Int}}()
    for (i, b) in enumerate(bs)
        k = (
            b.rows, b.ridx, String[skey(c) for c in b.cidx], b.contracted,
            b.guard === nothing ? nothing : skey(b.guard),
        )
        haskey(groups, k) || push!(keys_, k)
        push!(get!(groups, k, Int[]), i)
    end
    out = Band[]
    for k in keys_
        idx = groups[k]
        b = bs[idx[1]]
        (length(idx) == 1 || !co) && (push!(out, b); continue)
        push!(out, _rebuild(b; coef = op("+", ASTExpr[bs[i].coef for i in idx]...)))
    end
    return out
end

# The composed bands of observed `o` onto the targets, computed once: the
# bands of `o`'s own row (its definition over its shape, as a companion row
# is written, §4.2) by `_steps_row`.
function _steps_composed!(st::_Steps, o::String)::Vector{JacEntry}
    hit = get(st.composed, o, nothing)
    hit === nothing || return hit
    o in st.busy && throw(BandError("cyclic observed definition through `$o`"))
    push!(st.busy, o)
    shape = get(st.ctx.shapes, o, Int[])
    def = st.obs[o]
    rhs = if isempty(shape)
        VarExpr(o)
    else
        names = _row_names_of(def, length(shape))
        OpExpr(
            "faq", ASTExpr[];
            output_idx = Any[names...],
            expr_body = op("index", VarExpr(o), ASTExpr[VarExpr(n) for n in names]...),
            ranges = Dict{String, Any}(n => Any[1, shape[d]] for (d, n) in enumerate(names))
        )
    end
    inl = copy(st.inline)
    inl[o] = def
    out = try
        _steps_row(st, o, rhs, inl, shape)
    finally
        delete!(st.busy, o)
    end
    st.composed[o] = out
    return out
end
