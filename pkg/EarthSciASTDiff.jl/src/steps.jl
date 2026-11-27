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
band)`, `v` a target), each coefficient (full mode) a read of the band's
named intermediate; `letter` the axis letter of the intermediates' names
(`u`, `p`, `f`); `inters` every intermediate named so far, name → (observed,
definition, 1: declared without a shape); `taken` the names an
intermediate's name must be clear of.
"""
struct _Steps
    letter::String
    taken::Set{String}
    inters::Dict{String, Tuple{String, ASTExpr, Int}}
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
function _steps_setup(
        sv::SysView, ctx::Ctx, obs::Dict{String, ASTExpr}, targets, scalars, co::Bool;
        letter::String = "u"
    )
    taken = Set{String}(String(n) for n in keys(sv.variables))
    union!(taken, String(n) for n in keys(sv.index_sets))
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
        letter, taken, Dict{String, Tuple{String, ASTExpr, Int}}(), tset, cols, Dict{String, ASTExpr}(n => obs[n] for n in sub), obs, ctx2,
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
    # inlined, and the operands of a whole-array term aligned with `u` by
    # index-set name (`_inline_row`), as on every axis
    rhs = _inline_row(rhs, u, inl, ctx; rank_obs = st.obs, scalars = st.scalars)
    bycol = Dict{String, Vector{Band}}()
    for v in sort!(String[x for x in free_variables(rhs) if x in st.targets || x in st.cols])
        bs = Band[normalize_band(b) for b in merge_bands(_ref_bands(rhs, v, ctx, shape_u, st.co))]
        if v in st.targets
            append!(get!(bycol, v, Band[]), bs)
            continue
        end
        _check_cell_read(bs, v, ctx)
        inner = _steps_composed!(st, v)
        for w in sort!(unique(String[e.v for e in inner]))
            ib = Band[e.band for e in inner if e.v == w]
            for x in _compose(bs, ib, w, ctx, st.co)
                push!(get!(bycol, w, Band[]), _steps_sum_free(x, st.co))
            end
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

# A kept observed `v` is read with one index per dimension of its shape: a
# read of another arity selects no cell of it, refused as inlining refuses
# it (`_gather_cell`).
function _check_cell_read(bs::Vector{Band}, v::String, ctx::Ctx)
    sh = get(ctx.shapes, v, nothing)
    sh === nothing && return nothing
    for b in bs
        length(b.cidx) == length(sh) || throw(
            InlineError(
                "observed `$v` of rank $(length(sh)) is read at a cell with " *
                    "$(length(b.cidx)) indices"
            )
        )
    end
    return nothing
end

# A composite's contracted names that its column indices do not read are
# summed into its coefficient: a scalar `faq` (reduce `+`) over their ranges
# whose filter is the `where` conjuncts that read them (in order); the other
# conjuncts stay the band's `where`. The band keeps only the contracted
# dimensions a column index reads, as a band read off an inlined right-hand
# side does (§3.5), so the structure passes see the same form.
function _steps_sum_free(b::Band, co::Bool)
    isempty(b.contracted) && return b
    read = Set{String}(_join_names(b))      # a joined name is never summed away
    for c in b.cidx
        union!(read, free_variables(c))
    end
    free = Set{String}(n for (n, _, _) in b.contracted if !(n in read))
    isempty(free) && return b
    keep = ASTExpr[]; filt = ASTExpr[]
    if b.guard !== nothing
        for c in _conjuncts(b.guard)
            push!(isdisjoint(free_variables(c), free) ? keep : filt, c)
        end
    end
    con = Tuple{String, Int, Int}[x for x in b.contracted if !(x[1] in free)]
    coef = co ? OpExpr(
            "faq", ASTExpr[];
            output_idx = Any[], reduce = "+", expr_body = b.coef,
            ranges = Dict{String, Any}(n => Any[lo, hi] for (n, lo, hi) in b.contracted if n in free),
            filter = isempty(filt) ? nothing : _conj(filt)
        ) : nothing
    return normalize_band(
        Band(b.rows, b.ridx, b.cidx, coef, con, isempty(keep) ? nothing : _conj(keep), b.join)
    )
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
            b.guard === nothing ? nothing : skey(b.guard), b.join,
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
    if def isa OpExpr && def.op == "faq" && def.join !== nothing &&
            rhs isa OpExpr && Any[rhs.output_idx...] == Any[def.output_idx...]
        # a joined definition over its own row names is its row (a joined
        # node is never read through β-reduction, §3.5)
        rhs = def
    else
        inl[o] = def
    end
    out = try
        _steps_row(st, o, rhs, inl, shape)
    finally
        delete!(st.busy, o)
    end
    st.co && (out = _steps_name!(st, o, shape, out))
    st.composed[o] = out
    return out
end

# ── named intermediates (full mode) ──────────────────────────────────────────

# The composed bands of observed `o` in their canonical order (the code-point
# order of each band's structure-only entry written as stable JSON, §2.1,
# §5), each coefficient that is not a literal written once as a named
# intermediate `_d<letter>_<o>_<k>` (k its 1-based position) and replaced by
# a read of it at the band's row and contracted names. One intermediate per
# BAND, never one per (column, column index): two bands of one column index
# may overlap with different `where`, and a summed definition read by each
# would count the overlap twice.
function _steps_name!(st::_Steps, o::String, shape::Vector{Int}, bands::Vector{JacEntry})
    keyof(en) = stable_json(serialize_band(JacEntry(en.u, en.v, _rebuild(en.band; coef = nothing))))
    sorted = sort(bands; by = keyof)
    out = JacEntry[]
    for (k, en) in enumerate(sorted)
        b = en.band
        if b.coef isa IntExpr || b.coef isa NumExpr
            push!(out, en)
            continue
        end
        name = _inter_name!(st, o, k)
        dims = _inter_dims(b, shape)
        st.inters[name] = (o, _inter_definition(b, dims), 1)
        push!(out, JacEntry(en.u, en.v, _rebuild(b; coef = _inter_read(name, dims))))
    end
    return out
end

# `_d<letter>_<o>_<k>`; for a scoped `P.v`, `P._d<letter>_<v>_<k>` (in `o`'s
# component); `_` appended until clear of every variable and index-set name
# of the source.
function _inter_name!(st::_Steps, o::String, k::Int)
    i = findlast('.', o)
    base = i === nothing ? "_d$(st.letter)_$(o)_$k" :
        "$(o[1:prevind(o, i)])._d$(st.letter)_$(o[nextind(o, i):end])_$k"
    return _fresh!(st.taken, base, "_")
end

# The dimensions of a band's intermediate: the band's row names (in row
# order) and then contracted names that its coefficient or `where` reads,
# each with its box and its whole extent (a row dimension's extent is the
# observed's, a contracted one's its upper bound). Only those: an
# intermediate depends on what its coefficient depends on, so a read of it
# carries no dependence the coefficient lacks (a rank-one term still splits,
# §6) and it holds no more values than it varies over.
function _inter_dims(b::Band, shape::Vector{Int})
    fv = free_variables(b.coef)
    b.guard === nothing || union!(fv, free_variables(b.guard))
    dims = Tuple{String, Tuple{Int, Int}, Int}[]
    for (d, r) in enumerate(b.ridx)
        r isa String && r in fv && push!(dims, (r, b.rows[d], shape[d]))
    end
    for (n, lo, hi) in b.contracted
        n in fv && push!(dims, (n, (lo, hi), hi))
    end
    return dims
end

# The read of intermediate `name` at its dimensions' names; the bare name
# when it has none.
_inter_read(name::String, dims) =
    isempty(dims) ? VarExpr(name) : op("index", VarExpr(name), ASTExpr[VarExpr(n) for (n, _, _) in dims]...)

# The definition of a band's intermediate over its dimensions (`dims`): the
# coefficient, gated by the band's `where`. A band whose box is the whole
# extent on every dimension is one `faq`; one whose box is narrower is the
# `makearray` of zero over the whole extent and the box over it, read at each
# cell of the extent (`_region_array`), so a reader never needs a gate.
function _inter_definition(b::Band, dims)
    body = b.guard === nothing ? b.coef : op("ifelse", b.guard, b.coef, lit(0))
    isempty(dims) && return body
    names = String[n for (n, _, _) in dims]
    box = Tuple{Int, Int}[bx for (_, bx, _) in dims]
    full = Int[n for (_, _, n) in dims]
    faq(bx) = OpExpr(
        "faq", ASTExpr[];
        output_idx = Any[names...], expr_body = body,
        ranges = Dict{String, Any}(n => Any[lo, hi] for (n, (lo, hi)) in zip(names, bx))
    )
    all(((bx, n),) -> bx[1] == 1 && bx[2] == n, zip(box, full)) && return faq(box)
    return _region_array(
        Vector{Vector{Any}}[
            Vector{Any}[Any[1, n] for n in full], Vector{Any}[Any[lo, hi] for (lo, hi) in box],
        ],
        ASTExpr[lit(0), faq(box)], full
    )
end

# ── one pass per (view, axis, forcing set), shared ───────────────────────────

# The axis letter of an intermediate's name.
const _STEP_LETTER = Dict(:states => "u", :parameters => "p", :forcing => "f")

# The named-steps layouts already built, keyed by the view, the axis, the
# forcing set and the mode, so that the derived documents of a block read the
# very intermediates its pass named. A few views at a time.
const _STEP_CACHE = Pair{Any, _Steps}[]

function _steps_pass(sv::SysView, ctx::Ctx, src_obs, recs, wrt::Symbol, forcing, targets, scalars, co::Bool)
    fl = forcing === nothing ? nothing :
        forcing isa AbstractString ? [String(forcing)] : sort!(String[String(x) for x in forcing])
    key = (wrt, fl, co)
    for (k, st) in _STEP_CACHE
        k[1] === sv && k[2] == key && return st
    end
    rn = Set{String}(r.name for r in recs)
    st = _steps_setup(
        sv, ctx, Dict{String, ASTExpr}(k => v for (k, v) in src_obs if !(k in rn)),
        targets, scalars, co; letter = _STEP_LETTER[wrt]
    )
    push!(_STEP_CACHE, (sv, key) => st)
    length(_STEP_CACHE) > 8 && popfirst!(_STEP_CACHE)
    return st
end

# ── the intermediates a band list reads ──────────────────────────────────────

const _INTER_RE = r"^_d([upf])_(.*)_([0-9]+)_*$"

# The axis and observed of intermediate name `n`, or `nothing`.
# The observed a companion name `_dt_<o>` (`P._dt_<v>` for `P.v`) is of, or
# `nothing` for another name.
function _dt_observed(n::AbstractString)
    i = findlast('.', n)
    scope, base = i === nothing ? ("", n) : (n[1:i], n[nextind(n, i):end])
    startswith(base, "_dt_") || return nothing
    return scope * base[5:end]
end

function _inter_parse(n::AbstractString)
    i = findlast('.', n)
    scope, base = i === nothing ? ("", n) : (n[1:i], n[nextind(n, i):end])
    m = match(_INTER_RE, base)
    m === nothing && return nothing
    wrt = Dict("u" => :states, "p" => :parameters, "f" => :forcing)[m.captures[1]]
    return wrt, scope * m.captures[2]
end

"""
    _intermediates(sv, entries; forcing = nothing, info = nothing) -> Vector{Tuple{String, String, ASTExpr, Int}}

Every named intermediate the coefficients and `where` of `entries` read,
directly or through another intermediate's definition, as `(name, observed,
definition, 0 for a time companion (declared with the observed's shape) or 1
for a named step (declared without one))`, in the code-point order of the
observed, then of the name: the time companions (§4.2) and the named steps
of the other axes (§4.6). What a block's `intermediates` field and every
derived document of the list carry.
"""
function _intermediates(sv::SysView, entries; forcing = nothing, info = nothing)
    work = String[]
    for en in entries, x in (en.band.coef, en.band.guard)
        x === nothing && continue
        for n in free_variables(x)
            _inter_parse(n) === nothing || push!(work, n)
            # a time pair's companion of a named intermediate (§4.4)
            o = _dt_observed(n)
            o === nothing || _inter_parse(o) === nothing || push!(work, o)
        end
    end
    if isempty(work)
        return Tuple{String, String, ASTExpr, Int}[
            (n, o, d, 0) for (n, o, d) in _time_companions(sv, entries; info = info)
        ]
    end
    ctx = _ctx(sv)
    obs = _observed_defs(sv)
    recs = recurrences(sv, obs, ctx)
    scalars = _declared_scalars(sv, obs)
    defs = Dict{String, Tuple{String, ASTExpr, Int}}()
    # A pair's second-axis step of a first-axis intermediate (§4.4) is named
    # after that intermediate (`_dp__du_o_1_1`): the intermediates whose
    # observed is a source observed are resolved first, then those over them,
    # each level's passes taking the levels below as observed.
    level(n) = (o = _inter_parse(n)[2]; _inter_parse(o) === nothing ? 0 : 1 + level(o))
    passes = Dict{Tuple{Symbol, Int}, _Steps}()
    # a nested name's own observed is an intermediate of the level below
    function chain!(acc, n)
        push!(acc, n)
        o = _inter_parse(n)[2]
        _inter_parse(o) === nothing || chain!(acc, o)
        return acc
    end
    pending = String[]
    foreach(n -> chain!(pending, n), work)
    while !isempty(pending)
        lv = minimum(level, pending)
        # passes over this level's names are rebuilt with the levels below
        filter!(kv -> kv[1][2] <= lv, passes)
        here = String[n for n in pending if level(n) == lv]
        filter!(n -> level(n) != lv, pending)
        while !isempty(here)
            n = pop!(here)
            haskey(defs, n) && continue
            wrt, o = _inter_parse(n)
            st = get!(passes, (wrt, lv)) do
                obs2 = Dict{String, ASTExpr}(k => v for (k, v) in obs if !(k in Set(r.name for r in recs)))
                lv == 0 || merge!(obs2, Dict{String, ASTExpr}(m => d for (m, (_, d, _)) in defs))
                lv == 0 ? _steps_pass(sv, ctx, obs, recs, wrt, forcing, _targets(sv, wrt, forcing, obs), scalars, true) :
                    _steps_setup(sv, ctx, obs2, _targets(sv, wrt, forcing, obs), scalars, true; letter = _STEP_LETTER[wrt])
            end
            haskey(st.inters, n) || _steps_composed!(st, o)
            hit = get(st.inters, n, nothing)
            hit === nothing && throw(BandError("no named intermediate `$n` (observed `$o`, $(wrt))"))
            defs[n] = hit
            for m in free_variables(hit[2])
                (_inter_parse(m) === nothing || haskey(defs, m)) && continue
                level(m) == lv ? push!(here, m) : chain!(pending, m)
            end
        end
    end
    extra = Dict{String, ASTExpr}(n => d for (n, (_, d, _)) in defs)
    out = Tuple{String, String, ASTExpr, Int}[
        (n, o, d, 0) for (n, o, d) in _time_companions(sv, entries; info = info, extra = extra)
    ]
    axis = sort!([(o, n, d, c) for (n, (o, d, c)) in defs])
    append!(out, Tuple{String, String, ASTExpr, Int}[(n, o, d, c) for (o, n, d, c) in axis])
    return sort!(out; by = x -> (x[2], x[1]))
end
