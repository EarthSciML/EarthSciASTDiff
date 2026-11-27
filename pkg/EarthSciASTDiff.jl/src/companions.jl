# Time companions (esm-jacobian-spec.md §4.2): the time bands keep the chain
# through the observed that move with `t` as named steps. Each such observed
# `o` gets ONE derived observed, its companion `_dt_<o>` = ∂o/∂t (data held
# fixed), and every band reads it through the chain rule instead of
# differentiating `o`'s definition again inside every row:
#
#     a read of o at column index φ, with ∂f/∂o = c there   ↦   c · _dt_o[φ]
#
# A companion's own definition is built the same way from `o`'s definition:
# its direct derivative in `t` plus, for each moving observed `o′` it reads,
# `∂def_o/∂o′ · _dt_o′`. Nothing is inlined and then deduplicated: the moving
# observed are extra COLUMNS of the band calculus (as a recurrence is, §4.5),
# so each definition is differentiated once.

"""
    _Companions

The time pass's companion layout: `names` maps every observed that moves with
`t` to its companion's name (assigned over all of them, so a name never
depends on the pass); `cols` is the set that are columns of this pass (those
a column index does not need inlined); `inline` the pass's inlining map (the
observed a column index reads); `ctx` the band context with every column's
extents; `bands[o]` the companion's time bands once computed (empty: `∂o/∂t
≡ 0`, and a read of `o` contributes nothing).
"""
struct _Companions
    names::Dict{String, String}
    cols::Set{String}
    inline::Dict{String, ASTExpr}
    obs::Dict{String, ASTExpr}           # the observed definitions, recurrences left out
    rank_obs::Dict{String, ASTExpr}      # every observed definition (ranks, extents)
    ctx::Ctx
    scalars::Set{String}
    co::Bool
    bands::Dict{String, Vector{Band}}
    pending::Set{String}                 # companions whose bands are being built
end

# The companion name of each observed in `moving`, in name order: `_dt_<o>`
# (`P._dt_<v>` for a scoped `P.v`, so it lives in `o`'s own component), with
# `_` appended until clear of every variable, component path and index-set
# name of the source and of the companion names before it.
function _companion_names(sv::SysView, moving)::Dict{String, String}
    taken = Set{String}(String(n) for n in keys(sv.variables))
    union!(taken, String(n) for n in keys(sv.index_sets))
    # a companion named like a component path would be both a variable and a
    # subsystem of a derived model
    _component_paths!(taken, keys(sv.variables))
    out = Dict{String, String}()
    for o in sort!(collect(String, moving))
        i = findlast('.', o)
        base = i === nothing ? "_dt_$o" :
            "$(o[1:prevind(o, i)])._dt_$(o[nextind(o, i):end])"
        out[o] = _fresh!(taken, base, "_")
    end
    return out
end

# Every component path of the scoped names `names` (`A` and `A.B` of
# `A.B.v`), added to `taken`.
function _component_paths!(taken::Set{String}, names)
    for n in names
        n = String(n)
        for (i, ch) in pairs(n)
            ch == '.' && push!(taken, n[1:prevind(n, i)])
        end
    end
    return taken
end

# The companion layout of the time pass over the view. `obs` is every
# observed definition with the recurrences left out (the pass never inlines a
# recurrence), `rank_obs` every observed definition, recurrences included
# (what an elementwise read at a cell and an undeclared shape are ranked
# from); `scalars` the declared scalars that are not observed.
function _companion_setup(
        sv::SysView, ctx::Ctx, obs::Dict{String, ASTExpr}, scalars, co::Bool;
        rank_obs::Dict{String, ASTExpr} = obs
    )
    moving = _moving_observed(obs, Set{String}(["t"]))
    sub = _subscript_observed!(Set{String}(), obs, sv.equations, moving)
    cols = setdiff(moving, sub)
    shapes = copy(ctx.shapes)
    memo = Dict{String, Vector{Int}}()
    for o in sort!(collect(cols))
        ext = _obs_extents(o, sv, rank_obs, ctx, memo, Set{String}())
        isempty(ext) || (shapes[o] = ext)
    end
    ctx2 = Ctx(ctx.index_sets, shapes, ctx.params, ctx.refs, ctx.origins, ctx.axes)
    return _Companions(
        _companion_names(sv, moving), cols,
        Dict{String, ASTExpr}(n => obs[n] for n in sub), obs, rank_obs, ctx2,
        Set{String}(scalars), co, Dict{String, Vector{Band}}(), Set{String}()
    )
end

# The extents of observed `o`: declared, or inferred from its definition —
# a `faq`'s output ranges, a `makearray`'s regions, an array-valued
# reference's body extents, or the one extent list every array operand of an
# elementwise definition shares. A definition whose extents cannot be read at
# build time (operands of different extents, whose union needs index-set
# names; a partial `index` read) is refused: its companion has no shape.
function _obs_extents(o::String, sv::SysView, obs, ctx::Ctx, memo, seen::Set{String})::Vector{Int}
    haskey(ctx.shapes, o) && return ctx.shapes[o]
    var = get(sv.variables, o, nothing)
    # a named intermediate (a time pair's `a2` steps, §4.4) is no variable of
    # the view: declared without a shape, its extents are its definition's
    var === nothing && !haskey(obs, o) && return Int[]
    var === nothing || var.shape === nothing || return Int[]        # declared `[]`: a scalar
    haskey(obs, o) || return Int[]
    haskey(memo, o) && return memo[o]
    o in seen && throw(BandError("cyclic observed definition through `$o`"))
    push!(seen, o)
    ext = _expr_extents(obs[o], o, sv, obs, ctx, memo, seen)
    memo[o] = ext
    return ext
end

function _expr_extents(e::ASTExpr, o::String, sv::SysView, obs, ctx::Ctx, memo, seen)::Vector{Int}
    e isa VarExpr && return _obs_extents(e.name, sv, obs, ctx, memo, seen)
    e isa OpExpr || return Int[]
    refuse() = throw(
        BandError(
            "the extents of observed `$o` (declared without a shape) cannot be read " *
                "from its definition; declare its shape"
        )
    )
    if e.op == "faq"
        out = Int[]
        for x in something(e.output_idx, Any[])
            x isa AbstractString || refuse()
            r = get(something(e.ranges, Dict{String, Any}()), String(x), nothing)
            r === nothing && refuse()
            lo, hi = _range_of(r, ctx)
            lo == 1 || refuse()
            push!(out, hi)
        end
        return out
    elseif e.op == "makearray"
        (e.regions === nothing || isempty(e.regions)) && refuse()
        out = zeros(Int, length(e.regions[1]))
        for region in e.regions, (d, r) in enumerate(region)
            out[d] = max(out[d], _region_bound(r[2]))
        end
        return out
    elseif e.op == "index"
        # a read at a cell is a scalar; a read with fewer indices than its
        # array's rank is an array slice, whose extents are not read here
        a = e.args[1]
        a isa VarExpr || return Int[]
        ext = _obs_extents(a.name, sv, obs, ctx, memo, seen)
        length(e.args) - 1 < length(ext) && refuse()
        return Int[]
    elseif _is_ref(e)
        L = ctx.refs
        L === nothing && return Int[]
        D = _with_layer(() -> _def!(L, e.name), L)
        D.rank == 0 && return Int[]
        ext, org = try
            _with_layer(() -> _ref_extents(L, D, something(e.bindings, Dict{String, ASTExpr}()), ctx), L)
        catch err
            err isa BandError || rethrow()
            refuse()
        end
        all(==(1), org) || refuse()
        return ext
    end
    out = Int[]
    for a in e.args
        ext = _expr_extents(a, o, sv, obs, ctx, memo, seen)
        isempty(ext) && continue
        isempty(out) || out == ext || refuse()
        out = ext
    end
    return out
end

# The time bands of companion `o` (row `o` over its shape), computed once.
function _companion_bands!(cs::_Companions, o::String)::Vector{Band}
    haskey(cs.bands, o) && return cs.bands[o]
    # moving observed that read each other (an algebraic loop that is not a
    # recognized recurrence) have no companion: refused, never followed
    o in cs.pending && throw(BandError("cyclic observed definition through `$o`"))
    push!(cs.pending, o)
    shape = get(cs.ctx.shapes, o, Int[])
    def = cs.obs[o]
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
    inl = copy(cs.inline)
    inl[o] = def
    bs = _time_row_bands(cs, rhs, o, inl, shape)
    delete!(cs.pending, o)
    cs.bands[o] = bs
    return bs
end

# The row index names of a companion row: the definition's own output
# indices when it is a `faq` naming one per dimension, else `_r<d>` clear of
# every name of the definition.
function _row_names_of(def::ASTExpr, k::Int)::Vector{String}
    if def isa OpExpr && def.op == "faq"
        oi = something(def.output_idx, Any[])
        if length(oi) == k && all(x -> x isa AbstractString, oi) && allunique(oi)
            return String[String(x) for x in oi]
        end
    end
    taken = _all_names!(Set{String}(), def)
    return String[_fresh!(taken, "_r$d", "_") for d in 1:k]
end

# The time bands of one row (a state equation's right-hand side `rhs`, or a
# companion row), all with column `t`: the direct bands in `t` and each band
# of a moving-observed column composed with that observed's companion, summed
# where they share a box (`_sum_same_box`), in column-name order. `u` is the
# row's variable (the state, or the companion's observed): an elementwise
# observed read at a cell and the operands of a whole-array term are aligned
# with it by index-set name (`_inline_row`).
function _time_row_bands(cs::_Companions, rhs::ASTExpr, u::String, inl, shape_u::Vector{Int})
    ctx = cs.ctx
    rhs = _inline_row(rhs, u, inl, ctx; rank_obs = cs.rank_obs, scalars = cs.scalars)
    out = Band[]
    for v in sort!(String[x for x in free_variables(rhs) if x == "t" || x in cs.cols])
        bs = Band[normalize_band(b) for b in merge_bands(_ref_bands(rhs, v, ctx, shape_u, cs.co))]
        if v == "t"
            append!(out, bs)
            continue
        end
        isempty(_companion_bands!(cs, v)) && continue
        sv_shape = get(ctx.shapes, v, Int[])
        for b in bs
            push!(out, _through_companion(b, v, cs.names[v], sv_shape, cs.co))
        end
    end
    return Band[normalize_band(b) for b in merge_bands(_sum_same_box(out, cs.co))]
end

# Band `b` (column: a moving observed of shape `shape`, read at `b.cidx`)
# composed with that observed's companion `name`: column `t`, coefficient
# `coef · name[cidx…]` (the bare name for a scalar). A column index not
# provably inside the observed's extent over the band's box guards the term
# by `1 <= φ_d <= n_d` (a read the source never evaluates contributes
# nothing). Contracted dimensions become a scalar `faq` sum over their
# ranges, the band's `where` its filter, so a time band keeps the row's
# shape.
function _through_companion(b::Band, o::String, name::String, shape::Vector{Int}, co::Bool)
    _check_read_rank(b, o, shape)
    b.join === nothing || throw(
        BandError("a time companion read through a joined faq node is not supported")
    )
    if !co
        return Band(
            b.rows, b.ridx, ASTExpr[], nothing, Tuple{String, Int, Int}[],
            isempty(b.contracted) ? b.guard : nothing
        )
    end
    read = isempty(b.cidx) ? VarExpr(name) : op("index", VarExpr(name), b.cidx...)
    term = _simp(mul(b.coef, read))
    box = _index_box(b)
    conds = ASTExpr[]
    for (d, c) in enumerate(b.cidx)
        bd = _index_bounds(c, box)
        bd !== nothing && bd[1] >= 1 && bd[2] <= shape[d] && continue
        push!(conds, op(">=", c, lit(1)))
        push!(conds, op("<=", c, lit(shape[d])))
    end
    isempty(conds) || (term = op("ifelse", _conj(conds), term, lit(0)))
    isempty(b.contracted) && return Band(b.rows, b.ridx, ASTExpr[], term, Tuple{String, Int, Int}[], b.guard)
    sum = OpExpr(
        "faq", ASTExpr[];
        output_idx = Any[], reduce = "+", expr_body = term,
        ranges = Dict{String, Any}(n => Any[lo, hi] for (n, lo, hi) in b.contracted),
        filter = b.guard
    )
    return Band(b.rows, b.ridx, ASTExpr[], sum, Tuple{String, Int, Int}[], nothing)
end

# A band's column index must read every dimension of the moving observed
# `o`, whose extents are `shape`: a read of another rank has no companion
# cell (an undeclared observed whose extents were read wrong, or a malformed
# read), refused rather than read out of bounds.
function _check_read_rank(b::Band, o::String, shape::Vector{Int})
    length(b.cidx) == length(shape) || throw(
        BandError(
            "observed `$o` of rank $(length(shape)) is read with $(length(b.cidx)) " *
                "indices; declare its shape"
        )
    )
    return
end

# Bands of one row with the same box (rows, row names and `where`) summed
# into one: an n-ary `+` of their coefficients in order, at the first one's
# position.
function _sum_same_box(bs::Vector{Band}, co::Bool)
    keys_ = Any[]
    groups = Dict{Any, Vector{Int}}()
    for (i, b) in enumerate(bs)
        k = (b.rows, b.ridx, b.guard === nothing ? nothing : skey(b.guard))
        haskey(groups, k) || push!(keys_, k)
        push!(get!(groups, k, Int[]), i)
    end
    out = Band[]
    for k in keys_
        idx = groups[k]
        b = bs[idx[1]]
        length(idx) == 1 && (push!(out, b); continue)
        coef = co ? op("+", ASTExpr[bs[i].coef for i in idx]...) : nothing
        push!(out, _rebuild(b; coef = coef))
    end
    return out
end

# A companion's definition from its time bands: the scalar sum of their
# terms, or one `faq` over the observed's whole shape whose body sums them,
# each band's term gated by its region's membership (and its `where`) unless
# it covers the whole shape. Row names are the bands' own when they all
# share them, else `_r<d>` clear of every name the bands hold.
function _companion_definition(bs::Vector{Band}, shape::Vector{Int})::ASTExpr
    term(b, conds) = isempty(conds) ? b.coef : op("ifelse", _conj(conds), b.coef, lit(0))
    if isempty(shape)
        ts = ASTExpr[term(b, b.guard === nothing ? ASTExpr[] : _conjuncts(b.guard)) for b in bs]
        return length(ts) == 1 ? ts[1] : op("+", ts...)
    end
    k = length(shape)
    first_r = bs[1].ridx
    names = if all(r -> r isa String, first_r) && allunique(first_r) &&
            all(b -> b.ridx == first_r, bs)
        String[String(r) for r in first_r]
    else
        taken = Set{String}()
        for b in bs
            union!(taken, String[r for r in b.ridx if r isa String])
            _all_names!(taken, b.coef)
            b.guard === nothing || _all_names!(taken, b.guard)
        end
        String[_fresh!(taken, "_r$d", "_") for d in 1:k]
    end
    # each band: its box over the shape and its term (row names unified,
    # gated by its `where` only)
    boxes = Vector{Tuple{Int, Int}}[]
    terms = ASTExpr[]
    for b in bs
        m = Dict{String, ASTExpr}()
        box = Tuple{Int, Int}[]
        for d in 1:k
            r = b.ridx[d]
            if r isa String
                r == names[d] || (m[r] = VarExpr(names[d]))
                push!(box, b.rows[d])
            else
                push!(box, (r, r))
            end
        end
        coef = isempty(m) ? b.coef : _subst_free(b.coef, m)
        conds = b.guard === nothing ? ASTExpr[] :
            _conjuncts(isempty(m) ? b.guard : _subst_free(b.guard, m))
        push!(boxes, box)
        push!(terms, isempty(conds) ? coef : op("ifelse", _conj(conds), coef, lit(0)))
    end
    sumof(ts) = length(ts) == 1 ? ts[1] : op("+", ts...)
    faq(box, body) = OpExpr(
        "faq", ASTExpr[];
        output_idx = Any[names...], expr_body = body,
        ranges = Dict{String, Any}(n => Any[box[d][1], box[d][2]] for (d, n) in enumerate(names))
    )
    full = Tuple{Int, Int}[(1, n) for n in shape]
    all(==(full), boxes) && return faq(full, sumof(terms))
    # A region narrower than the shape: a `makearray`, never a whole-shape
    # gate (a reader built per cell, a lane program, needs the region). The
    # boxes are cut at every box's bounds into disjoint cells, each the sum
    # of the bands that cover it, over zero where none does.
    cuts = [sort!(unique!(vcat(1, shape[d] + 1, [bx[d][1] for bx in boxes], [bx[d][2] + 1 for bx in boxes]))) for d in 1:k]
    regions = Vector{Vector{Any}}[]
    values = ASTExpr[]
    ncells = 0
    for c in Iterators.product((1:(length(cuts[d]) - 1) for d in k:-1:1)...)
        cell = Tuple{Int, Int}[(cuts[d][c[k - d + 1]], cuts[d][c[k - d + 1] + 1] - 1) for d in 1:k]
        cover = [i for (i, bx) in enumerate(boxes) if all(bx[d][1] <= cell[d][1] && cell[d][2] <= bx[d][2] for d in 1:k)]
        isempty(cover) && continue
        ncells += prod(cell[d][2] - cell[d][1] + 1 for d in 1:k)
        push!(regions, Vector{Any}[Any[lo, hi] for (lo, hi) in cell])
        push!(values, faq(cell, sumof(terms[cover])))
    end
    if ncells < prod(shape)
        pushfirst!(regions, Vector{Any}[Any[1, n] for n in shape])
        pushfirst!(values, lit(0))
    end
    return _region_array(regions, values, shape)
end

# An array over `shape` (row dimensions, then any contracted ones) given by
# `makearray` regions: one `faq` over the whole extent whose body reads the
# `makearray` at its own cell. The definition stays an array aggregate (a
# lane program builds an observed only from one) while each region keeps its
# own box, so a read inside a region one cell thick along a lane set stays
# on the lane's cell. The `faq`'s names are `_m<d>`, `_` appended until clear
# of every name of the regions' values.
function _region_array(regions, values, shape)
    taken = Set{String}()
    for v in values
        _all_names!(taken, v)
    end
    names = String[_fresh!(taken, "_m$d", "_") for d in eachindex(shape)]
    ma = OpExpr("makearray", ASTExpr[]; regions = regions, values = values)
    return OpExpr(
        "faq", ASTExpr[];
        output_idx = Any[names...],
        expr_body = op("index", ma, ASTExpr[VarExpr(n) for n in names]...),
        ranges = Dict{String, Any}(n => Any[1, shape[d]] for (d, n) in enumerate(names))
    )
end

# The companions a band list reads, directly or through other companions'
# definitions, as `(name, observed, definition)` in the observed's name
# order — what the block's `intermediates` field and every derived document of
# the list carry (spec §4.2, §8.1). Empty unless some coefficient (or
# `where`) reads a companion name, which no source name can be.
function _time_companions(
        sv::SysView, entries; info::Union{Nothing, _RecInfo} = nothing,
        extra = Dict{String, ASTExpr}()
    )
    out = Tuple{String, String, ASTExpr}[]
    isempty(entries) && return out
    # Every companion name's last segment starts with `_dt_`: a list none of
    # whose coefficients reads such a name (every block of another axis)
    # reads no companion, and the analysis below is skipped.
    cand = Set{String}()
    for en in entries
        for x in (en.band.coef, en.band.guard)
            x === nothing && continue
            for n in free_variables(x)
                i = findlast('.', n)
                startswith(i === nothing ? n : n[nextind(n, i):end], "_dt_") && push!(cand, n)
            end
        end
    end
    isempty(cand) && return out
    info === nothing && (info = _rec_info(sv))
    obs, ctx, recs = info.obs, info.ctx, info.recs
    rn = Set{String}(r.name for r in recs)
    obs2 = Dict{String, ASTExpr}(k => v for (k, v) in obs if !(k in rn))
    # a time pair's `a2` intermediates are observed of the time pass (§4.4)
    merge!(obs2, extra)
    obs = isempty(extra) ? obs : merge(obs, extra)
    moving = _moving_observed(obs2, Set{String}(["t"]))
    isempty(moving) && return out
    names = _companion_names(sv, moving)
    byname = Dict{String, String}(n => o for (o, n) in names)
    read = Set{String}(n for n in cand if haskey(byname, n))
    isempty(read) && return out
    cs = _companion_setup(sv, ctx, obs2, _declared_scalars(sv, obs), true; rank_obs = obs)
    defs = Dict{String, ASTExpr}()
    work = collect(read)
    while !isempty(work)
        n = pop!(work)
        haskey(defs, n) && continue
        o = byname[n]
        d = _companion_definition(_companion_bands!(cs, o), get(cs.ctx.shapes, o, Int[]))
        defs[n] = d
        for m in free_variables(d)
            haskey(byname, m) && !haskey(defs, m) && push!(work, m)
        end
    end
    for o in sort!(collect(String, moving))
        n = names[o]
        haskey(defs, n) && push!(out, (n, o, defs[n]))
    end
    return out
end

# The companions as observed of a derived model (spec §8.1): each declared
# like its observed (`{"type": "unknown"}`, with the observed's declared
# `shape` when it has one) and defined by a bare-LHS equation, after the
# source's equations.
function _add_companions!(m, sv::SysView, comps)
    for c in comps
        n, o = c[1], c[2]
        d = OrderedDict{String, Any}("type" => "unknown")
        var = get(sv.variables, o, nothing)
        shp = var === nothing ? nothing : var.shape
        # a named step of another axis (§4.6), and a time pair's companion of
        # one (§4.4), is declared without a shape: its definition gives its
        # extents
        noshape = (length(c) >= 4 && c[4] != 0) || var === nothing
        noshape || shp === nothing || isempty(shp) || (d["shape"] = Any[x for x in shp])
        m["variables"][n] = d
    end
    for c in comps
        push!(m["equations"], OrderedDict{String, Any}("lhs" => c[1], "rhs" => _ser_expr(c[3])))
    end
    return
end

# The block's `intermediates` field (spec §4, §4.2, §4.6): `[{name, of,
# rhs}]`.
_companions_field(comps) = Any[
    OrderedDict{String, Any}("name" => c[1], "of" => c[2], "rhs" => _ser_expr(c[3]))
        for c in comps
]

# ── second-order pairs `["time", a2]` (spec §4.4) ───────────────────────────
#
# With every branch frozen (spec §3.2) each branch is smooth, so the mixed
# partials commute: ∂²f/∂t∂x = ∂/∂t (∂f/∂x). A time pair is therefore built
# with `t` differentiated SECOND: each first-order entry of the `a2` axis has
# its coefficient's time bands taken by the time pass's own rule — the
# observed that move with `t` are columns composed with their companions,
# never inlined. The `a2` pass keeps its named steps (§4.6), and its named
# intermediates are observed of the time pass: one that moves with `t` is a
# column composed with its own companion. `nothing` when a recurrence is
# active on either axis: that pair keeps the inlining path, as the time axis
# does.
function _time_pair_bands(sv::SysView, a2::Symbol, forcing, memo, t2)
    info = _rec_info!(memo, sv)
    isempty(info.recs) ||
        isempty(_active(info.recs, info.obs, union(Set{String}(["t"]), t2))) || return nothing
    # the a2 pass keeps its named steps (§4.6): its coefficients read named
    # intermediates, which are observed of the time pass here, each moving
    # with `t` a column composed with its own companion — never inlined
    e2 = _extended_pass(sv, info.ctx, info.obs, a2, forcing, String[]; info = info).rows
    inter = Dict{String, ASTExpr}(
        n => d for (n, _, d, k) in _intermediates(sv, e2; forcing = forcing, info = info) if k == 1
    )
    rn = Set{String}(r.name for r in info.recs)
    obs = Dict{String, ASTExpr}(k => v for (k, v) in info.obs if !(k in rn))
    merge!(obs, inter)
    cs = _companion_setup(
        sv, info.ctx, obs, _declared_scalars(sv, info.obs), true; rank_obs = merge(info.obs, inter)
    )
    out = JacEntry2[]
    for en in e2
        for b in _time_coef_bands(cs, en.band)
            push!(out, JacEntry2(en.u, "t", en.v, b, 0))
        end
    end
    return out
end

# The time bands of a first-order band's coefficient, broadcast over its
# rows (its contracted names bound too): `_time_row_bands` on a coefficient.
# Every band keeps the first-order column indices, which become the pair's
# `col2_idx`; a companion read whose index is not provably inside the
# observed's extent is gated as on the time axis. A site under a nested sum
# adds contracted dimensions (spec §4.4), which stay contracted.
function _time_coef_bands(cs::_Companions, b::Band)
    ctx = cs.ctx
    coef = _with_layer(ctx.refs) do
        inline_observed(
            b.coef, cs.inline; shapes = ctx.shapes, axes = ctx.axes,
            rank_obs = cs.rank_obs, scalars = cs.scalars
        )
    end
    b0 = _rebuild(b; coef = coef)
    n = length(b.cidx)
    out = Band[]
    for v in sort!(String[x for x in free_variables(coef) if x == "t" || x in cs.cols])
        bs = Band[normalize_band(x) for x in merge_bands(_coef_bands(b0, v, ctx))]
        if v == "t"
            append!(out, bs)
            continue
        end
        isempty(_companion_bands!(cs, v)) && continue
        shape = get(ctx.shapes, v, Int[])
        for x in bs
            push!(out, _pair_through_companion(x, n, v, cs.names[v], shape))
        end
    end
    return Band[normalize_band(x) for x in merge_bands(_sum_same_box2(out))]
end

# Band `x` of a coefficient (column indices: the first-order ones, then the
# moving observed's `φ`) composed with that observed's companion `name`:
# coefficient `coef · name[φ…]`, gated by `1 <= φ_d <= n_d` where `φ_d` is
# not provably in extent over the band's box, column indices back to the
# first `n`.
function _pair_through_companion(x::Band, n::Int, o::String, name::String, shape::Vector{Int})
    φ = x.cidx[(n + 1):end]
    length(φ) == length(shape) || throw(
        BandError(
            "observed `$o` of rank $(length(shape)) is read with $(length(φ)) " *
                "indices; declare its shape"
        )
    )
    read = isempty(φ) ? VarExpr(name) : op("index", VarExpr(name), φ...)
    term = _simp(mul(x.coef, read))
    box = _index_box(x)
    conds = ASTExpr[]
    for (d, c) in enumerate(φ)
        bd = _index_bounds(c, box)
        bd !== nothing && bd[1] >= 1 && bd[2] <= shape[d] && continue
        push!(conds, op(">=", c, lit(1)))
        push!(conds, op("<=", c, lit(shape[d])))
    end
    isempty(conds) || (term = op("ifelse", _conj(conds), term, lit(0)))
    return _rebuild(x; cidx = x.cidx[1:n], coef = term)
end

# `_sum_same_box` for pair bands: the box also holds the column indices and
# the contracted dimensions.
function _sum_same_box2(bs::Vector{Band})
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
        length(idx) == 1 && (push!(out, b); continue)
        push!(out, _rebuild(b; coef = op("+", ASTExpr[bs[i].coef for i in idx]...)))
    end
    return out
end
