# Product documents (esm-jacobian-spec.md §9): derived ESM documents whose
# right-hand side, evaluated once by an unmodified EarthSciAST evaluator,
# yields `J v`, `Jᵀ λ` or the §4.4 second-order contraction. Like the
# evaluation document (emit.jl), they are ASTs only: this file evaluates
# nothing at run time. The one numeric step is §9.1's build-time decision of
# which points read a column inside its extent, over closed column indices.

_sop(op, args...) = OrderedDict{String, Any}("op" => op, "args" => Any[args...])

# `index(v, idx…)`, or the bare name for a scalar read.
_sread(v::String, idx) = isempty(idx) ? v : _sop("index", v, idx...)

# A `+` over terms as a left fold, so the association is fixed (§9.1).
_sfold(op, xs) = foldl((a, b) -> _sop(op, a, b), xs)

_sand(cs) = length(cs) == 1 ? only(cs) : _sop("and", cs...)

# Every name of a derived document (§9.1): the variable and index-set names,
# every `ranges` key, and every string its equations hold other than an `op`.
function _doc_names(doc)::Set{String}
    taken = Set{String}(keys(doc["index_sets"]))
    for (_, m) in doc["models"]
        union!(taken, keys(m["variables"]))
        _collect_names!(taken, m["equations"])
    end
    return taken
end
_collect_names!(s, x::AbstractString) = push!(s, String(x))
_collect_names!(s, x::AbstractVector) = foreach(v -> _collect_names!(s, v), x)
_collect_names!(s, x) = nothing
function _collect_names!(s, x::AbstractDict)
    for (k, v) in x
        k == "op" && continue
        k == "ranges" && union!(s, String.(keys(v)))
        _collect_names!(s, v)
    end
    return
end

# An input or output name of a product document: `base` with every `.` of the
# source name it holds made `_`, then fresh (§9.1). A flattened system's
# column `SimpleOzone.O3` gives `_v_SimpleOzone_O3`, an unscoped name of the
# derived model, which a dotted name would not be (§8.1).
_iofresh!(taken::Set{String}, base::String) = _fresh!(taken, replace(base, '.' => '_'), "_")

# The column variables of one first-order axis, in canonical order (§9.1).
function _column_vars(sv::SysView, wrt::Symbol, forcing)::Vector{String}
    wrt == :time && return ["t"]
    wrt == :states && return view_states(sv)
    return first.(jacobian_columns(sv; wrt = wrt, forcing = forcing))
end

# Declare an input or output of the product document: an unknown shaped like
# `like` (`"t"`: a scalar) with default 0 and the equation `D(name) = rhs`,
# `rhs === nothing` meaning zero.
function _declare!(m, sv::SysView, ctx, name::String, like::String, rhs)
    d = OrderedDict{String, Any}("type" => "unknown", "default" => 0.0)
    shape = get(ctx.shapes, like, Int[])
    isempty(shape) || (d["shape"] = Any[x for x in sv.variables[like].shape])
    m["variables"][name] = d
    rhs === nothing && (rhs = _zeros(ctx, like))
    push!(
        m["equations"], OrderedDict{String, Any}(
            "lhs" => OrderedDict{String, Any}("op" => "D", "args" => Any[name], "wrt" => "t"),
            "rhs" => rhs
        )
    )
    return
end

# The zero value shaped like variable `name`: `0.0`, or a zero `makearray`
# over its shape.
function _zeros(ctx, name::String)
    shape = get(ctx.shapes, name, Int[])
    isempty(shape) && return 0.0
    return OrderedDict{String, Any}(
        "op" => "makearray", "args" => Any[],
        "regions" => Any[Any[Any[1, n] for n in shape]], "values" => Any[0.0]
    )
end

# Base of every product document (§9.1): the evaluation document's base, with
# the observed kept that the coefficients, column indices and `where` guards
# read, and the coefficients optionally hoisted. Returns (doc, model, ctx,
# coefficients).
function _product_base(sv::SysView, name::String, entries::Vector{JacEntry}, cse::Bool, cse_min_nodes::Int)
    doc = _base_document(sv, name)
    m = doc["models"][name]
    ctx = _ctx(sv)
    roots = ASTExpr[]
    for en in entries
        push!(roots, en.band.coef)
        append!(roots, en.band.cidx)
        en.band.guard === nothing || push!(roots, en.band.guard)
    end
    _zero_source_derivatives!(m, sv, ctx, roots)
    return doc, m, ctx, _hoist!(doc, m, sv, entries, cse, cse_min_nodes)
end

# A band's index names (row names, then contracted names) and their ranges.
function _index_box(b::Band)::Dict{String, Tuple{Int, Int}}
    box = Dict{String, Tuple{Int, Int}}()
    for (k, r) in enumerate(b.ridx)
        r isa String && (box[r] = b.rows[k])
    end
    for (n, lo, hi) in b.contracted
        box[n] = (lo, hi)
    end
    return box
end

# Bounds `(lo, hi, exact)` of an index expression over `box`, by interval
# arithmetic over the integer `+`, `-`, `*` core; `nothing` outside it (a
# `const` gather, a division, …). The bounds always enclose every value.
# `exact` when they are also attained at corners of the box: an affine
# expression reading each name at most once. Returns the names read too.
function _index_bounds(e::ASTExpr, box)
    if e isa IntExpr
        return (Int(e.value), Int(e.value), true, String[])
    elseif e isa NumExpr
        isinteger(e.value) || return nothing
        return (Int(e.value), Int(e.value), true, String[])
    elseif e isa VarExpr
        haskey(box, e.name) || return nothing
        lo, hi = box[e.name]
        return (lo, hi, true, [e.name])
    elseif e isa OpExpr && e.op in ("+", "-", "*") && !isempty(e.args)
        parts = [_index_bounds(a, box) for a in e.args]
        any(isnothing, parts) && return nothing
        lo, hi, ex, ns = parts[1]
        ns = copy(ns)
        if e.op == "-" && length(parts) == 1
            return (-hi, -lo, ex, ns)
        end
        for (plo, phi, pex, pns) in parts[2:end]
            if e.op == "+"
                lo, hi = lo + plo, hi + phi
            elseif e.op == "-"
                lo, hi = lo - phi, hi - plo
            else
                # affine only while one factor is a constant
                ex &= isempty(ns) || isempty(pns)
                c = (lo * plo, lo * phi, hi * plo, hi * phi)
                lo, hi = minimum(c), maximum(c)
            end
            ex &= pex
            append!(ns, pns)
        end
        return (lo, hi, ex && allunique(ns), ns)
    end
    return nothing
end

# Whether every point of `b` reads `ci` inside `shape`; `nothing` when that
# has no build-time answer: an index not closed (a gather through a table
# that is not `const`), or a `where` guard not closed that a point outside
# the extent depends on. Decided from the indices' bounds where they settle
# it (§9.1's affine breakpoints), by walking the band's points otherwise.
function _in_extent(b::Band, ci::Vector{ASTExpr}, shape::Vector{Int})
    isempty(ci) && return true
    box = _index_box(b)
    any(((lo, hi),) -> lo > hi, values(box)) && return true   # no points
    names = keys(box)
    all(e -> _index_only(e, names), ci) || return nothing
    bounds = [_index_bounds(e, box) for e in ci]
    if !any(isnothing, bounds)
        all(d -> 1 <= bounds[d][1] && bounds[d][2] <= shape[d], eachindex(ci)) &&
            return true
        # a corner outside the extent is a point of the band unless a guard
        # may drop it
        b.guard === nothing && all(x -> x[3], bounds) && return false
    end
    inside(_, _, c) = all(d -> 1 <= c[d] <= shape[d], eachindex(c))
    b.guard === nothing || _index_only(b.guard, names) ||
        return _all_points(inside, _rebuild(b; cidx = ci, guard = nothing)) ? true : nothing
    return _all_points(inside, _rebuild(b; cidx = ci))
end

# The forcing axis has no ghost read (§9.1): a point outside its buffer is
# refused rather than dropped.
function _forcing_ghost(v::String, ok)
    why = ok === nothing ?
        " or where that is not decidable at build time (an index or `where` guard not closed)" : ""
    throw(
        BandError(
            "a point of a band reads forcing buffer `$v` outside its extent$why; " *
                "a product document would drop it (esm-jacobian-spec.md §9.1)"
        )
    )
end

# Refuse a read of `v` with more indices than its shape has dimensions.
function _check_rank(v::String, ci::Vector{ASTExpr}, shape::Vector{Int})
    length(ci) > length(shape) && throw(
        BandError(
            "a band reads `$v` with $(length(ci)) indices, but it has " *
                "$(length(shape)) dimensions"
        )
    )
    return
end

# §9.1's ghost rule for a read of column variable `v` at `ci` on axis `wrt`:
# whether a point may read outside `v`'s extent, so the read must be filtered
# by the in-extent conjuncts `1 <= ci_d <= shape_d`. A ghost read on the
# forcing axis is refused.
function _needs_ghost(b::Band, v::String, ci::Vector{ASTExpr}, shape::Vector{Int}, wrt::Symbol)
    _check_rank(v, ci, shape)
    ok = _in_extent(b, ci, shape)
    ok === true && return false
    wrt == :forcing && _forcing_ghost(v, ok)
    return true
end

# The in-extent conjuncts `G` of `_needs_ghost`, as `(conjunct, names)`
# pairs; none when every point is in extent (the region split then leaves
# the box whole).
function _ghost_conjuncts(b::Band, v::String, ci::Vector{ASTExpr}, shape::Vector{Int}, wrt::Symbol)
    _needs_ghost(b, v, ci, shape, wrt) || return Tuple{Any, Set{String}}[]
    out = Tuple{Any, Set{String}}[]
    for (d, e) in enumerate(ci)
        ns = Set{String}(free_variables(e))
        push!(out, (_sop("<=", 1, _ser_expr(e)), ns))
        push!(out, (_sop("<=", _ser_expr(e), shape[d]), ns))
    end
    return out
end

# A band with no index names has one point (none when its closed `where`
# fails), and §9.2 has its column indices closed: whether that point exists
# and reads `v` inside its extent. A ghost is dropped, except on the forcing
# axis, where it is refused. Decided from the literal indices, no walk.
function _point_in_extent(b::Band, v::String, ci::Vector{ASTExpr}, shape::Vector{Int}, wrt::Symbol)
    _check_rank(v, ci, shape)
    all(e -> isempty(free_variables(e)), ci) || throw(
        BandError(
            "a band with no index names reads `$v` at an index not closed at build " *
                "time (esm-jacobian-spec.md §9.2)"
        )
    )
    _closed_point(b) || return false
    ok = _in_extent(_rebuild(b; guard = nothing), ci, shape)::Bool
    ok || wrt != :forcing || _forcing_ghost(v, ok)
    return ok
end

# Fresh output-index names `_j1 … _jn`, clear of the entry's and the
# document's names (§9.1).
function _jnames(taken::Set{String}, b::Band, coef::ASTExpr, n::Int)
    # the document's names are looked up, never copied: a per-entry copy of
    # the document-wide set makes the build quadratic
    local_taken = union(_row_names(b), first.(b.contracted), _names_of((coef, b.cidx...)))
    b.guard === nothing || union!(local_taken, free_variables(b.guard))
    out = String[]
    for d in 1:n
        nm = "_j$d"
        while nm in taken || nm in local_taken
            nm *= "_"
        end
        push!(local_taken, nm)
        push!(out, nm)
    end
    return out
end

# The `faq` of one entry's term: `ranges` (the output names first, then the
# row names, then the contracted names), the `filter` conjunction, and `body`
# under §9.1's `ifelse` guard when a reduced name would not occur in it.
function _term_faq(b::Band, output_idx, outranges, conj, body, bodynames::Set{String})
    ranges = OrderedDict{String, Any}(n => Any[lo, hi] for (n, lo, hi) in outranges)
    for (k, r) in enumerate(b.ridx)
        r isa String && (ranges[r] = Any[b.rows[k][1], b.rows[k][2]])
    end
    for (n, lo, hi) in b.contracted
        ranges[n] = Any[lo, hi]
    end
    outnames = Set{String}(x for x in output_idx if x isa String)
    miss = [n for n in keys(ranges) if !(n in outnames) && !(n in bodynames)]
    expr = body
    if !isempty(miss)
        phi = [c for (c, ns) in conj if any(in(ns), miss)]
        all(n -> any(((_, ns),) -> n in ns, conj), miss) || throw(
            BandError(
                "the reduced index name(s) $(miss) occur neither in the product's body " *
                    "nor in its filter (esm-jacobian-spec.md §9.1)"
            )
        )
        expr = _sop("ifelse", _sand(phi), body, 0.0)
    end
    f = OrderedDict{String, Any}(
        "op" => "faq", "args" => Any[], "output_idx" => Any[output_idx...],
        "ranges" => ranges
    )
    isempty(conj) || (f["filter"] = _sand(first.(conj)))
    f["expr"] = expr
    return f
end

_row_names(b::Band) = Set{String}(r for r in b.ridx if r isa String)
_has_names(b::Band) = any(r -> r isa String, b.ridx) || !isempty(b.contracted)
_where_conj(b::Band) = b.guard === nothing ? Tuple{Any, Set{String}}[] :
    Tuple{Any, Set{String}}[(_ser_expr(b.guard), Set{String}(free_variables(b.guard)))]
_names_of(es) = Set{String}(n for e in es for n in free_variables(e))

# Whether a band with no index names has its one point: its `where` guard,
# which §9.2 has closed, holds.
function _closed_point(b::Band)
    b.guard === nothing && return true
    isempty(free_variables(b.guard)) || throw(
        BandError(
            "a band with no index names has a `where` guard not closed at build " *
                "time (esm-jacobian-spec.md §9.2)"
        )
    )
    return _eval_guard(b.guard, Dict{String, Float64}())
end

# ── Jv (§9.2) ───────────────────────────────────────────────────────────────

function _jvp_term(en::JacEntry, coef::ASTExpr, vin::String, ctx, wrt::Symbol)
    b = en.band
    shape_v = get(ctx.shapes, en.v, Int[])
    body = _sop("*", _ser_expr(coef), _sread(vin, Any[_ser_expr(e) for e in b.cidx]))
    shape_u = get(ctx.shapes, en.u, Int[])
    val = if _has_names(b)
        conj = vcat(_where_conj(b), _ghost_conjuncts(b, en.v, b.cidx, shape_v, wrt))
        bodynames = union(_names_of((coef, b.cidx...)), _row_names(b))
        out = isempty(shape_u) ? Any[] : Any[r isa String ? r : 1 for r in b.ridx]
        _term_faq(b, out, (), conj, body, bodynames)
    else
        # no index names: `where` and the column are closed, decided here
        _point_in_extent(b, en.v, b.cidx, shape_v, wrt) || return nothing
        body
    end
    isempty(shape_u) && return val
    return OrderedDict{String, Any}(
        "op" => "makearray", "args" => Any[],
        "regions" => Any[Any[Any[1, n] for n in shape_u], Any[Any[lo, hi] for (lo, hi) in b.rows]],
        "values" => Any[0.0, val]
    )
end

# ── the equality-filter scatter (§9.3, §9.4) ─────────────────────────────────

# One entry's term landing in output variable `out` (shape `shape_o`) at
# column index `ci`: Σ over the points with `ci == j` of `body`, under the
# extra filter conjuncts `conj`.
function _scatter_term(
        b::Band, coef::ASTExpr, out::String, ci::Vector{ASTExpr}, shape_o::Vector{Int},
        wrt::Symbol, conj, body, bodynames::Set{String}, taken::Set{String}
    )
    # the equality drops every ghost, but on the forcing axis a dropped
    # point is a lost gradient
    wrt == :forcing && _ghost_conjuncts(b, out, ci, shape_o, wrt)
    !_has_names(b) && isempty(shape_o) && return _closed_point(b) ? body : nothing
    j = _jnames(taken, b, coef, length(shape_o))
    eqs = Tuple{Any, Set{String}}[
        (_sop("==", _ser_expr(e), j[d]), push!(Set{String}(free_variables(e)), j[d]))
            for (d, e) in enumerate(ci)
    ]
    return _term_faq(
        b, j, ((j[d], 1, shape_o[d]) for d in eachindex(j)),
        vcat(_where_conj(b), conj, eqs), body, bodynames
    )
end

_rho(b::Band) = Any[r for r in b.ridx]

_vjp_term(en::JacEntry, coef::ASTExpr, lam::String, ctx, wrt::Symbol, taken) =
    _vjp_term_read(en, coef, _sread(lam, _rho(en.band)), ctx, wrt, taken)

# `_vjp_term` with the cotangent read at the row point given: a plain input,
# or a reversed adjoint sweep (§9.7).
function _vjp_term_read(en::JacEntry, coef::ASTExpr, lam, ctx, wrt::Symbol, taken)
    b = en.band
    body = _sop("*", _ser_expr(coef), lam)
    return _scatter_term(
        b, coef, en.v, b.cidx, get(ctx.shapes, en.v, Int[]), wrt,
        Tuple{Any, Set{String}}[], body, union(_names_of((coef,)), _row_names(b)), taken
    )
end

# ── second-order contraction (§9.4) ──────────────────────────────────────────

# `form = :contract`: scatter by `col2_idx`, M summed over the stages.
function _hvp_term(en::JacEntry2, coef::ASTExpr, mu, w, ctx, a1::Symbol, a2::Symbol, taken)
    b = en.band
    ci, c2 = col_idx(en), col2_idx(en)
    m = _sfold(
        "+", [
            _sop("*", _sread(mu[i][en.u], _rho(b)), _sread(w[i][en.v], Any[_ser_expr(e) for e in ci]))
                for i in eachindex(mu)
        ]
    )
    shape1 = get(ctx.shapes, en.v, Int[])
    g = if _has_names(b)
        _ghost_conjuncts(b, en.v, ci, shape1, a1)
    else
        _point_in_extent(b, en.v, ci, shape1, a1) || return nothing
        Tuple{Any, Set{String}}[]
    end
    return _scatter_term(
        b, coef, en.w, c2, get(ctx.shapes, en.w, Int[]), a2, g,
        _sop("*", _ser_expr(coef), m),
        union(_names_of((coef, ci...)), _row_names(b)), taken
    )
end

# `form = :symmetric` on (states, states): scatter by the FIRST column.
function _hvp_sym_term(en::JacEntry2, coef::ASTExpr, mu, w, ctx, taken)
    b = en.band
    ci, c2 = col_idx(en), col2_idx(en)
    body = _sfold(
        "+", [
            _sop(
                "*", _sread(mu[i][en.u], _rho(b)), _ser_expr(coef),
                _sread(w[i][en.w], Any[_ser_expr(e) for e in c2])
            ) for i in eachindex(mu)
        ]
    )
    shape2 = get(ctx.shapes, en.w, Int[])
    g = if _has_names(b)
        _ghost_conjuncts(b, en.w, c2, shape2, :states)
    else
        _point_in_extent(b, en.w, c2, shape2, :states) || return nothing
        Tuple{Any, Set{String}}[]
    end
    return _scatter_term(
        b, coef, en.v, ci, get(ctx.shapes, en.v, Int[]), :states, g, body,
        union(_names_of((coef, c2...)), _row_names(b)), taken
    )
end

# ── the documents ────────────────────────────────────────────────────────────

# Each output's equation: the left fold of its entries' terms (§9.1).
# `owner` yields `(k, the output variable entry k lands in)`.
function _outputs!(m, sv, ctx, outs::OrderedDict{String, String}, term, owner)
    landing = Dict{String, Vector{Int}}()
    for (k, o) in owner
        push!(get!(landing, o, Int[]), k)
    end
    for (c, nm) in outs
        terms = Any[]
        for k in get(landing, c, Int[])
            t = term(k)
            t === nothing || push!(terms, t)
        end
        _declare!(m, sv, ctx, nm, c, isempty(terms) ? nothing : _sfold("+", terms))
    end
    return
end

function _jvp_document(
        sv::SysView, entries::Vector{JacEntry}, wrt::Symbol, forcing;
        cse::Bool = true, cse_min_nodes::Int = 12
    )
    doc, m, ctx, coefs = _product_base(sv, "JvpEval", entries, cse, cse_min_nodes)
    taken = _doc_names(doc)
    vin = OrderedDict{String, String}(c => _iofresh!(taken, "_v_$c") for c in _column_vars(sv, wrt, forcing))
    out = OrderedDict{String, String}(r => _iofresh!(taken, "_jvp_$r") for r in view_states(sv))
    for (c, nm) in vin
        _declare!(m, sv, ctx, nm, c, nothing)
    end
    _outputs!(
        m, sv, ctx, out, k -> _jvp_term(entries[k], coefs[k], vin[entries[k].v], ctx, wrt),
        ((k, en.u) for (k, en) in enumerate(entries))
    )
    return _finish_derived!(doc, sv, "JvpEval"), (model = "JvpEval", v = vin, jvp = out)
end

function _vjp_document(
        sv::SysView, entries::Vector{JacEntry}, wrt::Symbol, forcing;
        cse::Bool = true, cse_min_nodes::Int = 12
    )
    doc, m, ctx, coefs = _product_base(sv, "VjpEval", entries, cse, cse_min_nodes)
    taken = _doc_names(doc)
    lam = OrderedDict{String, String}(r => _iofresh!(taken, "_lam_$r") for r in view_states(sv))
    out = OrderedDict{String, String}(c => _iofresh!(taken, "_vjp_$c") for c in _column_vars(sv, wrt, forcing))
    for (r, nm) in lam
        _declare!(m, sv, ctx, nm, r, nothing)
    end
    _outputs!(
        m, sv, ctx, out, k -> _vjp_term(entries[k], coefs[k], lam[entries[k].u], ctx, wrt, taken),
        ((k, en.v) for (k, en) in enumerate(entries))
    )
    return _finish_derived!(doc, sv, "VjpEval"), (model = "VjpEval", lam = lam, vjp = out)
end

function _hvp_document(
        sv::SysView, entries::Vector{JacEntry2}, (a1, a2)::Tuple{Symbol, Symbol}, forcing,
        stages::Int; form::Symbol = :contract, cse::Bool = true, cse_min_nodes::Int = 12,
        memo::_FirstOrder = _FirstOrder()
    )
    form in (:contract, :symmetric) ||
        throw(ArgumentError("form must be :contract or :symmetric, got :$form"))
    form == :symmetric && (a1, a2) != (:states, :states) && throw(
        ArgumentError("form = :symmetric needs a (:states, :states) pair, got ($a1, $a2)")
    )
    stages >= 1 || throw(ArgumentError("a contraction needs at least one stage, got $stages"))
    model = form == :contract ? "HvpEval" : "HvpEvalSymmetric"
    pe = _pair_extended(sv, a1, a2, forcing, memo)
    pe === nothing || return _hvp_sweeps(
        sv, entries, pe, (a1, a2), forcing, stages, form, model, cse, cse_min_nodes
    )
    # the coefficients only: a second-order entry rides through the base and
    # the hoisting as a first-order one (its `cidx` holds both columns)
    doc, m, ctx, coefs = _product_base(
        sv, model, JacEntry[JacEntry(en.u, en.v, en.band) for en in entries], cse, cse_min_nodes
    )
    taken = _doc_names(doc)
    states = view_states(sv)
    cols1 = a1 == :time ? ["t"] : states
    mu = OrderedDict{String, String}[]
    w = OrderedDict{String, String}[]
    for i in 1:stages
        push!(mu, OrderedDict{String, String}(r => _iofresh!(taken, "_mu$(i)_$r") for r in states))
        push!(w, OrderedDict{String, String}(c => _iofresh!(taken, "_w$(i)_$c") for c in cols1))
    end
    cols2 = form == :contract ? _column_vars(sv, a2, forcing) : states
    out = OrderedDict{String, String}(c => _iofresh!(taken, "_hvp_$c") for c in cols2)
    for i in 1:stages
        for (r, nm) in mu[i]
            _declare!(m, sv, ctx, nm, r, nothing)
        end
        for (c, nm) in w[i]
            _declare!(m, sv, ctx, nm, c, nothing)
        end
    end
    if form == :contract
        _outputs!(
            m, sv, ctx, out, k -> _hvp_term(entries[k], coefs[k], mu, w, ctx, a1, a2, taken),
            ((k, en.w) for (k, en) in enumerate(entries))
        )
    else
        _outputs!(
            m, sv, ctx, out, k -> _hvp_sym_term(entries[k], coefs[k], mu, w, ctx, taken),
            ((k, en.v) for (k, en) in enumerate(entries))
        )
    end
    return _finish_derived!(doc, sv, model), (model = model, mu = mu, w = w, hvp = out)
end

# ── recurrences: tangent and adjoint sweeps (§9.7) ───────────────────────────

# The extended system of axis `wrt` (§4.5): with an active recurrence the
# document is the §9.7 sweeps over it, never the materialized bands (the
# sweeps need no transfer recurrence); without one its `rows` are
# `jacobian_bands`'s list.
_first_pass(sv::SysView, wrt::Symbol, forcing) =
    _extended_pass(sv, _ctx(sv), _observed_defs(sv), wrt, forcing, String[])

# The two extended systems of a pair with an active recurrence: the first
# axis's (seeded with the second axis's targets, as `jacobian_bands2`
# differentiates it — `memo` shares that pass) and the second axis's (its
# recurrence rows carry the adjoint back to the second axis's columns), both
# over the recurrences that move with either axis. `nothing` without one.
function _pair_extended(sv::SysView, a1::Symbol, a2::Symbol, forcing, memo::_FirstOrder)
    obs = _observed_defs(sv)
    isempty(recurrences(sv, obs, _ctx(sv))) && return nothing
    t1 = _targets(sv, a1, forcing, obs)
    t2 = _targets(sv, a2, forcing, obs)
    e1 = _first_order_ext!(memo, sv, a1, a2, forcing, t2)
    isempty(e1.recurrences) && return nothing
    e2 = a1 == a2 ? e1 : _first_order_ext!(memo, sv, a2, a1, forcing, t1)
    return e1, e2
end

# §9.1's ghost rule for a read of `v` at `ci` from band `b`, as expression
# conjuncts over the substituted indices `ci_sub` (the forcing axis refuses).
function _ghost_exprs(b::Band, v::String, ci, ci_sub, shape::Vector{Int}, wrt::Symbol)
    out = ASTExpr[]
    _needs_ghost(b, v, ci, shape, wrt) || return out
    for (e, n) in zip(ci_sub, shape)
        push!(out, op(">=", e, lit(1)))
        push!(out, op("<=", e, lit(n)))
    end
    return out
end

# Declare an intermediate of the product document (§9.7): an observed
# unknown shaped like recurrence `like`, defined by `name ~ rhs`.
function _observe!(m, sv::SysView, name::String, like::String, rhs)
    shape = something(sv.variables[like].shape, Any[])
    m["variables"][name] = OrderedDict{String, Any}(
        "type" => "unknown", "shape" => Any[string(x) for x in shape]
    )
    push!(m["equations"], OrderedDict{String, Any}("lhs" => name, "rhs" => rhs))
    return
end

# The forward sweep variable of `rec` (`target`): its own bands among `rows`
# (with their hoisted coefficients), reading a target column through `dir`
# and another recurrence's sweep through `sweeps`.
function _tangent(rec::Recurrence, rows, coefs, target::String, dir, sweeps, ctx, wrt::Symbol)
    own = Band[]
    cols = String[]
    for (en, c) in zip(rows, coefs)
        en.u == rec.name || continue
        push!(own, _rebuild(en.band; coef = c))
        push!(cols, en.v)
    end
    function col(b, c, ci)
        shape = get(ctx.shapes, c, Int[])
        nm, w = haskey(sweeps, c) ? (sweeps[c], :states) : (dir[c], wrt)
        return _index(nm, ci), _ghost_exprs(b, c, b.cidx, ci, shape, w)
    end
    return _ser_expr(_forward_sweep(rec, own, cols, target, col, lit(0)))
end

# The adjoint sweep variable of `rec` (`target`, reversed axis) fed by the
# cotangent source `src`: `rec`'s self bands among `rows`.
function _adjoint(rec::Recurrence, rows, coefs, target::String, src::String)
    own = Band[
        _rebuild(en.band; coef = c) for (en, c) in zip(rows, coefs)
            if en.u == rec.name && en.v == rec.name
    ]
    src_read = _reversed_read(rec, src, rec.frame)
    return _ser_expr(_adjoint_sweep(rec, own, target, src_read, lit(0)))
end

# A cotangent read at the row point of `b`: a plain input (a state row) or a
# reversed adjoint sweep (a recurrence row).
function _cot_read(b::Band, u::String, plain, rev, ext::ExtendedBands)
    rec = _recurrence(ext, u)
    rec !== nothing && haskey(rev, u) && return _ser_expr(_reversed_read(rec, rev[u], b.ridx))
    return _sread(plain[u], _rho(b))
end

# The cotangent source of recurrence `v`: Σ over the bands reading `v` (other
# than `v`'s own) of `coef · λ[row]`, scattered onto `v`'s cells (an
# equality-filter scatter, §9.3); zero when none reads it.
function _cot_source(v::String, entries, coefs, lam, rev, ext::ExtendedBands, ctx, taken)
    terms = Any[]
    for (en, c) in zip(entries, coefs)
        (en.v != v || en.u == v) && continue
        t = _vjp_term_read(en, c, _cot_read(en.band, en.u, lam, rev, ext), ctx, :states, taken)
        t === nothing || push!(terms, t)
    end
    return isempty(terms) ? _zeros(ctx, v) : _sfold("+", terms)
end

_rec_names(taken::Set{String}, ext::ExtendedBands, tag::String) = OrderedDict{String, String}(
    r.name => _iofresh!(taken, "_$(tag)_$(r.name)") for r in ext.recurrences
)

function _jvp_sweeps(sv::SysView, ext::ExtendedBands, wrt::Symbol, forcing, cse::Bool, cse_min_nodes::Int)
    every = vcat(ext.rows, ext.rec_rows)
    doc, m, ctx, coefs = _product_base(sv, "JvpEval", every, cse, cse_min_nodes)
    taken = _doc_names(doc)
    vin = OrderedDict{String, String}(c => _iofresh!(taken, "_v_$c") for c in _column_vars(sv, wrt, forcing))
    vdot = _rec_names(taken, ext, "vdot")
    out = OrderedDict{String, String}(r => _iofresh!(taken, "_jvp_$r") for r in view_states(sv))
    for (c, nm) in vin
        _declare!(m, sv, ctx, nm, c, nothing)
    end
    nrow = length(ext.rows)
    for rec in ext.recurrences
        t = _tangent(
            rec, every[(nrow + 1):end], coefs[(nrow + 1):end], vdot[rec.name], vin, vdot, ctx, wrt
        )
        _observe!(m, sv, vdot[rec.name], rec.name, t)
    end
    function term(k)
        en = ext.rows[k]
        haskey(vdot, en.v) && return _jvp_term(en, coefs[k], vdot[en.v], ctx, :states)
        return _jvp_term(en, coefs[k], vin[en.v], ctx, wrt)
    end
    _outputs!(m, sv, ctx, out, term, ((k, en.u) for (k, en) in enumerate(ext.rows)))
    return _finish_derived!(doc, sv, "JvpEval"), (model = "JvpEval", v = vin, jvp = out)
end

function _vjp_sweeps(sv::SysView, ext::ExtendedBands, wrt::Symbol, forcing, cse::Bool, cse_min_nodes::Int)
    every = vcat(ext.rows, ext.rec_rows)
    doc, m, ctx, coefs = _product_base(sv, "VjpEval", every, cse, cse_min_nodes)
    taken = _doc_names(doc)
    lam = OrderedDict{String, String}(r => _iofresh!(taken, "_lam_$r") for r in view_states(sv))
    vbar = _rec_names(taken, ext, "vbar")
    vrev = _rec_names(taken, ext, "vrev")
    out = OrderedDict{String, String}(c => _iofresh!(taken, "_vjp_$c") for c in _column_vars(sv, wrt, forcing))
    for (r, nm) in lam
        _declare!(m, sv, ctx, nm, r, nothing)
    end
    nrow = length(ext.rows)
    for rec in ext.recurrences
        v = rec.name
        _observe!(m, sv, vbar[v], v, _cot_source(v, every, coefs, lam, vrev, ext, ctx, taken))
        _observe!(
            m, sv, vrev[v], v,
            _adjoint(rec, every[(nrow + 1):end], coefs[(nrow + 1):end], vrev[v], vbar[v])
        )
    end
    function term(k)
        en = every[k]
        lr = _cot_read(en.band, en.u, lam, vrev, ext)
        return _vjp_term_read(en, coefs[k], lr, ctx, wrt, taken)
    end
    _outputs!(m, sv, ctx, out, term, ((k, en.v) for (k, en) in enumerate(every)))
    return _finish_derived!(doc, sv, "VjpEval"), (model = "VjpEval", lam = lam, vjp = out)
end

# The §4.4 contraction through recurrences (§9.7): per stage i, the tangent
# sweep of w_i and the adjoint sweep of μ_i over the first axis's extended
# system make the contraction's M read a recurrence row's cotangent and a
# recurrence column's direction; the second-order bands scattered onto a
# recurrence column feed one more adjoint sweep Ĥ, which the second axis's
# recurrence rows carry back to its columns.
function _hvp_sweeps(
        sv::SysView, entries::Vector{JacEntry2}, pe, (a1, a2)::Tuple{Symbol, Symbol}, forcing,
        stages::Int, form::Symbol, model::String, cse::Bool, cse_min::Int
    )
    e1, e2 = pe
    first_ = vcat(e1.rows, e1.rec_rows)
    nrow = length(e1.rows)
    # on a pair of one axis the second pass IS the first: its recurrence rows
    # (and their hoisted coefficients) are shared, not emitted twice
    shared = e2 === e1
    back = shared ? first_[(nrow + 1):end] : copy(e2.rec_rows)
    n2 = length(entries)
    nf = length(first_)
    doc, m, ctx, coefs = _product_base(
        sv, model,
        vcat(JacEntry[JacEntry(en.u, en.v, en.band) for en in entries], first_, shared ? JacEntry[] : back),
        cse, cse_min
    )
    c2s = coefs[1:n2]
    c1s = coefs[(n2 + 1):(n2 + nf)]
    cbs = shared ? c1s[(nrow + 1):end] : coefs[(n2 + nf + 1):end]
    taken = _doc_names(doc)
    states = view_states(sv)
    cols1 = a1 == :time ? ["t"] : states
    mu = OrderedDict{String, String}[]
    w = OrderedDict{String, String}[]
    for i in 1:stages
        push!(mu, OrderedDict{String, String}(r => _iofresh!(taken, "_mu$(i)_$r") for r in states))
        push!(w, OrderedDict{String, String}(c => _iofresh!(taken, "_w$(i)_$c") for c in cols1))
    end
    vdot = OrderedDict{String, String}[]
    vbar = OrderedDict{String, String}[]
    vrev = OrderedDict{String, String}[]
    for i in 1:stages
        push!(vdot, _rec_names(taken, e1, "vdot$i"))
        push!(vbar, _rec_names(taken, e1, "vbar$i"))
        push!(vrev, _rec_names(taken, e1, "vrev$i"))
    end
    hbar = _rec_names(taken, e1, "hbar")
    hrev = _rec_names(taken, e1, "hrev")
    cols2 = form == :contract ? _column_vars(sv, a2, forcing) : states
    out = OrderedDict{String, String}(c => _iofresh!(taken, "_hvp_$c") for c in cols2)
    for i in 1:stages
        for (r, nm) in mu[i]
            _declare!(m, sv, ctx, nm, r, nothing)
        end
        for (c, nm) in w[i]
            _declare!(m, sv, ctx, nm, c, nothing)
        end
    end
    own1, oc1 = first_[(nrow + 1):end], c1s[(nrow + 1):end]
    for i in 1:stages, rec in e1.recurrences
        v = rec.name
        _observe!(m, sv, vdot[i][v], v, _tangent(rec, own1, oc1, vdot[i][v], w[i], vdot[i], ctx, a1))
        _observe!(m, sv, vbar[i][v], v, _cot_source(v, first_, c1s, mu[i], vrev[i], e1, ctx, taken))
        _observe!(m, sv, vrev[i][v], v, _adjoint(rec, own1, oc1, vrev[i][v], vbar[i][v]))
    end
    cot(i, b, u) = _cot_read(b, u, mu[i], vrev[i], e1)
    dirr(i, v, ci) = _sread(haskey(vdot[i], v) ? vdot[i][v] : w[i][v], Any[_ser_expr(e) for e in ci])
    axis(v, a) = _is_recurrence(e1, v) ? :states : a
    function term2(k)
        en = entries[k]
        b = en.band
        coef = c2s[k]
        ci, c2 = col_idx(en), col2_idx(en)
        bodynames = union(_names_of((coef, ci..., c2...)), _row_names(b))
        if form == :contract
            mm = _sfold("+", [_sop("*", cot(i, b, en.u), dirr(i, en.v, ci)) for i in 1:stages])
            shape1 = get(ctx.shapes, en.v, Int[])
            a = axis(en.v, a1)
            g = if _has_names(b)
                _ghost_conjuncts(b, en.v, ci, shape1, a)
            else
                _point_in_extent(b, en.v, ci, shape1, a) || return nothing
                Tuple{Any, Set{String}}[]
            end
            return _scatter_term(
                b, coef, en.w, c2, get(ctx.shapes, en.w, Int[]), axis(en.w, a2), g,
                _sop("*", _ser_expr(coef), mm), bodynames, taken
            )
        end
        body = _sfold(
            "+", [_sop("*", cot(i, b, en.u), _ser_expr(coef), dirr(i, en.w, c2)) for i in 1:stages]
        )
        shape2 = get(ctx.shapes, en.w, Int[])
        g = if _has_names(b)
            _ghost_conjuncts(b, en.w, c2, shape2, :states)
        else
            _point_in_extent(b, en.w, c2, shape2, :states) || return nothing
            Tuple{Any, Set{String}}[]
        end
        return _scatter_term(
            b, coef, en.v, ci, get(ctx.shapes, en.v, Int[]), :states, g, body, bodynames, taken
        )
    end
    owner2 = String[form == :contract ? en.w : en.v for en in entries]
    function back_term(k)
        en = back[k]
        lr = _cot_read(en.band, en.u, Dict{String, String}(), hrev, e2)
        return _vjp_term_read(en, cbs[k], lr, ctx, axis(en.v, a2), taken)
    end
    for rec in e1.recurrences
        v = rec.name
        terms = Any[]
        for k in 1:n2
            owner2[k] == v || continue
            t = term2(k)
            t === nothing || push!(terms, t)
        end
        for (k, en) in enumerate(back)
            (en.v == v && en.u != v) || continue
            t = back_term(k)
            t === nothing || push!(terms, t)
        end
        _observe!(m, sv, hbar[v], v, isempty(terms) ? _zeros(ctx, v) : _sfold("+", terms))
        _observe!(m, sv, hrev[v], v, _adjoint(rec, back, cbs, hrev[v], hbar[v]))
    end
    owner = vcat(
        Tuple{Int, String}[(k, owner2[k]) for k in 1:n2],
        Tuple{Int, String}[(n2 + k, en.v) for (k, en) in enumerate(back)]
    )
    _outputs!(m, sv, ctx, out, k -> k <= n2 ? term2(k) : back_term(k - n2), owner)
    return _finish_derived!(doc, sv, model), (model = model, mu = mu, w = w, hvp = out)
end

"""
    jvp_document(input; wrt = :states, model_name = nothing, forcing = nothing,
                 cse = true) -> (doc, io)
    vjp_document(input; …) -> (doc, io)

The product documents of esm-jacobian-spec.md §9.2 / §9.3 for one
first-order axis: a derived ESM document (model `"JvpEval"` / `"VjpEval"`)
whose right-hand side, evaluated by an unmodified EarthSciAST evaluator at
`(u, inputs, p, t)`, leaves `J v` (resp. `Jᵀ λ`) in the derivative slots of
its outputs. `input` is an `EsmFile` or a `FlattenedSystem`, as for
[`jacobian_bands`](@ref).

`io` names the inputs and outputs: `io.v` maps each column variable to its
direction input and `io.jvp` each state to its output (`io.lam` / `io.vjp`
for `vjp_document`). The consumer writes the inputs into the evaluator's
state slots beside the source states and reads the outputs from `du`
through the evaluator's variable map.

A `VjpEval` document is an equality-filtered `faq` scatter: correct under
any evaluator, linear-time only under one that implements §9.3's
equality-filter scatter.
"""
function jvp_document(
        input; wrt::Symbol = :states, model_name = nothing, forcing = nothing,
        cse::Bool = true, cse_min_nodes::Int = 12, data_sources = nothing
    )
    sv, _ = _sv_src(input, model_name; data_sources = data_sources)
    _check_axis(wrt)
    ext = _first_pass(sv, wrt, forcing)
    isempty(ext.recurrences) ||
        return _jvp_sweeps(sv, ext, wrt, forcing, cse, cse_min_nodes)
    return _jvp_document(sv, ext.rows, wrt, forcing; cse = cse, cse_min_nodes = cse_min_nodes)
end

@doc (@doc jvp_document) function vjp_document(
        input; wrt::Symbol = :states, model_name = nothing, forcing = nothing,
        cse::Bool = true, cse_min_nodes::Int = 12, data_sources = nothing
    )
    sv, _ = _sv_src(input, model_name; data_sources = data_sources)
    _check_axis(wrt)
    ext = _first_pass(sv, wrt, forcing)
    isempty(ext.recurrences) ||
        return _vjp_sweeps(sv, ext, wrt, forcing, cse, cse_min_nodes)
    return _vjp_document(sv, ext.rows, wrt, forcing; cse = cse, cse_min_nodes = cse_min_nodes)
end

"""
    hvp_document(input; stages, wrt = (:states, :states), form = :contract,
                 model_name = nothing, forcing = nothing, cse = true) -> (doc, io)

The second-order product document of esm-jacobian-spec.md §9.4 for the
axis pair `wrt` and `stages` stage pairs: its right-hand side leaves
`g = ∇_{a2} Σᵢ μᵢᵀ J(u) wᵢ` (`∇_{a2} (Σᵢ wᵢ μᵢ)ᵀ f_t` on a time pair) in
the derivative slots of its outputs. `io.mu[i]` / `io.w[i]` map each state
(each first-axis column variable; `"t"` on a time pair) to stage `i`'s
inputs and `io.hvp` each second-axis column variable to its output.
`form = :symmetric` (a `(:states, :states)` pair only) scatters by the
first column instead (model `"HvpEvalSymmetric"`); both forms give the same
`g`.
"""
function hvp_document(
        input; stages::Int, wrt = (:states, :states), form::Symbol = :contract,
        model_name = nothing, forcing = nothing, cse::Bool = true, cse_min_nodes::Int = 12,
        data_sources = nothing
    )
    a1, a2 = _axis_pair(wrt)
    sv, _ = _sv_src(input, model_name; data_sources = data_sources)
    memo = _FirstOrder()   # the first pass, shared with the sweeps' (§9.7)
    _, entries = _bands2(sv, a1, a2, forcing, memo)
    return _hvp_document(
        sv, entries, (a1, a2), forcing, stages;
        form = form, cse = cse, cse_min_nodes = cse_min_nodes, memo = memo
    )
end
