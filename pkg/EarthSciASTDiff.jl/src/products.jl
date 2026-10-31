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

# The first of `base`, `base_`, `base__`, … not in `taken`, then taken.
function _fresh!(taken::Set{String}, base::String)
    nm = base
    while nm in taken
        nm *= "_"
    end
    push!(taken, nm)
    return nm
end

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
    rhs === nothing && (
        rhs = isempty(shape) ? 0.0 : OrderedDict{String, Any}(
                "op" => "makearray", "args" => Any[],
                "regions" => Any[Any[Any[1, n] for n in shape]], "values" => Any[0.0]
            )
    )
    push!(
        m["equations"], OrderedDict{String, Any}(
            "lhs" => OrderedDict{String, Any}("op" => "D", "args" => Any[name], "wrt" => "t"),
            "rhs" => rhs
        )
    )
    return
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

# Whether every point of `b` reads `ci` inside `shape`; `nothing` when an
# index is not closed at build time (a gather through a table that is not
# `const`), so the question has no build-time answer.
function _in_extent(b::Band, ci::Vector{ASTExpr}, shape::Vector{Int})
    isempty(ci) && return true
    return try
        _all_points(_rebuild(b; cidx = ci)) do _, _, c
            all(d -> 1 <= c[d] <= shape[d], eachindex(c))
        end
    catch err
        err isa InterruptException && rethrow()
        nothing
    end
end

# §9.1's ghost rule for a read of column variable `v` at `ci` on axis `wrt`:
# the in-extent conjuncts `G` to filter by, none when every point is in
# extent (the region split then leaves the box whole). On the forcing axis
# there is no ghost read: a point outside its buffer is refused.
function _ghost_conjuncts(b::Band, v::String, ci::Vector{ASTExpr}, shape::Vector{Int}, wrt::Symbol)
    ok = _in_extent(b, ci, shape)
    ok === true && return Tuple{Any, Set{String}}[]
    wrt == :forcing && throw(
        BandError(
            "a point of a band reads forcing buffer `$v` outside its extent" *
                (ok === nothing ? " or at an index not closed at build time" : "") *
                "; a product document would drop it (esm-jacobian-spec.md §9.1)"
        )
    )
    out = Tuple{Any, Set{String}}[]
    for (d, e) in enumerate(ci)
        ns = Set{String}(free_variables(e))
        push!(out, (_sop("<=", 1, _ser_expr(e)), ns))
        push!(out, (_sop("<=", _ser_expr(e), shape[d]), ns))
    end
    return out
end

# A band with no index names has one point (none when its closed `where`
# fails), and §9.2 has its column indices closed: whether that point reads
# `v` inside its extent. A ghost is dropped, except on the forcing axis.
function _point_in_extent(b::Band, v::String, ci::Vector{ASTExpr}, shape::Vector{Int}, wrt::Symbol)
    ok = _in_extent(b, ci, shape)
    ok === nothing && throw(
        BandError(
            "a band with no index names reads `$v` at an index not closed at build " *
                "time (esm-jacobian-spec.md §9.2)"
        )
    )
    ok || wrt != :forcing || _ghost_conjuncts(b, v, ci, shape, wrt)  # throws
    return ok
end

# Fresh output-index names `_j1 … _jn`, clear of the entry's and the
# document's names (§9.1).
function _jnames(taken::Set{String}, b::Band, coef::ASTExpr, n::Int)
    local_taken = union(taken, _row_names(b), first.(b.contracted), _names_of((coef, b.cidx...)))
    b.guard === nothing || union!(local_taken, free_variables(b.guard))
    return [_fresh!(local_taken, "_j$d") for d in 1:n]
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

# Whether a band with no index names has its one point: its closed `where`
# guard holds.
_closed_point(b::Band) = !_all_points((_...) -> false, b)

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
        (_closed_point(b) && _point_in_extent(b, en.v, b.cidx, shape_v, wrt)) || return nothing
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

function _vjp_term(en::JacEntry, coef::ASTExpr, lam::String, ctx, wrt::Symbol, taken)
    b = en.band
    body = _sop("*", _ser_expr(coef), _sread(lam, _rho(b)))
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
    _has_names(b) || _point_in_extent(b, en.v, ci, shape1, a1) || return nothing
    g = _ghost_conjuncts(b, en.v, ci, shape1, a1)
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
    _has_names(b) || _point_in_extent(b, en.w, c2, shape2, :states) || return nothing
    g = _ghost_conjuncts(b, en.w, c2, shape2, :states)
    return _scatter_term(
        b, coef, en.v, ci, get(ctx.shapes, en.v, Int[]), :states, g, body,
        union(_names_of((coef, c2...)), _row_names(b)), taken
    )
end

# ── the documents ────────────────────────────────────────────────────────────

# Each output's equation: the left fold of its entries' terms (§9.1).
# `owner` yields `(k, the output variable entry k lands in)`.
function _outputs!(m, sv, ctx, outs::OrderedDict{String, String}, term, owner)
    for (c, nm) in outs
        terms = Any[]
        for (k, o) in owner
            o == c || continue
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
    vin = OrderedDict{String, String}(c => _fresh!(taken, "_v_$c") for c in _column_vars(sv, wrt, forcing))
    out = OrderedDict{String, String}(r => _fresh!(taken, "_jvp_$r") for r in view_states(sv))
    for (c, nm) in vin
        _declare!(m, sv, ctx, nm, c, nothing)
    end
    _outputs!(
        m, sv, ctx, out, k -> _jvp_term(entries[k], coefs[k], vin[entries[k].v], ctx, wrt),
        ((k, en.u) for (k, en) in enumerate(entries))
    )
    return _derived_version!(doc), (model = "JvpEval", v = vin, jvp = out)
end

function _vjp_document(
        sv::SysView, entries::Vector{JacEntry}, wrt::Symbol, forcing;
        cse::Bool = true, cse_min_nodes::Int = 12
    )
    doc, m, ctx, coefs = _product_base(sv, "VjpEval", entries, cse, cse_min_nodes)
    taken = _doc_names(doc)
    lam = OrderedDict{String, String}(r => _fresh!(taken, "_lam_$r") for r in view_states(sv))
    out = OrderedDict{String, String}(c => _fresh!(taken, "_vjp_$c") for c in _column_vars(sv, wrt, forcing))
    for (r, nm) in lam
        _declare!(m, sv, ctx, nm, r, nothing)
    end
    _outputs!(
        m, sv, ctx, out, k -> _vjp_term(entries[k], coefs[k], lam[entries[k].u], ctx, wrt, taken),
        ((k, en.v) for (k, en) in enumerate(entries))
    )
    return _derived_version!(doc), (model = "VjpEval", lam = lam, vjp = out)
end

function _hvp_document(
        sv::SysView, entries::Vector{JacEntry2}, (a1, a2)::Tuple{Symbol, Symbol}, forcing,
        stages::Int; form::Symbol = :contract, cse::Bool = true, cse_min_nodes::Int = 12
    )
    form in (:contract, :symmetric) ||
        throw(ArgumentError("form must be :contract or :symmetric, got :$form"))
    form == :symmetric && (a1, a2) != (:states, :states) && throw(
        ArgumentError("form = :symmetric needs a (:states, :states) pair, got ($a1, $a2)")
    )
    stages >= 1 || throw(ArgumentError("a contraction needs at least one stage, got $stages"))
    model = form == :contract ? "HvpEval" : "HvpEvalSymmetric"
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
        push!(mu, OrderedDict{String, String}(r => _fresh!(taken, "_mu$(i)_$r") for r in states))
        push!(w, OrderedDict{String, String}(c => _fresh!(taken, "_w$(i)_$c") for c in cols1))
    end
    cols2 = form == :contract ? _column_vars(sv, a2, forcing) : states
    out = OrderedDict{String, String}(c => _fresh!(taken, "_hvp_$c") for c in cols2)
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
    return _derived_version!(doc), (model = model, mu = mu, w = w, hvp = out)
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
        cse::Bool = true, cse_min_nodes::Int = 12
    )
    sv, _ = _sv_src(input, model_name)
    entries = jacobian_bands(sv; wrt = wrt, forcing = forcing)
    return _jvp_document(sv, entries, wrt, forcing; cse = cse, cse_min_nodes = cse_min_nodes)
end

@doc (@doc jvp_document) function vjp_document(
        input; wrt::Symbol = :states, model_name = nothing, forcing = nothing,
        cse::Bool = true, cse_min_nodes::Int = 12
    )
    sv, _ = _sv_src(input, model_name)
    entries = jacobian_bands(sv; wrt = wrt, forcing = forcing)
    return _vjp_document(sv, entries, wrt, forcing; cse = cse, cse_min_nodes = cse_min_nodes)
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
        model_name = nothing, forcing = nothing, cse::Bool = true, cse_min_nodes::Int = 12
    )
    a1, a2 = _axis_pair(wrt)
    sv, _ = _sv_src(input, model_name)
    entries = jacobian_bands2(sv; wrt = (a1, a2), forcing = forcing)
    return _hvp_document(
        sv, entries, (a1, a2), forcing, stages;
        form = form, cse = cse, cse_min_nodes = cse_min_nodes
    )
end
