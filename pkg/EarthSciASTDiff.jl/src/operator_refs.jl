# Operator references (esm-spec §9.6): the ONE layer of this package that
# knows what an `apply_operator` node is.
#
# A reference `{op: apply_operator, name: D, bindings: B}` denotes its
# expansion (§9.6.4 rule 2), and nothing here ever builds that expansion, not
# even transiently. Each definition is analysed once — memoized per
# definition, and per binding shape where the analysis needs shapes — and the
# result is reused at every use (rule 2: "analyse and compile each definition
# once"):
#
#   * A SCALAR definition (its body is a scalar) is differentiated once per
#     parameter, into a derived definition `<D>.d_<p>` (`_dparam!`). The
#     chain rule at a reference is then
#         ∂ D(B) / ∂s = Σ_p D.d_p(B) · ∂B[p]/∂s
#     (`_dref`, the `apply_operator` row of the derivative table): a reference
#     to the derived definition with the SAME bindings, times the derivative
#     of the argument. A derived body that calls another definition is itself
#     written with references (the chain rule recurses through bodies), so the
#     derived definitions form a DAG beside the source's.
#   * An ARRAY definition (its body is a `faq`/`makearray` array: a stencil
#     operator) is differentiated once per parameter and binding shape into
#     BANDS of its own index space (`_def_bands!`): the band calculus runs on
#     the body with the parameter as the column variable. Each band's
#     coefficient becomes a derived definition `<D>.d_<p>.b<k>@<shape>` whose
#     parameters are D's plus the band's row and contracted index names. A
#     use reads the array as an intermediate: the band calculus sees a
#     placeholder array variable in its place (`_abstract_refs`), and the
#     bands of the use with respect to the placeholder are COMPOSED with the
#     definition's bands, bound to the use's arguments (`_compose`). Only
#     index expressions (a band's row box, column indices and guard) are
#     combined with the call site's bindings; a body never is.
#
# Sparsity, bands, locality and structure follow from the same two analyses:
# the structure-only mode runs them without coefficients.
#
# Derived definitions are carried into every derived document
# (`_operators_blocks!`) beside the source's definitions their references
# reach, marked `reference_only` (§9.6.1: an engine-generated entry), so the
# documents write references, never expanded trees.
#
# Both spellings of the reference op are read alike: `apply_operator`
# (esm 2.4.0) and `apply_expression_template`, the name EarthSciAST releases
# before it give the same node (esm-spec §9.6.12, "Old names"). Derived
# references and blocks are written as `apply_operator` / `operators`: the
# loader of every supported EarthSciAST rewrites the old spelling, so a
# loaded source never carries it.

const _REF_OPS = ("apply_operator", "apply_expression_template")

_is_ref(e) = e isa OpExpr && e.op in _REF_OPS

# A parsed definition: its parameters, typed body, and the roles its body
# gives each parameter (esm-spec §9.6.11 "Positions and roles"): read in a
# VALUE position, read as an ARRAY (an `index` operand, a `faq` operand,
# a `makearray` value), or in a non-expression field (`ranges`, `from`,
# `regions`, `wrt`, `dim`: STRUCTURAL, bound to a literal).
mutable struct _DefInfo
    name::String
    params::Vector{String}
    body::ASTExpr
    entry::Any                       # the registry entry as a document writes it
    derived::Bool
    value::Set{String}
    array::Set{String}
    structural::Set{String}
    rank::Int                        # 0: a scalar body; else its output rank
end

"""
    RefLayer

The reference-aware analysis state of one [`SysView`](@ref): the definitions
its references reach (the flattened system's merged registry, or a model's
own block closed over its free names), the derived definitions the analyses
have written, and the memo tables that keep every analysis to once per
definition (and binding shape). Created by [`sysview`](@ref); shared by every
pass over the view.
"""
mutable struct RefLayer
    registry::OrderedDict{String, Any}            # name → entry (closed bodies)
    defs::Dict{String, _DefInfo}                  # parsed, source and derived
    derived::OrderedDict{String, Any}             # derived name → entry
    refop::String                                 # the op name references use
    dparam::Dict{Tuple{String, String}, ASTExpr}  # (D, p) → generic ∂D/∂p
    defbands::Dict{Any, Vector{Band}}             # (D, shape key, p, co) → bands
    bandkey::Dict{String, Any}                    # band definition → its shape key
    extents::Dict{Any, Tuple{Vector{Int}, Vector{Int}}}
    index_sets::Dict{String, Int}
end

function RefLayer(registry::AbstractDict, index_sets::Dict{String, Int}; refop = nothing)
    reg = OrderedDict{String, Any}(String(k) => v for (k, v) in registry)
    # EarthSciAST ≥ 0.15 reads `apply_expression_template` as `apply_operator`
    # at load (esm-spec §9.6.12), so derived references use the current name.
    op = refop === nothing ? "apply_operator" : String(refop)
    return RefLayer(
        reg, Dict{String, _DefInfo}(), OrderedDict{String, Any}(), op,
        Dict{Tuple{String, String}, ASTExpr}(), Dict{Any, Vector{Band}}(),
        Dict{String, Any}(), Dict{Any, Tuple{Vector{Int}, Vector{Int}}}(), index_sets
    )
end

_entry_get(d::AbstractDict, k) = get(d, k, nothing)
_entry_get(d, k) = hasproperty(d, Symbol(k)) ? getproperty(d, Symbol(k)) : nothing

# ── the current layer ───────────────────────────────────────────────────────
#
# The derivative table (`dscalar`) meets references deep in a pass that
# carries no view; the pass's entry installs its view's layer for its
# duration.

function _with_layer(f, L)
    L === nothing && return f()
    old = get(task_local_storage(), :esad_ref_layer, nothing)
    task_local_storage(:esad_ref_layer, L)
    try
        return f()
    finally
        task_local_storage(:esad_ref_layer, old)
    end
end

function _layer()
    L = get(task_local_storage(), :esad_ref_layer, nothing)
    L === nothing && throw(
        BandError(
            "an operator reference (`apply_operator`) reached the band calculus " *
                "without the registry of the system it belongs to; differentiate a " *
                "SysView (`sysview`) of that system"
        )
    )
    return L::RefLayer
end

# The parameters of the definition whose body is being analysed. Inside a
# body two parameters are independent inputs, so a read of one array
# parameter at a subscript that depends on another is piecewise-constant in
# it (the `index` rule of the derivative table).
_defmode_params() = get(task_local_storage(), :esad_ref_defparams, nothing)

function _with_defparams(f, ps)
    old = _defmode_params()
    task_local_storage(:esad_ref_defparams, ps)
    try
        return f()
    finally
        task_local_storage(:esad_ref_defparams, old)
    end
end

_with_defmode(f, params) = _with_defparams(f, Set{String}(params))
_no_defmode(f) = _with_defparams(f, nothing)

# ── definitions ─────────────────────────────────────────────────────────────

# `<D>.d_<p>` (a scalar definition's derivative), `<D>.d_<p>.b<k>@<shape>`
# (band k of an array definition's at one binding shape, `_shape_tag`) or
# `<D>.d_<p>.s<k>@<shape>` (band k of a scalar definition's with respect to
# an array parameter: its sites, §3.6).
const _DERIVED_NAME = r"^(.+)\.d_([^.@]+)(?:\.([bs])(\d+)@(.*))?$"

# The definition `name`: a registry entry, or a derived definition — one this
# layer wrote, or one a document written from another view of the same
# system names (regenerated from its name alone: `<D>.d_<p>` from D's body,
# `<D>.d_<p>.b<k>@<shape>` from D's bands at the binding shape it names).
function _def!(L::RefLayer, name::AbstractString)::_DefInfo
    name = String(name)
    d = get(L.defs, name, nothing)
    d === nothing || return d
    if haskey(L.registry, name)
        entry = L.registry[name]
        params = String[String(p) for p in something(_entry_get(entry, "params"), Any[])]
        body = expression_from_json(_entry_get(entry, "body"))
        d = _DefInfo(
            name, params, body, entry, false, Set{String}(), Set{String}(),
            Set{String}(), 0
        )
        L.defs[name] = d
        _roles!(L, d)
        return d
    end
    m = match(_DERIVED_NAME, name)
    m === nothing && throw(
        BandError("operator reference to `$name`, which the system does not define")
    )
    base = _def!(L, m.captures[1])
    p = String(m.captures[2])
    p in base.params || throw(
        BandError("operator reference to `$name`: `$(base.name)` has no parameter `$p`")
    )
    if m.captures[3] === nothing
        _dparam!(L, base, p)
    else
        # the binding shape is the name's own: a derived band definition binds
        # only the parameters its coefficient reads
        _def_bands!(L, base, _parse_shape_tag(base, String(m.captures[5]), name), p, true)
    end
    d = get(L.defs, name, nothing)
    d === nothing && throw(
        BandError(
            "derived definition `$name` is not produced by differentiating " *
                "`$(base.name)` at this use"
        )
    )
    return d
end

# Each parameter's roles in `d`'s body (and through the references it makes,
# where a parameter takes the roles of the callee parameter it is bound to),
# and the body's output rank.
function _roles!(L::RefLayer, d::_DefInfo)
    ps = Set{String}(d.params)
    # a structural field holds a name as a string, or (a range bound in the
    # typed IR) as a `VarExpr`
    function scal!(x)
        x isa AbstractString && String(x) in ps && push!(d.structural, String(x))
        x isa VarExpr && x.name in ps && push!(d.structural, x.name)
        x isa AbstractVector && foreach(scal!, x)
        x isa AbstractDict && foreach(scal!, values(x))
        x isa EarthSciAST.IndexSetRef && scal!(x.from)
        return
    end
    function walk(e, pos::Symbol)
        if e isa VarExpr
            e.name in ps && push!(pos == :array ? d.array : d.value, e.name)
            return
        end
        e isa OpExpr || return
        for f in (:wrt, :dim, :int_var)
            scal!(getfield(e, f))
        end
        e.ranges === nothing || scal!(e.ranges)
        e.regions === nothing || scal!(e.regions)
        if _is_ref(e)
            callee = _def!(L, e.name)
            for (q, b) in something(e.bindings, Dict{String, ASTExpr}())
                if b isa VarExpr && b.name in ps
                    q in callee.value && push!(d.value, b.name)
                    q in callee.array && push!(d.array, b.name)
                    q in callee.structural && push!(d.structural, b.name)
                else
                    walk(b, q in callee.array ? :array : :value)
                end
            end
            return
        elseif e.op == "index"
            walk(e.args[1], :array)
            for x in e.args[2:end]
                walk(x, :value)
            end
            return
        elseif e.op == "faq"
            for x in e.args
                walk(x, :array)
            end
            e.expr_body === nothing || walk(e.expr_body, :value)
            e.filter === nothing || walk(e.filter, :value)
            return
        elseif e.op == "makearray"
            for x in something(e.values, ASTExpr[])
                walk(x, :array)
            end
            return
        end
        map_children(c -> (walk(c, pos); c), e)
        return
    end
    walk(d.body, :value)
    d.rank = _body_rank(L, d.body)
    return d
end

# The output rank of an expression: a `faq`'s string output indices, a
# `makearray`'s region rank, an array reference's definition's rank; an
# elementwise node takes its operands' largest. A bare name reads as a
# scalar here (a body's array parameters are read through `index`).
function _body_rank(L::RefLayer, e)::Int
    e isa OpExpr || return 0
    if e.op == "faq"
        return count(x -> x isa AbstractString, something(e.output_idx, Any[]))
    elseif e.op == "makearray"
        return (e.regions === nothing || isempty(e.regions)) ? 0 : length(e.regions[1])
    elseif _is_ref(e)
        return _def!(L, e.name).rank
    elseif e.op == "index"
        return 0
    end
    r = 0
    map_children(c -> (r = max(r, _body_rank(L, c)); c), e)
    return r
end

# ── derived definitions ─────────────────────────────────────────────────────

# A reference node to `name`.
_ref_node(L::RefLayer, name::String, bindings::Dict{String, ASTExpr}) =
    OpExpr(L.refop, ASTExpr[]; name = name, bindings = bindings)

_identity_bindings(params) = Dict{String, ASTExpr}(p => VarExpr(p) for p in params)

# Register the derived definition `name` (params, body). A name already
# registered must carry the same body (band names carry their binding shape,
# `_shape_tag`, so two shapes never share one).
function _register_derived!(L::RefLayer, name::String, params::Vector{String}, body::ASTExpr, what::String)
    haskey(L.registry, name) && throw(
        BandError(
            "the derived definition name `$name` ($what) is already a definition of " *
                "the system"
        )
    )
    old = get(L.defs, name, nothing)
    if old !== nothing
        (old.params == params && skey(old.body) == skey(body)) || throw(
            BandError(
                "the derived definition `$name` ($what) was produced twice with " *
                    "different bodies"
            )
        )
        return old
    end
    entry = OrderedDict{String, Any}(
        "params" => Any[params...],
        "body" => EarthSciAST.serialize_expression(body),
        "description" => what,
        "reference_only" => true,
    )
    d = _DefInfo(name, params, body, entry, true, Set{String}(), Set{String}(), Set{String}(), 0)
    L.defs[name] = d
    L.derived[name] = entry
    _roles!(L, d)
    return d
end

# The parameters of `params` a derived body reads (in their order): a
# derived definition takes only those. An argument the body never reads would
# still be bound at every use, and a whole array bound there inside a `faq`
# body is refused as a bare array read (esm-spec §4.3.1).
function _used_params(body::ASTExpr, params::Vector{String})
    used = Set{String}()
    function scal(x)
        x isa AbstractString && push!(used, String(x))
        x isa VarExpr && push!(used, x.name)
        x isa AbstractVector && foreach(scal, x)
        x isa AbstractDict && foreach(scal, values(x))
        x isa EarthSciAST.IndexSetRef && push!(used, x.from)
        return
    end
    function walk(e)
        if e isa VarExpr
            push!(used, e.name)
        elseif e isa OpExpr
            for f in (:wrt, :dim, :int_var)
                scal(getfield(e, f))
            end
            e.ranges === nothing || scal(e.ranges)
            e.regions === nothing || scal(e.regions)
            map_children(c -> (walk(c); c), e)
        end
        return
    end
    walk(body)
    return String[p for p in params if p in used]
end

# A generic derivative or coefficient (an expression over a definition's
# parameter names) written out: a leaf (a literal or one name) as itself, a
# lone reference as itself, anything else as a reference to a new derived
# definition (§9.6.3 constraint 7 writes a single-leaf body the same way).
_leafy(e) = !(e isa OpExpr) || _is_ref(e)

"""
    _dparam!(L, D, p) -> ASTExpr

`∂D/∂p` for a value parameter `p` of a scalar definition `D`, generic in D's
parameter names: `0`, a leaf, or a reference `D.d_p(params…)` to the derived
definition holding it (registered once, memoized).
"""
function _dparam!(L::RefLayer, D::_DefInfo, p::String)::ASTExpr
    hit = get(L.dparam, (D.name, p), nothing)
    hit === nothing || return hit
    D.rank == 0 || throw(
        BandError("`$(D.name)` is an array definition; its derivative is a band list")
    )
    body = _with_defmode(D.params) do
        _simp(dscalar(D.body, Site(VarExpr(p))))
    end
    T = if _leafy(body)
        body
    else
        name = D.name * ".d_" * p
        ps = _used_params(body, D.params)
        _register_derived!(L, name, ps, body, "∂$(D.name)/∂$(p)")
        _ref_node(L, name, _identity_bindings(ps))
    end
    L.dparam[(D.name, p)] = T
    return T
end

# A generic expression `T` over a definition's parameter names, at a use
# binding them to `B`: a name is its argument, a reference has `B`
# substituted into its bindings. Nothing else is ever substituted.
function _bind(T::ASTExpr, B::AbstractDict)::ASTExpr
    T isa VarExpr && return get(B, T.name, T)
    _is_ref(T) || return T
    nb = Dict{String, ASTExpr}(k => _subst_names(v, B) for (k, v) in T.bindings)
    return EarthSciAST.reconstruct(T; bindings = nb)
end

# Simultaneous substitution of names into a small index-level expression
# (bindings, column indices, guards); a name bound by a `faq` inside is
# shadowed.
_subst_names(e::ASTExpr, B::AbstractDict) =
    isempty(B) ? e : _subst_free(e, Dict{String, ASTExpr}(String(k) => v for (k, v) in B))

# ── the derivative-table row ────────────────────────────────────────────────

# `dscalar` of a scalar reference: Σ over the value parameters whose argument
# holds the site of D.d_p(B) · ∂B[p]. An array parameter whose argument holds
# the site would need the body's reads of it; refused (never a silent 0).
function _dref(e::OpExpr, s::Site, d)
    L = _layer()
    D = _def!(L, e.name)
    D.rank == 0 || throw(
        DerivativeRuleError(
            "an array-valued operator reference (`$(D.name)`) in a scalar position; " *
                "the band calculus reads it as an intermediate array"
        )
    )
    B = e.bindings
    terms = ASTExpr[]
    for p in D.params
        b = get(B, p, nothing)
        b === nothing && throw(
            BandError("operator reference to `$(D.name)` does not bind parameter `$p`")
        )
        db = d(b)
        iszero_lit(db) && continue
        (p in D.array) && throw(
            DerivativeRuleError(
                "operator `$(D.name)` reads its parameter `$p` as an array, and the " *
                    "argument bound to it depends on the differentiation site " *
                    "$(skey(s.expr)); differentiating through an array argument of a " *
                    "scalar definition is not supported"
            )
        )
        T = _dparam!(L, D, p)
        iszero_lit(T) && continue
        push!(terms, mul(_bind(T, B), db))
    end
    return add(terms...)
end

# ── array definitions: shapes, extents, bands ───────────────────────────────

# The integer extents of an array expression in `ctx` (`nothing` when not
# statically known).
function _expr_shape(L::RefLayer, e::ASTExpr, ctx::Ctx)
    if e isa VarExpr
        return get(ctx.shapes, e.name, Int[])
    elseif e isa OpExpr && e.op == "const"
        v = e.value
        v isa AbstractVector || return Int[]
        dims = Int[]
        while v isa AbstractVector
            push!(dims, length(v))
            isempty(v) && break
            v = v[1]
        end
        return dims
    elseif _is_ref(e)
        D = _def!(L, e.name)
        D.rank == 0 && return Int[]
        return first(_ref_extents(L, D, e.bindings, ctx))
    elseif e isa OpExpr && e.op in ("faq", "makearray")
        return first(_body_extents(e, ctx))
    elseif e isa OpExpr && e.op == "index"
        return Int[]
    elseif e isa OpExpr
        for c in e.args
            s = _expr_shape(L, c, ctx)
            (s === nothing || !isempty(s)) && return s
        end
        return Int[]
    end
    return Int[]
end

# A literal a structural parameter is bound to: an integer, a float, or a
# name (an index-set name, an axis name).
function _literal(b::ASTExpr)
    b isa IntExpr && return Int(b.value)
    b isa NumExpr && return isinteger(b.value) ? Int(b.value) : b.value
    b isa VarExpr && return b.name
    return nothing
end

# The binding shape of a use of `D`: each array parameter's extents and each
# structural parameter's literal, in parameter order.
function _shapekey(L::RefLayer, D::_DefInfo, B::AbstractDict, ctx::Ctx)
    key = Any[]
    for p in D.params
        if p in D.structural
            lv = _literal(B[p])
            lv === nothing && throw(
                BandError(
                    "operator `$(D.name)`: parameter `$p` fills a structural field and " *
                        "must be bound to a literal, not $(skey(B[p]))"
                )
            )
            push!(key, p => lv)
        elseif p in D.array
            s = _expr_shape(L, B[p], ctx)
            s === nothing && throw(
                BandError("operator `$(D.name)`: the extents of argument `$p` are not static")
            )
            push!(key, p => Tuple(s))
        end
    end
    return Tuple(key)
end

# The body of `D` at shape key `sk`: structural parameters replaced by their
# literals, everywhere they occur (the analysis needs its ranges and regions
# as numbers). The definition itself is unchanged.
function _specialized(D::_DefInfo, sk)
    lits = Dict{String, Any}(String(k) => v for (k, v) in sk if !(v isa Tuple))
    isempty(lits) && return D.body
    return _specialize(D.body, lits)
end

function _specialize(e::ASTExpr, lits::Dict{String, Any})::ASTExpr
    if e isa VarExpr
        haskey(lits, e.name) || return e
        v = lits[e.name]
        return v isa Real ? lit(v) : VarExpr(String(v))
    end
    e isa OpExpr || return e
    function sv(x)
        x isa AbstractString && haskey(lits, String(x)) && return lits[String(x)]
        x isa VarExpr && haskey(lits, x.name) && return _specialize(x, lits)
        return x
    end
    function sr(r)
        r isa AbstractVector && return Any[sv(x) for x in r]
        r isa EarthSciAST.IndexSetRef && return EarthSciAST.IndexSetRef(String(sv(r.from)); of = r.of)
        r isa AbstractDict && return Dict{String, Any}(String(k) => sv(x) for (k, x) in r)
        return r
    end
    kw = Dict{Symbol, Any}()
    for f in (:wrt, :dim, :int_var)
        x = getfield(e, f)
        x isa AbstractString && haskey(lits, x) && (kw[f] = String(lits[x]))
    end
    e.ranges === nothing ||
        (kw[:ranges] = Dict{String, Any}(String(k) => sr(r) for (k, r) in e.ranges))
    e.regions === nothing || (
        kw[:regions] = [[Any[sv(x) for x in r] for r in reg] for reg in e.regions]
    )
    e2 = isempty(kw) ? e : EarthSciAST.reconstruct(e; kw...)
    return map_children(c -> _specialize(c, lits), e2)
end

# The extents and origins (first index value) of an array body: a `faq`'s
# output ranges, a `makearray`'s region positions (origin 1).
function _body_extents(e::OpExpr, ctx::Ctx)
    if e.op == "faq"
        ranges = something(e.ranges, Dict{String, Any}())
        ext = Int[]; org = Int[]
        for x in something(e.output_idx, Any[])
            x isa AbstractString || continue
            haskey(ranges, String(x)) || throw(
                BandError("an array definition's output index `$x` has no range")
            )
            lo, hi = _range_of(ranges[String(x)], ctx)
            push!(ext, hi - lo + 1); push!(org, lo)
        end
        return ext, org
    elseif e.op == "makearray"
        (e.regions === nothing || isempty(e.regions)) && return Int[], Int[]
        n = length(e.regions[1])
        hi = zeros(Int, n)
        for reg in e.regions, (k, r) in enumerate(reg)
            hi[k] = max(hi[k], _region_bound(r[2]))
        end
        return hi, ones(Int, n)
    elseif _is_ref(e) && ctx.refs !== nothing
        # a body that is itself a reference to an array definition: that
        # definition's extents at the reference's bindings
        L = ctx.refs
        D2 = _def!(L, e.name)
        D2.rank > 0 || throw(BandError("an array definition's body is a scalar reference"))
        return _ref_extents(L, D2, e.bindings, ctx)
    end
    throw(BandError("an array definition's body is a `$(e.op)`, not a faq or makearray"))
end

function _ref_extents(L::RefLayer, D::_DefInfo, B::AbstractDict, ctx::Ctx)
    sk = _shapekey(L, D, B, ctx)
    hit = get(L.extents, (D.name, sk), nothing)
    hit === nothing || return hit
    body = _specialized(D, sk)
    body isa OpExpr || throw(BandError("`$(D.name)`: an array definition's body is a leaf"))
    return L.extents[(D.name, sk)] = _body_extents(body, _def_ctx(L, D, sk))
end

# The band context of `D`'s body at shape key `sk`: the system's index sets,
# and D's array parameters at their bound extents.
_def_ctx(L::RefLayer, D::_DefInfo, sk) = Ctx(
    L.index_sets,
    Dict{String, Vector{Int}}(
        String(k) => collect(Int, v) for (k, v) in sk if v isa Tuple && !isempty(v)
    ),
    Set{String}(), L
)

"""
    _def_bands!(L, D, sk, p, co) -> Vector{Band}

The bands of array definition `D`'s body with respect to its parameter `p`
at binding shape `sk`, in D's own index space and generic in D's parameter
names (memoized). With `co`, each coefficient is a leaf, a reference, or a
reference `<D>.d_<p>.b<k>@<shape>(params…, index names…)` to the derived definition
holding it.
"""
function _def_bands!(L::RefLayer, D::_DefInfo, sk, p::String, co::Bool)::Vector{Band}
    key = (D.name, sk, p, co)
    hit = get(L.defbands, key, nothing)
    hit === nothing || return hit
    # a scalar definition has bands only with respect to an array parameter
    # (its sites, §3.6); a value parameter's derivative is `D.d_p`
    D.rank == 0 && !(p in D.array) && throw(BandError("`$(D.name)` is a scalar definition"))
    body = _specialized(D, sk)
    ctx = _def_ctx(L, D, sk)
    ext = D.rank == 0 ? Int[] : _body_extents(body, ctx)[1]
    raw = _with_defmode(D.params) do
        D.rank > 0 && return _ref_bands(body, p, ctx, ext, co)
        # a scalar body's sites: its value parameters index like names
        task_local_storage(:esad_index_params, Set{String}(D.value)) do
            _ref_bands(body, p, ctx, ext, co)
        end
    end
    bs = Band[normalize_band(b) for b in merge_bands(raw)]
    out = Band[]
    for (k, b) in enumerate(bs)
        coef = co ? _generic_coef!(L, D, sk, p, k, b) : nothing
        push!(out, _rebuild(b; coef = coef))
    end
    L.defbands[key] = out
    return out
end

# A band coefficient of `D`'s body written out generically: a leaf or a lone
# reference as itself, anything else as `<D>.d_<p>.b<k>@<shape>` over the parameters
# of D it reads and the band's row and contracted index names it reads.
function _generic_coef!(L::RefLayer, D::_DefInfo, sk, p::String, k::Int, b::Band)
    c = b.coef
    names = Set{String}(r for r in b.ridx if r isa String)
    for (n, _, _) in b.contracted
        push!(names, n)
    end
    fv = free_variables(c)
    for n in fv
        (n in names || n in D.params) || throw(
            BandError(
                "a band coefficient of `$(D.name)` reads `$n`, which is neither a " *
                    "parameter nor an index of the band (the body is not closed)"
            )
        )
    end
    _leafy(c) && return c
    extra = sort!(String[n for n in fv if n in names])
    for n in extra
        n in D.params && throw(
            BandError("`$(D.name)`: index name `$n` is also a parameter name")
        )
    end
    name = D.name * ".d_" * p * (D.rank == 0 ? ".s" : ".b") * string(k) * "@" * _shape_tag(D, sk)
    params = vcat(_used_params(c, D.params), extra)
    _register_derived!(L, name, params, c, "band $k of ∂$(D.name)/∂$(p)")
    L.bandkey[name] = sk
    return _ref_node(L, name, _identity_bindings(params))
end

# The binding shape `sk` as a derived band name carries it: per array or
# structural parameter of D, in parameter order, its argument's extents
# joined by `x` or its literal, comma-separated. Band k at one shape and band
# k at another are different definitions (a boundary band empty at one
# extent shifts the numbering), so the name says which shape it is of.
function _shape_tag(D::_DefInfo, sk)
    parts = String[]
    for (p, v) in sk
        push!(
            parts, if v isa Tuple
                join(string.(v), "x")
            elseif v isa Integer || v isa AbstractString
                string(v)
            else
                throw(
                    BandError(
                        "operator `$(D.name)`: structural parameter `$p` is bound to " *
                            "the non-integer $v"
                    )
                )
            end
        )
    end
    return join(parts, ",")
end

# The shape key a derived band name's `<shape>` tag denotes (`_shape_tag`
# read back): one part per structural or array parameter of D, in order.
function _parse_shape_tag(D::_DefInfo, tag::String, name::String)
    parts = split(tag, ","; keepempty = true)
    key = Any[]
    i = 0
    bad() = BandError("derived definition `$name`: `$tag` is not a binding shape of `$(D.name)`")
    for p in D.params
        (p in D.structural || p in D.array) || continue
        i += 1
        i <= length(parts) || throw(bad())
        x = String(parts[i])
        if p in D.structural
            push!(key, p => something(tryparse(Int, x), x))
        else
            dims = isempty(x) ? Int[] : [tryparse(Int, y) for y in split(x, "x")]
            any(isnothing, dims) && throw(bad())
            push!(key, p => Tuple(Int.(dims)))
        end
    end
    i == length(parts) || (i == 0 && tag == "") || throw(bad())
    return Tuple(key)
end

# ── uses of array definitions: the intermediate and the composition ─────────

# Placeholder names: `§` occurs in no ESM name.
_pseudo_name(k::Int) = "§ref$k"

# `e` with every reference read as an intermediate replaced by a placeholder
# array (the same reference under the same scope, structurally, by the same
# placeholder), and the placeholders: name → `_Pseudo`. A reference is read
# as an intermediate when it is array-valued, or when it is scalar and binds
# an array parameter to an argument that reads `v` (§3.6; with `v ===
# nothing`, array-valued ones only). A placeholder's cells are, first, the
# values of the indices bound around the reference that its bindings read
# (its scope dimensions, in the order they are bound: outer `faq` first,
# within one its ranges in code-point order then its output indices, as
# `_binders`), then the index values the reference's own array takes (a
# `faq` body's own range values, a `makearray`'s positions). A read
# `index(ref, i…)` becomes `index(placeholder, s…, i…)`; a scalar reference
# with scope dimensions `index(placeholder, s…)`.
struct _Pseudo
    ref::OpExpr
    ext::Vector{Int}
    org::Vector{Int}
    scope::Vector{String}     # the scope dimensions' index names
end

function _abstract_refs(L::RefLayer, e::ASTExpr, ctx::Ctx; v = nothing, scope = Pair{String, Any}[])
    found = OrderedDict{Any, Tuple{OpExpr, String, Vector{Pair{String, Any}}}}()
    out = _abstract!(L, e, ctx, found, collect(Pair{String, Any}, scope), v)
    ps = OrderedDict{String, _Pseudo}()
    for (_, (r, n, sc)) in found
        D = _def!(L, r.name)
        ext, org = D.rank > 0 ? _ref_extents(L, D, r.bindings, ctx) : (Int[], Int[])
        sext = Int[]; sorg = Int[]
        for (nm, spec) in sc
            lo, hi = _scope_range(spec, nm, D, ctx)
            push!(sext, hi - lo + 1); push!(sorg, lo)
        end
        ps[n] = _Pseudo(r, vcat(sext, ext), vcat(sorg, org), String[first(x) for x in sc])
    end
    return out, ps
end

# The static `[lo, hi]` of a scope index's range spec.
function _scope_range(spec, nm::String, D::_DefInfo, ctx::Ctx)
    if spec isa AbstractDict || spec isa EarthSciAST.IndexSetRef
        from = spec isa AbstractDict ? spec["from"] : spec.from
        from isa AbstractString && return (1, _set_size(String(from), ctx))
    elseif spec isa AbstractVector && length(spec) == 2 &&
            all(x -> x isa Integer || (x isa IntExpr) || (x isa NumExpr && isinteger(x.value)), spec)
        lit(x) = x isa Integer ? Int(x) : Int(x.value)
        return (lit(spec[1]), lit(spec[2]))
    end
    throw(
        BandError(
            "operator reference to `$(D.name)` binds an argument that reads the index " *
                "`$nm` bound around it, whose range is not a static interval or index " *
                "set; not supported"
        )
    )
end

# Does `D` (scalar) bind an array parameter to an argument that reads `v`?
_reads_through_array(D::_DefInfo, B, v) = v !== nothing &&
    any(p -> p in D.array && v in free_variables(B[p]), D.params)

function _abstract!(L, e::ASTExpr, ctx, found, scope::Vector{Pair{String, Any}}, v)
    e isa OpExpr || return e
    # `index(ref, i…)` of an array-valued reference: the reference's own
    # indices follow its scope dimensions
    if e.op == "index" && _is_ref(e.args[1]) && _def!(L, e.args[1].name).rank > 0
        r = e.args[1]
        own = ASTExpr[_abstract!(L, x, ctx, found, scope, v) for x in e.args[2:end]]
        n, sc = _placeholder!(L, r, found, scope)
        return op("index", VarExpr(n), ASTExpr[VarExpr(first(x)) for x in sc]..., own...)
    end
    if _is_ref(e)
        D = _def!(L, e.name)
        if D.rank > 0 || _reads_through_array(D, e.bindings, v)
            n, sc = _placeholder!(L, e, found, scope)
            isempty(sc) && return VarExpr(n)
            D.rank > 0 && throw(
                BandError(
                    "the array-valued operator reference `$(D.name)` binds an argument " *
                        "that reads an index bound around it " *
                        "($(join(first.(sc), ", "))) and is read whole, not through " *
                        "`index`; not supported"
                )
            )
            return op("index", VarExpr(n), ASTExpr[VarExpr(first(x)) for x in sc]...)
        end
    end
    if e.op == "faq" && e.ranges !== nothing
        scope = copy(scope)
        for b in _binders(e)
            filter!(x -> first(x) != b, scope)
            push!(scope, b => get(e.ranges, b, nothing))
        end
    end
    return map_children(c -> _abstract!(L, c, ctx, found, scope, v), e)
end

# The placeholder of reference `r` under `scope`: its name and the scope
# dimensions it has (the scope indices its bindings read, in scope order).
function _placeholder!(L, r::OpExpr, found, scope)
    fv = free_variables(r)
    sc = Pair{String, Any}[x for x in scope if first(x) in fv]
    k = (skey(r), [first(x) for x in sc], [string(last(x)) for x in sc])
    hit = get(found, k, nothing)
    hit === nothing && (hit = found[k] = (r, _pseudo_name(length(found) + 1), sc))
    return hit[2], hit[3]
end

"""
    _ref_bands(rhs, v, ctx, shape_u, co) -> Vector{Band}

[`bands`](@ref) of an expression that may hold operator references. Scalar
references are differentiated by the derivative table's reference row
(`_dref`); every array reference is read as an intermediate array whose own
bands (`_ref_value_bands`) are composed with the bands of `rhs` with respect
to it. Not merged (the caller merges per row and column variable).
"""
function _ref_bands(rhs::ASTExpr, v::String, ctx::Ctx, shape_u::Vector{Int}, co::Bool)
    L = ctx.refs
    L === nothing && return bands(rhs, v, ctx; shape_u = shape_u, coefficients = co)
    return _with_layer(L) do
        rhs2, ps = _abstract_refs(L, rhs, ctx; v = v)
        isempty(ps) && return bands(rhs, v, ctx; shape_u = shape_u, coefficients = co)
        shapes = copy(ctx.shapes); origins = copy(ctx.origins)
        for (n, P) in ps
            shapes[n] = P.ext; origins[n] = P.org
        end
        ctx2 = Ctx(ctx.index_sets, shapes, ctx.params, L, origins, ctx.axes)
        out = Band[]
        fv = free_variables(rhs2)
        v in fv && append!(out, bands(rhs2, v, ctx2; shape_u = shape_u, coefficients = co))
        for (n, P) in ps
            n in fv || continue
            v in free_variables(P.ref) || continue
            inner = _ref_value_bands(L, P.ref, v, ctx, co)
            isempty(inner) && continue
            inner = Band[_one_hot_band(_with_scope(b, P), v, ctx) for b in inner]
            outer = bands(rhs2, n, ctx2; shape_u = shape_u, coefficients = co)
            append!(out, _compose(outer, inner, v, ctx, co))
        end
        return Band[_restore(b, ps) for b in out]
    end
end

# A band of a reference's value with its placeholder's scope dimensions put
# first: rows over their ranges, named by the scope indices (the bound
# arguments already read them by those names).
function _with_scope(b::Band, P::_Pseudo)
    isempty(P.scope) && return b
    m = length(P.scope)
    rows = Tuple{Int, Int}[(P.org[d], P.org[d] + P.ext[d] - 1) for d in 1:m]
    return _rebuild(
        b; rows = vcat(rows, b.rows),
        ridx = Union{String, Int}[P.scope..., b.ridx...]
    )
end

# A band whose column reads a value at the use (a scalar definition's
# parameter bound to data) made one-hot (`_one_hot`).
function _one_hot_band(b::Band, v::String, ctx::Ctx)
    b.coef === nothing && return b
    cidx, coef, contracted = _one_hot(b.cidx, b.coef, b.contracted, b.ridx, v, ctx)
    return _rebuild(b; cidx = cidx, coef = coef, contracted = contracted)
end

# A band with every placeholder read back as its reference: a read
# `index(placeholder, a…, i…)` is the reference with its scope indices bound
# to `a…`, read at `i…` (itself when it has no own dimensions).
function _restore(b::Band, ps::AbstractDict)
    function f(x)
        x === nothing && return nothing
        return _restore_expr(x, ps)
    end
    return _rebuild(b; cidx = ASTExpr[f(c) for c in b.cidx], coef = f(b.coef), guard = f(b.guard))
end

function _restore_expr(e::ASTExpr, ps::AbstractDict)
    if e isa VarExpr
        P = get(ps, e.name, nothing)
        P === nothing && return e
        isempty(P.scope) || throw(BandError("a placeholder with scope dimensions read whole"))
        return P.ref
    end
    e isa OpExpr || return e
    if e.op == "index" && e.args[1] isa VarExpr && haskey(ps, e.args[1].name)
        P = ps[e.args[1].name]
        m = length(P.scope)
        a = ASTExpr[_restore_expr(x, ps) for x in e.args[2:end]]
        sub = Dict{String, ASTExpr}(P.scope[d] => a[d] for d in 1:m)
        r = isempty(sub) ? P.ref : _bind(P.ref, sub)
        own = a[(m + 1):end]
        return isempty(own) ? r : op("index", r, own...)
    end
    return map_children(c -> _restore_expr(c, ps), e)
end

"""
    _ref_value_bands(L, r, v, ctx, co) -> Vector{Band}

The bands of the array an array reference `r` denotes with respect to `v`:
rows are the array's cells (its own index values). For each parameter whose
argument reads `v`, the definition's bands with respect to that parameter
(`_def_bands!`, once per definition and binding shape), bound to the use's
arguments, composed with the bands of the argument itself.
"""
function _ref_value_bands(L::RefLayer, r::OpExpr, v::String, ctx::Ctx, co::Bool)
    D = _def!(L, r.name)
    B = r.bindings
    sk = _shapekey(L, D, B, ctx)
    out = Band[]
    for p in D.params
        b = B[p]
        v in free_variables(b) || continue
        (p in D.structural && !(p in D.value) && !(p in D.array)) && continue
        db = if D.rank == 0 && !(p in D.array)
            # a value parameter of a scalar definition: `D.d_p`, one band
            T = co ? _dparam!(L, D, p) : nothing
            co && iszero_lit(T) ? Band[] : Band[
                    Band(Tuple{Int, Int}[], Union{String, Int}[], ASTExpr[], T, Tuple{String, Int, Int}[], nothing),
                ]
        else
            _def_bands!(L, D, sk, p, co)
        end
        isempty(db) && continue
        shape_p = p in D.array ? collect(Int, Dict(sk)[p]) : Int[]
        inner = _no_defmode() do
            _ref_bands(b, v, ctx, shape_p, co)
        end
        isempty(inner) && continue
        bound = Band[_bind_band(x, B, D) for x in db]
        append!(out, _compose(bound, inner, v, ctx, co))
    end
    return out
end

# A band of `D`'s body at a use binding D's parameters to `B`: its column
# indices and guard read D's parameter names (an offset, a literal-bound
# size), substituted; its coefficient bound (`_bind`). Its own index names
# are renamed apart from the names the arguments read first.
function _bind_band(b::Band, B::AbstractDict, D::_DefInfo)
    argnames = Set{String}()
    for (_, x) in B
        union!(argnames, free_variables(x))
    end
    own = String[r for r in b.ridx if r isa String]
    append!(own, String[n for (n, _, _) in b.contracted])
    ren = Dict{String, ASTExpr}()
    taken = union(argnames, Set(own), Set(D.params))
    for n in own
        n in argnames || continue
        ren[n] = VarExpr(_fresh!(taken, n * "_r", "_r"))
    end
    if !isempty(ren)
        rn(n) = haskey(ren, n) ? ren[n].name : n
        b = Band(
            b.rows, Union{String, Int}[r isa String ? rn(r) : r for r in b.ridx],
            ASTExpr[_subst_names(c, ren) for c in b.cidx],
            b.coef === nothing ? nothing : _bind(b.coef, ren),
            Tuple{String, Int, Int}[(rn(n), lo, hi) for (n, lo, hi) in b.contracted],
            b.guard === nothing ? nothing : _subst_names(b.guard, ren),
            _rename_join(b.join, Dict{String, String}(k => v.name for (k, v) in ren))
        )
    end
    Bs = Dict{String, ASTExpr}(String(k) => x for (k, x) in B)
    # a join clause names the definition's parameters (envelope factors, key
    # columns) as strings: each is the name its argument binds, which must be
    # a variable (§3.5, §3.6)
    jm = Dict{String, String}()
    if b.join !== nothing
        for (p, x) in Bs
            x isa VarExpr ? (jm[p] = x.name) : (jm[p] = "")
        end
    end
    j = _rename_join(b.join, filter(kv -> kv[2] != "", jm))
    if j !== nothing
        bad = [p for (p, n) in jm if n == "" && occursin("\"$p\"", JSON3.write(j[1]))]
        isempty(bad) || throw(
            BandError(
                "a join of `$(D.name)` names parameter `$(bad[1])`, bound to an " *
                    "expression: a join reads variables only"
            )
        )
    end
    return _rebuild(
        b; cidx = ASTExpr[_subst_names(c, Bs) for c in b.cidx],
        coef = b.coef === nothing ? nothing : _bind(b.coef, Bs),
        guard = b.guard === nothing ? nothing : _subst_names(b.guard, Bs), join = j
    )
end

# A comparison of two integer literals, folded; `nothing` otherwise.
function _fold_cmp(c::ASTExpr)
    (c isa OpExpr && c.op in _CMP_OPS && length(c.args) == 2) || return nothing
    a, b = c.args
    (a isa IntExpr || (a isa NumExpr && isinteger(a.value))) || return nothing
    (b isa IntExpr || (b isa NumExpr && isinteger(b.value))) || return nothing
    return _cmp_truth(c.op, 1, -Int(b.value), Int(a.value))
end

"""
    _compose(outer, inner, v, ctx, co) -> Vector{Band}

The chain rule through an intermediate array `W`: `outer` are bands of an
expression with respect to `W` (their column indices are cells of `W`),
`inner` the bands of `W` with respect to `v` (rows: cells of `W`). Each pair
gives the bands of the expression with respect to `v` where the outer
column lands in the inner row box: inner row names read the outer column
indices, the coefficients multiply, contractions concatenate (the inner's
renamed apart), and the row-box membership joins the guard, decided over
the outer box as far as it can be (`_split_guard`).
"""
# Whether `e` is an `ifelse` with a literal-zero branch: a guard.
_is_guard(e) = e isa OpExpr && e.op == "ifelse" && length(e.args) == 3 &&
    (iszero_lit(e.args[2]) || iszero_lit(e.args[3]))

# `outer · inner`, with a guard of `outer` (an `ifelse` with a literal-zero
# branch, `outer` itself or the first such factor of a product) kept outside
# the product: `ifelse(c, a, 0) · y` is `ifelse(c, a · y, 0)`. The guard of a
# definition (`ifelse(s > 0, …/s, …)`) then also guards its composed
# derivative where the inner derivative `y` is NaN (spec §3.6).
function guarded_mul(outer::ASTExpr, inner::ASTExpr)
    if _is_guard(outer)
        return op("ifelse", outer.args[1], guarded_mul(outer.args[2], inner), guarded_mul(outer.args[3], inner))
    end
    if outer isa OpExpr && outer.op == "*"
        k = findfirst(_is_guard, outer.args)
        if k !== nothing
            g = outer.args[k]
            branch(x) = guarded_mul(mul(outer.args[1:(k - 1)]..., x, outer.args[(k + 1):end]...), inner)
            return op("ifelse", g.args[1], branch(g.args[2]), branch(g.args[3]))
        end
    end
    return mul(outer, inner)
end

function _compose(outer::Vector{Band}, inner::Vector{Band}, v::String, ctx::Ctx, co::Bool)
    out = Band[]
    for o in outer, b in inner
        length(o.cidx) == length(b.ridx) || throw(
            BandError(
                "composing through an intermediate array: a read of rank $(length(o.cidx)) " *
                    "of an array of rank $(length(b.ridx))"
            )
        )
        taken = Set{String}(r for r in o.ridx if r isa String)
        for (n, _, _) in o.contracted
            push!(taken, n)
        end
        for c in o.cidx
            union!(taken, free_variables(c))
        end
        o.coef === nothing || _all_names!(taken, o.coef)
        sigma = Dict{String, ASTExpr}()
        bcon = Tuple{String, Int, Int}[]
        for (n, lo, hi) in b.contracted
            f = n in taken ? _fresh!(taken, n * "_c", "_c") : (push!(taken, n); n)
            f == n || (sigma[n] = VarExpr(f))
            push!(bcon, (f, lo, hi))
        end
        conds = ASTExpr[]
        dead = false
        jnames = _join_names(b)
        tie = ASTExpr[]
        for (d, r) in enumerate(b.ridx)
            lo, hi = b.rows[d]
            if r isa String && r in jnames && !(o.cidx[d] isa VarExpr)
                # a joined row name read at a non-bare index stays a name of
                # the composite (the join reads it): contracted over its
                # row range, tied to the index by an equality (§3.5)
                f = r in taken ? _fresh!(taken, r * "_c", "_c") : (push!(taken, r); r)
                sigma[r] = VarExpr(f)
                push!(bcon, (f, lo, hi))
                push!(tie, op("==", VarExpr(f), o.cidx[d]))
            elseif r isa String
                sigma[r] = o.cidx[d]
            end
            for c in (op(">=", o.cidx[d], lit(lo)), op("<=", o.cidx[d], lit(hi)))
                t = _fold_cmp(c)
                t === true && continue
                t === false && (dead = true; break)
                push!(conds, c)
            end
            dead && break
        end
        dead && continue
        sub(x) = x === nothing ? nothing : _subst_names(x, sigma)
        # a join travels with the band it gates (§3.5): the inner's names
        # renamed as the composite reads them (a row name must land on a
        # bare name); two joins in one composite are refused
        jn = o.join
        if b.join !== nothing
            o.join === nothing || throw(BandError("a band through two joined faq nodes is not supported"))
            jm = Dict{String, String}()
            for n in _join_names(b)
                x = get(sigma, n, VarExpr(n))
                x isa VarExpr || throw(
                    BandError("a joined band composed at a non-bare index ($(_json(x))) is not supported")
                )
                jm[n] = x.name
            end
            jn = _rename_join(b.join, jm)
        end
        g = _conj(o.guard, sub(b.guard))
        isempty(conds) || (g = _conj(g, _conj(conds)))
        isempty(tie) || (g = _conj(g, _conj(tie)))
        cidx = ASTExpr[_canon_lits(simplify(sub(c))) for c in b.cidx]
        # The inner coefficient is any expression (an argument's band, or a
        # composed band `ref * argcoef`), not only a generic leaf: substitute
        # its row and contracted names everywhere, references' bindings included.
        coef = co ? guarded_mul(o.coef, _subst_names(b.coef, sigma)) : nothing
        contracted = vcat(o.contracted, bcon)
        names = String[r for r in o.ridx if r isa String]
        append!(names, String[n for (n, _, _) in contracted])
        box = Tuple{Int, Int}[o.rows[k] for (k, r) in enumerate(o.ridx) if r isa String]
        append!(box, Tuple{Int, Int}[(lo, hi) for (_, lo, hi) in contracted])
        pieces = g === nothing ? [(box, nothing)] : _split_guard(g, names, box)
        nr = count(r -> r isa String, o.ridx)
        c = co ? _simp(coef) : nothing
        co && iszero_lit(c) && continue
        for (cell, g2) in pieces
            rows = copy(o.rows)
            j = 0
            for (k, r) in enumerate(o.ridx)
                r isa String || continue
                j += 1
                rows[k] = cell[j]
            end
            con = Tuple{String, Int, Int}[
                (n, cell[nr + i]...) for (i, (n, _, _)) in enumerate(contracted)
            ]
            push!(out, normalize_band(Band(rows, o.ridx, cidx, c, con, g2, jn)))
        end
    end
    return out
end

# ── closing a model's own block ─────────────────────────────────────────────
#
# A model's `operators` bodies may read names of the model freely: the
# reference's site supplies them (esm-spec §9.6.4 rules 1–2). `flatten`
# closes them (rule 7): every body's free names — its own, and those of the
# definitions it references — become trailing parameters, sorted, and every
# reference binds each to its own name. The view of a single model is closed
# the same way, so every analysis meets closed bodies whatever the input.

# (closed registry, name → the free names its references must bind).
function _close_registry(raw)
    reg = OrderedDict{String, Any}(String(k) => v for (k, v) in raw)
    bodies = Dict{String, ASTExpr}()
    params = Dict{String, Vector{String}}()
    for (n, d) in reg
        b = _entry_get(d, "body")
        b === nothing && continue
        bodies[n] = expression_from_json(b)
        params[n] = String[String(p) for p in something(_entry_get(d, "params"), Any[])]
    end
    free = Dict{String, Vector{String}}()
    function fr(n::String, stack::Vector{String})
        hit = get(free, n, nothing)
        hit === nothing || return hit
        n in stack && throw(BandError("operator `$n` references itself"))
        haskey(bodies, n) || return String[]
        ps = Set{String}(params[n])
        acc = Set{String}(x for x in free_variables(bodies[n]) if !(x in ps))
        _foreach_ref(bodies[n]) do r
            for x in fr(String(r.name), vcat(stack, n))
                x in ps || push!(acc, x)
            end
        end
        return free[n] = sort!(collect(acc))
    end
    closing = Dict{String, Vector{String}}(n => fr(n, String[]) for n in keys(bodies))
    all(isempty, values(closing)) && return reg, Dict{String, Vector{String}}()
    out = OrderedDict{String, Any}()
    for (n, d) in reg
        if !haskey(bodies, n)
            out[n] = d
            continue
        end
        e = OrderedDict{String, Any}(String(k) => v for (k, v) in pairs(d))
        e["params"] = Any[params[n]..., (x for x in closing[n] if !(x in params[n]))...]
        e["body"] = EarthSciAST.serialize_expression(_close_refs(bodies[n], closing))
        out[n] = e
    end
    return out, closing
end

# Every reference node of `e` (not looking into bodies, which are elsewhere).
function _foreach_ref(f, e::ASTExpr)
    e isa OpExpr || return
    _is_ref(e) && f(e)
    map_children(c -> (_foreach_ref(f, c); c), e)
    return
end

# `e` with every reference binding its definition's closing names to
# themselves (a binding already given is kept).
function _close_refs(e::ASTExpr, closing)::ASTExpr
    e isa OpExpr || return e
    e2 = map_children(c -> _close_refs(c, closing), e)
    _is_ref(e2) || return e2
    names = get(closing, String(e2.name), String[])
    isempty(names) && return e2
    b = Dict{String, ASTExpr}(something(e2.bindings, Dict{String, ASTExpr}()))
    for x in names
        haskey(b, x) || (b[x] = VarExpr(x))
    end
    return EarthSciAST.reconstruct(e2; bindings = b)
end

_close_equation(eq, closing) = EarthSciAST.Equation(
    _close_refs(eq.lhs, closing), _close_refs(eq.rhs, closing); _comment = eq._comment
)

# ── carrying definitions into derived documents ─────────────────────────────
#
# Every derived document writes references as references, so each top-level
# model whose subtree holds one carries an `operators` block with the
# transitive closure of the definitions its references reach (esm-spec §9.6.4
# rule 5; the planner's derived documents do the same): the system's own
# entries as its registry holds them (bodies closed), in registry order, then
# the derived definitions this package wrote, in name order, each marked
# `reference_only` (§9.6.1: an engine-generated entry). Reloading and
# flattening the document dedupes equal entries across models (rule 7).

# The (name, bindings) of every reference in a serialized subtree.
function _raw_refs!(acc, x)
    if x isa AbstractDict
        o = get(x, "op", nothing)
        if o isa AbstractString && o in _REF_OPS
            push!(acc, (String(x["name"]), get(x, "bindings", nothing)))
        end
        for (_, v) in x
            _raw_refs!(acc, v)
        end
    elseif x isa AbstractVector
        for v in x
            _raw_refs!(acc, v)
        end
    end
    return acc
end

# The definitions a set of references reaches, transitively through bodies,
# each made present in `L` (a derived definition another view wrote is
# regenerated from its name).
function _reached!(L::RefLayer, refs)
    seen = Set{String}()
    work = collect(refs)
    while !isempty(work)
        name, _ = popfirst!(work)
        name in seen && continue
        d = _with_layer(() -> _def!(L, name), L)
        push!(seen, name)
        for r in _raw_refs!(Tuple{String, Any}[], _entry_get(d.entry, "body"))
            r[1] in seen || push!(work, r)
        end
    end
    return seen
end

# The `operators` block of the definitions `names`.
function _operators_block(L::RefLayer, names::Set{String})
    block = OrderedDict{String, Any}()
    for (n, entry) in L.registry
        n in names && (block[n] = entry)
    end
    for n in sort!([n for n in names if !haskey(L.registry, n)])
        block[n] = L.defs[n].entry
    end
    return block
end

_operators_key(L::RefLayer) = L.refop == "apply_operator" ? "operators" : "expression_templates"

"""
    _operators_blocks!(doc, sv) -> doc

Give each top-level model of the derived document `doc` that holds an
operator reference (anywhere in its subtree) the `operators` block of the
definitions its references reach.
"""
function _operators_blocks!(doc, sv)
    L = sv.refs
    L === nothing && return doc
    for (_, m) in doc["models"]
        refs = _raw_refs!(Tuple{String, Any}[], m)
        isempty(refs) && continue
        m[_operators_key(L)] = _operators_block(L, _reached!(L, refs))
    end
    return doc
end
