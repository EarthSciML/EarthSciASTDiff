# Chain rule through algebraic intermediates: observed-variable inlining and
# `index`-of-array-expression β-reduction.
#
# Differentiation needs a closed expression in states/parameters. Two
# constructs stand between the flattened RHS and that form:
#
#   1. OBSERVED variables (algebraic intermediates). Scalar observed inline by
#      plain substitution; an array observed defined by a `faq` node inlines
#      at each read site by β-reduction (`index(F, e…)` → F's body with output
#      indices := e). A CONTRACTING definition (ranges binding names beyond
#      output_idx) keeps its sum: the read becomes a scalar `faq` over the
#      contracted names (see `_beta_faq`). Every substitution is
#      capture-avoiding, and a faq node binding a name an enclosing faq
#      already binds is renamed apart (`_inline_once`).
#   2. `Pre(a)` → `a`. esm-spec §5.1 gives `Pre` a meaning of its own only in
#      an event affect; in a right-hand side it is the value of `a` (spec
#      §3.2). Lowering it here, before differentiation, keeps it out of every
#      coefficient, so `Pre(x)` and `x` intern, simplify and share as one
#      structure and no emitted document asks its evaluator for `Pre`.
#   3. `index(makearray(…), i, j)` — the form the pointwise lift produces when
#      an operator's `makearray` is read per cell. Lowered to a nested
#      region-membership `ifelse`: later regions override earlier ones, which
#      is exactly the spec's §4.3.2 overwrite semantics, folded from the first
#      region outward.
#
# Iterated to a bounded fixpoint (observed may reference observed), mirroring
# the template engine's MAX_REWRITE_PASSES discipline.

const MAX_INLINE_PASSES = 64

struct InlineError <: Exception
    msg::String
end
Base.showerror(io::IO, e::InlineError) = print(io, "InlineError: ", e.msg)

"""
    inline_observed(e, obs::Dict{String,ASTExpr}) -> ASTExpr

Inline every observed-variable read in `e` (`obs` maps observed name →
defining expression), lower `Pre(a)` to `a` (spec §3.2), β-reduce
`index`-of-`faq`, and lower `index`-of-`makearray` to region `ifelse`. Fixpoint-iterated; throws
[`InlineError`](@ref) on non-convergence (cyclic observed definitions).
`obs_names` caches the names used by `obs`'s definitions; a caller inlining
many expressions against one `obs` passes one `Ref` to all of them.
`shapes` (variable → declared extents, the band context's) gives the rank of
an array variable used as a `makearray` region value; a variable it does not
name (nor an observed definition) is read at the region's free rank.
`rank_obs` (default `obs`) holds every observed definition, including those a
pruned `obs` leaves out, so that an elementwise observed read at a cell indexes
each array operand, inlined or not. `scalars` names the variables declared
without a shape that are not observed (scalar states and parameters); `t` is
always scalar. Any other name in such a definition has no known rank and is
refused. `axes` (variable → its declared index-set names) aligns the operands
of an elementwise observed read at a cell by name (esm-spec §4.3.4): an
operand over a subset of the observed's index sets is read at those of the
cell's indices and repeats along the others.
"""
function inline_observed(
        e::ASTExpr, obs::Dict{String, ASTExpr};
        obs_names = Ref{Union{Nothing, Set{String}}}(nothing),
        shapes::AbstractDict = Dict{String, Vector{Int}}(),
        rank_obs::AbstractDict = obs,
        scalars::AbstractSet = Set{String}(),
        axes::AbstractDict = Dict{String, Vector{String}}()
    )::ASTExpr
    isempty(obs) && !_needs_inline(e) && return e
    # FIXPOINT TEST, AT A COST THAT DOES NOT GROW WITH THE TREE. The test is
    # "did this pass change anything?", and `skey(e2) == skey(e)` answers it
    # by serializing the whole tree TWICE PER PASS — that tree is the
    # post-inlining RHS of one equation, the largest object in the calculus
    # (measured on ReSEACT: 4.2×10⁵ nodes per equation), and the
    # serialization is redone from scratch every pass. Two cheaper tests in
    # order, same relation: `_inline_once` routes untouched nodes through the
    # identity-preserving `map_children`, so at the fixpoint it returns the
    # SAME object — a pointer compare; and otherwise interning
    # ([`sid`](@ref)) re-keys only the subtrees the pass rebuilt, because the
    # index's memo is carried across passes.
    ix = StructIndex()
    fresh = _Fresh(
        e, obs, obs_names, nothing, shapes, Dict{String, Int}(),
        IdDict{OpExpr, Int}(), rank_obs, scalars, axes,
        IdDict{OpExpr, Tuple{Int, Int}}()
    )
    for _ in 1:MAX_INLINE_PASSES
        e2 = _inline_once(e, obs, fresh, Set{String}())
        e2 === e && return e2
        sid(ix, e2) == sid(ix, e) && return e2
        e = e2
    end
    throw(InlineError("no fixpoint after $MAX_INLINE_PASSES passes (cyclic observed?)"))
end

# Fresh names for renamed-apart binders: `_c<n>_<name>`, the smallest `n`
# clear of every name the document can put in this tree (the tree's own and
# every observed definition's, which later passes splice in) and of the
# caller's `avoid` set. Deterministic in (`name`, `avoid`): two copies of one
# sum renamed in the same scope get the same name, so they stay structurally
# equal and interning, template CSE and `hoist_observed` still share them.
# The taken set is collected on the first rename only (and `obs_names`, when
# given, is filled once and shared across the equations of one document), so
# a tree needing no rename pays nothing for it.
mutable struct _Fresh
    root::ASTExpr
    obs::Dict{String, ASTExpr}
    obs_names::Base.RefValue{Union{Nothing, Set{String}}}
    taken::Union{Set{String}, Nothing}
    shapes::AbstractDict
    ranks::Dict{String, Int}   # observed name → its rank at a cell (`_cell_rank`)
    op_ranks::IdDict{OpExpr, Int}  # node → its rank at a cell (`_cell_rank`)
    # every observed definition, the pruned-away ones too: an operand left an
    # opaque read still has a rank (`_cell_rank`)
    rank_obs::AbstractDict
    scalars::AbstractSet       # declared scalars that are not observed
    axes::AbstractDict         # variable → its declared index-set names (§4.3.4)
    # node → the two ranks some elementwise node in its subtree combines
    # (`_def_rank`): refused inside an observed declared without a shape
    mixed::IdDict{OpExpr, Tuple{Int, Int}}
end

function (f::_Fresh)(name::String, avoid)::String
    if f.taken === nothing
        if f.obs_names[] === nothing
            t = Set{String}()
            for d in values(f.obs)
                _all_names!(t, d)
            end
            f.obs_names[] = t
        end
        f.taken = _all_names!(Set{String}(), f.root)
    end
    base = replace(name, r"^_c\d+_" => "")
    n = 0
    while true
        n += 1
        c = "_c$(n)_$base"
        c in f.taken || c in f.obs_names[] || c in avoid || return c
    end
    return
end

# Every variable name and every faq-bound index name in `e`.
function _all_names!(acc::Set{String}, e::ASTExpr)::Set{String}
    if e isa VarExpr
        push!(acc, e.name)
    elseif e isa OpExpr
        e.op == "faq" && union!(acc, _binders(e))
        map_children(x -> (_all_names!(acc, x); x), e)
    end
    return acc
end

# The index names a faq node binds: its ranges (sorted, so renaming is
# deterministic across bindings) and its named output indices.
function _binders(e::OpExpr)::Vector{String}
    b = sort!(String[String(n) for n in keys(something(e.ranges, Dict{String, Any}()))])
    for x in something(e.output_idx, Any[])
        x isa AbstractString && !(String(x) in b) && push!(b, String(x))
    end
    return b
end

# Rename the binders `names` of faq node `e` to fresh ones (clear of `avoid`
# and of every name under `e`, so no inner binder can capture them).
function _rename_binders(e::OpExpr, names, fresh::_Fresh, avoid)::OpExpr
    e.join === nothing || throw(
        InlineError(
            "cannot rename the indices of a joined faq node (join keys name them)"
        )
    )
    av = _all_names!(Set{String}(avoid), e)
    ren = Dict{String, ASTExpr}()
    for n in names
        f = fresh(n, av)
        push!(av, f)
        ren[n] = VarExpr(f)
    end
    rn(n) = haskey(ren, String(n)) ? ren[String(n)].name : n
    fv = Set{String}(x.name for x in values(ren))
    return EarthSciAST.reconstruct(
        e;
        ranges = e.ranges === nothing ? nothing :
            Dict{String, Any}(rn(n) => r for (n, r) in e.ranges),
        output_idx = e.output_idx === nothing ? nothing :
            Any[x isa AbstractString ? rn(x) : x for x in e.output_idx],
        expr_body = _csubst(e.expr_body, ren, fv, fresh),
        filter = e.filter === nothing ? nothing : _csubst(e.filter, ren, fv, fresh)
    )
end

# CAPTURE-AVOIDING substitution (upstream `substitute` is not): a faq node
# shadows the names it binds, and one binding a name free in a replacement
# (`fv`) is renamed apart before the substitution enters it.
function _csubst(
        e::ASTExpr, m::Dict{String, ASTExpr}, fv::Set{String},
        fresh::_Fresh
    )::ASTExpr
    isempty(m) && return e
    e isa VarExpr && return get(m, e.name, e)
    e isa OpExpr || return e
    if e.op == "faq"
        b = _binders(e)
        if any(n -> haskey(m, n), b)
            m = Dict{String, ASTExpr}(k => v for (k, v) in m if !(k in b))
            isempty(m) && return e
        end
        clash = [n for n in b if n in fv]
        isempty(clash) || (e = _rename_binders(e, clash, fresh, fv))
    end
    return map_children(x -> _csubst(x, m, fv, fresh), e)
end

# β-reduce a read `F[sub…]` of a faq-defined array F. Output indices take the
# subscripts. Any other index F's ranges bind is CONTRACTED: the read is the
# scalar faq summing over it (F's own `reduce`, `filter` and `semiring`
# carried), never the bare body, which would leave that index free — or
# captured by a faq around the reading site that binds the same name, which
# turns the sum into one term with no error. The substitution is
# capture-avoiding: a contracted name, or a name bound inside the body, that
# the subscripts read freely is renamed apart first. A contracted name that
# shadows an index bound around the reading site (`scope`) is renamed too,
# since the band calculus reads a free index in a derivative by name. A
# filter on a non-contracting F is carried the same way (a sum over the one
# point it gates); the band calculus then refuses it, as it refuses a filter
# written inline.
function _beta_faq(def::OpExpr, sub::Vector{ASTExpr}, fresh::_Fresh, scope)::ASTExpr
    oidx = [String(x) for x in something(def.output_idx, Any[])]
    length(sub) == length(oidx) || throw(
        InlineError(
            "index arity $(length(sub)) ≠ rank $(length(oidx)) of a faq read"
        )
    )
    ranges = def.ranges === nothing ? Dict{String, Any}() : def.ranges
    cnames = sort!([String(n) for n in keys(ranges) if !(String(n) in oidx)])
    subst = Dict{String, ASTExpr}(n => sub[k] for (k, n) in enumerate(oidx))
    fv = Set{String}()
    for x in sub
        union!(fv, free_variables(x))
    end
    isempty(cnames) && def.filter === nothing &&
        return _csubst(def.expr_body, subst, fv, fresh)
    def.join === nothing || throw(
        InlineError(
            "cannot inline a read of a joined faq definition (join keys name its indices)"
        )
    )
    newranges = Dict{String, Any}()
    av = union(fv, scope, cnames)
    for n in cnames
        if n in fv || n in scope
            f = fresh(n, av)
            push!(av, f)
            subst[n] = VarExpr(f)
            newranges[f] = ranges[n]
        else
            newranges[n] = ranges[n]
        end
    end
    fv2 = copy(fv)
    for x in values(subst)
        x isa VarExpr && push!(fv2, x.name)
    end
    return EarthSciAST.reconstruct(
        def; output_idx = Any[], ranges = newranges,
        expr_body = _csubst(def.expr_body, subst, fv2, fresh),
        filter = def.filter === nothing ? nothing :
            _csubst(def.filter, subst, fv2, fresh)
    )
end

# Whether a pass could change `e` with no observed to inline: a `Pre` to
# lower, an `index`-of-array-expression to reduce, or a faq node shadowing an
# index bound around it (renamed apart by `_inline_once`).
function _needs_inline(e::ASTExpr, scope = Set{String}())::Bool
    e isa OpExpr || return false
    e.op == "Pre" && return true
    e.op == "index" && e.args[1] isa OpExpr && return true
    if e.op == "faq"
        b = _binders(e)
        any(n -> n in scope, b) && return true
        scope = union(scope, b)
    end
    found = false
    map_children(e) do c
        found || (found = _needs_inline(c, scope))
        c
    end
    return found
end

# index(makearray(regions, values), e…) → nested ifelse on region membership.
# Fold from the FIRST region outward so later regions override (§4.3.2).
# Where a faq range starts: an interval's lower bound, 1 for an index-set
# range.
_range_origin(r) = r isa AbstractVector ? _region_bound(r[1]) : 1

_ones_rank(n::Int) = (n, ones(Int, n))

# The rank of an array-valued `makearray` region value and, per dim it spans,
# the index at which the value starts (its ORIGIN): a faq node's own range
# start for each output index, 1 for any other array. An `index` value is its
# base's rank less its subscripts (0 for a cell read: a scalar broadcast over
# the region); a base whose rank is not read here is ranked as an operand of
# an elementwise definition (`_cell_rank`), and refused when it has none.
# `nothing` for a bare variable of no known shape: such a value is read at
# the region's free rank.
function _value_rank(val, fresh::_Fresh, scope)
    if val isa OpExpr && val.op == "faq"
        vr = val.ranges === nothing ? Dict{String, Any}() : val.ranges
        oidx = [String(x) for x in something(val.output_idx, Any[])]
        return (
            length(oidx),
            Int[haskey(vr, n) ? _range_origin(vr[n]) : 1 for n in oidx],
        )
    elseif val isa OpExpr && val.op == "makearray"
        (val.regions === nothing || isempty(val.regions)) && return nothing
        return _ones_rank(length(val.regions[1]))
    elseif val isa OpExpr && val.op == "index" && !isempty(val.args)
        # a read of a variable: its rank less the subscripts it is read at
        # (0 for a cell read, a scalar broadcast over the region)
        base = val.args[1]
        b = _value_rank(base, fresh, scope)
        if b === nothing
            base isa VarExpr && base.name in scope && throw(
                InlineError("a makearray region value reads `index` of the index `$(base.name)`")
            )
            b = _ones_rank(_index_base_rank(base, fresh))
        end
        nsub = length(val.args) - 1
        r = b[1] - nsub
        r < 0 && throw(
            InlineError(
                "a makearray region value reads an array of rank $(b[1]) with $nsub subscripts"
            )
        )
        # a partial read of an array whose ranges do not all start at 1: where
        # its remaining dims start is not read here, so it is refused
        r > 0 && any(!=(1), b[2]) && throw(
            InlineError(
                "a makearray region value is a partial read of an array whose ranges " *
                    "do not all start at 1; read it at a cell or as a whole"
            )
        )
        return _ones_rank(r)
    elseif val isa VarExpr && !(val.name in scope)
        haskey(fresh.shapes, val.name) && return _ones_rank(length(fresh.shapes[val.name]))
        d = get(fresh.obs, val.name, nothing)
        if d isa OpExpr && d.op == "faq"
            return _ones_rank(length(something(d.output_idx, Any[])))
        elseif d isa OpExpr && d.op == "makearray" && d.regions !== nothing &&
                !isempty(d.regions)
            return _ones_rank(length(d.regions[1]))
        end
    end
    return nothing
end

# The rank of the base of an `index` makearray value that `_value_rank` does
# not read: a variable ranked as an operand of an elementwise definition
# (`_cell_rank`: declared, an observed, `t` or a declared scalar), a literal
# table, or an operator reference. Anything else is refused, never guessed.
function _index_base_rank(base, fresh::_Fresh)::Int
    base isa VarExpr && return _cell_rank(base, fresh, Set{String}())
    base isa OpExpr && base.op == "const" && return _lit_rank(base.value)
    _is_ref(base) && return _def!(_layer(), base.name).rank
    throw(
        InlineError(
            "a makearray region value reads `index` of " *
                (base isa OpExpr ? "a `$(base.op)`" : "a literal") *
                ", whose rank is not read here"
        )
    )
end

function _index_makearray(ma::OpExpr, idx::Vector{ASTExpr}, obs, fresh, scope)::ASTExpr
    result = lit(0)
    for (region, val) in zip(ma.regions, ma.values)
        length(region) == length(idx) || throw(
            InlineError(
                "makearray region rank $(length(region)) ≠ index arity $(length(idx))"
            )
        )
        conds = ASTExpr[]; los = Int[]; nonsing = Int[]
        for (k, r) in enumerate(region)
            lo = r[1] isa ASTExpr ? r[1] : lit(r[1])
            hi = r[2] isa ASTExpr ? r[2] : lit(r[2])
            push!(conds, op(">=", idx[k], lo))
            push!(conds, op("<=", idx[k], hi))
            push!(los, _region_bound(r[1]))
            _region_bound(r[2]) > los[end] && push!(nonsing, k)
        end
        _is_ref(val) && _def!(_layer(), val.name).rank > 0 && throw(
            InlineError(
                "a makearray region value that is an array-valued operator reference " *
                    "(`$(val.name)`) read through an observed is not supported"
            )
        )
        v = if val isa VarExpr ||
                (val isa OpExpr && val.op in ("faq", "makearray", "index"))
            # TWO legal value shapes, the pair `_bands_makearray!` tells apart
            # with `val_fullrank` (bands.jl): a FULL-RANK value spans every
            # region dim, singleton dims included (a 3-D field's boundary
            # face); a REDUCED-RANK one spans only the non-singleton dims. The
            # value fills the region by position: its element at index
            # `origin + p` lands on region index `lo + p`, so a dim it spans
            # is read at `idx − lo + origin`, the shift `_bands_makearray!`
            # applies to the value's bands.
            vr = _value_rank(val, fresh, scope)
            rank, origins = vr === nothing ? _ones_rank(length(nonsing)) : vr
            scalar = rank == 0 && !(val isa OpExpr && val.op == "faq")
            dims = scalar ? Int[] :
                rank == length(idx) ? collect(1:length(idx)) :
                rank == length(nonsing) ? nonsing :
                throw(
                    InlineError(
                        "makearray value rank $rank matches neither the " *
                        "region rank $(length(idx)) nor the region free rank " *
                        "$(length(nonsing))"
                    )
                )
            tgt = ASTExpr[add(idx[k], lit(o - los[k])) for (k, o) in zip(dims, origins)]
            if scalar
                # a scalar value (a cell read): broadcast over the region,
                # never wrapped in an `index` with no subscripts
                val
            elseif val isa OpExpr && val.op == "faq"
                _beta_faq(val, tgt, fresh, scope)
            else
                _inline_once(op("index", val, tgt...), obs, fresh, scope)
            end
        else
            val   # scalar broadcast over the region
        end
        cond = length(conds) == 1 ? conds[1] : op("and", conds...)
        result = op("ifelse", cond, v, result)
    end
    return result
end

# `scope`: the index names bound by the faq nodes enclosing `e`.
function _inline_once(e::ASTExpr, obs, fresh::_Fresh, scope::Set{String})::ASTExpr
    if e isa VarExpr
        # A name an enclosing faq binds is that index, not an observed read.
        (e.name in scope || !haskey(obs, e.name)) && return e
        return obs[e.name]
    elseif e isa OpExpr
        # `Pre(a)` in a right-hand side is `a` (spec §3.2)
        e.op == "Pre" && return _inline_once(e.args[1], obs, fresh, scope)
        args = () -> ASTExpr[_inline_once(x, obs, fresh, scope) for x in e.args[2:end]]
        if e.op == "index" && e.args[1] isa OpExpr && e.args[1].op == "makearray"
            return _index_makearray(e.args[1], args(), obs, fresh, scope)
        elseif e.op == "index" && e.args[1] isa OpExpr && e.args[1].op == "faq"
            return _inline_once(_beta_faq(e.args[1], args(), fresh, scope), obs, fresh, scope)
        elseif e.op == "faq"
            # A faq node binding a name already bound around it — a spliced
            # definition's own sum (`t = 2·Σ_k u[k]` read under a faq over k)
            # or a hand-written one — is renamed apart: the band calculus
            # reads a free index in a derivative by name, so a shadowed one
            # would conflate the two indices (spec §3.5).
            b = _binders(e)
            clash = [n for n in b if n in scope]
            if !isempty(clash)
                e = _rename_binders(e, clash, fresh, scope)
                b = _binders(e)
            end
            inner = union(scope, b)
            return map_children(x -> _inline_once(x, obs, fresh, inner), e)
        end
        if e.op == "index" && e.args[1] isa VarExpr && haskey(obs, e.args[1].name) &&
                !(e.args[1].name in scope)
            def = obs[e.args[1].name]
            if def isa OpExpr && def.op == "faq"
                oidx = something(def.output_idx, Any[])
                length(oidx) == length(e.args) - 1 || throw(
                    InlineError(
                        "index arity $(length(e.args) - 1) ≠ rank $(length(oidx)) of observed `$(e.args[1].name)`"
                    )
                )
                return _beta_faq(def, args(), fresh, scope)
            elseif def isa OpExpr && def.op == "makearray"
                # A named flux-style field: same lowering as an anonymous
                # `index(makearray…)` read (region-membership ifelse, later
                # clipped exactly by clip.jl where the guards are decidable).
                return _index_makearray(def, args(), obs, fresh, scope)
            elseif _is_ref(def)
                # An observed defined by an operator reference: its read is
                # a read of the reference's array (operator_refs.jl reads it
                # as an intermediate; the body is never substituted).
                return op("index", def, args()...)
            elseif def isa OpExpr && def.op == "const"
                # A literal connectivity/weight table (`conn[i,k]`): keep the
                # gather, inline the table itself so downstream consumers
                # (site column evaluation, the derived evaluation model) see
                # a closed `index(const, …)` form.
                return op("index", def, args()...)
            elseif def isa VarExpr || def isa OpExpr
                # An observed declared without a shape whose definition
                # evaluates to an array (esm-spec §4.3.4, "An undeclared
                # result takes the union"; the pointwise lift of a 0-D
                # component coupled to a gridded host), or a shaped observed
                # defined by an elementwise expression of arrays: its element
                # is the definition with each array operand read at the same
                # cell (`_gather_cell`).
                return _gather_cell(def, args(), fresh, scope, e.args[1].name)
            else
                # a number: the same value at every cell
                return def
            end
        end
        return map_children(x -> _inline_once(x, obs, fresh, scope), e)
    end
    return e
end


# ── elementwise observeds read at a cell ────────────────────────────────────
#
# `index(x, i…)` of an observed `x` whose definition is an elementwise
# expression (arithmetic, elementary, comparison, logical, `ifelse`) of
# arrays and scalars: the definition with every array operand read at
# `i…` and every scalar operand unchanged (esm-spec §4.3.4, §9.6.11). The
# read's arity must be the rank of the observed's declared shape. An
# array operand is aligned with the read observed by index-set name when
# both declare theirs (`axes`): an operand over a subset of the observed's
# index sets is read at those of the cell's indices and repeats along the
# others, as §4.3.4 aligns an array-level expression, and one over an index
# set the observed does not have is refused. Without names, an operand must
# have the rank of the read and is aligned by position, as every operand of
# one rank in a flattened system spans the same index sets in the same
# order. An array
# operand that is a variable, a `faq`, a `makearray` or an operator
# reference is read whole at the cell (`index(operand, i…)`, reduced by the
# next pass); an `index` inside the definition must select a scalar. Anything
# this cannot decide is refused, never guessed: a name whose rank is not
# known (neither declared, an observed, nor `t`), an operand whose rank
# differs from the read's where names cannot align it, an observed declared
# without a shape that mixes ranks anywhere, a partial `index`, a literal array or shape op
# (`reshape`, `transpose`, `concat`, `broadcast`) as an operand, an
# elementwise application of a definition (an array bound to a value-role
# parameter), or a definition reading a name an index bound around the read
# site would capture.
#
# Ranks are computed on the closed definition (no enclosing scope), so they
# are memoized per observed name and per node: one bottom-up walk per
# definition, however many cells read it.

# The rank of `e` in an elementwise position: 0 for a scalar.
function _cell_rank(e, fresh::_Fresh, seen::Set{String})::Int
    if e isa VarExpr
        n = e.name
        (n == "t" || n in fresh.scalars) && return 0
        haskey(fresh.shapes, n) && return length(fresh.shapes[n])
        haskey(fresh.ranks, n) && return fresh.ranks[n]
        d = get(fresh.rank_obs, n, nothing)
        d === nothing && throw(
            InlineError(
                "`$n`, read in an elementwise observed definition read at a cell, " *
                    "has no known rank (not a declared variable, nor an observed)"
            )
        )
        n in seen && throw(InlineError("cyclic observed `$n`"))
        push!(seen, n)
        r = _def_rank(d, fresh, seen, n)
        delete!(seen, n)
        fresh.ranks[n] = r
        return r
    elseif e isa OpExpr
        r = get(fresh.op_ranks, e, -1)
        r >= 0 && return r
        r = _def_rank(e, fresh, seen, nothing)
        fresh.op_ranks[e] = r
        return r
    end
    return 0
end

# The rank of a literal array value (its nesting depth).
function _lit_rank(v)::Int
    r = 0
    while v isa AbstractVector
        r += 1
        isempty(v) && break
        v = v[1]
    end
    return r
end

function _def_rank(d, fresh::_Fresh, seen::Set{String}, name)::Int
    d isa VarExpr && return _cell_rank(d, fresh, seen)
    d isa OpExpr || return 0
    what = something(name, "expression")
    if d.op == "faq"
        return length(something(d.output_idx, Any[]))
    elseif d.op == "makearray"
        (d.regions === nothing || isempty(d.regions)) &&
            throw(InlineError("makearray with no region read at a cell"))
        return length(d.regions[1])
    elseif d.op == "index"
        # an `index` in an elementwise definition selects one element: its
        # arity is its operand's rank (a partial index is still an array, and
        # aligning what it leaves by position is not decidable here)
        a = d.args[1]
        r0 = a isa OpExpr && a.op == "const" ? _lit_rank(a.value) :
            _cell_rank(a, fresh, seen)
        r0 == length(d.args) - 1 || throw(
            InlineError(
                "observed $what reads `index` with $(length(d.args) - 1) indices into " *
                    "an operand of rank $r0 in an elementwise definition read at a cell"
            )
        )
        return 0
    elseif _is_ref(d)
        D = _def!(_layer(), d.name)
        D.rank > 0 && return D.rank
        for (q, b) in something(d.bindings, Dict{String, ASTExpr}())
            if q in D.value && _cell_rank(b, fresh, seen) > 0
                throw(
                    InlineError(
                        "observed $what applies `$(d.name)` " *
                            "elementwise (an array bound to its value parameter `$q`); " *
                            "reading such an application at a cell is not supported"
                    )
                )
            end
        end
        return 0
    elseif d.op == "const" && !(d.value isa AbstractVector)
        return 0   # a scalar literal
    elseif d.op in ("const", "reshape", "transpose", "concat", "broadcast")
        throw(
            InlineError(
                "observed $what reads `$(d.op)` in an " *
                    "elementwise definition read at a cell; not supported"
            )
        )
    end
    # Operands of different ranks are aligned by index-set name where the
    # definition is read (`_gather_cell`); a node's rank is then its widest
    # operand's, and the node is recorded as mixing ranks (`fresh.mixed`, as
    # is every node above one). A named observed reaches here only when it is
    # declared without a shape, and its shape is the union of its operands'
    # index sets (esm-spec §4.3.4), which ranks alone cannot give: mixed
    # ranks anywhere in its definition are refused.
    r = 0
    mix = nothing
    for a in d.args
        ra = _cell_rank(a, fresh, seen)
        mix === nothing && a isa OpExpr && (mix = get(fresh.mixed, a, nothing))
        ra == 0 && continue
        mix === nothing && r != 0 && r != ra && (mix = (r, ra))
        r = max(r, ra)
    end
    if mix !== nothing
        name === nothing || throw(
            InlineError(
                "observed $what, declared without a shape, combines arrays of rank " *
                    "$(mix[1]) and $(mix[2]); declare its shape so its operands can be " *
                    "aligned by index set"
            )
        )
        fresh.mixed[d] = mix
    end
    return r
end

# The cell indices at which an operand over the index sets `oax` is read
# inside a read of observed `name` over `tax` at `idx` (esm-spec §4.3.4): the
# k-th dimension of the operand on index set X takes the index of the k-th
# dimension of the observed on X. An index set the observed does not have
# (or has fewer times) is refused.
function _aligned_idx(oax, tax, idx, name::String, operand::String)
    out = ASTExpr[]
    for (d, X) in enumerate(oax)
        k = count(==(X), oax[1:d])
        hits = findall(==(X), tax)
        k <= length(hits) || throw(
            InlineError(
                "observed `$name` over ($(join(tax, ", "))) reads `$operand` over " *
                    "($(join(oax, ", "))), whose index set `$X` it is not shaped over"
            )
        )
        push!(out, idx[hits[k]])
    end
    return out
end

# `def` read at the cell `idx`, spliced under the indices `scope` binds.
function _gather_cell(def, idx::Vector{ASTExpr}, fresh::_Fresh, scope, name::String)::ASTExpr
    # the definition is closed; a name it reads that an index around the
    # read site binds would be captured by it, and a global cannot be renamed
    cap = sort!([n for n in free_variables(def) if n in scope])
    isempty(cap) || throw(
        InlineError(
            "observed `$name` reads $(join(("`$n`" for n in cap), ", ")), which an " *
                "index bound around its read at a cell would capture"
        )
    )
    tax = get(fresh.axes, name, nothing)
    # the read selects one element of the observed: one index per dimension
    sh = get(fresh.shapes, name, nothing)
    sh === nothing || length(sh) == length(idx) || throw(
        InlineError(
            "observed `$name` of rank $(length(sh)) is read at a cell with " *
                "$(length(idx)) indices"
        )
    )
    function g(e)
        e isa OpExpr || e isa VarExpr || return e
        r = _cell_rank(e, fresh, Set{String}())
        r == 0 && return e
        leaf = e isa VarExpr || _is_ref(e) || e.op == "faq" || e.op == "makearray"
        leaf || return map_children(g, e)
        oax = e isa VarExpr ? get(fresh.axes, e.name, nothing) : nothing
        tax === nothing || oax === nothing ||
            return op("index", e, _aligned_idx(oax, tax, idx, name, e.name)...)
        r == length(idx) || throw(
            InlineError(
                "observed `$name`, read with $(length(idx)) indices, has an operand of rank $r" * (
                    r >= length(idx) ? "" : sh === nothing ?
                        "; the observed is declared without a shape, so the operand cannot be aligned by index set" :
                        tax === nothing ?
                        "; the observed is shaped over a dimension that is not a declared index set, so the operand cannot be aligned by index set" :
                        " whose index sets are not known, so it cannot be aligned by index set"
                )
            )
        )
        return op("index", e, idx...)
    end
    return g(def)
end
