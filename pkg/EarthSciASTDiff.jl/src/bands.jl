# The array-level band calculus: differentiate an array-valued RHS w.r.t. one
# variable, producing a list of structured bands.

"""
    Band

One structured block of `∂(rhs of the equation for u)/∂v`:

  - `rows::Vector{Tuple{Int,Int}}` — a rectangular region of `u`'s index
    space, per-dimension inclusive `(lo, hi)`. Empty for a scalar `u`.
  - `ridx::Vector{Union{String,Int}}` — per-dimension row index: a free index
    NAME ranging over `rows[k]`, or a constant Int for a singleton dimension.
  - `cidx::Vector{ASTExpr}` — `v`'s column index expressions, one per
    dimension of `v`, in terms of the row index names. Empty for scalar `v`.
  - `coef::ASTExpr` — scalar ESM coefficient expression in terms of the row
    index names (plus parameters and other states).
  - `contracted::Vector{Tuple{String,Int,Int}}` — free CONTRACTED column
    dimensions `(name, lo, hi)`: when non-empty the band denotes, for every
    row cell AND every value of the contracted names over their inclusive
    ranges, an entry `J[row, cidx(row, contracted…)] += coef(row,
    contracted…)` — entries meeting at one column ACCUMULATE (this is how a
    `reduce: "+"` faq node's sum over a gathered column materializes).
    Empty for ordinary bands.
  - `guard::Union{Nothing,ASTExpr}` — an exact point predicate (the block's
    `where` field) over the row and contracted index names, literals and
    `const` gathers: a point contributes iff it holds. What a filtered faq
    node's non-rectangular window (`gk < gke`) leaves after its box is split
    at affine breakpoints; `nothing` for every rectangular band.

The band is the symbolic, portable analogue of the tree-walk's `_AccDesc`
state-read descriptors: `cidx` of the form `row ± const` is an affine stencil
offset, `cidx == ridx` is a diagonal (per-cell) coupling, and a `cidx`
referencing a contracted name is an indirect/contracted gather
(source–receptor matrices, regrid joins, `conn[]`-table stencils).
"""
struct Band
    rows::Vector{Tuple{Int, Int}}
    ridx::Vector{Union{String, Int}}
    cidx::Vector{ASTExpr}
    coef::ASTExpr
    contracted::Vector{Tuple{String, Int, Int}}
    guard::Union{Nothing, ASTExpr}
end
Band(rows, ridx, cidx, coef) = Band(rows, ridx, cidx, coef, Tuple{String, Int, Int}[])
Band(rows, ridx, cidx, coef, contracted) = Band(rows, ridx, cidx, coef, contracted, nothing)

# A copy of `b` with some fields replaced; the guard travels unless replaced,
# so no rebuild can drop it silently.
_rebuild(
    b::Band; rows = b.rows, ridx = b.ridx, cidx = b.cidx, coef = b.coef,
    contracted = b.contracted, guard = b.guard
) = Band(rows, ridx, cidx, coef, contracted, guard)

"Differentiation context: index-set sizes, array-variable shapes, parameter names."
struct Ctx
    index_sets::Dict{String, Int}
    shapes::Dict{String, Vector{Int}}
    params::Set{String}
end
is_array(ctx::Ctx, name) = haskey(ctx.shapes, name)

"""
    BandError(msg)

An expression has a form the band calculus cannot differentiate (for example a
filtered `faq` node or a non-`"+"` semiring).
"""
struct BandError <: Exception
    msg::String
end
Base.showerror(io::IO, e::BandError) = print(io, "BandError: ", e.msg)

function _range_of(r, ctx::Ctx)::Tuple{Int, Int}
    if r isa IndexSetRef
        return (1, _set_size(r.from, ctx))
    elseif r isa AbstractVector
        return (_region_bound(r[1]), _region_bound(r[end]))
    elseif r isa AbstractDict
        return (1, _set_size(String(r["from"]), ctx))
    end
    throw(BandError("unsupported range spec $r"))
end

_set_size(name, ctx::Ctx)::Int = haskey(ctx.index_sets, name) ? ctx.index_sets[name] :
    throw(BandError("range over undeclared index set `$name`"))

# All distinct sites of `v` inside a scalar body: the bare VarExpr and every
# `index(v, …)` node, keyed by canonical JSON.
function sites_of(e::ASTExpr, v::String, acc = Dict{String, Site}())
    if e isa VarExpr && e.name == v
        s = Site(e); acc[s.key] = s
    elseif e isa OpExpr && e.op == "index" && e.args[1] isa VarExpr && e.args[1].name == v
        s = Site(e); acc[s.key] = s
    elseif e isa OpExpr
        map_children(x -> (sites_of(x, v, acc); x), e)
    end
    return acc
end

# For each site of `v` under a scalar `faq` sum nested in a body (where a
# contracting observed's sum survives inlining), the indices the sums
# ENCLOSING that site bind: name → range spec. Position-aware, so a sum
# elsewhere in the body binding the same name does not affect a site outside
# it. `:ambiguous` marks a name whose free occurrence in the site's
# derivative could mean two indices — also bound by the band's own row or
# contraction indices (`outer`), bound twice on the way down, or bound over
# different ranges at different occurrences of the site — which
# `_lift_nested` refuses if the column reads it. The inliner renames apart
# every such shadowing it produces, so this fires only on a hand-written
# shadowing sum. Each site also keeps the filters of the sums around it
# (`filters`): the same list at every occurrence, or `:mixed`.
struct _Nested
    ranges::Dict{String, Dict{String, Any}}
    filters::Dict{String, Any}            # site key → Vector{ASTExpr} | :mixed
end

function _nested_sums(e::ASTExpr, v::String, outer)
    acc = _Nested(Dict{String, Dict{String, Any}}(), Dict{String, Any}())
    _nested_sums!(acc, e, v, Set{String}(outer), Pair{String, Any}[], ASTExpr[])
    return acc
end

function _nested_sums!(acc::_Nested, e::ASTExpr, v::String, outer::Set{String}, stack, fstack)
    if (e isa VarExpr && e.name == v) || (
            e isa OpExpr && e.op == "index" &&
                e.args[1] isa VarExpr && e.args[1].name == v
        )
        key = Site(e).key
        f = get(acc.filters, key, nothing)
        if f === nothing
            acc.filters[key] = copy(fstack)
        elseif f !== :mixed && (
                length(f) != length(fstack) || any(skey(a) != skey(b) for (a, b) in zip(f, fstack))
            )
            acc.filters[key] = :mixed
        end
        isempty(stack) && return
        d = get!(() -> Dict{String, Any}(), acc.ranges, key)
        seen = Set{String}()
        for (n, r) in stack
            d[n] = n in outer || n in seen || (haskey(d, n) && d[n] != r) ? :ambiguous : r
            push!(seen, n)
        end
        return
    end
    e isa OpExpr || return
    pushed = 0
    filtered = false
    if e.op == "faq" && isempty(something(e.output_idx, Any[])) && e.ranges !== nothing
        for (n, r) in e.ranges
            push!(stack, String(n) => r)
            pushed += 1
        end
        e.filter === nothing || (push!(fstack, e.filter); filtered = true)
    end
    map_children(x -> (_nested_sums!(acc, x, v, outer, stack, fstack); x), e)
    for _ in 1:pushed
        pop!(stack)
    end
    filtered && pop!(fstack)
    return
end

# Substitution that does not descend into a faq node for the names it binds.
function _subst_free(e::ASTExpr, m::Dict{String, ASTExpr})::ASTExpr
    e isa VarExpr && return get(m, e.name, e)
    e isa OpExpr || return e
    if e.op == "faq"
        bound = _binders(e)
        if any(n -> haskey(m, n), bound)
            m = Dict{String, ASTExpr}(k => v for (k, v) in m if !(k in bound))
            isempty(m) && return e
        end
    end
    return map_children(x -> _subst_free(x, m), e)
end

# A site under a nested sum whose column reads the sum's index (`u[i,k]` in
# Σ_k u[i,k]): `dscalar` leaves k free in the derivative (`_dsum`), and here
# it becomes a CONTRACTED column dimension of the band — entries for every k
# accumulating on their columns, exactly as for a contracted index of the
# enclosing faq. It is renamed apart in `cidx` and in the FREE occurrences of
# the coefficient, because the coefficient may still carry the sum itself
# (the chain rule through f(Σ_k …)) binding the same name, and assembly
# substitutes contracted values without regard to binders. The sums'
# index-only filter conjuncts over a lifted name become the band's guard
# (`_nested_guard`). `nested` is `_nested_sums` of `body`, and `taken` a
# lazily filled cache of its names; both are per body, not per site.
function _lift_nested(
        nested::_Nested, s::Site, body::ASTExpr, cidx::Vector{ASTExpr},
        coef::ASTExpr, ctx::Ctx, outer, taken
    )
    none = Tuple{String, Int, Int}[]
    d = get(nested.ranges, s.key, nothing)
    d === nothing && return cidx, coef, none, nothing
    read = Set{String}()
    for c in cidx
        union!(read, free_variables(c))
    end
    names = sort!([n for n in keys(d) if n in read])
    isempty(names) && return cidx, coef, none, nothing
    taken[] === nothing && (taken[] = _all_names!(Set{String}(String.(outer)), body))
    ren = Dict{String, ASTExpr}(); extra = Tuple{String, Int, Int}[]
    for n in names
        d[n] === :ambiguous && throw(
            BandError(
                "a column reads `$n`, which the sums around it bind ambiguously " *
                    "(shadowing an enclosing index, or over two different ranges)"
            )
        )
        f = n * "_j"
        while f in taken[] || any(x -> x.name == f, values(ren))
            f *= "_j"
        end
        ren[n] = VarExpr(f)
        push!(extra, (f, _range_of(d[n], ctx)...))
    end
    guard = _nested_guard(get(nested.filters, s.key, :mixed), d, names, outer, ren)
    return ASTExpr[substitute(c, ren) for c in cidx], _subst_free(coef, ren), extra, guard
end

# The index-only conjuncts of the filters around a site that read a lifted
# name — and no name a sum around it still binds — renamed like the column:
# the band's guard. The coefficient keeps them too (as the remaining sum's
# filter, or as an `ifelse` once every bound name is lifted), so the guard
# only makes the pattern exact; a site under different filters at different
# occurrences gets none.
function _nested_guard(filters, d, lifted, outer, ren)
    filters === :mixed && return nothing
    ok = union(Set{String}(String.(outer)), lifted)
    gs = ASTExpr[]
    for f in filters, c in _conjuncts(f)
        fv = free_variables(c)
        any(in(fv), lifted) || continue
        issubset(fv, ok) || continue
        any(n -> haskey(d, n) && !(n in lifted), fv) && continue
        push!(gs, substitute(c, ren))
    end
    return _conj(gs)
end

"Sites of `v` in `e`, in deterministic (sorted canonical-key) order."
function sorted_sites(e::ASTExpr, v::String)
    d = sites_of(e, v)
    return [d[k] for k in sort!(collect(keys(d)))]
end

"Does an expression have array shape (vs scalar) in this context?"
function is_array_expr(e::ASTExpr, ctx::Ctx)::Bool
    e isa VarExpr && return is_array(ctx, e.name)
    e isa OpExpr || return false
    # `arrayop` is deliberately absent: it was REMOVED at esm 0.8.0 and is not
    # a deprecated alias, so EarthSciAST's parser rejects it by name
    # (`[E_REMOVED_OP]`) and no tree reaching here can carry it. `faq` is the
    # unified query node, spelled `aggregate` before esm 1.1.0 and normalized
    # to `faq` at the wire boundary (docs/content/rfcs/faq-node-rename.md).
    e.op in (
        "faq", "makearray", "broadcast",
        "reshape", "transpose", "concat",
    ) && return true
    e.op == "index" && return false
    return any(x -> is_array_expr(x, ctx), e.args)
end

"""
    bands(rhs, v, ctx; shape_u = Int[]) -> Vector{Band}

Differentiate the RHS of an equation for a variable of shape `shape_u`
(`Int[]` for scalar) with respect to variable `v`. `rhs` must already be
inlined ([`inline_observed`](@ref)).
"""
function bands(rhs::ASTExpr, v::String, ctx::Ctx; shape_u::Vector{Int} = Int[])::Vector{Band}
    out = Band[]
    if isempty(shape_u)
        nested = _nested_sums(rhs, v, String[])
        taken = Ref{Union{Nothing, Set{String}}}(nothing)
        for s in sorted_sites(rhs, v)
            cidx = s.expr isa VarExpr ? ASTExpr[] : ASTExpr[s.expr.args[2:end]...]
            cidx, d, extra, g = _lift_nested(
                nested, s, rhs, cidx, dscalar(rhs, s), ctx,
                String[], taken
            )
            coef = _simp(d)
            iszero_lit(coef) && continue
            _push_band!(out, Tuple{Int, Int}[], Union{String, Int}[], cidx, coef, extra, g, v, ctx)
        end
        return out
    end
    _bands_array!(out, rhs, v, ctx, shape_u, ASTExpr[lit(1)])
    return out
end

# Push one band, its value-reading column dimensions made one-hot first.
function _push_band!(out, rows, ridx, cidx, coef, contracted, guard, v::String, ctx::Ctx)
    ridx = collect(Union{String, Int}, ridx)
    cidx, coef, contracted = _one_hot(cidx, coef, contracted, ridx, v, ctx)
    push!(out, Band(collect(Tuple{Int, Int}, rows), ridx, cidx, coef, contracted, guard))
    return
end

# Emit one band per site of a scalar-valued expression broadcast over `rows`,
# statically clipping decidable region guards first (clip.jl) so membership
# `ifelse` never reaches the coefficients and the pattern is per-subrange
# exact.
function _scalar_bands!(out, e::ASTExpr, v::String, ctx::Ctx, rows, ridx, scale)
    outer = String[r for r in ridx if r isa String]
    for (rows2, body) in clip_regions(e, ridx, rows)
        nested = _nested_sums(body, v, outer)
        taken = Ref{Union{Nothing, Set{String}}}(nothing)
        for s in sorted_sites(body, v)
            cidx = s.expr isa VarExpr ? ASTExpr[] : ASTExpr[s.expr.args[2:end]...]
            cidx, d, extra, g = _lift_nested(
                nested, s, body, cidx, dscalar(body, s), ctx,
                outer, taken
            )
            coef = _simp(mul(d, scale...))
            iszero_lit(coef) && continue
            _push_band!(out, rows2, ridx, cidx, coef, extra, g, v, ctx)
        end
    end
    return
end

# `scale` accumulates elementwise scalar multipliers from enclosing `*`, `-`,
# `/`, and `neg` nodes on the way down to array leaves.
function _bands_array!(out, e::ASTExpr, v::String, ctx::Ctx, shape_u, scale::Vector{ASTExpr})
    if e isa VarExpr
        e.name == v || return out
        names = ["_r$k" for k in 1:length(shape_u)]           # identity band
        push!(
            out, Band(
                [(1, n) for n in shape_u], Union{String, Int}[names...],
                ASTExpr[VarExpr(n) for n in names], mul(scale...)
            )
        )
        return out
    end
    e isa OpExpr || return out
    o = e.op
    if o == "+"
        for x in e.args
            _bands_array!(out, x, v, ctx, shape_u, scale)
        end
    elseif o == "-" && length(e.args) == 1
        _bands_array!(out, e.args[1], v, ctx, shape_u, vcat(scale, [lit(-1)]))
    elseif o == "-"
        _bands_array!(out, e.args[1], v, ctx, shape_u, scale)
        _bands_array!(out, e.args[2], v, ctx, shape_u, vcat(scale, [lit(-1)]))
    elseif o == "neg"
        _bands_array!(out, e.args[1], v, ctx, shape_u, vcat(scale, [lit(-1)]))
    elseif o == "*"
        arr = [x for x in e.args if is_array_expr(x, ctx)]
        sc = [x for x in e.args if !is_array_expr(x, ctx)]
        length(arr) == 1 || throw(
            BandError(
                "elementwise product of $(length(arr)) array-valued factors at array " *
                    "level; wrap in a faq node"
            )
        )
        any(x -> occurs_var(x, v), sc) && throw(
            BandError(
                "scalar factor depends on `$v` in an array-level product"
            )
        )
        _bands_array!(out, arr[1], v, ctx, shape_u, vcat(scale, sc))
    elseif o == "/"
        is_array_expr(e.args[2], ctx) &&
            throw(BandError("array-valued divisor at array level"))
        _bands_array!(
            out, e.args[1], v, ctx, shape_u,
            vcat(scale, [op("/", lit(1), e.args[2])])
        )
    elseif o == "makearray"
        _bands_makearray!(out, e, v, ctx, scale)
    elseif o == "faq"
        _bands_faq!(out, e, v, ctx, shape_u, scale)
    elseif o == "index"
        # A scalar read of some array in an array-level position: a broadcast.
        names = ["_r$k" for k in 1:length(shape_u)]
        _scalar_bands!(out, e, v, ctx, [(1, n) for n in shape_u], names, scale)
    else
        if is_array_expr(e, ctx)
            # Generic elementwise op over whole arrays (exp(c), c^2, ifelse):
            # push the index inward, then recurse as a faq node.
            names = ["_r$k" for k in 1:length(shape_u)]
            idx = ASTExpr[VarExpr(n) for n in names]
            body = _push_index(e, idx, ctx)
            agg = OpExpr(
                "faq", ASTExpr[];
                output_idx = Any[names...], expr_body = body,
                ranges = Dict{String, Any}(
                    n => Any[1, shape_u[k]] for (k, n) in enumerate(names)
                )
            )
            _bands_array!(out, agg, v, ctx, shape_u, scale)
        else
            # Scalar expression broadcast over u's shape.
            names = ["_r$k" for k in 1:length(shape_u)]
            _scalar_bands!(out, e, v, ctx, [(1, n) for n in shape_u], names, scale)
        end
    end
    return out
end

# A faq node at array level. Each site of its body differentiates as a term
# of the node's reduction over its window (reductions.jl): the derivative is
# weighted per term (`_window_weight`), and the filter's index-only conjuncts
# split the band box (`_split_guard`) while its value-dependent conjuncts
# gate the coefficient. Per box piece, a site whose column reads no
# contracted name yields one band whose coefficient sums the weighted
# derivative over the piece's contraction (filtered by the residual guard's
# contracted conjuncts; its row-only conjuncts guard the band); a gathered
# column yields a band with contracted dimensions and the whole residual
# guard.
function _bands_faq!(out, e::OpExpr, v::String, ctx::Ctx, shape_u, scale)
    oidx = [String(x) for x in e.output_idx if x isa AbstractString]
    length(oidx) == length(e.output_idx) || throw(
        BandError(
            "singleton `1` output_idx entries not supported yet"
        )
    )
    red = _reduce_op(e, BandError)
    ranges = e.ranges === nothing ? Dict{String, Any}() : e.ranges
    cnames = sort!([String(n) for n in keys(ranges) if !(String(n) in oidx)])
    crange0 = Tuple{String, Int, Int}[
        (n, _range_of(ranges[n], ctx)...) for n in cnames
    ]
    rows0 = Tuple{Int, Int}[]
    for (k, n) in enumerate(oidx)
        push!(rows0, haskey(ranges, n) ? _range_of(ranges[n], ctx) : (1, shape_u[k]))
    end
    outer = vcat(oidx, cnames)
    gidx = ASTExpr[]; gval = ASTExpr[]
    e.filter === nothing || for c in _conjuncts(e.filter)
        push!(_index_only(c, outer) ? gidx : gval, c)
    end
    pieces = if isempty(gidx)
        [(rows0, crange0, nothing)]
    else
        no = length(oidx)
        box = vcat(rows0, Tuple{Int, Int}[(lo, hi) for (_, lo, hi) in crange0])
        [
            (
                cell[1:no],
                Tuple{String, Int, Int}[(n, cell[no + k]...) for (k, n) in enumerate(cnames)],
                g,
            ) for (cell, g) in _split_guard(_conj(gidx), outer, box)
        ]
    end
    vfilt = _conj(gval)
    body = e.expr_body
    for (rows_p, crange, guard) in pieces, (rows2, body2) in clip_regions(body, oidx, rows_p)
        nested = _nested_sums(body2, v, outer)
        taken = Ref{Union{Nothing, Set{String}}}(nothing)
        for s in sorted_sites(body2, v)
            s.expr isa VarExpr && is_array(ctx, v) && throw(
                BandError(
                    "whole-array reference to `$v` inside a faq body"
                )
            )
            cidx = s.expr isa VarExpr ? ASTExpr[] : ASTExpr[s.expr.args[2:end]...]
            cidx, d, extra, g = _lift_nested(
                nested, s, body2, cidx, dscalar(body2, s),
                ctx, outer, taken
            )
            d = _window_weight(red, e, cnames, d, BandError)
            vfilt === nothing || iszero_lit(d) || (d = op("ifelse", vfilt, d, lit(0)))
            if isempty(crange)
                coef = _simp(mul(d, scale...))
                iszero_lit(coef) && continue
                _push_band!(out, rows2, oidx, cidx, coef, extra, _conj(guard, g), v, ctx)
            elseif !any(
                    n -> any(c -> n in free_variables(c), cidx),
                    first.(crange)
                )
                # The column is independent of every contracted index: a
                # plain band whose coefficient is the (scalar) `reduce: "+"`
                # of the site derivative over the contraction.
                gc = ASTExpr[]; gr = ASTExpr[]
                guard === nothing || for c in _conjuncts(guard)
                    push!(any(n -> n in cnames, free_variables(c)) ? gc : gr, c)
                end
                red_sum = OpExpr(
                    "faq", ASTExpr[];
                    output_idx = Any[], reduce = "+", expr_body = d,
                    ranges = Dict{String, Any}(
                        n => Any[lo, hi] for (n, lo, hi) in crange
                    ),
                    filter = _conj(gc)
                )
                coef = _simp(mul(red_sum, scale...))
                iszero_lit(coef) && continue
                _push_band!(out, rows2, oidx, cidx, coef, extra, _conj(_conj(gr), g), v, ctx)
            else
                # A gathered column (`u[j]`, `u[i+k]`, `u[conn[i,k]]`): a band
                # with free contracted column dimensions; entries landing on
                # one column accumulate at assembly.
                coef = _simp(mul(d, scale...))
                iszero_lit(coef) && continue
                _push_band!(out, rows2, oidx, cidx, coef, vcat(crange, extra), _conj(guard, g), v, ctx)
            end
        end
    end
    return
end

function _bands_makearray!(out, e::OpExpr, v::String, ctx::Ctx, scale)
    for (region, val) in zip(e.regions, e.values)
        rows = [(_region_bound(r[1]), _region_bound(r[2])) for r in region]
        any(((lo, hi),) -> hi < lo, rows) && continue        # folded-empty region
        nonsing = [k for (k, (lo, hi)) in enumerate(rows) if hi > lo]
        if is_array_expr(val, ctx)
            sub = Band[]
            # A full-rank faq value names every region dim explicitly
            # (singleton dims included); a reduced-rank value covers only the
            # non-singleton dims.
            val_fullrank = val isa OpExpr && val.op == "faq" &&
                length(val.output_idx) == length(rows)
            sub_shape = val_fullrank ?
                [hi - lo + 1 for (lo, hi) in rows] :
                [rows[k][2] - rows[k][1] + 1 for k in nonsing]
            _bands_array!(sub, val, v, ctx, sub_shape, scale)
            # The value's bands are in the value's OWN index values (an
            # faq node ranging i ∈ [olo, ohi] fills region positions
            # lo..lo+(ohi−olo)); shift each dim by (lo − origin), where the
            # ORIGIN is the value range's start — NOT the band's own row
            # start, which for a region-clipped sub-band (clip.jl) sits past
            # the origin. Singletons are pinned.
            origins = if val isa OpExpr && val.op == "faq"
                vr = val.ranges === nothing ? Dict{String, Any}() : val.ranges
                Int[
                    haskey(vr, String(n)) ? _range_of(vr[String(n)], ctx)[1] : 1
                        for n in val.output_idx
                ]
            else
                ones(Int, length(nonsing))   # identity/broadcast bands are 1-based
            end
            for b in sub
                ridx = Union{String, Int}[]; subst = Dict{String, ASTExpr}()
                rows2 = Tuple{Int, Int}[]
                fullrank = length(b.ridx) == length(rows)
                kk = 0
                for (k, (lo, hi)) in enumerate(rows)
                    if fullrank || hi > lo
                        kk += 1
                        r = b.ridx[kk]; rlo, rhi = b.rows[kk]
                        shift = lo - origins[kk]
                        if r isa String
                            push!(ridx, r)
                            shift != 0 && (subst[r] = add(VarExpr(r), lit(-shift)))
                            push!(rows2, (rlo + shift, rhi + shift))
                        else
                            push!(ridx, r + shift); push!(rows2, (r + shift, r + shift))
                        end
                    else
                        push!(ridx, lo); push!(rows2, (lo, lo))
                    end
                end
                coef = isempty(subst) ? b.coef : substitute(b.coef, subst)
                cidx = isempty(subst) ? b.cidx :
                    ASTExpr[substitute(c, subst) for c in b.cidx]
                guard = isempty(subst) || b.guard === nothing ? b.guard :
                    substitute(b.guard, subst)
                push!(out, Band(rows2, ridx, cidx, coef, b.contracted, guard))
            end
        else
            # Scalar value broadcast over the region.
            _scalar_bands!(
                out, val, v, ctx, rows,
                Union{String, Int}[lo for (lo, _) in rows], scale
            )
        end
    end
    return
end

# Index-pushdown for bare elementwise array expressions: A ↦ index(A, idx…).
function _push_index(e::ASTExpr, idx::Vector{ASTExpr}, ctx::Ctx)::ASTExpr
    if e isa VarExpr
        return is_array(ctx, e.name) ? op("index", e, idx...) : e
    elseif e isa OpExpr
        e.op in ("faq", "makearray", "index") &&
            throw(
            BandError(
                "cannot push an index into `$(e.op)` " *
                    "nested under an elementwise op"
            )
        )
        return map_children(x -> _push_index(x, idx, ctx), e)
    end
    return e
end

"""
    normalize_band(b::Band) -> Band

Pin any free row index whose range is a single cell (a periodic wrap region,
a one-cell boundary face) to that constant, substituting it through `cidx`,
`coef` and `guard`. Keeps emitted coefficient faq nodes rank-consistent.
"""
function normalize_band(b::Band)::Band
    subst = Dict{String, ASTExpr}(); ridx = Union{String, Int}[]
    for (k, r) in enumerate(b.ridx)
        lo, hi = b.rows[k]
        if r isa String && lo == hi
            subst[r] = lit(lo); push!(ridx, lo)
        else
            push!(ridx, r)
        end
    end
    contracted = Tuple{String, Int, Int}[]
    for (n, lo, hi) in b.contracted           # a one-value contraction pins too
        lo == hi ? (subst[n] = lit(lo)) : push!(contracted, (n, lo, hi))
    end
    isempty(subst) && return b
    return Band(
        b.rows, ridx,
        ASTExpr[_canon_lits(simplify(substitute(c, subst))) for c in b.cidx],
        _simp(substitute(b.coef, subst)), contracted,
        b.guard === nothing ? nothing : _canon_lits(simplify(substitute(b.guard, subst)))
    )
end
