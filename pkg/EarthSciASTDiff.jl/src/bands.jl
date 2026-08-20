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
    `reduce: "+"` aggregate's sum over a gathered column materializes).
    Empty for ordinary bands.

The band is the symbolic, portable analogue of the tree-walk's `_AccDesc`
state-read descriptors: `cidx` of the form `row ± const` is an affine stencil
offset, `cidx == ridx` is a diagonal (per-cell) coupling, and a `cidx`
referencing a contracted name is an indirect/contracted gather
(source–receptor matrices, regrid joins, `conn[]`-table stencils).
"""
struct Band
    rows::Vector{Tuple{Int,Int}}
    ridx::Vector{Union{String,Int}}
    cidx::Vector{ASTExpr}
    coef::ASTExpr
    contracted::Vector{Tuple{String,Int,Int}}
end
Band(rows, ridx, cidx, coef) = Band(rows, ridx, cidx, coef, Tuple{String,Int,Int}[])

"Differentiation context: index-set sizes, array-variable shapes, parameter names."
struct Ctx
    index_sets::Dict{String,Int}
    shapes::Dict{String,Vector{Int}}
    params::Set{String}
end
is_array(ctx::Ctx, name) = haskey(ctx.shapes, name)

struct BandError <: Exception
    msg::String
end
Base.showerror(io::IO, e::BandError) = print(io, "BandError: ", e.msg)

function _range_of(r, ctx::Ctx)::Tuple{Int,Int}
    if r isa IndexSetRef
        return (1, ctx.index_sets[r.from])
    elseif r isa AbstractVector
        return (_region_bound(r[1]), _region_bound(r[end]))
    elseif r isa AbstractDict
        return (1, ctx.index_sets[String(r["from"])])
    end
    throw(BandError("unsupported range spec $r"))
end

# All distinct sites of `v` inside a scalar body: the bare VarExpr and every
# `index(v, …)` node, keyed by canonical JSON.
function sites_of(e::ASTExpr, v::String, acc = Dict{String,Site}())
    if e isa VarExpr && e.name == v
        s = Site(e); acc[s.key] = s
    elseif e isa OpExpr && e.op == "index" && e.args[1] isa VarExpr && e.args[1].name == v
        s = Site(e); acc[s.key] = s
    elseif e isa OpExpr
        map_children(x -> (sites_of(x, v, acc); x), e)
    end
    return acc
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
    e.op in ("aggregate", "makearray", "arrayop", "broadcast",
             "reshape", "transpose", "concat") && return true
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
        for s in sorted_sites(rhs, v)
            coef = _simp(dscalar(rhs, s))
            iszero_lit(coef) && continue
            cidx = s.expr isa VarExpr ? ASTExpr[] : ASTExpr[s.expr.args[2:end]...]
            push!(out, Band(Tuple{Int,Int}[], Union{String,Int}[], cidx, coef))
        end
        return out
    end
    _bands_array!(out, rhs, v, ctx, shape_u, ASTExpr[lit(1)])
    return out
end

# Emit one band per site of a scalar-valued expression broadcast over `rows`,
# statically clipping decidable region guards first (clip.jl) so membership
# `ifelse` never reaches the coefficients and the pattern is per-subrange
# exact.
function _scalar_bands!(out, e::ASTExpr, v::String, rows, ridx, scale)
    for (rows2, body) in clip_regions(e, ridx, rows)
        for s in sorted_sites(body, v)
            coef = _simp(mul(dscalar(body, s), scale...))
            iszero_lit(coef) && continue
            cidx = s.expr isa VarExpr ? ASTExpr[] : ASTExpr[s.expr.args[2:end]...]
            push!(out, Band(rows2, collect(Union{String,Int}, ridx), cidx, coef))
        end
    end
end

# `scale` accumulates elementwise scalar multipliers from enclosing `*`, `-`,
# `/`, and `neg` nodes on the way down to array leaves.
function _bands_array!(out, e::ASTExpr, v::String, ctx::Ctx, shape_u, scale::Vector{ASTExpr})
    if e isa VarExpr
        e.name == v || return out
        names = ["_r$k" for k in 1:length(shape_u)]           # identity band
        push!(out, Band([(1, n) for n in shape_u], Union{String,Int}[names...],
                        ASTExpr[VarExpr(n) for n in names], mul(scale...)))
        return out
    end
    e isa OpExpr || return out
    o = e.op
    if o == "+"
        for x in e.args
            _bands_array!(out, x, v, ctx, shape_u, scale)
        end
    elseif o == "-"
        _bands_array!(out, e.args[1], v, ctx, shape_u, scale)
        length(e.args) == 2 &&
            _bands_array!(out, e.args[2], v, ctx, shape_u, vcat(scale, [lit(-1)]))
    elseif o == "neg"
        _bands_array!(out, e.args[1], v, ctx, shape_u, vcat(scale, [lit(-1)]))
    elseif o == "*"
        arr = [x for x in e.args if is_array_expr(x, ctx)]
        sc  = [x for x in e.args if !is_array_expr(x, ctx)]
        length(arr) == 1 || throw(BandError(
            "elementwise product of $(length(arr)) array-valued factors at array " *
            "level; wrap in an aggregate"))
        any(x -> occurs_var(x, v), sc) && throw(BandError(
            "scalar factor depends on `$v` in an array-level product"))
        _bands_array!(out, arr[1], v, ctx, shape_u, vcat(scale, sc))
    elseif o == "/"
        is_array_expr(e.args[2], ctx) &&
            throw(BandError("array-valued divisor at array level"))
        _bands_array!(out, e.args[1], v, ctx, shape_u,
                      vcat(scale, [op("/", lit(1), e.args[2])]))
    elseif o == "makearray"
        _bands_makearray!(out, e, v, ctx, scale)
    elseif o == "aggregate"
        _bands_aggregate!(out, e, v, ctx, shape_u, scale)
    elseif o == "index"
        # A scalar read of some array in an array-level position: a broadcast.
        names = ["_r$k" for k in 1:length(shape_u)]
        _scalar_bands!(out, e, v, [(1, n) for n in shape_u], names, scale)
    else
        if is_array_expr(e, ctx)
            # Generic elementwise op over whole arrays (exp(c), c^2, ifelse):
            # push the index inward, then recurse as an aggregate.
            names = ["_r$k" for k in 1:length(shape_u)]
            idx = ASTExpr[VarExpr(n) for n in names]
            body = _push_index(e, idx, ctx)
            agg = OpExpr("aggregate", ASTExpr[];
                         output_idx = Any[names...], expr_body = body,
                         ranges = Dict{String,Any}(
                             n => Any[1, shape_u[k]] for (k, n) in enumerate(names)))
            _bands_array!(out, agg, v, ctx, shape_u, scale)
        else
            # Scalar expression broadcast over u's shape.
            names = ["_r$k" for k in 1:length(shape_u)]
            _scalar_bands!(out, e, v, [(1, n) for n in shape_u], names, scale)
        end
    end
    return out
end

function _bands_aggregate!(out, e::OpExpr, v::String, ctx::Ctx, shape_u, scale)
    oidx = [String(x) for x in e.output_idx if x isa AbstractString]
    length(oidx) == length(e.output_idx) || throw(BandError(
        "singleton `1` output_idx entries not supported yet"))
    e.filter === nothing || throw(BandError(
        "filtered aggregates not supported (the filter gate would be dropped " *
        "from the derivative)"))
    ranges = e.ranges === nothing ? Dict{String,Any}() : e.ranges
    cnames = sort!([String(n) for n in keys(ranges) if !(String(n) in oidx)])
    e.reduce === nothing || e.reduce == "+" || throw(BandError(
        "reduce=`$(e.reduce)` (non-smooth semiring reductions have no bands)"))
    crange = Tuple{String,Int,Int}[
        (n, _range_of(ranges[n], ctx)...) for n in cnames]
    rows = Tuple{Int,Int}[]
    for (k, n) in enumerate(oidx)
        push!(rows, haskey(ranges, n) ? _range_of(ranges[n], ctx) : (1, shape_u[k]))
    end
    body = e.expr_body
    for (rows2, body2) in clip_regions(body, oidx, rows)
        for s in sorted_sites(body2, v)
            d = dscalar(body2, s)
            s.expr isa VarExpr && is_array(ctx, v) && throw(BandError(
                "whole-array reference to `$v` inside an aggregate body"))
            cidx = s.expr isa VarExpr ? ASTExpr[] : ASTExpr[s.expr.args[2:end]...]
            if isempty(crange)
                coef = _simp(mul(d, scale...))
                iszero_lit(coef) && continue
                push!(out, Band(rows2, Union{String,Int}[oidx...], cidx, coef))
            elseif !any(n -> any(c -> n in free_variables(c), cidx),
                        first.(crange))
                # The column is independent of every contracted index: a
                # plain band whose coefficient is the (scalar) `reduce: "+"`
                # of the site derivative over the contraction.
                red = OpExpr("aggregate", ASTExpr[];
                             output_idx = Any[], reduce = "+", expr_body = d,
                             ranges = Dict{String,Any}(
                                 n => Any[lo, hi] for (n, lo, hi) in crange))
                coef = _simp(mul(red, scale...))
                iszero_lit(coef) && continue
                push!(out, Band(rows2, Union{String,Int}[oidx...], cidx, coef))
            else
                # A gathered column (`u[j]`, `u[i+k]`, `u[conn[i,k]]`): a band
                # with free contracted column dimensions; entries landing on
                # one column accumulate at assembly.
                coef = _simp(mul(d, scale...))
                iszero_lit(coef) && continue
                push!(out, Band(rows2, Union{String,Int}[oidx...], cidx, coef,
                                crange))
            end
        end
    end
end

function _bands_makearray!(out, e::OpExpr, v::String, ctx::Ctx, scale)
    for (region, val) in zip(e.regions, e.values)
        rows = [( _region_bound(r[1]), _region_bound(r[2]) ) for r in region]
        any(((lo, hi),) -> hi < lo, rows) && continue        # folded-empty region
        nonsing = [k for (k, (lo, hi)) in enumerate(rows) if hi > lo]
        if is_array_expr(val, ctx)
            sub = Band[]
            # A full-rank aggregate value names every region dim explicitly
            # (singleton dims included); a reduced-rank value covers only the
            # non-singleton dims.
            val_fullrank = val isa OpExpr && val.op == "aggregate" &&
                           length(val.output_idx) == length(rows)
            sub_shape = val_fullrank ?
                [hi - lo + 1 for (lo, hi) in rows] :
                [rows[k][2] - rows[k][1] + 1 for k in nonsing]
            _bands_array!(sub, val, v, ctx, sub_shape, scale)
            # The value's bands are in the value's OWN index values (an
            # aggregate ranging i ∈ [olo, ohi] fills region positions
            # lo..lo+(ohi−olo)); shift each dim by (lo − origin), where the
            # ORIGIN is the value range's start — NOT the band's own row
            # start, which for a region-clipped sub-band (clip.jl) sits past
            # the origin. Singletons are pinned.
            origins = if val isa OpExpr && val.op == "aggregate"
                vr = val.ranges === nothing ? Dict{String,Any}() : val.ranges
                Int[haskey(vr, String(n)) ? _range_of(vr[String(n)], ctx)[1] : 1
                    for n in val.output_idx]
            else
                ones(Int, length(nonsing))   # identity/broadcast bands are 1-based
            end
            for b in sub
                ridx = Union{String,Int}[]; subst = Dict{String,ASTExpr}()
                rows2 = Tuple{Int,Int}[]
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
                push!(out, Band(rows2, ridx, cidx, coef, b.contracted))
            end
        else
            # Scalar value broadcast over the region.
            _scalar_bands!(out, val, v, rows,
                           Union{String,Int}[lo for (lo, _) in rows], scale)
        end
    end
end

# Index-pushdown for bare elementwise array expressions: A ↦ index(A, idx…).
function _push_index(e::ASTExpr, idx::Vector{ASTExpr}, ctx::Ctx)::ASTExpr
    if e isa VarExpr
        return is_array(ctx, e.name) ? op("index", e, idx...) : e
    elseif e isa OpExpr
        e.op in ("aggregate", "makearray", "index") &&
            throw(BandError("cannot push an index into `$(e.op)` " *
                            "nested under an elementwise op"))
        return map_children(x -> _push_index(x, idx, ctx), e)
    end
    return e
end

"""
    normalize_band(b::Band) -> Band

Pin any free row index whose range is a single cell (a periodic wrap region,
a one-cell boundary face) to that constant, substituting it through `cidx`
and `coef`. Keeps emitted coefficient aggregates rank-consistent.
"""
function normalize_band(b::Band)::Band
    subst = Dict{String,ASTExpr}(); ridx = Union{String,Int}[]
    for (k, r) in enumerate(b.ridx)
        lo, hi = b.rows[k]
        if r isa String && lo == hi
            subst[r] = lit(lo); push!(ridx, lo)
        else
            push!(ridx, r)
        end
    end
    contracted = Tuple{String,Int,Int}[]
    for (n, lo, hi) in b.contracted           # a one-value contraction pins too
        lo == hi ? (subst[n] = lit(lo)) : push!(contracted, (n, lo, hi))
    end
    isempty(subst) && return b
    return Band(b.rows, ridx,
                ASTExpr[_canon_lits(simplify(substitute(c, subst))) for c in b.cidx],
                _simp(substitute(b.coef, subst)), contracted)
end
