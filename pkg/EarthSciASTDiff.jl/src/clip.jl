# Static region clipping (README roadmap item). The `index(makearray…)`
# lowering (inline.jl) turns every region read into a nested membership
# `ifelse` — `ifelse(and(f(i) >= lo, f(i) <= hi), value, older)` — whose
# conditions are AFFINE in the band's row indices with LITERAL bounds. Those
# guards are statically decidable once the row range is split at their
# breakpoints: on each subrange every membership condition is uniformly true
# or uniformly false, the `ifelse` collapses to one branch, and the band
# calculus emits EXACT per-subrange bands — no membership guards in the
# coefficients, no branch-union slack in the pattern.
#
# Everything here is conservative: a condition that is not a literal-bounded
# affine comparison in exactly one row index (a state-dependent limiter
# switch, a mixed-index test, a non-affine map) contributes no breakpoints
# and is simply left in place, so the result degrades gracefully to today's
# branch-union behavior.

# ── affine forms ─────────────────────────────────────────────────────────────

# `e` as α·name + β over the single index `name` (other variables forbidden).
# Returns (α, β) or nothing. Integer arithmetic only.
function _affine(e::ASTExpr, name::String)
    if e isa VarExpr
        return e.name == name ? (1, 0) : nothing
    elseif e isa IntExpr
        return (0, Int(e.value))
    elseif e isa NumExpr
        return isinteger(e.value) ? (0, Int(e.value)) : nothing
    elseif e isa OpExpr
        a = e.args
        if e.op == "+"
            α = 0; β = 0
            for x in a
                t = _affine(x, name); t === nothing && return nothing
                α += t[1]; β += t[2]
            end
            return (α, β)
        elseif e.op == "-" && length(a) == 1
            t = _affine(a[1], name); t === nothing && return nothing
            return (-t[1], -t[2])
        elseif e.op == "-" && length(a) == 2
            t1 = _affine(a[1], name); t1 === nothing && return nothing
            t2 = _affine(a[2], name); t2 === nothing && return nothing
            return (t1[1] - t2[1], t1[2] - t2[2])
        elseif e.op == "*"
            α = 0; β = 1   # product of affines is affine only if ≤1 has a slope
            for x in a
                t = _affine(x, name); t === nothing && return nothing
                if t[1] == 0
                    α *= t[2]; β *= t[2]
                elseif α == 0 && β != 0    # β is the accumulated constant
                    α = β * t[1]; β = β * t[2]
                else
                    return nothing
                end
            end
            return (α, β)
        end
    end
    return nothing
end

# The single row-index name occurring in `e`, if exactly one of `names` does.
function _sole_name(e::ASTExpr, names)
    found = nothing
    ok = Ref(true)
    function walk(x)
        x isa VarExpr && x.name in names && begin
            found === nothing ? (found = x.name) : (found == x.name || (ok[] = false))
        end
        x isa OpExpr && map_children(c -> (walk(c); c), x)
        return
    end
    walk(e)
    return ok[] ? found : nothing
end

# ── breakpoints ──────────────────────────────────────────────────────────────

const _CMP_OPS = ("<", "<=", ">", ">=")

# A decidable comparison: one side affine in exactly one row index with the
# other side (folded into β) constant. The truth of `α·i + β ⋈ 0` is
# monotone in i, so it flips between exactly one pair of consecutive
# integers around the rational root −β/α — record that single flip point
# (the first i whose truth differs from i−1) as the interval-start cut.
_cmp_truth(o::String, α::Int, β::Int, i::Int) =
    o == "<"  ? α * i + β < 0  :
    o == "<=" ? α * i + β <= 0 :
    o == ">"  ? α * i + β > 0  : α * i + β >= 0

function _breakpoints!(bp::Dict{String,Set{Int}}, e::ASTExpr, names)
    e isa OpExpr || return
    if e.op in _CMP_OPS && length(e.args) == 2
        n = _sole_name(e, names)
        if n !== nothing
            diff = op("-", e.args[1], e.args[2])
            t = _affine(diff, n)
            if t !== nothing && t[1] != 0
                q = fld(-t[2], t[1])           # floor of the rational root
                for c in (q, q + 1)
                    _cmp_truth(e.op, t[1], t[2], c - 1) ==
                        _cmp_truth(e.op, t[1], t[2], c) ||
                        push!(get!(bp, n, Set{Int}()), c)
                end
            end
        end
    end
    map_children(c -> (_breakpoints!(bp, c, names); c), e)
    return
end

# ── uniform-truth evaluation and guard decision ──────────────────────────────

# Truth of `cond` over the box `env` (name → (lo, hi) inclusive), three-valued:
# true / false when uniform, nothing when it cannot be decided statically.
function _uniform(cond::ASTExpr, env::Dict{String,Tuple{Int,Int}})
    cond isa OpExpr || return nothing
    if cond.op in _CMP_OPS && length(cond.args) == 2
        n = _sole_name(cond, keys(env))
        n === nothing && return nothing
        haskey(env, n) || return nothing
        diff = op("-", cond.args[1], cond.args[2])
        t = _affine(diff, n)
        t === nothing && return nothing
        lo, hi = env[n]
        vlo = t[1] * lo + t[2]; vhi = t[1] * hi + t[2]
        a, b = minmax(vlo, vhi)
        f = cond.op == "<"  ? (x -> x < 0)  :
            cond.op == "<=" ? (x -> x <= 0) :
            cond.op == ">"  ? (x -> x > 0)  : (x -> x >= 0)
        return f(a) == f(b) ? f(a) : nothing
    elseif cond.op == "and"
        anynothing = false
        for x in cond.args
            u = _uniform(x, env)
            u === false && return false
            u === nothing && (anynothing = true)
        end
        return anynothing ? nothing : true
    elseif cond.op == "or"
        anynothing = false
        for x in cond.args
            u = _uniform(x, env)
            u === true && return true
            u === nothing && (anynothing = true)
        end
        return anynothing ? nothing : false
    elseif cond.op == "not" && length(cond.args) == 1
        u = _uniform(cond.args[1], env)
        return u === nothing ? nothing : !u
    end
    return nothing
end

# Collapse every statically-decided `ifelse` in `e` over the box `env`, and
# fold decided comparisons used as VALUES (`(i >= 2) * x`) to their 1/0
# literal so `simplify` can absorb them.
function _decide(e::ASTExpr, env::Dict{String,Tuple{Int,Int}})::ASTExpr
    e isa OpExpr || return e
    if e.op == "ifelse" && length(e.args) == 3
        u = _uniform(e.args[1], env)
        u === true  && return _decide(e.args[2], env)
        u === false && return _decide(e.args[3], env)
    elseif e.op in _CMP_OPS
        u = _uniform(e, env)
        u === nothing || return lit(u ? 1 : 0)
    end
    return map_children(x -> _decide(x, env), e)
end

# ── the clip driver ──────────────────────────────────────────────────────────

"""
    clip_regions(body, ridx, rows) -> Vector{(rows′, body′)}

Partition the band row ranges `rows` (per-dimension inclusive `(lo, hi)`,
free dimensions named by the strings in `ridx`) at the breakpoints of every
statically-decidable comparison in `body`, and on each cell of the partition
collapse the decided `ifelse` guards. Returns the per-subrange bodies; a
body with no decidable guards returns `[(rows, body)]` unchanged. Subranges
are emitted in ascending order per dimension (deterministic).
"""
function clip_regions(body::ASTExpr, ridx, rows)
    names = String[r for r in ridx if r isa String]
    isempty(names) && return [(collect(Tuple{Int,Int}, rows), body)]
    bp = Dict{String,Set{Int}}()
    _breakpoints!(bp, body, names)
    isempty(bp) && return [(collect(Tuple{Int,Int}, rows), body)]

    # per-dimension subinterval lists (split only dims that have breakpoints)
    dims = Vector{Vector{Tuple{Int,Int}}}()
    dimname = Vector{Union{String,Nothing}}()
    for (k, r) in enumerate(ridx)
        lo, hi = rows[k]
        if r isa String && haskey(bp, r)
            cuts = sort!([c for c in bp[r] if lo < c <= hi])
            ivs = Tuple{Int,Int}[]
            a = lo
            for c in cuts
                push!(ivs, (a, c - 1)); a = c
            end
            push!(ivs, (a, hi))
            push!(dims, ivs); push!(dimname, r)
        else
            push!(dims, [(lo, hi)]); push!(dimname, r isa String ? r : nothing)
        end
    end

    out = Vector{Tuple{Vector{Tuple{Int,Int}},ASTExpr}}()
    for cell in Iterators.product(dims...)
        rows2 = collect(Tuple{Int,Int}, cell)
        env = Dict{String,Tuple{Int,Int}}(
            dimname[k] => rows2[k] for k in eachindex(rows2) if dimname[k] !== nothing)
        push!(out, (rows2, _decide(body, env)))
    end
    return out
end

# ── re-merging ───────────────────────────────────────────────────────────────
#
# Clipping partitions per BODY, so every site in a composed equation is split
# at every breakpoint — including sites (a pointwise chemistry term under
# `operator_compose`) whose coefficient is identical on both sides of a cut.
# Because the split is a per-dimension product partition, bands from one site
# whose (row names, column indices, coefficient) agree re-merge exactly by
# repeated single-dimension coalescing of adjacent row ranges.

"""
    merge_bands(bs::Vector{Band}) -> Vector{Band}

Coalesce bands that are identical in `ridx`, `cidx`, and `coef` (structural
keys) and whose `rows` boxes are adjacent along exactly one dimension.
Iterated to a fixpoint; band order is preserved (a merge keeps the
earlier band's position), so emission stays deterministic.

SCALING. Only bands that already agree on (`ridx`, `cidx`, `coef`,
`contracted`) can ever merge, but the sweep compared EVERY pair to discover
that — Θ(b²) key comparisons per sweep and Θ(b³) to the fixpoint, each
comparison walking strings as long as the coefficients are big. Bucketing by
key first costs one key per band and makes the search Θ(Σ bₖ²) over the
buckets. The merges themselves are unchanged: a cross-bucket pair could
never have merged. Buckets are searched in ascending original index and the
survivors are read out in original order, so the emitted band order cannot
see the bucket iteration order.

(The key stays `skey`. It is taken ONCE per band and never compared
elementwise afterwards, which is the one regime where a whole-subtree
serialization beats interning it node by node.)
"""
function merge_bands(bs::Vector{Band})::Vector{Band}
    length(bs) < 2 && return bs
    key(b) = (b.ridx, [skey(c) for c in b.cidx], skey(b.coef), b.contracted)
    buckets = Dict{Any,Vector{Int}}()
    for (i, b) in enumerate(bs)
        push!(get!(buckets, key(b), Int[]), i)
    end
    out = collect(Band, bs)
    alive = trues(length(out))
    for idxs in values(buckets)
        length(idxs) < 2 && continue
        changed = true
        while changed
            changed = false
            for ai in eachindex(idxs)
                a = idxs[ai]
                alive[a] || continue
                for bi in (ai+1):length(idxs)
                    b = idxs[bi]
                    alive[b] || continue
                    ra, rb = out[a].rows, out[b].rows
                    length(ra) == length(rb) || continue
                    d = 0; ok = true
                    for k in eachindex(ra)
                        if ra[k] == rb[k]
                            continue
                        elseif d == 0 &&
                               (ra[k][2] + 1 == rb[k][1] || rb[k][2] + 1 == ra[k][1])
                            d = k
                        else
                            ok = false; break
                        end
                    end
                    (ok && d != 0) || continue
                    lo = min(ra[d][1], rb[d][1]); hi = max(ra[d][2], rb[d][2])
                    rows = copy(ra); rows[d] = (lo, hi)
                    out[a] = Band(rows, out[a].ridx, out[a].cidx, out[a].coef,
                                  out[a].contracted)
                    alive[b] = false
                    changed = true
                end
            end
        end
    end
    return out[alive]
end
