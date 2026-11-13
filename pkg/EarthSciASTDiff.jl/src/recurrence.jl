# Causal self-reference recurrences (esm-jacobian-spec.md §4.5; esm-spec
# §4.3.1.1): an observed `V` whose defining `faq` reads `V` itself at a
# strictly earlier cell along one of its output axes. Such an observed has no
# inlining fixpoint, so it is differentiated as an INTERMEDIATE: the band
# calculus runs on its definition like on an equation (rows `V`), and on every
# reader with `V` as an extra column (the extended system, `ExtendedBands`,
# jacobian.jl). The chain rule through the sweep is then either materialized —
# composed with the transfer recurrence `T = (I − G)⁻¹` of the line (the
# block's `auxiliary`, §4.5) — or left matrix-free as tangent and adjoint
# sweeps in the product documents (§9.7, products.jl). Everything here is an
# AST: the sweeps are ESM recurrences that EarthSciAST executes. The Rust
# (`recurrence.rs`) and Python (`recurrence.py`) bindings mirror this file.

"""
    Recurrence

One causal recurrence (esm-jacobian-spec.md §4.5): the observed `name`,
defined by a `faq` over the cell frame `frame` (its `output_idx`) with ranges
`rows`, folding along frame dimension `dim` (1-based). `shape` is the
declared shape of `name`, as written (index-set names).
"""
struct Recurrence
    name::String
    frame::Vector{String}
    rows::Vector{Tuple{Int, Int}}
    dim::Int
    shape::Vector{String}
end

# `lo + hi` of the recurrence axis: the reversal `k ↦ lo + hi − k`.
_flip(r::Recurrence) = r.rows[r.dim][1] + r.rows[r.dim][2]
_lo(r::Recurrence) = r.rows[r.dim][1]

# An index name, a pinned literal or an expression, as an expression.
_ex(x::ASTExpr) = x
_ex(x::AbstractString) = VarExpr(String(x))
_ex(x::Integer) = lit(x)

_vars(names) = ASTExpr[VarExpr(n) for n in names]

_index(v::String, idx::Vector{ASTExpr}) = isempty(idx) ? VarExpr(v) : op("index", VarExpr(v), idx...)

_and_all(cs) = isempty(cs) ? nothing : length(cs) == 1 ? cs[1] : op("and", cs...)

# A `faq` node with the given output indices, ranges and body.
_faq(output_idx, ranges, expr::ASTExpr) = OpExpr(
    "faq", ASTExpr[]; output_idx = Any[output_idx...],
    ranges = Dict{String, Any}(n => Any[lo, hi] for (n, lo, hi) in ranges),
    expr_body = expr
)

# A left fold of binary `+` (§9.1), `zero` for no term.
_plus_fold(terms, zero::ASTExpr) =
    isempty(terms) ? zero : foldl((a, t) -> op("+", a, t), terms)

_frame_ranges(rec::Recurrence) =
    Tuple{String, Int, Int}[(f, lo, hi) for (f, (lo, hi)) in zip(rec.frame, rec.rows)]

# ── recognition ──────────────────────────────────────────────────────────────

# The self-reads of `v` in `e` (each read's index arguments) and whether `v`
# also occurs bare. A faq that binds `v` as an index hides it: there the name
# is that index, never a read of the observed.
function _self_reads!(e::ASTExpr, v::String, reads::Vector{Vector{ASTExpr}}, bare::Ref{Bool})
    if e isa VarExpr
        e.name == v && (bare[] = true)
        return
    end
    e isa OpExpr || return
    e.op == "faq" && v in _binders(e) && return
    if e.op == "index" && !isempty(e.args) && e.args[1] isa VarExpr && e.args[1].name == v
        push!(reads, ASTExpr[e.args[2:end]...])
        for x in e.args[2:end]
            _self_reads!(x, v, reads, bare)
        end
        return
    end
    map_children(x -> (_self_reads!(x, v, reads, bare); x), e)
    return
end

function _is_candidate(d::ASTExpr, v::String)
    reads = Vector{ASTExpr}[]
    bare = Ref(false)
    _self_reads!(d, v, reads, bare)
    return !isempty(reads) || bare[]
end

"""
    recurrences(sv::SysView, obs = _observed_defs(sv), ctx = _ctx(sv)) -> Vector{Recurrence}

Every recurrence of the view (esm-jacobian-spec.md §4.5), in equation order:
each observed whose definition reads itself, with its frame and axis read off
the definition. A self-read that cannot be placed on one axis is refused
with a `BandError` (EarthSciAST rejects such a document anyway; the producer
never guesses).
"""
function recurrences(sv::SysView, obs = _observed_defs(sv), ctx = _ctx(sv))
    out = Recurrence[]
    for eq in sv.equations
        n = _defined_name(eq)
        (n === nothing || !haskey(obs, n)) && continue
        (any(r -> r.name == n, out) || !_is_candidate(obs[n], n)) && continue
        push!(out, _recognize(n, obs[n], sv, ctx))
    end
    return out
end

function _recognize(v::String, d::ASTExpr, sv::SysView, ctx::Ctx)
    fail(why) = BandError("recurrence `$v`: $why (esm-spec §4.3.1.1)")
    var = get(sv.variables, v, nothing)
    shape = var === nothing || var.shape === nothing ? String[] : String[string(x) for x in var.shape]
    ext = get(ctx.shapes, v, Int[])
    (
        d isa OpExpr && d.op == "faq" && d.output_idx !== nothing &&
            !isempty(d.output_idx) && length(d.output_idx) == length(ext)
    ) || throw(fail("its definition is not a faq over its frame"))
    all(x -> x isa AbstractString, d.output_idx) ||
        throw(fail("a singleton output index is not a frame symbol"))
    frame = String[String(x) for x in d.output_idx]
    ranges = something(d.ranges, Dict{String, Any}())
    rows = Tuple{Int, Int}[
        haskey(ranges, f) ? _range_of(ranges[f], ctx) : (1, ext[m]) for (m, f) in enumerate(frame)
    ]
    reads = Vector{ASTExpr}[]
    bare = Ref(false)
    _self_reads!(d, v, reads, bare)
    bare[] && throw(fail("a bare self-reference has no cell"))
    dims = Int[]
    for args in reads
        length(args) == length(frame) ||
            throw(fail("a self-read does not index every frame dimension"))
        for (m, a) in enumerate(args)
            (a isa VarExpr && a.name == frame[m]) || m in dims || push!(dims, m)
        end
    end
    length(dims) == 1 && return Recurrence(v, frame, rows, only(dims), shape)
    isempty(dims) && throw(fail("a self-read reads its own cell"))
    throw(fail("self-reads move along more than one axis"))
end

# The names reachable from `seeds` through the observed definitions, read
# backwards: every observed whose definition (transitively) reads a seed.
function _reader_closure(obs, seeds)
    deps = Dict{String, Set{String}}(n => free_variables(e) for (n, e) in obs)
    keep = Set{String}()
    while true
        before = length(keep)
        for (n, fv) in deps
            n in keep && continue
            any(x -> x in seeds || x in keep, fv) && push!(keep, n)
        end
        length(keep) == before && return keep
    end
    return
end

# The recurrences that move with `seeds` (an axis's targets, or a pair's two
# target sets): the active intermediates of a pass.
function _active(recs::Vector{Recurrence}, obs, seeds)
    moving = _reader_closure(obs, seeds)
    return Recurrence[r for r in recs if r.name in moving]
end

# ── the lag of a self band ───────────────────────────────────────────────────

# `lag = k − (column index along the axis)` of a self band over its row index
# `k` (a name or a pinned literal). The column index must be affine in `k`
# with coefficient 1, and every other dimension the identity (esm-spec
# §4.3.1.1), or the band is refused; so is a literal lag below 1, a self-read
# that is not strictly earlier on the axis (its sweep term would be gated to
# zero, never evaluated).
function _lag(rec::Recurrence, b::Band)
    lag = _lag_expr(rec, b)
    s = _simp(lag)
    s isa IntExpr && s.value < 1 && throw(
        BandError(
            "recurrence `$(rec.name)`: a self-read at lag $(s.value) along " *
                "`$(rec.frame[rec.dim])` is not strictly earlier on its axis (esm-spec §4.3.1.1)"
        )
    )
    return lag
end

function _lag_expr(rec::Recurrence, b::Band)
    fail() = BandError(
        "recurrence `$(rec.name)`: a self-read is not `$(rec.frame[rec.dim])` minus a " *
            "lag along its axis (esm-spec §4.3.1.1)"
    )
    for (m, (r, c)) in enumerate(zip(b.ridx, b.cidx))
        m == rec.dim && continue
        same = (r isa String && c isa VarExpr && c.name == r) ||
            (r isa Int && c isa IntExpr && c.value == r)
        same || throw(
            BandError(
                "recurrence `$(rec.name)`: a self-read moves along dimension $m " *
                    "besides its axis (esm-spec §4.3.1.1)"
            )
        )
    end
    c = b.cidx[rec.dim]
    r = b.ridx[rec.dim]
    if !(r isa String)
        isempty(intersect(free_variables(c), rec.frame)) || throw(fail())
        return _simp(op("-", lit(r), c))
    end
    k = r
    has(e) = k in free_variables(e)
    isk(e) = e isa VarExpr && e.name == k
    c isa OpExpr || throw(fail())
    if c.op == "-" && length(c.args) == 2 && isk(c.args[1]) && !has(c.args[2])
        return c.args[2]
    end
    if c.op == "+"
        ks = count(isk, c.args)
        rest = ASTExpr[a for a in c.args if !isk(a)]
        (ks != 1 || any(has, rest)) && throw(fail())
        return _simp(op("-", add(rest...)))
    end
    throw(fail())
end

_lit_at_least(e::ASTExpr, lo::Int) = e isa IntExpr && e.value >= lo

# ── sweep terms ──────────────────────────────────────────────────────────────

# A band of V's definition placed on the sweep frame: the substitution of its
# row names by frame expressions and the conditions that keep the frame point
# inside its row box.
function _place(rec::Recurrence, b::Band, at::Vector{ASTExpr}, plain::Vector{Bool})
    sigma = Dict{String, ASTExpr}()
    conds = ASTExpr[]
    for (m, r) in enumerate(b.ridx)
        if r isa String
            sigma[r] = at[m]
            lo, hi = b.rows[m]
            flo, fhi = rec.rows[m]
            (!plain[m] || lo > flo) && push!(conds, op(">=", at[m], lit(lo)))
            (!plain[m] || hi < fhi) && push!(conds, op("<=", at[m], lit(hi)))
        else
            push!(conds, op("==", at[m], lit(r)))
        end
    end
    return sigma, conds
end

# One band's term in a sweep body: `coef · read` at the frame point, under
# `conds`, summed over the band's contracted names. The conditions gate the
# read through `ifelse`, which a recurrence body evaluates branch-first, so a
# guarded self-read is never reached outside its axis.
function _term(b::Band, sigma, conds, read::ASTExpr, zero::ASTExpr)
    conds = copy(conds)
    b.guard === nothing || push!(conds, _subst_free(b.guard, sigma))
    body = op("*", _subst_free(b.coef, sigma), read)
    c = _and_all(conds)
    c === nothing || (body = op("ifelse", c, body, zero))
    isempty(b.contracted) && return body
    return _faq(String[], b.contracted, body)
end

# The forward read of a self band: `self[frame with axis − lag, extra…]`, with
# the conditions that the lag is positive and the read cell is on the axis.
function _self_read(rec::Recurrence, b::Band, sigma, target::String, extra::Vector{ASTExpr})
    lag = _subst_free(_lag(rec, b), sigma)
    d = rec.dim
    pos = op("-", VarExpr(rec.frame[d]), lag)
    conds = ASTExpr[]
    _lit_at_least(lag, 1) || push!(conds, op(">=", lag, lit(1)))
    row_lo = b.ridx[d] isa String ? b.rows[d][1] : b.ridx[d]
    provable = lag isa IntExpr && row_lo - lag.value >= _lo(rec)
    provable || push!(conds, op(">=", pos, lit(_lo(rec))))
    idx = _vars(rec.frame)
    idx[d] = pos
    append!(idx, extra)
    return _index(target, idx), conds
end

# The forward (tangent) sweep of recurrence `rec`: one `faq` over its frame
# whose body adds, per band of `V`'s definition (`own`, rows `V`, columns
# `own_cols`), `coef · col(read)`; a self band reads the sweep variable
# `target` itself one lag back. `col(b, c, ci)` gives `(read, conds)` for
# another column, or `nothing` to skip it.
function _forward_sweep(rec::Recurrence, own, own_cols, target::String, col, zero::ASTExpr)
    at = _vars(rec.frame)
    plain = fill(true, length(at))
    terms = ASTExpr[]
    for (b, c) in zip(own, own_cols)
        sigma, conds = _place(rec, b, at, plain)
        if c == rec.name
            r, cs = _self_read(rec, b, sigma, target, ASTExpr[])
        else
            got = col(b, c, ASTExpr[_subst_free(x, sigma) for x in b.cidx])
            got === nothing && continue
            r, cs = got
        end
        push!(terms, _term(b, sigma, vcat(conds, cs), r, zero))
    end
    return _faq(rec.frame, _frame_ranges(rec), _plus_fold(terms, zero))
end

# The adjoint sweep of recurrence `rec`, written ascending on the REVERSED
# axis (the spec admits no backward lag): `target[o, k̃]` is the accumulated
# cotangent of `V[o, lo + hi − k̃]`, `src` plus, per self point `p` whose
# column is that cell, `coef(p) · V̂[row(p)]` — row `m + lag(p)` on the axis,
# i.e. `k̃ − lag` reversed.
function _adjoint_sweep(rec::Recurrence, own_self, target::String, src::ASTExpr, zero::ASTExpr)
    d = rec.dim
    actual = op("-", lit(_flip(rec)), VarExpr(rec.frame[d]))
    terms = ASTExpr[src]
    for b in own_self
        lag0 = _lag(rec, b)
        at = _vars(rec.frame)
        plain = fill(true, length(at))
        o_sigma = Dict{String, ASTExpr}(
            r => at[m] for (m, r) in enumerate(b.ridx) if m != d && r isa String
        )
        lag = _subst_free(lag0, o_sigma)
        at[d] = op("+", actual, lag)
        plain[d] = false
        sigma, conds = _place(rec, b, at, plain)
        pos = op("-", VarExpr(rec.frame[d]), lag)
        _lit_at_least(lag, 1) || push!(conds, op(">=", lag, lit(1)))
        push!(conds, op(">=", pos, lit(_lo(rec))))
        idx = _vars(rec.frame)
        idx[d] = pos
        push!(terms, _term(b, sigma, conds, _index(target, idx), zero))
    end
    return _faq(rec.frame, _frame_ranges(rec), _plus_fold(terms, zero))
end

# The actual cell of `rec` a reversed sweep `target` holds at row point `rho`:
# `target[rho with the axis reversed]`.
function _reversed_read(rec::Recurrence, target::String, rho)
    idx = ASTExpr[_ex(x) for x in rho]
    idx[rec.dim] = op("-", lit(_flip(rec)), idx[rec.dim])
    return _index(target, idx)
end

# ── the transfer recurrence and materialization ─────────────────────────────

function _source_names(sv::SysView)
    t = Set{String}(String(n) for n in keys(sv.variables))
    union!(t, String(n) for n in keys(sv.index_sets))
    for eq in sv.equations
        _all_names!(t, eq.lhs)
        _all_names!(t, eq.rhs)
    end
    return t
end

# The transfer variable of each recurrence: `_tr_<V>`, `_` appended until
# clear of every name of the source and of the names before it. Named over
# EVERY recurrence of the view, so a name never depends on which recurrences
# a pass has active.
function _transfer_names(sv::SysView, recs::Vector{Recurrence})
    taken = _source_names(sv)
    out = Dict{String, String}()
    for r in recs
        out[r.name] = _fresh!(taken, "_tr_$(r.name)", "_")
    end
    return out
end

# The self bands of `rec`, from its definition with every other observed
# inlined (no recurrence): the line operator `G`, independent of the axis a
# pass differentiates for.
function _self_bands(rec::Recurrence, obs, recs::Vector{Recurrence}, ctx::Ctx)
    names = Set{String}(r.name for r in recs)
    m = Dict{String, ASTExpr}(k => v for (k, v) in obs if !(k in names))
    d = inline_observed(obs[rec.name], m)
    shape = get(ctx.shapes, rec.name, Int[])
    return Band[normalize_band(b) for b in merge_bands(bands(d, rec.name, ctx; shape_u = shape))]
end

function _transfer_j(rec::Recurrence, bs::Vector{Band})
    taken = Set{String}(rec.frame)
    for b in bs
        _all_names!(taken, b.coef)
        union!(taken, first.(b.contracted))
    end
    j = "_tj"
    while j in taken
        j *= "_j"
    end
    return j
end

"""
    transfer_definition(rec::Recurrence, obs, recs, ctx, tname) -> ASTExpr

The transfer recurrence `T` of `rec` (esm-jacobian-spec.md §4.5): over the
frame plus one output index `j` on the axis, `T[o, k, j] = [k == j] +
Σ_{self bands} coef · T[o, k − lag, j]`, so `T[o, ·, ·] = (I − G)⁻¹` on each
line. Lower triangular on each line.
"""
function transfer_definition(rec::Recurrence, obs, recs::Vector{Recurrence}, ctx::Ctx, tname::String)
    bs = _self_bands(rec, obs, recs, ctx)
    j = _transfer_j(rec, bs)
    d = rec.dim
    at = _vars(rec.frame)
    plain = fill(true, length(at))
    terms = ASTExpr[op("ifelse", op("==", VarExpr(rec.frame[d]), VarExpr(j)), lit(1), lit(0))]
    for b in bs
        sigma, conds = _place(rec, b, at, plain)
        r, cs = _self_read(rec, b, sigma, tname, ASTExpr[VarExpr(j)])
        push!(terms, _term(b, sigma, vcat(conds, cs), r, lit(0)))
    end
    ranges = _frame_ranges(rec)
    push!(ranges, (j, rec.rows[d][1], rec.rows[d][2]))
    return _canon_lits(_faq([rec.frame; j], ranges, _plus_fold(terms, lit(0))))
end

# The transfer band of `rec`: rows `V` at the frame, column `V` at the frame
# with the axis replaced by a contracted `κ ≤ k`, coefficient `T[o, k, κ]`.
function _transfer_band(rec::Recurrence, tname::String)
    d = rec.dim
    kap = "$(rec.frame[d])_t"
    while kap in rec.frame
        kap *= "_t"
    end
    cidx = _vars(rec.frame)
    cidx[d] = VarExpr(kap)
    tidx = copy(cidx)
    tidx[d] = VarExpr(rec.frame[d])
    push!(tidx, VarExpr(kap))
    return Band(
        copy(rec.rows), Union{String, Int}[rec.frame...], cidx, _index(tname, tidx),
        Tuple{String, Int, Int}[(kap, rec.rows[d][1], rec.rows[d][2])],
        op("<=", VarExpr(kap), VarExpr(rec.frame[d]))
    )
end

function _binder_names!(e::ASTExpr, acc::Set{String})
    e isa OpExpr || return acc
    e.op == "faq" && union!(acc, _binders(e))
    map_children(x -> (_binder_names!(x, acc); x), e)
    return acc
end

function _band_names(b::Band)
    t = Set{String}(r for r in b.ridx if r isa String)
    union!(t, first.(b.contracted))
    b.coef === nothing || _all_names!(t, b.coef)
    foreach(c -> _all_names!(t, c), b.cidx)
    b.guard === nothing || _all_names!(t, b.guard)
    return t
end

# The chain rule on two bands: `outer` (rows `R`, column `V` at `φ`) followed
# by `inner` (rows `V`, column `x`). Each inner row name becomes the outer
# column index it is read at; the inner contracted names are renamed apart
# (`_c` appended) and become contracted names of the composite. The inner row
# box and `V`'s extent restrict the composite exactly. `nothing` when the
# composite is empty.
function _compose(outer::Band, inner::Band, shape_v::Vector{Int})
    onames = _band_names(outer)
    taken = union(onames, _band_names(inner))
    sigma = Dict{String, ASTExpr}()
    extra = Tuple{String, Int, Int}[]
    for (n, lo, hi) in inner.contracted
        nm = n
        if n in onames
            nm = _fresh!(taken, "$(n)_c", "_c")
            sigma[n] = VarExpr(nm)
        end
        push!(extra, (nm, lo, hi))
    end
    rows = copy(outer.rows)
    ctr = copy(outer.contracted)
    conds = ASTExpr[]
    function restrict(e, lo, hi)
        if e isa VarExpr
            for (k, r) in enumerate(outer.ridx)
                if r isa String && r == e.name
                    rows[k] = (max(rows[k][1], lo), min(rows[k][2], hi))
                    return
                end
            end
            for (k, (n, clo, chi)) in enumerate(ctr)
                if n == e.name
                    ctr[k] = (n, max(clo, lo), min(chi, hi))
                    return
                end
            end
        end
        if lo == hi
            push!(conds, op("==", e, lit(lo)))
        else
            push!(conds, op("<=", lit(lo), e))
            push!(conds, op("<=", e, lit(hi)))
        end
        return
    end
    for (m, e) in enumerate(outer.cidx)
        n = m <= length(shape_v) ? shape_v[m] : 1
        r = inner.ridx[m]
        lo, hi = r isa String ? inner.rows[m] : (r, r)
        lo, hi = max(lo, 1), min(hi, n)
        _in_range(e, outer, lo, hi) || restrict(e, lo, hi)
        r isa String && (sigma[r] = e)
    end
    (any(((lo, hi),) -> hi < lo, rows) || any(((_, lo, hi),) -> hi < lo, ctr)) && return nothing
    bound = Set{String}()
    _binder_names!(inner.coef, bound)
    foreach(c -> _binder_names!(c, bound), inner.cidx)
    inner.guard === nothing || _binder_names!(inner.guard, bound)
    read = Set{String}()
    for v in values(sigma)
        union!(read, free_variables(v))
    end
    hit = sort!(collect(intersect(bound, read)))
    isempty(hit) || throw(
        BandError(
            "composing through a recurrence would capture `$(hit[1])` under a nested sum"
        )
    )
    guard = _conj(
        _conj(outer.guard, _and_all(conds)),
        inner.guard === nothing ? nothing : _subst_free(inner.guard, sigma)
    )
    b = Band(
        rows, copy(outer.ridx), ASTExpr[_subst_free(c, sigma) for c in inner.cidx],
        _simp(mul(outer.coef, _subst_free(inner.coef, sigma))), vcat(ctr, extra), guard
    )
    return _tidy(b)
end

# The span of index expression `e` over `b`'s box (its row and contracted
# names): the interval bounds of the `+ - *` core (`_index_bounds`, the
# bounds §9.1's extent decision uses); `nothing` outside it or when `e` reads
# another name. The bounds always enclose `e`'s values, so a comparison they
# decide holds (or fails) at every point.
function _span(e::ASTExpr, b::Band)
    r = _index_bounds(e, _index_box(b))
    return r === nothing ? nothing : (r[1], r[2])
end

function _in_range(e::ASTExpr, b::Band, lo::Int, hi::Int)
    s = _span(e, b)
    return s !== nothing && lo <= s[1] && s[2] <= hi
end

const _INF = 1 << 62

# Decide what the box decides of a composite's `where` guard: a conjunct that
# holds at every point is dropped, one that holds at none empties the band,
# and a bound on a bare name tightens that name's range instead. Repeated to
# a fixpoint; the remaining conjuncts are kept once each, in order.
function _tidy(b0::Band)
    rows = copy(b0.rows)
    ctr = copy(b0.contracted)
    cur() = Band(rows, b0.ridx, b0.cidx, b0.coef, ctr, nothing)
    function tighten(nm, lo, hi)
        for (k, r) in enumerate(b0.ridx)
            if r isa String && r == nm
                rows[k] = (max(rows[k][1], lo), min(rows[k][2], hi))
                return true
            end
        end
        for (k, (n, clo, chi)) in enumerate(ctr)
            if n == nm
                ctr[k] = (n, max(clo, lo), min(chi, hi))
                return true
            end
        end
        return false
    end
    guard = b0.guard
    while guard !== nothing
        g = guard
        guard = nothing
        keep = ASTExpr[]
        seen = Set{String}()
        changed = false
        for c0 in _conjuncts(g)
            c = _canon_lits(c0)
            if c isa IntExpr
                c.value == 0 && return nothing
                changed = true
                continue
            end
            if c isa OpExpr && length(c.args) == 2 && c.op in ("<=", "<", ">=", ">", "==")
                a, z, off = c.op in ("<=", "==") ? (c.args[1], c.args[2], 0) :
                    c.op == "<" ? (c.args[1], c.args[2], 1) :
                    c.op == ">=" ? (c.args[2], c.args[1], 0) : (c.args[2], c.args[1], 1)
                sa, sz = _span(a, cur()), _span(z, cur())
                if sa !== nothing && sz !== nothing
                    if c.op == "=="
                        if sa[1] == sa[2] && sz[1] == sz[2]
                            sa[1] != sz[1] && return nothing
                            changed = true
                            continue
                        end
                        pin = a isa VarExpr && z isa IntExpr ? (a.name, Int(z.value)) :
                            a isa IntExpr && z isa VarExpr ? (z.name, Int(a.value)) : nothing
                        if pin !== nothing && tighten(pin[1], pin[2], pin[2])
                            changed = true
                            continue
                        end
                    else
                        if sa[2] <= sz[1] - off
                            changed = true
                            continue
                        end
                        sa[1] > sz[2] - off && return nothing
                        bnd = a isa VarExpr && z isa IntExpr ? (a.name, -_INF, Int(z.value) - off) :
                            a isa IntExpr && z isa VarExpr ? (z.name, Int(a.value) + off, _INF) : nothing
                        if bnd !== nothing && tighten(bnd...)
                            changed = true
                            continue
                        end
                    end
                end
            end
            k = skey(c)
            if k in seen
                changed = true
            else
                push!(seen, k)
                push!(keep, c)
            end
        end
        (any(((lo, hi),) -> hi < lo, rows) || any(((_, lo, hi),) -> hi < lo, ctr)) &&
            return nothing
        guard = _and_all(keep)
        changed || break
    end
    return normalize_band(Band(rows, b0.ridx, b0.cidx, b0.coef, ctr, guard))
end

"""
    materialize(sv::SysView, ext::ExtendedBands; eqs = nothing) -> Vector{JacEntry}

The materialized band list of an extended system (esm-jacobian-spec.md §4.5):
every state band whose column is a target, plus, for each state band reading
a recurrence `V`, its composition with `dV/dx`. Per row equation (two
equations of one state are never merged) the entries are stably sorted by
column name, a column's direct bands before its composites. With `eqs` (an
`Int` vector) the row equation of each entry is appended to it.
"""
function materialize(sv::SysView, ext; eqs = nothing)
    ctx = _ctx(sv)
    memo = Dict{String, Vector{JacEntry}}()
    out = JacEntry[]
    group = Tuple{String, Int, JacEntry}[]   # (column, 0 direct | 1 composite, entry)
    cur = Ref(0)                             # the row equation of `group`
    function flush!()
        sort!(group; by = g -> (g[1], g[2]), alg = MergeSort)
        append!(out, last.(group))
        eqs === nothing || append!(eqs, fill(cur[], length(group)))
        empty!(group)
        return
    end
    for (k, en) in enumerate(ext.rows)
        k > 1 && ext.row_eq[k - 1] != ext.row_eq[k] && flush!()
        cur[] = ext.row_eq[k]
        rec = _recurrence(ext, en.v)
        if rec === nothing
            push!(group, (en.v, 0, en))
            continue
        end
        shape = get(ctx.shapes, rec.name, Int[])
        for dd in _total(rec, ext, ctx, memo)
            b = _compose(en.band, dd.band, shape)
            b === nothing || push!(group, (dd.v, 1, JacEntry(en.u, dd.v, b)))
        end
    end
    flush!()
    return out
end

# `dV/dx` of `rec` as bands with rows V and target columns: the transfer band
# composed with each band of V's definition on a target, and with each band
# on another recurrence W composed with `dW/dx` first.
function _total(rec::Recurrence, ext, ctx::Ctx, memo)
    haskey(memo, rec.name) && return memo[rec.name]
    tb = _transfer_band(rec, ext.transfer[rec.name])
    shape = get(ctx.shapes, rec.name, Int[])
    out = JacEntry[]
    for en in ext.rec_rows
        (en.u != rec.name || en.v == rec.name) && continue
        w = _recurrence(ext, en.v)
        direct = if w === nothing
            JacEntry[en]
        else
            sw = get(ctx.shapes, w.name, Int[])
            ds = JacEntry[]
            for dd in _total(w, ext, ctx, memo)
                b = _compose(en.band, dd.band, sw)
                b === nothing || push!(ds, JacEntry(en.u, dd.v, b))
            end
            ds
        end
        for dd in direct
            b = _compose(tb, dd.band, shape)
            b === nothing || push!(out, JacEntry(rec.name, dd.v, b))
        end
    end
    memo[rec.name] = out
    return out
end

# The `auxiliary` definitions a band list reads (esm-jacobian-spec.md §4.5):
# for each recurrence whose transfer variable some coefficient reads,
# `(name, shape, definition)` — the shape is the recurrence's followed by its
# axis once more. In recurrence order.
function _auxiliary(sv::SysView, entries)
    obs = _observed_defs(sv)
    ctx = _ctx(sv)
    out = Tuple{String, Vector{String}, ASTExpr}[]
    recs = recurrences(sv, obs, ctx)
    isempty(recs) && return out
    names = _transfer_names(sv, recs)
    read = Set{String}()
    for en in entries
        en.band.coef === nothing || union!(read, free_variables(en.band.coef))
    end
    for rec in recs
        t = names[rec.name]
        t in read || continue
        push!(
            out, (t, [rec.shape; rec.shape[rec.dim]], transfer_definition(rec, obs, recs, ctx, t))
        )
    end
    return out
end
