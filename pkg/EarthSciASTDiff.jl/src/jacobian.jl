# Model-level driver: differentiate every `D(u) = rhs` equation w.r.t. every
# state (or parameter) that structurally occurs in its RHS.

"""
    JacEntry

One band of the model Jacobian: `∂(equation for state `u`)/∂`v`` with its
[`Band`](@ref).
"""
struct JacEntry
    u::String
    v::String
    band::Band
end

"""
    forcing_names(sv::SysView, forcing = nothing) -> Vector{String}

The forcing set of a system, sorted by name: the parameters whose cells are
the columns of the `wrt = :forcing` axis (esm-jacobian-spec.md §4.3). With
`forcing = nothing` the set is inferred — every ARRAY-valued parameter with an
`update` of kind `"data"` (a live buffer a data provider refills). An explicit
list is what a planner that builds the providers itself passes (a document
need not declare the update), and every name in it must be a parameter of the
system: a name that resolves to no buffer throws, never yields an empty axis.
`forcing` may be one name (a `String` or `Symbol`) or a collection of them.
"""
function forcing_names(sv::SysView, forcing = nothing)::Vector{String}
    if forcing === nothing
        return sort!(
            String[
                String(n) for (n, var) in sv.variables
                    if var.shape !== nothing && !isempty(var.shape) &&
                    !isempty(_data_rules(var))
            ]
        )
    end
    # One buffer name is the natural call; iterating a String would split it
    # into characters.
    names = forcing isa Union{AbstractString, Symbol} ? [String(forcing)] :
        String[String(n) for n in forcing]
    for n in names
        var = get(sv.variables, n, nothing)
        (var !== nothing && var.type == EarthSciAST.ParameterVariable) || throw(
            ArgumentError(
                "forcing name `$n` resolves to no buffer: it is not a parameter " *
                    "of the system"
            )
        )
    end
    return sort!(unique!(names))
end

# The column variables of one axis, in canonical (name-sorted) order: every
# parameter outside the forcing set (`:parameters`), the forcing set
# (`:forcing`), the solver-slot states (`:states`) or the single site `t`.
function _targets(sv::SysView, wrt::Symbol, forcing, obs)::Vector{String}
    wrt == :time && return ["t"]
    wrt == :states && return view_states(sv, obs)
    fset = forcing_names(sv, forcing)
    wrt == :forcing && return fset
    return sort!(
        String[
            String(n) for (n, var) in sv.variables
                if var.type == EarthSciAST.ParameterVariable && !(String(n) in fset)
        ]
    )
end

"""
    jacobian_columns(sv::SysView; wrt, forcing = nothing) -> Vector{Tuple{String,Vector{Int}}}

The column space of a `:parameters` or `:forcing` block: each column variable
with its integer shape (`Int[]` for a scalar), sorted by name — the `columns`
field of the emitted block (esm-jacobian-spec.md §4). The canonical column
ordinal enumerates these variables in order, each one's cells column-major
(first index fastest).
"""
function jacobian_columns(sv::SysView; wrt::Symbol, forcing = nothing)
    wrt in (:parameters, :forcing) ||
        throw(ArgumentError("columns are defined for wrt = :parameters or :forcing, got $wrt"))
    ctx = _ctx(sv)
    return Tuple{String, Vector{Int}}[
        (n, get(ctx.shapes, n, Int[])) for n in _targets(sv, wrt, forcing, nothing)
    ]
end

"""
    jacobian_bands(file::EsmFile, model_name; wrt = :states, forcing = nothing, coefficients = true) -> Vector{JacEntry}
    jacobian_bands(flat::FlattenedSystem;    wrt = :states, forcing = nothing, coefficients = true) -> Vector{JacEntry}
    jacobian_bands(sv::SysView;              wrt = :states, forcing = nothing, coefficients = true) -> Vector{JacEntry}

Differentiate every differential equation of the system with respect to every
state (`wrt = :states`), every parameter outside the forcing set
(`wrt = :parameters`, scalar and array-valued alike), every forcing buffer
(`wrt = :forcing`, the cells of the buffers [`forcing_names`](@ref) selects,
their record axis included) that structurally occurs in its (observed-inlined)
RHS — or with respect to time itself (`wrt = :time`, the single scalar site
`t`; the integer-valued `datetime.*` calendar fields are piecewise-constant
and differentiate to 0, the continuous `datetime.julian_day` to `d[t]/86400`,
and `interp.linear`/`interp.bilinear` of `t` to the active-segment slope).

`forcing` names the forcing set explicitly; `nothing` infers it from the
document (every array parameter with a `data` update). It also decides which
parameters the `:parameters` axis leaves to the forcing axis.

The pattern implied by the result is GLOBAL: `ifelse`/`min`/`max` branches
are unioned, so it is valid at every point and is a superset of any local
(point-wise) pattern.

On the `:parameters` and `:forcing` axes every column index must be closed at
build time (literals, folded metaparameters, `const` gathers, and the entry's
row and contracted index names); an unclosed one throws
(esm-jacobian-spec.md §4.3). A model whose right-hand sides read the forcing
set only where the derivative is identically zero (an `ifelse` condition, a
piecewise-constant op) has an empty forcing list.

`coefficients = false` returns STRUCTURE-ONLY entries (esm-jacobian-spec.md
§2.1): regions are clipped and sites found as in the full mode, but no
derivative is taken, no coefficient simplified, and bands merge on
(`ridx`, `cidx`, `contracted`, `guard`) alone. Every `band.coef` is
`nothing`. Their points include the full mode's, and equal them unless the
full mode drops a band whose coefficient simplifies to zero. Use them where
only the pattern is needed ([`jacobian_pattern`](@ref),
[`detect_structure`](@ref)); a document or product needs the full mode.
Through an active recurrence the pass runs in full and the coefficients are
dropped.

An observed that reads itself at an earlier cell (a causal recurrence,
esm-jacobian-spec.md §4.5, [`recurrences`](@ref)) is differentiated as an
intermediate ([`extended_bands`](@ref)), and the chain rule through it is
composed with its line's transfer recurrence `T = (I − G)⁻¹`
([`transfer_definition`](@ref)): the coefficients read `T`, which the emitted
block carries as `auxiliary` and the evaluation document declares.
"""
jacobian_bands(file::EarthSciAST.EsmFile, model_name; kw...) =
    jacobian_bands(sysview(file, model_name); kw...)
jacobian_bands(flat::EarthSciAST.FlattenedSystem; kw...) =
    jacobian_bands(sysview(flat); kw...)

function jacobian_bands(
        sv::SysView; wrt::Symbol = :states, forcing = nothing, coefficients::Bool = true
    )
    _check_axis(wrt)
    return _jacobian_bands(sv, wrt, forcing, String[]; coefficients = coefficients)
end

_check_axis(wrt::Symbol) = wrt in (:states, :parameters, :forcing, :time) || throw(
    ArgumentError("wrt must be :states, :parameters, :forcing or :time, got $wrt")
)

"""
    ExtendedBands

The extended first-order system of one pass (esm-jacobian-spec.md §4.5): the
active `recurrences`, the bands of the state equations (`rows`; columns: the
axis's targets and the active recurrences) and the bands of the recurrences'
own definitions (`rec_rows`; rows: a recurrence). `row_eq[k]` is the
equation `rows[k]` came from (two equations may share a state), and
`transfer` the transfer variable name of EVERY recurrence of the view, so a
name never depends on which recurrences are active. Without an active
recurrence `rows` is [`jacobian_bands`](@ref)'s list and the rest is empty.
"""
struct ExtendedBands
    recurrences::Vector{Recurrence}
    rows::Vector{JacEntry}
    rec_rows::Vector{JacEntry}
    row_eq::Vector{Int}
    transfer::Dict{String, String}
end

_recurrence(ext::ExtendedBands, v::String) =
    (k = findfirst(r -> r.name == v, ext.recurrences); k === nothing ? nothing : ext.recurrences[k])
_is_recurrence(ext::ExtendedBands, v::String) = any(r -> r.name == v, ext.recurrences)

# The first-order passes of one emission (`jacobian_document`), and the
# structure class of each. A pass depends only on its axis and the inlining
# seed a second axis adds (`_bands2`); that seed changes nothing when the
# second axis is the first, so such a pair shares the plain pass's key: a
# document asking for `:states` and `(:states, :states)` differentiates the
# system once. On the time axis the seed changes no inlining, so every time
# pair shares the plain time pass too — unless the view has a recurrence,
# which the seed can make active (§4.5).
struct _FirstOrder
    ext::Dict{Tuple{Symbol, Union{Nothing, Symbol}}, ExtendedBands}
    bands::Dict{Tuple{Symbol, Union{Nothing, Symbol}}, Vector{JacEntry}}
    structure::IdDict{Vector{JacEntry}, Symbol}
    has_recs::Base.RefValue{Union{Nothing, Bool}}
end
_FirstOrder() = _FirstOrder(
    Dict{Tuple{Symbol, Union{Nothing, Symbol}}, ExtendedBands}(),
    Dict{Tuple{Symbol, Union{Nothing, Symbol}}, Vector{JacEntry}}(),
    IdDict{Vector{JacEntry}, Symbol}(), Ref{Union{Nothing, Bool}}(nothing)
)

function _memo_key(memo::_FirstOrder, sv::SysView, a1::Symbol, a2)
    a2 === a1 && return (a1, nothing)
    if a1 == :time
        memo.has_recs[] === nothing && (memo.has_recs[] = !isempty(recurrences(sv)))
        memo.has_recs[] || return (a1, nothing)
    end
    return (a1, a2)
end

# The extended `a1` pass seeded with `seed`.
function _first_order_ext!(memo::_FirstOrder, sv::SysView, a1::Symbol, a2, forcing, seed = String[])
    return get!(memo.ext, _memo_key(memo, sv, a1, a2)) do
        _extended_pass(sv, _ctx(sv), _observed_defs(sv), a1, forcing, seed)
    end
end

# The materialized `a1` pass (the extended one composed through its
# recurrences, §4.5).
function _first_order!(memo::_FirstOrder, sv::SysView, a1::Symbol, a2, forcing, seed = String[])
    return get!(memo.bands, _memo_key(memo, sv, a1, a2)) do
        ext = _first_order_ext!(memo, sv, a1, a2, forcing, seed)
        isempty(ext.recurrences) ? ext.rows : _intern_entries(materialize(sv, ext))
    end
end

_structure!(memo::_FirstOrder, e1::Vector{JacEntry}, sv::SysView) =
    get!(() -> detect_structure(e1, sv), memo.structure, e1)

# `seed` adds names to the inlining seed without making them targets: the
# second-order driver (second_order.jl) differentiates these coefficients
# again with respect to its second axis, so an observed that moves only with
# that axis must be inlined too, or its sites stay hidden behind its name.
# Through an active recurrence the extended system is materialized (§4.5).
function _jacobian_bands(
        sv::SysView, wrt::Symbol, forcing, seed::Vector{String}; coefficients::Bool = true
    )
    ext = _extended_pass(
        sv, _ctx(sv), _observed_defs(sv), wrt, forcing, seed; coefficients = coefficients
    )
    isempty(ext.recurrences) && return ext.rows
    coefficients && return _intern_entries(materialize(sv, ext))
    eqs = Int[]
    entries = materialize(sv, ext; eqs = eqs)
    return _structure_only(entries, eqs)
end

# Full entries (the pass through an active recurrence runs in full) as
# structure-only ones (spec §2.1): without `coef`, merged again on structure
# alone (item 2) — two entries the full mode kept apart for their
# coefficients alone coalesce — and pinned (item 3). A merge group is one
# column of one row EQUATION (`eqs[k]`, the equation of entry k), as without
# a recurrence: two equations of one state are never merged, even where
# one's run of a column meets the next one's.
function _structure_only(entries::Vector{JacEntry}, eqs::Vector{Int})
    out = JacEntry[]
    i = 1
    while i <= length(entries)
        j = i
        while j <= length(entries) && (eqs[j], entries[j].v) == (eqs[i], entries[i].v)
            j += 1
        end
        bs = merge_bands(Band[_rebuild(en.band; coef = nothing) for en in entries[i:(j - 1)]])
        for b in bs
            push!(out, JacEntry(entries[i].u, entries[i].v, normalize_band(b)))
        end
        i = j
    end
    return out
end

"""
    extended_bands(file::EsmFile, model_name; wrt = :states, forcing = nothing, second = nothing) -> ExtendedBands
    extended_bands(flat::FlattenedSystem;    wrt = :states, forcing = nothing, second = nothing) -> ExtendedBands
    extended_bands(sv::SysView;              wrt = :states, forcing = nothing, second = nothing) -> ExtendedBands

The extended first-order system of axis `wrt` (esm-jacobian-spec.md §4.5,
[`ExtendedBands`](@ref)): every state equation's bands with respect to the
axis's targets and the active recurrences, and each active recurrence
definition's own bands. With `second` (an axis) the pass is a pair's first
pass: seeded with the second axis's targets, so the recurrences that move
with either axis are active and the observed that move with either are
inlined (§4.4). Without a recurrence `rows` is [`jacobian_bands`](@ref)'s
list and the rest is empty.
"""
extended_bands(file::EarthSciAST.EsmFile, model_name; kw...) =
    extended_bands(sysview(file, model_name); kw...)
extended_bands(flat::EarthSciAST.FlattenedSystem; kw...) =
    extended_bands(sysview(flat); kw...)
function extended_bands(sv::SysView; wrt::Symbol = :states, forcing = nothing, second = nothing)
    _check_axis(wrt)
    obs = _observed_defs(sv)
    seed = second === nothing || second == wrt ? String[] :
        _targets(sv, Symbol(second), forcing, obs)
    return _extended_pass(sv, _ctx(sv), obs, wrt, forcing, seed)
end

# The extended first-order system of one pass (§4.5): the bands of every
# state equation with respect to the axis's targets AND the active
# recurrences, and the bands of each active recurrence's own definition (rows
# V). A recurrence is active when it moves with the targets or `seed` (a
# pair's second axis); recurrences are never inlined (no fixpoint exists), so
# an inactive one stays an opaque read, constant with respect to every target.
function _extended_pass(
        sv::SysView, ctx::Ctx, obs, wrt::Symbol, forcing, seed;
        coefficients::Bool = true
    )
    src_obs = obs
    targets = _targets(sv, wrt, forcing, obs)
    tset = Set{String}(targets)
    recs = recurrences(sv, obs, ctx)
    act = isempty(recs) ? Recurrence[] : _active(recs, obs, union(tset, seed))
    # Structure-only (spec §2.1) runs on a placeholder coefficient, except
    # through an active recurrence: its composition through the transfer
    # (§4.5) reads the coefficients, so that pass runs in full and only the
    # caller drops them.
    co = coefficients || !isempty(act)
    # Only the observed that can actually move with a target are inlined; the
    # rest have an identically-zero derivative and stay opaque reads that the
    # derived evaluation model resolves from their own definitions. Exact, and
    # seeded from THIS `wrt`'s targets — see `inlinable_observed`
    # (system.jl) for why the state and parameter axes must not share a seed.
    # `wrt = :time` is excluded: time also enters implicitly, through
    # externally-provided time-varying data, so `{"t"}` is not a sound
    # over-approximation of the set of time-dependent names.
    wrt == :time ||
        (obs = inlinable_observed(obs, sv.equations, Set{String}(vcat(targets, seed))))
    if !isempty(recs)
        rn = Set{String}(r.name for r in recs)
        obs = Dict{String, ASTExpr}(k => v for (k, v) in obs if !(k in rn))
    end
    cols = union(tset, Set{String}(r.name for r in act))
    entries = JacEntry[]
    row_eq = Int[]
    obs_names = Ref{Union{Nothing, Set{String}}}(nothing)
    for (q, eq) in enumerate(sv.equations)
        u = lhs_state(eq)
        if u === nothing
            _is_derivative(eq) && throw(
                BandError(
                    "unsupported derivative LHS $(JSON3.write(EarthSciAST.serialize_expression(eq.lhs))) " *
                        "(a state's derivative is `D(u)` or the lifted `faq{D(index(u, i…))}`)"
                )
            )
            continue
        end
        _row_bands!(entries, u, eq.rhs, obs, obs_names, cols, ctx, co)
        append!(row_eq, fill(q, length(entries) - length(row_eq)))
    end
    rec_rows = JacEntry[]
    for r in act
        _row_bands!(rec_rows, r.name, src_obs[r.name], obs, obs_names, cols, ctx, co)
    end
    for en in rec_rows
        # every self band is a lag back along the axis, or refused
        en.u == en.v && _lag(act[findfirst(r -> r.name == en.u, act)], en.band)
    end
    if wrt in (:parameters, :forcing)
        for en in Iterators.flatten((entries, rec_rows))
            en.v in tset && _check_closed_columns(en)
        end
    end
    if co
        # One object per distinct subtree across every coefficient (hashcons.jl
        # `_Interner`): the entries reference shared copies, not their own.
        t = _Interner()
        entries = _intern_entries(entries, t)
        rec_rows = _intern_entries(rec_rows, t)
    end
    isempty(act) &&
        return ExtendedBands(Recurrence[], entries, JacEntry[], row_eq, Dict{String, String}())
    return ExtendedBands(act, entries, rec_rows, row_eq, _transfer_names(sv, recs))
end

# The bands of one row variable `u` (a state, or a recurrence over its own
# definition) with respect to every column in `cols` its inlined right-hand
# side reads, in sorted column order. The structural occurrence gate is
# driven from the EQUATION rather than from the target list: scanning
# `targets` per equation is O(|equations| × |targets|) even when each RHS
# names a handful of them, which is quadratic on a document lowered to scalar
# cells. Intersecting the RHS's own free variables with the column SET is
# O(|free variables|), and re-sorting the hits by name reproduces the target
# order the downstream pattern/scatter construction (assemble.jl) reads.
function _row_bands!(out, u::String, rhs::ASTExpr, obs, obs_names, cols, ctx::Ctx, co::Bool)
    shape_u = get(ctx.shapes, u, Int[])
    rhs = inline_observed(rhs, obs; obs_names = obs_names, shapes = ctx.shapes)
    hits = sort!(String[v for v in free_variables(rhs) if v in cols])
    for v in hits
        for b in merge_bands(bands(rhs, v, ctx; shape_u = shape_u, coefficients = co))
            push!(out, JacEntry(u, v, normalize_band(b)))
        end
    end
    return out
end

_intern_entries(entries::Vector{JacEntry}, t::_Interner = _Interner()) =
    JacEntry[JacEntry(en.u, en.v, _intern_band(t, en.band)) for en in entries]

# `b` with its coefficient, guard and column indices interned.
_intern_band(t::_Interner, b::Band) = _rebuild(
    b;
    cidx = ASTExpr[_intern(t, c) for c in b.cidx],
    coef = b.coef === nothing ? nothing : _intern(t, b.coef),
    guard = b.guard === nothing ? nothing : _intern(t, b.guard),
)

# A column of the parameters / forcing axes is a cell of a buffer, so it must
# be placeable at build time: after inlining, its index expressions may read
# only the entry's own row and contracted index names (literals, folded
# metaparameters and `const` gathers carry no free name). A column reading
# anything else (a parameter offset, a state) cannot be placed, so the entry
# is refused rather than emitted. The second column of a pair
# (second_order.jl) is held to the same rule.
_check_closed_columns(en::JacEntry) = _check_closed(
    en.band, en.band.cidx, c -> "column index $(_json(c)) of `$(en.v)` in the equation for `$(en.u)`"
)

function _check_closed(b::Band, cols, what)
    bound = Set{String}(r for r in b.ridx if r isa String)
    for (n, _, _) in b.contracted
        push!(bound, n)
    end
    for c in cols, n in free_variables(c)
        n in bound || throw(
            BandError(
                "$(what(c)) reads `$n`, which is not closed at build time (a parameter " *
                    "or forcing column may read only literals, folded metaparameters, " *
                    "const gathers and its row or contracted index names)"
            )
        )
    end
    return
end

_json(c::ASTExpr) = JSON3.write(EarthSciAST.serialize_expression(c))
