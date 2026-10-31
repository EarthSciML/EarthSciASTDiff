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
                    if var.type == EarthSciAST.ParameterVariable &&
                    var.shape !== nothing && !isempty(var.shape) &&
                    any(u -> u.kind == "data", something(var.update, ()))
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
    jacobian_bands(file::EsmFile, model_name; wrt = :states, forcing = nothing) -> Vector{JacEntry}
    jacobian_bands(flat::FlattenedSystem;    wrt = :states, forcing = nothing) -> Vector{JacEntry}
    jacobian_bands(sv::SysView;              wrt = :states, forcing = nothing) -> Vector{JacEntry}

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
"""
jacobian_bands(file::EarthSciAST.EsmFile, model_name; kw...) =
    jacobian_bands(sysview(file, model_name); kw...)
jacobian_bands(flat::EarthSciAST.FlattenedSystem; kw...) =
    jacobian_bands(sysview(flat); kw...)

function jacobian_bands(sv::SysView; wrt::Symbol = :states, forcing = nothing)
    _check_axis(wrt)
    return _jacobian_bands(sv, wrt, forcing, String[])
end

_check_axis(wrt::Symbol) = wrt in (:states, :parameters, :forcing, :time) || throw(
    ArgumentError("wrt must be :states, :parameters, :forcing or :time, got $wrt")
)

# The first-order passes of one emission (`jacobian_document`), and the
# structure class of each. A pass depends only on its axis and the inlining
# seed a second axis adds (`_bands2`); that seed changes nothing on the time
# axis (never seeded) or when the second axis is the first, so those share
# the plain pass's key: a document asking for `:states` and
# `(:states, :states)` differentiates the system once.
struct _FirstOrder
    bands::Dict{Tuple{Symbol, Union{Nothing, Symbol}}, Vector{JacEntry}}
    structure::IdDict{Vector{JacEntry}, Symbol}
end
_FirstOrder() = _FirstOrder(Dict{Tuple{Symbol, Union{Nothing, Symbol}}, Vector{JacEntry}}(), IdDict{Vector{JacEntry}, Symbol}())

function _first_order!(memo::_FirstOrder, sv::SysView, a1::Symbol, a2, forcing, seed = String[])
    key = (a1, a1 == :time || a2 === a1 ? nothing : a2)
    return get!(() -> _jacobian_bands(sv, a1, forcing, seed), memo.bands, key)
end

_structure!(memo::_FirstOrder, e1::Vector{JacEntry}, sv::SysView) =
    get!(() -> detect_structure(e1, sv), memo.structure, e1)

# `seed` adds names to the inlining seed without making them targets: the
# second-order driver (second_order.jl) differentiates these coefficients
# again with respect to its second axis, so an observed that moves only with
# that axis must be inlined too, or its sites stay hidden behind its name.
function _jacobian_bands(sv::SysView, wrt::Symbol, forcing, seed::Vector{String})
    ctx = _ctx(sv)
    obs = _observed_defs(sv)
    targets = _targets(sv, wrt, forcing, obs)
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
    entries = JacEntry[]
    # The structural occurrence gate, driven from the EQUATION rather than
    # from the target list: scanning `targets` per equation is O(|equations| ×
    # |targets|) even when each RHS names a handful of them. `targets` is one
    # entry per state VARIABLE, so this is invisible on a document whose
    # states stay array-valued (ReSEACT: 13 either way) and quadratic on one
    # lowered to scalar cells, where a variable is a cell — a cost driven by
    # the flattening style rather than by how coupled the system is.
    # Intersecting the RHS's own free variables with the target SET is
    # O(|free variables|). Order is unchanged: `targets` is sorted by name, so
    # re-sorting the hits by name reproduces the same per-equation order, and
    # the `entries` order is what downstream pattern/scatter construction
    # (assemble.jl) reads.
    tset = Set{String}(targets)
    obs_names = Ref{Union{Nothing, Set{String}}}(nothing)
    for eq in sv.equations
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
        shape_u = get(ctx.shapes, u, Int[])
        rhs = inline_observed(eq.rhs, obs; obs_names = obs_names)
        hits = sort!(String[v for v in free_variables(rhs) if v in tset])
        for v in hits
            for b in merge_bands(bands(rhs, v, ctx; shape_u = shape_u))
                push!(entries, JacEntry(u, v, normalize_band(b)))
            end
        end
    end
    wrt in (:parameters, :forcing) && foreach(_check_closed_columns, entries)
    return entries
end

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
