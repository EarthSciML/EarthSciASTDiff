# Numeric assembly: the structural pattern, a one-shot Jacobian, and a
# prepared evaluator with a fixed sparse prototype + precomputed scatter map.

_cellname(v::String, cell) = isempty(cell) ? v : "$(v)[$(join(cell, ","))]"

# Column-index evaluation. `evaluate_expr` covers the scalar core; a literal
# `index(const, …)` gather (a `conn[]` connectivity table) is folded here
# first, because the tree-walk's inline-const branch is a build-time resolve,
# not a runtime read.
function _fold_const_gathers(e::ASTExpr, bind::Dict{String, Float64})::ASTExpr
    e isa OpExpr || return e
    if e.op == "index" && e.args[1] isa OpExpr && (e.args[1]::OpExpr).op == "const"
        v = (e.args[1]::OpExpr).value
        for x in e.args[2:end]
            v = v[_eval_cidx(x, bind)]
        end
        return v isa Integer ? IntExpr(Int64(v)) : NumExpr(Float64(v))
    end
    return map_children(x -> _fold_const_gathers(x, bind), e)
end
_eval_cidx(c::ASTExpr, bind::Dict{String, Float64})::Int =
    Int(round(evaluate_expr(_fold_const_gathers(c, bind), bind)))

# Every point of one band: `f(cell, cc, cidx)` per (row cell, contracted cell),
# first index fastest, with the column index evaluated there. Column index
# expressions are evaluated per cell ONCE (build-time, not solve-time).
function _foreach_point(f, b::Band)
    rowcells = isempty(b.rows) ? [Int[]] :
        [collect(c) for c in Iterators.product(((lo:hi) for (lo, hi) in b.rows)...)]
    ccells = isempty(b.contracted) ? [Int[]] :
        [
            collect(c) for c in
            Iterators.product(((lo:hi) for (_, lo, hi) in b.contracted)...)
        ]
    for cell in rowcells, cc in ccells
        bind = Dict{String, Float64}(
            r => Float64(cell[k2]) for (k2, r) in enumerate(b.ridx) if r isa String
        )
        for (k2, (n, _, _)) in enumerate(b.contracted)
            bind[n] = Float64(cc[k2])
        end
        f(cell, cc, [_eval_cidx(c, bind) for c in b.cidx])
    end
    return nothing
end

# Enumerate every (band, row cell) → (row name, col name) pair.
function _scatter_pairs(sv::SysView, entries::Vector{JacEntry})
    pairs = Vector{Tuple{Int, String, String, String}}()  # (band k, rowname, colname, jname-cell)
    for (k, en) in enumerate(entries)
        _foreach_point(en.band) do cell, cc, cidx
            push!(
                pairs, (
                    k, _cellname(en.u, cell), _cellname(en.v, cidx),
                    _cellname("J_$k", vcat(cell, cc)),
                )
            )
        end
    end
    return pairs
end

# Column slot lookup: state columns index the state vector; time is the
# single column `t`; parameter and forcing columns index the axis's CANONICAL
# column space (esm-jacobian-spec.md §4: the column variables sorted by name,
# each one's cells column-major), built from the system's declared shapes, so
# an array parameter's cells are real columns rather than names the scalar
# `p` NamedTuple cannot hold.
function _colmap(vm0, cols, wrt::Symbol)
    if wrt == :time
        return name -> name == "t" ? 1 : 0, 1, ["t"]
    elseif wrt == :states
        return name -> get(vm0, name, 0), length(vm0),
            first.(sort(collect(vm0), by = last))
    else
        cnames = _column_names(cols)
        lut = Dict(n => i for (i, n) in enumerate(cnames))
        return name -> get(lut, name, 0), length(cnames), cnames
    end
end

# The canonical column names of a `jacobian_columns` space, in ordinal order.
function _column_names(cols)::Vector{String}
    names = String[]
    for (v, shape) in cols
        if isempty(shape)
            push!(names, v)
        else
            for c in Iterators.product((1:n for n in shape)...)
                push!(names, _cellname(v, collect(c)))
            end
        end
    end
    return names
end

# The column slot of one scatter pair. On the forcing axis a column is a cell
# of a live buffer, and the ghost-read skip that is legitimate for a stencil
# reading past the domain is not: an unresolvable column there would silently
# drop a gradient, so it is an error (esm-jacobian-spec.md §4). The parameters
# axis follows the state axis: an array parameter read past its extent (a
# face coefficient `K[i+1]` beside `c[i+1]`) is a ghost read.
function _colslot(colslot, colname::String, rowname::String, wrt::Symbol)
    cs = colslot(colname)
    cs == 0 && wrt == :forcing && throw(
        BandError(
            "column `$colname` (row `$rowname`) is outside the declared extent of its " *
                "variable; on the $wrt axis every column must resolve (no ghost skip)"
        )
    )
    return cs
end

# --------------------------------------------------------------------------- #
# The SOURCE-model layout: `(n, p0, vm0)`.
#
# Both `jacobian_pattern` and `prepare_jacobian` need three things about the
# source model -- its state count, its scalar-parameter NamedTuple and its
# name -> state-slot map -- and by default they get them by running a COMPLETE
# `build_evaluator` of the source and throwing the RHS away. On a large grid
# that build is half of `prepare_jacobian`'s wall time (measured 45% at 288
# cells, 51% at 648, and the share GROWS with the grid because the band build
# is grid-independent while the source build is not).
#
# A caller that has ALREADY built the source evaluator can hand the layout in
# and skip the rebuild. That is a sharp edge: a layout from a build with
# different `build_kwargs` (different `const_arrays`, `parameter_overrides`,
# metaparameters, ...) would put every band in the wrong cell with no visible
# symptom -- so the hand-in carries a contract (see `prepare_jacobian`), the
# cheap structural half of which is checked here and the name half at the two
# use sites below.
#
# `source_layout = (u0 = …, p = …, var_map = …)` is that hand-in: the 2nd, 3rd
# and 5th returns of `EarthSciAST.build_evaluator`. `nothing` (the default)
# rebuilds, which is exactly the historical behavior.
function _source_layout(src, model_name, build_kwargs, layout, wrt::Symbol)
    if layout === nothing
        _, u00, p0, _, vm0 = _build_eval(src, model_name; build_kwargs...)
        return (length(u00), p0, vm0)
    end
    for f in (:u0, :var_map)
        hasproperty(layout, f) || throw(
            ArgumentError(
                "source_layout must carry (u0, var_map[, p]); `$f` is missing"
            )
        )
    end
    u0 = getproperty(layout, :u0)
    vm0 = getproperty(layout, :var_map)
    p0 = hasproperty(layout, :p) ? getproperty(layout, :p) : nothing
    n = length(u0)
    length(vm0) == n || throw(
        ArgumentError(
            "source_layout: length(u0) = $n but length(var_map) = $(length(vm0)). " *
                "The layout must come from a `build_evaluator` of the same source with " *
                "the same build_kwargs."
        )
    )
    slots = sort!(collect(values(vm0)))
    slots == 1:n || throw(
        ArgumentError(
            "source_layout: var_map values are not a permutation of 1:$n, so it is " *
                "not a state layout of this model."
        )
    )
    return (n, p0, vm0)
end

# Name lookup with a legible failure: a `source_layout` from the wrong build
# shows up here first, as a name that resolves nowhere.
function _slot(map, name::String, what::String)
    j = get(map, name, 0)
    j == 0 && throw(
        ArgumentError(
            "`$name` is not in the $what model's variable map -- if `source_layout` " *
                "was passed, it does not match a `build_evaluator` of this source with " *
                "these build_kwargs."
        )
    )
    return j
end

# The ROW slot of one scatter pair. A row whose column fell outside the layout
# is never used, so an unresolvable name there is (historically) silent -- but
# with a caller-supplied layout it is the loudest evidence available that the
# layout is from the wrong build, and a layout whose names ALL miss would
# otherwise assemble a silently EMPTY pattern. So under `strict` every row name
# must resolve, column or no column.
function _rowslot(vm0, rowname::String, cs::Int, strict::Bool)
    r = get(vm0, rowname, 0)
    (r == 0 && (strict || cs != 0)) && _slot(vm0, rowname, "source")
    return r
end

"""
    jacobian_pattern(input; wrt = :states, forcing = nothing) -> (pattern, rownames, colnames, entries)

The structural (global, point-independent) sparsity pattern of the Jacobian
as a `SparseMatrixCSC{Bool}`. `input` is an `EsmFile` + model name pair
(pass a `Tuple`), an `EsmFile` with a single model, or a `FlattenedSystem`.
On the `:parameters` and `:forcing` axes the columns are the canonical column
space of [`jacobian_columns`](@ref).

`source_layout` skips this function's own rebuild of the source model — see
[`prepare_jacobian`](@ref) for the keyword and its contract.
"""
function jacobian_pattern(
        input; wrt::Symbol = :states, model_name = nothing, forcing = nothing,
        build_kwargs = NamedTuple(), source_layout = nothing
    )
    sv, src = _sv_src(input, model_name)
    entries = jacobian_bands(sv; wrt = wrt, forcing = forcing)
    _, _, vm0 = _source_layout(src, model_name, build_kwargs, source_layout, wrt)
    colslot, ncol, colnames = _colmap(vm0, _axis_columns(sv, wrt, forcing), wrt)
    n = length(vm0)
    strict = source_layout !== nothing
    I = Int[]; J = Int[]
    for (_, rowname, colname, _) in _scatter_pairs(sv, entries)
        cs = _colslot(colslot, colname, rowname, wrt)
        r = _rowslot(vm0, rowname, cs, strict)
        cs == 0 && continue                          # ghost / out-of-layout column
        push!(I, r); push!(J, cs)
    end
    pattern = sparse(I, J, trues(length(I)), n, ncol, |)
    return (
        pattern = pattern,
        rownames = first.(sort(collect(vm0), by = last)),
        colnames = colnames, entries = entries,
    )
end

_axis_columns(sv::SysView, wrt::Symbol, forcing) =
    wrt in (:parameters, :forcing) ? jacobian_columns(sv; wrt = wrt, forcing = forcing) :
    nothing

_sv_src(file::EarthSciAST.EsmFile, model_name) =
    (sysview(file, model_name === nothing ? _only_model(file) : model_name), file)
_sv_src(flat::EarthSciAST.FlattenedSystem, ::Any) = (sysview(flat), flat)

# Build the ordinary RHS evaluator for the SAME model `_sv_src` selected: an
# `EsmFile` threads `model_name` through `build_evaluator`'s own selection
# (so multi-model documents work), a `FlattenedSystem` has no model to name.
_build_eval(src::EarthSciAST.EsmFile, model_name; kw...) =
    build_evaluator(src; model_name = model_name === nothing ? nothing : String(model_name), kw...)
_build_eval(src, model_name; kw...) = build_evaluator(src; kw...)

function _only_model(file::EarthSciAST.EsmFile)
    length(file.models) == 1 ||
        throw(ArgumentError("document has $(length(file.models)) models; pass model_name"))
    return first(keys(file.models))
end

"""
    JacobianEvaluator

Prepared analytical Jacobian: call as `jac!(J, u, p, t)` where `J` is (a copy
of) `jac.prototype`. Fields of note: `prototype::SparseMatrixCSC{Float64}`,
`pattern::SparseMatrixCSC{Bool}`, `entries`, `colnames`, `structure`.
"""
struct JacobianEvaluator{F, P}
    fJ!::F                        # derived-model RHS (fills band fields into duj)
    oop::Bool                     # true when `fJ!` is `f(u, p, t) -> du`
    uj::Vector{Float64}           # derived-model state buffer
    duj::Vector{Float64}
    umap::Vector{Int}             # uj[umap[i]] = u[i]
    p_default::P
    scatter::Vector{Tuple{Int, Int}}  # (slot in duj, position in nzval)
    items::Vector{Tuple{Int, Int, Int}} # (slot in duj, row, col): the transposed scatter
    field_shapes::Vector{Vector{Int}} # band k → field shape (scatter plans)
    slots::Dict{String, Int}          # derived-model name → slot in duj
    prototype::SparseMatrixCSC{Float64, Int}
    pattern::SparseMatrixCSC{Bool, Int}
    entries::Vector{JacEntry}
    rownames::Vector{String}
    colnames::Vector{String}
    columns::Union{Nothing, Vector{Tuple{String, Vector{Int}}}} # parameters / forcing axes
    structure::Union{Nothing, Symbol} # `nothing` on the rectangular parameters / forcing axes
    wrt::Symbol
end

"""
    prepare_jacobian(input; wrt = :states, model_name = nothing, forcing = nothing) -> JacobianEvaluator

Build the analytical Jacobian evaluator once: differentiates the system,
compiles the derived band-field model through the ordinary
`EarthSciAST.build_evaluator`, and precomputes the (row, col) → nzval scatter
map on the fixed structural pattern. `build_kwargs` (a NamedTuple) is
forwarded to both `build_evaluator` calls — pass `(const_arrays = …,)` etc.
for models that need build-time bindings.

The second of those two builds exists only to learn the source model's state
layout — `length(u0)`, the parameter names and the name → slot map — and its
RHS is thrown away. On a real grid it is about HALF this call's wall time (a
share that grows with the grid, because the band build is grid-independent and
the source build is not). A caller that has already built the source evaluator
can delete that build by passing its layout:

    # u0, p, var_map: build_evaluator's 2nd, 3rd and 5th returns
    source_layout = (u0 = u0, p = p, var_map = var_map)

!!! warning "Contract"
    The layout MUST come from a `build_evaluator` of the SAME source (same
    document / `FlattenedSystem`, same `model_name`) with the SAME
    `build_kwargs`. Only the cheap parts of that are checked (the state count,
    that the slots are a permutation, that every name resolves in the derived
    band model); a layout from a build that merely *differs* — other
    `const_arrays`, other `parameter_overrides`, another grid — would put
    bands in the wrong cells with no visible symptom. `p` may be omitted:
    parameter and forcing columns come from the system's declared shapes.
    The default, `nothing`, rebuilds.

    jac = prepare_jacobian(file)
    J = copy(jac.prototype)
    jac(J, u, p, t)              # fills J in place; stored zeros stay

On the `:parameters` and `:forcing` axes the columns are the canonical column
space of [`jacobian_columns`](@ref) (`jac.columns`), and `jac.structure` is
`nothing` (the classes describe square matrices). A forcing column that does
not resolve to a cell throws; a parameter column outside its extent is a
ghost read, skipped as on the state axis. Forcing buffers are bound like any
other array parameter of the derived band model: through
`build_kwargs = (param_arrays = buffers,)` they are read live, by reference
(after writing one in place, call `EarthSciAST.notify_forcing_refresh!(buf)`
as for the RHS); without it the declared defaults are used. [`vjp!`](@ref) /
[`vjp_forcing!`](@ref) apply the transposed bands without forming `J`.
"""
function prepare_jacobian(
        input; wrt::Symbol = :states, model_name = nothing, forcing = nothing,
        build_kwargs = NamedTuple(), source_layout = nothing,
        eval_cse::Bool = true, eval_cse_min_nodes::Int = 12
    )
    sv, src = _sv_src(input, model_name)
    entries = jacobian_bands(sv; wrt = wrt, forcing = forcing)
    doc, _ = _evaluation_document(
        sv, entries; cse = eval_cse,
        cse_min_nodes = eval_cse_min_nodes
    )
    fJ!, u0j, pj, _, vmj = build_evaluator(
        doc; model_name = "JacobianEval",
        build_kwargs...
    )
    n, _, vm0 = _source_layout(src, model_name, build_kwargs, source_layout, wrt)
    columns = _axis_columns(sv, wrt, forcing)
    colslot, ncol, colnames = _colmap(vm0, columns, wrt)

    umap = zeros(Int, n)
    for (name, i) in vm0
        umap[i] = _slot(vmj, name, "band")
    end

    strict = source_layout !== nothing
    items = Tuple{Int, Int, Int}[]     # (duj slot, row, col)
    for (k, rowname, colname, jname) in _scatter_pairs(sv, entries)
        cs = _colslot(colslot, colname, rowname, wrt)
        r = _rowslot(vm0, rowname, cs, strict)
        cs == 0 && continue
        push!(items, (_slot(vmj, jname, "band"), r, cs))
    end
    I = [it[2] for it in items]; J = [it[3] for it in items]
    pattern = sparse(I, J, trues(length(I)), n, ncol, |)
    proto = SparseMatrixCSC(
        n, ncol, copy(pattern.colptr), copy(pattern.rowval),
        zeros(length(pattern.rowval))
    )
    # nzval position of each (row, col): binary search the fixed pattern.
    scatter = Tuple{Int, Int}[]
    for (slot, r, c) in items
        rng = proto.colptr[c]:(proto.colptr[c + 1] - 1)
        pos = rng[searchsortedfirst(view(proto.rowval, rng), r)]
        push!(scatter, (slot, pos))
    end
    # The structure classes describe square matrices; a parameters or forcing
    # Jacobian is rectangular, so it has none (as its emitted block omits it).
    structure = columns === nothing ? detect_structure(entries, sv) : nothing
    # Field shapes and the derived name → slot map, for scatter plans
    # (scatter.jl): `bind_plan` resolves only the field cells a plan reads.
    ctx = _ctx(sv)
    field_shapes = [_field_shape(en, ctx) for en in entries]
    # WHICH FORM did `build_evaluator` hand back? The derived band model goes
    # through the ORDINARY builder, so it comes back in whichever form
    # `build_kwargs` asked for: the default in-place `f!(du, u, p, t)`, or the
    # out-of-place `f(u, p, t) -> du` that `form = :oop` produces. The
    # out-of-place form is not an optional nicety here — it is the only form
    # that traces (an in-place `f!` captures host scratch per node), so a
    # consumer compiling the Jacobian alongside a traced RHS builds BOTH that
    # way. Probed once, by method applicability, rather than by reaching into
    # the builder's return types.
    oop = !applicable(fJ!, zeros(1), zeros(1), pj, 0.0) &&
        applicable(fJ!, zeros(1), pj, 0.0)
    return JacobianEvaluator(
        fJ!, oop, zeros(length(u0j)), zeros(length(u0j)), umap,
        pj, scatter, items, field_shapes,
        convert(Dict{String, Int}, vmj), proto, pattern, entries,
        first.(sort(collect(vm0), by = last)), colnames, columns,
        structure, wrt
    )
end

# Evaluate every band field at (u, p, t); returns the derived-model `du`.
function _eval_fields!(jac::JacobianEvaluator, u::AbstractVector, p, t::Real)
    for (i, s) in enumerate(jac.umap)
        jac.uj[s] = u[i]
    end
    pp = p === nothing ? jac.p_default : p
    return jac.oop ? jac.fJ!(jac.uj, pp, t) :
        (jac.fJ!(jac.duj, jac.uj, pp, t); jac.duj)
end

function (jac::JacobianEvaluator)(J::SparseMatrixCSC, u::AbstractVector, p, t::Real)
    @assert size(J) == size(jac.prototype)
    duj = _eval_fields!(jac, u, p, t)
    fill!(J.nzval, 0.0)
    for (slot, pos) in jac.scatter
        J.nzval[pos] += duj[slot]
    end
    return J
end

"""
    vjp!(ḡ, jac::JacobianEvaluator, λ, u, p, t) -> ḡ

The transposed product `ḡ += Jᵀ λ` at `(u, p, t)`, as a scatter-add over the
band points: `ḡ[col] += coef · λ[row]` for every point, with no matrix (and
no transpose) formed. `λ` is indexed like the rows (`jac.rownames`, the
state vector) and `ḡ` like the columns (`jac.colnames`); `ḡ` ACCUMULATES —
zero it first for a single product.
"""
function vjp!(ḡ::AbstractVector, jac::JacobianEvaluator, λ::AbstractVector, u::AbstractVector, p, t::Real)
    length(ḡ) == length(jac.colnames) || throw(
        DimensionMismatch("ḡ has length $(length(ḡ)), the column space $(length(jac.colnames))")
    )
    length(λ) == length(jac.rownames) || throw(
        DimensionMismatch("λ has length $(length(λ)), the row space $(length(jac.rownames))")
    )
    duj = _eval_fields!(jac, u, p, t)
    for (slot, r, c) in jac.items
        ḡ[c] += duj[slot] * λ[r]
    end
    return ḡ
end

"""
    vjp_forcing!(ḡ, jac::JacobianEvaluator, λ, u, p, t) -> ḡ

[`vjp!`](@ref) on a `wrt = :forcing` evaluator: the forcing cotangent of one
stage evaluation, `ḡ_B += (∂f/∂B)ᵀ λ` (esm-jacobian-spec.md §4.3). `ḡ` is
either a flat vector over the canonical column ordinal (`jac.colnames`) or a
`Dict` of buffer-shaped accumulators keyed by buffer name (one per
`jac.columns` entry, each the buffer's shape). It accumulates; which dataset
record a buffer slot holds, and when to flush, is the caller's bookkeeping.
"""
function vjp_forcing!(ḡ::AbstractVector, jac::JacobianEvaluator, λ, u, p, t::Real)
    jac.wrt == :forcing ||
        throw(ArgumentError("vjp_forcing! needs a wrt = :forcing evaluator, got wrt = :$(jac.wrt)"))
    return vjp!(ḡ, jac, λ, u, p, t)
end

function vjp_forcing!(ḡ::AbstractDict, jac::JacobianEvaluator, λ, u, p, t::Real)
    flat = vjp_forcing!(zeros(length(jac.colnames)), jac, λ, u, p, t)
    off = 0
    for (name, shape) in jac.columns
        len = prod(shape; init = 1)
        acc = get(ḡ, name, nothing)
        acc === nothing && throw(ArgumentError("no accumulator for forcing buffer `$name`"))
        size(acc) == Tuple(shape) || throw(
            DimensionMismatch("accumulator `$name` has size $(size(acc)), the buffer $(Tuple(shape))")
        )
        for i in 1:len
            acc[i] += flat[off + i]      # both column-major
        end
        off += len
    end
    return ḡ
end

"""
    assemble_jacobian(input; wrt = :states, u = nothing, p = nothing, t = 0.0,
                      model_name = nothing, forcing = nothing)
        -> (J, pattern, rownames, colnames, entries, structure)

One-shot convenience over [`prepare_jacobian`](@ref): evaluate the analytical
Jacobian at a point (defaults: the document's initial state and parameters).
"""
function assemble_jacobian(
        input; wrt::Symbol = :states, u = nothing, p = nothing,
        t::Real = 0.0, model_name = nothing, forcing = nothing,
        build_kwargs = NamedTuple(), source_layout = nothing
    )
    jac = prepare_jacobian(
        input; wrt = wrt, model_name = model_name, forcing = forcing,
        build_kwargs = build_kwargs, source_layout = source_layout
    )
    u00, p0 = if source_layout === nothing
        b = _build_eval(_sv_src(input, model_name)[2], model_name; build_kwargs...)
        (b[2], b[3])
    else
        (
            getproperty(source_layout, :u0),
            hasproperty(source_layout, :p) ? getproperty(source_layout, :p) : nothing,
        )
    end
    uu = u === nothing ? u00 : u
    J = copy(jac.prototype)
    jac(J, uu, p === nothing ? p0 : p, t)
    return (
        J = J, pattern = jac.pattern, rownames = jac.rownames,
        colnames = jac.colnames, entries = jac.entries,
        structure = jac.structure,
    )
end
