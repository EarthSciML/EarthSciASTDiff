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

# Column slot lookup: state columns index the state vector; parameter columns
# index the scalar-parameter order of the RHS `p` NamedTuple; time is the
# single column `t`.
function _colmap(vm0, p0, wrt::Symbol)
    if wrt == :time
        return name -> name == "t" ? 1 : 0, 1, ["t"]
    elseif wrt == :states
        return name -> get(vm0, name, 0), length(vm0),
            first.(sort(collect(vm0), by = last))
    else
        pnames = String[String(k) for k in keys(p0)]
        lut = Dict(n => i for (i, n) in enumerate(pnames))
        return name -> get(lut, name, 0), length(pnames), pnames
    end
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
    p0 === nothing && !(wrt in (:states, :time)) && throw(
        ArgumentError(
            "source_layout: wrt = :$wrt indexes columns by parameter name, so the " *
                "layout must carry `p` (the build_evaluator parameter NamedTuple)."
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
    jacobian_pattern(input; wrt = :states) -> (pattern, rownames, colnames, entries)

The structural (global, point-independent) sparsity pattern of the Jacobian
as a `SparseMatrixCSC{Bool}`. `input` is an `EsmFile` + model name pair
(pass a `Tuple`), an `EsmFile` with a single model, or a `FlattenedSystem`.

`source_layout` skips this function's own rebuild of the source model — see
[`prepare_jacobian`](@ref) for the keyword and its contract.
"""
function jacobian_pattern(
        input; wrt::Symbol = :states, model_name = nothing,
        build_kwargs = NamedTuple(), source_layout = nothing
    )
    sv, src = _sv_src(input, model_name)
    entries = jacobian_bands(sv; wrt = wrt)
    _, p0, vm0 = _source_layout(src, model_name, build_kwargs, source_layout, wrt)
    colslot, ncol, colnames = _colmap(vm0, p0, wrt)
    n = length(vm0)
    strict = source_layout !== nothing
    I = Int[]; J = Int[]
    for (_, rowname, colname, _) in _scatter_pairs(sv, entries)
        cs = colslot(colname)
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
    field_shapes::Vector{Vector{Int}} # band k → field shape (scatter plans)
    slots::Dict{String, Int}          # derived-model name → slot in duj
    prototype::SparseMatrixCSC{Float64, Int}
    pattern::SparseMatrixCSC{Bool, Int}
    entries::Vector{JacEntry}
    rownames::Vector{String}
    colnames::Vector{String}
    structure::Symbol
    wrt::Symbol
end

"""
    prepare_jacobian(input; wrt = :states, model_name = nothing) -> JacobianEvaluator

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
    bands in the wrong cells with no visible symptom. `p` may be omitted
    unless `wrt` selects parameter columns. The default, `nothing`, rebuilds.

    jac = prepare_jacobian(file)
    J = copy(jac.prototype)
    jac(J, u, p, t)              # fills J in place; stored zeros stay
"""
function prepare_jacobian(
        input; wrt::Symbol = :states, model_name = nothing,
        build_kwargs = NamedTuple(), source_layout = nothing,
        eval_cse::Bool = true, eval_cse_min_nodes::Int = 12
    )
    sv, src = _sv_src(input, model_name)
    entries = jacobian_bands(sv; wrt = wrt)
    doc, _ = _evaluation_document(
        sv, entries; cse = eval_cse,
        cse_min_nodes = eval_cse_min_nodes
    )
    fJ!, u0j, pj, _, vmj = build_evaluator(
        doc; model_name = "JacobianEval",
        build_kwargs...
    )
    n, p0, vm0 = _source_layout(src, model_name, build_kwargs, source_layout, wrt)
    colslot, ncol, colnames = _colmap(vm0, p0, wrt)

    umap = zeros(Int, n)
    for (name, i) in vm0
        umap[i] = _slot(vmj, name, "band")
    end

    strict = source_layout !== nothing
    items = Tuple{Int, Int, Int}[]     # (duj slot, row, col)
    for (k, rowname, colname, jname) in _scatter_pairs(sv, entries)
        cs = colslot(colname)
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
    structure = detect_structure(entries, sv)
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
        pj, scatter, field_shapes,
        convert(Dict{String, Int}, vmj), proto, pattern, entries,
        first.(sort(collect(vm0), by = last)), colnames,
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
    assemble_jacobian(input; wrt = :states, u = nothing, p = nothing, t = 0.0,
                      model_name = nothing)
        -> (J, pattern, rownames, colnames, entries, structure)

One-shot convenience over [`prepare_jacobian`](@ref): evaluate the analytical
Jacobian at a point (defaults: the document's initial state and parameters).
"""
function assemble_jacobian(
        input; wrt::Symbol = :states, u = nothing, p = nothing,
        t::Real = 0.0, model_name = nothing,
        build_kwargs = NamedTuple(), source_layout = nothing
    )
    jac = prepare_jacobian(
        input; wrt = wrt, model_name = model_name,
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
