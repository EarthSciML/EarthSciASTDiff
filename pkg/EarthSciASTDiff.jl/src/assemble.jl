# The structural pattern: every band point placed at its (row, column) slot of
# the source model's layout. No value is computed here; the values of `J` and
# its products are EarthSciAST's to evaluate, from the documents of emit.jl
# and products.jl.

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
_eval_guard(g::ASTExpr, bind::Dict{String, Float64})::Bool =
    evaluate_expr(_fold_const_gathers(g, bind), bind) != 0

# Every point of one band: `f(cell, cc, cidx)` per (row cell, contracted cell),
# first index fastest, with the column index evaluated there. Column index
# expressions are evaluated per cell ONCE (build-time, not solve-time). A
# point the band's guard (`where`) excludes is not a point of the band.
_foreach_point(f, b::Band) = (_all_points((a...) -> (f(a...); true), b); nothing)

# As `_foreach_point`, stopping at the first point where `f` returns false;
# whether it never did.
function _all_points(f, b::Band)::Bool
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
        b.guard === nothing || _eval_guard(b.guard, bind) || continue
        f(cell, cc, [_eval_cidx(c, bind) for c in b.cidx]) || return false
    end
    return true
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
        # A cell past a state's extent is a ghost read (0); a column variable
        # with no cell in the layout at all is not a state the layout holds
        # (-1), which `_colslot` refuses rather than skip.
        vars = Set{String}(first(split(String(n), '[')) for n in keys(vm0))
        slot(name) = (
            j = get(vm0, name, 0);
            j != 0 ? j : first(split(name, '[')) in vars ? 0 : -1
        )
        return slot, length(vm0), first.(sort(collect(vm0), by = last))
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

# The column slot of one scatter pair. A state column whose variable the
# layout does not hold at all resolves nowhere — an observed the band
# calculus took for a state, say — and is an error on every axis rather than
# a silently dropped derivative. On the forcing axis a column is a cell
# of a live buffer, and the ghost-read skip that is legitimate for a stencil
# reading past the domain is not: an unresolvable column there would silently
# drop a gradient, so it is an error (esm-jacobian-spec.md §4). The parameters
# axis follows the state axis: an array parameter read past its extent (a
# face coefficient `K[i+1]` beside `c[i+1]`) is a ghost read.
function _colslot(colslot, colname::String, rowname::String, wrt::Symbol)
    cs = colslot(colname)
    cs == -1 && throw(
        BandError(
            "column `$colname` (row `$rowname`) names no state of the layout; a band " *
                "column must resolve to a state slot or read past a state's extent"
        )
    )
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
# `jacobian_pattern` needs three things about the source model -- its state
# count, its scalar-parameter NamedTuple and its name -> state-slot map -- and
# by default gets them by running a COMPLETE `build_evaluator` of the source
# and throwing the RHS away, a cost that grows with the grid.
#
# A caller that has ALREADY built the source evaluator can hand the layout in
# and skip the rebuild. That is a sharp edge: a layout from a build with
# different `build_kwargs` (different `const_arrays`, `parameter_overrides`,
# metaparameters, ...) would put every band in the wrong cell with no visible
# symptom -- so the hand-in carries a contract (see `jacobian_pattern`), the
# cheap structural half of which is checked here and the name half at the use
# site.
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

The pattern needs the source model's state layout (the name → slot map),
and by default learns it by running a complete `build_evaluator` of the
source and discarding the right-hand side. A caller that has already built
the source evaluator can skip that build by passing its layout:

    # u0, p, var_map: build_evaluator's 2nd, 3rd and 5th returns
    source_layout = (u0 = u0, p = p, var_map = var_map)

!!! warning "Contract"
    The layout MUST come from a `build_evaluator` of the SAME source (same
    document / `FlattenedSystem`, same `model_name`) with the SAME
    `build_kwargs`. Only the cheap parts of that are checked (the state count,
    that the slots are a permutation, that every row name resolves); a layout
    from a build that merely *differs* — other `const_arrays`, other
    `parameter_overrides`, another grid — would put entries in the wrong cells
    with no visible symptom. The default, `nothing`, rebuilds.
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
        r = _rowslot(vm0, rowname, colslot(colname), strict)  # a wrong layout first
        cs = _colslot(colslot, colname, rowname, wrt)
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
