# Values from documents: the package evaluates nothing, so the tests (and
# scripts/regenerate-value-goldens.jl) get every number by running one of its
# documents through EarthSciAST — `J` column by column from the `JvpEval`
# product document (spec §9.2), `Jᵀ λ` from `VjpEval` and the second-order
# contraction from `HvpEval`. Shared by the test suite and the value-golden
# script; needs `EarthSciAST`, `EarthSciASTDiff` and `SparseArrays` loaded.

using EarthSciASTDiff: _sv_src, _build_eval, _column_names, _ctx

# An EarthSciAST evaluator of a derived document. The native compiler refuses
# some coefficient forms (a filtered or non-`+` faq) and every
# equality-filtered scatter (spec §9.3); the interpreter evaluates them, so it
# is the fallback when the caller did not pick a compiler.
function _doc_evaluator(doc, model; build_kwargs = NamedTuple())
    d = EarthSciASTDiff.JSON3.read(stable_json(doc))
    haskey(build_kwargs, :compiler) &&
        return build_evaluator(d; model_name = model, build_kwargs...)
    return try
        build_evaluator(d; model_name = model, build_kwargs...)
    catch err
        occursin("compiler_refused", sprint(showerror, err)) || rethrow()
        build_evaluator(d; model_name = model, compiler = :interpreter, build_kwargs...)
    end
end

# The document's name for a cell of variable `v`, given `v`'s name map.
function _doc_cell(names::AbstractDict, cell::AbstractString)
    k = findfirst('[', cell)
    k === nothing && return names[cell]
    return names[cell[1:(k - 1)]] * cell[k:end]
end

# The source's state layout, initial point and parameters.
function _doc_source(input, model_name, build_kwargs)
    _, src = _sv_src(input, model_name)
    _, u0, p0, _, vm0 = _build_eval(src, model_name; build_kwargs...)
    rownames = first.(sort(collect(vm0), by = last))
    return (u0 = u0, p = p0, var_map = vm0, rownames = rownames)
end

# One call of a document's right-hand side at the source point `u` (in the
# source's state order), with `inputs` (document name → value) written into
# their slots and the source's parameters `p` bound by name. Returns a lookup
# of `du` by document name.
function _doc_call(ev, layout, u, p, t, inputs)
    f!, x0, pd, _, vm = ev
    x = copy(x0)
    for (i, r) in enumerate(layout.rownames)
        x[vm[r]] = u[i]
    end
    for (n, v) in inputs
        x[vm[n]] = v
    end
    pp = p === nothing || pd isa NamedTuple && isempty(pd) ? pd :
        NamedTuple{keys(pd)}(Tuple(haskey(p, k) ? p[k] : pd[k] for k in keys(pd)))
    du = zeros(length(x))
    f!(du, x, pp, t)
    return n -> du[vm[n]]
end

_doc_colnames(sv, wrt, forcing, rownames) =
    wrt == :states ? rownames : wrt == :time ? ["t"] :
    _column_names(jacobian_columns(sv; wrt = wrt, forcing = forcing))

"""
    doc_jacobian(input; wrt = :states, u = nothing, p = nothing, t = 0.0,
                 model_name = nothing, forcing = nothing, build_kwargs = NamedTuple(),
                 cse = true)
        -> (J, pattern, rownames, colnames, entries, structure)

`J` at `(u, p, t)` (defaults: the document's initial state and parameters),
column `j` read off one `JvpEval` call at the unit direction `e_j`, stored on
the structural pattern ([`jacobian_pattern`](@ref)); a nonzero off the
pattern throws. The rows are the source's state layout.
"""
function doc_jacobian(
        input; wrt::Symbol = :states, u = nothing, p = nothing, t::Real = 0.0,
        model_name = nothing, forcing = nothing, build_kwargs = NamedTuple(),
        cse::Bool = true
    )
    layout = _doc_source(input, model_name, build_kwargs)
    sv, _ = _sv_src(input, model_name)
    pat = jacobian_pattern(
        input; wrt = wrt, model_name = model_name, forcing = forcing,
        build_kwargs = build_kwargs, source_layout = layout
    )
    doc, io = jvp_document(input; wrt = wrt, model_name = model_name, forcing = forcing, cse = cse)
    ev = _doc_evaluator(doc, io.model; build_kwargs = build_kwargs)
    uu = u === nothing ? layout.u0 : u
    pp = p === nothing ? layout.p : p
    rows, cols = pat.rownames, pat.colnames
    vals = zeros(length(rows), length(cols))
    for (j, c) in enumerate(cols)
        out = _doc_call(ev, layout, uu, pp, t, Dict(_doc_cell(io.v, c) => 1.0))
        for (i, r) in enumerate(rows)
            vals[i, j] = out(_doc_cell(io.jvp, r))
        end
    end
    P = pat.pattern
    for j in axes(vals, 2), i in axes(vals, 1)
        P[i, j] || vals[i, j] == 0 || error(
            "J[$(rows[i]), $(cols[j])] = $(vals[i, j]) lies off the structural pattern"
        )
    end
    I, Jc, _ = findnz(P)
    J = sparse(I, Jc, [vals[I[k], Jc[k]] for k in eachindex(I)], size(P)...)
    structure = wrt in (:states, :time) ? detect_structure(pat.entries, sv) : nothing
    return (
        J = J, pattern = P, rownames = rows, colnames = cols,
        entries = pat.entries, structure = structure,
    )
end

"""
    doc_vjp(input, λ; wrt = :states, u = nothing, p = nothing, t = 0.0, …) -> (g, colnames)

`Jᵀ λ` from one `VjpEval` call; `λ` and `u` in the source's state order, `g`
over the axis's columns (the source layout, `["t"]`, or the canonical column
ordinal).
"""
function doc_vjp(
        input, λ; wrt::Symbol = :states, u = nothing, p = nothing, t::Real = 0.0,
        model_name = nothing, forcing = nothing, build_kwargs = NamedTuple()
    )
    layout = _doc_source(input, model_name, build_kwargs)
    sv, _ = _sv_src(input, model_name)
    doc, io = vjp_document(input; wrt = wrt, model_name = model_name, forcing = forcing)
    ev = _doc_evaluator(doc, io.model; build_kwargs = build_kwargs)
    inputs = Dict(_doc_cell(io.lam, r) => λ[i] for (i, r) in enumerate(layout.rownames))
    out = _doc_call(
        ev, layout, u === nothing ? layout.u0 : u, p === nothing ? layout.p : p, t, inputs
    )
    cols = _doc_colnames(sv, wrt, forcing, layout.rownames)
    return [out(_doc_cell(io.vjp, c)) for c in cols], cols
end

"""
    doc_hvp(input, pairs; wrt = (:states, :states), form = :contract, u = nothing,
            p = nothing, t = 0.0, …) -> (g, col2names)

The §4.4 contraction from one `HvpEval` (`HvpEvalSymmetric`) call. `pairs` is
a list of stage pairs `(μ, w)`: `μ` in the source's state order, `w` likewise
(a scalar on a time pair). `g` is over the second axis's columns.
"""
function doc_hvp(
        input, pairs; wrt = (:states, :states), form::Symbol = :contract, u = nothing,
        p = nothing, t::Real = 0.0, model_name = nothing, forcing = nothing,
        build_kwargs = NamedTuple()
    )
    layout = _doc_source(input, model_name, build_kwargs)
    sv, _ = _sv_src(input, model_name)
    doc, io = hvp_document(
        input; stages = length(pairs), wrt = wrt, form = form,
        model_name = model_name, forcing = forcing
    )
    ev = _doc_evaluator(doc, io.model; build_kwargs = build_kwargs)
    inputs = Dict{String, Float64}()
    for (s, (μ, w)) in enumerate(pairs)
        for (i, r) in enumerate(layout.rownames)
            inputs[_doc_cell(io.mu[s], r)] = μ[i]
        end
        if w isa Number
            inputs[io.w[s]["t"]] = w
        else
            for (i, r) in enumerate(layout.rownames)
                inputs[_doc_cell(io.w[s], r)] = w[i]
            end
        end
    end
    out = _doc_call(
        ev, layout, u === nothing ? layout.u0 : u, p === nothing ? layout.p : p, t, inputs
    )
    cols = _doc_colnames(sv, wrt[2], forcing, layout.rownames)
    return [out(_doc_cell(io.hvp, c)) for c in cols], cols
end
