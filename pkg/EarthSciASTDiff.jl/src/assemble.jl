# Numeric assembly: the structural pattern, a one-shot Jacobian, and a
# prepared evaluator with a fixed sparse prototype + precomputed scatter map.

_cellname(v::String, cell) = isempty(cell) ? v : "$(v)[$(join(cell, ","))]"

# Enumerate every (band, row cell) → (row name, col name) pair. Column index
# expressions are evaluated per cell ONCE here (build-time, not solve-time).
function _scatter_pairs(sv::SysView, entries::Vector{JacEntry})
    ctx = _ctx(sv)
    pairs = Vector{Tuple{Int,String,String,String}}()  # (band k, rowname, colname, jname-cell)
    for (k, en) in enumerate(entries)
        b = en.band
        rowcells = isempty(b.rows) ? [Int[]] :
            [collect(c) for c in Iterators.product(((lo:hi) for (lo, hi) in b.rows)...)]
        for cell in rowcells
            bind = Dict{String,Float64}(
                r => Float64(cell[k2]) for (k2, r) in enumerate(b.ridx) if r isa String)
            cidx = [Int(round(evaluate_expr(c, bind))) for c in b.cidx]
            push!(pairs, (k, _cellname(en.u, cell), _cellname(en.v, cidx),
                          _cellname("J_$k", cell)))
        end
    end
    return pairs
end

# Column slot lookup: state columns index the state vector; parameter columns
# index the scalar-parameter order of the RHS `p` NamedTuple.
function _colmap(vm0, p0, wrt::Symbol)
    if wrt == :states
        return name -> get(vm0, name, 0), length(vm0),
               first.(sort(collect(vm0), by = last))
    else
        pnames = String[String(k) for k in keys(p0)]
        lut = Dict(n => i for (i, n) in enumerate(pnames))
        return name -> get(lut, name, 0), length(pnames), pnames
    end
end

"""
    jacobian_pattern(input; wrt = :states) -> (pattern, rownames, colnames, entries)

The structural (global, point-independent) sparsity pattern of the Jacobian
as a `SparseMatrixCSC{Bool}`. `input` is an `EsmFile` + model name pair
(pass a `Tuple`), an `EsmFile` with a single model, or a `FlattenedSystem`.
"""
function jacobian_pattern(input; wrt::Symbol = :states, model_name = nothing,
                          build_kwargs = NamedTuple())
    sv, src = _sv_src(input, model_name)
    entries = jacobian_bands(sv; wrt = wrt)
    _, _, p0, _, vm0 = build_evaluator(src; build_kwargs...)
    colslot, ncol, colnames = _colmap(vm0, p0, wrt)
    n = length(vm0)
    I = Int[]; J = Int[]
    for (_, rowname, colname, _) in _scatter_pairs(sv, entries)
        cs = colslot(colname); cs == 0 && continue   # ghost / out-of-layout column
        push!(I, vm0[rowname]); push!(J, cs)
    end
    pattern = sparse(I, J, trues(length(I)), n, ncol, |)
    return (pattern = pattern,
            rownames = first.(sort(collect(vm0), by = last)),
            colnames = colnames, entries = entries)
end

_sv_src(file::EarthSciAST.EsmFile, model_name) =
    (sysview(file, model_name === nothing ? _only_model(file) : model_name), file)
_sv_src(flat::EarthSciAST.FlattenedSystem, ::Any) = (sysview(flat), flat)

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
struct JacobianEvaluator{F,P}
    fJ!::F                        # derived-model RHS (fills band fields into duj)
    uj::Vector{Float64}           # derived-model state buffer
    duj::Vector{Float64}
    umap::Vector{Int}             # uj[umap[i]] = u[i]
    p_default::P
    scatter::Vector{Tuple{Int,Int}}  # (slot in duj, position in nzval)
    prototype::SparseMatrixCSC{Float64,Int}
    pattern::SparseMatrixCSC{Bool,Int}
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

    jac = prepare_jacobian(file)
    J = copy(jac.prototype)
    jac(J, u, p, t)              # fills J in place; stored zeros stay
"""
function prepare_jacobian(input; wrt::Symbol = :states, model_name = nothing,
                          build_kwargs = NamedTuple())
    sv, src = _sv_src(input, model_name)
    entries = jacobian_bands(sv; wrt = wrt)
    doc, _ = _evaluation_document(sv, entries)
    fJ!, u0j, pj, _, vmj = build_evaluator(doc; model_name = "JacobianEval",
                                           build_kwargs...)
    _, u00, p0, _, vm0 = build_evaluator(src; build_kwargs...)
    n = length(u00)
    colslot, ncol, colnames = _colmap(vm0, p0, wrt)

    umap = zeros(Int, n)
    for (name, i) in vm0
        umap[i] = vmj[name]
    end

    items = Tuple{Int,Int,Int}[]     # (duj slot, row, col)
    for (k, rowname, colname, jname) in _scatter_pairs(sv, entries)
        cs = colslot(colname); cs == 0 && continue
        push!(items, (vmj[jname], vm0[rowname], cs))
    end
    I = [it[2] for it in items]; J = [it[3] for it in items]
    pattern = sparse(I, J, trues(length(I)), n, ncol, |)
    proto = SparseMatrixCSC(n, ncol, copy(pattern.colptr), copy(pattern.rowval),
                            zeros(length(pattern.rowval)))
    # nzval position of each (row, col): binary search the fixed pattern.
    scatter = Tuple{Int,Int}[]
    for (slot, r, c) in items
        rng = proto.colptr[c]:(proto.colptr[c+1]-1)
        pos = rng[searchsortedfirst(view(proto.rowval, rng), r)]
        push!(scatter, (slot, pos))
    end
    structure = detect_structure(entries, sv)
    return JacobianEvaluator(fJ!, zeros(length(u0j)), zeros(length(u0j)), umap,
                             pj, scatter, proto, pattern, entries,
                             first.(sort(collect(vm0), by = last)), colnames,
                             structure, wrt)
end

function (jac::JacobianEvaluator)(J::SparseMatrixCSC, u::AbstractVector, p, t::Real)
    @assert size(J) == size(jac.prototype)
    for (i, s) in enumerate(jac.umap)
        jac.uj[s] = u[i]
    end
    jac.fJ!(jac.duj, jac.uj, p === nothing ? jac.p_default : p, t)
    fill!(J.nzval, 0.0)
    for (slot, pos) in jac.scatter
        J.nzval[pos] += jac.duj[slot]
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
function assemble_jacobian(input; wrt::Symbol = :states, u = nothing, p = nothing,
                           t::Real = 0.0, model_name = nothing,
                           build_kwargs = NamedTuple())
    jac = prepare_jacobian(input; wrt = wrt, model_name = model_name,
                           build_kwargs = build_kwargs)
    _, u00, p0, _, _ = build_evaluator(_sv_src(input, model_name)[2]; build_kwargs...)
    uu = u === nothing ? u00 : u
    J = copy(jac.prototype)
    jac(J, uu, p === nothing ? p0 : p, t)
    return (J = J, pattern = jac.pattern, rownames = jac.rownames,
            colnames = jac.colnames, entries = jac.entries,
            structure = jac.structure)
end
