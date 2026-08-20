# Serialization: the draft `"jacobians"` document block (esm-jacobian-spec.md
# §4), a deterministic JSON writer for goldens/conformance, and the internal
# derived model that lets the ordinary tree-walk evaluate band fields.

"""
    stable_json(x) -> String

Deterministic JSON: object keys emitted in sorted order at every level,
scalars formatted by JSON3. Used for conformance goldens and for structural
expression keys ([`skey`](@ref)) — equal trees always produce equal text.
"""
function stable_json(x)::String
    io = IOBuffer()
    _sj(io, x)
    return String(take!(io))
end

function _sj(io::IO, x::AbstractDict)
    print(io, '{')
    for (i, k) in enumerate(sort!(collect(String.(keys(x)))))
        i > 1 && print(io, ',')
        _sj(io, k); print(io, ':'); _sj(io, x[k])
    end
    print(io, '}')
end
function _sj(io::IO, x::Union{AbstractVector,Tuple})
    print(io, '[')
    for (i, v) in enumerate(x)
        i > 1 && print(io, ',')
        _sj(io, v)
    end
    print(io, ']')
end
_sj(io::IO, x) = print(io, JSON3.write(x))

# ── the "jacobians" block ────────────────────────────────────────────────────

_ser_expr(e::ASTExpr) = EarthSciAST.serialize_expression(e)

function serialize_band(en::JacEntry)
    b = en.band
    d = OrderedDict{String,Any}(
        "row"     => en.u,
        "col"     => en.v,
        "rows"    => Any[Any[lo, hi] for (lo, hi) in b.rows],
        "row_idx" => Any[b.ridx...],
        "col_idx" => Any[_ser_expr(c) for c in b.cidx],
        "coef"    => _ser_expr(b.coef),
    )
    isempty(b.contracted) ||
        (d["contracted"] = Any[Any[n, lo, hi] for (n, lo, hi) in b.contracted])
    return d
end

function parse_band(d::AbstractDict)::JacEntry
    band = Band(
        Tuple{Int,Int}[(Int(r[1]), Int(r[2])) for r in d["rows"]],
        Union{String,Int}[x isa AbstractString ? String(x) : Int(x)
                          for x in d["row_idx"]],
        ASTExpr[parse_expression(c) for c in d["col_idx"]],
        parse_expression(d["coef"]),
        haskey(d, "contracted") ?
            Tuple{String,Int,Int}[(String(c[1]), Int(c[2]), Int(c[3]))
                                  for c in d["contracted"]] :
            Tuple{String,Int,Int}[],
    )
    return JacEntry(String(d["row"]), String(d["col"]), band)
end

"""
    jacobian_document(file::EsmFile, model_name; wrt = :states,
                      cse = true, cse_min_nodes = 12) -> Dict
    jacobian_document(flat::FlattenedSystem; wrt = :states, name = "Flattened",
                      cse = true, cse_min_nodes = 12) -> Dict

The serialized input document plus the draft `"jacobians"` top-level block
(esm-jacobian-spec.md §4): per model, the differentiation axis, the band
entries, the detected structure class, and — when a factorization plan
exists for that structure — the serialized plan.

With `cse = true` (default), repeated coefficient subtrees of at least
`cse_min_nodes` AST nodes are hoisted into zero-parameter
`expression_templates` carried inside the block ([`cse_templates`](@ref)),
and the emitted coefficients reference them via `apply_expression_template`
— the size control for limiter-heavy schemes. [`parse_jacobian_block`](@ref)
expands the references back, so the round-trip is exact either way.
"""
function jacobian_document(file::EarthSciAST.EsmFile, model_name; wrt::Symbol = :states,
                           cse::Bool = true, cse_min_nodes::Int = 12)
    sv = sysview(file, model_name)
    doc = EarthSciAST.serialize_esm_file(file)
    _attach_jacobians!(doc, sv, String(model_name); wrt = wrt,
                       cse = cse, cse_min_nodes = cse_min_nodes)
    return doc
end

function jacobian_document(flat::EarthSciAST.FlattenedSystem;
                           wrt::Symbol = :states, name::AbstractString = "Flattened",
                           cse::Bool = true, cse_min_nodes::Int = 12)
    sv = sysview(flat)
    doc = _base_document(sv, String(name))
    _attach_jacobians!(doc, sv, String(name); wrt = wrt,
                       cse = cse, cse_min_nodes = cse_min_nodes)
    return doc
end

function _attach_jacobians!(doc, sv::SysView, model_name::String; wrt::Symbol,
                            cse::Bool = true, cse_min_nodes::Int = 12)
    entries = jacobian_bands(sv; wrt = wrt)
    structure = detect_structure(entries, sv)
    ser_entries = entries
    templates = OrderedDict{String,ASTExpr}()
    if cse
        coefs = ASTExpr[en.band.coef for en in entries]
        templates, rewritten = cse_templates(coefs; min_nodes = cse_min_nodes)
        isempty(templates) || (ser_entries = JacEntry[
            JacEntry(en.u, en.v, Band(en.band.rows, en.band.ridx,
                                      en.band.cidx, rewritten[k],
                                      en.band.contracted))
            for (k, en) in enumerate(entries)])
    end
    block = OrderedDict{String,Any}(
        "wrt"       => String(wrt),
        "entries"   => Any[serialize_band(en) for en in ser_entries],
        "structure" => String(structure),
    )
    isempty(templates) || (block["expression_templates"] = OrderedDict{String,Any}(
        name => OrderedDict{String,Any}("params" => Any[], "body" => _ser_expr(body))
        for (name, body) in templates))
    plan = plan_factorization(entries, sv; structure = structure)
    plan === nothing || (block["factorization"] = serialize_plan(plan))
    jb = get!(doc, "jacobians", OrderedDict{String,Any}())
    jb[model_name] = block
    return doc
end

"""
    parse_jacobian_block(block) -> (wrt::Symbol, entries::Vector{JacEntry})

Inverse of the block emission (round-trip surface for conformance tests).
When the block carries `expression_templates` (CSE'd emission), every
`apply_expression_template` reference in the parsed coefficients is expanded
back to its body, so the returned entries are always closed expressions.
"""
function parse_jacobian_block(block::AbstractDict)
    wrt = Symbol(block["wrt"])
    entries = JacEntry[parse_band(d) for d in block["entries"]]
    if haskey(block, "expression_templates")
        reg = Dict{String,ASTExpr}(
            String(name) => parse_expression(t["body"])
            for (name, t) in block["expression_templates"])
        entries = JacEntry[
            JacEntry(en.u, en.v, Band(en.band.rows, en.band.ridx,
                                      ASTExpr[expand_templates(c, reg) for c in en.band.cidx],
                                      expand_templates(en.band.coef, reg),
                                      en.band.contracted))
            for en in entries]
    end
    return wrt, entries
end

# ── the internal evaluation model ────────────────────────────────────────────
#
# v1 evaluation strategy (deliberate reuse of existing machinery, zero new
# evaluator code): emit a derived single-model document in which each band's
# coefficient field is a STATE `J_k` whose RHS is the band field —
#
#   D(J_k) = makearray( zero default, band region ← aggregate(coef) )
#
# and run it through the unmodified `EarthSciAST.build_evaluator`; one RHS
# call then leaves every band value in `du`. A dedicated compiled `jac!`
# kernel is the planned upgrade once this is a measured bottleneck.

function _base_document(sv::SysView, name::String)
    m = OrderedDict{String,Any}("variables" => OrderedDict{String,Any}(),
                                "equations" => Any[])
    for n in sort!(collect(keys(sv.variables)))
        var = sv.variables[n]
        d = OrderedDict{String,Any}("type" =>
            var.type == EarthSciAST.StateVariable     ? "state" :
            var.type == EarthSciAST.ParameterVariable ? "parameter" : "observed")
        var.shape === nothing || isempty(var.shape) || (d["shape"] = var.shape)
        var.default === nothing || (d["default"] = var.default)
        var.expression === nothing || (d["expression"] = _ser_expr(var.expression))
        m["variables"][n] = d
    end
    for eq in sv.equations
        push!(m["equations"], OrderedDict{String,Any}(
            "lhs" => _ser_expr(eq.lhs), "rhs" => _ser_expr(eq.rhs)))
    end
    isets = OrderedDict{String,Any}(
        String(n) => OrderedDict{String,Any}("kind" => "interval", "size" => s.size)
        for (n, s) in sv.index_sets if s.size !== nothing)
    return OrderedDict{String,Any}(
        "esm" => "0.8.0",
        "metadata" => OrderedDict{String,Any}("name" => name),
        "index_sets" => isets,
        "models" => OrderedDict{String,Any}(name => m))
end

# A literal-size axis index set of the derived document (shared with the
# hoisted-observed axes: same `jcax<size>` convention, reused when present).
function _axis_name!(doc, sz::Int)
    isets = doc["index_sets"]
    nm = "jcax$(sz)"
    while haskey(isets, nm) && get(isets[nm], "size", nothing) != sz
        nm *= "x"
    end
    haskey(isets, nm) ||
        (isets[nm] = OrderedDict{String,Any}("kind" => "interval", "size" => sz))
    return nm
end

# Derived model: original system + one state per band whose RHS is the band's
# coefficient field. With `cse = true`, repeated coefficient subtrees are
# first hoisted into observed variables of the document
# ([`hoist_observed`](@ref)) so the tree-walk's factored array-observed
# buffers evaluate them once per cell per RHS call — the compile-time and
# duplicate-work control for limiter-heavy (PPM-scale) coefficients.
# Returns (document, band-state names).
function _evaluation_document(sv::SysView, entries::Vector{JacEntry};
                              cse::Bool = true, cse_min_nodes::Int = 12)
    doc = _base_document(sv, "JacobianEval")
    m = doc["models"]["JacobianEval"]
    ctx = _ctx(sv)
    h = cse ? hoist_observed(entries, sv; min_nodes = cse_min_nodes) : nothing
    coefs = h === nothing ? ASTExpr[en.band.coef for en in entries] : h.coefs
    if h !== nothing
        for (nm, sz) in h.axes
            doc["index_sets"][nm] =
                OrderedDict{String,Any}("kind" => "interval", "size" => sz)
        end
        for (nm, vd) in h.obs
            m["variables"][nm] = vd
        end
    end
    names = String[]
    for (k, en) in enumerate(entries)
        b = en.band; nm = "J_$k"; push!(names, nm)
        shape_u = get(ctx.shapes, en.u, Int[])
        cr = b.contracted
        if isempty(shape_u) && isempty(cr)
            m["variables"][nm] = OrderedDict{String,Any}("type" => "state", "default" => 0.0)
            rhs = _ser_expr(coefs[k])
        else
            # The band field spans the row dims PLUS any free contracted
            # column dims — the scatter reads one slot per (row, contracted)
            # cell and ACCUMULATES into the column each cell gathers.
            ushape = isempty(shape_u) ? Any[] : Any[x for x in sv.variables[en.u].shape]
            m["variables"][nm] = OrderedDict{String,Any}(
                "type" => "state", "default" => 0.0,
                "shape" => vcat(ushape, Any[_axis_name!(doc, hi) for (_, _, hi) in cr]))
            free = [(k2, r) for (k2, r) in enumerate(b.ridx) if r isa String]
            aggnames = Any[Any[r for (_, r) in free]..., Any[n for (n, _, _) in cr]...]
            val = if isempty(aggnames)
                _ser_expr(coefs[k])
            else
                rngs = OrderedDict{String,Any}(
                    r => Any[b.rows[k2][1], b.rows[k2][2]] for (k2, r) in free)
                for (n, lo, hi) in cr
                    rngs[n] = Any[lo, hi]
                end
                OrderedDict{String,Any}(
                    "op" => "aggregate", "args" => Any[],
                    "output_idx" => aggnames,
                    "ranges" => rngs,
                    "expr" => _ser_expr(coefs[k]))
            end
            rhs = OrderedDict{String,Any}(
                "op" => "makearray", "args" => Any[],
                "regions" => Any[
                    vcat(Any[Any[1, n] for n in shape_u],
                         Any[Any[1, hi] for (_, _, hi) in cr]),
                    vcat(Any[Any[lo, hi] for (lo, hi) in b.rows],
                         Any[Any[lo, hi] for (_, lo, hi) in cr])],
                "values" => Any[0.0, val])
        end
        push!(m["equations"], OrderedDict{String,Any}(
            "lhs" => OrderedDict{String,Any}("op" => "D", "args" => [nm], "wrt" => "t"),
            "rhs" => rhs))
    end
    return doc, names
end
