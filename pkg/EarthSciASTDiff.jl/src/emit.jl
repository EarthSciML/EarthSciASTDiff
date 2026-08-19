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
    OrderedDict{String,Any}(
        "row"     => en.u,
        "col"     => en.v,
        "rows"    => Any[Any[lo, hi] for (lo, hi) in b.rows],
        "row_idx" => Any[b.ridx...],
        "col_idx" => Any[_ser_expr(c) for c in b.cidx],
        "coef"    => _ser_expr(b.coef),
    )
end

function parse_band(d::AbstractDict)::JacEntry
    band = Band(
        Tuple{Int,Int}[(Int(r[1]), Int(r[2])) for r in d["rows"]],
        Union{String,Int}[x isa AbstractString ? String(x) : Int(x)
                          for x in d["row_idx"]],
        ASTExpr[parse_expression(c) for c in d["col_idx"]],
        parse_expression(d["coef"]),
    )
    return JacEntry(String(d["row"]), String(d["col"]), band)
end

"""
    jacobian_document(file::EsmFile, model_name; wrt = :states) -> Dict
    jacobian_document(flat::FlattenedSystem; wrt = :states, name = "Flattened") -> Dict

The serialized input document plus the draft `"jacobians"` top-level block
(esm-jacobian-spec.md §4): per model, the differentiation axis, the band
entries, the detected structure class, and — when a factorization plan
exists for that structure — the serialized plan.
"""
function jacobian_document(file::EarthSciAST.EsmFile, model_name; wrt::Symbol = :states)
    sv = sysview(file, model_name)
    doc = EarthSciAST.serialize_esm_file(file)
    _attach_jacobians!(doc, sv, String(model_name); wrt = wrt)
    return doc
end

function jacobian_document(flat::EarthSciAST.FlattenedSystem;
                           wrt::Symbol = :states, name::AbstractString = "Flattened")
    sv = sysview(flat)
    doc = _base_document(sv, String(name))
    _attach_jacobians!(doc, sv, String(name); wrt = wrt)
    return doc
end

function _attach_jacobians!(doc, sv::SysView, model_name::String; wrt::Symbol)
    entries = jacobian_bands(sv; wrt = wrt)
    structure = detect_structure(entries, sv)
    block = OrderedDict{String,Any}(
        "wrt"       => String(wrt),
        "entries"   => Any[serialize_band(en) for en in entries],
        "structure" => String(structure),
    )
    plan = plan_factorization(entries, sv; structure = structure)
    plan === nothing || (block["factorization"] = serialize_plan(plan))
    jb = get!(doc, "jacobians", OrderedDict{String,Any}())
    jb[model_name] = block
    return doc
end

"""
    parse_jacobian_block(block) -> (wrt::Symbol, entries::Vector{JacEntry})

Inverse of the block emission (round-trip surface for conformance tests).
"""
function parse_jacobian_block(block::AbstractDict)
    wrt = Symbol(block["wrt"])
    entries = JacEntry[parse_band(d) for d in block["entries"]]
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

# Derived model: original system + one state per band whose RHS is the band's
# coefficient field. Returns (document, band-state names).
function _evaluation_document(sv::SysView, entries::Vector{JacEntry})
    doc = _base_document(sv, "JacobianEval")
    m = doc["models"]["JacobianEval"]
    ctx = _ctx(sv)
    names = String[]
    for (k, en) in enumerate(entries)
        b = en.band; nm = "J_$k"; push!(names, nm)
        shape_u = get(ctx.shapes, en.u, Int[])
        if isempty(shape_u)
            m["variables"][nm] = OrderedDict{String,Any}("type" => "state", "default" => 0.0)
            rhs = _ser_expr(b.coef)
        else
            m["variables"][nm] = OrderedDict{String,Any}(
                "type" => "state", "default" => 0.0,
                "shape" => sv.variables[en.u].shape)
            free = [(k2, r) for (k2, r) in enumerate(b.ridx) if r isa String]
            val = if isempty(free)
                _ser_expr(b.coef)
            else
                OrderedDict{String,Any}(
                    "op" => "aggregate", "args" => Any[],
                    "output_idx" => Any[r for (_, r) in free],
                    "ranges" => OrderedDict{String,Any}(
                        r => Any[b.rows[k2][1], b.rows[k2][2]] for (k2, r) in free),
                    "expr" => _ser_expr(b.coef))
            end
            rhs = OrderedDict{String,Any}(
                "op" => "makearray", "args" => Any[],
                "regions" => Any[Any[Any[1, n] for n in shape_u],
                                 Any[Any[lo, hi] for (lo, hi) in b.rows]],
                "values" => Any[0.0, val])
        end
        push!(m["equations"], OrderedDict{String,Any}(
            "lhs" => OrderedDict{String,Any}("op" => "D", "args" => [nm], "wrt" => "t"),
            "rhs" => rhs))
    end
    return doc, names
end
