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
    return print(io, '}')
end
function _sj(io::IO, x::Union{AbstractVector, Tuple})
    print(io, '[')
    for (i, v) in enumerate(x)
        i > 1 && print(io, ',')
        _sj(io, v)
    end
    return print(io, ']')
end
_sj(io::IO, x) = print(io, JSON3.write(x))

# ── the "jacobians" block ────────────────────────────────────────────────────

_ser_expr(e::ASTExpr) = EarthSciAST.serialize_expression(e)

function serialize_band(en::JacEntry)
    b = en.band
    d = OrderedDict{String, Any}(
        "row" => en.u,
        "col" => en.v,
        "rows" => Any[Any[lo, hi] for (lo, hi) in b.rows],
        "row_idx" => Any[b.ridx...],
        "col_idx" => Any[_ser_expr(c) for c in b.cidx],
        "coef" => _ser_expr(b.coef),
    )
    isempty(b.contracted) ||
        (d["contracted"] = Any[Any[n, lo, hi] for (n, lo, hi) in b.contracted])
    b.guard === nothing || (d["where"] = _ser_expr(b.guard))
    return d
end

function parse_band(d::AbstractDict)::JacEntry
    band = Band(
        Tuple{Int, Int}[(Int(r[1]), Int(r[2])) for r in d["rows"]],
        Union{String, Int}[
            x isa AbstractString ? String(x) : Int(x)
                for x in d["row_idx"]
        ],
        ASTExpr[expression_from_json(c) for c in d["col_idx"]],
        expression_from_json(d["coef"]),
        haskey(d, "contracted") ?
            Tuple{String, Int, Int}[
                (String(c[1]), Int(c[2]), Int(c[3]))
                for c in d["contracted"]
            ] :
            Tuple{String, Int, Int}[],
        haskey(d, "where") ? expression_from_json(d["where"]) : nothing,
    )
    return JacEntry(String(d["row"]), String(d["col"]), band)
end

"""
    jacobian_document(file::EsmFile, model_name; wrt = :states, forcing = nothing,
                      cse = true, cse_min_nodes = 12, column_axis = nothing) -> Dict
    jacobian_document(flat::FlattenedSystem; wrt = :states, forcing = nothing,
                      name = "Flattened", cse = true, cse_min_nodes = 12,
                      column_axis = nothing) -> Dict

The serialized input document plus the draft `"jacobians"` top-level block
(esm-jacobian-spec.md §4): per model, the differentiation axis, the band
entries, and — on the state and time axes — the detected structure class and,
when a factorization plan exists for that structure, the serialized plan. A
`:parameters` or `:forcing` block carries its column space instead
(`columns`, [`jacobian_columns`](@ref)).

`wrt` may also be a vector of distinct axes (`[:states, :forcing]`): the
model's entry is then an ARRAY of blocks, one per axis in the given order
(esm-jacobian-spec.md §4); a single `wrt` keeps the single-object form. A
`Pair` `a1 => a2` is an axis PAIR, a block of second-order entries
([`jacobian_bands2`](@ref), §4.4), alone or in such a list, where a Tuple
`(a1, a2)` (or a two-name vector) is a pair too
(`[:states, (:states, :states)]`). A bare Tuple `wrt` is refused: it would
read either as a pair or as a list of axes, so neither is assumed.

With `cse = true` (default), repeated coefficient subtrees of at least
`cse_min_nodes` AST nodes are hoisted into zero-parameter
`expression_templates` carried inside the block ([`cse_templates`](@ref)),
and the emitted coefficients reference them via `apply_expression_template`
— the size control for limiter-heavy schemes. [`parse_jacobian_block`](@ref)
expands the references back, so the round-trip is exact either way.

`column_axis` (an index-set name) lets the state block report the
`"column_block"` structure class with its line-by-line factorization plan
when every coupling stays within one line along that axis
([`detect_structure`](@ref), [`column_lines`](@ref)).
"""
function jacobian_document(
        file::EarthSciAST.EsmFile, model_name; wrt = :states, forcing = nothing,
        cse::Bool = true, cse_min_nodes::Int = 12, column_axis = nothing
    )
    sv = sysview(file, model_name)
    doc = EarthSciAST.serialize_esm_file(file)
    _attach_jacobians!(
        doc, sv, String(model_name); wrt = wrt, forcing = forcing,
        cse = cse, cse_min_nodes = cse_min_nodes, column_axis = column_axis
    )
    return doc
end

function jacobian_document(
        flat::EarthSciAST.FlattenedSystem;
        wrt = :states, forcing = nothing, name::AbstractString = "Flattened",
        cse::Bool = true, cse_min_nodes::Int = 12, column_axis = nothing
    )
    sv = sysview(flat)
    doc = _base_document(sv, String(name))
    _attach_jacobians!(
        doc, sv, String(name); wrt = wrt, forcing = forcing,
        cse = cse, cse_min_nodes = cse_min_nodes, column_axis = column_axis
    )
    return doc
end

function _attach_jacobians!(
        doc, sv::SysView, model_name::String; wrt, forcing = nothing,
        cse::Bool = true, cse_min_nodes::Int = 12, column_axis = nothing
    )
    # One axis may be spelled as a Symbol or a String; iterating a String
    # would split it into one-character "axes". A Pair is one axis PAIR
    # (second order, §4.4); inside a list, a Tuple or a two-name vector is
    # one too. A bare Tuple is both a pair and a list of axes, so it is
    # refused rather than silently read as either.
    wrt isa Tuple && throw(
        ArgumentError(
            "a Tuple wrt $wrt is ambiguous: write a list of axes as a vector " *
                "([:states, :forcing]) and one second-order pair as a Pair " *
                "(:states => :forcing)"
        )
    )
    single = wrt isa Union{Symbol, AbstractString, Pair}
    axes = single ? Any[_axis_key(wrt)] : Any[_axis_key(w) for w in wrt]
    (isempty(axes) || !allunique(axes)) &&
        throw(ArgumentError("wrt must name one or more distinct axes, got $wrt"))
    memo = _FirstOrder()
    blocks = Any[
        _jacobian_block(
            sv, w, memo; forcing = forcing, cse = cse, cse_min_nodes = cse_min_nodes,
            column_axis = column_axis
        )
            for w in axes
    ]
    jb = get!(doc, "jacobians", OrderedDict{String, Any}())
    jb[model_name] = single ? only(blocks) : blocks
    _raise_faq_floor!(doc)
    return doc
end

_axis_key(w::Union{Symbol, AbstractString}) = Symbol(w)
_axis_key(w::Union{Tuple, AbstractVector, Pair}) = _axis_pair(w)

# One `"jacobians"` block (esm-jacobian-spec.md §4) for one axis. `memo`
# shares the first-order pass with the pairs of the same emission.
function _jacobian_block(
        sv::SysView, wrt::Symbol, memo::_FirstOrder = _FirstOrder(); forcing = nothing,
        cse::Bool = true, cse_min_nodes::Int = 12, column_axis = nothing
    )
    _check_axis(wrt)
    entries = _first_order!(memo, sv, wrt, nothing, forcing)
    # A parameter or forcing block describes a rectangular matrix, which the
    # structure classes and the factorization plan do not: it carries its
    # column space instead, so a consumer can size and order the columns
    # without the source document.
    rect = wrt in (:parameters, :forcing)
    structure = rect ? nothing : _structure!(memo, entries, sv)
    # Line-by-line structure is a property of the square state matrix.
    cax = wrt == :states ? column_axis : nothing
    cax === nothing || structure in (:empty, :block_diagonal) ||
        column_lines(entries, sv, String(cax)) === nothing || (structure = :column_block)
    block = _block(
        String(wrt), entries, rect ? _columns_field(sv, wrt, forcing) : nothing,
        structure, cse, cse_min_nodes
    )
    if !rect
        plan = plan_factorization(entries, sv; structure = structure, column_axis = cax)
        plan === nothing || (block["factorization"] = serialize_plan(plan))
    end
    return block
end

# The fields every block shares, in their emitted order: `wrt`, the entries
# (coefficients template-CSE'd when `cse`), the `columns` of a parameter or
# forcing axis and the `structure` class (each when given), and the
# block-local `expression_templates`.
function _block(wrt, entries, columns, structure, cse::Bool, cse_min_nodes::Int)
    ser_entries = entries
    templates = OrderedDict{String, ASTExpr}()
    if cse
        templates, rewritten = cse_templates(
            ASTExpr[en.band.coef for en in entries]; min_nodes = cse_min_nodes
        )
        isempty(templates) || (
            ser_entries = [
                _with_band(en, _rebuild(b; coef = rewritten[k]))
                    for (k, en) in enumerate(entries) for b in (en.band,)
            ]
        )
    end
    block = OrderedDict{String, Any}(
        "wrt" => wrt,
        "entries" => Any[serialize_band(en) for en in ser_entries],
    )
    columns === nothing || (block["columns"] = columns)
    structure === nothing || (block["structure"] = String(structure))
    isempty(templates) || (
        block["expression_templates"] = OrderedDict{String, Any}(
            name => OrderedDict{String, Any}("params" => Any[], "body" => _ser_expr(body))
                for (name, body) in templates
        )
    )
    return block
end

_columns_field(sv::SysView, wrt::Symbol, forcing) = Any[
    OrderedDict{String, Any}("name" => n, "shape" => Any[shape...])
        for (n, shape) in jacobian_columns(sv; wrt = wrt, forcing = forcing)
]

"""
    _raise_faq_floor!(doc) -> doc

Raise a document's declared `esm` version to the 1.1.0 floor when it is older.
The band coefficients emitted above may contain `faq` nodes, which arrive at
esm 1.1.0; a source model that legally declared 1.0.0 would otherwise end up
spelling a 1.1.0 construct under a version that forbids it, and the loader
would reject the result. This is EarthSciAST's own emit-side rule
(`docs/content/rfcs/faq-node-rename.md` §5.5).
"""
function _raise_faq_floor!(doc)
    v = get(doc, "esm", nothing)
    v isa AbstractString || return doc
    parts = split(v, '.')
    length(parts) >= 2 || return doc
    maj = tryparse(Int, parts[1]); min = tryparse(Int, parts[2])
    (maj === nothing || min === nothing) && return doc
    (maj, min) < (1, 1) && (doc["esm"] = "1.1.0")
    return doc
end

"""
    parse_jacobian_block(block) -> (wrt::Symbol, entries::Vector{JacEntry})
    parse_jacobian_block(block) -> (wrt::Tuple{Symbol,Symbol}, entries::Vector{JacEntry2})

Inverse of the block emission (round-trip surface for conformance tests).
A block whose `wrt` is a pair holds second-order entries (§4.4).
When the block carries `expression_templates` (CSE'd emission), every
`apply_expression_template` reference in the parsed coefficients is expanded
back to its body, so the returned entries are always closed expressions.
"""
function parse_jacobian_block(block::AbstractDict)
    pair = block["wrt"] isa AbstractVector
    wrt = pair ? _axis_pair(block["wrt"]) : Symbol(block["wrt"])
    entries = pair ? JacEntry2[parse_band2(d) for d in block["entries"]] :
        JacEntry[parse_band(d) for d in block["entries"]]
    if haskey(block, "expression_templates")
        reg = Dict{String, ASTExpr}(
            String(name) => expression_from_json(t["body"])
                for (name, t) in block["expression_templates"]
        )
        entries = map(en -> _with_band(en, _expand_band(en.band, reg)), entries)
    end
    return wrt, entries
end

_expand_band(b::Band, reg) = _rebuild(
    b; cidx = ASTExpr[expand_templates(c, reg) for c in b.cidx],
    coef = expand_templates(b.coef, reg)
)
_with_band(en::JacEntry, b::Band) = JacEntry(en.u, en.v, b)

"""
    parse_jacobian_blocks(x) -> Vector{Any}

[`parse_jacobian_block`](@ref) over a model's `"jacobians"` entry, which is
either one block or an array of blocks with distinct `wrt`
(esm-jacobian-spec.md §4); returns one `(wrt, entries)` per block, in order:
`(wrt::Symbol, Vector{JacEntry})` for a first-order block and
`(wrt::Tuple{Symbol,Symbol}, Vector{JacEntry2})` for a second-order pair, so
the element type is `Any` once a model mixes the two. Dispatch on `wrt`
(or on the entry type) rather than assuming the first form.
"""
function parse_jacobian_blocks(x)
    blocks = x isa AbstractDict ? [x] : collect(x)
    out = Any[parse_jacobian_block(b) for b in blocks]
    allunique(first.(out)) || throw(ArgumentError("a model's jacobian blocks must have distinct `wrt`"))
    return out
end

# ── the internal evaluation model ────────────────────────────────────────────
#
# v1 evaluation strategy (deliberate reuse of existing machinery, zero new
# evaluator code): emit a derived single-model document in which each band's
# coefficient field is a STATE `J_k` whose RHS is the band field —
#
#   D(J_k) = makearray( zero default, band region ← faq(coef) )
#
# and run it through the unmodified `EarthSciAST.build_evaluator`; one RHS
# call then leaves every band value in `du`. A dedicated compiled `jac!`
# kernel is the planned upgrade once this is a measured bottleneck.

function _base_document(sv::SysView, name::String)
    m = OrderedDict{String, Any}(
        "variables" => OrderedDict{String, Any}(),
        "equations" => Any[]
    )
    for n in sort!(collect(keys(sv.variables)))
        var = sv.variables[n]
        # esm 1.0.0 has TWO declared types. `state` / `observed` are gone, and
        # the distinction they carried is now made by the equations: see
        # `view_states` / `_observed_defs` (system.jl). Nothing is lost here
        # because an observed's definition is ALREADY an equation of `sv` and the
        # loop below copies every equation verbatim -- in 0.x it lived in the
        # variable and had to be serialized twice.
        d = OrderedDict{String, Any}(
            "type" =>
                var.type == EarthSciAST.ParameterVariable ? "parameter" : "unknown"
        )
        var.shape === nothing || isempty(var.shape) || (d["shape"] = var.shape)
        var.default === nothing || (d["default"] = _inline_value(var.default))
        m["variables"][n] = d
    end
    for eq in sv.equations
        push!(
            m["equations"], OrderedDict{String, Any}(
                "lhs" => _ser_expr(eq.lhs), "rhs" => _ser_expr(eq.rhs)
            )
        )
    end
    isets = OrderedDict{String, Any}(
        String(n) => OrderedDict{String, Any}("kind" => "interval", "size" => s.size)
            for (n, s) in sv.index_sets if s.size !== nothing
    )
    return OrderedDict{String, Any}(
        # 1.1.0, not 1.0.0: every document this file builds may carry `faq`
        # coefficient nodes, and `faq` arrives at esm 1.1.0. Declaring 1.0.0
        # makes the document unloadable (`[E_FAQ_VERSION_TOO_OLD]`).
        "esm" => "1.1.0",
        "metadata" => OrderedDict{String, Any}("name" => name),
        "index_sets" => isets,
        "models" => OrderedDict{String, Any}(name => m)
    )
end

# A variable's value in the document's inline form: an array value is
# row-major nested JSON (esm-spec §6.3), outermost list over the FIRST
# dimension. The loader hands an inline default back as a column-major Julia
# `Array`, which JSON3 would write flat, a different (and rank-less) value.
_inline_value(x) = x
_inline_value(a::AbstractVector) = Any[_inline_value(v) for v in a]
_inline_value(a::AbstractArray) =
    Any[_inline_value(collect(selectdim(a, 1, i))) for i in axes(a, 1)]

# A literal-size axis index set of the derived document (shared with the
# hoisted-observed axes: same `jcax<size>` convention, reused when present).
function _axis_name!(doc, sz::Int)
    isets = doc["index_sets"]
    nm = _jcax_name(n -> haskey(isets, n) && get(isets[n], "size", nothing) != sz, sz)
    haskey(isets, nm) ||
        (isets[nm] = OrderedDict{String, Any}("kind" => "interval", "size" => sz))
    return nm
end

# Trim the source's ODE right-hand side out of the evaluation model: every
# `D(u)` equation of the source becomes `D(u) = 0` (a zero `makearray` over
# the state's shape for an array state), in place of the first one and with
# any further ones dropped. The states keep their solver slots — `jac!`
# copies `u` into them exactly as before — so the observed definitions and
# band fields read them unchanged, but each call no longer recomputes `f`.
# Observed the coefficients do not read (directly or through other observed)
# are dropped with their declarations: they fed only `f`.
#
# Why zero derivatives rather than turning the states into parameters: an
# array parameter is a build-time `param_arrays` buffer, and EarthSciAST
# treats observed that read ONLY parameters as constant and caches them, so
# writing a new `u` into the buffer in place leaves those values stale
# (minmod_adv's limiters, off by up to 16.0 at a second point) unless an
# internal epoch hook is called on every write.
function _zero_source_derivatives!(m, sv::SysView, ctx, entries::Vector{JacEntry})
    states = Set{String}(
        u for u in (lhs_state(eq) for eq in sv.equations)
            if u !== nothing
    )
    defined = [
        lhs_state(eq) === nothing ? _defined_name(eq) : nothing
            for eq in sv.equations
    ]
    reads = Dict{String, Vector{ASTExpr}}()     # observed → its definitions' RHS
    for (eq, n) in zip(sv.equations, defined)
        (n === nothing || n in states) || push!(get!(reads, n, ASTExpr[]), eq.rhs)
    end
    live = Set{String}()
    stack = String[v for en in entries for v in free_variables(en.band.coef)]
    while !isempty(stack)
        n = pop!(stack)
        (haskey(reads, n) && !(n in live)) || continue
        push!(live, n)
        for r in reads[n]
            append!(stack, free_variables(r))
        end
    end
    done = Set{String}()
    eqs = Any[]
    for (eq, ser, n) in zip(sv.equations, m["equations"], defined)
        u = lhs_state(eq)
        if u === nothing
            _is_derivative(eq) && throw(
                BandError(
                    "unsupported derivative LHS $(JSON3.write(_ser_expr(eq.lhs))) (a state's " *
                        "derivative is `D(u)` or the lifted `faq{D(index(u, i…))}`)"
                )
            )
            if n !== nothing && haskey(reads, n) && !(n in live)
                delete!(m["variables"], n)
                continue
            end
            push!(eqs, ser)
        elseif !(u in done)
            push!(done, u)
            shape = get(ctx.shapes, u, Int[])
            rhs = isempty(shape) ? 0.0 : OrderedDict{String, Any}(
                    "op" => "makearray", "args" => Any[],
                    "regions" => Any[Any[Any[1, n] for n in shape]],
                    "values" => Any[0.0]
                )
            push!(
                eqs, OrderedDict{String, Any}(
                    "lhs" => OrderedDict{String, Any}("op" => "D", "args" => Any[u], "wrt" => "t"),
                    "rhs" => rhs
                )
            )
        end
    end
    m["equations"] = eqs
    return m
end

# Derived model: the source's observed definitions that the coefficients
# read, with its state derivatives zeroed, plus one state per band whose RHS
# is the band's coefficient field. With `cse = true`, repeated coefficient
# subtrees are first hoisted into observed variables of the document
# ([`hoist_observed`](@ref)) so the tree-walk's factored array-observed
# buffers evaluate them once per cell per RHS call — the compile-time and
# duplicate-work control for limiter-heavy (PPM-scale) coefficients.
# Returns (document, band-state names).
function _evaluation_document(
        sv::SysView, entries::Vector{JacEntry};
        cse::Bool = true, cse_min_nodes::Int = 12
    )
    doc = _base_document(sv, "JacobianEval")
    m = doc["models"]["JacobianEval"]
    ctx = _ctx(sv)
    _zero_source_derivatives!(m, sv, ctx, entries)
    h = cse ? hoist_observed(entries, sv; min_nodes = cse_min_nodes) : nothing
    coefs = h === nothing ? ASTExpr[en.band.coef for en in entries] : h.coefs
    if h !== nothing
        for (nm, sz) in h.axes
            doc["index_sets"][nm] =
                OrderedDict{String, Any}("kind" => "interval", "size" => sz)
        end
        for (nm, vd) in h.obs
            m["variables"][nm] = vd
        end
        # 1.0.0: a hoisted observed is an unknown plus a bare-LHS equation. The
        # equations go in BEFORE the J_k derivative equations below, matching the
        # declaration order the corpus migration produced -- ordering is not
        # semantically load-bearing, but a stable one keeps diffs readable.
        for (nm, rhs) in h.obs_eqs
            push!(
                m["equations"],
                OrderedDict{String, Any}("lhs" => nm, "rhs" => rhs)
            )
        end
    end
    names = String[]
    for (k, en) in enumerate(entries)
        b = en.band; nm = "J_$k"; push!(names, nm)
        shape_u = get(ctx.shapes, en.u, Int[])
        cr = b.contracted
        if isempty(shape_u) && isempty(cr)
            # an unknown that a D(.) equation targets IS an ODE state in 1.0.0;
            # the equation is pushed a few lines below.
            m["variables"][nm] = OrderedDict{String, Any}("type" => "unknown", "default" => 0.0)
            rhs = _ser_expr(coefs[k])
        else
            # The band field spans the row dims PLUS any free contracted
            # column dims — the scatter reads one slot per (row, contracted)
            # cell and ACCUMULATES into the column each cell gathers.
            ushape = isempty(shape_u) ? Any[] : Any[x for x in sv.variables[en.u].shape]
            m["variables"][nm] = OrderedDict{String, Any}(
                "type" => "unknown", "default" => 0.0,
                "shape" => vcat(ushape, Any[_axis_name!(doc, hi) for (_, _, hi) in cr])
            )
            free = [(k2, r) for (k2, r) in enumerate(b.ridx) if r isa String]
            aggnames = Any[Any[r for (_, r) in free]..., Any[n for (n, _, _) in cr]...]
            val = if isempty(aggnames)
                _ser_expr(coefs[k])
            else
                rngs = OrderedDict{String, Any}(
                    r => Any[b.rows[k2][1], b.rows[k2][2]] for (k2, r) in free
                )
                for (n, lo, hi) in cr
                    rngs[n] = Any[lo, hi]
                end
                OrderedDict{String, Any}(
                    "op" => "faq", "args" => Any[],
                    "output_idx" => aggnames,
                    "ranges" => rngs,
                    "expr" => _ser_expr(coefs[k])
                )
            end
            rhs = OrderedDict{String, Any}(
                "op" => "makearray", "args" => Any[],
                "regions" => Any[
                    vcat(
                        Any[Any[1, n] for n in shape_u],
                        Any[Any[1, hi] for (_, _, hi) in cr]
                    ),
                    vcat(
                        Any[Any[lo, hi] for (lo, hi) in b.rows],
                        Any[Any[lo, hi] for (_, lo, hi) in cr]
                    ),
                ],
                "values" => Any[0.0, val]
            )
        end
        push!(
            m["equations"], OrderedDict{String, Any}(
                "lhs" => OrderedDict{String, Any}("op" => "D", "args" => [nm], "wrt" => "t"),
                "rhs" => rhs
            )
        )
    end
    return doc, names
end
