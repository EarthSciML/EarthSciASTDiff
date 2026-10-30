# Regenerate tests/goldens/*.jvals.json — assembled Jacobian VALUES at a
# pinned state, produced by the Julia reference. Other bindings' numeric
# assemblies compare against these with a tolerance (evaluation order makes
# bit-equality across runtimes unrealistic), and must produce EXACTLY the
# same (row, col) key set. Run after an intended rule change, like
# regenerate-goldens.jl.
using Pkg
using EarthSciAST, EarthSciASTDiff, OrderedCollections, SparseArrays
root = normpath(joinpath(@__DIR__, ".."))
# `<name>.time.jvals.json` (wrt = :time, the single column `t`) for the
# fixtures in `TIME`, at the same pinned point.
const TIME = ("tgrad_photolysis",)
for (name, wrt) in Iterators.flatten(
        (
            (
                (n, :states) for n in (
                        "bd_chem", "adv_interior", "contracted_ops", "flux_form_adv",
                        "minmod_adv", "contracting_observed", "tgrad_photolysis",
                    )
            ),
            ((n, :time) for n in TIME),
        )
    )
    file = EarthSciAST.load_path(joinpath(root, "tests", "valid", "$name.esm"))
    mname = first(keys(file.models))
    f!, u0, p0, _, vm = build_evaluator(file)
    n = length(u0)
    u = [0.2 + 0.6 * abs(sin(2.3 * i)) for i in 1:n]
    t = 0.7
    # The native compiler declines a derived model whose coefficients carry
    # a sum (`compiler_refused_rule`); the interpreter evaluates them.
    kw = name in ("contracted_ops", "contracting_observed") ?
        (build_kwargs = (compiler = :interpreter,),) : NamedTuple()
    res = assemble_jacobian(file; wrt = wrt, u = u, t = t, kw...)
    rn, cn = res.rownames, res.colnames
    entries = Any[]
    I, J, V = findnz(res.J)
    for k in eachindex(I)
        push!(entries, Any[rn[I[k]], cn[J[k]], V[k]])
    end
    sort!(entries; by = e -> (e[1], e[2]))
    doc = OrderedDict{String, Any}(
        "t" => t,
        "u" => OrderedDict{String, Any}(rn[i] => u[i] for i in 1:n),
        "params" => p0 === nothing ? OrderedDict{String, Any}() :
            OrderedDict{String, Any}(String(k) => v for (k, v) in pairs(p0)),
        "entries" => entries
    )
    path = joinpath(
        root, "tests", "goldens",
        wrt == :time ? "$name.time.jvals.json" : "$name.jvals.json"
    )
    write(path, stable_json(doc))
    println("wrote ", path, " (", length(entries), " stored entries)")
end
