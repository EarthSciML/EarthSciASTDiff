# Regenerate tests/goldens/*.jvals.json — assembled Jacobian VALUES at a
# pinned state, produced by the Julia reference. Other bindings' numeric
# assemblies compare against these with a tolerance (evaluation order makes
# bit-equality across runtimes unrealistic), and must produce EXACTLY the
# same (row, col) key set. Run after an intended rule change, like
# regenerate-goldens.jl.
using Pkg
using EarthSciAST, EarthSciASTDiff, OrderedCollections, SparseArrays
root = normpath(joinpath(@__DIR__, ".."))
# `<name>.<axis>.jvals.json` for the other axes: wrt = :time (the single
# column `t`) at the same pinned point, and the forcing / parameter axes of
# the forcing fixture (canonical column space, spec §4) at two points of its
# frac-blend weight w = t - floor(t): t = 0.37 (`w = 0.37`) and t = 2.0
# (`.w0.`, w = 0 exactly: both bracket records still read, the second with a
# zero coefficient). The buffers and the array parameter take the document
# defaults, so the golden needs no array values of its own.
const STATE_NAMES = (
    "bd_chem", "adv_interior", "contracted_ops", "flux_form_adv",
    "minmod_adv", "contracting_observed", "tgrad_photolysis", "forcing_bracket",
)
for (name, wrt, t, tag) in Iterators.flatten(
        (
            ((n, :states, 0.7, "") for n in STATE_NAMES),
            (("tgrad_photolysis", :time, 0.7, ".time"),),
            (
                ("forcing_bracket", w, t, ".$w$sfx") for w in (:forcing, :parameters)
                    for (t, sfx) in ((0.37, ""), (2.0, ".w0"))
            ),
        )
    )
    file = EarthSciAST.load_path(joinpath(root, "tests", "valid", "$name.esm"))
    mname = first(keys(file.models))
    f!, u0, p0, _, vm = build_evaluator(file)
    n = length(u0)
    u = [0.2 + 0.6 * abs(sin(2.3 * i)) for i in 1:n]
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
    path = joinpath(root, "tests", "goldens", "$name$tag.jvals.json")
    write(path, stable_json(doc))
    println("wrote ", path, " (", length(entries), " stored entries)")
end
