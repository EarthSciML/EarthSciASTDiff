# Regenerate tests/goldens/*.jvals.json — Jacobian VALUES at a pinned state —
# and *.hvals.json. The values come from EarthSciAST evaluating the Julia
# reference's product documents (pkg/EarthSciASTDiff.jl/test/docvals.jl): this
# package evaluates nothing itself. Other bindings compare against these with
# a tolerance (evaluation order makes bit-equality across runtimes
# unrealistic), and must produce EXACTLY the same (row, col) key set. Run
# after an intended rule change, like regenerate-goldens.jl.
using Pkg
using EarthSciAST, EarthSciASTDiff, OrderedCollections, SparseArrays
root = normpath(joinpath(@__DIR__, ".."))
include(joinpath(root, "pkg", "EarthSciASTDiff.jl", "test", "docvals.jl"))
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
    "filtered_scan", "semiring_reduce",
)
# The native compiler declines a derived model whose coefficients carry a sum,
# a filter or a non-`+` reduction (`compiler_refused_rule`); the interpreter
# evaluates them.
const INTERPRETED = ("contracted_ops", "contracting_observed", "filtered_scan", "semiring_reduce")
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
    bkw = name in INTERPRETED ? (compiler = :interpreter,) : NamedTuple()
    f!, u0, p0, _, vm = build_evaluator(file; bkw...)
    n = length(u0)
    u = [0.2 + 0.6 * abs(sin(2.3 * i)) for i in 1:n]
    res = doc_jacobian(file; wrt = wrt, u = u, t = t, build_kwargs = bkw)
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
# The second columns a point of the pair's bands reaches with both of its
# columns inside their extents (a ghost point reaches nothing): structural,
# read off the bands by a point walk.
function _reached2(file, wrt)
    sv = sysview(file, first(keys(file.models)))
    shapes = EarthSciASTDiff._ctx(sv).shapes
    inext(v, c) = all(1 <= c[d] <= get(shapes, v, Int[])[d] for d in eachindex(c))
    out = Set{String}()
    for en in jacobian_bands2(sv; wrt = wrt)
        EarthSciASTDiff._foreach_point(en.band) do _, _, c
            c1, c2 = c[1:(en.ncol)], c[(en.ncol + 1):end]
            inext(en.v, c1) && inext(en.w, c2) &&
                push!(out, EarthSciASTDiff._cellname(en.w, c2))
        end
    end
    return out
end
# `<name>.<a1>-<a2>.hvals.json`: the second-order contraction (spec §4.4)
# `g = ∇_{a2} Σᵢ μᵢᵀ J wᵢ` (on a time pair `∇_{a2} (Σᵢ wᵢ μᵢ)ᵀ f_t`) at the
# pinned state and two pinned stage pairs, all carried in the golden by name.
# `g` lists every second column a band point reaches (the exact key set),
# sorted by name. Parameters take the document defaults.
for (name, wrt, t) in (
        ("bd_chem", (:states, :states), 0.7),
        ("arrhenius_chem", (:states, :states), 0.7),
        ("arrhenius_chem", (:states, :parameters), 0.7),
        ("forcing_bracket", (:states, :forcing), 0.37),
        ("forcing_bracket", (:time, :states), 0.37),
        ("minmod_adv", (:states, :states), 0.7),
        ("filtered_scan", (:states, :states), 0.7),
        ("semiring_reduce", (:states, :states), 0.7),
    )
    file = EarthSciAST.load_path(joinpath(root, "tests", "valid", "$name.esm"))
    bkw = name in INTERPRETED ? (compiler = :interpreter,) : NamedTuple()
    rn = _doc_source(file, nothing, bkw).rownames
    cn = wrt[1] == :time ? ["t"] : rn
    u = [0.2 + 0.6 * abs(sin(2.3 * i)) for i in eachindex(rn)]
    ps = [
        (
            [sin(1.7 * k + 0.9 * s) for k in eachindex(rn)],
            wrt[1] == :time ? 0.5 + 0.25 * s : [cos(0.6 * k + 1.1 * s) for k in eachindex(cn)],
        ) for s in 1:2
    ]
    g, gn = doc_hvp(file, ps; wrt = wrt, u = u, t = t, build_kwargs = bkw)
    reached = sort!(
        [findfirst(==(c), gn) for c in _reached2(file, wrt)]; by = c -> gn[c]
    )
    doc = OrderedDict{String, Any}(
        "wrt" => Any[String(wrt[1]), String(wrt[2])],
        "t" => t,
        "u" => OrderedDict{String, Any}(rn[i] => u[i] for i in eachindex(rn)),
        "pairs" => Any[
            OrderedDict{String, Any}(
                "mu" => OrderedDict{String, Any}(rn[i] => μ[i] for i in eachindex(rn)),
                "w" => w isa Number ? w : OrderedDict{String, Any}(cn[i] => w[i] for i in eachindex(cn)),
            ) for (μ, w) in ps
        ],
        "g" => Any[Any[gn[c], g[c]] for c in reached],
    )
    path = joinpath(root, "tests", "goldens", "$name.$(wrt[1])-$(wrt[2]).hvals.json")
    write(path, stable_json(doc))
    println("wrote ", path, " (", length(reached), " second columns)")
end
