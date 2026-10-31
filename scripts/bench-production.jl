# #4 W5, target T1: the Julia band-field evaluation per call on the documents
# `examples/bench_production.rs` times (write them with
# `cargo run --release --example bench_production -- <cells> --write-docs DIR`).
#
#     julia --project=pkg/EarthSciASTDiff.jl scripts/bench-production.jl DIR
#
# EarthSciASTDiff evaluates nothing itself: the Julia path is EarthSciAST
# running the evaluation document (`evaluation_document(file; cse = true)`).
# Median of 50 warm right-hand-side calls of that document, which leave every
# band value in `du` (placing them in a matrix is a scatter plan's work and is
# not timed), next to one call of the source RHS (`esm_problem(file, tspan).f!`).
using EarthSciAST, EarthSciASTDiff

function median_time(f; n = 50)
    for _ in 1:5
        f()
    end
    return sort!([(t0 = time_ns(); f(); (time_ns() - t0) / 1.0e3) for _ in 1:n])[n ÷ 2 + 1]
end

dir = ARGS[1]
for name in ("bd_chem", "column")
    file = EarthSciAST.load_path(joinpath(dir, name * ".esm"))
    t0 = time()
    doc, fields = evaluation_document(file; cse = true)
    fJ!, x0, pJ, _, vmJ = build_evaluator(
        EarthSciASTDiff.JSON3.read(stable_json(doc)); model_name = "JacobianEval"
    )
    tprep = time() - t0
    prob = EarthSciAST.esm_problem(file, (0.0, 1.0))
    f!, p = prob.f!, prob.p
    u = [0.3 + 0.5 * mod((i - 1) * 0.618, 1.0) for i in eachindex(prob.u0)]
    x = copy(x0)
    for (n, i) in prob.var_map
        x[vmJ[n]] = u[i]
    end
    dx = similar(x)
    tj = median_time(() -> fJ!(dx, x, pJ, 0.5))   # the parameter defaults
    du = similar(u)
    tf = median_time(() -> f!(du, u, p, 0.5))
    println(
        rpad(name, 10), " build ", round(tprep; digits = 2), " s | band fields ",
        round(tj; digits = 1), " µs | RHS ", round(tf; digits = 1), " µs | ",
        length(u), " states, ", length(fields), " band fields"
    )
end
