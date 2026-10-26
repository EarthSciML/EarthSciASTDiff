# #4 W5, target T1: the Julia reference's `jac!` per call on the documents
# `examples/bench_production.rs` times (write them with
# `cargo run --release --example bench_production -- <cells> --write-docs DIR`).
#
#     julia --project=pkg/EarthSciASTDiff.jl scripts/bench-production.jl DIR
#
# Median of 50 warm calls of `jac!(J, u, p, t)` from
# `prepare_jacobian(file; eval_cse = true)`, CSC scatter included, next to
# one call of the source RHS (`esm_problem(file, tspan).f!`).
using EarthSciAST, EarthSciASTDiff

function median_time(f; n = 50)
    for _ in 1:5; f(); end
    sort!([(t0 = time_ns(); f(); (time_ns() - t0) / 1e3) for _ in 1:n])[n ÷ 2 + 1]
end

dir = ARGS[1]
for name in ("bd_chem", "column")
    file = EarthSciAST.load_path(joinpath(dir, name * ".esm"))
    t0 = time()
    jac = prepare_jacobian(file; eval_cse = true)
    tprep = time() - t0
    prob = EarthSciAST.esm_problem(file, (0.0, 1.0))
    f!, p = prob.f!, prob.p
    u = [0.3 + 0.5 * mod((i - 1) * 0.618, 1.0) for i in eachindex(prob.u0)]
    J = copy(jac.prototype)
    tj = median_time(() -> jac(J, u, nothing, 0.5))   # the parameter defaults
    du = similar(u)
    tf = median_time(() -> f!(du, u, p, 0.5))
    println(rpad(name, 10), " prepare_jacobian ", round(tprep; digits = 2), " s | jac! ",
            round(tj; digits = 1), " µs | RHS ", round(tf; digits = 1), " µs | ",
            length(u), " states, nnz ", length(J.nzval))
end
