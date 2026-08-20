#!/usr/bin/env julia
# Scaling benchmark for the SYMBOLIC step, `jacobian_bands`.
#
# The band calculus is cheap per node and expensive per PASS OVER A NODE, so
# the thing to watch is not absolute time at one size but the EXPONENT: a
# document flattened from templates over index sets grows its per-equation
# RHS linearly with the index sets (`flatten` materializes an operator's
# stencil per cell), and any pass that is Θ(n·depth) rather than Θ(n) stops
# finishing somewhere between a toy grid and a real one. Both modes below
# therefore run a LADDER and print the fitted exponent.
#
#   --synthetic                 no document needed: a generated region-`ifelse`
#                               chain, the exact shape `index(makearray…)`
#                               lowering produces, timed per primitive.
#   --esm PATH                  a real document. Repeat --metaparameters once
#                               per rung; `--include`/`--select` reach a
#                               sub-system (e.g. one half of a split).
#
# Examples
#   julia --project=pkg/EarthSciASTDiff.jl scripts/bench-jacobian-bands.jl --synthetic
#   julia --project=<env> scripts/bench-jacobian-bands.jl \
#       --esm /path/reseact.esm \
#       --metaparameters '{"LON0":11,"LAT0":29,"NLON":6,"NLAT":6,"NLEV":8}' \
#       --metaparameters '{"LON0":11,"LAT0":29,"NLON":6,"NLAT":6,"NLEV":12}'

using Printf, JSON3
using EarthSciAST
using EarthSciAST: ASTExpr, NumExpr, IntExpr, VarExpr, OpExpr, map_children
using EarthSciASTDiff
const EA = EarthSciAST
const ED = EarthSciASTDiff

say(s) = (println(s); flush(stdout))
rssgib() = Sys.maxrss() / 2^30
nodes(e) = (n = Ref(0); EA.foreach_subexpr(x -> (n[] += 1), e); n[])

# Least-squares slope of log(t) vs log(size) — 1.0 is linear, 2.0 quadratic.
function exponent(sizes, times)
    ok = [i for i in eachindex(sizes) if sizes[i] > 0 && times[i] > 0]
    length(ok) < 2 && return NaN
    x = [log(Float64(sizes[i])) for i in ok]; y = [log(times[i]) for i in ok]
    xm = sum(x)/length(x); ym = sum(y)/length(y)
    return sum((x .- xm) .* (y .- ym)) / max(sum((x .- xm).^2), eps())
end

function report_ladder(label, sizes, times)
    say("")
    say("$label   (size = RHS nodes)")
    say(@sprintf("  %10s %12s %10s", "size", "seconds", "s/node"))
    for i in eachindex(sizes)
        say(@sprintf("  %10d %12.3f %10.2e", sizes[i], times[i],
                     times[i]/max(sizes[i], 1)))
    end
    say(@sprintf("  fitted exponent d log t / d log n = %.2f", exponent(sizes, times)))
end

# ── synthetic mode ───────────────────────────────────────────────────────────
#
# `k` region guards nested exactly as `_index_makearray` (inline.jl) nests
# them, each region's value a small stencil. Everything the ladder varies is
# `k`, so `size` and the cost per pass move together and the exponent is read
# straight off.
#
# Only ONE region reads the differentiation target `u`; the rest read `w`.
# That is the realistic shape — a flattened system has one state per species
# and a region per cell, so any one target is absent from almost every region
# — and it is the shape that exercises the occurrence scan, which can only
# stop early where the site IS present.
function synthetic_body(k::Int; target_region::Int = 1)
    i = VarExpr("i")
    rd(nm, off) = OpExpr("index", ASTExpr[VarExpr(nm),
                         off == 0 ? i : OpExpr("+", ASTExpr[i, IntExpr(off)])])
    body = IntExpr(0)
    for r in 1:k
        nm = r == target_region ? "u" : "w"
        cond = OpExpr("and", ASTExpr[
            OpExpr(">=", ASTExpr[i, IntExpr(r)]),
            OpExpr("<=", ASTExpr[i, IntExpr(r)])])
        val = OpExpr("*", ASTExpr[
            OpExpr("-", ASTExpr[rd(nm, 1), rd(nm, 0)]),
            OpExpr("max", ASTExpr[rd(nm, 0), NumExpr(1e-30)])])
        body = OpExpr("ifelse", ASTExpr[cond, val, body])
    end
    return body
end

# `b` bands over adjacent row ranges. Coefficient SIZE is held fixed while
# `b` grows, so the ladder isolates the pairwise SEARCH from the cost of
# keying bigger coefficients: half the bands share a key (and merge, as after
# a clip split), half do not (and the search has to discover that).
function synthetic_bands(b::Int, site)
    base = ED.dscalar(synthetic_body(4; target_region = 1), site)
    out = ED.Band[]
    for j in 1:b
        coef = OpExpr("*", ASTExpr[IntExpr(1 + (j ÷ 2) % max(b ÷ 2, 1)), base])
        push!(out, ED.Band(Tuple{Int,Int}[(j, j)], Union{String,Int}["i"],
                           ASTExpr[VarExpr("i")], coef))
    end
    return out
end

function run_synthetic(ks)
    say("=== synthetic region-chain ladder ===")
    site = ED.Site(OpExpr("index", ASTExpr[VarExpr("u"), VarExpr("i")]))
    # Warm up every code path so the first rung is not a compile measurement.
    let w = synthetic_body(4)
        ED.merge_bands(synthetic_bands(4, site))
        ED.simplify_branches(ED.dscalar(w, site))
    end
    sizes = Int[]; td = Float64[]; ts = Float64[]; tm = Float64[]; bs = Int[]
    for k in ks
        b = synthetic_body(k)
        n = nodes(b); push!(sizes, n)
        GC.gc()
        t = time(); d = ED.dscalar(b, site);               push!(td, time() - t)
        t = time(); ED.simplify_branches(d);               push!(ts, time() - t)
        bands = synthetic_bands(k, site); push!(bs, length(bands))
        GC.gc()
        t = time(); ED.merge_bands(bands);                 push!(tm, time() - t)
        say(@sprintf("k=%-6d nodes=%-9d dscalar=%8.3f s  simplify_branches=%8.3f s  merge_bands(%d)=%8.3f s",
                     k, n, td[end], ts[end], bs[end], tm[end]))
    end
    report_ladder("dscalar", sizes, td)
    report_ladder("simplify_branches", sizes, ts)
    report_ladder("merge_bands", bs, tm)
    return nothing
end

# ── document mode ────────────────────────────────────────────────────────────

function load_system(esm, mp, incl, select)
    for f in incl
        Base.include(Main, f)
    end
    file = EA.load(esm; metaparameters = mp)
    flat = EA.promote_downstream_shapes(EA.algebraic_states_to_observeds(EA.flatten(file)))
    select === nothing && return flat
    return Base.invokelatest(Main.eval, :(let flat = $flat; $(Meta.parse(select)); end))
end

function run_document(esm, mps, incl, select, wrt)
    say("=== document ladder: $esm ===")
    sizes = Int[]; times = Float64[]
    for mp in mps
        t = time(); src = load_system(esm, mp, incl, select); tload = time() - t
        sv = ED.sysview(src)
        deqs = [eq for eq in sv.equations if ED.lhs_state(eq) !== nothing]
        n = sum(nodes(eq.rhs) for eq in deqs; init = 0)
        GC.gc()
        t = time(); entries = ED.jacobian_bands(src; wrt = wrt); dt = time() - t
        push!(sizes, n); push!(times, dt)
        say(@sprintf("mp=%s  load=%.1f s  D-eqs=%d  rhs nodes=%d  jacobian_bands=%.2f s  bands=%d  peak rss=%.2f GiB",
                     JSON3.write(mp), tload, length(deqs), n, dt, length(entries), rssgib()))
    end
    report_ladder("jacobian_bands", sizes, times)
    return nothing
end

function main(args)
    esm = nothing; mps = Any[]; incl = String[]; select = nothing
    wrt = :states; synth = false; ks = [50, 100, 200, 400]
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--synthetic";           synth = true; i += 1
        elseif a == "--k";               ks = parse.(Int, split(args[i+1], ",")); i += 2
        elseif a == "--esm";             esm = args[i+1]; i += 2
        elseif a == "--metaparameters"   # metaparameters are integers by spec
            push!(mps, Dict{String,Int}(String(k) => Int(v)
                                        for (k, v) in JSON3.read(args[i+1], Dict{String,Any})))
            i += 2
        elseif a == "--include";         push!(incl, args[i+1]); i += 2
        elseif a == "--select";          select = args[i+1]; i += 2
        elseif a == "--wrt";             wrt = Symbol(args[i+1]); i += 2
        else error("unknown argument $a")
        end
    end
    synth && run_synthetic(ks)
    esm === nothing || run_document(esm, isempty(mps) ? [Dict{String,Int}()] : mps,
                                    incl, select, wrt)
    (synth || esm !== nothing) ||
        error("nothing to do: pass --synthetic and/or --esm PATH")
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
