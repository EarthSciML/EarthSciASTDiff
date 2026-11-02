---
title: "prepare_jacobian"
description: "Stable function · Evaluate the Jacobian"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Evaluate the Jacobian](/generated/api/topics/evaluate)

| Julia | Python | Rust |
|---|---|---|
| `prepare_jacobian` | not exported | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
prepare_jacobian(input; wrt, model_name, build_kwargs, source_layout, eval_cse, eval_cse_min_nodes)
```

```julia
prepare_jacobian(input; wrt = :states, model_name = nothing) -> JacobianEvaluator
```

Build the analytical Jacobian evaluator once: differentiates the system,
compiles the derived band-field model through the ordinary
`EarthSciAST.build_evaluator`, and precomputes the (row, col) → nzval scatter
map on the fixed structural pattern. `build_kwargs` (a NamedTuple) is
forwarded to both `build_evaluator` calls — pass `(const_arrays = …,)` etc.
for models that need build-time bindings.

The second of those two builds exists only to learn the source model's state
layout — `length(u0)`, the parameter names and the name → slot map — and its
RHS is thrown away. On a real grid it is about HALF this call's wall time (a
share that grows with the grid, because the band build is grid-independent and
the source build is not). A caller that has already built the source evaluator
can delete that build by passing its layout:

    # u0, p, var_map: build_evaluator's 2nd, 3rd and 5th returns
    source_layout = (u0 = u0, p = p, var_map = var_map)

> **Contract**
>
> The layout MUST come from a `build_evaluator` of the SAME source (same
> document / `FlattenedSystem`, same `model_name`) with the SAME
> `build_kwargs`. Only the cheap parts of that are checked (the state count,
> that the slots are a permutation, that every name resolves in the derived
> band model); a layout from a build that merely *differs* — other
> `const_arrays`, other `parameter_overrides`, another grid — would put
> bands in the wrong cells with no visible symptom. `p` may be omitted
> unless `wrt` selects parameter columns. The default, `nothing`, rebuilds.
>
> jac = prepare_jacobian(file)
> J = copy(jac.prototype)
> jac(J, u, p, t)              # fills J in place; stored zeros stay

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/EarthSciASTDiff.jl/src/assemble.jl#L218)

**Example** (`prepare`)

```julia
jac = prepare_jacobian(file)                 # differentiate and compile once
J = jac_prototype(jac)                       # sparse matrix on the fixed pattern
u = fill(2.0, 12)
p = jac.p_default                            # the document's parameter values
jac(J, u, p, 0.0)                            # fill J in place, at any (u, p, t)
@assert size(J) == (12, 12)
@assert nnz(J) == 24
@assert J[1, 1] == -0.04
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/docs/api/examples/julia/jacobians.jl#L41-L48">docs/api/examples/julia/jacobians.jl</a></p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

<p class="api-absent">Python does not export this function.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this function.</p>

</div>

</div>
