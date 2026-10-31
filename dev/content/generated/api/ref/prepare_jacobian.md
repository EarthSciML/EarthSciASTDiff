---
title: "prepare_jacobian"
description: "Stable function · Evaluate the Jacobian"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Evaluate the Jacobian](/generated/api/topics/evaluate)

| Julia | Python | Rust |
|---|---|---|
| `prepare_jacobian` | `prepare_jacobian` | `prepare_jacobian` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
prepare_jacobian(input; wrt, model_name, forcing, build_kwargs, source_layout, eval_cse, eval_cse_min_nodes)
```

```julia
prepare_jacobian(input; wrt = :states, model_name = nothing, forcing = nothing) -> JacobianEvaluator
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
> bands in the wrong cells with no visible symptom. `p` may be omitted:
> parameter and forcing columns come from the system's declared shapes.
> The default, `nothing`, rebuilds.
>
> jac = prepare_jacobian(file)
> J = copy(jac.prototype)
> jac(J, u, p, t)              # fills J in place; stored zeros stay

On the `:parameters` and `:forcing` axes the columns are the canonical column
space of [`jacobian_columns`](/generated/api/ref/jacobian_columns) (`jac.columns`), and `jac.structure` is
`nothing` (the classes describe square matrices). A forcing column that does
not resolve to a cell throws; a parameter column outside its extent is a
ghost read, skipped as on the state axis. Forcing buffers are bound like any
other array parameter of the derived band model: through
`build_kwargs = (param_arrays = buffers,)` they are read live, by reference
(after writing one in place, call `EarthSciAST.notify_forcing_refresh!(buf)`
as for the RHS); without it the declared defaults are used. [`vjp!`](/generated/api/ref/vjp) /
[`vjp_forcing!`](/generated/api/ref/vjp_forcing) apply the transposed bands without forming `J`.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/EarthSciASTDiff.jl/src/assemble.jl#L281)

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

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/docs/api/examples/julia/jacobians.jl#L41-L48">docs/api/examples/julia/jacobians.jl</a></p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
prepare_jacobian(obj, model_name: str | None = None, wrt: str = 'states', forcing=None) -> JacobianEvaluator
```

Build a [`JacobianEvaluator`](/generated/api/ref/jacobian_evaluator) once — differentiation, column
space and point placement — for repeated evaluation at different points
(an adjoint loop calling [`vjp_forcing`](/generated/api/ref/vjp_forcing) once per stage).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-py/src/earthsci_astdiff/numeric.py#L228)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn prepare_jacobian(file: &EsmFile, model_name: Option<&str>, wrt: Wrt, forcing: Option<&[String]>) -> Result<JacobianEvaluator, BandError>
```

Build a [`JacobianEvaluator`](/generated/api/ref/jacobian_evaluator) once: differentiation, observed inlining,
the column space (parameters and forcing axes) and the parameter defaults.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-rs/src/numeric.rs#L442) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/numeric/fn.prepare_jacobian.html)

</div>

</div>
