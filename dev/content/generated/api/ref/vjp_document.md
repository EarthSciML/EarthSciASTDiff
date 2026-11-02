---
title: "vjp_document"
description: "Stable function · Evaluate the Jacobian"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Evaluate the Jacobian](/generated/api/topics/evaluate)

| Julia | Python | Rust |
|---|---|---|
| `vjp_document` | `vjp_document` | `vjp_document` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
vjp_document(input; wrt, model_name, forcing, cse, cse_min_nodes)
```

```julia
jvp_document(input; wrt = :states, model_name = nothing, forcing = nothing,
             cse = true) -> (doc, io)
vjp_document(input; …) -> (doc, io)
```

The product documents of esm-jacobian-spec.md §9.2 / §9.3 for one
first-order axis: a derived ESM document (model `"JvpEval"` / `"VjpEval"`)
whose right-hand side, evaluated by an unmodified EarthSciAST evaluator at
`(u, inputs, p, t)`, leaves `J v` (resp. `Jᵀ λ`) in the derivative slots of
its outputs. `input` is an `EsmFile` or a `FlattenedSystem`, as for
[`jacobian_bands`](/generated/api/ref/jacobian_bands).

`io` names the inputs and outputs: `io.v` maps each column variable to its
direction input and `io.jvp` each state to its output (`io.lam` / `io.vjp`
for `vjp_document`). The consumer writes the inputs into the evaluator's
state slots beside the source states and reads the outputs from `du`
through the evaluator's variable map.

A `VjpEval` document is an equality-filtered `faq` scatter: correct under
any evaluator, linear-time only under one that implements §9.3's
equality-filter scatter.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/EarthSciASTDiff.jl/src/products.jl#L519)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
vjp_document(obj, model_name: str | None = None, wrt: str = 'states', forcing=None, cse: bool = True, cse_min_nodes: int = 12)
```

The `Jᵀ λ` product document of spec §9.3 for one first-order axis
(model `"VjpEval"`), as [`jvp_document`](/generated/api/ref/jvp_document): `io["lam"]` maps each
state to its adjoint input and `io["vjp"]` each column variable to its
output.

A `VjpEval` document is an equality-filtered `faq` scatter: correct
under any evaluator, linear-time only under one that implements §9.3's
equality-filter scatter.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/earthsci-astdiff-py/src/earthsci_astdiff/products.py#L973)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn vjp_document(sv: &SysView<'_>, entries: &[JacEntry], wrt: Wrt, opts: &ProductOptions<'_>) -> Result<(Value, VjpIo), BandError>
```

The `Jᵀ λ` product document of spec §9.3 (model `"VjpEval"`): as
[`jvp_document`](/generated/api/ref/jvp_document), with the cotangent inputs `VjpIo::lam` and the
outputs `VjpIo::vjp`. An equality-filtered `faq` scatter: correct under
any evaluator, linear-time only under one that implements §9.3's
equality-filter scatter. `entries` and `opts.forcing` must agree, as for
[`jvp_document`](/generated/api/ref/jvp_document), which also says what to pass through a recurrence.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/earthsci-astdiff-rs/src/products.rs#L952) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/products/fn.vjp_document.html)

</div>

</div>
