---
title: "hvp_document"
description: "Stable function · Evaluate the Jacobian"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Evaluate the Jacobian](/generated/api/topics/evaluate)

| Julia | Python | Rust |
|---|---|---|
| `hvp_document` | `hvp_document` | `hvp_document` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
hvp_document(input; stages, wrt, form, model_name, forcing, cse, cse_min_nodes)
```

```julia
hvp_document(input; stages, wrt = (:states, :states), form = :contract,
             model_name = nothing, forcing = nothing, cse = true) -> (doc, io)
```

The second-order product document of esm-jacobian-spec.md §9.4 for the
axis pair `wrt` and `stages` stage pairs: its right-hand side leaves
`g = ∇_{a2} Σᵢ μᵢᵀ J(u) wᵢ` (`∇_{a2} (Σᵢ wᵢ μᵢ)ᵀ f_t` on a time pair) in
the derivative slots of its outputs. `io.mu[i]` / `io.w[i]` map each state
(each first-axis column variable; `"t"` on a time pair) to stage `i`'s
inputs and `io.hvp` each second-axis column variable to its output.
`form = :symmetric` (a `(:states, :states)` pair only) scatters by the
first column instead (model `"HvpEvalSymmetric"`); both forms give the same
`g`.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/EarthSciASTDiff.jl/src/products.jl#L559)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
hvp_document(obj, stages: int, model_name: str | None = None, wrt=('states', 'states'), form: str = 'contract', forcing=None, cse: bool = True, cse_min_nodes: int = 12)
```

The second-order product document of spec §9.4 for the axis pair
`wrt` and `stages` stage pairs: its right-hand side leaves
`g = ∇_{a2} Σᵢ μᵢᵀ J(u) wᵢ` (`∇_{a2} (Σᵢ wᵢ μᵢ)ᵀ f_t` on a time pair)
in the derivative slots of its outputs.

Returns `(doc, io)`: `io["mu"][i]` / `io["w"][i]` map each state
(each first-axis column variable; `"t"` on a time pair) to stage
`i`'s inputs and `io["hvp"]` each second-axis column variable to its
output. `form="symmetric"` (a `("states", "states")` pair only)
scatters by the first column instead (model `"HvpEvalSymmetric"`);
both forms give the same `g`.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/earthsci-astdiff-py/src/earthsci_astdiff/products.py#L996)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn hvp_document(sv: &SysView<'_>, entries: &[JacEntry2], wrt: (Wrt, Wrt), stages: usize, form: HvpForm, opts: &ProductOptions<'_>) -> Result<(Value, HvpIo), BandError>
```

The second-order product document of spec §9.4 for the axis pair `wrt`,
the second-order `entries` (as [`crate::jacobian_bands2`](/generated/api/ref/jacobian_bands2) gives them) and
`stages` stage pairs: its right-hand side leaves `g = ∇_{a2} Σᵢ μᵢᵀ J(u)
wᵢ` (`∇_{a2} (Σᵢ wᵢ μᵢ)ᵀ f_t` on a time pair) in the derivative slots of
its outputs. `HvpForm::Symmetric` (a `(states, states)` pair only)
scatters by the first column instead (model `"HvpEvalSymmetric"`).
`entries` and `opts.forcing` must agree, as for [`jvp_document`](/generated/api/ref/jvp_document).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/earthsci-astdiff-rs/src/products.rs#L1011) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/products/fn.hvp_document.html)

</div>

</div>
