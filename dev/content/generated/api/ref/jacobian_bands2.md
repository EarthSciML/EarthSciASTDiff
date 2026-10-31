---
title: "jacobian_bands2"
description: "Stable function · Differentiate a model"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Differentiate a model](/generated/api/topics/differentiate)

| Julia | Python | Rust |
|---|---|---|
| `jacobian_bands2` | `jacobian_bands2` | `jacobian_bands2` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
jacobian_bands2(file::EarthSciAST.EsmFile, model_name; kw...)
jacobian_bands2(flat::EarthSciAST.FlattenedSystem; kw...)
jacobian_bands2(sv::SysView; wrt, forcing)
```

```julia
jacobian_bands2(file::EsmFile, model_name; wrt = (:states, :states), forcing = nothing) -> Vector{JacEntry2}
jacobian_bands2(flat::FlattenedSystem;    wrt = (:states, :states), forcing = nothing) -> Vector{JacEntry2}
jacobian_bands2(sv::SysView;              wrt = (:states, :states), forcing = nothing) -> Vector{JacEntry2}
```

Second-order bands for the axis pair `wrt = (a1, a2)`, `a1 ∈ (:states, :time)`
and `a2 ∈ (:states, :parameters, :forcing)`: every first-order entry of the
`a1` axis ([`jacobian_bands`](/generated/api/ref/jacobian_bands)) with its coefficient differentiated by
the same calculus with respect to every site of every `a2` target, broadcast
over the entry's rows with region clipping and re-merging. No new rules: the
non-smooth policy (spec §3.2) applies again, and the pattern is the global
structural union over branches.

`(r, c, s)` and `(r, s, c)` come out independently, from distinct first-order
entries; consumers must not assume either is deduplicated. On the
`:parameters` and `:forcing` second axes every `col2_idx` must be closed at
build time, as on the first-order axes (esm-jacobian-spec.md §4.3).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/EarthSciASTDiff.jl/src/second_order.jl#L43)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
jacobian_bands2(obj, model_name: str | None = None, wrt=('states', 'states'), forcing=None) -> list[JacEntry2]
```

Second-order bands for the axis pair `wrt = (a1, a2)`, `a1` in
`("states", "time")` and `a2` in `("states", "parameters", "forcing")`: every first-order entry of the `a1` axis
([`jacobian_bands`](/generated/api/ref/jacobian_bands)) with its coefficient differentiated by the same
calculus with respect to every site of every `a2` target, broadcast
over the entry's rows with region clipping and re-merging. No new rules:
the non-smooth policy (spec §3.2) applies again, and the pattern is the
global structural union over branches.

`(r, c, s)` and `(r, s, c)` come out independently, from distinct
first-order entries; consumers must not assume either is deduplicated.
On the `"parameters"` and `"forcing"` second axes every second column
index must be closed at build time (spec §4.3), else [`BandError`](/generated/api/ref/band_error).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-py/src/earthsci_astdiff/second_order.py#L76)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn jacobian_bands2(file: &EsmFile, model_name: Option<&str>, wrt: (Wrt, Wrt)) -> Result<Vec<JacEntry2>, BandError>
```

Second-order bands for the axis pair `wrt = (a1, a2)`, `a1` states or
time and `a2` states, parameters or forcing: every first-order entry of
the `a1` axis ([`crate::jacobian_bands`](/generated/api/ref/jacobian_bands)) with its coefficient
differentiated by the same calculus with respect to every site of every
`a2` target, broadcast over the entry's rows with region clipping and
re-merging. No new rules: the non-smooth policy (spec §3.2) applies
again, and the pattern is the global structural union over branches.

`(r, c, s)` and `(r, s, c)` come out independently, from distinct
first-order entries; consumers must not assume either is deduplicated.
The forcing set is inferred (see `jacobian_bands2_with_forcing`).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-rs/src/second_order.rs#L86) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/second_order/fn.jacobian_bands2.html)

</div>

</div>
