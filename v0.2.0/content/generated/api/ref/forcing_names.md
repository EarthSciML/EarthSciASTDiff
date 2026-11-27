---
title: "forcing_names"
description: "Stable function · Differentiate a model"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Differentiate a model](/generated/api/topics/differentiate)

| Julia | Python | Rust |
|---|---|---|
| `forcing_names` | `forcing_names` | `forcing_names` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
forcing_names(sv::SysView)
forcing_names(sv::SysView, forcing)
```

```julia
forcing_names(sv::SysView, forcing = nothing) -> Vector{String}
```

The forcing set of a system, sorted by name: the parameters whose cells are
the columns of the `wrt = :forcing` axis (esm-jacobian-spec.md §4.3). With
`forcing = nothing` the set is inferred — every ARRAY-valued parameter with an
`update` of kind `"data"` (a live buffer a data provider refills). An explicit
list is what a planner that builds the providers itself passes (a document
need not declare the update), and every name in it must be a parameter of the
system: a name that resolves to no buffer throws, never yields an empty axis.
`forcing` may be one name (a `String` or `Symbol`) or a collection of them.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/EarthSciASTDiff.jl/src/jacobian.jl#L16)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
forcing_names(sv: SysView, forcing=None) -> list[str]
```

The forcing set of a system, sorted by name: the parameters whose cells
are the columns of the `wrt="forcing"` axis (esm-jacobian-spec.md §4.3).

With `forcing=None` the set is inferred — every ARRAY-valued parameter
with an `update` of kind `"data"` (a live buffer a data provider
refills). An explicit list must name only parameters of the system: a
name that resolves to no buffer raises `ValueError`, never yields an
empty axis. `forcing` may be one name (a `str`) or a collection.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/earthsci-astdiff-py/src/earthsci_astdiff/jacobian.py#L53)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn forcing_names(sv: &SysView<'_>, forcing: Option<&[String]>) -> Result<Vec<String>, BandError>
```

The forcing set of a system, sorted by name: the parameters whose cells
are the columns of the `Wrt::Forcing` axis (spec §4.3). With `forcing =
None` the set is inferred — every ARRAY-valued parameter with an `update`
of kind `"data"` (a live buffer a data provider refills). An explicit list
is what a planner that builds the providers itself passes, and every name
in it must be a parameter of the system: a name that resolves to no buffer
is an error, never an empty axis.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/earthsci-astdiff-rs/src/jacobian.rs#L73) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/jacobian/fn.forcing_names.html)

</div>

</div>
