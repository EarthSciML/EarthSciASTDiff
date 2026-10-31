---
title: "parse_jacobian_blocks"
description: "Stable function · Serialize as a document block"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Serialize as a document block](/generated/api/topics/serialize)

| Julia | Python | Rust |
|---|---|---|
| `parse_jacobian_blocks` | `parse_jacobian_blocks` | `parse_jacobian_blocks` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
parse_jacobian_blocks(x)
```

```julia
parse_jacobian_blocks(x) -> Vector{Any}
```

`parse_jacobian_block` over a model's `"jacobians"` entry, which is
either one block or an array of blocks with distinct `wrt`
(esm-jacobian-spec.md §4); returns one `(wrt, entries)` per block, in order:
`(wrt::Symbol, Vector{JacEntry})` for a first-order block and
`(wrt::Tuple{Symbol,Symbol}, Vector{JacEntry2})` for a second-order pair, so
the element type is `Any` once a model mixes the two. Dispatch on `wrt`
(or on the entry type) rather than assuming the first form.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/EarthSciASTDiff.jl/src/emit.jl#L281)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
parse_jacobian_blocks(x) -> list
```

`parse_jacobian_block` over a model's `jacobians` entry, which
is either one block or a list of blocks with distinct `wrt` (spec §4);
returns one `(wrt, entries)` per block, in order (`wrt` a str, or a
tuple for a second-order pair).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-py/src/earthsci_astdiff/emit.py#L300)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn parse_jacobian_blocks(x: &Value) -> Result<Vec<ParsedBlock>, BandError>
```

`parse_jacobian_block` / `parse_jacobian_block2` over a model's
`jacobians` entry, which is either one block or an array of blocks with
distinct `wrt` (spec §4); one `ParsedBlock` per block, in order.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-rs/src/emit.rs#L856) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/emit/fn.parse_jacobian_blocks.html)

</div>

</div>
