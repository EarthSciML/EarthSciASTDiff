---
title: "serialize_plan"
description: "Stable function · Sparsity and structure"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Sparsity and structure](/generated/api/topics/structure)

| Julia | Python | Rust |
|---|---|---|
| `serialize_plan` | `serialize_plan` | `serialize_plan` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
serialize_plan(p::ColumnBlockPlan)
serialize_plan(p::BlockDiagonalPlan)
```

Serialize a plan into the `"factorization"` field of the document block.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/EarthSciASTDiff.jl/src/factorization.jl#L76)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
serialize_plan(p) -> dict
```

<p class="api-absent">No docstring.</p>

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/earthsci-astdiff-py/src/earthsci_astdiff/factorization.py#L155)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn serialize_plan(plan: &FactorizationPlan) -> Value
```

A factorization plan as the `factorization` field of a `jacobians`
block (spec §4): the reference's `serialize_plan`. A block LU is an
object with `type` (`"block_diagonal_lu"`), `block_vars` (the block's
variables in elimination order), `ordering` (the 1-based Markowitz
ordering of the name-sorted variables), and `pattern` and `lu_pattern`
(the permuted block's pattern and its LU pattern including fill-in, as
sorted 1-based `[row, column]` pairs). Serialized with
[`crate::stable_json`](/generated/api/ref/stable_json) it is byte-identical to the Julia and Python
bindings' output.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/earthsci-astdiff-rs/src/emit.rs#L781) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/emit/fn.serialize_plan.html)

</div>

</div>
