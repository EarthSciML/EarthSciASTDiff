---
title: "lu_fill"
description: "Stable function · Sparsity and structure"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Sparsity and structure](/generated/api/topics/structure)

| Julia | Python | Rust |
|---|---|---|
| `lu_fill` | `lu_fill` | `lu_fill` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
lu_fill(pattern::AbstractMatrix{Bool})
```

```julia
lu_fill(pattern::AbstractMatrix{Bool}) -> BitMatrix
```

Symbolic (no-pivoting) LU of a structural pattern: the input pattern plus
every fill-in entry Gaussian elimination creates, in natural order.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/EarthSciASTDiff.jl/src/factorization.jl#L119)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
lu_fill(pattern: list[list[bool]]) -> list[list[bool]]
```

Symbolic (no-pivoting) LU: the pattern plus every fill-in entry.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/earthsci-astdiff-py/src/earthsci_astdiff/factorization.py#L47)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn lu_fill(pattern: &[Vec<bool>]) -> Vec<Vec<bool>>
```

Symbolic (no-pivoting) LU: the pattern plus every fill-in entry.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/earthsci-astdiff-rs/src/emit.rs#L562) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/emit/fn.lu_fill.html)

</div>

</div>
