---
title: "block_diagonal_plan"
description: "Stable type · Sparsity and structure"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Sparsity and structure](/generated/api/topics/structure)

| Julia | Python | Rust |
|---|---|---|
| `BlockDiagonalPlan` | `BlockDiagonalPlan` | `BlockLu` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
struct BlockDiagonalPlan <: FactorizationPlan
BlockDiagonalPlan(block_vars::Vector{String}, pattern::BitMatrix, ordering::Vector{Int64}, lu::BitMatrix)
BlockDiagonalPlan(block_vars, pattern, ordering, lu)
```

```julia
BlockDiagonalPlan <: FactorizationPlan
```

Per-cell block factorization info for a `:block_diagonal` Jacobian (the
chemistry shape: every coupling is within-cell, so the global matrix is one
identical `ns × ns`-patterned block per cell):

  - `block_vars`   — the state variables forming the block, in block order.
  - `pattern`      — the block's structural pattern (`ns × ns`, block order).
  - `ordering`     — elimination ordering (a permutation of `1:ns`) chosen by
    a greedy Markowitz heuristic to reduce fill-in.
  - `lu`           — the LU pattern **including fill-in** under `ordering`
    (of the permuted block), so a solver can allocate and generate
    straight-line factorization code without symbolic analysis at runtime.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/EarthSciASTDiff.jl/src/factorization.jl#L81)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
class BlockDiagonalPlan
BlockDiagonalPlan(block_vars: list[str], pattern: list[list[bool]], ordering: list[int], lu: list[list[bool]])
```

Per-cell block LU of a block-diagonal Jacobian: the block variables in
Markowitz order, the block pattern, the 1-based ordering and the pattern
of the LU factors including fill-in.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/earthsci-astdiff-py/src/earthsci_astdiff/factorization.py#L15)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct BlockLu {
    pub block_vars: Vec<String>,
    pub ordering: Vec<usize>,
    pub pattern: Vec<Vec<bool>>,
    pub lu: Vec<Vec<bool>>,
}
```

The per-cell block LU of a block-diagonal band list (factorization.jl's
`BlockDiagonalPlan`): block variables in Markowitz order, the 1-based
ordering, and the permuted pattern and its fill-in LU pattern.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/earthsci-astdiff-rs/src/emit.rs#L627) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/emit/struct.BlockLu.html)

</div>

</div>
