---
title: "column_block_plan"
description: "Stable type · Sparsity and structure"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Sparsity and structure](/generated/api/topics/structure)

| Julia | Python | Rust |
|---|---|---|
| `ColumnBlockPlan` | `ColumnBlockPlan` | `ColumnLines` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
struct ColumnBlockPlan <: FactorizationPlan
ColumnBlockPlan(axis::String, line_length::Int64, block_vars::Vector{String}, pattern::String, kl::Int64, ku::Int64)
ColumnBlockPlan(axis, line_length, block_vars, pattern, kl, ku)
```

```julia
ColumnBlockPlan <: FactorizationPlan
```

Line-by-line factorization info for a `:column_block` Jacobian: every
coupling stays within one line along `axis`, so the matrix is one
`(length(block_vars)·line_length)`-square block per line. `pattern` names the
coupling along the line (`"dense"`, `"lower"`, `"upper"`, `"banded"`) and
`kl` / `ku` bound it below and above the diagonal, so a solver can pick a
column solver (banded LU, a triangular sweep, dense LU) without looking at
the entries.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/EarthSciASTDiff.jl/src/factorization.jl#L176)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
class ColumnBlockPlan
ColumnBlockPlan(axis: str, line_length: int, block_vars: list[str], pattern: str, kl: int, ku: int, rank1: list = list())
```

Line-by-line factorization info for a `"column_block"` Jacobian:
every coupling stays within one line along `axis`, so the matrix is one
`(len(block_vars)·line_length)`-square block per line. `pattern`
names the coupling along the line (`"dense"`, `"lower"`,
`"upper"`, `"banded"`, `"diag_plus_rank1"`) and `kl` / `ku`
bound it below and above the diagonal.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/earthsci-astdiff-py/src/earthsci_astdiff/factorization.py#L27)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct ColumnLines {
    pub axis: String,
    pub dim: usize,
    pub line_length: i64,
    pub block_vars: Vec<String>,
    pub kl: i64,
    pub ku: i64,
    pub pattern: &'static str,
    pub rank1: Vec<Rank1>,
}
```

The per-line structure of a state Jacobian whose couplings all stay
within one line along an index set (structure.jl `column_lines`).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/earthsci-astdiff-rs/src/emit.rs#L321) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/emit/struct.ColumnLines.html)

</div>

</div>
