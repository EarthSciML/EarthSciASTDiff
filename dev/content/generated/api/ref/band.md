---
title: "band"
description: "Stable type · Differentiate a model"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Differentiate a model](/generated/api/topics/differentiate)

| Julia | Python | Rust |
|---|---|---|
| `Band` | `Band` | `Band` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
struct Band
Band(rows::Vector{Tuple{Int64, Int64}}, ridx::Vector{Union{Int64, String}}, cidx::Vector{EarthSciAST.ASTExpr}, coef::EarthSciAST.ASTExpr, contracted::Vector{Tuple{String, Int64, Int64}})
Band(rows, ridx, cidx, coef, contracted)
Band(rows, ridx, cidx, coef)
```

```julia
Band
```

One structured block of `∂(rhs of the equation for u)/∂v`:

  - `rows::Vector{Tuple{Int,Int}}` — a rectangular region of `u`'s index
    space, per-dimension inclusive `(lo, hi)`. Empty for a scalar `u`.
  - `ridx::Vector{Union{String,Int}}` — per-dimension row index: a free index
    NAME ranging over `rows[k]`, or a constant Int for a singleton dimension.
  - `cidx::Vector{ASTExpr}` — `v`'s column index expressions, one per
    dimension of `v`, in terms of the row index names. Empty for scalar `v`.
  - `coef::ASTExpr` — scalar ESM coefficient expression in terms of the row
    index names (plus parameters and other states).
  - `contracted::Vector{Tuple{String,Int,Int}}` — free CONTRACTED column
    dimensions `(name, lo, hi)`: when non-empty the band denotes, for every
    row cell AND every value of the contracted names over their inclusive
    ranges, an entry `J[row, cidx(row, contracted…)] += coef(row,
    contracted…)` — entries meeting at one column ACCUMULATE (this is how a
    `reduce: "+"` faq node's sum over a gathered column materializes).
    Empty for ordinary bands.

The band is the symbolic, portable analogue of the tree-walk's `_AccDesc`
state-read descriptors: `cidx` of the form `row ± const` is an affine stencil
offset, `cidx == ridx` is a diagonal (per-cell) coupling, and a `cidx`
referencing a contracted name is an indirect/contracted gather
(source–receptor matrices, regrid joins, `conn[]`-table stencils).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/EarthSciASTDiff.jl/src/bands.jl#L4)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
class Band
Band(rows: list[tuple[int, int]], ridx: list, cidx: list, coef: Expr, contracted: list[tuple[str, int, int]] = list())
```

One structured block of ∂(rhs of the equation for u)/∂v — see the
Julia reference docstring and esm-jacobian-spec.md §4. `contracted`
holds free contracted column dimensions `(name, lo, hi)`.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-py/src/earthsci_astdiff/bands.py#L21)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct Band {
    pub rows: Vec<(i64, i64)>,
    pub ridx: Vec<RIdx>,
    pub cidx: Vec<Expr>,
    pub coef: Expr,
    pub contracted: Vec<(String, i64, i64)>,
}
```

One rectangular region of a Jacobian block: the rows it covers
(`rows`, 1-based inclusive ranges, named by `ridx`), the column-index
expressions (`cidx`) and the coefficient expression (`coef`). `contracted`
lists extra column dimensions that accumulate (summed) at assembly.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-rs/src/bands.rs#L22) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/bands/struct.Band.html)

</div>

</div>
