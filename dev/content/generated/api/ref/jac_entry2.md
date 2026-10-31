---
title: "jac_entry2"
description: "Stable type · Differentiate a model"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Differentiate a model](/generated/api/topics/differentiate)

| Julia | Python | Rust |
|---|---|---|
| `JacEntry2` | `JacEntry2` | `JacEntry2` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
struct JacEntry2
JacEntry2(u::String, v::String, w::String, band::Band, ncol::Int64)
JacEntry2(u, v, w, band, ncol)
```

```julia
JacEntry2
```

One second-order band: `∂²(equation for u)/∂v ∂w`. `band` is the first-order
entry's band with its coefficient replaced by `∂coef/∂(site of w)`: the same
`rows`, `ridx` and `contracted` (plus any dimensions the second site adds
under a nested sum), and `cidx` holding `v`'s column indices FOLLOWED by
`w`'s. The first `ncol` of them are `v`'s (`col_idx` /
`col2_idx` split them). Keeping both in one `Band` lets clipping,
merging and normalization run unchanged.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/EarthSciASTDiff.jl/src/second_order.jl#L6)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
class JacEntry2
JacEntry2(u: str, v: str, w: str, band: Band, ncol: int)
JacEntry2.col_idx: list  # property
JacEntry2.col2_idx: list  # property
```

One second-order band: ∂²(equation for `u`)/∂`v` ∂`w`. `band`
is the first-order entry's band with its coefficient replaced by
∂coef/∂(site of `w`): the same `rows`, `ridx` and `contracted`
(plus any dimensions the second site adds under a nested sum), and
`cidx` holding `v`'s column indices FOLLOWED by `w`'s. The first
`ncol` of them are `v`'s (`col_idx` / `col2_idx` split
them). Keeping both in one [`Band`](/generated/api/ref/band) lets clipping, merging and
normalization run unchanged.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-py/src/earthsci_astdiff/second_order.py#L35)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct JacEntry2 {
    pub u: String,
    pub v: String,
    pub w: String,
    pub band: Band,
    pub ncol: usize,
}

impl JacEntry2 { pub fn col_idx(&self) -> &[Expr] }

impl JacEntry2 { pub fn col2_idx(&self) -> &[Expr] }
```

One second-order band: `∂²(equation for u)/∂v ∂w`. `band` is the
first-order entry's band with its coefficient replaced by `∂coef/∂(site
of w)`: the same `rows`, `ridx` and `contracted` (plus any dimensions the
second site adds under a nested sum), and `cidx` holding `v`'s column
indices FOLLOWED by `w`'s. The first `ncol` of them are `v`'s
(`JacEntry2::col_idx` / `JacEntry2::col2_idx` split them). Keeping
both in one [`Band`](/generated/api/ref/band) lets clipping, merging and normalization run
unchanged.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/pkg/earthsci-astdiff-rs/src/second_order.rs#L36) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/second_order/struct.JacEntry2.html)

</div>

</div>
