---
title: "site"
description: "Stable type · Differentiate a model"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [Differentiate a model](/generated/api/topics/differentiate)

| Julia | Python | Rust |
|---|---|---|
| `Site` | `Site` | `Site` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
struct Site
Site(expr::EarthSciAST.ASTExpr, key::String, name::String, rank::Int64)
Site(expr, key, name, rank)
Site(e::EarthSciAST.ASTExpr)
```

```julia
Site(expr)
```

The thing a scalar derivative is taken with respect to: a bare `VarExpr`
(scalar variable) or an `index(v, i…)` node (one symbolic cell of an array
variable). Two nodes denote the same site iff their canonical JSON agrees —
`d/d(u[i+1,j])` treats `index(u, +(i,1), j)` and `index(u, +(1,i), j)` as the
same site because canonicalization sorts commutative arguments.

`name`/`rank` are a PREFILTER cached off `expr`, not extra identity: the site
reads variable `name` at `rank` subscripts (`rank == -1` for a bare
`VarExpr`, `-2` for a shape this fast path does not describe, which then
falls back to the plain key compare). See `is_site` for why.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/EarthSciASTDiff.jl/src/expr_helpers.jl#L63)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
class Site
Site(expr: Expr, key: str)
@staticmethod Site.of(e: Expr) -> Site
```

A bare variable reference or one symbolic `index(v, i…)` cell.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/earthsci-astdiff-py/src/earthsci_astdiff/expr_helpers.py#L164)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct Site {
    pub expr: Expr,
    pub key: String,
}

impl Site { pub fn of(e: &Expr) -> Site }
```

A bare variable reference or one symbolic `index(v, i…)` cell.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/earthsci-astdiff-rs/src/helpers.rs#L224) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/helpers/struct.Site.html)

</div>

</div>
