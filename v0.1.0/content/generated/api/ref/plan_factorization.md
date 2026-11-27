---
title: "plan_factorization"
description: "Stable function · Sparsity and structure"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Sparsity and structure](/generated/api/topics/structure)

| Julia | Python | Rust |
|---|---|---|
| `plan_factorization` | `plan_factorization` | `block_diagonal_lu` |


**Note:** Rust exposes the first plan implementation, the per-cell block LU, directly as `block_diagonal_lu`.


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
plan_factorization(entries::Vector{JacEntry}, sv::SysView; structure)
```

```julia
plan_factorization(entries, sv::SysView;
                   structure = detect_structure(entries, sv))
    -> Union{FactorizationPlan, Nothing}
```

Symbolic factorization info for the detected structure class, or `nothing`
when no plan is implemented for that class (the Jacobian itself is still
fully usable — a plan is an optimization, never a requirement).

Currently implemented: `:block_diagonal` → [`BlockDiagonalPlan`](/generated/api/ref/block_diagonal_plan).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/EarthSciASTDiff.jl/src/factorization.jl#L21)

**Example** (`structure`)

```julia
@assert detect_structure(entries, view) == :block_diagonal
plan = plan_factorization(entries, view)
@assert sort(plan.block_vars) == ["A", "B", "C"]
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/docs/api/examples/julia/jacobians.jl#L28-L30">docs/api/examples/julia/jacobians.jl</a></p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
plan_factorization(entries: list[JacEntry], sv: SysView, structure: str | None = None)
```

Symbolic factorization info for the detected structure.

Returns a [`BlockDiagonalPlan`](/generated/api/ref/block_diagonal_plan) when the structure is
`"block_diagonal"` and `None` otherwise; banded and general plans are
not implemented yet.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/earthsci-astdiff-py/src/earthsci_astdiff/factorization.py#L62)

**Example** (`structure`)

```python
assert earthsci_astdiff.detect_structure(entries, view) == "block_diagonal"
plan = earthsci_astdiff.plan_factorization(entries, view)
assert sorted(plan.block_vars) == ["A", "B", "C"]
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/docs/api/examples/python/jacobians.py#L33-L35">docs/api/examples/python/jacobians.py</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn block_diagonal_lu(entries: &[JacEntry]) -> BlockLu
```

Symbolic factorization info for a block-diagonal band list: the per-cell
block LU with Markowitz ordering and fill-in ([`BlockLu`](/generated/api/ref/block_diagonal_plan)).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/earthsci-astdiff-rs/src/emit.rs#L316) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/emit/fn.block_diagonal_lu.html)

**Example** (`structure`)

```rust
let ctx = ctx_of(&view)?;
assert_eq!(detect_structure(&entries, &ctx), "block_diagonal");
let plan = block_diagonal_lu(&entries);
let mut vars = plan.block_vars.clone();
vars.sort();
assert_eq!(vars, ["A", "B", "C"]);
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/docs/api/examples/rust/jacobians.rs#L31-L36">docs/api/examples/rust/jacobians.rs</a></p>

</div>

</div>
