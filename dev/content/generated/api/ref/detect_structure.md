---
title: "detect_structure"
description: "Stable function · Sparsity and structure"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Sparsity and structure](/generated/api/topics/structure)

| Julia | Python | Rust |
|---|---|---|
| `detect_structure` | `detect_structure` | `detect_structure` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
detect_structure(entries::Vector{JacEntry}, sv::SysView; column_axis)
```

```julia
detect_structure(entries, sv::SysView) -> Symbol
```

Classify the Jacobian's structure from the band list:

  - `:empty`          — no bands.
  - `:block_diagonal` — every array-state band is diagonal (`col == row` in
    every dimension) and no band couples a scalar state to an array state:
    after the species-major → cell-major permutation the matrix is
    block-diagonal with one dense-patterned block per cell.
  - `:banded`         — off-diagonal couplings exist but every column index
    is `row ± literal` (bounded stencil offsets): banded/block-banded once
    strides are fixed by the layout.
  - `:general`        — anything else (fixed boundary columns, periodic wrap
    corners, indirect gathers): use a general sparse type.
  - `:column_block`    — only when `column_axis` (an index-set name) is given,
    in place of `:banded` or `:general`: every coupling stays within one
    LINE along that axis (see [`column_lines`](/generated/api/ref/column_lines)) — a column scan, a
    contraction over the axis, a stencil along it — so a solver can
    factorize line by line.

`column_axis` comes from the caller (an integrator's locality pass): an
index-set name carries no vertical meaning by itself.

This is a CLASSIFICATION of the exact pattern, not an approximation — the
pattern itself is always available via [`jacobian_pattern`](/generated/api/ref/jacobian_pattern).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/EarthSciASTDiff.jl/src/structure.jl#L32)

**Example** (`structure`)

```julia
@assert detect_structure(entries, view) == :block_diagonal
plan = plan_factorization(entries, view)
@assert sort(plan.block_vars) == ["A", "B", "C"]
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/docs/api/examples/julia/jacobians.jl#L30-L32">docs/api/examples/julia/jacobians.jl</a></p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
detect_structure(entries: list[JacEntry], sv: SysView, column_axis: str | None = None) -> str
```

Classify the band list: `"block_diagonal"`, `"banded"`, `"general"`
(or `"empty"` / `"other"`), read off the column-index expressions
without evaluating anything. Use it to choose a matrix type.

Given `column_axis` (an index-set name, from the caller's locality
pass), `"column_block"` replaces `"banded"` / `"general"` when
every coupling stays within one LINE along that axis
([`column_lines`](/generated/api/ref/column_lines)), so a solver can factorize line by line.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/earthsci-astdiff-py/src/earthsci_astdiff/structure.py#L36)

**Example** (`structure`)

```python
assert earthsci_astdiff.detect_structure(entries, view) == "block_diagonal"
plan = earthsci_astdiff.plan_factorization(entries, view)
assert sorted(plan.block_vars) == ["A", "B", "C"]
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/docs/api/examples/python/jacobians.py#L38-L40">docs/api/examples/python/jacobians.py</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn detect_structure(entries: &[JacEntry], ctx: &Ctx) -> &'static str
```

Classify the band list as `"block_diagonal"`, `"banded"` or `"general"`
(or `"empty"` / `"other"`), read off the column-index expressions without
evaluating anything. Use it to choose a matrix type.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/earthsci-astdiff-rs/src/emit.rs#L253) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/emit/fn.detect_structure.html)

**Example** (`structure`)

```rust
let ctx = ctx_of(&view)?;
assert_eq!(detect_structure(&entries, &ctx), "block_diagonal");
let Some(FactorizationPlan::BlockDiagonal(plan)) =
    plan_factorization(&entries, &view, &ctx, None)?
else {
    unreachable!("a block-diagonal Jacobian has a block LU plan")
};
let mut vars = plan.block_vars.clone();
vars.sort();
assert_eq!(vars, ["A", "B", "C"]);
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/docs/api/examples/rust/jacobians.rs#L28-L37">docs/api/examples/rust/jacobians.rs</a></p>

</div>

</div>
