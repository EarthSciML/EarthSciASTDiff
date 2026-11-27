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
detect_structure(entries::Vector{JacEntry}, sv::SysView; column_axis, cell_axes)
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
  - `:cell_block`      — only when `cell_axes` (index-set names) is given, in
    place of `:banded` or `:general` and before `:column_block`: every
    coupling stays inside one CELL of those axes (see `cell_blocks`),
    so the matrix is one block per cell holding every element of every
    variable along the other axes (a species or salt axis summed inside the
    cell, variables of different ranks), factorized cell by cell.

`:block_diagonal` and `:banded` are POSITIONAL: a band's column is compared
with its row dimension by dimension at the same position, because their
consumers (the lane-block scatter plan, the block-diagonal LU plan) pair a
row cell with the column cell at the same storage position. So a band whose
column and row have different ranks (a variable over `cell` reading one
over `cell × salt`), whose column is over an index set its row is not (a
variable over `cell × salt` reading one over `cell × el` with matching
indices), or whose column is over the row's index sets in another order (a
variable over `x × y` reading one over `y × x` at `[j, i]`) is never
`:block_diagonal` or `:banded`: it is `:general` unless `cell_axes` makes it
`:cell_block`, whose plan pairs the cell axes by index-set name. When either
variable's index-set names are not known, the dimensions pair by position.

`column_axis` and `cell_axes` come from the caller (an integrator's locality
pass): an index-set name carries no vertical or cell meaning by itself.

This is a CLASSIFICATION of the exact pattern, not an approximation — the
pattern itself is always available via [`jacobian_pattern`](/generated/api/ref/jacobian_pattern).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2621db67c22b0f489f82486ef9c881b3a8a27a4d/pkg/EarthSciASTDiff.jl/src/structure.jl#L32)

**Example** (`structure`)

```julia
@assert detect_structure(entries, view) == :block_diagonal
plan = plan_factorization(entries, view)
@assert sort(plan.block_vars) == ["A", "B", "C"]
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2621db67c22b0f489f82486ef9c881b3a8a27a4d/docs/api/examples/julia/jacobians.jl#L30-L32">docs/api/examples/julia/jacobians.jl</a></p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
detect_structure(entries: list[JacEntry], sv: SysView, column_axis: str | None = None, cell_axes: list[str] | None = None) -> str
```

Classify the band list: `"block_diagonal"`, `"banded"`, `"general"`
(or `"empty"` / `"other"`), read off the column-index expressions
without evaluating anything. Use it to choose a matrix type.

Given `column_axis` (an index-set name, from the caller's locality
pass), `"column_block"` replaces `"banded"` / `"general"` when
every coupling stays within one LINE along that axis
([`column_lines`](/generated/api/ref/column_lines)), so a solver can factorize line by line.

Given `cell_axes` (index-set names, from the caller), `"cell_block"`
replaces `"banded"` / `"general"`, before `"column_block"`, when
every coupling stays inside one CELL of those axes (`cell_blocks`),
whose plan pairs the cell axes by index-set name. `"block_diagonal"`
and `"banded"` are positional (their consumers pair row and column
cells by storage position): a band whose column and row have different
ranks, whose column is over an index set its row is not, or whose column
is over the row's index sets in another order is `"general"` (by
position when either variable's names are not known). A lone `str`
`cell_axes` is one axis.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2621db67c22b0f489f82486ef9c881b3a8a27a4d/pkg/earthsci-astdiff-py/src/earthsci_astdiff/structure.py#L36)

**Example** (`structure`)

```python
assert earthsci_astdiff.detect_structure(entries, view) == "block_diagonal"
plan = earthsci_astdiff.plan_factorization(entries, view)
assert sorted(plan.block_vars) == ["A", "B", "C"]
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2621db67c22b0f489f82486ef9c881b3a8a27a4d/docs/api/examples/python/jacobians.py#L38-L40">docs/api/examples/python/jacobians.py</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn detect_structure(entries: &[JacEntry], ctx: &Ctx) -> &'static str
```

Classify the band list as `"block_diagonal"`, `"banded"` or `"general"`
(or `"empty"` / `"other"`), read off the column-index expressions without
evaluating anything. Use it to choose a matrix type. The classes are
positional (their consumers pair row and column cells by storage
position): a band whose column and row have different ranks, whose
column is over an index set its row is not, or whose column is over the
row's index sets in another order (`ctx.axes`; by position when either
variable's names are not known) is `"general"`.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2621db67c22b0f489f82486ef9c881b3a8a27a4d/pkg/earthsci-astdiff-rs/src/emit.rs#L349) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/emit/fn.detect_structure.html)

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

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2621db67c22b0f489f82486ef9c881b3a8a27a4d/docs/api/examples/rust/jacobians.rs#L28-L37">docs/api/examples/rust/jacobians.rs</a></p>

</div>

</div>
