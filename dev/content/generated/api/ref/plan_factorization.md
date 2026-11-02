---
title: "plan_factorization"
description: "Stable function · Sparsity and structure"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Sparsity and structure](/generated/api/topics/structure)

| Julia | Python | Rust |
|---|---|---|
| `plan_factorization` | `plan_factorization` | `plan_factorization` |


**Note:** Every binding refuses an unknown structure class and a band list with a parameter column. Rust takes the view's context too (`plan_factorization(entries, sv, ctx, structure)`) and returns a `FactorizationPlan` enum; it also exposes the first plan implementation, the per-cell block LU, directly as `block_diagonal_lu`. A `column_block` plan needs the column axis the class was detected with: a `column_axis` keyword in Julia and Python, `plan_factorization_along(entries, sv, ctx, structure, column_axis)` in Rust.


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
plan_factorization(entries::Vector{JacEntry}, sv::SysView; column_axis, structure)
```

```julia
plan_factorization(entries, sv::SysView;
                   structure = detect_structure(entries, sv))
    -> Union{FactorizationPlan, Nothing}
```

Symbolic factorization info for the detected structure class, or `nothing`
when no plan is implemented for that class (the Jacobian itself is still
fully usable — a plan is an optimization, never a requirement).

Currently implemented: `:block_diagonal` → [`BlockDiagonalPlan`](/generated/api/ref/block_diagonal_plan), and
`:column_block` (which needs the `column_axis` it was detected with) →
[`ColumnBlockPlan`](/generated/api/ref/column_block_plan).

Throws an `ArgumentError` when `structure` is not one of
`STRUCTURE_CLASSES` (a misspelled class never silently gives no
plan), or when a column of `entries` is a parameter: a `:parameters` or
`:forcing` band list is a rectangular matrix, which no plan factorizes.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/EarthSciASTDiff.jl/src/factorization.jl#L21)

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
plan_factorization(entries: list[JacEntry], sv: SysView, structure: str | None = None, column_axis: str | None = None)
```

Symbolic factorization info for the detected structure.

Returns a [`BlockDiagonalPlan`](/generated/api/ref/block_diagonal_plan) when the structure is
`"block_diagonal"`, a [`ColumnBlockPlan`](/generated/api/ref/column_block_plan) for `"column_block"`
(which needs the `column_axis` it was detected with), and `None`
otherwise; banded and general plans are not implemented yet.

Raises `ValueError` when `structure` is not one of
`STRUCTURE_CLASSES` (a misspelled class never silently gives no
plan), or when a column of `entries` is a parameter: a parameters or
forcing band list is a rectangular matrix, which no plan factorizes.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/earthsci-astdiff-py/src/earthsci_astdiff/factorization.py#L87)

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
pub fn plan_factorization(entries: &[JacEntry], sv: &SysView<'_>, ctx: &Ctx, structure: Option<&str>) -> Result<Option<FactorizationPlan>, BandError>
```

Symbolic factorization info for the band list's structure class: the
reference's `plan_factorization(entries, sv; structure)`.

`structure` defaults (`None`) to [`detect_structure`](/generated/api/ref/detect_structure) of `entries` over
`ctx` (the view's context, `crate::ctx_of`). A `"block_diagonal"`
class gives its per-cell block LU (`FactorizationPlan::BlockDiagonal`);
`"column_block"` needs its column axis (`plan_factorization_along`);
every other class gives `None`, since no plan is implemented for it yet
(the Jacobian is still fully usable: a plan is an optimization, never a
requirement).

Errors: `structure` is not one of `STRUCTURE_CLASSES`, or `entries`
has a parameter column (a parameters or forcing band list is a
rectangular matrix, which no plan factorizes).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/earthsci-astdiff-rs/src/emit.rs#L710) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/emit/fn.plan_factorization.html)

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
