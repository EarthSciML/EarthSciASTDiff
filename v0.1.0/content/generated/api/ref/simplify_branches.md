---
title: "simplify_branches"
description: "Stable function · Differentiate a model"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Differentiate a model](/generated/api/topics/differentiate)

| Julia | Python | Rust |
|---|---|---|
| `simplify_branches` | `simplify_branches` | `simplify_branches` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
simplify_branches(e::EarthSciAST.ASTExpr)
```

```julia
simplify_branches(e::ASTExpr) -> ASTExpr
```

Conditional-specific simplification, complementing `EarthSciAST.simplify`:

  * `ifelse(lit, a, b)` → `a` (lit ≠ 0) or `b` (lit == 0);
  * `ifelse(c, a, a)`   → `a` when the branches are structurally equal;
  * same-condition pruning along a path: inside the true branch of
    `ifelse(c, …)` any nested `ifelse(c, a₂, b₂)` collapses to `a₂`, inside
    the false branch to `b₂` (conditions compared structurally).

Bottom-up; nodes are rebuilt only where something changed.

STRUCTURAL EQUALITY, AND ITS COST. Conditions and branches are compared by
structure, and the pass asks for that comparison at every `ifelse` node it
meets. Doing it with `skey` — a full re-serialization of the subtree
— makes the pass Θ(n·depth): each of the `k` levels of an `index(makearray…)`
region chain re-serializes the whole chain below it, and the product rule
replicates such chains along a path. The
comparisons therefore go through hashcons.jl: a `shash` mismatch —
O(1) per node — settles the common case where the two differ, and only a
match pays for the exact answer from `StructIndex`, whose bottom-up
interning keys a node from its own shell plus its children's ids (Θ(n) for
the tree, O(1) per compare) and memoizes on node identity, so the
identity-preserving rewrites below re-key nothing they did not change. Same
equality relation as `skey`, so the pass is unchanged.

The second Θ(n·depth) term is `_assume`, which re-descends a branch looking
for a condition that is usually not there at all. A running count of the
conditions the bottom-up pass has already built decides that in O(1) at the
branch, so a `k`-level chain costs Θ(n) instead of Θ(n·k); see `_sbr`.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/EarthSciASTDiff.jl/src/simplify_branches.jl#L20)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
simplify_branches(e: Expr) -> Expr
```

Literal-condition fold, equal-branch collapse, same-condition pruning.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/earthsci-astdiff-py/src/earthsci_astdiff/simplify_branches.py#L26)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn simplify_branches(e: &Expr) -> Expr
```

`ifelse`-aware simplification, the swell control for coefficients of
limiter-heavy schemes: equal branches collapse and same-condition tests
are pruned.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/earthsci-astdiff-rs/src/branches.rs#L12) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/branches/fn.simplify_branches.html)

</div>

</div>
