---
title: "cse_templates"
description: "Stable function · Serialize as a document block"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Serialize as a document block](/generated/api/topics/serialize)

| Julia | Python | Rust |
|---|---|---|
| `cse_templates` | `cse_templates` | `cse_templates` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
cse_templates(exprs::Vector{EarthSciAST.ASTExpr}; min_nodes, prefix)
```

```julia
cse_templates(exprs; min_nodes = 12, prefix = "jt")
    -> (templates::OrderedDict{String,ASTExpr}, rewritten::Vector{ASTExpr})
```

Extract repeated subtrees of `exprs` by the rule of esm-jacobian-spec §4.1.
Every structurally distinct subtree is visited once, largest first (ties in
first-encounter post-order over the inputs). Its count is its number of
occurrences that remain once every larger extracted subtree is replaced by a
reference and kept once as a template body. A subtree that spans at least
`min_nodes` AST nodes and still counts at least 2 is extracted, named
`"$(prefix)$k"` in visiting order, and every occurrence (in the inputs and
in larger template bodies) is replaced by an `apply_expression_template`
reference. The result is a reference DAG, acyclic by strict containment, and
a body only references templates named after it. Θ(distinct subtrees ·
log) after one walk of the inputs (a subtree shared in memory is walked
once).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2c83701848fda145f8d65f939fb8cf382ee2681e/pkg/EarthSciASTDiff.jl/src/cse.jl#L141)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
cse_templates(exprs: list[Expr], min_nodes: int = 12, prefix: str = 'jt')
```

Extract repeated subtrees by the rule of esm-jacobian-spec §4.1 (the
reference's cse.jl). Every distinct subtree is visited once, largest
first (ties in first-encounter order); its count is its number of
occurrences left once every larger extracted subtree is a reference whose
body is kept once; one of at least `min_nodes` nodes that still counts
at least 2 is extracted and named `<prefix><k>` in visiting order.
Returns (templates: {name: body}, rewritten exprs).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2c83701848fda145f8d65f939fb8cf382ee2681e/pkg/earthsci-astdiff-py/src/earthsci_astdiff/cse.py#L128)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn cse_templates(exprs: &[Expr], min_nodes: usize, prefix: &str) -> (Vec<(String, Expr)>, Vec<Expr>)
```

Extract repeated subtrees of `exprs` by the rule of esm-jacobian-spec
§4.1: every distinct subtree is visited once, largest first (ties in
first-encounter order); its count is its number of occurrences left once
every larger extracted subtree is a reference whose body is kept once;
one of at least `min_nodes` nodes that still counts at least 2 is
extracted and named `<prefix><k>` in visiting order. Byte-identical to the
reference.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2c83701848fda145f8d65f939fb8cf382ee2681e/pkg/earthsci-astdiff-rs/src/emit.rs#L215) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/emit/fn.cse_templates.html)

</div>

</div>
