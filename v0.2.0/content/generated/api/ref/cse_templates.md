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

Greedy largest-first extraction of repeated subtrees over `exprs`. Each
round interns the whole working set (input expressions plus the bodies of
templates already extracted), picks the largest subtree that occurs at least
twice and spans at least `min_nodes` AST nodes, names it `"$(prefix)$k"`,
and replaces every occurrence — in the inputs AND in earlier template
bodies — with an `apply_expression_template` reference. Terminates when no
candidate remains; the result is a reference DAG (acyclic by strict
containment). Deterministic: ties between equally-sized candidates break by
interning (first-encounter) order over the given expression order.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/EarthSciASTDiff.jl/src/cse.jl#L87)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
cse_templates(exprs: list[Expr], min_nodes: int = 12, prefix: str = 'jt')
```

Greedy largest-first extraction of repeated subtrees. Returns
(templates: {name: body}, rewritten exprs).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/earthsci-astdiff-py/src/earthsci_astdiff/cse.py#L80)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn cse_templates(exprs: &[Expr], min_nodes: usize, prefix: &str) -> (Vec<(String, Expr)>, Vec<Expr>)
```

Greedy largest-first extraction; deterministic (first-encounter order,
first-max tie-break) to match the reference byte-for-byte.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/earthsci-astdiff-rs/src/emit.rs#L149) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/emit/fn.cse_templates.html)

</div>

</div>
