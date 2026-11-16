---
title: "jacobian_document"
description: "Stable function · Serialize as a document block"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Serialize as a document block](/generated/api/topics/serialize)

| Julia | Python | Rust |
|---|---|---|
| `jacobian_document` | not exported | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
jacobian_document(file::EarthSciAST.EsmFile, model_name; wrt, cse, cse_min_nodes)
jacobian_document(flat::EarthSciAST.FlattenedSystem; wrt, name, cse, cse_min_nodes)
```

```julia
jacobian_document(file::EsmFile, model_name; wrt = :states,
                  cse = true, cse_min_nodes = 12) -> Dict
jacobian_document(flat::FlattenedSystem; wrt = :states, name = "Flattened",
                  cse = true, cse_min_nodes = 12) -> Dict
```

The serialized input document plus the draft `"jacobians"` top-level block
(esm-jacobian-spec.md §4): per model, the differentiation axis, the band
entries, the detected structure class, and — when a factorization plan
exists for that structure — the serialized plan.

With `cse = true` (default), repeated coefficient subtrees of at least
`cse_min_nodes` AST nodes are hoisted into zero-parameter
`expression_templates` carried inside the block ([`cse_templates`](/generated/api/ref/cse_templates)),
and the emitted coefficients reference them via `apply_expression_template`
— the size control for limiter-heavy schemes. `parse_jacobian_block`
expands the references back, so the round-trip is exact either way.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/EarthSciASTDiff.jl/src/emit.jl#L70)

**Example** (`block`)

```julia
doc = jacobian_document(file, "Chem")
block = doc["jacobians"]["Chem"]
@assert block["structure"] == "block_diagonal"
@assert block["factorization"]["type"] == "block_diagonal_lu"
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/docs/api/examples/julia/jacobians.jl#L52-L55">docs/api/examples/julia/jacobians.jl</a></p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

<p class="api-absent">Python does not export this function.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this function.</p>

</div>

</div>
