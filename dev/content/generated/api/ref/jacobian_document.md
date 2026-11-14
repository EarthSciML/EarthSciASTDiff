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
jacobian_document(file::EarthSciAST.EsmFile, model_name; wrt, forcing, cse, cse_min_nodes, column_axis)
jacobian_document(flat::EarthSciAST.FlattenedSystem; wrt, forcing, name, cse, cse_min_nodes, column_axis, data_sources)
```

```julia
jacobian_document(file::EsmFile, model_name; wrt = :states, forcing = nothing,
                  cse = true, cse_min_nodes = 12, column_axis = nothing) -> Dict
jacobian_document(flat::FlattenedSystem; wrt = :states, forcing = nothing,
                  name = "Flattened", cse = true, cse_min_nodes = 12,
                  column_axis = nothing, data_sources = nothing) -> Dict
```

The serialized input document plus the draft `"jacobians"` top-level block
(esm-jacobian-spec.md §4): per model, the differentiation axis, the band
entries, and — on the state and time axes — the detected structure class and,
when a factorization plan exists for that structure, the serialized plan. A
`:parameters` or `:forcing` block carries its column space instead
(`columns`, [`jacobian_columns`](/generated/api/ref/jacobian_columns)).

`wrt` may also be a vector of distinct axes (`[:states, :forcing]`): the
model's entry is then an ARRAY of blocks, one per axis in the given order
(esm-jacobian-spec.md §4); a single `wrt` keeps the single-object form. A
`Pair` `a1 => a2` is an axis PAIR, a block of second-order entries
([`jacobian_bands2`](/generated/api/ref/jacobian_bands2), §4.4), alone or in such a list, where a Tuple
`(a1, a2)` (or a two-name vector) is a pair too
(`[:states, (:states, :states)]`). A bare Tuple `wrt` is refused: it would
read either as a pair or as a list of axes, so neither is assumed.

With `cse = true` (default), repeated coefficient subtrees of at least
`cse_min_nodes` AST nodes are hoisted into zero-parameter
`expression_templates` carried inside the block ([`cse_templates`](/generated/api/ref/cse_templates)),
and the emitted coefficients reference them via `apply_expression_template`
— the size control for limiter-heavy schemes. `parse_jacobian_block`
expands the references back, so the round-trip is exact either way.

`column_axis` (an index-set name) lets the state block report the
`"column_block"` structure class with its line-by-line factorization plan
when every coupling stays within one line along that axis
([`detect_structure`](/generated/api/ref/detect_structure), [`column_lines`](/generated/api/ref/column_lines)).

The flattened form builds its model from the system, keeping each data-fed
parameter's `data` rules (esm-jacobian-spec.md §8.1), so it also declares the
`data_sources` entries they name. A `FlattenedSystem` carries none: pass the
flattened document's as `data_sources` when its parameters read data; a named
source that is not found is a `BandError`.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/9891c3005a31b74a0547498d0d2981ebb69cdc14/pkg/EarthSciASTDiff.jl/src/emit.jl#L95)

**Example** (`block`)

```julia
doc = jacobian_document(file, "Chem")
block = doc["jacobians"]["Chem"]
@assert block["structure"] == "block_diagonal"
@assert block["factorization"]["type"] == "block_diagonal_lu"
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/9891c3005a31b74a0547498d0d2981ebb69cdc14/docs/api/examples/julia/jacobians.jl#L61-L64">docs/api/examples/julia/jacobians.jl</a></p>

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
