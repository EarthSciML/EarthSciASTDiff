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
jacobian_document(file::EarthSciAST.EsmFile, model_name; wrt, forcing, cse, cse_min_nodes, column_axis, cell_axes)
jacobian_document(flat::EarthSciAST.FlattenedSystem; wrt, forcing, name, cse, cse_min_nodes, column_axis, cell_axes, data_sources)
```

```julia
jacobian_document(file::EsmFile, model_name; wrt = :states, forcing = nothing,
                  cse = false, cse_min_nodes = 12, column_axis = nothing,
                  cell_axes = nothing) -> Dict
jacobian_document(flat::FlattenedSystem; wrt = :states, forcing = nothing,
                  name = "Flattened", cse = false, cse_min_nodes = 12,
                  column_axis = nothing, cell_axes = nothing,
                  data_sources = nothing) -> Dict
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

The coefficients are written as the band calculus produced them, with
operator references where the source has them (esm-jacobian-spec.md §3.6);
no repeated subtree is extracted after the fact. `cse` and `cse_min_nodes`
are accepted for compatibility and have no effect (a deprecation warning
when `cse = true`). `parse_jacobian_block` still expands the
block-local `expression_templates` of a block an earlier version wrote.

`column_axis` (an index-set name) lets the state block report the
`"column_block"` structure class with its line-by-line factorization plan
when every coupling stays within one line along that axis
([`detect_structure`](/generated/api/ref/detect_structure), [`column_lines`](/generated/api/ref/column_lines)). `cell_axes` (index-set
names) lets it report the `"cell_block"` class with its per-cell block plan
when every coupling stays inside one cell of those axes (`cell_blocks`),
the class and plan [`detect_structure`](/generated/api/ref/detect_structure) and
[`plan_factorization`](/generated/api/ref/plan_factorization) give with the same `cell_axes`.

The flattened form builds its model from the system, keeping each data-fed
parameter's `data` rules (esm-jacobian-spec.md §8.1), so it also declares the
`data_sources` entries they name. A `FlattenedSystem` carries none: pass the
flattened document's as `data_sources` when its parameters read data; a named
source that is not found is a `BandError`.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2621db67c22b0f489f82486ef9c881b3a8a27a4d/pkg/EarthSciASTDiff.jl/src/emit.jl#L95)

**Example** (`block`)

```julia
doc = jacobian_document(file, "Chem")
block = doc["jacobians"]["Chem"]
@assert block["structure"] == "block_diagonal"
@assert block["factorization"]["type"] == "block_diagonal_lu"
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2621db67c22b0f489f82486ef9c881b3a8a27a4d/docs/api/examples/julia/jacobians.jl#L61-L64">docs/api/examples/julia/jacobians.jl</a></p>

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
