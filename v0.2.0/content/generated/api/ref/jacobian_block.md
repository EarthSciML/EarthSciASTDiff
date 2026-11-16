---
title: "jacobian_block"
description: "Stable function · Serialize as a document block"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Serialize as a document block](/generated/api/topics/serialize)

| Julia | Python | Rust |
|---|---|---|
| not exported | `jacobian_block` | `jacobian_block` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this function.</p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
jacobian_block(obj, model_name: str | None = None, wrt='states', cse: bool = True, cse_min_nodes: int = 12, forcing=None, column_axis: str | None = None)
```

The §4 `jacobians` block for one model: entries (template-CSE'd by
default) and the block-local `expression_templates`; on the state and
time axes the structure classification plus the `factorization` plan
where one exists for the detected structure (per-cell block LU with
Markowitz ordering), on the parameters and forcing axes the column space
(`columns`, [`jacobian_columns`](/generated/api/ref/jacobian_columns)) instead.

`wrt` may also be a list of distinct axes: the result is then the list
of blocks, one per axis in the given order (a model's `jacobians` entry
in its array form, spec §4). Inside such a list a `tuple` `(a1, a2)`
(or a two-name list) is an axis PAIR, a block of second-order entries
([`jacobian_bands2`](/generated/api/ref/jacobian_bands2), §4.4): `["states", ("states", "states")]`,
and `[("states", "states")]` for a pair block alone. A bare `tuple`
`wrt` is refused: it reads both as a pair and as a list of axes, so
neither is assumed.

`column_axis` (an index-set name) lets the state block report the
`"column_block"` structure class with its line-by-line factorization
plan when every coupling stays within one line along that axis
([`column_lines`](/generated/api/ref/column_lines)).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/earthsci-astdiff-py/src/earthsci_astdiff/emit.py#L200)

**Example** (`block`)

```python
block = earthsci_astdiff.jacobian_block(file)
assert block["structure"] == "block_diagonal"
assert block["factorization"]["type"] == "block_diagonal_lu"
json.dumps(block)  # a plain, serializable dict
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/docs/api/examples/python/jacobians.py#L59-L62">docs/api/examples/python/jacobians.py</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn jacobian_block(file: &EsmFile, model_name: Option<&str>, wrt: Wrt, cse: bool, cse_min_nodes: usize) -> Result<Value, BandError>
```

The `jacobians` block for one model (spec §4): entries (template-CSE'd when
`cse` is set, hoisting subtrees of at least `cse_min_nodes` nodes) and —
on the state and time axes — the structure classification and the
factorization plan where one exists. A parameters or forcing block carries
its column space instead (`columns`, [`crate::jacobian_columns`](/generated/api/ref/jacobian_columns)). The
forcing set is inferred; see `jacobian_block_with_forcing`.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/earthsci-astdiff-rs/src/emit.rs#L820) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/emit/fn.jacobian_block.html)

**Example** (`block`)

```rust
let block = jacobian_block(&file, None, Wrt::States, true, 12)?;
assert_eq!(block["structure"], "block_diagonal");
assert_eq!(block["factorization"]["type"], "block_diagonal_lu");
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/docs/api/examples/rust/jacobians.rs#L55-L57">docs/api/examples/rust/jacobians.rs</a></p>

</div>

</div>
