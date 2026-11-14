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
jacobian_block(obj, model_name: str | None = None, wrt: str = 'states', cse: bool = True, cse_min_nodes: int = 12) -> dict
```

The §4 `jacobians` block for one model: entries (template-CSE'd by
default), the structure classification, and the block-local
`expression_templates`, plus the `factorization` plan where one exists
for the detected structure (per-cell block LU with Markowitz ordering).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/earthsci-astdiff-py/src/earthsci_astdiff/emit.py#L106)

**Example** (`block`)

```python
block = earthsci_astdiff.jacobian_block(file)
assert block["structure"] == "block_diagonal"
assert block["factorization"]["type"] == "block_diagonal_lu"
json.dumps(block)  # a plain, serializable dict
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/docs/api/examples/python/jacobians.py#L47-L50">docs/api/examples/python/jacobians.py</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn jacobian_block(file: &EsmFile, model_name: Option<&str>, wrt: Wrt, cse: bool, cse_min_nodes: usize) -> Result<Value, BandError>
```

The `jacobians` block for one model (spec §4): entries (template-CSE'd when
`cse` is set, hoisting subtrees of at least `cse_min_nodes` nodes), the
structure classification, and the factorization plan where one exists.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/pkg/earthsci-astdiff-rs/src/emit.rs#L359) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/emit/fn.jacobian_block.html)

**Example** (`block`)

```rust
let block = jacobian_block(&file, None, Wrt::States, true, 12)?;
assert_eq!(block["structure"], "block_diagonal");
assert_eq!(block["factorization"]["type"], "block_diagonal_lu");
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/6f151823f210d8951df2266333997c204f54e68e/docs/api/examples/rust/jacobians.rs#L59-L61">docs/api/examples/rust/jacobians.rs</a></p>

</div>

</div>
