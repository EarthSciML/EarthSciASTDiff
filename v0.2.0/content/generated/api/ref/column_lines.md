---
title: "column_lines"
description: "Stable function · Sparsity and structure"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Sparsity and structure](/generated/api/topics/structure)

| Julia | Python | Rust |
|---|---|---|
| `column_lines` | `column_lines` | `column_lines` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
column_lines(entries::Vector{JacEntry}, sv::SysView, axis::String)
```

```julia
column_lines(entries, sv, axis::String) -> Union{NamedTuple, Nothing}
```

The per-line structure of a state Jacobian whose couplings all stay within
one line along index set `axis`, or `nothing` when one does not: every band
couples two array states of one common shape that holds `axis` once, and
at every band point the column equals the row in every other dimension. A
point reading past the axis is a ghost read and does not count. Returns
`(axis, dim, line_length, block_vars, kl, ku, pattern)`: `kl` / `ku` the
largest distances below / above the diagonal any point reaches along the
axis, and `pattern` their name — `"dense"` (both span the line), `"lower"`,
`"upper"`, or `"banded"` — read off the exact band points, never probed.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/EarthSciASTDiff.jl/src/structure.jl#L85)

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
column_lines(entries: list[JacEntry], sv: SysView, axis: str) -> dict | None
```

The per-line structure of a state Jacobian whose couplings all stay
within one line along index set `axis`, or `None` when one does not
(structure.jl `column_lines`): every band couples two array states of
one common shape that holds `axis` once, and at every band point the
column equals the row in every other dimension. A point reading past the
axis is a ghost read and does not count. Returns a dict with `axis`,
`dim` (1-based), `line_length`, `block_vars`, `kl` / `ku` (the
largest distances below / above the diagonal any point reaches along the
axis) and `pattern` — `"dense"`, `"lower"`, `"upper"`,
`"banded"` or `"diag_plus_rank1"` — read off the exact band points,
never probed, with `rank1` the rank-one terms of a
`"diag_plus_rank1"` line block (spec §6; empty otherwise).

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/earthsci-astdiff-py/src/earthsci_astdiff/structure.py#L51)

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub fn column_lines(entries: &[JacEntry], sv: &SysView<'_>, ctx: &Ctx, axis: &str) -> Result<Option<ColumnLines>, BandError>
```

The per-line structure of a state Jacobian along index set `axis`, or
`None` when a coupling leaves its line: every band couples two array
states of one common shape that holds `axis` once, and at every band
point the column equals the row in every other dimension. A point
reading past the axis is a ghost read and does not count. `kl` / `ku`
and their `pattern` name are read off the exact band points (guards
applied), never probed.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a7603b0bd3d2f1c582d4b8dc2f8f9c2960584df/pkg/earthsci-astdiff-rs/src/emit.rs#L439) · [Rust docs](https://docs.rs/earthsci-astdiff/latest/earthsci_astdiff/emit/fn.column_lines.html)

</div>

</div>
