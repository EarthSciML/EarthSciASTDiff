---
title: "jacobian_pattern"
description: "Stable function · Sparsity and structure"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** function · **Topic:** [Sparsity and structure](/generated/api/topics/structure)

| Julia | Python | Rust |
|---|---|---|
| `jacobian_pattern` | `jacobian_pattern` | not exported |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
jacobian_pattern(input; wrt, model_name, forcing, build_kwargs, source_layout)
```

```julia
jacobian_pattern(input; wrt = :states, forcing = nothing) -> (pattern, rownames, colnames, entries)
```

The structural (global, point-independent) sparsity pattern of the Jacobian
as a `SparseMatrixCSC{Bool}`. `input` is an `EsmFile` + model name pair
(pass a `Tuple`), an `EsmFile` with a single model, or a `FlattenedSystem`.
On the `:parameters` and `:forcing` axes the columns are the canonical column
space of [`jacobian_columns`](/generated/api/ref/jacobian_columns).

The pattern needs the source model's state layout (the name → slot map),
and by default learns it by running a complete `build_evaluator` of the
source and discarding the right-hand side. A caller that has already built
the source evaluator can skip that build by passing its layout:

    # u0, p, var_map: build_evaluator's 2nd, 3rd and 5th returns
    source_layout = (u0 = u0, p = p, var_map = var_map)

> **Contract**
>
> The layout MUST come from a `build_evaluator` of the SAME source (same
> document / `FlattenedSystem`, same `model_name`) with the SAME
> `build_kwargs`. Only the cheap parts of that are checked (the state count,
> that the slots are a permutation, that every row name resolves); a layout
> from a build that merely *differs* — other `const_arrays`, other
> `parameter_overrides`, another grid — would put entries in the wrong cells
> with no visible symptom. The default, `nothing`, rebuilds.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/EarthSciASTDiff.jl/src/assemble.jl#L213)

**Example** (`pattern`)

```julia
pat = jacobian_pattern(file)
# 4 cells x 6 variable pairs, one stored entry each, even where a value is 0.
@assert nnz(pat.pattern) == 24
@assert pat.pattern[1, 1]                    # A[1] depends on A[1]
@assert !pat.pattern[1, 2]                   # but not on A[2]
@assert pat.rownames[1] == "A[1]"
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/docs/api/examples/julia/jacobians.jl#L21-L26">docs/api/examples/julia/jacobians.jl</a></p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
jacobian_pattern(obj, model_name: str | None = None, wrt: str = 'states', forcing=None)
```

The structural sparsity pattern as sorted (rowname, colname) pairs
plus the entries — dependency-light (no scipy). On the `"parameters"`
and `"forcing"` axes the columns are cells of the canonical column
space ([`jacobian_columns`](/generated/api/ref/jacobian_columns)): a forcing column must resolve to one,
and a parameter column outside its variable's extent is a ghost read.

[Source](https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/pkg/earthsci-astdiff-py/src/earthsci_astdiff/emit.py#L122)

**Example** (`pattern`)

```python
pairs, _ = earthsci_astdiff.jacobian_pattern(file)
# 4 cells x 6 variable pairs, one stored entry each, even where a value is 0.
assert len(pairs) == 24
assert ("A[1]", "B[1]") in pairs
assert ("A[1]", "A[2]") not in pairs
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/docs/api/examples/python/jacobians.py#L29-L33">docs/api/examples/python/jacobians.py</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

<p class="api-absent">Rust does not export this function.</p>

</div>

</div>
