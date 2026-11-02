---
title: "Getting started"
description: "Differentiate a model, read its sparsity pattern and structure, emit the Jacobian as documents EarthSciAST evaluates, and serialize it as a document block."
weight: 10
api_tabs: true
---

EarthSciASTDiff turns an EarthSciAST (`.esm`) model into its analytical sparse
Jacobian by symbolic differentiation. This guide takes one small model through
each step. Each step shows the same task in every binding; pick your language
on any tab and the rest of the site follows. Every snippet is cut from a
program that runs, and checks its own result, in CI.

The examples use `bd_chem.esm` from
[`tests/valid/`](https://github.com/EarthSciML/EarthSciASTDiff/tree/main/tests/valid):
per-cell chemistry among three species `A`, `B` and `C` on a four-cell column,
so the Jacobian is block-diagonal: four blocks of three species.

## Differentiate

The Jacobian is a list of **bands**: each is a rectangular region of the
equation's index space, a symbolic column-index expression, and a coefficient
expression. One entry is produced for each (row variable, column variable)
pair that occurs in the model, so the derivative is exact and does not depend
on the point where you later evaluate it.


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
file = EarthSciAST.load_path(joinpath(FIXTURES, "bd_chem.esm"))
view = sysview(file, "Chem")
entries = jacobian_bands(view)
# One entry per (row variable, column variable) pair that occurs in the model.
@assert sort(unique([(en.u, en.v) for en in entries])) ==
    [("A", "A"), ("A", "B"), ("A", "C"), ("B", "A"), ("B", "B"), ("C", "B")]
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/docs/api/examples/julia/jacobians.jl#L12-L17">docs/api/examples/julia/jacobians.jl</a></p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
file = earthsci_ast.load_path(FIXTURES / "bd_chem.esm")
entries = earthsci_astdiff.jacobian_bands(file)
# One entry per (row variable, column variable) pair that occurs in the model.
assert sorted({(en.u, en.v) for en in entries}) == [
    ("A", "A"),
    ("A", "B"),
    ("A", "C"),
    ("B", "A"),
    ("B", "B"),
    ("C", "B"),
]
view = earthsci_astdiff.sysview(file)
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/docs/api/examples/python/jacobians.py#L13-L24">docs/api/examples/python/jacobians.py</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
let file = load_path(fixture("bd_chem.esm"))?;
let entries = jacobian_bands(&file, None, Wrt::States)?;
// One entry per (row variable, column variable) pair that occurs in the model.
let mut pairs: Vec<(&str, &str)> =
    entries.iter().map(|en| (en.u.as_str(), en.v.as_str())).collect();
pairs.sort();
pairs.dedup();
assert_eq!(
    pairs,
    [("A", "A"), ("A", "B"), ("A", "C"), ("B", "A"), ("B", "B"), ("C", "B")]
);
let view = sysview(&file, None)?;
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/docs/api/examples/rust/jacobians.rs#L13-L24">docs/api/examples/rust/jacobians.rs</a></p>

</div>

</div>


See [`jacobian_bands`](/generated/api/ref/jacobian_bands) and
[`sysview`](/generated/api/ref/sysview).

## Sparsity pattern

The pattern is read off structural occurrence, never off numeric probing, so it
is valid at every point. Branches of `ifelse`, `min` and `max` are unioned, and
entries that happen to be zero at one point stay in the pattern.


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

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
pairs, _ = earthsci_astdiff.jacobian_pattern(file)
# 4 cells x 6 variable pairs, one stored entry each, even where a value is 0.
assert len(pairs) == 24
assert ("A[1]", "B[1]") in pairs
assert ("A[1]", "A[2]") not in pairs
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/docs/api/examples/python/jacobians.py#L29-L33">docs/api/examples/python/jacobians.py</a></p>

</div>

</div>


See [`jacobian_pattern`](/generated/api/ref/jacobian_pattern).

## Structure and factorization

The bands are classified as block-diagonal, banded or general, which tells a
solver what matrix type to use. For block-diagonal systems a symbolic
per-cell block LU plan (Markowitz ordering and fill-in) is available too.


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
@assert detect_structure(entries, view) == :block_diagonal
plan = plan_factorization(entries, view)
@assert sort(plan.block_vars) == ["A", "B", "C"]
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/docs/api/examples/julia/jacobians.jl#L30-L32">docs/api/examples/julia/jacobians.jl</a></p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
assert earthsci_astdiff.detect_structure(entries, view) == "block_diagonal"
plan = earthsci_astdiff.plan_factorization(entries, view)
assert sorted(plan.block_vars) == ["A", "B", "C"]
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/docs/api/examples/python/jacobians.py#L38-L40">docs/api/examples/python/jacobians.py</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
let ctx = ctx_of(&view)?;
assert_eq!(detect_structure(&entries, &ctx), "block_diagonal");
let Some(FactorizationPlan::BlockDiagonal(plan)) =
    plan_factorization(&entries, &view, &ctx, None)?
else {
    unreachable!("a block-diagonal Jacobian has a block LU plan")
};
let mut vars = plan.block_vars.clone();
vars.sort();
assert_eq!(vars, ["A", "B", "C"]);
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/docs/api/examples/rust/jacobians.rs#L28-L37">docs/api/examples/rust/jacobians.rs</a></p>

</div>

</div>


See [`detect_structure`](/generated/api/ref/detect_structure) and
[`plan_factorization`](/generated/api/ref/plan_factorization).

## Evaluate

EarthSciASTDiff evaluates nothing itself. It emits the Jacobian as derived
ESM documents, and EarthSciAST evaluates them like any other model: the band
fields `J_k` (the evaluation document), and the matrix-free products `J·v`,
`Jᵀ·λ` and the second-order contraction (spec §9).


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
# EarthSciASTDiff evaluates nothing: it emits ESM documents that EarthSciAST
# evaluates. The band fields J_k, as a derived model whose right-hand side,
# run once, leaves every band value in `du`. `fields[k]` names entry k's field.
doc, fields = evaluation_document(file)
@assert length(fields) == length(entries)
prob = EarthSciAST.esm_problem(
    JSON3.read(stable_json(doc)), (0.0, 1.0)
)
du = zero(prob.u0)
prob.f!(du, prob.u0, prob.p, 0.0)            # at the document's initial state
k = findfirst(en -> (en.u, en.v) == ("A", "A"), entries)
@assert du[prob.var_map["JacobianEval.$(fields[k])[1]"]] == -0.04   # -k1
# J·v as a derived model: inputs `_v_<C>`, outputs `_jvp_<R>`.
jdoc, (model, v, jvp) = jvp_document(file)
@assert model == "JvpEval"
J = jac_prototype(file)                      # sparse matrix on the fixed pattern
@assert size(J) == (12, 12) && nnz(J) == 24
# A scatter plan places the band fields in your own storage (here CSR),
# checked against the source before it is used.
bound = bind_plan(file, scatter_plan(file, Csr()))
@assert bound.len == 24
@assert all(1 <= k <= length(fields) for (k, _, _) in bound.pairs)
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/docs/api/examples/julia/jacobians.jl#L36-L57">docs/api/examples/julia/jacobians.jl</a></p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
# EarthSciASTDiff evaluates nothing: it emits the band fields `J_k` as a derived
# ESM model that EarthSciAST evaluates.
doc, fields = earthsci_astdiff.evaluation_document(file)
assert "JacobianEval" in doc["models"]
assert len(fields) == len(entries)
# `J v` likewise: write `v` into the inputs io["v"], read `J v` from the
# derivatives of the outputs io["jvp"].
jdoc, io = earthsci_astdiff.jvp_document(file)
assert io["model"] in jdoc["models"]
assert io["v"]["A"] == "_v_A" and io["jvp"]["A"] == "_jvp_A"
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/docs/api/examples/python/jacobians.py#L45-L54">docs/api/examples/python/jacobians.py</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
// EarthSciASTDiff evaluates nothing: it emits the band fields `J_k` as a
// derived ESM model that EarthSciAST evaluates.
let (doc, fields) = evaluation_document(&view, &entries, true, 12)?;
assert!(doc["models"]["JacobianEval"].is_object());
assert_eq!(fields.len(), entries.len());
// `J v` likewise: write `v` into the inputs `io.v`, read `J v` from the
// derivatives of the outputs `io.jvp`.
let (jdoc, io) = jvp_document(&view, &entries, Wrt::States, &ProductOptions::default())?;
assert!(jdoc["models"][io.model.as_str()].is_object());
assert_eq!(io.v["A"], "_v_A");
assert_eq!(io.jvp["A"], "_jvp_A");
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/docs/api/examples/rust/jacobians.rs#L41-L51">docs/api/examples/rust/jacobians.rs</a></p>

</div>

</div>


See [`evaluation_document`](/generated/api/ref/evaluation_document),
[`jvp_document`](/generated/api/ref/jvp_document) and
[`jac_prototype`](/generated/api/ref/jac_prototype).

## Serialize

The bands can be written into the document itself as a `jacobians` block, so
any binding that can evaluate `.esm` expressions can evaluate the Jacobian.
The block carries the entries, the structure classification and the
factorization plan.


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
doc = jacobian_document(file, "Chem")
block = doc["jacobians"]["Chem"]
@assert block["structure"] == "block_diagonal"
@assert block["factorization"]["type"] == "block_diagonal_lu"
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/docs/api/examples/julia/jacobians.jl#L61-L64">docs/api/examples/julia/jacobians.jl</a></p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
block = earthsci_astdiff.jacobian_block(file)
assert block["structure"] == "block_diagonal"
assert block["factorization"]["type"] == "block_diagonal_lu"
json.dumps(block)  # a plain, serializable dict
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/docs/api/examples/python/jacobians.py#L59-L62">docs/api/examples/python/jacobians.py</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
let block = jacobian_block(&file, None, Wrt::States, true, 12)?;
assert_eq!(block["structure"], "block_diagonal");
assert_eq!(block["factorization"]["type"], "block_diagonal_lu");
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/2a9b4a92b2b429cd94fc2ddac236c438425d94f9/docs/api/examples/rust/jacobians.rs#L55-L57">docs/api/examples/rust/jacobians.rs</a></p>

</div>

</div>


See [`jacobian_block`](/generated/api/ref/jacobian_block) and
[`jacobian_document`](/generated/api/ref/jacobian_document).
