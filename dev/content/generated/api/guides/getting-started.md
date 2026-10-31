---
title: "Getting started"
description: "Differentiate a model, read its sparsity pattern and structure, evaluate the Jacobian, and serialize it as a document block."
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

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/docs/api/examples/julia/jacobians.jl#L10-L15">docs/api/examples/julia/jacobians.jl</a></p>

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

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/docs/api/examples/python/jacobians.py#L13-L24">docs/api/examples/python/jacobians.py</a></p>

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

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/docs/api/examples/rust/jacobians.rs#L16-L27">docs/api/examples/rust/jacobians.rs</a></p>

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

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/docs/api/examples/julia/jacobians.jl#L19-L24">docs/api/examples/julia/jacobians.jl</a></p>

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

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/docs/api/examples/python/jacobians.py#L29-L33">docs/api/examples/python/jacobians.py</a></p>

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

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/docs/api/examples/julia/jacobians.jl#L28-L30">docs/api/examples/julia/jacobians.jl</a></p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
assert earthsci_astdiff.detect_structure(entries, view) == "block_diagonal"
plan = earthsci_astdiff.plan_factorization(entries, view)
assert sorted(plan.block_vars) == ["A", "B", "C"]
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/docs/api/examples/python/jacobians.py#L38-L40">docs/api/examples/python/jacobians.py</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
let ctx = ctx_of(&view)?;
assert_eq!(detect_structure(&entries, &ctx), "block_diagonal");
let plan = block_diagonal_lu(&entries);
let mut vars = plan.block_vars.clone();
vars.sort();
assert_eq!(vars, ["A", "B", "C"]);
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/docs/api/examples/rust/jacobians.rs#L31-L36">docs/api/examples/rust/jacobians.rs</a></p>

</div>

</div>


See [`detect_structure`](/generated/api/ref/detect_structure) and
[`plan_factorization`](/generated/api/ref/plan_factorization).

## Evaluate

The one-shot call evaluates the Jacobian at a point, using the document's
initial state and parameters unless you pass others.


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
res = assemble_jacobian(file)                # at the document's initial state
@assert res.structure == :block_diagonal
@assert res.J[1, 1] == -0.04                 # -k1
@assert res.J[5, 5] == -3.0                  # -2 k2 B, with B = 0.5 by default
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/docs/api/examples/julia/jacobians.jl#L34-L37">docs/api/examples/julia/jacobians.jl</a></p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
values = earthsci_astdiff.assemble_jacobian(file, states={"A": [2.0] * 4})
assert values[("A[1]", "A[1]")] == -0.04  # -k1
assert values[("B[1]", "B[1]")] == -3.0  # -2 k2 B, with B = 0.5 by default
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/docs/api/examples/python/jacobians.py#L45-L47">docs/api/examples/python/jacobians.py</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
let env = Env {
    states: HashMap::from([
        ("A".to_string(), json!([2.0, 2.0, 2.0, 2.0])),
        ("B".to_string(), json!([0.5, 0.5, 0.5, 0.5])),
        ("C".to_string(), json!([0.1, 0.1, 0.1, 0.1])),
    ]),
    params: HashMap::from([
        ("k1".to_string(), json!(0.04)),
        ("k2".to_string(), json!(3.0)),
        ("k3".to_string(), json!(1.0)),
    ]),
    t: 0.0,
};
let values = assemble_jacobian(&file, None, Wrt::States, &env)?;
assert_eq!(values[&("A[1]".to_string(), "A[1]".to_string())], -0.04); // -k1
assert_eq!(values[&("B[1]".to_string(), "B[1]".to_string())], -3.0); // -2 k2 B
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/docs/api/examples/rust/jacobians.rs#L40-L55">docs/api/examples/rust/jacobians.rs</a></p>

</div>

</div>


See [`assemble_jacobian`](/generated/api/ref/assemble_jacobian).

## Prepare once, evaluate many times

To use the Jacobian inside a solver, differentiate and compile once, then fill
a fixed sparse prototype in place at each step.


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

```julia
jac = prepare_jacobian(file)                 # differentiate and compile once
J = jac_prototype(jac)                       # sparse matrix on the fixed pattern
u = fill(2.0, 12)
p = jac.p_default                            # the document's parameter values
jac(J, u, p, 0.0)                            # fill J in place, at any (u, p, t)
@assert size(J) == (12, 12)
@assert nnz(J) == 24
@assert J[1, 1] == -0.04
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/docs/api/examples/julia/jacobians.jl#L41-L48">docs/api/examples/julia/jacobians.jl</a></p>

</div>

</div>


See [`prepare_jacobian`](/generated/api/ref/prepare_jacobian).

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

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/docs/api/examples/julia/jacobians.jl#L52-L55">docs/api/examples/julia/jacobians.jl</a></p>

</div>

<div class="api-tab" data-lang="python" data-label="Python">
<p class="api-tab-label">Python</p>

```python
block = earthsci_astdiff.jacobian_block(file)
assert block["structure"] == "block_diagonal"
assert block["factorization"]["type"] == "block_diagonal_lu"
json.dumps(block)  # a plain, serializable dict
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/docs/api/examples/python/jacobians.py#L52-L55">docs/api/examples/python/jacobians.py</a></p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
let block = jacobian_block(&file, None, Wrt::States, true, 12)?;
assert_eq!(block["structure"], "block_diagonal");
assert_eq!(block["factorization"]["type"], "block_diagonal_lu");
```

<p class="api-src">From <a href="https://github.com/EarthSciML/EarthSciASTDiff/blob/a9ca79cb254eab4630ba5d3dc17b785c72fcfc79/docs/api/examples/rust/jacobians.rs#L59-L61">docs/api/examples/rust/jacobians.rs</a></p>

</div>

</div>


See [`jacobian_block`](/generated/api/ref/jacobian_block) and
[`jacobian_document`](/generated/api/ref/jacobian_document).
