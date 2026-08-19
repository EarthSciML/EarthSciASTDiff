# ESM Analytical Jacobians — Draft Specification

**Status: DRAFT.** This document specifies the `"jacobians"` top-level block
and the derivative semantics of every evaluable-core operator. It is
developed in this repository and is intended to be merged into
[`esm-spec.md`](https://github.com/EarthSciML/EarthSciAST) (the block as a new
top-level section, the derivative table as an appendix) once stable. Until
then, bindings MUST ignore an unrecognized `"jacobians"` block.

## 1. Overview and scope

A **Jacobian document** is an ordinary ESM document plus a `"jacobians"`
top-level block that carries, per model, the analytical derivative of the
model's ODE right-hand side with respect to its state variables
(`wrt: "states"`) or parameters (`wrt: "parameters"`), as a list of
**bands** (§4). Coefficients are ordinary ESM expressions over the model's
own variables, so any binding that evaluates ESM expressions evaluates the
Jacobian — no binding needs a differentiator to *consume* the block.

Producing the block requires the derivative semantics of §3, defined on the
**expanded, lowered** expression tree: expression templates expanded
(esm-spec §9.6.4), rewrite-target operators (spatial `D`, `grad`,
`table_lookup`, …) already rewritten away. Differentiating an unlowered tree
is a conformance violation, mirroring the `unlowered_operator` evaluation
gate.

Consistent with esm-spec §4.3.6, the block describes symbolic *structure*
(regions, index expressions); the choice of a runtime matrix type
(block-diagonal, banded, CSC, …) remains a runtime concern. §6 defines an
optional, structure-agnostic `"factorization"` field carrying symbolic
factorization information.

## 2. Sparsity semantics: structural, global

The nonzero pattern implied by a band list is **structural**: an entry exists
wherever the column variable *occurs* in the row equation's RHS, regardless
of whether the derivative value happens to be zero at some point.
Consequences (normative):

- `ifelse`, `min`, `max`: the pattern is the **union of all branches**. The
  pattern is therefore valid at every state point ("global"), and a consumer
  may cache factorization symbolics across the whole integration.
- An entry whose coefficient simplifies to a constant zero MAY be dropped by
  the producer; a consumer MUST NOT rely on such dropping.
- The pattern is a superset of any local (single-point) pattern. Producers
  MUST NOT emit a pattern derived from numeric probing at a point.

**Region exactness.** Reads of region-structured arrays
(`index(makearray…)`, directly or through an observed field) lower to
region-membership conditionals whose conditions are affine in the entry's
row indices with literal bounds. Such conditions are statically decidable:
producers SHOULD split entries at the conditions' breakpoints and decide
them exactly, so region structure appears as entry `rows` boundaries — never
as `ifelse` guards inside `coef`, and never as branch-union padding in the
pattern. Only genuinely value-dependent branches (limiters) remain unioned.

## 3. Derivative semantics (per-operator, normative)

`d[e, s]` below is the derivative of expression `e` with respect to a
**site** `s`. A site is a bare variable reference or an `index(v, i…)` node;
two nodes denote the same site iff their canonical forms (RFC §5.4) are
equal. `d[s, s] = 1`; `d[e, s] = 0` whenever `s` does not occur in `e`
(structural short-circuit).

### 3.1 Smooth core

Textbook rules, with these bindings-relevant details:

| op | rule |
|---|---|
| `+` (n-ary) | `Σ d[aᵢ]` |
| `-` (unary/binary), `neg` | `−d[a₁]` / `d[a₁] − d[a₂]` |
| `*` (n-ary) | `Σᵢ d[aᵢ]·Πⱼ≠ᵢ aⱼ` |
| `/` | `d[n]/m − n·d[m]/m²` |
| `^`, `pow` | `x·b^(x−1)·d[b] + b^x·log(b)·d[x]`; a term with structurally-zero inner derivative MUST be omitted (so a constant exponent never evaluates `log(b)`) |
| `exp log log10 sqrt sin cos tan asin acos atan`(1–2)`/atan2 sinh cosh tanh asinh acosh atanh` | standard chain rule |

### 3.2 Non-smooth operators (policy)

| op | rule | note |
|---|---|---|
| `ifelse(c, a, b)` | `ifelse(c, d[a], d[b])` | the condition's Dirac term is dropped |
| `max`/`min` (n-ary) | left fold; `d[max(p,q)] = ifelse(p >= q, d[p], d[q])` (`<=` for `min`) | a.e. derivative; **ties resolve to the earlier argument** |
| `abs` | `ifelse(a < 0, −1, 1) · d[a]` | subgradient −1/+1 at 0 per the tie rule |
| comparisons, `and`/`or`/`not` | `0` | comparisons return 1.0/0.0 *values*; still 0 even in multiplicative position |
| `sign`, `floor`, `ceil` | `0` | a.e. |
| `Pre`, `const`, `enum`, niladic constants | `0` | |

### 3.3 Rewrite-target operators

Spatial `D`, `grad`, `div`, `laplacian`, `integral`, `table_lookup`, and any
op outside the evaluable core: **no derivative rule exists**. A producer
encountering one MUST fail (diagnostic `underivable_operator`), never emit 0.

### 3.4 Closed functions (esm-spec §9.2)

| name | rule |
|---|---|
| `datetime.*` | `0` (piecewise-constant by the registry's a.e.-zero contract) |
| `interp.searchsorted` | `0` (integer-valued) |
| `interp.linear(table, axis, x)` | `slope · d[x]`, where `slope` is the active segment's slope `(table[k+1] − table[k]) / (axis[k+1] − axis[k])` for `axis[k] ≤ x < axis[k+1]`, and **0 outside the axis** (extrapolate-flat, §9.2). Since `table`/`axis` MUST be literal `const` arrays (§9.2 `interp_table_not_const`), the producer precomputes the per-segment slopes and emits a nested segment `ifelse` (`ifelse(x < axis[k+1], sₖ, …)`, flat-zero guards at both ends) — scalar-evaluable, no dynamic table gather, only existing operators. |
| `interp.bilinear` | the two-axis analogue (one slope per axis); RESERVED until fixtures pin it |

### 3.5 Array operators

Derivatives of `aggregate` / `makearray` / elementwise-over-array expressions
are not expressed as rewritten array expressions; they are expressed as
**bands** (§4). `makearray` overlap follows §4.3.2: later regions overwrite
earlier ones, so a band from an overwritten sub-region MUST NOT be emitted
(equivalently: emission may lower a per-cell read of a `makearray` to nested
region-membership `ifelse`, folded first-region-outward).

Reserved (producers MUST fail rather than guess): contracted (reduced)
`aggregate` indices, non-`"+"` semiring reductions, `argmin`/`argmax`,
singleton `1` output indices, `reshape`/`transpose`/`concat` operands.

## 4. The `"jacobians"` top-level block

```json
"jacobians": {
  "<ModelName>": {
    "wrt": "states",
    "entries": [
      {
        "row": "c",
        "col": "c",
        "rows": [[2, 4], [1, 3]],
        "row_idx": ["i", "j"],
        "col_idx": [ {"op": "+", "args": ["i", 1]}, "j" ],
        "coef": {"op": "/", "args": [1, {"op": "*", "args": [2, "dx"]}]}
      }
    ],
    "structure": "banded",
    "factorization": { "...": "see §6" }
  }
}
```

Each **entry** (band) is one structured block of `∂(RHS of the equation for
row)/∂(col)`:

- `row` — a state variable name (the equation's integrated variable).
- `col` — a state (`wrt: "states"`) or parameter (`wrt: "parameters"`) name.
- `rows` — a rectangular region of `row`'s index space, one inclusive
  `[lo, hi]` integer pair per dimension; `[]` for a scalar `row`.
- `row_idx` — per dimension: a free index NAME (string) ranging over the
  corresponding `rows` pair, or an integer for a singleton dimension. Names
  are local to the entry.
- `col_idx` — `col`'s index expressions, one per dimension of `col`, over the
  `row_idx` names; `[]` for a scalar `col`.
- `coef` — a scalar Expression over the `row_idx` names, the model's
  variables and parameters, and `t`. Literals MUST be in the wire-canonical
  form of the ESM conformance rules (CONFORMANCE_SPEC §5.5.3.1: an integral,
  Int64-representable float literal is an integer literal) — producers whose
  in-memory simplification folds mixed int/float constants must normalize
  before emission, or parse round-trips and cross-binding goldens diverge.

**Meaning.** For every index point `r` in `rows`,

```
J[ row[r], col[col_idx(r)] ] += coef(r)
```

Entries are additive: multiple entries may target the same (row, col) pair.
A `col_idx(r)` that falls outside `col`'s declared extent contributes nothing
(it denotes a boundary/ghost read that is not a state column).

`structure` (optional, derived): `"block_diagonal"` | `"banded"` |
`"general"` — a *classification* of the exact pattern for consumers choosing
a matrix type; the pattern itself is always recoverable from the entries.

**Round-trip.** The block is derived data: producers MUST be able to
regenerate it from the document, and `Expand(document)` conformance (§9.6.4)
is unaffected by its presence.

### 4.1 Shared coefficient fragments: `expression_templates`

Limiter-heavy schemes (minmod, superbee, PPM) produce coefficient
expressions whose subtrees repeat heavily across entries. A block MAY carry
an `expression_templates` field factoring them out:

```json
"jacobians": {
  "<ModelName>": {
    "wrt": "states",
    "expression_templates": {
      "jt1": { "params": [], "body": { "op": "...", "args": ["..."] } }
    },
    "entries": [ { "coef": {"op": "apply_expression_template",
                            "args": [], "name": "jt1"}, "...": "..." } ]
  }
}
```

Each entry is a **zero-parameter, match-less** template in the sense of
esm-spec §9.6 (`params` MUST be `[]`; no `match` field): a fixed named
fragment whose expansion is pure syntactic substitution of `body` for the
`apply_expression_template` reference. Free index names in a `body` (entry
`row_idx` names) re-bind at the reference site — extraction is exact subtree
factoring, not hygienic abstraction.

- References may appear in entry `coef` (and, in principle, `col_idx`)
  expressions and inside other template bodies. The reference graph MUST be
  acyclic (it is, by construction, when templates are extracted by strict
  subtree containment).
- Consumers MUST expand all references before evaluating a coefficient; a
  reference to a name absent from the block's `expression_templates` is an
  error.
- The field is an emission-size optimization only: a block with templates
  and its fully-expanded form denote the same Jacobian, and producers MAY
  emit either. Deterministic producers SHOULD extract greedily
  largest-first with deterministic naming (`jt1`, `jt2`, … in extraction
  order) so byte-comparison goldens are stable.

## 5. Conformance

Fixture corpus in `tests/` of this repository:

1. **Scalar rule goldens** — (expression, site) → canonical derivative JSON,
   one per operator row of §3, including the tie/branch policies.
2. **Band goldens** — per fixture model: the `entries` list in
   deterministic JSON (sorted object keys), byte-compared across bindings.
3. **Value oracle** — per fixture model and sample point: the assembled
   Jacobian must match a machine-precision reference (AD through the
   binding's evaluator, or the shipped reference values) to `rtol 1e-12`,
   and the structural pattern must be a superset of the local pattern.

## 6. Factorization information (optional)

The `"factorization"` field carries **symbolic factorization info** so a
consumer can factorize without runtime symbolic analysis. The field is a
tagged object; the `type` tag names the layout, and new types may be added
per structure class (banded, general CSC, …) without changing existing ones.
The first defined type:

```json
"factorization": {
  "type": "block_diagonal_lu",
  "block_vars": ["NO", "O3", "NO2"],
  "ordering":   [2, 1, 3],
  "pattern":    [[1,1], [1,2], [2,1], [2,2], [3,2], [3,3]],
  "lu_pattern": [[1,1], [1,2], [2,1], [2,2], [3,2], [3,3]]
}
```

- `block_vars` — the per-cell block's variables in permuted block order
  (after applying `ordering` to the producer's canonical variable sort).
- `ordering` — the elimination ordering (a permutation), chosen by the
  producer (e.g. Markowitz) to reduce fill-in; consumers MUST apply it.
- `pattern` — the block's structural pattern as `[row, col]` pairs
  (1-based, permuted order), diagonal included.
- `lu_pattern` — the no-pivoting LU pattern **including fill-in** under
  `ordering`; superset of `pattern`. This is the allocation a straight-line
  LU (KPP-style) needs.

The field is advisory: a consumer MAY ignore it and factorize however it
likes. A producer MUST NOT emit a `factorization` whose `lu_pattern` is not
a fill-correct superset of `pattern`.

## 7. Out of scope (this draft)

Time gradients (`tgrad`), second derivatives, derivatives with respect to
forcing buffers / const arrays, contracted-index bands, and
CSE/`expression_templates`-based coefficient sharing are planned extensions;
each will be specified with fixtures before implementation is required.
