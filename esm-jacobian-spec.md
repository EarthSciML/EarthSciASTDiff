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
(`wrt: "states"`), its parameters (`wrt: "parameters"`) or time itself
(`wrt: "time"`, §4.2), as a list of **bands** (§4). Coefficients are ordinary ESM expressions over the model's
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
| `datetime.year`, `.month`, `.day`, `.hour`, `.minute`, `.second`, `.day_of_year`, `.is_leap_year` | `0` (integer-valued, so piecewise-constant: the registry's a.e.-zero contract) |
| `datetime.julian_day(x)` | `d[x] / 86400` — the Julian day is continuous (it carries the fractional time of day, §9.2), one day per 86400 s of UTC time. A `datetime.*` name outside the §9.2 registry has no rule (`underivable_operator`). |
| `interp.searchsorted` | `0` (integer-valued) |
| `interp.linear(table, axis, x)` | `slope · d[x]`, where `slope` is the active segment's slope `(table[k+1] − table[k]) / (axis[k+1] − axis[k])` for `axis[k] ≤ x < axis[k+1]`, and **0 outside the axis** (extrapolate-flat, §9.2). Since `table`/`axis` MUST be literal `const` arrays (§9.2 `interp_table_not_const`), the producer precomputes the per-segment slopes and emits a nested segment `ifelse` (`ifelse(x < axis[k+1], sₖ, …)`, flat-zero guards at both ends) — scalar-evaluable, no dynamic table gather, only existing operators. |
| `interp.bilinear(table, axis_x, axis_y, x, y)` | `∂B/∂x · d[x] + ∂B/∂y · d[y]`. On cell `(i, j)` (located per §9.2's clamp + largest-knot-below convention), `∂B/∂x = sxᵢⱼ + wy · (sxᵢ,ⱼ₊₁ − sxᵢⱼ)` where `sxᵢⱼ = (table[i+1][j] − table[i][j]) / (axis_x[i+1] − axis_x[i])` and `wy` is the **clamped** y-weight; symmetrically for `∂B/∂y`. Outside `axis_x` the blend is constant in `x`, so `∂B/∂x = 0` there (and symmetrically for `y`), while the *other* partial persists at the clamped edge row/column. Since the table and axes MUST be literal `const` arrays, the producer precomputes the per-cell slope constants and emits each partial as a nested segment `ifelse` over the partial's own axis whose leaves are inner segment chains over the other axis (edge rows as the clamp leaves) — scalar-evaluable, no dynamic table gather, only existing operators. A query exactly on an interior knot selects the cell whose weight is 0, matching §9.2 step 2. |

### 3.5 Array operators

Derivatives of `faq` / `makearray` / elementwise-over-array expressions
are not expressed as rewritten array expressions; they are expressed as
**bands** (§4). `makearray` overlap follows §4.3.2: later regions overwrite
earlier ones, so a band from an overwritten sub-region MUST NOT be emitted
(equivalently: emission may lower a per-cell read of a `makearray` to nested
region-membership `ifelse`, folded first-region-outward).

Contracted (reduced) `faq` indices with `reduce: "+"` (explicit or
the default) differentiate per site of the faq body: a site whose
column indices are independent of every contracted name yields an ordinary
band whose `coef` is the scalar `reduce: "+"` of the site derivative over
the contraction; a site whose column indices reference a contracted name (a
gathered column) yields a band with `contracted` dimensions (§4), whose
points accumulate. Filtered faq nodes and relational joins are NOT
differentiated — producers MUST fail rather than drop the gate.

The same holds for a scalar `+`-reducing faq node NESTED in a
differentiated expression (Σ_k g inside f(…)) — the form a read of a
contracting observed takes. A faq node reduces by `+` when its `semiring` is
`"sum_product"`, or when it has no `semiring` and its `reduce` is absent or
`"+"`; a `semiring` supersedes `reduce`. A read `F[e…]` of an observed
defined by a faq node whose ranges bind indices beyond its `output_idx` MUST
inline to a scalar faq node over those contracted indices, carrying
`reduce`, `semiring` and `filter`. Inlining the bare body instead drops the
sum, and a faq around the reading site that binds the same name captures the
leftover index. All inlining substitution MUST be capture-avoiding, and no
nested faq node may bind a name a faq enclosing it already binds: producers
rename such a binder apart (the reference uses `_c<n>_<name>`, the smallest
`n` clear of every name in the document, which keeps repeated reads of one
observed structurally equal). A name a faq around the reading site binds is
that index, never a read of an observed with the same name. The nested sum
is linear, so its derivative is the sum of its body's derivatives: for a
site whose column does not read a bound index, the coefficient keeps that
sum; for a site whose column does read one (`u[i,k]` under Σ_k), that index
becomes a `contracted` dimension of the band (§4), renamed apart from the
sum's own bound name, since the coefficient may still contain the sum (the
chain rule through f(Σ_k …)). Producers MUST fail when a column reads a name
the sums around its site bind ambiguously (twice on the way down, over two
different ranges at different occurrences of the site, or shadowing the
band's own row or contraction index). A filtered or non-`+` nested sum is
reserved, as above. Fixture: `contracting_observed.esm`.

Reserved (producers MUST fail rather than guess): non-`"+"` semiring
reductions, filtered faq nodes, `argmin`/`argmax`, singleton `1` output
indices, `reshape`/`transpose`/`concat` operands.

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
- `col` — a state (`wrt: "states"`) or parameter (`wrt: "parameters"`)
  name, or `"t"` (`wrt: "time"`, §4.2).
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
- `contracted` (optional) — free CONTRACTED column dimensions, each a
  `[name, lo, hi]` triple (string name, inclusive integer bounds). The names
  are local to the entry (disjoint from its `row_idx` names) and may appear
  in `col_idx` and `coef`. Produced by differentiating through a
  `reduce: "+"` faq node whose gathered column depends on a contracted
  index (`u[j]`, `u[i+k]`, a `conn[]`-table gather): the sum over the
  contraction materializes as accumulation over the contracted points.
  Absent or `[]` for ordinary entries. When a contracted site's column does
  NOT depend on any contracted index, producers SHOULD instead emit an
  ordinary entry whose `coef` is a scalar `reduce: "+"` faq node over the
  contraction — keeping the entry count independent of the contraction
  extent.

**Meaning.** For every index point `r` in `rows` AND every point `c` of the
`contracted` box (the empty product when absent),

```
J[ row[r], col[col_idx(r, c)] ] += coef(r, c)
```

Entries are additive: multiple entries — and multiple contracted points of
ONE entry — may target the same (row, col) pair.
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

### 4.2 Time-derivative bands (`wrt: "time"`)

A block with `"wrt": "time"` carries `f_t = ∂f/∂t`, the derivative of the
right-hand side with respect to the independent variable — the term a
Rosenbrock stage adds when `f` depends on `t` explicitly. Its single column
is the scalar site `t`, so every entry has `"col": "t"`, `"col_idx": []` and
no `contracted` dimensions; `rows`, `row_idx` and `coef` are as for state
bands. The meaning is §4's with `J` an `N × 1` matrix: for every point `r`
of `rows`, `f_t[row[r]] += coef(r)`. Entry order is the state bands' order —
equation order, and the single column within each equation.

**Coverage (normative).**

- Every explicit occurrence of `t` in a right-hand side is differentiated,
  including occurrences inside the observed it reads: a producer MUST inline
  every observed variable the right-hand side reads, transitively, before
  differentiating. (The state and parameter axes prune the inlining to the
  observed that can depend on a target; the time axis does not — an
  observed's time dependence can also arrive through data its definition
  reads, so `{t}` is not a sound seed.)
- Closed functions of `t` follow §3.4: the integer `datetime.*` fields
  differentiate to 0, `datetime.julian_day` to `1/86400`, and
  `interp.linear` / `interp.bilinear` of `t` to the active segment's slope.
- `f_t` is the **partial** derivative with every parameter held fixed. Time
  dependence that enters only through data — two-record forcing blend weights
  a provider sets at each refresh, parameters a `data` source updates — is
  NOT in the band list, and a consumer that needs it MUST account for it
  itself (for example by not stepping across a data refresh). Bands with
  respect to forcing values are a separate, planned axis (§7). A blend weight
  that the model computes from `t` (a `frac` expression) is an explicit
  occurrence and IS covered.

**Autonomous systems.** An empty `"time"` entry list means `f_t ≡ 0`: no
row's right-hand side reads `t`. A consumer MAY skip evaluating `f_t` for
such a model and use the zero vector.

A `"jacobians"` block holds one `wrt` per model, so a document carries its
time bands separately from its state bands (a derived document per axis).

## 5. Conformance

Fixture corpus in `tests/` of this repository:

1. **Scalar rule goldens** — (expression, site) → canonical derivative JSON,
   one per operator row of §3, including the tie/branch policies.
2. **Band goldens** — per fixture model: the `entries` list in
   deterministic JSON (sorted object keys; an integer node as an integer
   literal; a float node in shortest round-trip digits, positional iff the
   first significant digit's decimal exponent is in `-4..5`, else
   `d.f…e<exp>`, always with at least one fraction digit — `0.0`, `1.0`,
   `100.0`, `1.0e6`, `2.4405875e6`, `7.27220521664304e-5`. This is the float
   layout of EarthSciAST CONFORMANCE_SPEC §5.5.3.1 rule 2, but an integral
   float stays a float: `1.0` is never written `1`), byte-compared across
   bindings. Each fixture
   `<name>` carries `<name>.bands.json`, `<name>.block.json` (§4 block),
   `<name>.evaldoc.json` (§8 evaluation document; `-nocse` when it differs
   without hoisting) and `<name>.jvals.json` (values, item 3). A fixture
   whose right-hand side reads `t` also carries the same four for
   `wrt: "time"` as `<name>.time.*.json` (`tgrad_photolysis.esm`: a
   photolysis rate `max(0, cos(ω t + φᵢ))` through an array observed,
   `interp.linear` of `t` through a scalar observed, a `datetime.hour`
   factor, a `datetime.julian_day` source, and one autonomous row).
3. **Value oracle** — per fixture model and sample point: the assembled
   Jacobian must match a machine-precision reference (AD through the
   binding's evaluator, or the shipped reference values) to `rtol 1e-12`,
   and the structural pattern must be a superset of the local pattern.
   Shipped value goldens (`*.jvals.json`) are compared with exactly the same
   (row, column) key set and `rtol 1e-10` (evaluation order differs between
   runtimes).

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

Second derivatives and derivatives with respect to forcing buffers / const
arrays (and so the data-borne time dependence §4.2 excludes) are planned
extensions; each will be specified with fixtures before implementation is
required. (Time-derivative bands, contracted-index bands and
`expression_templates` coefficient sharing, once listed here, are specified
in §4.2, §3.5/§4 and §4.1.)

## 8. Scatter plans (conformance artifact)

A **scatter plan** maps the band fields of the derived evaluation document
(one field `J_k` per entry, emitted with the block's band list) to a
consumer's own Jacobian storage. It is not part of the `"jacobians"` block —
a plan is derived from the band list, so any producer can compute it — but
its serialization is a conformance artifact (`tests/goldens/<fixture>.<layout>.plan.json`),
byte-identical across bindings, so that the storage map can be checked
rather than re-derived by each consumer.

**Conventions.** Every index in a plan is 0-based.

- *Band field `k`, cell `f`*: field `J_{k+1}`, whose shape is the row
  variable's shape followed by each contracted dimension's extent (`1..hi`);
  `f` is the column-major flat index of a cell of that shape. A field is
  zero outside its band's region.
- *Canonical state order*: state names sorted, species-major, each state's
  cells column-major. A scalar state is one slot.
- *Ghost reads* — points whose column index falls outside the column
  variable — have no storage position and appear in no plan.

**Time bands.** The evaluation document for `wrt: "time"` is built the same
way from the time band list: one field `J_k` per entry, of the row variable's
shape (the scalar site `t` contributes no contracted dimension). No plan
layout is defined for it, since `f_t` is one dense vector: field `k`'s cell
`f` (column-major, as for the state Jacobian fields above) adds into the
canonical-state-order position of state `row_k`'s cell `f` (the state's
offset plus `f`, both column-major, so no permutation is involved), and every
cell outside the band's region reads zero.

**Common fields.** `layout` (the tag below), `wrt` (`"states"`; plans cover
the state Jacobian only), `states` (`[{name, shape}]` in canonical order),
`bands` (`[{u, v, shape}]`: row variable, column variable and field shape
per entry, in entry order), and `items` (layout-specific tuples, in
accumulation order). A consumer MUST refuse a plan whose `states` or `bands`
differ from those of the evaluator it binds to (field shapes compared
dimension by dimension, not by cell count). Equal `states` and `bands` do
not make a plan correct — two stencils with the same variables have the same
field shapes — so binding MUST also decode every contribution against the
evaluator's own bands: each contribution that reads a band point MUST land on
the (row, column) cell that point differentiates, no point may be read
twice or given a position if it is a ghost read, and every non-ghost point
MUST be read. Contributions that read a field cell outside its band's region
read zero and are allowed.

Accumulation is `+=` in `items` order into an output zeroed first: several
bands may land on one element (§4 additivity). Items are sorted by output
position, then band, then field cell, so each element's contributions sum
in band order.

**Layouts.** `L` is the lane capacity, chosen by the consumer at bind time
(`L ≥` the number of lanes); lane `ℓ` of a lane layout is the cell with flat
index `ℓ` of the lane domain, and padding lanes stay zero.

| `layout` | Extra fields | `items` | Position of a contribution |
|---|---|---|---|
| `lane_blocks` | `cells`, `block_vars` | `[k, o]`, `o = r + n·c` | `o·L + ℓ`, reading field `k` cell `ℓ` |
| `lane_sparse_blocks` | `cells`, `block_vars`, `lu_pattern` | `[k, q]` | `q·L + ℓ`, reading field `k` cell `ℓ` |
| `column_banded` | `cells`, `axis`, `axis_dim`, `vars`, `kl`, `ku` | `[k, lev, o]` | `o·L + ℓ`, reading field `k` at row level `lev` of column `ℓ` |
| `csr` | `rowptr`, `colind` | `[k, f, p]` | `p` (the nonzero index), reading field `k` cell `f` |

- `lane_blocks` / `lane_sparse_blocks` require a `block_diagonal` structure
  with every state on one cell shape, and no band with contracted dimensions
  (a lane reads one field cell, so a contracted band's other points would
  be dropped). `block_vars` is the §6 plan's
  `block_vars` (Markowitz order) followed by any state no band touches,
  sorted; `n = len(block_vars)`. `lu_pattern` is the §6 `lu_pattern` made
  0-based, plus a diagonal `[i, i]` for each appended state, sorted; fill
  positions receive no items and are therefore zero.
- `column_banded` requires every state to be an array of one shape that
  contains index set `axis` (dimension `axis_dim`), and every band to couple
  cells only along `axis`, at a literal offset `d = col − lev`
  (`structure.jl` `:diag` / `:affine`). Columns (lanes) enumerate the other
  dimensions column-major. `vars` is the canonical state order, `N` the
  axis size, `kl`/`ku` the largest sub-/super-diagonal offsets, and
  `o = (((c·nvar + r)·(kl+ku+1) + (d+kl))·N + lev)`.
- `csr` applies to every structure. Rows and columns are the canonical state
  order; `rowptr` has one more entry than there are rows, `colind` is sorted
  within each row, and duplicate (row, column) pairs are merged.

A producer MUST refuse (not emit) a plan for an empty band list on a model
with array states, for a band column that is not a state, and for a layout
whose preconditions above do not hold.
