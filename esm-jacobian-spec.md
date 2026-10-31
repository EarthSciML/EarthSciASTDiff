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
(`wrt: "states"`), its parameters (`wrt: "parameters"`, array-valued ones
cell by cell), the cells of its live forcing buffers (`wrt: "forcing"`,
§4.3) or time itself (`wrt: "time"`, §4.2), as a list of **bands** (§4),
and optionally the second derivatives a Rosenbrock adjoint contracts
(`wrt: [a1, a2]`, §4.4). Coefficients are ordinary ESM expressions over the
model's own variables, so any binding that evaluates ESM expressions
evaluates the Jacobian — no binding needs a differentiator to *consume* the
block.

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
(structural short-circuit). An `index` node whose subscripts hold an array
operator the canonical form refuses (a level search used as a subscript,
§3.5) is keyed by `~` followed by its §5 deterministic JSON instead; sites
are visited in ascending key order, which fixes entry order.

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
points accumulate. Relational joins and value-invention nodes (`distinct`,
`key`) and the `bool_and_or` semiring are NOT differentiated — producers MUST
fail. A faq node's reduction ⊕ is its `semiring`'s when it has one
(`sum_product` → `+`, `max_product` and `max_sum` → `max`, `min_sum` →
`min`; its ⊗ is already written out in the body), otherwise its `reduce`
(absent → `+`).

**Filters and non-`+` reductions.** A `filter` is a gate whose Dirac term is
dropped, exactly like an `ifelse` condition (§3.2): the derivative of a
reduction over the window `W(o) = {c : φ(o, c)}` is a `+` over the same
window of each term's derivative `d(o, c) = d[b(o, c), s]`, weighted by what
⊕ does with that term:

| ⊕ | weighted term |
|---|---|
| `+` | `d` |
| `max` | `ifelse(and(b >= M, b > M<), d, 0)` |
| `min` | `ifelse(and(b <= M, b < M<), d, 0)` |
| `*` | `d · E` |

where `M = ⊕_{c' ∈ W(o)} b(o, c')` is the node's own reduction,
`M< = ⊕_{c' ∈ W(o), c' < c} b(o, c')` the reduction over the earlier terms
(the semiring identity, `∓∞`, on an empty window), and
`E = Π_{c' ∈ W(o), c' ≠ c} b(o, c')` the exclusion product — never a division,
so a zero factor is exact. `M`, `M<` and `E` are scalar faq nodes with the
node's `ranges` and `filter` over the contracted names renamed apart: each
name `c` becomes `c_p` (`_p` appended again until clear of every name in the
node), and `M<` conjoins `c_p < c`, `E` conjoins `c_p != c` (`or` of one
`!=` per name for several). `M` and `M<` select the FIRST extremal term in
ascending contraction order — the §3.2 left-fold tie rule, which the
normative ascending accumulation order of esm-spec §4.3.1 makes the fold's
own; a reverse scan's tie still resolves to the lowest index (an a.e.
choice). A `max` or `min` over more than one contracted name has no single
ascending order and MUST be refused. A one-point window (no contracted name)
weighs `d` alone. The weighted terms then differentiate as a `+` contraction:

- the filter's **index-only** conjuncts — those reading only the node's
  output and contracted names, literals, folded metaparameters and `const`
  gathers — are decided exactly: the band box (rows and contracted ranges)
  is split at their affine breakpoints with every one-value name pinned
  first (the §2 region clipping), cells where they are false are dropped,
  and what no box can express (a triangular window `gk < gke`) becomes the
  entry's `where` guard (§4). A gathered site's entry takes the whole
  residual guard; a site whose column reads no contracted name keeps the
  residual's contracted conjuncts as the `filter` of its coefficient sum and
  takes the rest as its `where`;
- the filter's **value-dependent** conjuncts stay in the coefficient as
  `ifelse(φ, d, 0)`, and the pattern is the whole window (§2's branch union).

The same holds for a scalar faq node NESTED in a differentiated expression
(Σ_k g inside f(…)) — the form a read of a contracting observed takes. A
read `F[e…]` of an observed defined by a faq node whose ranges bind indices
beyond its `output_idx` MUST inline to a scalar faq node over those
contracted indices, carrying `reduce`, `semiring` and `filter`. Inlining the
bare body instead drops the sum, and a faq around the reading site that
binds the same name captures the leftover index. All inlining substitution
MUST be capture-avoiding, and no nested faq node may bind a name a faq
enclosing it already binds: producers rename such a binder apart (the
reference uses `_c<n>_<name>`, the smallest `n` clear of every name in the
document, which keeps repeated reads of one observed structurally equal). A
name a faq around the reading site binds is that index, never a read of an
observed with the same name. The derivative of the nested node is the `+`
of its weighted terms above, with its own filter kept: for a site whose
column does not read a bound index, the coefficient keeps the sum (a
`reduce: "+"` node, `semiring` dropped); for a site whose column does read
one (`u[i,k]` under Σ_k), that index is left free — the sum over the other
indices keeps the filter, and once every index is free the filter becomes
`ifelse(φ, ·, 0)` — and becomes a `contracted` dimension of the band (§4),
renamed apart from the sum's own bound name (`<name>_j`, `_j` appended until
clear), since the coefficient may still contain the sum (the chain rule
through f(Σ_k …)). The index-only conjuncts of the filters around the site
that read such a lifted index — and no index a sum around it still binds —
are also the entry's `where` guard, renamed alike, unless the site occurs
under different filters at different occurrences. A guard conjunct that
also reads an index the enclosing node contracts but the entry does not
carry as a `contracted` dimension (`Σ_k Σ_{j≤k}` with a column `u[j]`)
filters the entry's sum over that index instead, never the `where` guard,
which may read only the entry's own row and contracted names. Producers
MUST fail when a column reads a name the sums around its site bind
ambiguously (twice on the way down, over two different ranges at different
occurrences of the site, or shadowing the band's own row or contraction
index). Fixtures:
`contracting_observed.esm`, `filtered_scan.esm`, `semiring_reduce.esm`.

**Value-indexed reads.** A column index that reads anything but the entry's
own row and contracted index names (a level search `κ = min{k : …}` used as
`x[κ + d]`, a parameter offset) is integer-valued, hence piecewise constant
— the `floor` rule of §3.2 — so its own derivative is 0, and the read is the
one-hot sum `Σ_k [k == κ + d]·x[k]`. Each such column dimension `k` (1-based)
becomes a `contracted` dimension named `_oh<k>` (`_j` appended until clear of
every name in the entry) over the static extent of that axis of the column
variable, its index expression moves into the coefficient as
`ifelse(_oh<k> == e, coef, 0)`, and the pattern is the structural union: every
cell of the axis. A level search's own derivative needs no rule: its body is
an index name, and its filter's Dirac term is dropped.

Reserved (producers MUST fail rather than guess): joins, value-invention
`faq` nodes, the `bool_and_or` semiring, `argmin`/`argmax`, a `max` or `min`
over more than one contracted index, singleton `1` output indices,
`reshape`/`transpose`/`concat` operands. A causal self-reference recurrence
(esm-spec §4.3.1.1) is not yet differentiated: the inliner finds no
fixpoint and fails.

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
- `col` — a state (`wrt: "states"`), parameter (`wrt: "parameters"`) or
  forcing-buffer (`wrt: "forcing"`, §4.3) name, or `"t"` (`wrt: "time"`,
  §4.2).
- `rows` — a rectangular region of `row`'s index space, one inclusive
  `[lo, hi]` integer pair per dimension; `[]` for a scalar `row`.
- `row_idx` — per dimension: a free index NAME (string) ranging over the
  corresponding `rows` pair, or an integer for a singleton dimension. Names
  are local to the entry.
- `col_idx` — `col`'s index expressions, one per dimension of `col`, over the
  `row_idx` names; `[]` for a scalar `col`. Full rank: a forcing buffer's
  record axis is an ordinary dimension (§4.3).
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
- `where` (optional) — an exact point predicate: a boolean Expression over
  the entry's `row_idx` and `contracted` names, literals, folded
  metaparameters and `const` gathers, never a state, parameter or forcing
  value. A point contributes iff it holds. Produced by a filtered faq
  node's window that no box can express (§3.5: the triangular `gk < gke` of
  a level scan); producers SHOULD split boxes at affine breakpoints first,
  so a rectangular window never needs one. Absent for every other entry.

**Meaning.** For every index point `r` in `rows` AND every point `c` of the
`contracted` box (the empty product when absent) at which `where(r, c)`
holds (every point when it is absent),

```
J[ row[r], col[col_idx(r, c)] ] += coef(r, c)
```

Entries are additive: multiple entries — and multiple contracted points of
ONE entry — may target the same (row, col) pair.
On the `states` and `parameters` axes, a `col_idx(r)` that falls outside
`col`'s declared extent contributes nothing (it denotes a boundary/ghost read,
such as a face coefficient `K[i+1]` read beside `c[i+1]`, that is not a
column). On the `forcing` axis a column is a cell of a live buffer and there
is no ghost read: a column that does not resolve to a cell of `col` is an
error, and a consumer MUST refuse the block rather than skip the point.

`structure` (optional, derived): `"block_diagonal"` | `"banded"` |
`"general"` | `"column_block"` — a *classification* of the exact pattern for
consumers choosing a matrix type; the pattern itself is always recoverable
from the entries. `"column_block"` is reported only when the producer is
told a column axis (an index-set name, e.g. by an integrator's locality
pass: a name carries no vertical meaning by itself), in place of `"banded"`
or `"general"`, when every coupling stays within one LINE along it: every
entry couples two array states of one common shape that holds the axis
once, and at every point the column equals the row in every other
dimension (a point past the axis's extent is a ghost read). Its §6 plan
names the per-line pattern.
Omitted, with `factorization`, on the `parameters` and `forcing` axes: the
classes describe square matrices.

`columns` (required when `wrt` is `"parameters"` or `"forcing"`, absent
otherwise): the axis's column variables, `[{"name": …, "shape": [n₁, …]}]`
with integer extents (`[]` for a scalar), sorted by name — every variable of
the axis, whether or not an entry reads it. It fixes the **canonical column
ordinal**: the variables in this order, each one's cells column-major (first
index fastest), so a consumer can size and order the column space without
the source document. Each binding maps the ordinal onto its own memory.

**Several axes per model.** `jacobians.<ModelName>` MAY be an ARRAY of
blocks with pairwise-distinct `wrt` instead of one block, so a producer
writing several axes of one operator need not write several documents. A
single object stays legal, and a consumer MUST accept both forms. A block
whose `wrt` is an axis pair holds second-order entries (§4.4); it is distinct
from every single-axis block and from the pair's reverse.

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
  respect to forcing values are the forcing axis (§4.3). A blend weight
  that the model computes from `t` (a `frac` expression) is an explicit
  occurrence and IS covered.

**Autonomous systems.** An empty `"time"` entry list means `f_t ≡ 0`: no
row's right-hand side reads `t`. A consumer MAY skip evaluating `f_t` for
such a model and use the zero vector.

A `"jacobians"` block holds one `wrt`, so a model's time bands are a block
of their own: in a derived document per axis, or beside the state block in
the model's block array (§4).

### 4.3 Forcing and parameter columns (`wrt: "forcing"`, `wrt: "parameters"`)

A block with `"wrt": "forcing"` carries `∂f/∂B`, the derivative of the
right-hand side with respect to the cells of the **forcing buffers** `B` a
runtime binds and a data provider refills. A block with `"wrt": "parameters"`
carries `∂f/∂p` for every parameter outside the forcing set, scalar and
array-valued alike: an array parameter's cells (a gridded emission field,
say) are real columns, never dropped.

**The forcing set** is an input to the producer, a list of parameter names.
When none is given it is inferred: every array-valued parameter with an
`update` of kind `"data"`. A planner that builds the providers itself passes
the list explicitly (a document need not declare the update). A name that is
not a parameter of the system resolves to no buffer, and the producer MUST
fail. The two axes partition the parameters: a forcing-set member is a column
of the forcing axis only.

**Columns.** A column is `(buffer name, multi-index over the declared
shape)`, every dimension included. The canonical ordinal is §4's `columns`
order: buffers sorted by name, each column-major. Column-index expressions
MUST be closed at build time: literals, folded metaparameters, `const`
gathers, and the entry's row and contracted index names. A producer that
cannot close one (an offset read from a parameter, say) MUST fail rather than
emit an entry it cannot place. Inlining follows the state axis: only the
observed that can move with a target are inlined, seeded from the forcing set
(for `"parameters"`, from that axis's parameters), plus those a column index
reads.

**The record axis.** A provider that brackets the current time hands the
model its two bracketing records as an ordinary leading length-2 dimension of
the buffer (`F[1, …]` the floor record, `F[2, …]` its successor), and the
model writes the time blend itself. No construct is special: `index(F, 1, …)`
and `index(F, 2, …)` are distinct sites (§3), and the blend is model AST. For a
row reading `F̃ = (1 − w)·F[1, m] + w·F[2, m]` with `φ = ∂f/∂F̃` by the ordinary
rules,

```
∂f/∂F[1, m] = (1 − w)·φ        ∂f/∂F[2, m] = w·φ
```

from the `*` and `+` rules alone; the weight's own `floor` differentiates to
0 (§3.2). A non-blend read of the records, such as `(F[2, m] − F[1, m])/Δ`,
adds `±φ'/Δ` to the same two columns (entries are additive). A regrid between
the buffer and the blend is a `reduce: "+"` contraction with `const` weights,
which yields `contracted` bands over the native grid (§3.5). Producers MUST
NOT pattern-match the blend. At a record time the frac weight is exactly 0,
so the second record keeps its columns with a zero blend coefficient
(structural sparsity, §2). The weight's dependence on `t` stays on the time
axis (§4.2).

**Empty blocks.** A model that reads none of the forcing set has an empty
forcing block. So does one that reads it only where the derivative is
identically zero, such as an `ifelse` condition or the argument of a
piecewise-constant op (§3.2): structural occurrence alone is not a band.

**Meaning and use.** The block describes `∂f/∂B` at one evaluation point
`(u, B, t)`, with §4's meaning. A consumer applies its transpose by a
scatter-add over the band points, without forming a matrix: for a stage
cotangent `λ`,

```
ḡ_B[col(r, c)] += coef(u, B, t; r, c) · λ[row(r)]
```

into one accumulator per buffer, of the buffer's shape. Which dataset record
a buffer slot held at that stage, and the per-record accumulation across
provider refreshes, are runtime state that the consumer (an integrator's
tape) owns. They are not part of the block. A degenerate final bracket that
repeats one record in both slots maps both slots' gradients to that record,
where they sum. §9.3 gives the same scatter-add as a document an EarthSciAST
evaluator runs.

### 4.4 Second-order entries (`wrt: [a1, a2]`)

A Rosenbrock step contains `J = ∂f/∂u` at the step's start state, so its
discrete adjoint differentiates `J` itself, once per step:

```
H = ∇_{a2} Σᵢ μᵢᵀ J(u) wᵢ                (a1 = "states")
H = ∇_{a2} νᵀ f_t,   ν = Σᵢ wᵢ μᵢ         (a1 = "time"; each wᵢ a scalar)
```

with stage cotangents `μᵢ` and stage vectors `wᵢ`. A block whose `wrt` is a
pair `[a1, a2]`, `a1 ∈ {"states", "time"}` the first derivative's axis and
`a2 ∈ {"states", "parameters", "forcing"}` the second's, carries the
entries of that contraction. `["states", "states"]` serves the state
cotangent, `["states", "parameters"]` and `["states", "forcing"]` the mixed
terms of the parameter and forcing cotangents (§4.3), and `["time", ·]` the
derivative of `f_t` when the method passes it.

```json
"jacobians": { "Chem": [
  { "wrt": "states", "entries": [ … ] },
  { "wrt": ["states", "states"],
    "entries": [
      { "row": "O3", "col": "NO", "col2": "O3",
        "rows": [[1, 3]], "row_idx": ["i"],
        "col_idx": ["i"], "col2_idx": ["i"],
        "coef": {"op": "*", "args": [{"op": "index", "args": ["k1", "i"]}, -1]} } ],
    "structure": "block_diagonal" } ] }
```

**Production.** No new derivative rules. For each first-order entry `e` of
the `a1` axis, the producer differentiates `e.coef` with respect to every
site of every `a2` target, treating it as a scalar body broadcast over
`e.rows`: region clipping, re-merging of equal-coefficient entries and
singleton pinning as for any scalar body. The first-order pass inlines the
observed that move with either axis (§4.3's seeding, over the union of the
two target sets): an observed rate that reads only parameters is an opaque
read on the state axis and would hide its parameter sites from the second
derivative. The non-smooth policy (§3.2) applies again — `ifelse`
differentiates branch-wise with the condition's Dirac term dropped, ties of
`max`/`min` keep the earlier argument — and the pattern is the structural,
global union over branches (§2). A piecewise-linear limiter therefore has an
empty second-order block.

**Entries** carry §4's fields with the same meaning, plus:

- `col2` — the second axis's variable: a state, a parameter outside the
  forcing set, or a forcing buffer.
- `col2_idx` — its index expressions over the `row_idx` and `contracted`
  names, one per dimension of `col2`; `[]` for a scalar.

`col` and `col_idx` are inherited from `e` (`"t"` and `[]` on a time pair),
and so is `contracted`. A second site inside a scalar `reduce: "+"`
coefficient whose index the column reads adds contracted dimensions by the
rule of §3.5.

**Meaning.** For every row point `r` and contracted point `c`,

```
T[ row[r], col[col_idx(r, c)], col2[col2_idx(r, c)] ] += coef(r, c)     ( = ∂²f_row / ∂col ∂col2 )
```

Entries are additive. `(r, c, s)` and `(r, s, c)` come out independently,
from distinct first-order entries; consumers MUST NOT assume either is
deduplicated, and either orientation alone is not the tensor.
An out-of-extent `col_idx` or `col2_idx` follows its axis's rule (§4): a
ghost read on `states` and `parameters`, an error on `forcing`. On the
`parameters` and `forcing` second axes `col2_idx` MUST be closed at build
time (§4.3).

**Block fields.** `columns` is the second axis's column space when `a2` is
`"parameters"` or `"forcing"` (§4's form and canonical ordinal), absent
otherwise. `structure` is `"block_diagonal"` when `a1` is `"states"`, the
first-order state block is block-diagonal and every `col2_idx` equals the
row index (`col2` an array of the row's rank, or both scalar): the
contraction is then lane-local. Otherwise it is omitted. There is no
`factorization`. `expression_templates` (§4.1) apply unchanged.

**Contraction.** The tensor is never formed. A consumer evaluates the
coefficients at `(u, p, t)` and contracts:

```
M(r, c)          = Σᵢ μᵢ[row(r)] · wᵢ[col(r, c)]
g[col2(r, c)]   += coef(r, c) · M(r, c)
```

On a `["states", "states"]` pair the symmetry of `∇²(μᵀf)` gives an
equivalent form with no `M`, one pass per stage:
`g[col(r, c)] += μᵢ[row(r)] · coef(r, c) · wᵢ[col2(r, c)]`. Implementations
choose by cost; conformance checks both against the oracle. §9.4 gives both
forms as documents an EarthSciAST evaluator runs.

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
   without hoisting; like every derived document of §8 and §9 it declares
   the lowest `esm` version that admits what it holds: `"1.2.0"` when any
   expression node carries `units`, a unit-bearing `const` copied from the
   source, and `"1.1.0"` otherwise, for its `faq` nodes) and `<name>.jvals.json` (values, item 3). A fixture
   whose right-hand side reads `t` also carries the same four for
   `wrt: "time"` as `<name>.time.*.json` (`tgrad_photolysis.esm`: a
   photolysis rate `max(0, cos(ω t + φᵢ))` through an array observed,
   `interp.linear` of `t` through a scalar observed, a `datetime.hour`
   factor, a `datetime.julian_day` source, and one autonomous row). A fixture
   with forcing carries the same four for `wrt: "forcing"` and
   `wrt: "parameters"` as `<name>.forcing.*.json` and
   `<name>.parameters.*.json`, its forcing set inferred, plus a second value
   golden per axis, `<name>.<axis>.w0.jvals.json`, at a record time
   (`forcing_bracket.esm`: an array and a scalar state reading a two-record
   buffer through a frac-blend weight, a record difference and a gather at a
   constant offset; a regrid-like `reduce: "+"` contraction with `const`
   weights over one record of a second buffer; and a const array parameter
   standing in for an emission field. Its value goldens are at `w = 0.37` and
   at `w = 0` exactly, with the buffers at the document defaults). A
   second-order pair carries `<name>.<a1>-<a2>.bands.json` and `.block.json`
   (§4.4) and a value golden `<name>.<a1>-<a2>.hvals.json`: the contracted
   `g` at a pinned point and two pinned stage pairs `(μᵢ, wᵢ)`, all carried
   in the golden by name, with `g` keyed by every second column a band point
   reaches. The pairs: `["states", "states"]` on `bd_chem.esm`
   (block-diagonal bimolecular chemistry), `arrhenius_chem.esm` (per-cell
   NOx–O₃ chemistry whose rates are Arrhenius observed of a temperature
   parameter `T[z]`, opaque on the state pair) and `minmod_adv.esm` (a
   limiter kink: empty); `["states", "parameters"]` on `arrhenius_chem.esm`;
   `["states", "forcing"]` and `["time", "states"]` on
   `forcing_bracket.esm`; `["states", "states"]` on `filtered_scan.esm` and
   `semiring_reduce.esm` (below). The §3.5 window fixtures: `filtered_scan.esm`
   (inclusive forward and exclusive reverse scans, a filtered contracting
   observed read through a nested sum, a window with an index conjunct and
   a value conjunct, reseact's `Mz` level scan `gk < gke` over `(gi, gke)`
   rows — `where` guards — and a window clipped by a constant, which splits
   the box instead) and `semiring_reduce.esm` (a running `max`, a nested
   `min` observed, a suffix product, a `max_product` semiring, and a level
   search used as a subscript — one-hot).
3. **Value oracle** — per fixture model and sample point: the assembled
   Jacobian must match a machine-precision reference (AD through the
   binding's evaluator, or the shipped reference values) to `rtol 1e-12`,
   and the structural pattern must be a superset of the local pattern.
   Shipped value goldens (`*.jvals.json`) are compared with exactly the same
   (row, column) key set and `rtol 1e-10` (evaluation order differs between
   runtimes); `*.hvals.json` with the same tolerance and exactly the same
   second-column key set, in both contraction forms where both apply. The
   Julia reference checks the contraction against ForwardDiff over ForwardDiff
   through its evaluator. A column key is the variable name, with `[i,j,…]` (1-based) for
   a cell of an array variable.

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

A `"column_block"` structure (§4) carries a line-by-line plan:

```json
"factorization": {
  "type": "column_block", "axis": "lev", "line_length": 72,
  "block_vars": ["NO", "O3"], "pattern": "lower", "bandwidth": [71, 0]
}
```

- `axis` / `line_length` — the column axis and its extent: the matrix is one
  `(length(block_vars) · line_length)`-square block per line.
- `block_vars` — the state variables of the line block, sorted.
- `bandwidth` — `[kl, ku]`, the largest distance below / above the diagonal
  any entry point reaches along the axis (a variable-to-variable coupling
  at one level is distance 0).
- `pattern` — the name of that coupling: `"dense"` (`kl = ku =
  line_length − 1`), `"lower"` (`ku = 0`, `kl = line_length − 1`), `"upper"`
  (the mirror), else `"banded"` — so a solver can pick a column solver
  (a banded LU, a triangular sweep, a dense LU) without reading the
  entries.

## 7. Out of scope (this draft)

Third and higher derivatives, a Hessian assembly API, and the second-order
terms of a lagged-`J` W-method (which move to the step where `J` was
formed). (Time-derivative bands, contracted-index bands,
`expression_templates` coefficient sharing, forcing-buffer and
array-parameter columns, and second-order entries, once listed here, are
specified in §4.2, §3.5/§4, §4.1, §4.3 and §4.4.)

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

## 9. Product documents (PROPOSED)

**Status: PROPOSED. Produced by the Julia reference (`jvp_document`,
`vjp_document`, `hvp_document`), with goldens (§9.6); not yet by the other
bindings.** Any change to the text below lands with its goldens.

A consumer needs more than the coefficients: an implicit integrator applies
`J` and `Jᵀ` to vectors, and an adjoint contracts the second-order bands
(§4.4). Evaluation of every one of these is EarthSciAST's job, not the
producer's. For each product the producer emits a **product document**: a
derived ESM document whose right-hand side, evaluated once by an unmodified
EarthSciAST evaluator, yields the product. No binding of this repository
evaluates band points itself, and no product needs a kernel beyond what the
esm-spec's `faq` already expresses.

### 9.1 Common form

A product document is built like the §8 evaluation document: the source
model's variables, the observed definitions the coefficients read
(transitively), every source `D(u)` equation replaced by `D(u) = 0`, and the
coefficients' `expression_templates` expanded (or hoisted as observed, as the
evaluation document does). On top of that base:

- **Inputs.** Every vector the product contracts with is declared as an
  `unknown` with `D(x) = 0` and default `0.0`, of the shape of the space it
  lives in: a row variable's shape for a cotangent, a column variable's for
  a direction. The consumer writes it into the evaluator's state slots at
  each call, beside the source states. (A state, not a parameter, for the
  reason the evaluation document zeroes derivatives rather than freezing
  states: an evaluator may cache observed that read only parameters.)
- **Outputs.** Every result is an `unknown` with default `0.0` whose `D`
  equation is the product expression. One right-hand-side call at
  `(u, inputs, p, t)` leaves the product in the derivative slots of the
  outputs; the consumer reads them through the evaluator's variable map.
  No equation may read an output, and an output's own state value is never
  read.
- **Names.** Inputs and outputs take the names below, with `_` appended
  until clear of every name in the document and of every name generated
  before it. The names of a document are its variable and index-set names,
  every `ranges` key, and every string its equations hold other than an
  `op` value. Names are generated in one fixed order, so every binding
  reaches the same names: the table's input column, then its output column;
  within a column by stage number `i` ascending, then (within one stage)
  every `_mu<i>` before every `_w<i>`, then by variable in canonical order
  (states in the canonical state order of §8, the columns of a `parameters`
  or `forcing` axis in `columns` order, then `t`). `R` is a row variable, `C`
  a column variable, `C2` a second-axis column variable, and `i` a 1-based
  stage number.
- **Layout.** The `D` equations of the inputs, then of the outputs, follow
  the base's equations (and any hoisted observed), in name-generation order.
  A zero right-hand side is `0.0`, or for an array a `makearray` of `0.0`
  over the whole shape, as for the zeroed source derivatives. A `faq` below
  omits `reduce` (its default is `"+"`); its `ranges` list the output names,
  then the row names, then the contracted names. A `filter` with one
  conjunct is that conjunct, with several an `and` of them in the order
  written; with none the field is omitted. A `+` over terms is a nest of
  binary `+` folded from the left.

| Product | Model | Inputs | Outputs |
|---|---|---|---|
| `Jv` | `JvpEval` | `_v_<C>` per column variable | `_jvp_<R>` per state |
| `Jᵀλ` | `VjpEval` | `_lam_<R>` per state | `_vjp_<C>` per column variable |
| §4.4 contraction | `HvpEval` | `_mu<i>_<R>`, `_w<i>_<C>` per stage | `_hvp_<C2>` per second-axis column variable |

The column variables of an axis are every state (`wrt: "states"`), every
entry of the block's `columns` (`"parameters"`, `"forcing"`), or the single
scalar `t` (`"time"`, §4.2: input `_v_t`, output `_vjp_t`), whether or not
an entry reaches them: an output nothing reaches is `0`. The number of stages
`S` of an `HvpEval` document is an input to the producer.

**Notation.** For entry `k`: `R_k` its `row` with row box `rows_k`;
`ρ_k` its `row_idx` read as an index tuple (names, and the literal of each
singleton dimension), `ρ̂_k` the same tuple as a `faq` `output_idx` (each
literal replaced by the reshape singleton `1`, esm-spec §4.3.1), and
"`rows_k` over `ρ_k`" one `ranges` entry per NAME of `ρ_k` (a literal
dimension has none); `γ_k` its `contracted` names with their ranges, `C_k`
its `col` with index expressions `e_k(ρ, γ)`, `a_k(ρ, γ)` its `coef` and
`ω_k(ρ, γ)` its `where` (`true` when absent, and then no conjunct).
`n_{C,d}` is the extent of dimension `d` of `C`. Fresh output-index names `j = (j_1, …, j_n)` are
`_j1`, …, `_jn`, each with `_` appended until clear of every name in the
entry and the document. Each product below is a `+` over entries in entry
order (a left fold), so the association is fixed.

**Every reduced name occurs in `expr`.** esm-spec §4.3.1 reduces a `faq`
over the index names that occur in `expr` and not in `output_idx`; a name
that occurs only in `ranges` or `filter` is not iterated. Where a body below
does not read a non-output range name `ν` (a contracted name that appears
only in an `e_k` the filter carries, as in §9.3), the producer MUST write
`expr` as `ifelse(Φ, body, 0.0)`, `Φ` the conjunction of the filter's
conjuncts that read such names, in filter order. `Φ` holds at every point
the filter admits, so the value is unchanged; the guard exists only to
make `ν` occur. A producer that cannot make every name occur this way MUST
fail.

**Ghost reads.** On the `states` and `parameters` axes a point whose column
falls outside `C`'s extent contributes nothing (§4). Where a product reads
`C` at `e_k`, the producer adds the in-extent conjunct
`G_k = and(1 <= e_{k,d}, e_{k,d} <= n_{C,d} for every d)` to the filter
(its conjuncts in that order, after `ω_k`); it SHOULD instead split the box
at `G_k`'s affine breakpoints (the §2 region clipping) so that no filter
remains for an affine `e_k`. Where every point of the entry evaluates in
extent at build time the split leaves the box whole, and no `G_k` is
written; the goldens' producer does not split otherwise. On the
`forcing` axis there is no ghost read: the producer MUST check at build time
that every point of every entry resolves to a cell of its buffer, and MUST
fail rather than emit a document that would drop one.

### 9.2 `Jv` (`JvpEval`)

For each state `R`,

```
D(_jvp_R) = Σ_{k : R_k = R} makearray( 0 over shape(R),
               rows_k ← faq{ output_idx: ρ̂_k,
                             ranges:     rows_k over ρ_k, the ranges of γ_k,
                             filter:     and(ω_k, G_k),
                             expr:       a_k · index(_v_C_k, e_k) } )
```

with `reduce: "+"`. The contracted names are reduced; the row names are
the output. A scalar `R` with no `contracted` drops the `makearray` and the
`faq`: the term is `a_k · index(_v_C_k, e_k)` (a scalar `_v_C_k` read bare).
Such an entry has no index names, so `e_k`, `G_k` and `ω_k` read only
literals, folded metaparameters and `const` gathers, and the producer
decides them at build time: it drops the term outright when `ω_k` is false
or the column is a ghost (§9.1), and otherwise emits it unguarded. An array
`R` whose `row_idx` is all literals and that has no `contracted` likewise
keeps the `makearray` but writes the bare term as its region's value, with
no `faq`. This is
a gather: every output cell reads its own row point, and no optimization
beyond what a `faq` already gets is needed.

### 9.3 `Jᵀλ` (`VjpEval`)

For each column variable `C` of the axis, with fresh output-index names
`j = (j_1, …, j_n)` (`n` the rank of `C`, §9.1),

```
D(_vjp_C) = Σ_{k : C_k = C} faq{ output_idx: j,
                                 ranges:     j_d ∈ [1, n_{C,d}], rows_k over ρ_k,
                                             the ranges of γ_k,
                                 filter:     and(ω_k, e_{k,1} == j_1, …, e_{k,n} == j_n),
                                 expr:       a_k · index(_lam_R_k, ρ_k) }
```

with `reduce: "+"`. This is §4's meaning transposed: output cell `j`
receives `a_k · λ[row]` from every point whose column is `j`. The output
range is exactly `C`'s extent, so ghost points never satisfy the equality
and need no `G_k`. A contracted name that `e_k` reads but `a_k` does not
occurs in this body only through the filter, so `expr` takes §9.1's
`ifelse` guard. A scalar `C` has no output index and no equality: the
term is the filtered sum over the entry's points (on `wrt: "time"`,
`_vjp_t = Σ_k Σ_rows a_k · λ[row]`). On `wrt: "forcing"` this is §4.3's
`ḡ_B` scatter-add; on `wrt: "parameters"` it is the parameter cotangent.

None of the conjuncts compares a contracted name with the node's own output
index by `<`, `<=`, `>` or `>=`, so the node is not a scan (esm-spec
§4.3.1): it folds ascending over its contracted names, and its value is
fully determined by the esm-spec.

**Evaluation cost (required of EarthSciAST).** Evaluated as written, the node
costs `|output box| × |row box × contracted box|`: quadratic in the grid for a
state vjp. An evaluator used for production MUST recognize the
**equality-filter scatter**: a `faq` whose `filter` is a conjunction that
contains, for an output index `j_d`, a conjunct `j_d == e` or `e == j_d` with
`e` reading none of the node's output indices (the first such conjunct per
`j_d`, in filter order; any later one is an ordinary conjunct). Output
indices with no equality conjunct iterate outside the pass as ordinary
output indices. The evaluator evaluates such a node in ONE pass over the
box of the non-output names, in the same ascending order the naive
evaluation visits them. At each point it computes every bound `e` (none
reads an output index), skips the point if one is not an integer in its
`j_d`'s range, binds each `j_d := e`, and only then evaluates the remaining
conjuncts and, where they hold, the body, folding it into output cell `j`.
The remaining conjuncts and the body may read the output indices: with
`j_d` bound to `e` they see exactly the values the naive evaluation gives
them at that point of cell `j`, where every other `j_d` fails its equality.
Each output cell then receives exactly the terms it receives naively, in
the same order, so the result is bit-identical (the license has the form of
esm-spec §4.3.1's running-accumulation license); the cost is
`|row box × contracted box|`. When every bound `e` is closed at build time
(it reads only index names, literals, folded metaparameters and `const`
gathers: always for `col_idx` on the `parameters` and `forcing` axes, §4.3,
and for `col2_idx` on those second axes, §4.4), an evaluator MAY compute
the scatter map once at build time. A states-axis `col_idx` that gathers
through a table that is not `const` is not closed, and the map is then
recomputed at each call.

This optimization is an EarthSciAST change in each language, tracked
there; until it lands, `VjpEval` documents are correct but quadratic. (At
the time of writing EarthSciAST.jl's native compiler refuses such a node
outright, `compiler_refused_rule`, and its interpreter evaluates it.)

### 9.4 Second-order contraction (`HvpEval`)

For a block with `wrt: [a1, a2]` and `S` stages, write
`M_k(ρ, γ) = Σ_{i=1..S} index(_mu<i>_R_k, ρ_k) · index(_w<i>_C_k, e_k)`
(each stage's product a binary `*`, the stages a `+` fold; on a time pair,
`a1 = "time"`, `C_k` is `t` and each `_w<i>` is a scalar read bare). For each column variable `C2` of `a2`, with fresh output names
`j` as in §9.3,

```
D(_hvp_C2) = Σ_{k : col2_k = C2} faq{ output_idx: j,
                                      ranges:     j over C2's extent, rows_k, γ_k,
                                      filter:     and(ω_k, G_k, col2_idx_k == j),
                                      expr:       a_k · M_k }
```

with `G_k` the first column's in-extent conjunct (§9.1) and `col2_idx_k == j`
the per-dimension equalities. This is §4.4's `contract` form: the same
equality-filter scatter as `Jᵀλ`. On a `["states", "states"]` pair the
producer MAY instead emit the `symmetric` form, which sums over the FIRST
column and scatters by it: for each state `C`,

```
D(_hvp_C) = Σ_{k : C_k = C} faq{ output_idx: j,
                                 ranges:     j over C's extent, rows_k, γ_k,
                                 filter:     and(ω_k, G2_k, e_k == j),
                                 expr:       Σ_i index(_mu<i>_R_k, ρ_k) · a_k
                                                 · index(_w<i>_col2_k, col2_idx_k) }
```

with `G2_k` `col2`'s in-extent conjunct (§9.1's `G_k` for `col2_k` at
`col2_idx_k`), and each stage's product one three-argument `*` in that
order. Both forms give the same `g` (§4.4), but their outputs
gather different entries, so neither may be emitted with the other's
summation. The document records which form it holds in its model name
(`HvpEval` or `HvpEvalSymmetric`). Both forms take §9.1's `ifelse` guard
where their body does not read a reduced name. An entry with no index
names whose gathered column (`C_k` in the contract form, `col2_k` in the
symmetric one) is a ghost is dropped at build time, as in §9.2.

### 9.5 Matrix storage and scatter plans

Assembling `J` into a consumer's storage stays §8's split: the evaluation
document gives the band fields, and the scatter plan, a conformance
artifact, says where each field cell goes. Executing a plan is the
consumer's (an integrator's) work, not this repository's.

### 9.6 Conformance

File names follow §5's per-axis naming: the states axis has no infix, every
other axis is named, `<name>.<axis>.*`. Each axis whose value golden exists
(`<name>.jvals.json`, `<name>.<axis>.jvals.json`) carries
`<name>[.<axis>].jvp.evaldoc.json` and `<name>[.<axis>].vjp.evaldoc.json`
(`tgrad_photolysis.time.jvp.evaldoc.json`, `forcing_bracket.forcing.vjp.evaldoc.json`).
A `.w0` value golden is a second point of the same axis and adds no
document. Each second-order pair carries
`<name>.<a1>-<a2>.hvp.evaldoc.json` (`HvpEval`) and, on a
`["states", "states"]` pair, also `<name>.states-states.hvp-sym.evaldoc.json`
(`HvpEvalSymmetric`), so both contraction forms are checked (§5 item 3).
Their stage count `S` is the number of stage pairs in the pair's
`*.hvals.json` (two). All are byte-compared across bindings. Their
values are checked by evaluating the document with EarthSciAST at each
pinned point of the axis's value goldens and comparing with the product
computed from the golden itself (`J v`, `Jᵀ λ` from the `*.jvals.json` at
seeded `v`, `λ`; the `*.hvals.json` `g` at its pinned stage pairs), under
§5 item 3's tolerances. No binding needs an evaluator of its own for this
check.
