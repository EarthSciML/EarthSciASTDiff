//! Product documents (the reference's products.jl; spec §9): derived ESM
//! documents whose right-hand side, evaluated once by an unmodified
//! EarthSciAST evaluator, yields `J v`, `Jᵀ λ` or the §4.4 second-order
//! contraction. Like the evaluation document ([`crate::evaluation_document`])
//! they are ASTs only: this crate evaluates nothing at run time. The one
//! numeric step is §9.1's build-time decision of which points read a column
//! inside its extent, over closed column indices. Byte-gated by the
//! `*.{jvp,vjp,hvp,hvp-sym}.evaldoc.json` goldens.

use std::collections::{HashMap, HashSet};

use earthsci_ast::{free_variables, Expr};
use indexmap::IndexMap;
use serde_json::{json, Map, Value};

use crate::bands::{Band, BandError, Ctx};
use crate::clip::{index_bounds, index_box, RIdx};
use crate::emit::{base_document, derived_version, hoist, zero_source_derivatives, zeros};
use crate::helpers::ser_expr;
use crate::jacobian::{extended_pass, jacobian_columns, targets, JacEntry, Wrt};
use crate::points::{all_points, eval_guard};
use crate::recurrence::{
    adjoint_sweep, forward_sweep, recurrences, reversed_read, ExtendedBands, Recurrence,
};
use crate::second_order::{axis_pair, JacEntry2};
use crate::system::{ctx_of, observed_defs, view_states, SysView};

type R<T> = Result<T, BandError>;

/// One input or output map of a product document: each variable (a state,
/// a column variable, or `"t"`) to the document variable that carries it.
pub type IoMap = IndexMap<String, String>;

/// The options of a product document.
#[derive(Clone, Debug)]
pub struct ProductOptions<'a> {
    /// The forcing set (`None` infers it).
    pub forcing: Option<&'a [String]>,
    /// Hoist repeated coefficient subtrees into observed
    /// ([`crate::hoist_observed`]).
    pub cse: bool,
    /// The smallest subtree (in nodes) hoisted.
    pub cse_min_nodes: usize,
}

impl Default for ProductOptions<'_> {
    fn default() -> Self {
        ProductOptions {
            forcing: None,
            cse: true,
            cse_min_nodes: 12,
        }
    }
}

/// The inputs and outputs of a `JvpEval` document: `v` maps each column
/// variable to its direction input, `jvp` each state to its output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JvpIo {
    pub model: String,
    pub v: IoMap,
    pub jvp: IoMap,
}

/// The inputs and outputs of a `VjpEval` document: `lam` maps each state to
/// its cotangent input, `vjp` each column variable to its output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VjpIo {
    pub model: String,
    pub lam: IoMap,
    pub vjp: IoMap,
}

/// The inputs and outputs of an `HvpEval` / `HvpEvalSymmetric` document:
/// `mu[i]` / `w[i]` map each state (each first-axis column variable; `"t"`
/// on a time pair) to stage `i`'s inputs, `hvp` each second-axis column
/// variable to its output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HvpIo {
    pub model: String,
    pub mu: Vec<IoMap>,
    pub w: Vec<IoMap>,
    pub hvp: IoMap,
}

/// The second-order contraction's form (spec §9.4): `Contract` scatters by
/// the second column, `Symmetric` (a `(states, states)` pair only) by the
/// first. Both give the same `g`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HvpForm {
    Contract,
    Symmetric,
}

fn sop(o: &str, args: Vec<Value>) -> Value {
    json!({"op": o, "args": args})
}

/// `index(v, idx…)`, or the bare name for a scalar read.
fn sread(v: &str, idx: Vec<Value>) -> Value {
    if idx.is_empty() {
        return json!(v);
    }
    let mut args = vec![json!(v)];
    args.extend(idx);
    sop("index", args)
}

/// An `op` over terms as a left fold, so the association is fixed (§9.1).
fn sfold(o: &str, xs: Vec<Value>) -> Value {
    let mut it = xs.into_iter();
    let first = it.next().expect("a fold over at least one term");
    it.fold(first, |a, b| sop(o, vec![a, b]))
}

fn sand(mut cs: Vec<Value>) -> Value {
    if cs.len() == 1 {
        return cs.pop().unwrap();
    }
    sop("and", cs)
}

fn ser_all(es: &[Expr]) -> Vec<Value> {
    es.iter().map(ser_expr).collect()
}

/// The first of `base`, `base_`, `base__`, … not in `taken`, then taken.
fn fresh(taken: &mut HashSet<String>, base: &str) -> String {
    let mut nm = base.to_string();
    while taken.contains(&nm) {
        nm.push('_');
    }
    taken.insert(nm.clone());
    nm
}

/// [`fresh`] clear of `local` too, which is left untouched: the document's
/// names stay in `taken` and an entry's own in `local`, so a per-entry
/// freshening never copies the document-wide set.
fn fresh2(taken: &HashSet<String>, local: &mut HashSet<String>, base: &str) -> String {
    let mut nm = base.to_string();
    while taken.contains(&nm) || local.contains(&nm) {
        nm.push('_');
    }
    local.insert(nm.clone());
    nm
}

/// Every name of a derived document (§9.1): the variable and index-set
/// names, every `ranges` key, and every string its equations hold other
/// than an `op`.
fn doc_names(doc: &Map<String, Value>, model: &Map<String, Value>) -> HashSet<String> {
    fn collect(s: &mut HashSet<String>, x: &Value) {
        match x {
            Value::String(t) => {
                s.insert(t.clone());
            }
            Value::Array(a) => a.iter().for_each(|v| collect(s, v)),
            Value::Object(m) => {
                for (k, v) in m {
                    if k == "op" {
                        continue;
                    }
                    if k == "ranges" {
                        if let Value::Object(r) = v {
                            s.extend(r.keys().cloned());
                        }
                    }
                    collect(s, v);
                }
            }
            _ => {}
        }
    }
    let mut taken: HashSet<String> = HashSet::new();
    if let Some(Value::Object(isets)) = doc.get("index_sets") {
        taken.extend(isets.keys().cloned());
    }
    if let Some(Value::Object(vars)) = model.get("variables") {
        taken.extend(vars.keys().cloned());
    }
    if let Some(eqs) = model.get("equations") {
        collect(&mut taken, eqs);
    }
    taken
}

/// The column variables of one first-order axis, in canonical order (§9.1).
fn column_vars(sv: &SysView, wrt: Wrt, forcing: Option<&[String]>) -> R<Vec<String>> {
    Ok(match wrt {
        Wrt::Time => vec!["t".into()],
        Wrt::States => view_states(sv, &observed_defs(sv)?),
        _ => jacobian_columns(sv, wrt, forcing)?
            .into_iter()
            .map(|(n, _)| n)
            .collect(),
    })
}

/// The document variable carrying `v` in `names`: an entry whose row or
/// column variable the axis does not hold (an observed the band calculus
/// took for a state, say) is refused.
fn io_name<'m>(names: &'m IoMap, v: &str) -> R<&'m str> {
    names.get(v).map(String::as_str).ok_or_else(|| {
        BandError(format!(
            "`{v}` names no state or column variable of the product document's axis"
        ))
    })
}

/// Refuse entries whose column variable `col` is not on the document's axis
/// (`cols`, from `wrt` and `opts.forcing`): the caller built the entries
/// under another forcing set. Checked before any term is built, so the
/// mismatch is named rather than met late in [`io_name`].
fn check_columns<'e>(cols: &IoMap, used: impl IntoIterator<Item = &'e str>, wrt: Wrt) -> R<()> {
    for c in used {
        if !cols.contains_key(c) {
            return Err(BandError(format!(
                "an entry reads `{c}`, which is no column variable of the {} axis the product \
                 document was asked for; build the entries and the document with the same \
                 forcing set (`ProductOptions::forcing`)",
                wrt.as_str()
            )));
        }
    }
    Ok(())
}

fn shape_of(ctx: &Ctx, v: &str) -> Vec<i64> {
    ctx.shapes.get(v).cloned().unwrap_or_default()
}

/// The derived document under construction: the document's top level and
/// its one model.
struct Doc<'s, 'a> {
    sv: &'s SysView<'a>,
    ctx: Ctx,
    doc: Map<String, Value>,
    model: Map<String, Value>,
}

impl Doc<'_, '_> {
    /// Declare an input or output of the product document: an unknown shaped
    /// like `like` (`"t"`: a scalar) with default 0 and the equation
    /// `D(name) = rhs`, `None` meaning zero.
    fn declare(&mut self, name: &str, like: &str, rhs: Option<Value>) {
        let shape = shape_of(&self.ctx, like);
        let mut d = Map::new();
        d.insert("type".into(), json!("unknown"));
        d.insert("default".into(), json!(0.0));
        if !shape.is_empty() {
            d.insert(
                "shape".into(),
                json!(self.sv.variables[like].shape.as_deref().unwrap_or(&[])),
            );
        }
        if let Some(Value::Object(vars)) = self.model.get_mut("variables") {
            vars.insert(name.into(), Value::Object(d));
        }
        let rhs = rhs.unwrap_or_else(|| zeros(&shape));
        if let Some(Value::Array(eqs)) = self.model.get_mut("equations") {
            eqs.push(json!({"lhs": {"op": "D", "args": [name], "wrt": "t"}, "rhs": rhs}));
        }
    }

    /// Declare an intermediate of the product document (§9.7): an observed
    /// unknown shaped like recurrence `like`, defined by `name ~ rhs`.
    fn observe(&mut self, name: &str, like: &str, rhs: Value) {
        let shape = self.sv.variables[like].shape.clone().unwrap_or_default();
        if let Some(Value::Object(vars)) = self.model.get_mut("variables") {
            vars.insert(name.into(), json!({"type": "unknown", "shape": shape}));
        }
        if let Some(Value::Array(eqs)) = self.model.get_mut("equations") {
            eqs.push(json!({"lhs": name, "rhs": rhs}));
        }
    }

    /// Each output's equation: the left fold of its entries' terms (§9.1).
    /// `owner[k]` is the output variable entry `k` lands in.
    fn outputs(
        &mut self,
        outs: &IoMap,
        owner: &[&str],
        mut term: impl FnMut(usize, &Ctx) -> R<Option<Value>>,
    ) -> R<()> {
        // each output's entries, in entry order: one pass, not one per output
        let mut by_owner: IndexMap<&str, Vec<usize>> = IndexMap::new();
        for (k, o) in owner.iter().enumerate() {
            by_owner.entry(o).or_default().push(k);
        }
        for (c, nm) in outs {
            let mut terms = Vec::new();
            for &k in by_owner.get(c.as_str()).map_or(&[][..], Vec::as_slice) {
                if let Some(t) = term(k, &self.ctx)? {
                    terms.push(t);
                }
            }
            let rhs = (!terms.is_empty()).then(|| sfold("+", terms));
            self.declare(nm, c, rhs);
        }
        Ok(())
    }

    fn finish(mut self, name: &str) -> Value {
        self.doc
            .insert("models".into(), json!({name: Value::Object(self.model)}));
        derived_version(&mut self.doc);
        Value::Object(self.doc)
    }
}

/// Base of every product document (§9.1): the evaluation document's base,
/// with the observed kept that the coefficients, column indices and `where`
/// guards read, and the coefficients optionally hoisted. Returns the
/// document and the coefficients.
fn product_base<'s, 'a>(
    sv: &'s SysView<'a>,
    name: &str,
    bands: &[&Band],
    opts: &ProductOptions,
) -> R<(Doc<'s, 'a>, Vec<Expr>)> {
    let (mut doc, mut model) = base_document(sv, name);
    let ctx = ctx_of(sv)?;
    let mut roots: Vec<&Expr> = Vec::new();
    for b in bands {
        roots.push(&b.coef);
        roots.extend(&b.cidx);
        roots.extend(&b.guard);
    }
    zero_source_derivatives(&mut model, sv, &ctx, &roots)?;
    let coefs = hoist(
        &mut doc,
        &mut model,
        sv,
        bands,
        opts.cse,
        opts.cse_min_nodes,
    );
    Ok((
        Doc {
            sv,
            ctx,
            doc,
            model,
        },
        coefs,
    ))
}

/// The point walk of `b` reading `ci`, under `guard`: the band's index
/// structure only, so a large coefficient is never copied.
fn walk_band(b: &Band, ci: &[Expr], guard: Option<Expr>) -> Band {
    Band {
        rows: b.rows.clone(),
        ridx: b.ridx.clone(),
        cidx: ci.to_vec(),
        coef: Expr::Integer(0),
        contracted: b.contracted.clone(),
        guard,
    }
}

/// Whether `e` reads only the band's index names.
fn index_only(e: &Expr, bx: &HashMap<String, (i64, i64)>) -> bool {
    free_variables(e).iter().all(|n| bx.contains_key(n))
}

/// Whether every point of `b` reads `ci` inside `shape`; `None` when that has
/// no build-time answer: an index not closed (a gather through a table that
/// is not `const`), or a `where` guard not closed that a point outside the
/// extent depends on. Decided from the indices' bounds where they settle it
/// (§9.1's affine breakpoints), by walking the band's points otherwise. An
/// evaluation failure of that walk is an error, never taken for "not
/// closed".
fn in_extent(b: &Band, ci: &[Expr], shape: &[i64]) -> R<Option<bool>> {
    if ci.is_empty() {
        return Ok(Some(true));
    }
    if ci.len() > shape.len() {
        return Ok(None);
    }
    let bx = index_box(b);
    if bx.values().any(|(lo, hi)| lo > hi) {
        return Ok(Some(true)); // no points
    }
    if !ci.iter().all(|e| index_only(e, &bx)) {
        return Ok(None);
    }
    let bounds: Option<Vec<_>> = ci.iter().map(|e| index_bounds(e, &bx)).collect();
    if let Some(bounds) = bounds {
        if bounds
            .iter()
            .zip(shape)
            .all(|((lo, hi, _, _), n)| 1 <= *lo && hi <= n)
        {
            return Ok(Some(true));
        }
        // a corner outside the extent is a point of the band unless a guard
        // may drop it
        if b.guard.is_none() && bounds.iter().all(|x| x.2) {
            return Ok(Some(false));
        }
    }
    let mut inside = |_: &[i64], _: &[i64], _: &HashMap<String, i64>, c: &[i64]| {
        Ok(c.iter().zip(shape).all(|(x, n)| 1 <= *x && x <= n))
    };
    if let Some(g) = &b.guard {
        if !index_only(g, &bx) {
            // a guard not closed cannot drop a point: only an all-inside
            // box decides
            let all = all_points(&walk_band(b, ci, None), &mut inside)?;
            return Ok(all.then_some(true));
        }
    }
    Ok(Some(all_points(
        &walk_band(b, ci, b.guard.clone()),
        &mut inside,
    )?))
}

/// Refuse a read of `v` with more indices than its shape has dimensions.
fn check_rank(v: &str, ci: &[Expr], shape: &[i64]) -> R<()> {
    if ci.len() > shape.len() {
        return Err(BandError(format!(
            "a band reads `{v}` with {} indices, but it has {} dimensions",
            ci.len(),
            shape.len()
        )));
    }
    Ok(())
}

/// The forcing axis has no ghost read (§9.1): a point outside its buffer is
/// refused rather than dropped.
fn forcing_ghost(v: &str, ok: Option<bool>) -> BandError {
    let why = if ok.is_none() {
        " or where that is not decidable at build time (an index or `where` guard not closed)"
    } else {
        ""
    };
    BandError(format!(
        "a point of a band reads forcing buffer `{v}` outside its extent{why}; a product \
         document would drop it (esm-jacobian-spec.md §9.1)"
    ))
}

/// An extra conjunct of a filter, with the index names it reads.
type Conj = (Value, HashSet<String>);

/// §9.1's ghost rule for a read of column variable `v` at `ci` on axis
/// `wrt`: whether a point may read outside `v`'s extent, so the read must
/// be filtered by the in-extent conjuncts `1 <= ci_d <= shape_d`. A ghost
/// read on the forcing axis is refused.
fn needs_ghost(b: &Band, v: &str, ci: &[Expr], shape: &[i64], wrt: Wrt) -> R<bool> {
    check_rank(v, ci, shape)?;
    let ok = in_extent(b, ci, shape)?;
    if ok == Some(true) {
        return Ok(false);
    }
    if wrt == Wrt::Forcing {
        return Err(forcing_ghost(v, ok));
    }
    Ok(true)
}

/// The in-extent conjuncts of [`needs_ghost`], none when every point is in
/// extent (the region split then leaves the box whole).
fn ghost_conjuncts(b: &Band, v: &str, ci: &[Expr], shape: &[i64], wrt: Wrt) -> R<Vec<Conj>> {
    if !needs_ghost(b, v, ci, shape, wrt)? {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for (e, n) in ci.iter().zip(shape) {
        let ns = free_variables(e);
        out.push((sop("<=", vec![json!(1), ser_expr(e)]), ns.clone()));
        out.push((sop("<=", vec![ser_expr(e), json!(n)]), ns));
    }
    Ok(out)
}

/// A band with no index names has one point (none when its closed `where`
/// fails), and §9.2 has its column indices closed: whether that point exists
/// and reads `v` inside its extent. A ghost is dropped, except on the
/// forcing axis, where it is refused. Decided from the literal indices.
fn point_in_extent(b: &Band, v: &str, ci: &[Expr], shape: &[i64], wrt: Wrt) -> R<bool> {
    check_rank(v, ci, shape)?;
    if !ci.iter().all(|e| free_variables(e).is_empty()) {
        return Err(BandError(format!(
            "a band with no index names reads `{v}` at an index not closed at build time \
             (esm-jacobian-spec.md §9.2)"
        )));
    }
    if !closed_point(b)? {
        return Ok(false);
    }
    let ok = in_extent(&walk_band(b, ci, None), ci, shape)? == Some(true);
    if !ok && wrt == Wrt::Forcing {
        return Err(forcing_ghost(v, Some(false)));
    }
    Ok(ok)
}

fn row_names(b: &Band) -> HashSet<String> {
    b.ridx
        .iter()
        .filter_map(|r| match r {
            RIdx::Name(s) => Some(s.clone()),
            RIdx::Fixed(_) => None,
        })
        .collect()
}

fn has_names(b: &Band) -> bool {
    b.ridx.iter().any(|r| matches!(r, RIdx::Name(_))) || !b.contracted.is_empty()
}

fn where_conj(b: &Band) -> Vec<Conj> {
    b.guard
        .iter()
        .map(|g| (ser_expr(g), free_variables(g)))
        .collect()
}

fn names_of<'e>(es: impl IntoIterator<Item = &'e Expr>) -> HashSet<String> {
    es.into_iter().flat_map(free_variables).collect()
}

/// Whether a band with no index names has its one point: its `where` guard,
/// which §9.2 has closed, holds.
fn closed_point(b: &Band) -> R<bool> {
    let Some(g) = &b.guard else {
        return Ok(true);
    };
    if !free_variables(g).is_empty() {
        return Err(BandError(
            "a band with no index names has a `where` guard not closed at build time \
             (esm-jacobian-spec.md §9.2)"
                .into(),
        ));
    }
    eval_guard(g, &HashMap::new())
}

/// The row index of `b` as a read: its names and fixed indices.
fn rho(b: &Band) -> Vec<Value> {
    b.ridx
        .iter()
        .map(|r| match r {
            RIdx::Name(s) => json!(s),
            RIdx::Fixed(n) => json!(n),
        })
        .collect()
}

/// Fresh output-index names `_j1 … _jn`, clear of the entry's and the
/// document's names (§9.1).
fn jnames(taken: &HashSet<String>, b: &Band, coef: &Expr, n: usize) -> Vec<String> {
    let mut local = row_names(b);
    local.extend(b.contracted.iter().map(|(n, _, _)| n.clone()));
    local.extend(names_of(std::iter::once(coef).chain(&b.cidx)));
    if let Some(g) = &b.guard {
        local.extend(free_variables(g));
    }
    (1..=n)
        .map(|d| fresh2(taken, &mut local, &format!("_j{d}")))
        .collect()
}

/// The `faq` of one entry's term: `ranges` (the output names first, then the
/// row names, then the contracted names), the `filter` conjunction, and
/// `body` under §9.1's `ifelse` guard when a reduced name would not occur in
/// it.
fn term_faq(
    b: &Band,
    output_idx: Vec<Value>,
    outranges: &[(String, i64, i64)],
    conj: Vec<Conj>,
    body: Value,
    bodynames: &HashSet<String>,
) -> R<Value> {
    let mut ranges: IndexMap<String, (i64, i64)> = IndexMap::new();
    for (n, lo, hi) in outranges {
        ranges.insert(n.clone(), (*lo, *hi));
    }
    for (k, r) in b.ridx.iter().enumerate() {
        if let RIdx::Name(r) = r {
            ranges.insert(r.clone(), b.rows[k]);
        }
    }
    for (n, lo, hi) in &b.contracted {
        ranges.insert(n.clone(), (*lo, *hi));
    }
    let outnames: HashSet<&str> = output_idx.iter().filter_map(Value::as_str).collect();
    let miss: Vec<&String> = ranges
        .keys()
        .filter(|n| !outnames.contains(n.as_str()) && !bodynames.contains(*n))
        .collect();
    let mut expr = body;
    if !miss.is_empty() {
        if let Some(n) = miss
            .iter()
            .find(|n| !conj.iter().any(|(_, ns)| ns.contains(**n)))
        {
            return Err(BandError(format!(
                "the reduced index name `{n}` occurs neither in the product's body nor in its \
                 filter (esm-jacobian-spec.md §9.1)"
            )));
        }
        let phi: Vec<Value> = conj
            .iter()
            .filter(|(_, ns)| miss.iter().any(|n| ns.contains(*n)))
            .map(|(c, _)| c.clone())
            .collect();
        expr = sop("ifelse", vec![sand(phi), expr, json!(0.0)]);
    }
    let mut f = Map::new();
    f.insert("op".into(), json!("faq"));
    f.insert("args".into(), json!([]));
    f.insert("output_idx".into(), Value::Array(output_idx));
    f.insert(
        "ranges".into(),
        Value::Object(
            ranges
                .into_iter()
                .map(|(n, (lo, hi))| (n, json!([lo, hi])))
                .collect(),
        ),
    );
    if !conj.is_empty() {
        f.insert(
            "filter".into(),
            sand(conj.into_iter().map(|(c, _)| c).collect()),
        );
    }
    f.insert("expr".into(), expr);
    Ok(Value::Object(f))
}

// ── Jv (§9.2) ───────────────────────────────────────────────────────────────

fn jvp_term(en: &JacEntry, coef: &Expr, vin: &str, ctx: &Ctx, wrt: Wrt) -> R<Option<Value>> {
    let b = &en.band;
    let shape_v = shape_of(ctx, &en.v);
    let body = sop("*", vec![ser_expr(coef), sread(vin, ser_all(&b.cidx))]);
    let shape_u = shape_of(ctx, &en.u);
    let val = if has_names(b) {
        let mut conj = where_conj(b);
        conj.extend(ghost_conjuncts(b, &en.v, &b.cidx, &shape_v, wrt)?);
        let mut bodynames = names_of(std::iter::once(coef).chain(&b.cidx));
        bodynames.extend(row_names(b));
        let out: Vec<Value> = if shape_u.is_empty() {
            Vec::new()
        } else {
            b.ridx
                .iter()
                .map(|r| match r {
                    RIdx::Name(s) => json!(s),
                    RIdx::Fixed(_) => json!(1),
                })
                .collect()
        };
        term_faq(b, out, &[], conj, body, &bodynames)?
    } else {
        // no index names: `where` and the column are closed, decided here
        if !point_in_extent(b, &en.v, &b.cidx, &shape_v, wrt)? {
            return Ok(None);
        }
        body
    };
    if shape_u.is_empty() {
        return Ok(Some(val));
    }
    Ok(Some(json!({
        "op": "makearray", "args": [],
        "regions": [shape_u.iter().map(|n| json!([1, n])).collect::<Vec<_>>(),
                    b.rows.iter().map(|(lo, hi)| json!([lo, hi])).collect::<Vec<_>>()],
        "values": [0.0, val],
    })))
}

// ── the equality-filter scatter (§9.3, §9.4) ────────────────────────────────

/// One entry's term landing in output variable `out` (shape `shape_o`) at
/// column index `ci`: Σ over the points with `ci == j` of `body`, under the
/// extra filter conjuncts `conj`.
#[allow(clippy::too_many_arguments)]
fn scatter_term(
    b: &Band,
    coef: &Expr,
    out: &str,
    ci: &[Expr],
    shape_o: &[i64],
    wrt: Wrt,
    conj: Vec<Conj>,
    body: Value,
    bodynames: &HashSet<String>,
    taken: &HashSet<String>,
) -> R<Option<Value>> {
    // the equality drops every ghost, but on the forcing axis a dropped
    // point is a lost gradient
    if wrt == Wrt::Forcing {
        ghost_conjuncts(b, out, ci, shape_o, wrt)?;
    }
    if !has_names(b) && shape_o.is_empty() {
        return Ok(closed_point(b)?.then_some(body));
    }
    let j = jnames(taken, b, coef, shape_o.len());
    let mut all = where_conj(b);
    all.extend(conj);
    for (e, jd) in ci.iter().zip(&j) {
        let mut ns = free_variables(e);
        ns.insert(jd.clone());
        all.push((sop("==", vec![ser_expr(e), json!(jd)]), ns));
    }
    let outranges: Vec<(String, i64, i64)> = j
        .iter()
        .zip(shape_o)
        .map(|(n, s)| (n.clone(), 1, *s))
        .collect();
    let output_idx = j.iter().map(|n| json!(n)).collect();
    term_faq(b, output_idx, &outranges, all, body, bodynames).map(Some)
}

fn vjp_term(
    en: &JacEntry,
    coef: &Expr,
    lam: &str,
    ctx: &Ctx,
    wrt: Wrt,
    taken: &HashSet<String>,
) -> R<Option<Value>> {
    vjp_term_read(en, coef, sread(lam, rho(&en.band)), ctx, wrt, taken)
}

/// [`vjp_term`] with the cotangent read at the row point given: a plain
/// input, or a reversed adjoint sweep (§9.7).
fn vjp_term_read(
    en: &JacEntry,
    coef: &Expr,
    lam: Value,
    ctx: &Ctx,
    wrt: Wrt,
    taken: &HashSet<String>,
) -> R<Option<Value>> {
    let b = &en.band;
    let body = sop("*", vec![ser_expr(coef), lam]);
    let mut bodynames = free_variables(coef);
    bodynames.extend(row_names(b));
    scatter_term(
        b,
        coef,
        &en.v,
        &b.cidx,
        &shape_of(ctx, &en.v),
        wrt,
        Vec::new(),
        body,
        &bodynames,
        taken,
    )
}

// ── second-order contraction (§9.4) ─────────────────────────────────────────

/// The stage inputs of a contraction.
struct Stages<'a> {
    mu: &'a [IoMap],
    w: &'a [IoMap],
}

/// `form = Contract`: scatter by `col2_idx`, M summed over the stages.
fn hvp_term(
    en: &JacEntry2,
    coef: &Expr,
    st: &Stages,
    ctx: &Ctx,
    (a1, a2): (Wrt, Wrt),
    taken: &HashSet<String>,
) -> R<Option<Value>> {
    let b = &en.band;
    let (ci, c2) = (en.col_idx(), en.col2_idx());
    let m = sfold(
        "+",
        st.mu
            .iter()
            .zip(st.w)
            .map(|(mu, w)| {
                Ok(sop(
                    "*",
                    vec![
                        sread(io_name(mu, &en.u)?, rho(b)),
                        sread(io_name(w, &en.v)?, ser_all(ci)),
                    ],
                ))
            })
            .collect::<R<_>>()?,
    );
    let shape1 = shape_of(ctx, &en.v);
    let g = if has_names(b) {
        ghost_conjuncts(b, &en.v, ci, &shape1, a1)?
    } else if point_in_extent(b, &en.v, ci, &shape1, a1)? {
        Vec::new()
    } else {
        return Ok(None);
    };
    let mut bodynames = names_of(std::iter::once(coef).chain(ci));
    bodynames.extend(row_names(b));
    scatter_term(
        b,
        coef,
        &en.w,
        c2,
        &shape_of(ctx, &en.w),
        a2,
        g,
        sop("*", vec![ser_expr(coef), m]),
        &bodynames,
        taken,
    )
}

/// `form = Symmetric` on (states, states): scatter by the FIRST column.
fn hvp_sym_term(
    en: &JacEntry2,
    coef: &Expr,
    st: &Stages,
    ctx: &Ctx,
    taken: &HashSet<String>,
) -> R<Option<Value>> {
    let b = &en.band;
    let (ci, c2) = (en.col_idx(), en.col2_idx());
    let body = sfold(
        "+",
        st.mu
            .iter()
            .zip(st.w)
            .map(|(mu, w)| {
                Ok(sop(
                    "*",
                    vec![
                        sread(io_name(mu, &en.u)?, rho(b)),
                        ser_expr(coef),
                        sread(io_name(w, &en.w)?, ser_all(c2)),
                    ],
                ))
            })
            .collect::<R<_>>()?,
    );
    let shape2 = shape_of(ctx, &en.w);
    let g = if has_names(b) {
        ghost_conjuncts(b, &en.w, c2, &shape2, Wrt::States)?
    } else if point_in_extent(b, &en.w, c2, &shape2, Wrt::States)? {
        Vec::new()
    } else {
        return Ok(None);
    };
    let mut bodynames = names_of(std::iter::once(coef).chain(c2));
    bodynames.extend(row_names(b));
    scatter_term(
        b,
        coef,
        &en.v,
        ci,
        &shape_of(ctx, &en.v),
        Wrt::States,
        g,
        body,
        &bodynames,
        taken,
    )
}

// ── the documents ───────────────────────────────────────────────────────────

/// The `J v` product document of spec §9.2 for the first-order `entries` on
/// axis `wrt` (as [`crate::jacobian_bands_view`] gives them): a derived ESM
/// document (model `"JvpEval"`) whose right-hand side, evaluated by an
/// unmodified EarthSciAST evaluator at `(u, inputs, p, t)`, leaves `J v` in
/// the derivative slots of its outputs. The consumer writes the inputs
/// ([`JvpIo::v`]) into the evaluator's state slots beside the source states
/// and reads the outputs ([`JvpIo::jvp`]) from `du`.
///
/// `entries` must be built under the same forcing set as `opts.forcing`
/// (`None` on both sides infers it): the document's columns come from
/// `wrt` and `opts.forcing`, and an entry reading a variable outside them is
/// refused. A column of that set the entries never read comes out zero.
///
/// Through an active recurrence (spec §4.5) the document is §9.7's tangent
/// sweep over the extended system, which this function builds itself:
/// `entries` then only serve that forcing-set check. Pass the extended
/// system's rows ([`crate::extended_bands`]`(sv, wrt, forcing,
/// None)?.rows`, whose columns may also be a recurrence) rather than the
/// materialized bands, so the transfer recurrence is never composed.
pub fn jvp_document(
    sv: &SysView,
    entries: &[JacEntry],
    wrt: Wrt,
    opts: &ProductOptions,
) -> R<(Value, JvpIo)> {
    let model = "JvpEval";
    let ext = extended_for(sv, wrt, opts.forcing)?;
    if !ext.recurrences.is_empty() {
        check_sweep_columns(sv, &ext, entries, wrt, opts)?;
        return jvp_sweeps(sv, &ext, wrt, opts);
    }
    let bands: Vec<&Band> = entries.iter().map(|en| &en.band).collect();
    let (mut d, coefs) = product_base(sv, model, &bands, opts)?;
    let mut taken = doc_names(&d.doc, &d.model);
    let vin: IoMap = column_vars(sv, wrt, opts.forcing)?
        .into_iter()
        .map(|c| {
            let nm = fresh(&mut taken, &format!("_v_{c}"));
            (c, nm)
        })
        .collect();
    let out: IoMap = view_states(sv, &observed_defs(sv)?)
        .into_iter()
        .map(|r| {
            let nm = fresh(&mut taken, &format!("_jvp_{r}"));
            (r, nm)
        })
        .collect();
    check_columns(&vin, entries.iter().map(|en| en.v.as_str()), wrt)?;
    for (c, nm) in &vin {
        d.declare(nm, c, None);
    }
    let owner: Vec<&str> = entries.iter().map(|en| en.u.as_str()).collect();
    d.outputs(&out, &owner, |k, ctx| {
        jvp_term(
            &entries[k],
            &coefs[k],
            io_name(&vin, &entries[k].v)?,
            ctx,
            wrt,
        )
    })?;
    let io = JvpIo {
        model: model.into(),
        v: vin,
        jvp: out,
    };
    Ok((d.finish(model), io))
}

/// The `Jᵀ λ` product document of spec §9.3 (model `"VjpEval"`): as
/// [`jvp_document`], with the cotangent inputs [`VjpIo::lam`] and the
/// outputs [`VjpIo::vjp`]. An equality-filtered `faq` scatter: correct under
/// any evaluator, linear-time only under one that implements §9.3's
/// equality-filter scatter. `entries` and `opts.forcing` must agree, as for
/// [`jvp_document`], which also says what to pass through a recurrence.
pub fn vjp_document(
    sv: &SysView,
    entries: &[JacEntry],
    wrt: Wrt,
    opts: &ProductOptions,
) -> R<(Value, VjpIo)> {
    let model = "VjpEval";
    let ext = extended_for(sv, wrt, opts.forcing)?;
    if !ext.recurrences.is_empty() {
        check_sweep_columns(sv, &ext, entries, wrt, opts)?;
        return vjp_sweeps(sv, &ext, wrt, opts);
    }
    let bands: Vec<&Band> = entries.iter().map(|en| &en.band).collect();
    let (mut d, coefs) = product_base(sv, model, &bands, opts)?;
    let mut taken = doc_names(&d.doc, &d.model);
    let lam: IoMap = view_states(sv, &observed_defs(sv)?)
        .into_iter()
        .map(|r| {
            let nm = fresh(&mut taken, &format!("_lam_{r}"));
            (r, nm)
        })
        .collect();
    let out: IoMap = column_vars(sv, wrt, opts.forcing)?
        .into_iter()
        .map(|c| {
            let nm = fresh(&mut taken, &format!("_vjp_{c}"));
            (c, nm)
        })
        .collect();
    check_columns(&out, entries.iter().map(|en| en.v.as_str()), wrt)?;
    for (r, nm) in &lam {
        d.declare(nm, r, None);
    }
    let owner: Vec<&str> = entries.iter().map(|en| en.v.as_str()).collect();
    d.outputs(&out, &owner, |k, ctx| {
        vjp_term(
            &entries[k],
            &coefs[k],
            io_name(&lam, &entries[k].u)?,
            ctx,
            wrt,
            &taken,
        )
    })?;
    let io = VjpIo {
        model: model.into(),
        lam,
        vjp: out,
    };
    Ok((d.finish(model), io))
}

/// The second-order product document of spec §9.4 for the axis pair `wrt`,
/// the second-order `entries` (as [`crate::jacobian_bands2`] gives them) and
/// `stages` stage pairs: its right-hand side leaves `g = ∇_{a2} Σᵢ μᵢᵀ J(u)
/// wᵢ` (`∇_{a2} (Σᵢ wᵢ μᵢ)ᵀ f_t` on a time pair) in the derivative slots of
/// its outputs. [`HvpForm::Symmetric`] (a `(states, states)` pair only)
/// scatters by the first column instead (model `"HvpEvalSymmetric"`).
/// `entries` and `opts.forcing` must agree, as for [`jvp_document`].
pub fn hvp_document(
    sv: &SysView,
    entries: &[JacEntry2],
    wrt: (Wrt, Wrt),
    stages: usize,
    form: HvpForm,
    opts: &ProductOptions,
) -> R<(Value, HvpIo)> {
    let (a1, a2) = axis_pair(wrt)?;
    if form == HvpForm::Symmetric && (a1, a2) != (Wrt::States, Wrt::States) {
        return Err(BandError(format!(
            "the symmetric form needs a (states, states) pair, got ({}, {})",
            a1.as_str(),
            a2.as_str()
        )));
    }
    if stages < 1 {
        return Err(BandError(
            "a contraction needs at least one stage, got 0".into(),
        ));
    }
    let model = match form {
        HvpForm::Contract => "HvpEval",
        HvpForm::Symmetric => "HvpEvalSymmetric",
    };
    if let Some((e1, e2)) = pair_extended(sv, (a1, a2), opts.forcing)? {
        let e2 = e2.as_ref().unwrap_or(&e1);
        return hvp_sweeps(sv, entries, (&e1, e2), (a1, a2), stages, form, model, opts);
    }
    // the bands only, borrowed: a second-order band rides through the base
    // and the hoisting as a first-order one (its `cidx` holds both columns)
    let bands: Vec<&Band> = entries.iter().map(|en| &en.band).collect();
    let (mut d, coefs) = product_base(sv, model, &bands, opts)?;
    let mut taken = doc_names(&d.doc, &d.model);
    let states = view_states(sv, &observed_defs(sv)?);
    let cols1 = if a1 == Wrt::Time {
        vec!["t".to_string()]
    } else {
        states.clone()
    };
    let mut mu: Vec<IoMap> = Vec::new();
    let mut w: Vec<IoMap> = Vec::new();
    for i in 1..=stages {
        let mut alloc = |cols: &[String], tag: &str| -> IoMap {
            cols.iter()
                .map(|c| (c.clone(), fresh(&mut taken, &format!("_{tag}{i}_{c}"))))
                .collect()
        };
        mu.push(alloc(&states, "mu"));
        w.push(alloc(&cols1, "w"));
    }
    let cols2 = match form {
        HvpForm::Contract => column_vars(sv, a2, opts.forcing)?,
        HvpForm::Symmetric => states,
    };
    let out: IoMap = cols2
        .into_iter()
        .map(|c| {
            let nm = fresh(&mut taken, &format!("_hvp_{c}"));
            (c, nm)
        })
        .collect();
    if form == HvpForm::Contract {
        check_columns(&out, entries.iter().map(|en| en.w.as_str()), a2)?;
    }
    for (mi, wi) in mu.iter().zip(&w) {
        for (r, nm) in mi {
            d.declare(nm, r, None);
        }
        for (c, nm) in wi {
            d.declare(nm, c, None);
        }
    }
    let st = Stages { mu: &mu, w: &w };
    match form {
        HvpForm::Contract => {
            let owner: Vec<&str> = entries.iter().map(|en| en.w.as_str()).collect();
            d.outputs(&out, &owner, |k, ctx| {
                hvp_term(&entries[k], &coefs[k], &st, ctx, (a1, a2), &taken)
            })?;
        }
        HvpForm::Symmetric => {
            let owner: Vec<&str> = entries.iter().map(|en| en.v.as_str()).collect();
            d.outputs(&out, &owner, |k, ctx| {
                hvp_sym_term(&entries[k], &coefs[k], &st, ctx, &taken)
            })?;
        }
    }
    let io = HvpIo {
        model: model.into(),
        mu,
        w,
        hvp: out,
    };
    Ok((d.finish(model), io))
}

// ── recurrences: tangent and adjoint sweeps (§9.7) ──────────────────────────

/// The forcing-set check of a sweep document's `entries`: materialized
/// bands or the extended system's rows, so a column may be a target or an
/// active recurrence.
fn check_sweep_columns(
    sv: &SysView,
    ext: &ExtendedBands,
    entries: &[JacEntry],
    wrt: Wrt,
    opts: &ProductOptions,
) -> R<()> {
    let cols: IoMap = column_vars(sv, wrt, opts.forcing)?
        .into_iter()
        .map(|c| (c.clone(), c))
        .collect();
    check_columns(
        &cols,
        entries
            .iter()
            .map(|en| en.v.as_str())
            .filter(|v| !ext.is_recurrence(v)),
        wrt,
    )
}

/// The extended system of axis `wrt` (spec §4.5).
/// Empty (no pass run) when the view has no recurrence at all.
fn extended_for(sv: &SysView, wrt: Wrt, forcing: Option<&[String]>) -> R<ExtendedBands> {
    let ctx = ctx_of(sv)?;
    let obs = observed_defs(sv)?;
    if recurrences(sv, &obs, &ctx)?.is_empty() {
        return Ok(ExtendedBands::default());
    }
    extended_pass(sv, &ctx, &obs, wrt, forcing, &[])
}

/// The two extended systems of a pair with an active recurrence: the first
/// axis's (state rows and recurrence rows, seeded with the second axis's
/// targets, as [`crate::jacobian_bands2`] differentiates it) and the second
/// axis's (its recurrence rows carry the adjoint back to the second axis's
/// columns), both over the recurrences that move with either axis — the
/// second `None` on a pair of one axis, whose second pass is the first.
/// `None` without an active recurrence.
fn pair_extended(
    sv: &SysView,
    (a1, a2): (Wrt, Wrt),
    forcing: Option<&[String]>,
) -> R<Option<(ExtendedBands, Option<ExtendedBands>)>> {
    let ctx = ctx_of(sv)?;
    let obs = observed_defs(sv)?;
    if recurrences(sv, &obs, &ctx)?.is_empty() {
        return Ok(None);
    }
    let t1 = targets(sv, a1, forcing, &obs)?;
    let t2 = targets(sv, a2, forcing, &obs)?;
    let e1 = extended_pass(sv, &ctx, &obs, a1, forcing, &t2)?;
    if e1.recurrences.is_empty() {
        return Ok(None);
    }
    let e2 = if a1 == a2 {
        None
    } else {
        Some(extended_pass(sv, &ctx, &obs, a2, forcing, &t1)?)
    };
    Ok(Some((e1, e2)))
}

/// The index expressions of a read of `v` at `ci` from band `b` as `Expr`
/// conjuncts of §9.1's ghost rule (the forcing axis refuses a ghost).
fn ghost_exprs(
    b: &Band,
    v: &str,
    ci: &[Expr],
    ci_sub: &[Expr],
    shape: &[i64],
    wrt: Wrt,
) -> R<Vec<Expr>> {
    if !needs_ghost(b, v, ci, shape, wrt)? {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for (e, n) in ci_sub.iter().zip(shape) {
        out.push(crate::helpers::op(">=", vec![e.clone(), Expr::Integer(1)]));
        out.push(crate::helpers::op("<=", vec![e.clone(), Expr::Integer(*n)]));
    }
    Ok(out)
}

/// A band with its coefficient replaced (the hoisted one).
fn with_coef(b: &Band, coef: &Expr) -> Band {
    Band {
        coef: coef.clone(),
        ..b.clone()
    }
}

/// `index(v, idx…)` (bare for a scalar) as an `Expr`.
fn eread(v: &str, idx: Vec<Expr>) -> Expr {
    if idx.is_empty() {
        return Expr::Variable(v.into());
    }
    let mut args = vec![Expr::Variable(v.into())];
    args.extend(idx);
    crate::helpers::op("index", args)
}

/// The row point of `b` as index expressions.
fn rho_e(b: &Band) -> Vec<Expr> {
    b.ridx
        .iter()
        .map(|r| match r {
            RIdx::Name(s) => Expr::Variable(s.clone()),
            RIdx::Fixed(n) => Expr::Integer(*n),
        })
        .collect()
}

/// The forward sweep variable of `rec` (`target`): its own bands among
/// `rows` (with their hoisted coefficients `coefs`), reading a target
/// column through `dir` and another recurrence's sweep through `sweeps`.
#[allow(clippy::too_many_arguments)]
fn tangent(
    rec: &Recurrence,
    rows: &[&JacEntry],
    coefs: &[Expr],
    target: &str,
    dir: &IoMap,
    sweeps: &IoMap,
    ctx: &Ctx,
    wrt: Wrt,
) -> R<Value> {
    let mut own = Vec::new();
    let mut cols = Vec::new();
    for (en, c) in rows.iter().zip(coefs) {
        if en.u == rec.name {
            own.push(with_coef(&en.band, c));
            cols.push(en.v.as_str());
        }
    }
    let own_refs: Vec<&Band> = own.iter().collect();
    let col = |b: &Band, c: &str, ci: Vec<Expr>| -> R<Option<(Expr, Vec<Expr>)>> {
        let shape = shape_of(ctx, c);
        let (nm, w) = match sweeps.get(c) {
            Some(nm) => (nm.as_str(), Wrt::States),
            None => (io_name(dir, c)?, wrt),
        };
        let conds = ghost_exprs(b, c, &b.cidx, &ci, &shape, w)?;
        Ok(Some((eread(nm, ci), conds)))
    };
    let e = forward_sweep(rec, &own_refs, &cols, target, &col, &Expr::Number(0.0))?;
    Ok(ser_expr(&e))
}

/// The adjoint sweep variable of `rec` (`target`, reversed axis) fed by the
/// cotangent source `src` (a variable of `rec`'s shape): `rec`'s self
/// bands among `rows`.
fn adjoint(
    rec: &Recurrence,
    rows: &[&JacEntry],
    coefs: &[Expr],
    target: &str,
    src: &str,
) -> R<Value> {
    let own: Vec<Band> = rows
        .iter()
        .zip(coefs)
        .filter(|(en, _)| en.u == rec.name && en.v == rec.name)
        .map(|(en, c)| with_coef(&en.band, c))
        .collect();
    let refs: Vec<&Band> = own.iter().collect();
    let frame: Vec<Expr> = rec
        .frame
        .iter()
        .map(|f| Expr::Variable(f.clone()))
        .collect();
    let e = adjoint_sweep(
        rec,
        &refs,
        target,
        reversed_read(rec, src, frame),
        &Expr::Number(0.0),
    )?;
    Ok(ser_expr(&e))
}

/// A cotangent read at the row point of `b`: a plain input (`plain`, a
/// state row) or a reversed adjoint sweep (`rev`, a recurrence row).
fn cot_read(b: &Band, u: &str, plain: &IoMap, rev: &IoMap, ext: &ExtendedBands) -> R<Value> {
    match (ext.recurrence(u), rev.get(u)) {
        (Some(rec), Some(nm)) => Ok(ser_expr(&reversed_read(rec, nm, rho_e(b)))),
        _ => Ok(sread(io_name(plain, u)?, rho(b))),
    }
}

/// The cotangent source of recurrence `v`: Σ over the bands reading `v`
/// (`entries`, other than `v`'s own) of `coef · λ[row]`, scattered onto
/// `v`'s cells (an equality-filter scatter, §9.3); zero when none reads it.
#[allow(clippy::too_many_arguments)]
fn cot_source(
    v: &str,
    entries: &[&JacEntry],
    coefs: &[Expr],
    lam: &IoMap,
    rev: &IoMap,
    ext: &ExtendedBands,
    ctx: &Ctx,
    taken: &HashSet<String>,
) -> R<Value> {
    let mut terms = Vec::new();
    for (en, c) in entries.iter().zip(coefs) {
        if en.v != v || en.u == v {
            continue;
        }
        let l = cot_read(&en.band, &en.u, lam, rev, ext)?;
        if let Some(t) = vjp_term_read(en, c, l, ctx, Wrt::States, taken)? {
            terms.push(t);
        }
    }
    Ok(if terms.is_empty() {
        zeros(&shape_of(ctx, v))
    } else {
        sfold("+", terms)
    })
}

/// Fresh names `_<tag><stage>_<V>` for every recurrence, in order.
fn rec_names(taken: &mut HashSet<String>, ext: &ExtendedBands, tag: &str) -> IoMap {
    ext.recurrences
        .iter()
        .map(|r| (r.name.clone(), fresh(taken, &format!("_{tag}_{}", r.name))))
        .collect()
}

fn jvp_sweeps(
    sv: &SysView,
    ext: &ExtendedBands,
    wrt: Wrt,
    opts: &ProductOptions,
) -> R<(Value, JvpIo)> {
    let model = "JvpEval";
    let all: Vec<&JacEntry> = ext.rows.iter().chain(&ext.rec_rows).collect();
    let bands: Vec<&Band> = all.iter().map(|en| &en.band).collect();
    let (mut d, coefs) = product_base(sv, model, &bands, opts)?;
    let mut taken = doc_names(&d.doc, &d.model);
    let vin: IoMap = column_vars(sv, wrt, opts.forcing)?
        .into_iter()
        .map(|c| {
            let nm = fresh(&mut taken, &format!("_v_{c}"));
            (c, nm)
        })
        .collect();
    let vdot = rec_names(&mut taken, ext, "vdot");
    let out: IoMap = view_states(sv, &observed_defs(sv)?)
        .into_iter()
        .map(|r| {
            let nm = fresh(&mut taken, &format!("_jvp_{r}"));
            (r, nm)
        })
        .collect();
    for (c, nm) in &vin {
        d.declare(nm, c, None);
    }
    let nrow = ext.rows.len();
    for rec in &ext.recurrences {
        let e = tangent(
            rec,
            &all[nrow..],
            &coefs[nrow..],
            &vdot[&rec.name],
            &vin,
            &vdot,
            &d.ctx,
            wrt,
        )?;
        d.observe(&vdot[&rec.name], &rec.name, e);
    }
    let owner: Vec<&str> = ext.rows.iter().map(|en| en.u.as_str()).collect();
    d.outputs(&out, &owner, |k, ctx| {
        let en = &ext.rows[k];
        match vdot.get(&en.v) {
            Some(nm) => jvp_term(en, &coefs[k], nm, ctx, Wrt::States),
            None => jvp_term(en, &coefs[k], io_name(&vin, &en.v)?, ctx, wrt),
        }
    })?;
    let io = JvpIo {
        model: model.into(),
        v: vin,
        jvp: out,
    };
    Ok((d.finish(model), io))
}

fn vjp_sweeps(
    sv: &SysView,
    ext: &ExtendedBands,
    wrt: Wrt,
    opts: &ProductOptions,
) -> R<(Value, VjpIo)> {
    let model = "VjpEval";
    let all: Vec<&JacEntry> = ext.rows.iter().chain(&ext.rec_rows).collect();
    let bands: Vec<&Band> = all.iter().map(|en| &en.band).collect();
    let (mut d, coefs) = product_base(sv, model, &bands, opts)?;
    let mut taken = doc_names(&d.doc, &d.model);
    let lam: IoMap = view_states(sv, &observed_defs(sv)?)
        .into_iter()
        .map(|r| {
            let nm = fresh(&mut taken, &format!("_lam_{r}"));
            (r, nm)
        })
        .collect();
    let vbar = rec_names(&mut taken, ext, "vbar");
    let vrev = rec_names(&mut taken, ext, "vrev");
    let out: IoMap = column_vars(sv, wrt, opts.forcing)?
        .into_iter()
        .map(|c| {
            let nm = fresh(&mut taken, &format!("_vjp_{c}"));
            (c, nm)
        })
        .collect();
    for (r, nm) in &lam {
        d.declare(nm, r, None);
    }
    let nrow = ext.rows.len();
    for rec in &ext.recurrences {
        let v = &rec.name;
        let src = cot_source(v, &all, &coefs, &lam, &vrev, ext, &d.ctx, &taken)?;
        d.observe(&vbar[v], v, src);
        let e = adjoint(rec, &all[nrow..], &coefs[nrow..], &vrev[v], &vbar[v])?;
        d.observe(&vrev[v], v, e);
    }
    let owner: Vec<&str> = all.iter().map(|en| en.v.as_str()).collect();
    d.outputs(&out, &owner, |k, ctx| {
        let en = all[k];
        let l = cot_read(&en.band, &en.u, &lam, &vrev, ext)?;
        vjp_term_read(en, &coefs[k], l, ctx, wrt, &taken)
    })?;
    let io = VjpIo {
        model: model.into(),
        lam,
        vjp: out,
    };
    Ok((d.finish(model), io))
}

/// The §4.4 contraction through recurrences (§9.7): per stage `i`, the
/// tangent sweep `V̇ᵢ` of `wᵢ` and the adjoint sweep `Âᵢ` of `μᵢ` over the
/// first axis's extended system make the contraction's `M` read a
/// recurrence row's cotangent and a recurrence column's direction; the
/// second-order bands scattered onto a recurrence column feed one more
/// adjoint sweep `Ĥ`, which the second axis's recurrence rows carry back to
/// its columns.
#[allow(clippy::too_many_arguments)]
fn hvp_sweeps(
    sv: &SysView,
    entries: &[JacEntry2],
    (e1, e2): (&ExtendedBands, &ExtendedBands),
    (a1, a2): (Wrt, Wrt),
    stages: usize,
    form: HvpForm,
    model: &str,
    opts: &ProductOptions,
) -> R<(Value, HvpIo)> {
    let first: Vec<&JacEntry> = e1.rows.iter().chain(&e1.rec_rows).collect();
    let nrow = e1.rows.len();
    // on a pair of one axis the second pass IS the first: its recurrence
    // rows (and their hoisted coefficients) are shared, not emitted twice
    let shared = a1 == a2;
    let back: Vec<&JacEntry> = if shared {
        first[nrow..].to_vec()
    } else {
        e2.rec_rows.iter().collect()
    };
    let mut bands: Vec<&Band> = entries.iter().map(|en| &en.band).collect();
    bands.extend(first.iter().map(|en| &en.band));
    if !shared {
        bands.extend(back.iter().map(|en| &en.band));
    }
    let (mut d, coefs) = product_base(sv, model, &bands, opts)?;
    let n2 = entries.len();
    let c2s = &coefs[..n2];
    let c1s = &coefs[n2..n2 + first.len()];
    let cbs = if shared {
        &c1s[nrow..]
    } else {
        &coefs[n2 + first.len()..]
    };
    let mut taken = doc_names(&d.doc, &d.model);
    let states = view_states(sv, &observed_defs(sv)?);
    let cols1 = if a1 == Wrt::Time {
        vec!["t".to_string()]
    } else {
        states.clone()
    };
    let mut mu: Vec<IoMap> = Vec::new();
    let mut w: Vec<IoMap> = Vec::new();
    for i in 1..=stages {
        let mut alloc = |cols: &[String], tag: &str| -> IoMap {
            cols.iter()
                .map(|c| (c.clone(), fresh(&mut taken, &format!("_{tag}{i}_{c}"))))
                .collect()
        };
        mu.push(alloc(&states, "mu"));
        w.push(alloc(&cols1, "w"));
    }
    let mut vdot = Vec::new();
    let mut vbar = Vec::new();
    let mut vrev = Vec::new();
    for i in 1..=stages {
        vdot.push(rec_names(&mut taken, e1, &format!("vdot{i}")));
        vbar.push(rec_names(&mut taken, e1, &format!("vbar{i}")));
        vrev.push(rec_names(&mut taken, e1, &format!("vrev{i}")));
    }
    let hbar = rec_names(&mut taken, e1, "hbar");
    let hrev = rec_names(&mut taken, e1, "hrev");
    let cols2 = match form {
        HvpForm::Contract => column_vars(sv, a2, opts.forcing)?,
        HvpForm::Symmetric => states,
    };
    let out: IoMap = cols2
        .into_iter()
        .map(|c| {
            let nm = fresh(&mut taken, &format!("_hvp_{c}"));
            (c, nm)
        })
        .collect();
    if form == HvpForm::Contract {
        check_columns(
            &out,
            entries
                .iter()
                .map(|en| en.w.as_str())
                .filter(|x| !e1.is_recurrence(x)),
            a2,
        )?;
    }
    for (mi, wi) in mu.iter().zip(&w) {
        for (r, nm) in mi {
            d.declare(nm, r, None);
        }
        for (c, nm) in wi {
            d.declare(nm, c, None);
        }
    }
    for i in 0..stages {
        for rec in &e1.recurrences {
            let v = &rec.name;
            let t = tangent(
                rec,
                &first[nrow..],
                &c1s[nrow..],
                &vdot[i][v],
                &w[i],
                &vdot[i],
                &d.ctx,
                a1,
            )?;
            d.observe(&vdot[i][v], v, t);
            let src = cot_source(v, &first, c1s, &mu[i], &vrev[i], e1, &d.ctx, &taken)?;
            d.observe(&vbar[i][v], v, src);
            let a = adjoint(rec, &first[nrow..], &c1s[nrow..], &vrev[i][v], &vbar[i][v])?;
            d.observe(&vrev[i][v], v, a);
        }
    }
    let ctx = d.ctx.clone();
    // per stage, the cotangent at a row and the direction at a column
    let cot = |i: usize, b: &Band, u: &str| cot_read(b, u, &mu[i], &vrev[i], e1);
    let dir = |i: usize, v: &str, ci: &[Expr]| -> R<Value> {
        Ok(match vdot[i].get(v) {
            Some(nm) => sread(nm, ser_all(ci)),
            None => sread(io_name(&w[i], v)?, ser_all(ci)),
        })
    };
    let axis = |v: &str, a: Wrt| if e1.is_recurrence(v) { Wrt::States } else { a };
    // one second-order entry's scatter term: by `col2` (contract) or by
    // the first column (symmetric)
    let term2 = |k: usize, ctx: &Ctx| -> R<Option<Value>> {
        let en = &entries[k];
        let b = &en.band;
        let coef = &c2s[k];
        let (ci, c2) = (en.col_idx(), en.col2_idx());
        let mut bodynames = names_of(std::iter::once(coef).chain(ci).chain(c2));
        bodynames.extend(row_names(b));
        match form {
            HvpForm::Contract => {
                let m = sfold(
                    "+",
                    (0..stages)
                        .map(|i| Ok(sop("*", vec![cot(i, b, &en.u)?, dir(i, &en.v, ci)?])))
                        .collect::<R<_>>()?,
                );
                let shape1 = shape_of(ctx, &en.v);
                let a = axis(&en.v, a1);
                let g = if has_names(b) {
                    ghost_conjuncts(b, &en.v, ci, &shape1, a)?
                } else if point_in_extent(b, &en.v, ci, &shape1, a)? {
                    Vec::new()
                } else {
                    return Ok(None);
                };
                scatter_term(
                    b,
                    coef,
                    &en.w,
                    c2,
                    &shape_of(ctx, &en.w),
                    axis(&en.w, a2),
                    g,
                    sop("*", vec![ser_expr(coef), m]),
                    &bodynames,
                    &taken,
                )
            }
            HvpForm::Symmetric => {
                let body = sfold(
                    "+",
                    (0..stages)
                        .map(|i| {
                            Ok(sop(
                                "*",
                                vec![cot(i, b, &en.u)?, ser_expr(coef), dir(i, &en.w, c2)?],
                            ))
                        })
                        .collect::<R<_>>()?,
                );
                let shape2 = shape_of(ctx, &en.w);
                let g = if has_names(b) {
                    ghost_conjuncts(b, &en.w, c2, &shape2, Wrt::States)?
                } else if point_in_extent(b, &en.w, c2, &shape2, Wrt::States)? {
                    Vec::new()
                } else {
                    return Ok(None);
                };
                scatter_term(
                    b,
                    coef,
                    &en.v,
                    ci,
                    &shape_of(ctx, &en.v),
                    Wrt::States,
                    g,
                    body,
                    &bodynames,
                    &taken,
                )
            }
        }
    };
    let owner2: Vec<&str> = entries
        .iter()
        .map(|en| match form {
            HvpForm::Contract => en.w.as_str(),
            HvpForm::Symmetric => en.v.as_str(),
        })
        .collect();
    // a recurrence row of the second axis carries Ĥ to its column
    let back_term = |k: usize, ctx: &Ctx| -> R<Option<Value>> {
        let en = back[k];
        let l = cot_read(&en.band, &en.u, &IoMap::new(), &hrev, e2)?;
        vjp_term_read(en, &cbs[k], l, ctx, axis(&en.v, a2), &taken)
    };
    for rec in &e1.recurrences {
        let v = &rec.name;
        let mut terms = Vec::new();
        for k in 0..n2 {
            if owner2[k] == v {
                if let Some(t) = term2(k, &ctx)? {
                    terms.push(t);
                }
            }
        }
        for k in 0..back.len() {
            if back[k].v == *v && back[k].u != *v {
                if let Some(t) = back_term(k, &ctx)? {
                    terms.push(t);
                }
            }
        }
        let src = if terms.is_empty() {
            zeros(&shape_of(&ctx, v))
        } else {
            sfold("+", terms)
        };
        d.observe(&hbar[v], v, src);
        let a = adjoint(rec, &back, cbs, &hrev[v], &hbar[v])?;
        d.observe(&hrev[v], v, a);
    }
    let mut owner: Vec<&str> = owner2.clone();
    owner.extend(back.iter().map(|en| en.v.as_str()));
    d.outputs(&out, &owner, |k, ctx| {
        if k < n2 {
            term2(k, ctx)
        } else {
            back_term(k - n2, ctx)
        }
    })?;
    let io = HvpIo {
        model: model.into(),
        mu,
        w,
        hvp: out,
    };
    Ok((d.finish(model), io))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::op;

    fn var(n: &str) -> Expr {
        Expr::Variable(n.into())
    }
    fn int(n: i64) -> Expr {
        Expr::Integer(n)
    }
    fn add(a: Expr, b: Expr) -> Expr {
        op("+", vec![a, b])
    }
    fn sub(a: Expr, b: Expr) -> Expr {
        op("-", vec![a, b])
    }
    fn mul(a: Expr, b: Expr) -> Expr {
        op("*", vec![a, b])
    }
    /// A band over rows `i ∈ lo..hi`, reading column `i`.
    fn rows(lo: i64, hi: i64) -> Band {
        Band {
            rows: vec![(lo, hi)],
            ridx: vec![RIdx::Name("i".into())],
            cidx: vec![var("i")],
            coef: Expr::Number(1.0),
            contracted: Vec::new(),
            guard: None,
        }
    }
    /// One point (row 3), reading its column at the literal index `ci`.
    fn lit(ci: &[i64]) -> Band {
        Band {
            rows: vec![(3, 3)],
            ridx: vec![RIdx::Fixed(3)],
            cidx: ci.iter().map(|n| int(*n)).collect(),
            coef: Expr::Number(1.0),
            contracted: Vec::new(),
            guard: None,
        }
    }
    fn walk(b: &Band, ci: &[Expr], n: i64) -> bool {
        all_points(&walk_band(b, ci, b.guard.clone()), &mut |_, _, _, c| {
            Ok(c.iter().all(|x| 1 <= *x && *x <= n))
        })
        .unwrap()
    }

    #[test]
    fn bounds_decide_what_a_walk_would() {
        let mut b = rows(2, 7);
        b.contracted = vec![("k".into(), 1, 3)];
        let (i, k) = (var("i"), var("k"));
        for e in [
            i.clone(),
            add(i.clone(), int(1)),
            sub(i.clone(), int(2)),
            add(i.clone(), k.clone()),
            sub(i.clone(), k.clone()),
            mul(int(2), i.clone()),
            sub(mul(int(2), i.clone()), k.clone()),
            sub(i.clone(), i.clone()),
            add(i.clone(), i.clone()),
        ] {
            for n in [5, 8, 10, 13] {
                let ci = [e.clone()];
                assert_eq!(
                    in_extent(&b, &ci, &[n]).unwrap(),
                    Some(walk(&b, &ci, n)),
                    "{e:?} over {n}"
                );
            }
        }
    }

    #[test]
    fn guards_gathers_and_failures() {
        let i = var("i");
        let im1 = [sub(i.clone(), int(1))];
        // a guard that drops the out-of-extent corner: decided by the walk
        let mut gb = rows(1, 8);
        gb.guard = Some(op(">=", vec![i.clone(), int(2)]));
        assert_eq!(in_extent(&gb, &im1, &[8]).unwrap(), Some(true));
        assert_eq!(in_extent(&rows(1, 8), &im1, &[8]).unwrap(), Some(false));
        // an index through a table that is not `const` has no build-time answer
        let gather = [op("index", vec![var("tbl"), i.clone()])];
        assert_eq!(in_extent(&rows(1, 8), &gather, &[8]).unwrap(), None);
        // a guard that is not closed cannot drop a point: only an all-inside
        // box decides
        let mut ob = rows(1, 8);
        ob.guard = Some(op(">", vec![var("u"), Expr::Number(0.0)]));
        assert_eq!(
            in_extent(&ob, std::slice::from_ref(&i), &[8]).unwrap(),
            Some(true)
        );
        assert_eq!(
            in_extent(&ob, &[add(i.clone(), int(1))], &[8]).unwrap(),
            None
        );
        // a real evaluation failure is not taken for "not closed"
        let tbl: Expr =
            serde_json::from_value(json!({"op": "const", "args": [], "value": [1, 2, 3]})).unwrap();
        let bad = [op("index", vec![tbl, i.clone()])];
        assert!(in_extent(&rows(1, 8), &bad, &[8]).is_err());
    }

    #[test]
    fn a_band_with_no_index_names() {
        // an open column index is §9.2's refusal, not an evaluation error
        let mut nb = lit(&[1]);
        nb.cidx = vec![op("index", vec![var("tbl"), int(2)])];
        let e = point_in_extent(&nb, "v", &nb.cidx, &[8], Wrt::States).unwrap_err();
        assert!(e.0.contains("§9.2"), "{}", e.0);
        let b = lit(&[9]);
        assert!(!point_in_extent(&b, "v", &b.cidx, &[8], Wrt::States).unwrap());
        assert!(point_in_extent(&b, "v", &b.cidx, &[8], Wrt::Forcing).is_err());
        let mut shut = lit(&[9]);
        shut.guard = Some(op("<", vec![int(2), int(1)]));
        // no point, so nothing to refuse
        assert!(!point_in_extent(&shut, "v", &shut.cidx, &[8], Wrt::Forcing).unwrap());
        // an open `where` is §9.2's refusal
        let mut open = lit(&[2]);
        open.guard = Some(op(">", vec![var("u"), int(0)]));
        assert!(point_in_extent(&open, "v", &open.cidx, &[8], Wrt::States).is_err());
    }

    #[test]
    fn a_read_with_more_indices_than_dimensions_is_refused() {
        let b = lit(&[1, 1]);
        assert_eq!(in_extent(&b, &b.cidx, &[4]).unwrap(), None);
        assert_eq!(in_extent(&b, &b.cidx, &[]).unwrap(), None);
        assert_eq!(in_extent(&b, &b.cidx, &[4, 4]).unwrap(), Some(true));
        assert_eq!(in_extent(&b, &b.cidx[..1], &[4, 4]).unwrap(), Some(true));
        for wrt in [Wrt::States, Wrt::Parameters, Wrt::Forcing] {
            assert!(ghost_conjuncts(&b, "c", &b.cidx, &[4], wrt).is_err());
            assert!(point_in_extent(&b, "c", &b.cidx, &[4], wrt).is_err());
        }
    }
}
