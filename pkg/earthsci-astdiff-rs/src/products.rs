//! Product documents (the reference's products.jl; spec §9): derived ESM
//! documents whose right-hand side, evaluated once by an unmodified
//! EarthSciAST evaluator, yields `J v`, `Jᵀ λ` or the §4.4 second-order
//! contraction. Like the evaluation document ([`crate::evaluation_document`])
//! they are ASTs only: this crate evaluates nothing at run time. The one
//! numeric step is §9.1's build-time decision of which points read a column
//! inside its extent, over closed column indices. Byte-gated by the
//! `*.{jvp,vjp,hvp,hvp-sym}.evaldoc.json` goldens.

use std::collections::HashSet;

use earthsci_ast::{free_variables, Expr};
use indexmap::IndexMap;
use serde_json::{json, Map, Value};

use crate::bands::{Band, BandError, Ctx};
use crate::clip::RIdx;
use crate::emit::{base_document, derived_version, hoist, zero_source_derivatives};
use crate::helpers::ser_expr;
use crate::jacobian::{jacobian_columns, JacEntry, Wrt};
use crate::points::all_points;
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
        let rhs = rhs.unwrap_or_else(|| {
            if shape.is_empty() {
                json!(0.0)
            } else {
                json!({"op": "makearray", "args": [],
                       "regions": [shape.iter().map(|n| json!([1, n])).collect::<Vec<_>>()],
                       "values": [0.0]})
            }
        });
        if let Some(Value::Array(eqs)) = self.model.get_mut("equations") {
            eqs.push(json!({"lhs": {"op": "D", "args": [name], "wrt": "t"}, "rhs": rhs}));
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

fn with_cidx(b: &Band, ci: &[Expr]) -> Band {
    Band {
        cidx: ci.to_vec(),
        ..b.clone()
    }
}

/// Whether every point of `b` reads `ci` inside `shape`; `None` when an
/// index is not closed at build time (a gather through a table that is not
/// `const`), or when `ci` has more indices than `shape` has dimensions, so
/// the question has no build-time answer.
fn in_extent(b: &Band, ci: &[Expr], shape: &[i64]) -> Option<bool> {
    if ci.is_empty() {
        return Some(true);
    }
    if ci.len() > shape.len() {
        return None;
    }
    all_points(&with_cidx(b, ci), &mut |_, _, _, c| {
        Ok(c.iter().zip(shape).all(|(x, n)| 1 <= *x && x <= n))
    })
    .ok()
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

/// An extra conjunct of a filter, with the index names it reads.
type Conj = (Value, HashSet<String>);

/// §9.1's ghost rule for a read of column variable `v` at `ci` on axis
/// `wrt`: the in-extent conjuncts to filter by, none when every point is in
/// extent (the region split then leaves the box whole). On the forcing axis
/// there is no ghost read: a point outside its buffer is refused.
fn ghost_conjuncts(b: &Band, v: &str, ci: &[Expr], shape: &[i64], wrt: Wrt) -> R<Vec<Conj>> {
    check_rank(v, ci, shape)?;
    let ok = in_extent(b, ci, shape);
    if ok == Some(true) {
        return Ok(Vec::new());
    }
    if wrt == Wrt::Forcing {
        return Err(BandError(format!(
            "a point of a band reads forcing buffer `{v}` outside its extent{}; a product \
             document would drop it (esm-jacobian-spec.md §9.1)",
            if ok.is_none() {
                " or at an index not closed at build time"
            } else {
                ""
            }
        )));
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
/// fails), and §9.2 has its column indices closed: whether that point reads
/// `v` inside its extent. A ghost is dropped, except on the forcing axis.
fn point_in_extent(b: &Band, v: &str, ci: &[Expr], shape: &[i64], wrt: Wrt) -> R<bool> {
    check_rank(v, ci, shape)?;
    let Some(ok) = in_extent(b, ci, shape) else {
        return Err(BandError(format!(
            "a band with no index names reads `{v}` at an index not closed at build time \
             (esm-jacobian-spec.md §9.2)"
        )));
    };
    if !ok && wrt == Wrt::Forcing {
        ghost_conjuncts(b, v, ci, shape, wrt)?; // refuses
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

/// Whether a band with no index names has its one point: its closed `where`
/// guard holds.
fn closed_point(b: &Band) -> R<bool> {
    Ok(!all_points(b, &mut |_, _, _, _| Ok(false))?)
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
        if !(closed_point(b)? && point_in_extent(b, &en.v, &b.cidx, &shape_v, wrt)?) {
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
    let b = &en.band;
    let body = sop("*", vec![ser_expr(coef), sread(lam, rho(b))]);
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
    if !has_names(b) && !point_in_extent(b, &en.v, ci, &shape1, a1)? {
        return Ok(None);
    }
    let g = ghost_conjuncts(b, &en.v, ci, &shape1, a1)?;
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
    if !has_names(b) && !point_in_extent(b, &en.w, c2, &shape2, Wrt::States)? {
        return Ok(None);
    }
    let g = ghost_conjuncts(b, &en.w, c2, &shape2, Wrt::States)?;
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
pub fn jvp_document(
    sv: &SysView,
    entries: &[JacEntry],
    wrt: Wrt,
    opts: &ProductOptions,
) -> R<(Value, JvpIo)> {
    let model = "JvpEval";
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
/// [`jvp_document`].
pub fn vjp_document(
    sv: &SysView,
    entries: &[JacEntry],
    wrt: Wrt,
    opts: &ProductOptions,
) -> R<(Value, VjpIo)> {
    let model = "VjpEval";
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

#[cfg(test)]
mod tests {
    use super::*;

    /// One point (row 1), reading its column at the literal index `ci`.
    fn band(ci: &[i64]) -> Band {
        Band {
            rows: vec![(1, 1)],
            ridx: vec![RIdx::Fixed(1)],
            cidx: ci.iter().map(|n| Expr::Integer(*n)).collect(),
            coef: Expr::Number(1.0),
            contracted: Vec::new(),
            guard: None,
        }
    }

    #[test]
    fn a_read_with_more_indices_than_dimensions_is_refused() {
        let b = band(&[1, 1]);
        // one index past the shape's dimensions has no build-time answer, as
        // in the reference (an out-of-bounds `shape[d]`), not `Some(true)`
        assert_eq!(in_extent(&b, &b.cidx, &[4]), None);
        assert_eq!(in_extent(&b, &b.cidx, &[]), None);
        assert_eq!(in_extent(&b, &b.cidx, &[4, 4]), Some(true));
        assert_eq!(in_extent(&b, &b.cidx[..1], &[4, 4]), Some(true));
        for wrt in [Wrt::States, Wrt::Parameters, Wrt::Forcing] {
            assert!(ghost_conjuncts(&b, "c", &b.cidx, &[4], wrt).is_err());
            assert!(point_in_extent(&b, "c", &b.cidx, &[4], wrt).is_err());
        }
    }
}
