//! Named steps on every axis (spec §4.6; the reference's steps.jl): the
//! bands of a row are composed through the observed graph rather than read
//! off the row with every observed inlined. Each observed that moves with the
//! axis is a column of the band calculus; its own definition is
//! differentiated once, and its bands onto the axis's targets are composed
//! once and memoized, so the cost grows with the number of observed, not
//! with the number of paths through them. Nothing is inlined and then
//! deduplicated, and there is no threshold.

use std::collections::{BTreeMap, HashMap, HashSet};

use earthsci_ast::{free_variables, Expr};

use crate::bands::{merge_bands, normalize_band, placeholder, Band, BandError, Ctx};
use crate::clip::RIdx;
use crate::companions::{faq_over, index_of, obs_extents, region_array, row_names_of};
use crate::emit::serialize_band;
use crate::helpers::stable_json;
use crate::helpers::{op, skey};
use crate::jacobian::JacEntry;
use crate::jacobian::Wrt;
use crate::operator_refs::{compose, ref_bands};
use crate::reductions::{conj_list, conjuncts, fresh};
use crate::system::{moving_observed, subscript_observed, SysView};
use earthsci_ast::extension::types::RangeSpec;
use earthsci_ast::ExpressionNode;
use serde_json::Value;

type R<T> = Result<T, BandError>;

/// One pass's named-step layout (steps.jl `_Steps`): `targets` the axis's
/// columns; `cols` the observed that move with them and are columns too
/// (those a column index does not need inlined); `inline` the pass's
/// inlining map (the observed a column index reads); `ctx` the band context
/// with every column's extents; `composed[o]` the bands of observed `o` onto
/// the targets once composed, each coefficient (full mode) a read of the
/// band's named intermediate; `letter` the axis letter of the intermediates'
/// names; `inters` every intermediate named so far, name → (observed,
/// definition, contracted dimensions added); `taken` the names an
/// intermediate's name must be clear of.
pub(crate) struct Steps {
    letter: String,
    taken: HashSet<String>,
    pub(crate) inters: HashMap<String, (String, Expr, usize)>,
    pub(crate) targets: HashSet<String>,
    pub(crate) cols: HashSet<String>,
    pub(crate) inline: HashMap<String, Expr>,
    obs: HashMap<String, Expr>,
    pub(crate) ctx: Ctx,
    pub(crate) scalars: HashSet<String>,
    co: bool,
    composed: HashMap<String, Vec<JacEntry>>,
    busy: HashSet<String>,
}

/// The layout of a pass over `targets`. `obs` is every observed definition
/// with the recurrences left out; `scalars` the declared scalars that are
/// not observed.
pub(crate) fn steps_setup(
    sv: &SysView,
    ctx: &Ctx,
    obs: HashMap<String, Expr>,
    targets: &HashSet<String>,
    scalars: &HashSet<String>,
    co: bool,
    letter: &str,
) -> R<Steps> {
    let mut taken: HashSet<String> = sv.variables.keys().cloned().collect();
    taken.extend(sv.index_sets.keys().cloned());
    let moving = moving_observed(&obs, targets);
    let mut sub = HashSet::new();
    let mut scan: Vec<String> = moving.iter().cloned().collect();
    scan.sort();
    subscript_observed(&mut sub, &obs, &sv.equations, &scan);
    let cols: HashSet<String> = moving.difference(&sub).cloned().collect();
    let mut ctx2 = ctx.clone();
    let mut memo = HashMap::new();
    let mut sorted: Vec<&String> = cols.iter().collect();
    sorted.sort();
    for o in sorted {
        let ext = obs_extents(o, sv, &obs, ctx, &mut memo, &mut HashSet::new())?;
        if !ext.is_empty() {
            ctx2.shapes.insert(o.clone(), ext);
        }
    }
    let inline = sub.iter().map(|n| (n.clone(), obs[n].clone())).collect();
    Ok(Steps {
        letter: letter.to_string(),
        taken,
        inters: HashMap::new(),
        targets: targets.clone(),
        cols,
        inline,
        obs,
        ctx: ctx2,
        scalars: scalars.clone(),
        co,
        composed: HashMap::new(),
        busy: HashSet::new(),
    })
}

/// The bands of one row (`rhs`, rows `u` of shape `shape_u`) onto the
/// targets: its direct bands onto each target, and each direct band onto an
/// observed column composed with that observed's composed bands. Columns in
/// code-point order; within a column the direct bands, then the composites
/// in the order of the observed they pass through; merged and normalized.
pub(crate) fn steps_row(
    st: &mut Steps,
    u: &str,
    rhs: &Expr,
    inl: &HashMap<String, Expr>,
    shape_u: &[i64],
) -> R<Vec<JacEntry>> {
    let mut obs_names = None;
    // inlined, and the operands of a whole-array term aligned with `u` by
    // index-set name (`inline_row`), as on every axis
    let rhs =
        crate::jacobian::inline_row(rhs, u, inl, &mut obs_names, &st.ctx, &st.obs, &st.scalars)?;
    let mut hits: Vec<String> = free_variables(&rhs)
        .into_iter()
        .filter(|v| st.targets.contains(v) || st.cols.contains(v))
        .collect();
    hits.sort();
    let mut bycol: BTreeMap<String, Vec<Band>> = BTreeMap::new();
    for v in &hits {
        let mut raw = ref_bands(&rhs, v, &st.ctx, shape_u, st.co)?;
        if !st.co {
            // structure-only bands merge on structure alone (jacobian.rs
            // `row_bands`): one-hot lifting has rewritten the placeholder
            for b in raw.iter_mut() {
                b.coef = placeholder();
            }
        }
        let bs: Vec<Band> = merge_bands(raw).iter().map(normalize_band).collect();
        if st.targets.contains(v) {
            bycol.entry(v.clone()).or_default().extend(bs);
            continue;
        }
        // a kept observed is read with one index per dimension of its
        // shape: another arity selects no cell, refused as inlining refuses
        // it (steps.jl `_check_cell_read`)
        if let Some(sh) = st.ctx.shapes.get(v) {
            if let Some(b) = bs.iter().find(|b| b.cidx.len() != sh.len()) {
                return Err(BandError(format!(
                    "observed `{v}` of rank {} is read at a cell with {} indices",
                    sh.len(),
                    b.cidx.len()
                )));
            }
        }
        let inner = steps_composed(st, v)?;
        let mut ws: Vec<String> = inner.iter().map(|e| e.v.clone()).collect();
        ws.sort();
        ws.dedup();
        for w in ws {
            let ib: Vec<Band> = inner
                .iter()
                .filter(|e| e.v == w)
                .map(|e| e.band.clone())
                .collect();
            let got = compose(&bs, &ib, st.co)?;
            bycol
                .entry(w)
                .or_default()
                .extend(got.iter().map(|x| steps_sum_free(x, st.co)));
        }
    }
    let mut out = Vec::new();
    for (w, bs) in bycol {
        for b in merge_bands(steps_combine(bs, st.co)) {
            out.push(JacEntry {
                u: u.to_string(),
                v: w.clone(),
                band: normalize_band(&b),
            });
        }
    }
    Ok(out)
}

/// The composed bands of observed `o` onto the targets, computed once: the
/// bands of `o`'s own row (its definition over its shape, as a companion
/// row is written, §4.2) by [`steps_row`].
pub(crate) fn steps_composed(st: &mut Steps, o: &str) -> R<Vec<JacEntry>> {
    if let Some(hit) = st.composed.get(o) {
        return Ok(hit.clone());
    }
    if st.busy.contains(o) {
        return Err(BandError(format!(
            "cyclic observed definition through `{o}`"
        )));
    }
    st.busy.insert(o.to_string());
    let shape = st.ctx.shapes.get(o).cloned().unwrap_or_default();
    let def = st.obs[o].clone();
    let rhs = if shape.is_empty() {
        Expr::Variable(o.to_string())
    } else {
        let names = row_names_of(&def, shape.len());
        faq_over(
            &names,
            &shape,
            index_of(o, names.iter().map(|n| Expr::Variable(n.clone())).collect()),
        )
    };
    let mut inl = st.inline.clone();
    let same_names = |a: &Expr, b: &Expr| match (a, b) {
        (Expr::Operator(x), Expr::Operator(y)) => x.output_idx == y.output_idx,
        _ => false,
    };
    let rhs = match &def {
        Expr::Operator(dn) if dn.op == "faq" && dn.join.is_some() && same_names(&rhs, &def) => {
            // a joined definition over its own row names is its row (a joined
            // node is never read through beta-reduction, §3.5)
            def.clone()
        }
        _ => {
            inl.insert(o.to_string(), def);
            rhs
        }
    };
    let out = steps_row(st, o, &rhs, &inl, &shape);
    st.busy.remove(o);
    let mut out = out?;
    if st.co {
        out = steps_name(st, o, &shape, out);
    }
    st.composed.insert(o.to_string(), out.clone());
    Ok(out)
}

// ── named intermediates (full mode) ─────────────────────────────────────────

/// The composed bands of observed `o` in their canonical order (the
/// code-point order of each band's structure-only entry written as stable
/// JSON), each coefficient that is not a literal written once as a named
/// intermediate `_d<letter>_<o>_<k>` (k its 1-based position) and replaced
/// by a read of it at the band's row and contracted names. One intermediate
/// per BAND, never one per (column, column index): two bands of one column
/// index may overlap with different `where`, and a summed definition read by
/// each would count the overlap twice (steps.jl `_steps_name!`).
fn steps_name(st: &mut Steps, o: &str, shape: &[i64], bands: Vec<JacEntry>) -> Vec<JacEntry> {
    let key = |en: &JacEntry| {
        let mut v = serialize_band(en);
        if let Value::Object(m) = &mut v {
            m.remove("coef");
        }
        stable_json(&v)
    };
    let mut keyed: Vec<(String, JacEntry)> = bands.into_iter().map(|en| (key(&en), en)).collect();
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out = Vec::new();
    for (k, (_, en)) in keyed.into_iter().enumerate() {
        if matches!(en.band.coef, Expr::Integer(_) | Expr::Number(_)) {
            out.push(en);
            continue;
        }
        let name = inter_name(st, o, k + 1);
        let dims = inter_dims(&en.band, shape);
        let def = inter_definition(&en.band, &dims);
        st.inters.insert(name.clone(), (o.to_string(), def, 1));
        let mut b = en.band.clone();
        b.coef = inter_read(&name, &dims);
        out.push(JacEntry {
            u: en.u,
            v: en.v,
            band: b,
        });
    }
    out
}

/// `_d<letter>_<o>_<k>`; for a scoped `P.v`, `P._d<letter>_<v>_<k>`; `_`
/// appended until clear of every variable and index-set name of the source.
fn inter_name(st: &mut Steps, o: &str, k: usize) -> String {
    let base = match o.rfind('.') {
        None => format!("_d{}_{o}_{k}", st.letter),
        Some(i) => format!("{}._d{}_{}_{k}", &o[..i], st.letter, &o[i + 1..]),
    };
    fresh(&mut st.taken, &base, "_")
}

/// One dimension of an intermediate: its name, box and whole extent.
type Dim = (String, (i64, i64), i64);

/// The dimensions of a band's intermediate: the band's row names (in row
/// order) and then contracted names that its coefficient or `where` reads,
/// each with its box and its whole extent. Only those: an intermediate
/// depends on what its coefficient depends on (steps.jl `_inter_dims`).
fn inter_dims(b: &Band, shape: &[i64]) -> Vec<Dim> {
    let mut fv: HashSet<String> = free_variables(&b.coef).into_iter().collect();
    if let Some(g) = &b.guard {
        fv.extend(free_variables(g));
    }
    let mut dims = Vec::new();
    for (d, r) in b.ridx.iter().enumerate() {
        if let RIdx::Name(n) = r {
            if fv.contains(n) {
                dims.push((n.clone(), b.rows[d], shape[d]));
            }
        }
    }
    for (n, lo, hi) in &b.contracted {
        if fv.contains(n) {
            dims.push((n.clone(), (*lo, *hi), *hi));
        }
    }
    dims
}

/// The read of intermediate `name` at its dimensions' names.
fn inter_read(name: &str, dims: &[Dim]) -> Expr {
    if dims.is_empty() {
        Expr::Variable(name.to_string())
    } else {
        index_of(
            name,
            dims.iter()
                .map(|(n, _, _)| Expr::Variable(n.clone()))
                .collect(),
        )
    }
}

/// The definition of a band's intermediate over its dimensions: one `faq`
/// when the box is the whole extent, else the `makearray` of zero over the
/// whole extent and the box over it, read at each cell (steps.jl
/// `_inter_definition`).
fn inter_definition(b: &Band, dims: &[Dim]) -> Expr {
    let body = match &b.guard {
        None => b.coef.clone(),
        Some(g) => op("ifelse", vec![g.clone(), b.coef.clone(), Expr::Integer(0)]),
    };
    if dims.is_empty() {
        return body;
    }
    let names: Vec<String> = dims.iter().map(|(n, _, _)| n.clone()).collect();
    let bx: Vec<(i64, i64)> = dims.iter().map(|(_, b, _)| *b).collect();
    let full: Vec<i64> = dims.iter().map(|(_, _, n)| *n).collect();
    let faq = |bx: &[(i64, i64)]| -> Expr {
        let ranges: HashMap<String, RangeSpec> = names
            .iter()
            .zip(bx)
            .map(|(n, (lo, hi))| (n.clone(), RangeSpec::Interval([*lo, *hi])))
            .collect();
        Expr::operator(ExpressionNode {
            op: "faq".into(),
            output_idx: Some(names.clone()),
            expr: Some(Box::new(body.clone())),
            ranges: Some(ranges),
            ..Default::default()
        })
    };
    if bx
        .iter()
        .zip(&full)
        .all(|((lo, hi), n)| *lo == 1 && hi == n)
    {
        return faq(&bx);
    }
    region_array(
        vec![
            full.iter().map(|n| [1, *n]).collect(),
            bx.iter().map(|(lo, hi)| [*lo, *hi]).collect(),
        ],
        vec![Expr::Integer(0), faq(&bx)],
        &full,
    )
}

// ── the intermediates a band list reads ─────────────────────────────────────

/// The axis and observed of intermediate name `n`, or `None` (steps.jl
/// `_inter_parse`).
/// The observed a companion name `_dt_<o>` (`P._dt_<v>` for `P.v`) is of,
/// or `None` for another name (steps.jl `_dt_observed`).
pub(crate) fn dt_observed(n: &str) -> Option<String> {
    let (scope, base) = match n.rfind('.') {
        None => ("", n),
        Some(i) => (&n[..=i], &n[i + 1..]),
    };
    base.strip_prefix("_dt_").map(|o| format!("{scope}{o}"))
}

pub(crate) fn inter_parse(n: &str) -> Option<(Wrt, String)> {
    let (scope, base) = match n.rfind('.') {
        None => ("", n),
        Some(i) => (&n[..=i], &n[i + 1..]),
    };
    let wrt = match base.get(..4)? {
        "_du_" => Wrt::States,
        "_dp_" => Wrt::Parameters,
        "_df_" => Wrt::Forcing,
        _ => return None,
    };
    let rest = base[4..].trim_end_matches('_');
    let (mid, k) = rest.rsplit_once('_')?;
    if k.is_empty() || !k.bytes().all(|c| c.is_ascii_digit()) || mid.is_empty() {
        return None;
    }
    Some((wrt, format!("{scope}{mid}")))
}

/// Every named intermediate the coefficients and `where` of `bands` read,
/// directly or through another intermediate's definition, as `(name,
/// observed, definition, added contracted dimensions)`, in the code-point
/// order of the observed, then of the name: the time companions (§4.2) and
/// the named steps of the other axes (§4.6) (steps.jl `_intermediates`).
pub(crate) fn intermediates(
    sv: &SysView,
    bands: &[&Band],
    forcing: Option<&[String]>,
) -> R<Vec<(String, String, Expr, usize)>> {
    let mut work: Vec<String> = Vec::new();
    for b in bands {
        for x in std::iter::once(&b.coef).chain(b.guard.iter()) {
            for n in free_variables(x) {
                if inter_parse(&n).is_some() {
                    work.push(n.clone());
                }
                // a time pair's companion of a named intermediate (§4.4)
                if let Some(o) = dt_observed(&n) {
                    if inter_parse(&o).is_some() {
                        work.push(o);
                    }
                }
            }
        }
    }
    if work.is_empty() {
        return Ok(
            crate::companions::time_companions(sv, bands, &HashMap::new())?
                .into_iter()
                .map(|(n, o, d)| (n, o, d, 0))
                .collect(),
        );
    }
    let ctx = crate::system::ctx_of(sv)?;
    let obs = crate::system::observed_defs(sv)?;
    let recs = crate::recurrence::recurrences(sv, &obs, &ctx)?;
    let scalars = crate::jacobian::declared_scalars(sv, &obs);
    let mut nr = obs.clone();
    for r in &recs {
        nr.remove(&r.name);
    }
    // A pair's second-axis step of a first-axis intermediate (§4.4) is named
    // after that intermediate (`_dp__du_o_1_1`): the intermediates whose
    // observed is a source observed are resolved first, then those over them,
    // each level's passes taking the levels below as observed (steps.jl).
    fn level(n: &str) -> usize {
        let (_, o) = inter_parse(n).expect("an intermediate name");
        if inter_parse(&o).is_some() {
            1 + level(&o)
        } else {
            0
        }
    }
    fn chain(acc: &mut Vec<String>, n: String) {
        let (_, o) = inter_parse(&n).expect("an intermediate name");
        acc.push(n);
        if inter_parse(&o).is_some() {
            chain(acc, o);
        }
    }
    let mut passes: HashMap<(Wrt, usize), Steps> = HashMap::new();
    let mut defs: BTreeMap<String, (String, Expr, usize)> = BTreeMap::new();
    let mut pending: Vec<String> = Vec::new();
    for n in work {
        chain(&mut pending, n);
    }
    while !pending.is_empty() {
        let lv = pending.iter().map(|n| level(n)).min().unwrap();
        passes.retain(|k, _| k.1 <= lv);
        let mut here: Vec<String> = pending.iter().filter(|n| level(n) == lv).cloned().collect();
        pending.retain(|n| level(n) != lv);
        while let Some(n) = here.pop() {
            if defs.contains_key(&n) {
                continue;
            }
            let (wrt, o) = inter_parse(&n).expect("parsed above");
            let key = (wrt, lv);
            let st = match passes.entry(key) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => {
                    let mut o2 = nr.clone();
                    if lv > 0 {
                        o2.extend(defs.iter().map(|(m, (_, d, _))| (m.clone(), d.clone())));
                    }
                    let tset: HashSet<String> = crate::jacobian::targets(sv, wrt, forcing, &obs)?
                        .into_iter()
                        .collect();
                    let letter = match wrt {
                        Wrt::States => "u",
                        Wrt::Parameters => "p",
                        _ => "f",
                    };
                    e.insert(steps_setup(sv, &ctx, o2, &tset, &scalars, true, letter)?)
                }
            };
            if !st.inters.contains_key(&n) {
                steps_composed(st, &o)?;
            }
            let hit = st.inters.get(&n).cloned().ok_or_else(|| {
                BandError(format!("no named intermediate `{n}` (observed `{o}`)"))
            })?;
            for m in free_variables(&hit.1) {
                if inter_parse(&m).is_none() || defs.contains_key(&m) {
                    continue;
                }
                if level(&m) == lv {
                    here.push(m);
                } else {
                    chain(&mut pending, m);
                }
            }
            defs.insert(n, hit);
        }
    }
    let extra: HashMap<String, Expr> = defs
        .iter()
        .map(|(n, (_, d, _))| (n.clone(), d.clone()))
        .collect();
    let mut out: Vec<(String, String, Expr, usize)> =
        crate::companions::time_companions(sv, bands, &extra)?
            .into_iter()
            .map(|(n, o, d)| (n, o, d, 0))
            .collect();
    for (n, (o, d, c)) in defs {
        out.push((n, o, d, c));
    }
    out.sort_by(|a, b| (&a.1, &a.0).cmp(&(&b.1, &b.0)));
    Ok(out)
}

/// A composite's contracted names that its column indices do not read are
/// summed into its coefficient: a scalar `faq` (reduce `+`) over their
/// ranges whose filter is the `where` conjuncts that read them (in order);
/// the other conjuncts stay the band's `where` (steps.jl `_steps_sum_free`).
pub(crate) fn steps_sum_free(b: &Band, co: bool) -> Band {
    if b.contracted.is_empty() {
        return b.clone();
    }
    // a joined name is never summed away
    let mut read: HashSet<String> = crate::bands::join_names(b).into_iter().collect();
    for c in &b.cidx {
        read.extend(free_variables(c));
    }
    let free: HashSet<String> = b
        .contracted
        .iter()
        .filter(|(n, _, _)| !read.contains(n))
        .map(|(n, _, _)| n.clone())
        .collect();
    if free.is_empty() {
        return b.clone();
    }
    let mut keep = Vec::new();
    let mut filt = Vec::new();
    if let Some(g) = &b.guard {
        for c in conjuncts(g) {
            if free_variables(&c).iter().any(|n| free.contains(n)) {
                filt.push(c);
            } else {
                keep.push(c);
            }
        }
    }
    let con: Vec<(String, i64, i64)> = b
        .contracted
        .iter()
        .filter(|(n, _, _)| !free.contains(n))
        .cloned()
        .collect();
    let coef = if co {
        let ranges: HashMap<String, RangeSpec> = b
            .contracted
            .iter()
            .filter(|(n, _, _)| free.contains(n))
            .map(|(n, lo, hi)| (n.clone(), RangeSpec::Interval([*lo, *hi])))
            .collect();
        Expr::operator(ExpressionNode {
            op: "faq".into(),
            output_idx: Some(Vec::new()),
            reduce: Some("+".into()),
            expr: Some(Box::new(b.coef.clone())),
            ranges: Some(ranges),
            filter: conj_list(filt).map(Box::new),
            ..Default::default()
        })
    } else {
        placeholder()
    };
    normalize_band(&Band {
        rows: b.rows.clone(),
        ridx: b.ridx.clone(),
        cidx: b.cidx.clone(),
        coef,
        contracted: con,
        guard: conj_list(keep),
        join: b.join.clone(),
    })
}

/// Bands of one column with the same structure (rows, row names, column
/// indices, contracted dimensions and `where`), reached by different paths
/// through the observed graph, combined into the first of them: their
/// coefficients summed (an n-ary `+`, in order), or, structure-only, kept
/// once (steps.jl `_steps_combine`).
pub(crate) fn steps_combine(bs: Vec<Band>, co: bool) -> Vec<Band> {
    type Key = (
        Vec<(i64, i64)>,
        Vec<RIdx>,
        Vec<String>,
        Vec<(String, i64, i64)>,
        Option<String>,
        Option<String>,
    );
    let mut order: Vec<Key> = Vec::new();
    let mut groups: BTreeMap<Key, Vec<usize>> = BTreeMap::new();
    for (i, b) in bs.iter().enumerate() {
        let k = (
            b.rows.clone(),
            b.ridx.clone(),
            b.cidx.iter().map(skey).collect(),
            b.contracted.clone(),
            b.guard.as_ref().map(skey),
            b.join.as_ref().map(|j| j.key()),
        );
        if !groups.contains_key(&k) {
            order.push(k.clone());
        }
        groups.entry(k).or_default().push(i);
    }
    let mut out = Vec::new();
    for k in order {
        let idx = &groups[&k];
        let mut b = bs[idx[0]].clone();
        if idx.len() > 1 && co {
            b.coef = op("+", idx.iter().map(|&i| bs[i].coef.clone()).collect());
        }
        out.push(b);
    }
    out
}
