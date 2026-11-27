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
use crate::companions::{faq_over, index_of, obs_extents, row_names_of};
use crate::helpers::{op, skey};
use crate::inline::inline_observed_aligned;
use crate::jacobian::JacEntry;
use crate::operator_refs::{compose, ref_bands, with_layer};
use crate::system::{moving_observed, subscript_observed, SysView};

type R<T> = Result<T, BandError>;

/// One pass's named-step layout (steps.jl `_Steps`): `targets` the axis's
/// columns; `cols` the observed that move with them and are columns too
/// (those a column index does not need inlined); `inline` the pass's
/// inlining map (the observed a column index reads); `ctx` the band context
/// with every column's extents; `composed[o]` the bands of observed `o` onto
/// the targets once composed.
pub(crate) struct Steps {
    targets: HashSet<String>,
    cols: HashSet<String>,
    pub(crate) inline: HashMap<String, Expr>,
    obs: HashMap<String, Expr>,
    ctx: Ctx,
    scalars: HashSet<String>,
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
) -> R<Steps> {
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
    let rhs = with_layer(&st.ctx.refs, || {
        inline_observed_aligned(
            rhs,
            inl,
            &mut obs_names,
            &st.ctx.shapes,
            &st.obs,
            &st.scalars,
            &st.ctx.axes,
        )
    })
    .map_err(|e| BandError(e.0))?;
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
            bycol.entry(w).or_default().extend(got);
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
    inl.insert(o.to_string(), def);
    let out = steps_row(st, o, &rhs, &inl, &shape);
    st.busy.remove(o);
    let out = out?;
    st.composed.insert(o.to_string(), out.clone());
    Ok(out)
}

/// Bands of one column with the same structure (rows, row names, column
/// indices, contracted dimensions and `where`), reached by different paths
/// through the observed graph, combined into the first of them: their
/// coefficients summed (an n-ary `+`, in order), or, structure-only, kept
/// once (steps.jl `_steps_combine`).
fn steps_combine(bs: Vec<Band>, co: bool) -> Vec<Band> {
    type Key = (
        Vec<(i64, i64)>,
        Vec<RIdx>,
        Vec<String>,
        Vec<(String, i64, i64)>,
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
