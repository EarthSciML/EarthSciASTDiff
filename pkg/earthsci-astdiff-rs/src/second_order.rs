//! Second-order bands (the reference's second_order.jl; spec §4.4): the band
//! calculus applied once more, to the coefficients of the first-order bands.
//! A Rosenbrock step contains J(uₙ) explicitly, so its discrete adjoint
//! needs ∇(μᵀ J(u) w) once per step; that is a contraction of these entries
//! (the `hvp` product document of the reference, spec §9), never a Hessian
//! tensor.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use earthsci_ast::{free_variables, EsmFile, Expr};
use serde_json::{json, Value};

use crate::bands::{
    is_array, lift_nested, merge_bands, nested_sums, normalize_band, site_cidx, sorted_sites, Band,
    BandError, Ctx,
};
use crate::branches::SimpMemo;
use crate::clip::{clip_regions, RIdx};
use crate::emit::{block_fields, columns_field, offset_class, serialize_band};
use crate::helpers::{is_zero, ser_expr, OccIndex, SiteScan};
use crate::jacobian::{check_closed, expr_json, targets, FirstOrder, JacEntry, Wrt};
use crate::operator_refs::{abstract_refs, compose, ref_value_bands, restore_band, with_layer};
use crate::recurrence::{ExtendedBands, Recurrence};
use crate::reductions::{conj, one_hot};
use crate::rules::dscalar_in;
use crate::system::{ctx_of, observed_defs, sysview, SysView};

type R<T> = Result<T, BandError>;

/// One second-order band: `∂²(equation for u)/∂v ∂w`. `band` is the
/// first-order entry's band with its coefficient replaced by `∂coef/∂(site
/// of w)`: the same `rows`, `ridx` and `contracted` (plus any dimensions the
/// second site adds under a nested sum), and `cidx` holding `v`'s column
/// indices FOLLOWED by `w`'s. The first `ncol` of them are `v`'s
/// ([`JacEntry2::col_idx`] / [`JacEntry2::col2_idx`] split them). Keeping
/// both in one [`Band`] lets clipping, merging and normalization run
/// unchanged.
#[derive(Clone, Debug)]
pub struct JacEntry2 {
    pub u: String,
    pub v: String,
    pub w: String,
    pub band: Band,
    pub ncol: usize,
}

impl JacEntry2 {
    /// The first column's (`v`'s) index expressions.
    pub fn col_idx(&self) -> &[Expr] {
        &self.band.cidx[..self.ncol]
    }

    /// The second column's (`w`'s) index expressions.
    pub fn col2_idx(&self) -> &[Expr] {
        &self.band.cidx[self.ncol..]
    }
}

/// Check a second-order axis pair (spec §4.4): the first derivative's axis
/// is states or time, the second's states, parameters or forcing.
pub fn axis_pair(wrt: (Wrt, Wrt)) -> R<(Wrt, Wrt)> {
    let (a1, a2) = wrt;
    if !matches!(a1, Wrt::States | Wrt::Time) {
        return Err(BandError(format!(
            "the first axis of a pair must be states or time, got {}",
            a1.as_str()
        )));
    }
    if !matches!(a2, Wrt::States | Wrt::Parameters | Wrt::Forcing) {
        return Err(BandError(format!(
            "the second axis of a pair must be states, parameters or forcing, got {}",
            a2.as_str()
        )));
    }
    Ok(wrt)
}

/// Second-order bands for the axis pair `wrt = (a1, a2)`, `a1` states or
/// time and `a2` states, parameters or forcing: every first-order entry of
/// the `a1` axis ([`crate::jacobian_bands`]) with its coefficient
/// differentiated by the same calculus with respect to every site of every
/// `a2` target, broadcast over the entry's rows with region clipping and
/// re-merging. No new rules: the non-smooth policy (spec §3.2) applies
/// again, and the pattern is the global structural union over branches.
///
/// `(r, c, s)` and `(r, s, c)` come out independently, from distinct
/// first-order entries; consumers must not assume either is deduplicated.
/// The forcing set is inferred (see [`jacobian_bands2_with_forcing`]).
pub fn jacobian_bands2(
    file: &EsmFile,
    model_name: Option<&str>,
    wrt: (Wrt, Wrt),
) -> R<Vec<JacEntry2>> {
    jacobian_bands2_with_forcing(file, model_name, wrt, None)
}

/// [`jacobian_bands2`] with an explicit forcing set (`None` infers it). On
/// the parameters and forcing second axes every `col2_idx` must be closed at
/// build time, as on the first-order axes (spec §4.3).
pub fn jacobian_bands2_with_forcing(
    file: &EsmFile,
    model_name: Option<&str>,
    wrt: (Wrt, Wrt),
    forcing: Option<&[String]>,
) -> R<Vec<JacEntry2>> {
    let sv = sysview(file, model_name)?;
    let ctx = ctx_of(&sv)?;
    let obs = observed_defs(&sv)?;
    Ok(bands2_view(&sv, &ctx, &obs, axis_pair(wrt)?, forcing)?.1)
}

/// (first-order entries, second-order entries). The first-order pass inlines
/// the observed that move with EITHER axis: one that reads only parameters
/// (an Arrhenius rate) is opaque on the state axis, and would hide its
/// parameter sites from the second derivative.
pub(crate) fn bands2_view(
    sv: &SysView,
    ctx: &Ctx,
    obs: &HashMap<String, Expr>,
    wrt: (Wrt, Wrt),
    forcing: Option<&[String]>,
) -> R<(Rc<ExtendedBands>, Vec<JacEntry2>)> {
    bands2_memo(sv, ctx, obs, wrt, forcing, &mut FirstOrder::default())
}

/// [`bands2_view`] sharing its first-order pass through `memo`.
pub(crate) fn bands2_memo(
    sv: &SysView,
    ctx: &Ctx,
    obs: &HashMap<String, Expr>,
    wrt: (Wrt, Wrt),
    forcing: Option<&[String]>,
    memo: &mut FirstOrder,
) -> R<(Rc<ExtendedBands>, Vec<JacEntry2>)> {
    let (a1, a2) = wrt;
    let t2 = targets(sv, a2, forcing, obs)?;
    // A time pair takes `t` second, through the companions (companions.rs);
    // through an active recurrence it inlines, as below.
    if a1 == Wrt::Time {
        if let Some(mut tp) =
            crate::companions::time_pair_bands(sv, ctx, obs, a2, forcing, memo, &t2)?
        {
            if a2.has_columns() {
                for en in &tp {
                    check_closed_col2(en)?;
                }
            }
            crate::jacobian::intern_entries(tp.iter_mut().map(|en| &mut en.band));
            let e1 = Rc::new(ExtendedBands {
                recurrences: Vec::new(),
                rows: Vec::new(),
                rec_rows: Vec::new(),
                row_eq: Vec::new(),
                transfer: HashMap::new(),
            });
            return Ok((e1, tp));
        }
    }
    // A pair with no active recurrence keeps both axes' named steps (§4.4).
    if a1 != Wrt::Time {
        if let Some((e1, mut ps)) = pair_steps_bands(sv, ctx, obs, a1, a2, forcing, &t2)? {
            if a2.has_columns() {
                for en in &ps {
                    check_closed_col2(en)?;
                }
            }
            crate::jacobian::intern_entries(ps.iter_mut().map(|en| &mut en.band));
            return Ok((Rc::new(e1), ps));
        }
    }
    let e1 = memo.ext(sv, ctx, obs, a1, Some(a2), forcing, &t2)?;
    // a recurrence is a second column too (spec §4.5): its rows and columns
    // are the extended system's
    let mut tset: HashSet<String> = t2.into_iter().collect();
    tset.extend(e1.recurrences.iter().map(|r| r.name.clone()));
    let mut out = Vec::new();
    // One occurrence index and coefficient memo per second column, shared by
    // every first-order band: the coefficient memo hands repeated first-order
    // coefficients out as one object, so their nodes are indexed and
    // differentiated once.
    let mut per_w: HashMap<String, (RefCell<OccIndex>, SimpMemo)> = HashMap::new();
    for en in e1.rows.iter().chain(&e1.rec_rows) {
        let mut ws: Vec<String> = free_variables(&en.band.coef)
            .into_iter()
            .filter(|x| tset.contains(x))
            .collect();
        ws.sort();
        for w in ws {
            let (occ, sm) = per_w
                .entry(w.clone())
                .or_insert_with(|| (RefCell::new(OccIndex::new(&w)), SimpMemo::default()));
            for b in merge_bands(coef_bands(&en.band, &w, ctx, occ, sm)?) {
                out.push(JacEntry2 {
                    u: en.u.clone(),
                    v: en.v.clone(),
                    w: w.clone(),
                    band: normalize_band(&b),
                    ncol: en.band.cidx.len(),
                });
            }
        }
    }
    if a2.has_columns() {
        for en in out.iter().filter(|en| !e1.is_recurrence(&en.w)) {
            check_closed_col2(en)?;
        }
    }
    // The second derivatives repeat the first-order coefficients' shared
    // subtrees band after band: intern them as the first-order pass is.
    crate::jacobian::intern_entries(out.iter_mut().map(|en| &mut en.band));
    Ok((e1, out))
}

/// A pair `[a1, a2]` (a1 not time) by named steps (second_order.jl
/// `_pair_steps_bands`): the `a1` pass keeps its named steps, its named
/// intermediates are observed of a named-steps pass over the `a2` targets,
/// and each coefficient's bands w.r.t. a column of that pass are composed
/// with the column's composed `a2` bands. `None` through an active
/// recurrence.
fn pair_steps_bands(
    sv: &SysView,
    ctx: &Ctx,
    obs: &HashMap<String, Expr>,
    a1: Wrt,
    a2: Wrt,
    forcing: Option<&[String]>,
    t2: &[String],
) -> R<Option<(ExtendedBands, Vec<JacEntry2>)>> {
    let recs = crate::recurrence::recurrences(sv, obs, ctx)?;
    if !recs.is_empty() {
        let mut s: HashSet<String> = targets(sv, a1, forcing, obs)?.into_iter().collect();
        s.extend(t2.iter().cloned());
        if !crate::recurrence::active(&recs, obs, &s).is_empty() {
            return Ok(None);
        }
    }
    let ext = crate::jacobian::extended_pass(sv, ctx, obs, a1, forcing, &[], true)?;
    let e1 = &ext.rows;
    let eb: Vec<&Band> = e1.iter().map(|en| &en.band).collect();
    let inter: HashMap<String, Expr> = crate::steps::intermediates(sv, &eb, forcing)?
        .into_iter()
        .filter(|x| x.3 == 1)
        .map(|(n, _, d, _)| (n, d))
        .collect();
    let mut nr = obs.clone();
    for r in &recs {
        nr.remove(&r.name);
    }
    nr.extend(inter.iter().map(|(k, v)| (k.clone(), v.clone())));
    let mut rank = obs.clone();
    rank.extend(inter);
    let tset: HashSet<String> = t2.iter().cloned().collect();
    let letter = match a2 {
        Wrt::States => "u",
        Wrt::Parameters => "p",
        _ => "f",
    };
    let scalars = crate::jacobian::declared_scalars(sv, obs);
    let mut st = crate::steps::steps_setup(sv, ctx, nr, &tset, &scalars, true, letter)?;
    let sctx = st.ctx.clone();
    let mut out = Vec::new();
    for en in e1 {
        let b = &en.band;
        let n = b.cidx.len();
        let mut obs_names = None;
        let coef = with_layer(&sctx.refs, || {
            crate::inline::inline_observed_aligned(
                &b.coef,
                &st.inline,
                &mut obs_names,
                &sctx.shapes,
                &rank,
                &st.scalars,
                &sctx.axes,
            )
        })
        .map_err(|e| BandError(e.0))?;
        let mut b0 = b.clone();
        b0.coef = coef;
        let mut xs: Vec<String> = free_variables(&b0.coef)
            .into_iter()
            .filter(|v| st.targets.contains(v) || st.cols.contains(v))
            .collect();
        xs.sort();
        let mut bycol: HashMap<String, Vec<Band>> = HashMap::new();
        for x in &xs {
            let occ = RefCell::new(OccIndex::new(x));
            let mut sm = SimpMemo::default();
            let bs = coef_bands(&b0, x, &sctx, &occ, &mut sm)?;
            if st.targets.contains(x) {
                bycol.entry(x.clone()).or_default().extend(bs);
                continue;
            }
            let inner = crate::steps::steps_composed(&mut st, x)?;
            let outer: Vec<Band> = bs
                .into_iter()
                .map(|mut y| {
                    y.cidx = y.cidx[n..].to_vec();
                    y
                })
                .collect();
            let mut ws: Vec<String> = inner.iter().map(|e| e.v.clone()).collect();
            ws.sort();
            ws.dedup();
            for w in ws {
                let ib: Vec<Band> = inner
                    .iter()
                    .filter(|e| e.v == w)
                    .map(|e| e.band.clone())
                    .collect();
                for mut z in compose(&outer, &ib, true)? {
                    z.cidx = [b.cidx.clone(), z.cidx].concat();
                    bycol
                        .entry(w.clone())
                        .or_default()
                        .push(crate::steps::steps_sum_free(&z, true));
                }
            }
        }
        let mut ws: Vec<&String> = bycol.keys().collect();
        ws.sort();
        for w in ws {
            for y in merge_bands(crate::steps::steps_combine(bycol[w].clone(), true)) {
                out.push(JacEntry2 {
                    u: en.u.clone(),
                    v: en.v.clone(),
                    w: w.clone(),
                    band: normalize_band(&y),
                    ncol: n,
                });
            }
        }
    }
    Ok(Some((ext, out)))
}

/// One band per site of `w` in a first-order coefficient, broadcast over the
/// entry's rows: the scalar band step with the entry's contracted names
/// bound too. `occ` and `sm` are the caller's, shared across the bands of
/// one second column `w`. A coefficient may hold operator references
/// (operator_refs.rs): scalar ones are differentiated by the derivative
/// table's reference row under the view's layer; an array one is read as an
/// intermediate, its bands w.r.t. `w` composed with the coefficient's w.r.t.
/// it (as `ref_bands`; second_order.jl `_coef_bands`).
pub(crate) fn coef_bands(
    b: &Band,
    w: &str,
    ctx: &Ctx,
    occ: &RefCell<OccIndex>,
    sm: &mut SimpMemo,
) -> R<Vec<Band>> {
    let Some(l) = ctx.refs.clone() else {
        return coef_site_bands(b, clip_regions(&b.coef, &b.ridx, &b.rows), w, ctx, occ, sm);
    };
    with_layer(&Some(l.clone()), || {
        let mut out = Vec::new();
        for (rows2, body) in clip_regions(&b.coef, &b.ridx, &b.rows) {
            let (body2, ps) = abstract_refs(&l, &body, ctx, Some(w))?;
            if ps.is_empty() {
                out.extend(coef_site_bands(b, vec![(rows2, body)], w, ctx, occ, sm)?);
                continue;
            }
            let mut ctx2 = ctx.clone();
            for p in &ps {
                p.declare(&mut ctx2);
            }
            let fv = free_variables(&body2);
            let mut got = Vec::new();
            if fv.contains(w) {
                let piece = vec![(rows2.clone(), body2.clone())];
                got.extend(coef_site_bands(b, piece, w, &ctx2, occ, sm)?);
            }
            for p in &ps {
                if !fv.contains(&p.name) || !free_variables(&p.r).contains(w) {
                    continue;
                }
                let inner = ref_value_bands(&l, &p.r, w, ctx, true)?;
                if inner.is_empty() {
                    continue;
                }
                let inner: Vec<Band> = inner
                    .into_iter()
                    .map(|x| crate::operator_refs::one_hot_band(p.with_scope(x), w, ctx))
                    .collect::<R<_>>()?;
                // the bands of the coefficient w.r.t. the intermediate, their
                // columns its cells only
                let pocc = RefCell::new(OccIndex::new(&p.name));
                let piece = vec![(rows2.clone(), body2.clone())];
                let outer: Vec<Band> = coef_site_bands(b, piece, &p.name, &ctx2, &pocc, sm)?
                    .into_iter()
                    .map(|mut x| {
                        x.cidx = x.cidx[b.cidx.len()..].to_vec();
                        x
                    })
                    .collect();
                for mut x in compose(&outer, &inner, true)? {
                    x.cidx = [b.cidx.clone(), x.cidx].concat();
                    got.push(x);
                }
            }
            for x in got {
                out.push(restore_band(x, &ps)?);
            }
        }
        Ok(out)
    })
}

/// The bands of the sites of `w` in the clipped regions `pieces` (row box,
/// coefficient body) of b's coefficient.
fn coef_site_bands(
    b: &Band,
    pieces: Vec<(Vec<(i64, i64)>, Expr)>,
    w: &str,
    ctx: &Ctx,
    occ: &RefCell<OccIndex>,
    sm: &mut SimpMemo,
) -> R<Vec<Band>> {
    let mut out = Vec::new();
    let outer: Vec<String> = b
        .ridx
        .iter()
        .filter_map(|r| match r {
            RIdx::Name(n) => Some(n.clone()),
            RIdx::Fixed(_) => None,
        })
        .chain(b.contracted.iter().map(|(n, _, _)| n.clone()))
        .collect();
    for (rows2, body) in pieces {
        let nested = nested_sums(&body, w, &outer);
        let mut taken = None;
        for s in sorted_sites(&body, w) {
            if matches!(&s.expr, Expr::Variable(_)) && is_array(ctx, w) {
                return Err(BandError(format!(
                    "whole-array reference to `{w}` in a band coefficient"
                )));
            }
            let (c2, d, extra, g) = lift_nested(
                &nested,
                &s,
                &body,
                site_cidx(&s),
                dscalar_in(&body, &s, &SiteScan::new(occ, &s))?,
                ctx,
                &outer,
                &mut taken,
            )?;
            let coef = sm.simp(&d);
            if is_zero(&coef) {
                continue;
            }
            let (c2, coef, contracted) = one_hot(
                c2,
                coef,
                [b.contracted.clone(), extra].concat(),
                &b.ridx,
                w,
                ctx,
            )?;
            out.push(Band {
                rows: rows2.clone(),
                ridx: b.ridx.clone(),
                cidx: [b.cidx.clone(), c2].concat(),
                coef,
                contracted,
                guard: conj(b.guard.clone(), g),
                join: b.join.clone(),
            });
        }
    }
    Ok(out)
}

/// The first-order closed-column check for the second column of a pair: on
/// the parameters and forcing second axes a `col2_idx` may read only the
/// entry's row and contracted index names.
fn check_closed_col2(en: &JacEntry2) -> R<()> {
    check_closed(&en.band, en.col2_idx(), |c| {
        format!(
            "second column index {} of `{}` in ∂²({})/∂{}∂{}",
            expr_json(c),
            en.w,
            en.u,
            en.v,
            en.w
        )
    })
}

/// `"block_diagonal"` when the first-order state block is and every second
/// column sits on the row's own cell, so the contraction is lane-local;
/// `None` (field omitted) otherwise, and always on a time pair.
/// `first` is the first-order block's class (asked only on a states pair).
pub(crate) fn structure2(
    first: impl FnOnce() -> &'static str,
    entries: &[JacEntry2],
    ctx: &Ctx,
    a1: Wrt,
) -> Option<&'static str> {
    if a1 != Wrt::States || first() != "block_diagonal" {
        return None;
    }
    for en in entries {
        if ctx.shapes.contains_key(&en.u) != ctx.shapes.contains_key(&en.w) {
            return None;
        }
        let c2 = en.col2_idx();
        if c2.is_empty() {
            continue;
        }
        if c2.len() != en.band.ridx.len()
            || !c2
                .iter()
                .zip(&en.band.ridx)
                .all(|(c, r)| offset_class(c, r) == "diag")
        {
            return None;
        }
    }
    Some("block_diagonal")
}

// ── emission ─────────────────────────────────────────────────────────────────

/// One second-order entry of a pair block, as JSON (spec §4.4): the
/// first-order keys plus `col2` and `col2_idx`.
pub fn serialize_band2(en: &JacEntry2) -> Value {
    let first = JacEntry {
        u: en.u.clone(),
        v: en.v.clone(),
        band: Band {
            cidx: en.col_idx().to_vec(),
            ..en.band.clone()
        },
    };
    let mut d = serialize_band(&first);
    let m = d.as_object_mut().expect("a band serializes to an object");
    m.insert("col2".into(), json!(en.w));
    m.insert(
        "col2_idx".into(),
        Value::Array(en.col2_idx().iter().map(ser_expr).collect()),
    );
    d
}

/// One second-order entry back from JSON (the inverse of
/// [`serialize_band2`]).
pub fn parse_band2(d: &Value) -> R<JacEntry2> {
    let en = crate::emit::parse_band(d)?;
    let w = d
        .get("col2")
        .and_then(Value::as_str)
        .ok_or_else(|| BandError("band without `col2`".into()))?
        .to_string();
    let c2: Vec<Expr> = d
        .get("col2_idx")
        .and_then(Value::as_array)
        .ok_or_else(|| BandError("band without `col2_idx`".into()))?
        .iter()
        .map(|c| {
            serde_json::from_value(c.clone()).map_err(|e| BandError(format!("bad expression: {e}")))
        })
        .collect::<R<_>>()?;
    let ncol = en.band.cidx.len();
    let mut band = en.band;
    band.cidx.extend(c2);
    Ok(JacEntry2 {
        u: en.u,
        v: en.v,
        w,
        band,
        ncol,
    })
}

/// The `jacobians` block of one axis pair (spec §4.4): `"wrt": [a1, a2]`,
/// the entries (template-CSE'd when `cse` is set), the second axis's
/// `columns` on a parameters or forcing pair, and `structure` only when the
/// contraction is lane-local. Never a factorization plan.
pub fn jacobian_block2(
    file: &EsmFile,
    model_name: Option<&str>,
    wrt: (Wrt, Wrt),
    forcing: Option<&[String]>,
    cse: bool,
    cse_min_nodes: usize,
) -> R<Value> {
    let sv = sysview(file, model_name)?;
    let ctx = ctx_of(&sv)?;
    let obs = observed_defs(&sv)?;
    let mut memo = FirstOrder::default();
    pair_block(&sv, &ctx, &obs, wrt, forcing, cse, cse_min_nodes, &mut memo)
}

/// [`jacobian_block2`] over a view, sharing its first-order pass through
/// `memo` with the other blocks of one emission.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pair_block(
    sv: &SysView,
    ctx: &Ctx,
    obs: &HashMap<String, Expr>,
    wrt: (Wrt, Wrt),
    forcing: Option<&[String]>,
    cse: bool,
    cse_min_nodes: usize,
    memo: &mut FirstOrder,
) -> R<Value> {
    let (a1, a2) = axis_pair(wrt)?;
    let (e1, entries) = bands2_memo(sv, ctx, obs, (a1, a2), forcing, memo)?;
    let columns = if a2.has_columns() {
        Some(columns_field(sv, a2, forcing)?)
    } else {
        None
    };
    // through a recurrence the contraction is not lane-local (§4.5)
    let structure = if e1.recurrences.is_empty() {
        structure2(
            || memo.structure(a1, Some(a2), &e1.rows, ctx),
            &entries,
            ctx,
            a1,
        )
    } else {
        None
    };
    let mut block = block_fields(
        json!([a1.as_str(), a2.as_str()]),
        &entries,
        serialize_band2,
        columns,
        structure,
        cse,
        cse_min_nodes,
    );
    if !e1.recurrences.is_empty() {
        block.insert("recurrences".into(), recurrences_field(&e1.recurrences));
    }
    let eb: Vec<&Band> = entries.iter().map(|en| &en.band).collect();
    let comps = crate::steps::intermediates(sv, &eb, forcing)?;
    if !comps.is_empty() {
        block.insert(
            "intermediates".into(),
            crate::companions::companions_field(&comps),
        );
    }
    Ok(Value::Object(block))
}

/// A pair block's `recurrences` field (spec §4.5): each active recurrence's
/// name and its 1-based recurrence dimension, in recurrence order. Entries
/// naming one as `row`, `col` or `col2` are terms of the chain rule
/// through it, which only the §9.7 contraction completes.
pub(crate) fn recurrences_field(recs: &[Recurrence]) -> Value {
    Value::Array(
        recs.iter()
            .map(|r| json!({"name": r.name, "dim": r.dim + 1}))
            .collect(),
    )
}
