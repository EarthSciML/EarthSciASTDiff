//! Second-order bands (the reference's second_order.jl; spec §4.4): the band
//! calculus applied once more, to the coefficients of the first-order bands.
//! A Rosenbrock step contains J(uₙ) explicitly, so its discrete adjoint
//! needs ∇(μᵀ J(u) w) once per step; that is a contraction of these entries
//! ([`crate::numeric::HvpEvaluator`]), never a Hessian tensor.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use earthsci_ast::{free_variables, EsmFile, Expr};
use serde_json::{json, Value};

use crate::bands::{
    is_array, lift_nested, merge_bands, nested_sums, normalize_band, site_cidx, sorted_sites, Band,
    BandError, Ctx,
};
use crate::branches::simp;
use crate::clip::{clip_regions, RIdx};
use crate::emit::{block_fields, columns_field, offset_class, serialize_band};
use crate::helpers::{is_zero, ser_expr};
use crate::jacobian::{check_closed, expr_json, targets, FirstOrder, JacEntry, Wrt};
use crate::reductions::{conj, one_hot};
use crate::rules::dscalar;
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
) -> R<(Rc<Vec<JacEntry>>, Vec<JacEntry2>)> {
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
) -> R<(Rc<Vec<JacEntry>>, Vec<JacEntry2>)> {
    let (a1, a2) = wrt;
    let t2 = targets(sv, a2, forcing, obs)?;
    let e1 = memo.bands(sv, ctx, obs, a1, Some(a2), forcing, &t2)?;
    let tset: HashSet<String> = t2.into_iter().collect();
    let mut out = Vec::new();
    for en in e1.iter() {
        let mut ws: Vec<String> = free_variables(&en.band.coef)
            .into_iter()
            .filter(|x| tset.contains(x))
            .collect();
        ws.sort();
        for w in ws {
            for b in merge_bands(coef_bands(&en.band, &w, ctx)?) {
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
        for en in &out {
            check_closed_col2(en)?;
        }
    }
    Ok((e1, out))
}

/// One band per site of `w` in a first-order coefficient, broadcast over the
/// entry's rows: the scalar band step with the entry's contracted names
/// bound too.
fn coef_bands(b: &Band, w: &str, ctx: &Ctx) -> R<Vec<Band>> {
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
    for (rows2, body) in clip_regions(&b.coef, &b.ridx, &b.rows) {
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
                dscalar(&body, &s)?,
                ctx,
                &outer,
                &mut taken,
            )?;
            let coef = simp(&d);
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
    let structure = structure2(|| memo.structure(a1, Some(a2), &e1, ctx), &entries, ctx, a1);
    Ok(Value::Object(block_fields(
        json!([a1.as_str(), a2.as_str()]),
        &entries,
        serialize_band2,
        columns,
        structure,
        cse,
        cse_min_nodes,
    )))
}
