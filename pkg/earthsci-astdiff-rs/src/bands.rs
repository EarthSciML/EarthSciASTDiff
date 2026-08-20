//! The array-level band calculus (the reference's bands.jl).

use std::collections::{BTreeMap, HashMap, HashSet};

use earthsci_ast::expression::free_variables;
use earthsci_ast::substitute::substitute;
use earthsci_ast::types::{Expr, ExpressionNode, RangeSpec};

use crate::branches::{canon_lits, simp};
use crate::clip::{clip_regions, RIdx};
use crate::helpers::{add, is_zero, map_expr_children, mul, op, skey, Site};
use crate::rules::{dscalar, DerivativeRuleError};
use crate::simplify_port::simplify_jl;

#[derive(Clone, Debug)]
pub struct Band {
    pub rows: Vec<(i64, i64)>,
    pub ridx: Vec<RIdx>,
    pub cidx: Vec<Expr>,
    pub coef: Expr,
    pub contracted: Vec<(String, i64, i64)>,
}

#[derive(Clone, Debug, Default)]
pub struct Ctx {
    pub index_sets: HashMap<String, i64>,
    pub shapes: HashMap<String, Vec<i64>>,
    pub params: HashSet<String>,
}

#[derive(Debug)]
pub struct BandError(pub String);

impl std::fmt::Display for BandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BandError: {}", self.0)
    }
}
impl std::error::Error for BandError {}

impl From<DerivativeRuleError> for BandError {
    fn from(e: DerivativeRuleError) -> Self {
        BandError(e.0)
    }
}

type R<T> = Result<T, BandError>;

pub fn range_of(r: &RangeSpec, ctx: &Ctx) -> R<(i64, i64)> {
    match r {
        RangeSpec::Interval([lo, hi]) => Ok((*lo, *hi)),
        RangeSpec::Strided([lo, _, hi]) => Ok((*lo, *hi)),
        RangeSpec::IndexSetRef { from, .. } => ctx
            .index_sets
            .get(from)
            .map(|n| (1, *n))
            .ok_or_else(|| BandError(format!("unknown index set `{from}`"))),
        #[allow(unreachable_patterns)]
        _ => Err(BandError("unsupported range spec".into())),
    }
}

pub fn is_array(ctx: &Ctx, name: &str) -> bool {
    ctx.shapes.contains_key(name)
}

fn sites_of(e: &Expr, v: &str, acc: &mut BTreeMap<String, Site>) {
    match e {
        Expr::Variable(n) if n == v => {
            let s = Site::of(e);
            acc.insert(s.key.clone(), s);
        }
        Expr::Operator(n)
            if n.op == "index"
                && matches!(n.args.first(), Some(Expr::Variable(t)) if t == v) =>
        {
            let s = Site::of(e);
            acc.insert(s.key.clone(), s);
        }
        Expr::Operator(n) => {
            n.map_children(&mut |c: &Expr| {
                sites_of(c, v, acc);
                c.clone()
            });
        }
        _ => {}
    }
}

pub fn sorted_sites(e: &Expr, v: &str) -> Vec<Site> {
    let mut acc = BTreeMap::new();
    sites_of(e, v, &mut acc);
    acc.into_values().collect()
}

pub fn is_array_expr(e: &Expr, ctx: &Ctx) -> bool {
    match e {
        Expr::Variable(n) => is_array(ctx, n),
        Expr::Operator(n) => {
            if matches!(n.op.as_str(),
                        "aggregate" | "makearray" | "arrayop" | "broadcast"
                        | "reshape" | "transpose" | "concat") {
                return true;
            }
            if n.op == "index" {
                return false;
            }
            n.args.iter().any(|x| is_array_expr(x, ctx))
        }
        _ => false,
    }
}

fn site_cidx(s: &Site) -> Vec<Expr> {
    match &s.expr {
        Expr::Operator(n) => n.args[1..].to_vec(),
        _ => Vec::new(),
    }
}

pub fn bands(rhs: &Expr, v: &str, ctx: &Ctx, shape_u: &[i64]) -> R<Vec<Band>> {
    let mut out = Vec::new();
    if shape_u.is_empty() {
        for s in sorted_sites(rhs, v) {
            let coef = simp(&dscalar(rhs, &s)?);
            if is_zero(&coef) {
                continue;
            }
            out.push(Band {
                rows: vec![],
                ridx: vec![],
                cidx: site_cidx(&s),
                coef,
                contracted: vec![],
            });
        }
        return Ok(out);
    }
    bands_array(&mut out, rhs, v, ctx, shape_u, &[Expr::Integer(1)])?;
    Ok(out)
}

fn scalar_bands(out: &mut Vec<Band>, e: &Expr, v: &str,
                rows: &[(i64, i64)], ridx: &[RIdx], scale: &[Expr]) -> R<()> {
    for (rows2, body) in clip_regions(e, ridx, rows) {
        for s in sorted_sites(&body, v) {
            let mut fac = vec![dscalar(&body, &s)?];
            fac.extend(scale.iter().cloned());
            let coef = simp(&mul(fac));
            if is_zero(&coef) {
                continue;
            }
            out.push(Band {
                rows: rows2.clone(),
                ridx: ridx.to_vec(),
                cidx: site_cidx(&s),
                coef,
                contracted: vec![],
            });
        }
    }
    Ok(())
}

fn row_names(shape_u: &[i64]) -> Vec<RIdx> {
    (1..=shape_u.len()).map(|k| RIdx::Name(format!("_r{k}"))).collect()
}

fn bands_array(out: &mut Vec<Band>, e: &Expr, v: &str, ctx: &Ctx,
               shape_u: &[i64], scale: &[Expr]) -> R<()> {
    if let Expr::Variable(name) = e {
        if name != v {
            return Ok(());
        }
        let names = row_names(shape_u);
        out.push(Band {
            rows: shape_u.iter().map(|n| (1, *n)).collect(),
            ridx: names.clone(),
            cidx: names.iter().map(|r| match r {
                RIdx::Name(n) => Expr::Variable(n.clone()),
                RIdx::Fixed(i) => Expr::Integer(*i),
            }).collect(),
            coef: mul(scale.to_vec()),
            contracted: vec![],
        });
        return Ok(());
    }
    let Expr::Operator(n) = e else { return Ok(()) };
    match n.op.as_str() {
        "+" => {
            for x in &n.args {
                bands_array(out, x, v, ctx, shape_u, scale)?;
            }
        }
        "-" => {
            bands_array(out, &n.args[0], v, ctx, shape_u, scale)?;
            if n.args.len() == 2 {
                let mut sc = scale.to_vec();
                sc.push(Expr::Integer(-1));
                bands_array(out, &n.args[1], v, ctx, shape_u, &sc)?;
            }
        }
        "neg" => {
            let mut sc = scale.to_vec();
            sc.push(Expr::Integer(-1));
            bands_array(out, &n.args[0], v, ctx, shape_u, &sc)?;
        }
        "*" => {
            let arr: Vec<&Expr> =
                n.args.iter().filter(|x| is_array_expr(x, ctx)).collect();
            let sc: Vec<&Expr> =
                n.args.iter().filter(|x| !is_array_expr(x, ctx)).collect();
            if arr.len() != 1 {
                return Err(BandError(format!(
                    "elementwise product of {} array-valued factors at array \
                     level; wrap in an aggregate", arr.len())));
            }
            if sc.iter().any(|x| crate::helpers::occurs_var(x, v)) {
                return Err(BandError(format!(
                    "scalar factor depends on `{v}` in an array-level product")));
            }
            let mut sc2 = scale.to_vec();
            sc2.extend(sc.into_iter().cloned());
            bands_array(out, arr[0], v, ctx, shape_u, &sc2)?;
        }
        "/" => {
            if is_array_expr(&n.args[1], ctx) {
                return Err(BandError("array-valued divisor at array level".into()));
            }
            let mut sc = scale.to_vec();
            sc.push(op("/", vec![Expr::Integer(1), n.args[1].clone()]));
            bands_array(out, &n.args[0], v, ctx, shape_u, &sc)?;
        }
        "makearray" => bands_makearray(out, n, v, ctx, scale)?,
        "aggregate" => bands_aggregate(out, n, v, ctx, shape_u, scale)?,
        "index" => {
            let names = row_names(shape_u);
            let rows: Vec<(i64, i64)> = shape_u.iter().map(|x| (1, *x)).collect();
            scalar_bands(out, e, v, &rows, &names, scale)?;
        }
        _ => {
            if is_array_expr(e, ctx) {
                let names = row_names(shape_u);
                let idx: Vec<Expr> = names.iter().map(|r| match r {
                    RIdx::Name(nm) => Expr::Variable(nm.clone()),
                    RIdx::Fixed(i) => Expr::Integer(*i),
                }).collect();
                let body = push_index(e, &idx, ctx)?;
                let mut ranges = HashMap::new();
                let mut oidx = Vec::new();
                for (k, r) in names.iter().enumerate() {
                    if let RIdx::Name(nm) = r {
                        oidx.push(nm.clone());
                        ranges.insert(nm.clone(),
                                      RangeSpec::Interval([1, shape_u[k]]));
                    }
                }
                let agg = Expr::operator(ExpressionNode {
                    op: "aggregate".into(),
                    output_idx: Some(oidx),
                    expr: Some(Box::new(body)),
                    ranges: Some(ranges),
                    ..Default::default()
                });
                bands_array(out, &agg, v, ctx, shape_u, scale)?;
            } else {
                let names = row_names(shape_u);
                let rows: Vec<(i64, i64)> =
                    shape_u.iter().map(|x| (1, *x)).collect();
                scalar_bands(out, e, v, &rows, &names, scale)?;
            }
        }
    }
    Ok(())
}

fn bands_aggregate(out: &mut Vec<Band>, n: &ExpressionNode, v: &str, ctx: &Ctx,
                   shape_u: &[i64], scale: &[Expr]) -> R<()> {
    let oidx: Vec<String> = n.output_idx.clone().unwrap_or_default();
    if oidx.iter().any(|x| x.parse::<i64>().is_ok()) {
        return Err(BandError(
            "singleton `1` output_idx entries not supported yet".into()));
    }
    if n.filter.is_some() {
        return Err(BandError(
            "filtered aggregates not supported (the filter gate would be \
             dropped from the derivative)".into()));
    }
    let empty = HashMap::new();
    let ranges = n.ranges.as_ref().unwrap_or(&empty);
    let mut cnames: Vec<String> = ranges.keys()
        .filter(|k| !oidx.contains(k)).cloned().collect();
    cnames.sort();
    if let Some(red) = &n.reduce {
        if red != "+" {
            return Err(BandError(format!(
                "reduce=`{red}` (non-smooth semiring reductions have no bands)")));
        }
    }
    let mut crange = Vec::new();
    for nm in &cnames {
        let (lo, hi) = range_of(&ranges[nm], ctx)?;
        crange.push((nm.clone(), lo, hi));
    }
    let mut rows = Vec::new();
    for (k, nm) in oidx.iter().enumerate() {
        rows.push(match ranges.get(nm) {
            Some(r) => range_of(r, ctx)?,
            None => (1, shape_u[k]),
        });
    }
    let body = n.expr.as_deref()
        .ok_or_else(|| BandError("aggregate without body".into()))?;
    let ridx: Vec<RIdx> = oidx.iter().map(|s| RIdx::Name(s.clone())).collect();
    for (rows2, body2) in clip_regions(body, &ridx, &rows) {
        for s in sorted_sites(&body2, v) {
            let d = dscalar(&body2, &s)?;
            if matches!(&s.expr, Expr::Variable(_)) && is_array(ctx, v) {
                return Err(BandError(format!(
                    "whole-array reference to `{v}` inside an aggregate body")));
            }
            let cidx = site_cidx(&s);
            if crange.is_empty() {
                let mut fac = vec![d];
                fac.extend(scale.iter().cloned());
                let coef = simp(&mul(fac));
                if !is_zero(&coef) {
                    out.push(Band { rows: rows2.clone(), ridx: ridx.clone(),
                                    cidx, coef, contracted: vec![] });
                }
            } else if !crange.iter().any(|(nm, _, _)| {
                cidx.iter().any(|c| free_variables(c).contains(nm))
            }) {
                // column independent of the contraction: symbolic sum coef
                let mut rngs = HashMap::new();
                for (nm, lo, hi) in &crange {
                    rngs.insert(nm.clone(), RangeSpec::Interval([*lo, *hi]));
                }
                let red = Expr::operator(ExpressionNode {
                    op: "aggregate".into(),
                    output_idx: Some(vec![]),
                    reduce: Some("+".into()),
                    expr: Some(Box::new(d)),
                    ranges: Some(rngs),
                    ..Default::default()
                });
                let mut fac = vec![red];
                fac.extend(scale.iter().cloned());
                let coef = simp(&mul(fac));
                if !is_zero(&coef) {
                    out.push(Band { rows: rows2.clone(), ridx: ridx.clone(),
                                    cidx, coef, contracted: vec![] });
                }
            } else {
                let mut fac = vec![d];
                fac.extend(scale.iter().cloned());
                let coef = simp(&mul(fac));
                if !is_zero(&coef) {
                    out.push(Band { rows: rows2.clone(), ridx: ridx.clone(),
                                    cidx, coef, contracted: crange.clone() });
                }
            }
        }
    }
    Ok(())
}

fn bands_makearray(out: &mut Vec<Band>, n: &ExpressionNode, v: &str,
                   ctx: &Ctx, scale: &[Expr]) -> R<()> {
    let regions = n.regions.as_ref()
        .ok_or_else(|| BandError("makearray without regions".into()))?;
    let values = n.values.as_ref()
        .ok_or_else(|| BandError("makearray without values".into()))?;
    for (region, val) in regions.iter().zip(values.iter()) {
        let rows: Vec<(i64, i64)> = region.iter().map(|r| (r[0], r[1])).collect();
        if rows.iter().any(|(lo, hi)| hi < lo) {
            continue;                                 // folded-empty region
        }
        let nonsing: Vec<usize> = rows.iter().enumerate()
            .filter(|(_, (lo, hi))| hi > lo).map(|(k, _)| k).collect();
        if is_array_expr(val, ctx) {
            let mut sub = Vec::new();
            let val_fullrank = matches!(val, Expr::Operator(vn)
                if vn.op == "aggregate"
                   && vn.output_idx.as_ref().map(|o| o.len()).unwrap_or(0)
                      == rows.len());
            let sub_shape: Vec<i64> = if val_fullrank {
                rows.iter().map(|(lo, hi)| hi - lo + 1).collect()
            } else {
                nonsing.iter().map(|&k| rows[k].1 - rows[k].0 + 1).collect()
            };
            bands_array(&mut sub, val, v, ctx, &sub_shape, scale)?;
            let origins: Vec<i64> = if let Expr::Operator(vn) = val {
                if vn.op == "aggregate" {
                    let empty = HashMap::new();
                    let vr = vn.ranges.as_ref().unwrap_or(&empty);
                    let mut o = Vec::new();
                    for nm in vn.output_idx.as_ref().unwrap_or(&vec![]) {
                        o.push(match vr.get(nm) {
                            Some(r) => range_of(r, ctx)?.0,
                            None => 1,
                        });
                    }
                    o
                } else {
                    vec![1; nonsing.len()]
                }
            } else {
                vec![1; nonsing.len()]
            };
            for b in sub {
                let mut ridx = Vec::new();
                let mut subst: HashMap<String, Expr> = HashMap::new();
                let mut rows2 = Vec::new();
                let fullrank = b.ridx.len() == rows.len();
                let mut kk: isize = -1;
                for (_k, (lo, hi)) in rows.iter().enumerate() {
                    if fullrank || hi > lo {
                        kk += 1;
                        let r = &b.ridx[kk as usize];
                        let (rlo, rhi) = b.rows[kk as usize];
                        let shift = lo - origins[kk as usize];
                        match r {
                            RIdx::Name(nm) => {
                                ridx.push(RIdx::Name(nm.clone()));
                                if shift != 0 {
                                    subst.insert(nm.clone(),
                                        add(vec![Expr::Variable(nm.clone()),
                                                 Expr::Integer(-shift)]));
                                }
                                rows2.push((rlo + shift, rhi + shift));
                            }
                            RIdx::Fixed(i) => {
                                ridx.push(RIdx::Fixed(i + shift));
                                rows2.push((i + shift, i + shift));
                            }
                        }
                    } else {
                        ridx.push(RIdx::Fixed(*lo));
                        rows2.push((*lo, *lo));
                    }
                }
                let (coef, cidx) = if subst.is_empty() {
                    (b.coef.clone(), b.cidx.clone())
                } else {
                    (substitute(&b.coef, &subst),
                     b.cidx.iter().map(|c| substitute(c, &subst)).collect())
                };
                out.push(Band { rows: rows2, ridx, cidx, coef,
                                contracted: b.contracted.clone() });
            }
        } else {
            let ridx: Vec<RIdx> =
                rows.iter().map(|(lo, _)| RIdx::Fixed(*lo)).collect();
            scalar_bands(out, val, v, &rows, &ridx, scale)?;
        }
    }
    Ok(())
}

fn push_index(e: &Expr, idx: &[Expr], ctx: &Ctx) -> R<Expr> {
    match e {
        Expr::Variable(nm) if is_array(ctx, nm) => {
            let mut args = vec![e.clone()];
            args.extend(idx.iter().cloned());
            Ok(op("index", args))
        }
        Expr::Operator(n) => {
            if matches!(n.op.as_str(), "aggregate" | "makearray" | "index") {
                return Err(BandError(format!(
                    "cannot push an index into `{}` nested under an \
                     elementwise op", n.op)));
            }
            let mut err = None;
            let out = map_expr_children(e, &mut |x| {
                match push_index(x, idx, ctx) {
                    Ok(v) => v,
                    Err(er) => {
                        if err.is_none() {
                            err = Some(er);
                        }
                        x.clone()
                    }
                }
            });
            match err {
                Some(er) => Err(er),
                None => Ok(out),
            }
        }
        _ => Ok(e.clone()),
    }
}

pub fn normalize_band(b: &Band) -> Band {
    let mut subst: HashMap<String, Expr> = HashMap::new();
    let mut ridx = Vec::new();
    for (k, r) in b.ridx.iter().enumerate() {
        let (lo, hi) = b.rows[k];
        match r {
            RIdx::Name(nm) if lo == hi => {
                subst.insert(nm.clone(), Expr::Integer(lo));
                ridx.push(RIdx::Fixed(lo));
            }
            other => ridx.push(other.clone()),
        }
    }
    let mut contracted = Vec::new();
    for (nm, lo, hi) in &b.contracted {
        if lo == hi {
            subst.insert(nm.clone(), Expr::Integer(*lo));
        } else {
            contracted.push((nm.clone(), *lo, *hi));
        }
    }
    if subst.is_empty() {
        return b.clone();
    }
    Band {
        rows: b.rows.clone(),
        ridx,
        cidx: b.cidx.iter()
            .map(|c| canon_lits(&simplify_jl(&substitute(c, &subst))))
            .collect(),
        coef: simp(&substitute(&b.coef, &subst)),
        contracted,
    }
}

pub fn merge_bands(bs: Vec<Band>) -> Vec<Band> {
    if bs.len() < 2 {
        return bs;
    }
    let key = |b: &Band| {
        (b.ridx.clone(),
         b.cidx.iter().map(skey).collect::<Vec<_>>(),
         skey(&b.coef),
         b.contracted.clone())
    };
    let ks: Vec<_> = bs.iter().map(key).collect();
    let mut out = bs;
    let mut alive = vec![true; out.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for a in 0..out.len() {
            if !alive[a] {
                continue;
            }
            for b in (a + 1)..out.len() {
                if !alive[b] || ks[a] != ks[b] {
                    continue;
                }
                let (ra, rb) = (out[a].rows.clone(), out[b].rows.clone());
                if ra.len() != rb.len() {
                    continue;
                }
                let mut d: isize = -1;
                let mut ok = true;
                for k in 0..ra.len() {
                    if ra[k] == rb[k] {
                        continue;
                    }
                    if d < 0 && (ra[k].1 + 1 == rb[k].0
                                 || rb[k].1 + 1 == ra[k].0) {
                        d = k as isize;
                    } else {
                        ok = false;
                        break;
                    }
                }
                if !ok || d < 0 {
                    continue;
                }
                let du = d as usize;
                let lo = ra[du].0.min(rb[du].0);
                let hi = ra[du].1.max(rb[du].1);
                out[a].rows[du] = (lo, hi);
                alive[b] = false;
                changed = true;
            }
        }
    }
    out.into_iter().zip(alive).filter(|(_, a)| *a).map(|(b, _)| b).collect()
}
