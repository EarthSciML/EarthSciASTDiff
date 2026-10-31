//! The array-level band calculus (the reference's bands.jl).

use std::collections::{BTreeMap, HashMap, HashSet};

use earthsci_ast::extension::types::RangeSpec;
use earthsci_ast::substitute;
use earthsci_ast::{Expr, ExpressionNode, RegionBound};

use crate::branches::{canon_lits, simp};
use crate::clip::{clip_regions, RIdx};
use crate::helpers::{add, free_vars, is_zero, map_expr_children, mul, op, skey, Site};
use crate::inline::{all_names, binders};
use crate::reductions::{
    conj, conj_list, conjuncts, index_only, one_hot, reduce_op, split_guard, window_weight,
};
use crate::rules::{dscalar, DerivativeRuleError};
use crate::simplify_port::simplify_jl;

/// One rectangular region of a Jacobian block: the rows it covers
/// (`rows`, 1-based inclusive ranges, named by `ridx`), the column-index
/// expressions (`cidx`) and the coefficient expression (`coef`). `contracted`
/// lists extra column dimensions that accumulate (summed) at assembly.
/// `guard` is an exact point predicate (the entry's `where` field) over the
/// row and contracted index names, literals and `const` gathers: a point
/// contributes iff it holds. What a filtered faq node's non-rectangular
/// window (`gk < gke`) leaves after its box is split at affine breakpoints;
/// `None` for every rectangular band.
#[derive(Clone, Debug)]
pub struct Band {
    pub rows: Vec<(i64, i64)>,
    pub ridx: Vec<RIdx>,
    pub cidx: Vec<Expr>,
    pub coef: Expr,
    pub contracted: Vec<(String, i64, i64)>,
    pub guard: Option<Expr>,
}

#[derive(Clone, Debug, Default)]
pub struct Ctx {
    pub index_sets: HashMap<String, i64>,
    pub shapes: HashMap<String, Vec<i64>>,
    pub params: HashSet<String>,
}

/// An expression has a form the band calculus cannot differentiate (for
/// example a joined `faq` node, the `bool_and_or` semiring, or a `max` over
/// two contracted indices).
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
            if n.op == "index" && matches!(n.args.first(), Some(Expr::Variable(t)) if t == v) =>
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
            if matches!(
                n.op.as_str(),
                "faq" | "makearray" | "broadcast" | "reshape" | "transpose" | "concat"
            ) {
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

/// Per body: for each site of `v` under a nested scalar faq sum, the
/// indices the sums ENCLOSING it bind (bands.jl `_nested_sums`): site key →
/// name → range, or `None` (ambiguous) for a name also bound by `outer`,
/// bound twice on the way down, or over different ranges at different
/// occurrences. Position-aware: a sum elsewhere in the body does not affect
/// a site outside it.
/// Each site under a sum also keeps the filters of the sums around it
/// (`filters`): the same list at every such occurrence, or `None` (mixed) —
/// compared by keys taken once per filter (`fkeys`), so an occurrence costs
/// no re-serialization of its filters.
#[derive(Default)]
pub(crate) struct Nested {
    ranges: HashMap<String, HashMap<String, Option<RangeSpec>>>,
    filters: HashMap<String, Option<Vec<Expr>>>,
    fkeys: HashMap<String, Vec<String>>,
}

/// The filters of the sums enclosing a walk position, with their keys.
#[derive(Default)]
struct FilterStack {
    exprs: Vec<Expr>,
    keys: Vec<String>,
}

pub(crate) fn nested_sums(e: &Expr, v: &str, outer: &[String]) -> Nested {
    let mut acc = Nested::default();
    nested_walk(
        e,
        v,
        outer,
        &mut Vec::new(),
        &mut FilterStack::default(),
        &mut acc,
    );
    acc
}

fn nested_walk(
    e: &Expr,
    v: &str,
    outer: &[String],
    stack: &mut Vec<(String, RangeSpec)>,
    fstack: &mut FilterStack,
    acc: &mut Nested,
) {
    let is_site = match e {
        Expr::Variable(nm) => nm == v,
        Expr::Operator(n) => {
            n.op == "index" && matches!(n.args.first(), Some(Expr::Variable(t)) if t == v)
        }
        _ => false,
    };
    if is_site {
        if stack.is_empty() {
            return;
        }
        let key = Site::of(e).key;
        match acc.filters.get_mut(&key) {
            None => {
                acc.filters.insert(key.clone(), Some(fstack.exprs.clone()));
                acc.fkeys.insert(key.clone(), fstack.keys.clone());
            }
            Some(f) => {
                if f.is_some() && acc.fkeys[&key] != fstack.keys {
                    *f = None;
                }
            }
        }
        let d = acc.ranges.entry(key).or_default();
        let mut seen: HashSet<&String> = HashSet::new();
        for (nm, r) in stack.iter() {
            let amb = outer.contains(nm)
                || seen.contains(nm)
                || d.get(nm).is_some_and(|prev| prev.as_ref() != Some(r));
            d.insert(nm.clone(), if amb { None } else { Some(r.clone()) });
            seen.insert(nm);
        }
        return;
    }
    let Expr::Operator(n) = e else { return };
    let depth = stack.len();
    let fdepth = fstack.exprs.len();
    if n.op == "faq" && n.output_idx.as_ref().is_none_or(|o| o.is_empty()) {
        if let Some(rs) = &n.ranges {
            let mut ks: Vec<&String> = rs.keys().collect();
            ks.sort();
            for k in ks {
                stack.push((k.clone(), rs[k].clone()));
            }
            if let Some(f) = &n.filter {
                fstack.keys.push(skey(f));
                fstack.exprs.push((**f).clone());
            }
        }
    }
    n.for_each_child(&mut |c: &Expr| nested_walk(c, v, outer, stack, fstack, acc));
    stack.truncate(depth);
    fstack.exprs.truncate(fdepth);
    fstack.keys.truncate(fdepth);
}

/// Substitution that does not descend into a faq for the names it binds.
pub(crate) fn subst_free(e: &Expr, m: &HashMap<String, Expr>) -> Expr {
    match e {
        Expr::Variable(nm) => m.get(nm).cloned().unwrap_or_else(|| e.clone()),
        Expr::Operator(n) => {
            if n.op == "faq" {
                let bound = binders(n);
                if m.keys().any(|k| bound.contains(k)) {
                    let m2: HashMap<String, Expr> = m
                        .iter()
                        .filter(|(k, _)| !bound.contains(k))
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    if m2.is_empty() {
                        return e.clone();
                    }
                    return map_expr_children(e, &mut |x| subst_free(x, &m2));
                }
            }
            map_expr_children(e, &mut |x| subst_free(x, m))
        }
        _ => e.clone(),
    }
}

/// A site under a nested sum whose column reads the sum's index: that index
/// becomes a CONTRACTED column dimension of the band, renamed apart in
/// `cidx` and in the coefficient's free occurrences (bands.jl
/// `_lift_nested`). The sums' index-only filter conjuncts over a lifted
/// name become the band's guard ([`nested_guard`]). `nested` and the lazily
/// filled `taken` (the body's names) are per body, not per site.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub(crate) fn lift_nested(
    nested: &Nested,
    s: &Site,
    body: &Expr,
    cidx: Vec<Expr>,
    coef: Expr,
    ctx: &Ctx,
    outer: &[String],
    taken: &mut Option<HashSet<String>>,
) -> R<Lifted> {
    let Some(d) = nested.ranges.get(&s.key) else {
        return Ok((cidx, coef, vec![], None));
    };
    let mut read: HashSet<String> = HashSet::new();
    for c in &cidx {
        read.extend(free_vars(c));
    }
    let mut names: Vec<&String> = d.keys().filter(|n| read.contains(*n)).collect();
    if names.is_empty() {
        return Ok((cidx, coef, vec![], None));
    }
    names.sort();
    let taken = taken.get_or_insert_with(|| {
        let mut t: HashSet<String> = outer.iter().cloned().collect();
        all_names(body, &mut t);
        t
    });
    let mut ren: HashMap<String, Expr> = HashMap::new();
    let mut chosen: HashSet<String> = HashSet::new();
    let mut extra = Vec::new();
    for &nm in &names {
        let Some(r) = &d[nm] else {
            return Err(BandError(format!(
                "a column reads `{nm}`, which the sums around it bind \
                 ambiguously (shadowing an enclosing index, or over two \
                 different ranges)"
            )));
        };
        let mut f = format!("{nm}_j");
        while taken.contains(&f) || chosen.contains(&f) {
            f.push_str("_j");
        }
        chosen.insert(f.clone());
        let (lo, hi) = range_of(r, ctx)?;
        ren.insert(nm.clone(), Expr::Variable(f.clone()));
        extra.push((f, lo, hi));
    }
    let lifted: Vec<String> = names.into_iter().cloned().collect();
    let guard = match nested.filters.get(&s.key) {
        Some(Some(fs)) => nested_guard(fs, d, &lifted, outer, &ren),
        _ => None,
    };
    let cidx = cidx.iter().map(|c| substitute(c, &ren)).collect();
    Ok((cidx, subst_free(&coef, &ren), extra, guard))
}

/// [`lift_nested`]'s result: column indices, coefficient, the lifted
/// contracted dimensions and the guard.
pub(crate) type Lifted = (Vec<Expr>, Expr, Vec<(String, i64, i64)>, Option<Expr>);

/// The index-only conjuncts of the filters around a site that read a lifted
/// name — and no name a sum around it still binds — renamed like the
/// column: the band's guard (bands.jl `_nested_guard`). The coefficient
/// keeps them too (as the remaining sum's filter, or as an `ifelse` once
/// every bound name is lifted), so the guard only makes the pattern exact;
/// a site under different filters at different occurrences gets none.
fn nested_guard(
    filters: &[Expr],
    d: &HashMap<String, Option<RangeSpec>>,
    lifted: &[String],
    outer: &[String],
    ren: &HashMap<String, Expr>,
) -> Option<Expr> {
    let ok: HashSet<&String> = outer.iter().chain(lifted.iter()).collect();
    let mut gs = Vec::new();
    for f in filters {
        for c in conjuncts(f) {
            let fv = free_vars(&c);
            if !lifted.iter().any(|n| fv.contains(n)) {
                continue;
            }
            if !fv.iter().all(|n| ok.contains(n)) {
                continue;
            }
            if fv.iter().any(|n| d.contains_key(n) && !lifted.contains(n)) {
                continue;
            }
            gs.push(substitute(&c, ren));
        }
    }
    conj_list(gs)
}

pub(crate) fn site_cidx(s: &Site) -> Vec<Expr> {
    match &s.expr {
        Expr::Operator(n) => n.args[1..].to_vec(),
        _ => Vec::new(),
    }
}

pub fn bands(rhs: &Expr, v: &str, ctx: &Ctx, shape_u: &[i64]) -> R<Vec<Band>> {
    let mut out = Vec::new();
    if shape_u.is_empty() {
        let nested = nested_sums(rhs, v, &[]);
        let mut taken = None;
        for s in sorted_sites(rhs, v) {
            let (cidx, d, extra, g) = lift_nested(
                &nested,
                &s,
                rhs,
                site_cidx(&s),
                dscalar(rhs, &s)?,
                ctx,
                &[],
                &mut taken,
            )?;
            let coef = simp(&d);
            if is_zero(&coef) {
                continue;
            }
            push_band(&mut out, vec![], vec![], cidx, coef, extra, g, v, ctx)?;
        }
        return Ok(out);
    }
    bands_array(&mut out, rhs, v, ctx, shape_u, &[Expr::Integer(1)])?;
    Ok(out)
}

/// Push one band, its value-reading column dimensions made one-hot first
/// (bands.jl `_push_band!`).
#[allow(clippy::too_many_arguments)]
fn push_band(
    out: &mut Vec<Band>,
    rows: Vec<(i64, i64)>,
    ridx: Vec<RIdx>,
    cidx: Vec<Expr>,
    coef: Expr,
    contracted: Vec<(String, i64, i64)>,
    guard: Option<Expr>,
    v: &str,
    ctx: &Ctx,
) -> R<()> {
    let (cidx, coef, contracted) = one_hot(cidx, coef, contracted, &ridx, v, ctx)?;
    out.push(Band {
        rows,
        ridx,
        cidx,
        coef,
        contracted,
        guard,
    });
    Ok(())
}

fn scalar_bands(
    out: &mut Vec<Band>,
    e: &Expr,
    v: &str,
    ctx: &Ctx,
    rows: &[(i64, i64)],
    ridx: &[RIdx],
    scale: &[Expr],
) -> R<()> {
    let outer: Vec<String> = ridx
        .iter()
        .filter_map(|r| match r {
            RIdx::Name(nm) => Some(nm.clone()),
            _ => None,
        })
        .collect();
    for (rows2, body) in clip_regions(e, ridx, rows) {
        let nested = nested_sums(&body, v, &outer);
        let mut taken = None;
        for s in sorted_sites(&body, v) {
            let (cidx, d, extra, g) = lift_nested(
                &nested,
                &s,
                &body,
                site_cidx(&s),
                dscalar(&body, &s)?,
                ctx,
                &outer,
                &mut taken,
            )?;
            let mut fac = vec![d];
            fac.extend(scale.iter().cloned());
            let coef = simp(&mul(fac));
            if is_zero(&coef) {
                continue;
            }
            push_band(
                out,
                rows2.clone(),
                ridx.to_vec(),
                cidx,
                coef,
                extra,
                g,
                v,
                ctx,
            )?;
        }
    }
    Ok(())
}

fn row_names(shape_u: &[i64]) -> Vec<RIdx> {
    (1..=shape_u.len())
        .map(|k| RIdx::Name(format!("_r{k}")))
        .collect()
}

fn bands_array(
    out: &mut Vec<Band>,
    e: &Expr,
    v: &str,
    ctx: &Ctx,
    shape_u: &[i64],
    scale: &[Expr],
) -> R<()> {
    if let Expr::Variable(name) = e {
        if name != v {
            return Ok(());
        }
        let names = row_names(shape_u);
        out.push(Band {
            rows: shape_u.iter().map(|n| (1, *n)).collect(),
            ridx: names.clone(),
            cidx: names
                .iter()
                .map(|r| match r {
                    RIdx::Name(n) => Expr::Variable(n.clone()),
                    RIdx::Fixed(i) => Expr::Integer(*i),
                })
                .collect(),
            coef: mul(scale.to_vec()),
            contracted: vec![],
            guard: None,
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
        "-" if n.args.len() == 1 => {
            let mut sc = scale.to_vec();
            sc.push(Expr::Integer(-1));
            bands_array(out, &n.args[0], v, ctx, shape_u, &sc)?;
        }
        "-" => {
            bands_array(out, &n.args[0], v, ctx, shape_u, scale)?;
            let mut sc = scale.to_vec();
            sc.push(Expr::Integer(-1));
            bands_array(out, &n.args[1], v, ctx, shape_u, &sc)?;
        }
        "neg" => {
            let mut sc = scale.to_vec();
            sc.push(Expr::Integer(-1));
            bands_array(out, &n.args[0], v, ctx, shape_u, &sc)?;
        }
        "*" => {
            let arr: Vec<&Expr> = n.args.iter().filter(|x| is_array_expr(x, ctx)).collect();
            let sc: Vec<&Expr> = n.args.iter().filter(|x| !is_array_expr(x, ctx)).collect();
            if arr.len() != 1 {
                return Err(BandError(format!(
                    "elementwise product of {} array-valued factors at array \
                     level; wrap in a faq node",
                    arr.len()
                )));
            }
            if sc.iter().any(|x| crate::helpers::occurs_var(x, v)) {
                return Err(BandError(format!(
                    "scalar factor depends on `{v}` in an array-level product"
                )));
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
        "faq" => bands_faq(out, n, v, ctx, shape_u, scale)?,
        "index" => {
            let names = row_names(shape_u);
            let rows: Vec<(i64, i64)> = shape_u.iter().map(|x| (1, *x)).collect();
            scalar_bands(out, e, v, ctx, &rows, &names, scale)?;
        }
        _ => {
            if is_array_expr(e, ctx) {
                let names = row_names(shape_u);
                let idx: Vec<Expr> = names
                    .iter()
                    .map(|r| match r {
                        RIdx::Name(nm) => Expr::Variable(nm.clone()),
                        RIdx::Fixed(i) => Expr::Integer(*i),
                    })
                    .collect();
                let body = push_index(e, &idx, ctx)?;
                let mut ranges = HashMap::new();
                let mut oidx = Vec::new();
                for (k, r) in names.iter().enumerate() {
                    if let RIdx::Name(nm) = r {
                        oidx.push(nm.clone());
                        ranges.insert(nm.clone(), RangeSpec::Interval([1, shape_u[k]]));
                    }
                }
                let agg = Expr::operator(ExpressionNode {
                    op: "faq".into(),
                    output_idx: Some(oidx),
                    expr: Some(Box::new(body)),
                    ranges: Some(ranges),
                    ..Default::default()
                });
                bands_array(out, &agg, v, ctx, shape_u, scale)?;
            } else {
                let names = row_names(shape_u);
                let rows: Vec<(i64, i64)> = shape_u.iter().map(|x| (1, *x)).collect();
                scalar_bands(out, e, v, ctx, &rows, &names, scale)?;
            }
        }
    }
    Ok(())
}

/// A faq node at array level (bands.jl `_bands_faq!`). Each site of its body
/// differentiates as a term of the node's reduction over its window
/// (reductions.rs): the derivative is weighted per term ([`window_weight`]),
/// and the filter's index-only conjuncts split the band box
/// ([`split_guard`]) while its value-dependent conjuncts gate the
/// coefficient. Per box piece, a site whose column reads no contracted name
/// yields one band whose coefficient sums the weighted derivative over the
/// piece's contraction (filtered by the residual guard's contracted
/// conjuncts; its row-only conjuncts guard the band); a gathered column
/// yields a band with contracted dimensions and the whole residual guard.
fn bands_faq(
    out: &mut Vec<Band>,
    n: &ExpressionNode,
    v: &str,
    ctx: &Ctx,
    shape_u: &[i64],
    scale: &[Expr],
) -> R<()> {
    let oidx: Vec<String> = n.output_idx.clone().unwrap_or_default();
    if oidx.iter().any(|x| x.parse::<i64>().is_ok()) {
        return Err(BandError(
            "singleton `1` output_idx entries not supported yet".into(),
        ));
    }
    let red = reduce_op(n).map_err(BandError)?;
    let empty = HashMap::new();
    let ranges = n.ranges.as_ref().unwrap_or(&empty);
    let mut cnames: Vec<String> = ranges
        .keys()
        .filter(|k| !oidx.contains(k))
        .cloned()
        .collect();
    cnames.sort();
    let mut crange0 = Vec::new();
    for nm in &cnames {
        let (lo, hi) = range_of(&ranges[nm], ctx)?;
        crange0.push((nm.clone(), lo, hi));
    }
    let mut rows0 = Vec::new();
    for (k, nm) in oidx.iter().enumerate() {
        rows0.push(match ranges.get(nm) {
            Some(r) => range_of(r, ctx)?,
            None => (1, shape_u[k]),
        });
    }
    let body = n
        .expr
        .as_deref()
        .ok_or_else(|| BandError("faq without body".into()))?;
    let ridx: Vec<RIdx> = oidx.iter().map(|s| RIdx::Name(s.clone())).collect();
    let outer: Vec<String> = oidx.iter().chain(cnames.iter()).cloned().collect();
    let (mut gidx, mut gval) = (Vec::new(), Vec::new());
    if let Some(f) = &n.filter {
        for c in conjuncts(f) {
            if index_only(&c, &outer) {
                gidx.push(c);
            } else {
                gval.push(c);
            }
        }
    }
    type Piece = (Vec<(i64, i64)>, Vec<(String, i64, i64)>, Option<Expr>);
    let pieces: Vec<Piece> = match conj_list(gidx) {
        None => vec![(rows0, crange0, None)],
        Some(g) => {
            let no = oidx.len();
            let bx: Vec<(i64, i64)> = rows0
                .iter()
                .copied()
                .chain(crange0.iter().map(|(_, lo, hi)| (*lo, *hi)))
                .collect();
            split_guard(&g, &outer, &bx)
                .into_iter()
                .map(|(cell, g)| {
                    let cr = cnames
                        .iter()
                        .enumerate()
                        .map(|(k, nm)| (nm.clone(), cell[no + k].0, cell[no + k].1))
                        .collect();
                    (cell[..no].to_vec(), cr, g)
                })
                .collect()
        }
    };
    let vfilt = conj_list(gval);
    for (rows_p, crange, guard) in pieces {
        for (rows2, body2) in clip_regions(body, &ridx, &rows_p) {
            let nested = nested_sums(&body2, v, &outer);
            let mut taken = None;
            for s in sorted_sites(&body2, v) {
                if matches!(&s.expr, Expr::Variable(_)) && is_array(ctx, v) {
                    return Err(BandError(format!(
                        "whole-array reference to `{v}` inside a faq body"
                    )));
                }
                let (cidx, d, extra, g) = lift_nested(
                    &nested,
                    &s,
                    &body2,
                    site_cidx(&s),
                    dscalar(&body2, &s)?,
                    ctx,
                    &outer,
                    &mut taken,
                )?;
                let mut d = window_weight(red, n, &cnames, d).map_err(BandError)?;
                if let Some(vf) = &vfilt {
                    if !is_zero(&d) {
                        d = op("ifelse", vec![vf.clone(), d, Expr::Integer(0)]);
                    }
                }
                if crange.is_empty() {
                    let mut fac = vec![d];
                    fac.extend(scale.iter().cloned());
                    let coef = simp(&mul(fac));
                    if is_zero(&coef) {
                        continue;
                    }
                    let guard = conj(guard.clone(), g);
                    push_band(
                        out,
                        rows2.clone(),
                        ridx.clone(),
                        cidx,
                        coef,
                        extra,
                        guard,
                        v,
                        ctx,
                    )?;
                } else if !crange
                    .iter()
                    .any(|(nm, _, _)| cidx.iter().any(|c| free_vars(c).contains(nm)))
                {
                    // The column is independent of every contracted index: a
                    // plain band whose coefficient is the (scalar)
                    // `reduce: "+"` of the site derivative over the
                    // contraction. A guard conjunct reading a contracted
                    // name — the piece's own, or a nested sum's filter tied
                    // to it (`Σ_k Σ_{j≤k}`) — is no condition on the band's
                    // point: it filters that sum.
                    let (mut gc, mut gr) = (Vec::new(), Vec::new());
                    if let Some(gd) = &conj(guard.clone(), g) {
                        for c in conjuncts(gd) {
                            let fv = free_vars(&c);
                            if cnames.iter().any(|nm| fv.contains(nm)) {
                                gc.push(c);
                            } else {
                                gr.push(c);
                            }
                        }
                    }
                    let mut rngs = HashMap::new();
                    for (nm, lo, hi) in &crange {
                        rngs.insert(nm.clone(), RangeSpec::Interval([*lo, *hi]));
                    }
                    let red_sum = Expr::operator(ExpressionNode {
                        op: "faq".into(),
                        output_idx: Some(vec![]),
                        reduce: Some("+".into()),
                        expr: Some(Box::new(d)),
                        ranges: Some(rngs),
                        filter: conj_list(gc).map(Box::new),
                        ..Default::default()
                    });
                    let mut fac = vec![red_sum];
                    fac.extend(scale.iter().cloned());
                    let coef = simp(&mul(fac));
                    if is_zero(&coef) {
                        continue;
                    }
                    let guard = conj_list(gr);
                    push_band(
                        out,
                        rows2.clone(),
                        ridx.clone(),
                        cidx,
                        coef,
                        extra,
                        guard,
                        v,
                        ctx,
                    )?;
                } else {
                    // A gathered column (`u[j]`, `u[i+k]`, `u[conn[i,k]]`): a
                    // band with free contracted column dimensions; entries
                    // landing on one column accumulate at assembly.
                    let mut fac = vec![d];
                    fac.extend(scale.iter().cloned());
                    let coef = simp(&mul(fac));
                    if is_zero(&coef) {
                        continue;
                    }
                    let contracted = [crange.clone(), extra].concat();
                    let guard = conj(guard.clone(), g);
                    push_band(
                        out,
                        rows2.clone(),
                        ridx.clone(),
                        cidx,
                        coef,
                        contracted,
                        guard,
                        v,
                        ctx,
                    )?;
                }
            }
        }
    }
    Ok(())
}

/// A makearray region's inclusive `[lo, hi]`. Load folds bounds to integers
/// (esm-spec §9.7.6); a bound still left as a closed expression is evaluated,
/// as the reference's `_region_bound` does. One that names anything (an
/// unfolded metaparameter) is an error. (Upstream's `region_bounds` would
/// cover the integer case but is not exported from `earthsci_ast`.)
pub fn region_bounds(r: &[RegionBound; 2]) -> R<(i64, i64)> {
    let one = |b: &RegionBound| -> R<i64> {
        if let Some(i) = b.as_i64() {
            return Ok(i);
        }
        let RegionBound::Expr(e) = b else {
            unreachable!("Int is as_i64")
        };
        earthsci_ast::evaluate(e, &HashMap::new())
            .map(|x| x.round() as i64)
            .map_err(|er| BandError(format!("unfolded symbolic makearray region bound: {er:?}")))
    };
    Ok((one(&r[0])?, one(&r[1])?))
}

fn bands_makearray(
    out: &mut Vec<Band>,
    n: &ExpressionNode,
    v: &str,
    ctx: &Ctx,
    scale: &[Expr],
) -> R<()> {
    let regions = n
        .regions
        .as_ref()
        .ok_or_else(|| BandError("makearray without regions".into()))?;
    let values = n
        .values
        .as_ref()
        .ok_or_else(|| BandError("makearray without values".into()))?;
    for (region, val) in regions.iter().zip(values.iter()) {
        let rows: Vec<(i64, i64)> = region.iter().map(region_bounds).collect::<R<_>>()?;
        if rows.iter().any(|(lo, hi)| hi < lo) {
            continue; // folded-empty region
        }
        let nonsing: Vec<usize> = rows
            .iter()
            .enumerate()
            .filter(|(_, (lo, hi))| hi > lo)
            .map(|(k, _)| k)
            .collect();
        if is_array_expr(val, ctx) {
            let mut sub = Vec::new();
            let val_fullrank = matches!(val, Expr::Operator(vn)
                if vn.op == "faq"
                   && vn.output_idx.as_ref().map(|o| o.len()).unwrap_or(0)
                      == rows.len());
            let sub_shape: Vec<i64> = if val_fullrank {
                rows.iter().map(|(lo, hi)| hi - lo + 1).collect()
            } else {
                nonsing.iter().map(|&k| rows[k].1 - rows[k].0 + 1).collect()
            };
            bands_array(&mut sub, val, v, ctx, &sub_shape, scale)?;
            let origins: Vec<i64> = if let Expr::Operator(vn) = val {
                if vn.op == "faq" {
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
                for (lo, hi) in rows.iter() {
                    if fullrank || hi > lo {
                        kk += 1;
                        let r = &b.ridx[kk as usize];
                        let (rlo, rhi) = b.rows[kk as usize];
                        let shift = lo - origins[kk as usize];
                        match r {
                            RIdx::Name(nm) => {
                                ridx.push(RIdx::Name(nm.clone()));
                                if shift != 0 {
                                    subst.insert(
                                        nm.clone(),
                                        add(vec![
                                            Expr::Variable(nm.clone()),
                                            Expr::Integer(-shift),
                                        ]),
                                    );
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
                let (coef, cidx, guard) = if subst.is_empty() {
                    (b.coef.clone(), b.cidx.clone(), b.guard.clone())
                } else {
                    (
                        substitute(&b.coef, &subst),
                        b.cidx.iter().map(|c| substitute(c, &subst)).collect(),
                        b.guard.as_ref().map(|g| substitute(g, &subst)),
                    )
                };
                out.push(Band {
                    rows: rows2,
                    ridx,
                    cidx,
                    coef,
                    contracted: b.contracted.clone(),
                    guard,
                });
            }
        } else {
            let ridx: Vec<RIdx> = rows.iter().map(|(lo, _)| RIdx::Fixed(*lo)).collect();
            scalar_bands(out, val, v, ctx, &rows, &ridx, scale)?;
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
            if matches!(n.op.as_str(), "faq" | "makearray" | "index") {
                return Err(BandError(format!(
                    "cannot push an index into `{}` nested under an \
                     elementwise op",
                    n.op
                )));
            }
            let mut err = None;
            let out = map_expr_children(e, &mut |x| match push_index(x, idx, ctx) {
                Ok(v) => v,
                Err(er) => {
                    if err.is_none() {
                        err = Some(er);
                    }
                    x.clone()
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
        cidx: b
            .cidx
            .iter()
            .map(|c| canon_lits(&simplify_jl(&substitute(c, &subst))))
            .collect(),
        coef: simp(&substitute(&b.coef, &subst)),
        contracted,
        guard: b
            .guard
            .as_ref()
            .map(|g| canon_lits(&simplify_jl(&substitute(g, &subst)))),
    }
}

pub fn merge_bands(bs: Vec<Band>) -> Vec<Band> {
    if bs.len() < 2 {
        return bs;
    }
    let key = |b: &Band| {
        (
            b.ridx.clone(),
            b.cidx.iter().map(skey).collect::<Vec<_>>(),
            skey(&b.coef),
            b.contracted.clone(),
            b.guard.as_ref().map(skey),
        )
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
                    if d < 0 && (ra[k].1 + 1 == rb[k].0 || rb[k].1 + 1 == ra[k].0) {
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
    out.into_iter()
        .zip(alive)
        .filter(|(_, a)| *a)
        .map(|(b, _)| b)
        .collect()
}
