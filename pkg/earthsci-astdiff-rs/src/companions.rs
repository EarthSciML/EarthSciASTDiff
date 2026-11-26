//! Time companions (spec §4.2; the reference's companions.jl): the time
//! bands keep the chain through the observed that move with `t` as named
//! steps. Each such observed `o` gets ONE derived observed, its companion
//! `_dt_<o>` = ∂o/∂t (data held fixed), and every band reads it through the
//! chain rule instead of differentiating `o`'s definition again inside every
//! row:
//!
//! ```text
//! a read of o at column index φ, with ∂f/∂o = c there   ↦   c · _dt_o[φ]
//! ```
//!
//! A companion's own definition is built the same way from `o`'s
//! definition: its direct derivative in `t` plus, for each moving observed
//! `o′` it reads, `∂def_o/∂o′ · _dt_o′`. Nothing is inlined and then
//! deduplicated: the moving observed are extra COLUMNS of the band calculus
//! (as a recurrence is, §4.5), so each definition is differentiated once.

use std::collections::{BTreeMap, HashMap, HashSet};

use earthsci_ast::extension::types::RangeSpec;
use earthsci_ast::free_variables;
use earthsci_ast::{Expr, ExpressionNode};
use serde_json::{json, Map, Value};

use crate::bands::{
    merge_bands, normalize_band, placeholder, range_of, region_bounds, subst_free, Band, BandError,
    Ctx,
};
use crate::branches::simp;
use crate::clip::{index_bounds, index_box, RIdx};
use crate::helpers::{mul, op, ser_expr, skey};
use crate::inline::{all_names, inline_observed_ranked};
use crate::operator_refs::{ref_node, with_layer};
use crate::reductions::{conj_list, conjuncts, fresh};
use crate::system::{moving_observed, subscript_observed, SysView};

type R<T> = Result<T, BandError>;

/// The time pass's companion layout (companions.jl `_Companions`): `names`
/// maps every observed that moves with `t` to its companion's name
/// (assigned over all of them, so a name never depends on the pass); `cols`
/// is the set that are columns of this pass (those a column index does not
/// need inlined); `inline` the pass's inlining map (the observed a column
/// index reads); `ctx` the band context with every column's extents;
/// `bands[o]` the companion's time bands once computed (empty: `∂o/∂t ≡ 0`,
/// and a read of `o` contributes nothing).
pub(crate) struct Companions {
    names: HashMap<String, String>,
    cols: HashSet<String>,
    pub(crate) inline: HashMap<String, Expr>,
    obs: HashMap<String, Expr>,
    pub(crate) ctx: Ctx,
    scalars: HashSet<String>,
    co: bool,
    bands: HashMap<String, Vec<Band>>,
}

/// The companion name of each observed in `moving`, in name order:
/// `_dt_<o>` (`P._dt_<v>` for a scoped `P.v`, so it lives in `o`'s own
/// component), with `_` appended until clear of every variable and
/// index-set name of the source and of the companion names before it.
pub(crate) fn companion_names(sv: &SysView, moving: &HashSet<String>) -> HashMap<String, String> {
    let mut taken: HashSet<String> = sv.variables.keys().cloned().collect();
    taken.extend(sv.index_sets.keys().cloned());
    let mut sorted: Vec<&String> = moving.iter().collect();
    sorted.sort();
    let mut out = HashMap::new();
    for o in sorted {
        let base = match o.rfind('.') {
            None => format!("_dt_{o}"),
            Some(i) => format!("{}._dt_{}", &o[..i], &o[i + 1..]),
        };
        out.insert(o.clone(), fresh(&mut taken, &base, "_"));
    }
    out
}

/// The companion layout of the time pass over the view. `obs` is every
/// observed definition with the recurrences left out (the pass never
/// inlines a recurrence); `scalars` the declared scalars that are not
/// observed.
pub(crate) fn companion_setup(
    sv: &SysView,
    ctx: &Ctx,
    obs: HashMap<String, Expr>,
    scalars: &HashSet<String>,
    co: bool,
) -> R<Companions> {
    let t: HashSet<String> = ["t".to_string()].into_iter().collect();
    let moving = moving_observed(&obs, &t);
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
    Ok(Companions {
        names: companion_names(sv, &moving),
        cols,
        inline,
        obs,
        ctx: ctx2,
        scalars: scalars.clone(),
        co,
        bands: HashMap::new(),
    })
}

fn refuse(o: &str) -> BandError {
    BandError(format!(
        "the extents of observed `{o}` (declared without a shape) cannot be read from its \
         definition; declare its shape"
    ))
}

/// The extents of observed `o`: declared, or inferred from its definition —
/// a `faq`'s output ranges, a `makearray`'s regions, or the first operand of
/// an elementwise definition that has extents. A definition whose extents
/// cannot be read at build time is refused: its companion has no shape.
fn obs_extents(
    o: &str,
    sv: &SysView,
    obs: &HashMap<String, Expr>,
    ctx: &Ctx,
    memo: &mut HashMap<String, Vec<i64>>,
    seen: &mut HashSet<String>,
) -> R<Vec<i64>> {
    if let Some(s) = ctx.shapes.get(o) {
        return Ok(s.clone());
    }
    let Some(var) = sv.variables.get(o) else {
        return Ok(Vec::new());
    };
    if var.shape.is_some() {
        return Ok(Vec::new()); // declared `[]`: a scalar
    }
    let Some(def) = obs.get(o) else {
        return Ok(Vec::new());
    };
    if let Some(e) = memo.get(o) {
        return Ok(e.clone());
    }
    if !seen.insert(o.to_string()) {
        return Err(BandError(format!(
            "cyclic observed definition through `{o}`"
        )));
    }
    let ext = expr_extents(def, o, sv, obs, ctx, memo, seen)?;
    memo.insert(o.to_string(), ext.clone());
    Ok(ext)
}

fn expr_extents(
    e: &Expr,
    o: &str,
    sv: &SysView,
    obs: &HashMap<String, Expr>,
    ctx: &Ctx,
    memo: &mut HashMap<String, Vec<i64>>,
    seen: &mut HashSet<String>,
) -> R<Vec<i64>> {
    let n = match e {
        Expr::Variable(v) => return obs_extents(v, sv, obs, ctx, memo, seen),
        Expr::Operator(n) => n,
        _ => return Ok(Vec::new()),
    };
    if n.op == "faq" {
        let mut out = Vec::new();
        for x in n.output_idx.iter().flatten() {
            let r = n
                .ranges
                .as_ref()
                .and_then(|m| m.get(x))
                .ok_or_else(|| refuse(o))?;
            let (lo, hi) = range_of(r, ctx)?;
            if lo != 1 {
                return Err(refuse(o));
            }
            out.push(hi);
        }
        return Ok(out);
    }
    if n.op == "makearray" {
        let regions = n
            .regions
            .as_ref()
            .filter(|r| !r.is_empty())
            .ok_or_else(|| refuse(o))?;
        let mut out = vec![0i64; regions[0].len()];
        for region in regions {
            for (d, r) in region.iter().enumerate() {
                out[d] = out[d].max(region_bounds(r)?.1);
            }
        }
        return Ok(out);
    }
    if n.op == "index" {
        return Ok(Vec::new());
    }
    if let Some(rn) = ref_node(e) {
        let Some(l) = &ctx.refs else {
            return Ok(Vec::new());
        };
        let name = rn.name.clone().unwrap_or_default();
        let rank = with_layer(&ctx.refs, || l.def(&name))?.rank;
        if rank == 0 {
            return Ok(Vec::new());
        }
        return Err(refuse(o));
    }
    for a in &n.args {
        let ext = expr_extents(a, o, sv, obs, ctx, memo, seen)?;
        if !ext.is_empty() {
            return Ok(ext);
        }
    }
    Ok(Vec::new())
}

fn faq_over(names: &[String], shape: &[i64], body: Expr) -> Expr {
    let ranges: HashMap<String, RangeSpec> = names
        .iter()
        .zip(shape)
        .map(|(n, s)| (n.clone(), RangeSpec::Interval([1, *s])))
        .collect();
    Expr::operator(ExpressionNode {
        op: "faq".into(),
        output_idx: Some(names.to_vec()),
        expr: Some(Box::new(body)),
        ranges: Some(ranges),
        ..Default::default()
    })
}

fn index_of(name: &str, idx: Vec<Expr>) -> Expr {
    let mut args = vec![Expr::Variable(name.to_string())];
    args.extend(idx);
    op("index", args)
}

/// The time bands of companion `o` (row `o` over its shape), computed once.
pub(crate) fn companion_bands(cs: &mut Companions, o: &str) -> R<Vec<Band>> {
    if let Some(b) = cs.bands.get(o) {
        return Ok(b.clone());
    }
    let shape = cs.ctx.shapes.get(o).cloned().unwrap_or_default();
    let def = cs.obs[o].clone();
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
    let mut inl = cs.inline.clone();
    inl.insert(o.to_string(), def);
    let bs = time_row_bands(cs, &rhs, &inl, &shape)?;
    cs.bands.insert(o.to_string(), bs.clone());
    Ok(bs)
}

/// The row index names of a companion row: the definition's own output
/// indices when it is a `faq` naming one per dimension, else `_r<d>` clear
/// of every name of the definition.
fn row_names_of(def: &Expr, k: usize) -> Vec<String> {
    if let Expr::Operator(n) = def {
        if n.op == "faq" {
            let oi = n.output_idx.clone().unwrap_or_default();
            let unique = oi.iter().collect::<HashSet<_>>().len() == oi.len();
            if oi.len() == k && unique {
                return oi;
            }
        }
    }
    let mut taken = HashSet::new();
    all_names(def, &mut taken);
    (1..=k)
        .map(|d| fresh(&mut taken, &format!("_r{d}"), "_"))
        .collect()
}

/// The time bands of one row (a state equation's right-hand side `rhs`, or
/// a companion row), all with column `t`: the direct bands in `t` and each
/// band of a moving-observed column composed with that observed's
/// companion, summed where they share a box ([`sum_same_box`]), in
/// column-name order.
pub(crate) fn time_row_bands(
    cs: &mut Companions,
    rhs: &Expr,
    inl: &HashMap<String, Expr>,
    shape_u: &[i64],
) -> R<Vec<Band>> {
    let mut obs_names = None;
    let rhs = with_layer(&cs.ctx.refs, || {
        inline_observed_ranked(
            rhs,
            inl,
            &mut obs_names,
            &cs.ctx.shapes,
            &cs.obs,
            &cs.scalars,
        )
    })
    .map_err(|e| BandError(e.0))?;
    let mut hits: Vec<String> = free_variables(&rhs)
        .into_iter()
        .filter(|v| v == "t" || cs.cols.contains(v))
        .collect();
    hits.sort();
    let mut out = Vec::new();
    for v in &hits {
        let mut raw = crate::operator_refs::ref_bands(&rhs, v, &cs.ctx, shape_u, cs.co)?;
        if !cs.co {
            for b in raw.iter_mut() {
                b.coef = placeholder();
            }
        }
        let bs: Vec<Band> = merge_bands(raw).iter().map(normalize_band).collect();
        if v == "t" {
            out.extend(bs);
            continue;
        }
        if companion_bands(cs, v)?.is_empty() {
            continue;
        }
        let shape_v = cs.ctx.shapes.get(v).cloned().unwrap_or_default();
        let name = cs.names[v].clone();
        for b in &bs {
            out.push(through_companion(b, &name, &shape_v, cs.co));
        }
    }
    Ok(merge_bands(sum_same_box(out, cs.co))
        .iter()
        .map(normalize_band)
        .collect())
}

/// Band `b` (column: a moving observed of shape `shape`, read at `b.cidx`)
/// composed with that observed's companion `name`: column `t`, coefficient
/// `coef · name[cidx…]` (the bare name for a scalar). A column index not
/// provably inside the observed's extent over the band's box guards the term
/// by `1 <= φ_d <= n_d`. Contracted dimensions become a scalar `faq` sum
/// over their ranges, the band's `where` its filter, so a time band keeps
/// the row's shape.
fn through_companion(b: &Band, name: &str, shape: &[i64], co: bool) -> Band {
    if !co {
        return Band {
            rows: b.rows.clone(),
            ridx: b.ridx.clone(),
            cidx: Vec::new(),
            coef: placeholder(),
            contracted: Vec::new(),
            guard: if b.contracted.is_empty() {
                b.guard.clone()
            } else {
                None
            },
        };
    }
    let read = if b.cidx.is_empty() {
        Expr::Variable(name.to_string())
    } else {
        index_of(name, b.cidx.clone())
    };
    let mut term = simp(&mul(vec![b.coef.clone(), read]));
    let bx = index_box(b);
    let mut conds = Vec::new();
    for (d, c) in b.cidx.iter().enumerate() {
        if let Some((lo, hi, _, _)) = index_bounds(c, &bx) {
            if lo >= 1 && hi <= shape[d] {
                continue;
            }
        }
        conds.push(op(">=", vec![c.clone(), Expr::Integer(1)]));
        conds.push(op("<=", vec![c.clone(), Expr::Integer(shape[d])]));
    }
    if let Some(g) = conj_list(conds) {
        term = op("ifelse", vec![g, term, Expr::Integer(0)]);
    }
    if b.contracted.is_empty() {
        return Band {
            rows: b.rows.clone(),
            ridx: b.ridx.clone(),
            cidx: Vec::new(),
            coef: term,
            contracted: Vec::new(),
            guard: b.guard.clone(),
        };
    }
    let ranges: HashMap<String, RangeSpec> = b
        .contracted
        .iter()
        .map(|(n, lo, hi)| (n.clone(), RangeSpec::Interval([*lo, *hi])))
        .collect();
    let sum = Expr::operator(ExpressionNode {
        op: "faq".into(),
        output_idx: Some(Vec::new()),
        reduce: Some("+".into()),
        expr: Some(Box::new(term)),
        ranges: Some(ranges),
        filter: b.guard.clone().map(Box::new),
        ..Default::default()
    });
    Band {
        rows: b.rows.clone(),
        ridx: b.ridx.clone(),
        cidx: Vec::new(),
        coef: sum,
        contracted: Vec::new(),
        guard: None,
    }
}

/// Bands of one row with the same box (rows, row names and `where`) summed
/// into one: an n-ary `+` of their coefficients in order, at the first
/// one's position.
fn sum_same_box(bs: Vec<Band>, co: bool) -> Vec<Band> {
    type Key = (Vec<(i64, i64)>, Vec<RIdx>, Option<String>);
    let mut order: Vec<Key> = Vec::new();
    let mut groups: BTreeMap<Key, Vec<usize>> = BTreeMap::new();
    for (i, b) in bs.iter().enumerate() {
        let k = (b.rows.clone(), b.ridx.clone(), b.guard.as_ref().map(skey));
        if !groups.contains_key(&k) {
            order.push(k.clone());
        }
        groups.entry(k).or_default().push(i);
    }
    let mut out = Vec::new();
    for k in order {
        let idx = &groups[&k];
        let mut b = bs[idx[0]].clone();
        if idx.len() > 1 {
            b.coef = if co {
                op("+", idx.iter().map(|&i| bs[i].coef.clone()).collect())
            } else {
                placeholder()
            };
        }
        out.push(b);
    }
    out
}

/// A companion's definition from its time bands: the scalar sum of their
/// terms, or one `faq` over the observed's whole shape whose body sums them,
/// each band's term gated by its region's membership (and its `where`)
/// unless it covers the whole shape. Row names are the bands' own when they
/// all share them, else `_r<d>` clear of every name the bands hold.
fn companion_definition(bs: &[Band], shape: &[i64]) -> Expr {
    let gate = |coef: Expr, conds: Vec<Expr>| match conj_list(conds) {
        None => coef,
        Some(g) => op("ifelse", vec![g, coef, Expr::Integer(0)]),
    };
    let sum = |ts: Vec<Expr>| {
        if ts.len() == 1 {
            ts.into_iter().next().unwrap()
        } else {
            op("+", ts)
        }
    };
    if shape.is_empty() {
        let ts = bs
            .iter()
            .map(|b| {
                gate(
                    b.coef.clone(),
                    b.guard.as_ref().map(conjuncts).unwrap_or_default(),
                )
            })
            .collect();
        return sum(ts);
    }
    let k = shape.len();
    let first = &bs[0].ridx;
    let all_named: Option<Vec<String>> = first
        .iter()
        .map(|r| match r {
            RIdx::Name(n) => Some(n.clone()),
            RIdx::Fixed(_) => None,
        })
        .collect();
    let names: Vec<String> = match all_named {
        Some(ns)
            if ns.iter().collect::<HashSet<_>>().len() == ns.len()
                && bs.iter().all(|b| &b.ridx == first) =>
        {
            ns
        }
        _ => {
            let mut taken = HashSet::new();
            for b in bs {
                for r in &b.ridx {
                    if let RIdx::Name(n) = r {
                        taken.insert(n.clone());
                    }
                }
                all_names(&b.coef, &mut taken);
                if let Some(g) = &b.guard {
                    all_names(g, &mut taken);
                }
            }
            (1..=k)
                .map(|d| fresh(&mut taken, &format!("_r{d}"), "_"))
                .collect()
        }
    };
    let mut ts = Vec::new();
    for b in bs {
        let mut m = HashMap::new();
        let mut conds = Vec::new();
        for d in 0..k {
            let (lo, hi) = b.rows[d];
            let x = Expr::Variable(names[d].clone());
            match &b.ridx[d] {
                RIdx::Name(r) => {
                    if *r != names[d] {
                        m.insert(r.clone(), x.clone());
                    }
                    if lo > 1 {
                        conds.push(op(">=", vec![x.clone(), Expr::Integer(lo)]));
                    }
                    if hi < shape[d] {
                        conds.push(op("<=", vec![x.clone(), Expr::Integer(hi)]));
                    }
                }
                RIdx::Fixed(c) => conds.push(op("==", vec![x.clone(), Expr::Integer(*c)])),
            }
        }
        let coef = if m.is_empty() {
            b.coef.clone()
        } else {
            subst_free(&b.coef, &m)
        };
        if let Some(g) = &b.guard {
            let g = if m.is_empty() {
                g.clone()
            } else {
                subst_free(g, &m)
            };
            conds.extend(conjuncts(&g));
        }
        ts.push(gate(coef, conds));
    }
    faq_over(&names, shape, sum(ts))
}

/// The companions the band list `bands` reads, directly or through other
/// companions' definitions, as `(name, observed, definition)` in the
/// observed's name order — what the block's `companions` field and every
/// derived document of the list carry (spec §4.2, §8.1). Empty unless some
/// coefficient (or `where`) reads a companion name, which no source name
/// can be.
pub(crate) fn time_companions(sv: &SysView, bands: &[&Band]) -> R<Vec<(String, String, Expr)>> {
    let mut out = Vec::new();
    if bands.is_empty() {
        return Ok(out);
    }
    let obs = crate::system::observed_defs(sv)?;
    let ctx = crate::system::ctx_of(sv)?;
    let recs = crate::recurrence::recurrences(sv, &obs, &ctx)?;
    let mut obs2 = obs.clone();
    for r in &recs {
        obs2.remove(&r.name);
    }
    let t: HashSet<String> = ["t".to_string()].into_iter().collect();
    let moving = moving_observed(&obs2, &t);
    if moving.is_empty() {
        return Ok(out);
    }
    let names = companion_names(sv, &moving);
    let byname: HashMap<&String, &String> = names.iter().map(|(o, n)| (n, o)).collect();
    let mut read: Vec<String> = Vec::new();
    for b in bands {
        for x in std::iter::once(&b.coef).chain(b.guard.iter()) {
            for n in free_variables(x) {
                if byname.contains_key(&n) {
                    read.push(n);
                }
            }
        }
    }
    if read.is_empty() {
        return Ok(out);
    }
    let scalars = declared_scalars(sv, &obs);
    let mut cs = companion_setup(sv, &ctx, obs2, &scalars, true)?;
    let mut defs: BTreeMap<String, Expr> = BTreeMap::new();
    while let Some(n) = read.pop() {
        if defs.contains_key(&n) {
            continue;
        }
        let o = byname[&n].clone();
        let shape = cs.ctx.shapes.get(&o).cloned().unwrap_or_default();
        let d = companion_definition(&companion_bands(&mut cs, &o)?, &shape);
        for m in free_variables(&d) {
            if byname.contains_key(&m) && !defs.contains_key(&m) {
                read.push(m);
            }
        }
        defs.insert(n, d);
    }
    let mut sorted: Vec<&String> = moving.iter().collect();
    sorted.sort();
    for o in sorted {
        let n = &names[o];
        if let Some(d) = defs.remove(n) {
            out.push((n.clone(), o.clone(), d));
        }
    }
    Ok(out)
}

/// The declared scalars that are not observed: an elementwise observed read
/// at a cell reads them unchanged.
pub(crate) fn declared_scalars(sv: &SysView, obs: &HashMap<String, Expr>) -> HashSet<String> {
    sv.variables
        .iter()
        .filter(|(n, v)| v.shape.as_ref().is_none_or(|s| s.is_empty()) && !obs.contains_key(*n))
        .map(|(n, _)| n.clone())
        .collect()
}

/// The companions as observed of a derived model (spec §8.1): each declared
/// like its observed (`{"type": "unknown"}`, with the observed's declared
/// `shape` when it has one) and defined by a bare-LHS equation, after the
/// source's equations.
pub(crate) fn add_companions(
    model: &mut Map<String, Value>,
    sv: &SysView,
    comps: &[(String, String, Expr)],
) {
    if let Some(Value::Object(vars)) = model.get_mut("variables") {
        for (n, o, _) in comps {
            let mut d = Map::new();
            d.insert("type".into(), json!("unknown"));
            if let Some(shp) = sv.variables[o].shape.as_ref().filter(|s| !s.is_empty()) {
                d.insert("shape".into(), json!(shp));
            }
            vars.insert(n.clone(), Value::Object(d));
        }
    }
    if let Some(Value::Array(eqs)) = model.get_mut("equations") {
        for (n, _, d) in comps {
            eqs.push(json!({"lhs": n, "rhs": ser_expr(d)}));
        }
    }
}

/// The block's `companions` field (spec §4.2): `[{name, of, rhs}]`.
pub(crate) fn companions_field(comps: &[(String, String, Expr)]) -> Value {
    Value::Array(
        comps
            .iter()
            .map(|(n, o, d)| json!({"name": n, "of": o, "rhs": ser_expr(d)}))
            .collect(),
    )
}
