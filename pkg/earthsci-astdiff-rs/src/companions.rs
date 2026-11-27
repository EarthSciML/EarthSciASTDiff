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
use earthsci_ast::RegionBound;
use earthsci_ast::{Expr, ExpressionNode};
use serde_json::{json, Map, Value};

use crate::bands::{
    merge_bands, normalize_band, placeholder, range_of, region_bounds, subst_free, Band, BandError,
    Ctx,
};
use crate::branches::simp;
use crate::clip::{index_bounds, index_box, RIdx};
use crate::helpers::{mul, op, ser_expr, skey};
use crate::inline::{all_names, inline_observed_aligned};
use crate::jacobian::declared_scalars;
use crate::operator_refs::{ref_node, with_layer};
use crate::reductions::{conj_list, conjuncts, fresh};
use crate::system::{moving_observed, subscript_observed, SysView};
use std::cell::RefCell;

use crate::branches::SimpMemo;
use crate::helpers::OccIndex;
use crate::jacobian::{FirstOrder, Wrt};
use crate::recurrence::{active, recurrences};
use crate::second_order::{coef_bands, JacEntry2};

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
    /// The observed definitions, recurrences left out.
    obs: HashMap<String, Expr>,
    /// Every observed definition, recurrences included (ranks, extents).
    rank_obs: HashMap<String, Expr>,
    pub(crate) ctx: Ctx,
    scalars: HashSet<String>,
    co: bool,
    bands: HashMap<String, Vec<Band>>,
    /// The companions whose bands are being built.
    pending: HashSet<String>,
}

/// The companion name of each observed in `moving`, in name order:
/// `_dt_<o>` (`P._dt_<v>` for a scoped `P.v`, so it lives in `o`'s own
/// component), with `_` appended until clear of every variable, component
/// path and index-set name of the source and of the companion names before
/// it.
pub(crate) fn companion_names(sv: &SysView, moving: &HashSet<String>) -> HashMap<String, String> {
    let mut taken: HashSet<String> = sv.variables.keys().cloned().collect();
    taken.extend(sv.index_sets.keys().cloned());
    // a companion named like a component path would be both a variable and
    // a subsystem of a derived model
    component_paths(&mut taken, sv.variables.keys());
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

/// Every component path of the scoped names `names` (`A` and `A.B` of
/// `A.B.v`), added to `taken`.
pub(crate) fn component_paths<'a>(
    taken: &mut HashSet<String>,
    names: impl IntoIterator<Item = &'a String>,
) {
    for n in names {
        for (i, ch) in n.char_indices() {
            if ch == '.' {
                taken.insert(n[..i].to_string());
            }
        }
    }
}

/// The companion layout of the time pass over the view. `obs` is every
/// observed definition with the recurrences left out (the pass never
/// inlines a recurrence), `rank_obs` every observed definition, recurrences
/// included (what an elementwise read at a cell and an undeclared shape are
/// ranked from); `scalars` the declared scalars that are not observed.
pub(crate) fn companion_setup(
    sv: &SysView,
    ctx: &Ctx,
    obs: HashMap<String, Expr>,
    rank_obs: HashMap<String, Expr>,
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
        let ext = obs_extents(o, sv, &rank_obs, ctx, &mut memo, &mut HashSet::new())?;
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
        rank_obs,
        ctx: ctx2,
        scalars: scalars.clone(),
        co,
        bands: HashMap::new(),
        pending: HashSet::new(),
    })
}

fn refuse(o: &str) -> BandError {
    BandError(format!(
        "the extents of observed `{o}` (declared without a shape) cannot be read from its \
         definition; declare its shape"
    ))
}

/// The extents of observed `o`: declared, or inferred from its definition —
/// a `faq`'s output ranges, a `makearray`'s regions, an array-valued
/// reference's body extents, or the one extent list every array operand of
/// an elementwise definition shares. A definition whose extents cannot be
/// read at build time (operands of different extents, whose union needs
/// index-set names; a partial `index` read) is refused: its companion has
/// no shape.
pub(crate) fn obs_extents(
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
    // a named intermediate (a time pair's `a2` steps, §4.4) is no variable
    // of the view: declared without a shape, its extents are its
    // definition's
    match sv.variables.get(o) {
        None if !obs.contains_key(o) => return Ok(Vec::new()),
        Some(var) if var.shape.is_some() => return Ok(Vec::new()), // declared `[]`
        _ => {}
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
        // a read at a cell is a scalar; a read with fewer indices than its
        // array's rank is an array slice, whose extents are not read here
        if let Some(Expr::Variable(a)) = n.args.first() {
            let ext = obs_extents(a, sv, obs, ctx, memo, seen)?;
            if n.args.len() - 1 < ext.len() {
                return Err(refuse(o));
            }
        }
        return Ok(Vec::new());
    }
    if let Some(rn) = ref_node(e) {
        let Some(l) = &ctx.refs else {
            return Ok(Vec::new());
        };
        let name = rn.name.clone().unwrap_or_default();
        let d = with_layer(&ctx.refs, || l.def(&name))?;
        if d.rank == 0 {
            return Ok(Vec::new());
        }
        let bindings = rn.bindings.clone().unwrap_or_default();
        let (ext, org) =
            with_layer(&ctx.refs, || l.ref_extents(&d, &bindings, ctx)).map_err(|_| refuse(o))?;
        if org.iter().any(|&x| x != 1) {
            return Err(refuse(o));
        }
        return Ok(ext);
    }
    let mut out: Vec<i64> = Vec::new();
    for a in &n.args {
        let ext = expr_extents(a, o, sv, obs, ctx, memo, seen)?;
        if ext.is_empty() {
            continue;
        }
        if !out.is_empty() && out != ext {
            return Err(refuse(o));
        }
        out = ext;
    }
    Ok(out)
}

pub(crate) fn faq_over(names: &[String], shape: &[i64], body: Expr) -> Expr {
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

pub(crate) fn index_of(name: &str, idx: Vec<Expr>) -> Expr {
    let mut args = vec![Expr::Variable(name.to_string())];
    args.extend(idx);
    op("index", args)
}

/// The time bands of companion `o` (row `o` over its shape), computed once.
pub(crate) fn companion_bands(cs: &mut Companions, o: &str) -> R<Vec<Band>> {
    if let Some(b) = cs.bands.get(o) {
        return Ok(b.clone());
    }
    // moving observed that read each other (an algebraic loop that is not a
    // recognized recurrence) have no companion: refused, never followed
    if !cs.pending.insert(o.to_string()) {
        return Err(BandError(format!(
            "cyclic observed definition through `{o}`"
        )));
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
    let bs = time_row_bands(cs, &rhs, o, &inl, &shape)?;
    cs.pending.remove(o);
    cs.bands.insert(o.to_string(), bs.clone());
    Ok(bs)
}

/// The row index names of a companion row: the definition's own output
/// indices when it is a `faq` naming one per dimension, else `_r<d>` clear
/// of every name of the definition.
pub(crate) fn row_names_of(def: &Expr, k: usize) -> Vec<String> {
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
    u: &str,
    inl: &HashMap<String, Expr>,
    shape_u: &[i64],
) -> R<Vec<Band>> {
    let mut obs_names = None;
    let rhs = crate::jacobian::inline_row(
        rhs,
        u,
        inl,
        &mut obs_names,
        &cs.ctx,
        &cs.rank_obs,
        &cs.scalars,
    )?;
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
            if b.join.is_some() {
                return Err(BandError(
                    "a time companion read through a joined faq node is not supported".into(),
                ));
            }
            out.push(through_companion(b, v, &name, &shape_v, cs.co)?);
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
pub(crate) fn through_companion(b: &Band, o: &str, name: &str, shape: &[i64], co: bool) -> R<Band> {
    check_read_rank(b, o, shape)?;
    if !co {
        return Ok(Band {
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
            join: None,
        });
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
        return Ok(Band {
            rows: b.rows.clone(),
            ridx: b.ridx.clone(),
            cidx: Vec::new(),
            coef: term,
            contracted: Vec::new(),
            guard: b.guard.clone(),
            join: None,
        });
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
    Ok(Band {
        rows: b.rows.clone(),
        ridx: b.ridx.clone(),
        cidx: Vec::new(),
        coef: sum,
        contracted: Vec::new(),
        guard: None,
        join: None,
    })
}

/// A band's column index must read every dimension of the moving observed
/// `o`, whose extents are `shape`: a read of another rank has no companion
/// cell (an undeclared observed whose extents were read wrong, or a
/// malformed read), refused rather than read out of bounds.
pub(crate) fn check_read_rank(b: &Band, o: &str, shape: &[i64]) -> R<()> {
    if b.cidx.len() != shape.len() {
        return Err(BandError(format!(
            "observed `{o}` of rank {} is read with {} indices; declare its shape",
            shape.len(),
            b.cidx.len()
        )));
    }
    Ok(())
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
    // each band: its box over the shape and its term (row names unified,
    // gated by its `where` only)
    let mut boxes: Vec<Vec<(i64, i64)>> = Vec::new();
    let mut terms: Vec<Expr> = Vec::new();
    for b in bs {
        let mut m = HashMap::new();
        let mut bx = Vec::new();
        for d in 0..k {
            match &b.ridx[d] {
                RIdx::Name(r) => {
                    if *r != names[d] {
                        m.insert(r.clone(), Expr::Variable(names[d].clone()));
                    }
                    bx.push(b.rows[d]);
                }
                RIdx::Fixed(c) => bx.push((*c, *c)),
            }
        }
        let coef = if m.is_empty() {
            b.coef.clone()
        } else {
            subst_free(&b.coef, &m)
        };
        let conds = match &b.guard {
            None => Vec::new(),
            Some(g) => {
                let g = if m.is_empty() {
                    g.clone()
                } else {
                    subst_free(g, &m)
                };
                conjuncts(&g)
            }
        };
        boxes.push(bx);
        terms.push(gate(coef, conds));
    }
    let faq_box = |bx: &[(i64, i64)], body: Expr| -> Expr {
        let ranges: HashMap<String, RangeSpec> = names
            .iter()
            .zip(bx)
            .map(|(n, (lo, hi))| (n.clone(), RangeSpec::Interval([*lo, *hi])))
            .collect();
        Expr::operator(ExpressionNode {
            op: "faq".into(),
            output_idx: Some(names.clone()),
            expr: Some(Box::new(body)),
            ranges: Some(ranges),
            ..Default::default()
        })
    };
    let full: Vec<(i64, i64)> = shape.iter().map(|n| (1, *n)).collect();
    if boxes.iter().all(|bx| *bx == full) {
        return faq_box(&full, sum(terms));
    }
    // A region narrower than the shape: a `makearray`, never a whole-shape
    // gate (a reader built per cell, a lane program, needs the region). The
    // boxes are cut at every box's bounds into disjoint cells, each the sum
    // of the bands that cover it, over zero where none does
    // (companions.jl `_companion_definition`).
    let cuts: Vec<Vec<i64>> = (0..k)
        .map(|d| {
            let mut c = vec![1, shape[d] + 1];
            for bx in &boxes {
                c.push(bx[d].0);
                c.push(bx[d].1 + 1);
            }
            c.sort();
            c.dedup();
            c
        })
        .collect();
    let mut regions = Vec::new();
    let mut values = Vec::new();
    let mut ncells: i64 = 0;
    // cells in lexicographic order, the first dimension slowest
    let mut pos = vec![0usize; k];
    'cells: loop {
        let cell: Vec<(i64, i64)> = (0..k)
            .map(|d| (cuts[d][pos[d]], cuts[d][pos[d] + 1] - 1))
            .collect();
        let cover: Vec<usize> = boxes
            .iter()
            .enumerate()
            .filter(|(_, bx)| (0..k).all(|d| bx[d].0 <= cell[d].0 && cell[d].1 <= bx[d].1))
            .map(|(i, _)| i)
            .collect();
        if !cover.is_empty() {
            ncells += cell.iter().map(|(lo, hi)| hi - lo + 1).product::<i64>();
            regions.push(cell.iter().map(|(lo, hi)| [*lo, *hi]).collect::<Vec<_>>());
            values.push(faq_box(
                &cell,
                sum(cover.iter().map(|&i| terms[i].clone()).collect()),
            ));
        }
        let mut d = k;
        loop {
            if d == 0 {
                break 'cells;
            }
            d -= 1;
            pos[d] += 1;
            if pos[d] + 1 < cuts[d].len() {
                break;
            }
            pos[d] = 0;
        }
    }
    if ncells < shape.iter().product::<i64>() {
        regions.insert(0, shape.iter().map(|n| [1, *n]).collect());
        values.insert(0, Expr::Integer(0));
    }
    region_array(regions, values, shape)
}

/// An array over `shape` given by `makearray` regions: one `faq` over the
/// whole extent whose body reads the `makearray` at its own cell, so the
/// definition stays an array aggregate (a lane program builds an observed
/// only from one) while each region keeps its own box. Its names are
/// `_m<d>`, `_` appended until clear of every name of the regions' values
/// (companions.jl `_region_array`).
pub(crate) fn region_array(regions: Vec<Vec<[i64; 2]>>, values: Vec<Expr>, shape: &[i64]) -> Expr {
    let mut taken = HashSet::new();
    for v in &values {
        all_names(v, &mut taken);
    }
    let names: Vec<String> = (1..=shape.len())
        .map(|d| fresh(&mut taken, &format!("_m{d}"), "_"))
        .collect();
    let ma = makearray_of(regions, values);
    let mut args = vec![ma];
    args.extend(names.iter().map(|n| Expr::Variable(n.clone())));
    faq_over(&names, shape, op("index", args))
}

/// The companions the band list `bands` reads, directly or through other
/// companions' definitions, as `(name, observed, definition)` in the
/// observed's name order — what the block's `intermediates` field and every
/// derived document of the list carry (spec §4.2, §8.1). Empty unless some
/// coefficient (or `where`) reads a companion name, which no source name
/// can be.
pub(crate) fn time_companions(
    sv: &SysView,
    bands: &[&Band],
    extra: &HashMap<String, Expr>,
) -> R<Vec<(String, String, Expr)>> {
    let mut out = Vec::new();
    if bands.is_empty() {
        return Ok(out);
    }
    // Every companion name's last segment starts with `_dt_`: a list none of
    // whose coefficients reads such a name (every block of another axis)
    // reads no companion, and the analysis below is skipped.
    let mut cand: Vec<String> = Vec::new();
    for b in bands {
        for x in std::iter::once(&b.coef).chain(b.guard.iter()) {
            for n in free_variables(x) {
                let last = n.rsplit('.').next().unwrap_or(&n);
                if last.starts_with("_dt_") {
                    cand.push(n);
                }
            }
        }
    }
    if cand.is_empty() {
        return Ok(out);
    }
    let obs = crate::system::observed_defs(sv)?;
    let ctx = crate::system::ctx_of(sv)?;
    let recs = crate::recurrence::recurrences(sv, &obs, &ctx)?;
    let mut obs2 = obs.clone();
    for r in &recs {
        obs2.remove(&r.name);
    }
    // a time pair's `a2` intermediates are observed of the time pass (§4.4)
    obs2.extend(extra.iter().map(|(k, v)| (k.clone(), v.clone())));
    let mut obs = obs;
    obs.extend(extra.iter().map(|(k, v)| (k.clone(), v.clone())));
    let t: HashSet<String> = ["t".to_string()].into_iter().collect();
    let moving = moving_observed(&obs2, &t);
    if moving.is_empty() {
        return Ok(out);
    }
    let names = companion_names(sv, &moving);
    let byname: HashMap<&String, &String> = names.iter().map(|(o, n)| (n, o)).collect();
    let mut read: Vec<String> = cand
        .into_iter()
        .filter(|n| byname.contains_key(n))
        .collect();
    if read.is_empty() {
        return Ok(out);
    }
    let scalars = declared_scalars(sv, &obs);
    let mut cs = companion_setup(sv, &ctx, obs2, obs.clone(), &scalars, true)?;
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

/// The companions as observed of a derived model (spec §8.1): each declared
/// like its observed (`{"type": "unknown"}`, with the observed's declared
/// `shape` when it has one) and defined by a bare-LHS equation, after the
/// source's equations.
pub(crate) fn add_companions(
    model: &mut Map<String, Value>,
    sv: &SysView,
    comps: &[(String, String, Expr, usize)],
) {
    if let Some(Value::Object(vars)) = model.get_mut("variables") {
        for (n, o, _, extra) in comps {
            let mut d = Map::new();
            d.insert("type".into(), json!("unknown"));
            // a named step of another axis (§4.6) is declared without a
            // shape: its definition gives its extents
            if *extra == 0 {
                if let Some(shp) = sv
                    .variables
                    .get(o)
                    .and_then(|v| v.shape.as_ref())
                    .filter(|s| !s.is_empty())
                {
                    d.insert("shape".into(), json!(shp));
                }
            }
            vars.insert(n.clone(), Value::Object(d));
        }
    }
    if let Some(Value::Array(eqs)) = model.get_mut("equations") {
        for (n, _, d, _) in comps {
            eqs.push(json!({"lhs": n, "rhs": ser_expr(d)}));
        }
    }
}

/// The block's `intermediates` field (spec §4, §4.2, §4.6): `[{name, of,
/// rhs}]`.
pub(crate) fn companions_field(comps: &[(String, String, Expr, usize)]) -> Value {
    Value::Array(
        comps
            .iter()
            .map(|(n, o, d, _)| json!({"name": n, "of": o, "rhs": ser_expr(d)}))
            .collect(),
    )
}

// ── second-order pairs `["time", a2]` (spec §4.4) ───────────────────────────

/// A time pair taken with `t` SECOND (companions.jl `_time_pair_bands`).
/// With every branch frozen (spec §3.2) each branch is smooth, so the mixed
/// partials commute: ∂²f/∂t∂x = ∂/∂t (∂f/∂x). Each first-order entry of the
/// `a2` axis has its coefficient's time bands taken by the time pass's own
/// rule — the observed that move with `t` are columns composed with their
/// companions, never inlined. The `a2` pass keeps its named steps (§4.6),
/// and its named intermediates are observed of the time pass: one that
/// moves with `t` is a column composed with its own companion. `None` when a
/// recurrence is
/// active on either axis: that pair keeps the inlining path, as the time
/// axis does.
#[allow(clippy::too_many_arguments)]
pub(crate) fn time_pair_bands(
    sv: &SysView,
    ctx: &Ctx,
    obs: &HashMap<String, Expr>,
    a2: Wrt,
    forcing: Option<&[String]>,
    memo: &mut FirstOrder,
    t2: &[String],
) -> R<Option<Vec<JacEntry2>>> {
    let recs = recurrences(sv, obs, ctx)?;
    if !recs.is_empty() {
        let mut s: HashSet<String> = t2.iter().cloned().collect();
        s.insert("t".to_string());
        if !active(&recs, obs, &s).is_empty() {
            return Ok(None);
        }
    }
    // the a2 pass keeps its named steps (§4.6): its coefficients read named
    // intermediates, which are observed of the time pass here, each moving
    // with `t` a column composed with its own companion — never inlined
    let _ = &memo;
    let e2 = crate::jacobian::extended_pass(sv, ctx, obs, a2, forcing, &[], true)?.rows;
    let eb: Vec<&Band> = e2.iter().map(|en| &en.band).collect();
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
    let mut cs = companion_setup(sv, ctx, nr, rank, &declared_scalars(sv, obs), true)?;
    let mut out = Vec::new();
    for en in e2.iter() {
        for b in time_coef_bands(&mut cs, &en.band)? {
            out.push(JacEntry2 {
                u: en.u.clone(),
                v: "t".to_string(),
                w: en.v.clone(),
                band: b,
                ncol: 0,
            });
        }
    }
    Ok(Some(out))
}

/// The time bands of a first-order band's coefficient, broadcast over its
/// rows (its contracted names bound too): [`time_row_bands`] on a
/// coefficient. Every band keeps the first-order column indices, which
/// become the pair's `col2_idx`; a companion read whose index is not
/// provably inside the observed's extent is gated as on the time axis. A
/// site under a nested sum adds contracted dimensions (spec §4.4), which
/// stay contracted.
fn time_coef_bands(cs: &mut Companions, b: &Band) -> R<Vec<Band>> {
    let mut obs_names = None;
    let coef = with_layer(&cs.ctx.refs, || {
        inline_observed_aligned(
            &b.coef,
            &cs.inline,
            &mut obs_names,
            &cs.ctx.shapes,
            &cs.rank_obs,
            &cs.scalars,
            &cs.ctx.axes,
        )
    })
    .map_err(|e| BandError(e.0))?;
    let mut b0 = b.clone();
    b0.coef = coef;
    let n = b.cidx.len();
    let mut hits: Vec<String> = free_variables(&b0.coef)
        .into_iter()
        .filter(|v| v == "t" || cs.cols.contains(v))
        .collect();
    hits.sort();
    let mut out = Vec::new();
    for v in &hits {
        let occ = RefCell::new(OccIndex::new(v));
        let mut sm = SimpMemo::default();
        let ctx = cs.ctx.clone();
        let bs: Vec<Band> = merge_bands(coef_bands(&b0, v, &ctx, &occ, &mut sm)?)
            .iter()
            .map(normalize_band)
            .collect();
        if v == "t" {
            out.extend(bs);
            continue;
        }
        if companion_bands(cs, v)?.is_empty() {
            continue;
        }
        let shape_v = cs.ctx.shapes.get(v).cloned().unwrap_or_default();
        let name = cs.names[v].clone();
        for x in &bs {
            out.push(pair_through_companion(x, n, v, &name, &shape_v)?);
        }
    }
    Ok(merge_bands(sum_same_box2(out))
        .iter()
        .map(normalize_band)
        .collect())
}

/// Band `x` of a coefficient (column indices: the first-order ones, then the
/// moving observed's `φ`) composed with that observed's companion `name`:
/// coefficient `coef · name[φ…]`, gated by `1 <= φ_d <= n_d` where `φ_d` is
/// not provably in extent over the band's box, column indices back to the
/// first `n`.
fn pair_through_companion(x: &Band, n: usize, o: &str, name: &str, shape: &[i64]) -> R<Band> {
    let phi = &x.cidx[n..];
    if phi.len() != shape.len() {
        return Err(BandError(format!(
            "observed `{o}` of rank {} is read with {} indices; declare its shape",
            shape.len(),
            phi.len()
        )));
    }
    let read = if phi.is_empty() {
        Expr::Variable(name.to_string())
    } else {
        index_of(name, phi.to_vec())
    };
    let mut term = simp(&mul(vec![x.coef.clone(), read]));
    let bx = index_box(x);
    let mut conds = Vec::new();
    for (d, c) in phi.iter().enumerate() {
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
    let mut out = x.clone();
    out.cidx = x.cidx[..n].to_vec();
    out.coef = term;
    Ok(out)
}

/// [`sum_same_box`] for pair bands: the box also holds the column indices
/// and the contracted dimensions.
fn sum_same_box2(bs: Vec<Band>) -> Vec<Band> {
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
        if idx.len() > 1 {
            b.coef = op("+", idx.iter().map(|&i| bs[i].coef.clone()).collect());
        }
        out.push(b);
    }
    out
}

/// A `makearray` node of integer regions (`[lo, hi]` per dimension) and
/// their values.
pub(crate) fn makearray_of(regions: Vec<Vec<[i64; 2]>>, values: Vec<Expr>) -> Expr {
    Expr::operator(ExpressionNode {
        op: "makearray".into(),
        regions: Some(
            regions
                .into_iter()
                .map(|r| {
                    r.into_iter()
                        .map(|[lo, hi]| [RegionBound::Int(lo), RegionBound::Int(hi)])
                        .collect()
                })
                .collect(),
        ),
        values: Some(values),
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_read_of_another_rank_is_refused() {
        let b = Band {
            rows: vec![(1, 3)],
            ridx: vec![RIdx::Name("i".into())],
            cidx: vec![Expr::Variable("i".into())],
            coef: Expr::Integer(1),
            contracted: Vec::new(),
            guard: None,
            join: None,
        };
        assert!(through_companion(&b, "jr", "_dt_jr", &[3, 2], true).is_err());
        assert!(through_companion(&b, "jr", "_dt_jr", &[3], true).is_ok());
        // and in a time pair: the first `n` column indices are the pair's own
        assert!(pair_through_companion(&b, 0, "jr", "_dt_jr", &[3, 2]).is_err());
        assert!(pair_through_companion(&b, 0, "jr", "_dt_jr", &[3])
            .unwrap()
            .cidx
            .is_empty());
    }

    #[test]
    fn component_paths_are_taken() {
        let mut taken = HashSet::new();
        component_paths(&mut taken, &["P._dt_v.x".to_string(), "w".to_string()]);
        let want: HashSet<String> = ["P", "P._dt_v"].iter().map(|s| s.to_string()).collect();
        assert_eq!(taken, want);
    }
}
