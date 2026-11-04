//! Static region clipping (the reference's clip.jl): decidable membership
//! guards decided exactly by splitting band row ranges at their affine
//! breakpoints; conservative everywhere else.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use earthsci_ast::Expr;

use crate::bands::Band;
use crate::hashcons::map_children_shared;
use crate::helpers::op;

const CMP_OPS: &[&str] = &["<", "<=", ">", ">="];

/// `e` as (α, β) over α·name + β, integer arithmetic only.
fn affine(e: &Expr, name: &str) -> Option<(i64, i64)> {
    match e {
        Expr::Variable(v) => {
            if v == name {
                Some((1, 0))
            } else {
                None
            }
        }
        Expr::Integer(i) => Some((0, *i)),
        Expr::Number(x) => {
            if x.fract() == 0.0 {
                Some((0, *x as i64))
            } else {
                None
            }
        }
        Expr::Operator(n) => {
            let a = &n.args;
            match n.op.as_str() {
                "+" => {
                    let (mut al, mut be) = (0i64, 0i64);
                    for x in a {
                        let (ta, tb) = affine(x, name)?;
                        al += ta;
                        be += tb;
                    }
                    Some((al, be))
                }
                "-" if a.len() == 1 => {
                    let (ta, tb) = affine(&a[0], name)?;
                    Some((-ta, -tb))
                }
                "-" if a.len() == 2 => {
                    let (t1a, t1b) = affine(&a[0], name)?;
                    let (t2a, t2b) = affine(&a[1], name)?;
                    Some((t1a - t2a, t1b - t2b))
                }
                "*" => {
                    let (mut al, mut be) = (0i64, 1i64);
                    for x in a {
                        let (ta, tb) = affine(x, name)?;
                        if ta == 0 {
                            al *= tb;
                            be *= tb;
                        } else if al == 0 && be != 0 {
                            al = be * ta;
                            be *= tb;
                        } else {
                            return None;
                        }
                    }
                    Some((al, be))
                }
                _ => None,
            }
        }
    }
}

fn sole_name<'a>(e: &Expr, names: &'a [String]) -> Option<&'a str> {
    fn walk<'a>(x: &Expr, names: &'a [String], found: &mut Option<&'a str>, ok: &mut bool) {
        if let Expr::Variable(v) = x {
            if let Some(nm) = names.iter().find(|n| *n == v) {
                match found {
                    None => *found = Some(nm.as_str()),
                    Some(f) => {
                        if *f != v {
                            *ok = false;
                        }
                    }
                }
            }
        }
        if let Expr::Operator(n) = x {
            n.for_each_child(&mut |c: &Expr| walk(c, names, found, ok));
        }
    }
    let mut found = None;
    let mut ok = true;
    walk(e, names, &mut found, &mut ok);
    if ok {
        found
    } else {
        None
    }
}

/// Julia's `fld` (floor division). NOT `div_euclid`, which differs for a
/// negative divisor (fld(7,-2) = -4, div_euclid = -3) — a reversed
/// comparison's slope is negative.
fn fld(a: i64, b: i64) -> i64 {
    let q = a / b;
    if a % b != 0 && ((a < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

fn cmp_truth(o: &str, al: i64, be: i64, i: i64) -> bool {
    let v = al * i + be;
    match o {
        "<" => v < 0,
        "<=" => v <= 0,
        ">" => v > 0,
        _ => v >= 0,
    }
}

fn breakpoints(bp: &mut BTreeMap<String, BTreeSet<i64>>, e: &Expr, names: &[String]) {
    let Expr::Operator(n) = e else { return };
    if CMP_OPS.contains(&n.op.as_str()) && n.args.len() == 2 {
        if let Some(nm) = sole_name(e, names) {
            let diff = op("-", vec![n.args[0].clone(), n.args[1].clone()]);
            if let Some((al, be)) = affine(&diff, nm) {
                if al != 0 {
                    let q = fld(-be, al);
                    for c in [q, q + 1] {
                        if cmp_truth(&n.op, al, be, c - 1) != cmp_truth(&n.op, al, be, c) {
                            bp.entry(nm.to_string()).or_default().insert(c);
                        }
                    }
                }
            }
        }
    }
    n.for_each_child(&mut |c: &Expr| breakpoints(bp, c, names));
}

fn uniform(cond: &Expr, env: &BTreeMap<String, (i64, i64)>) -> Option<bool> {
    let Expr::Operator(n) = cond else { return None };
    if CMP_OPS.contains(&n.op.as_str()) && n.args.len() == 2 {
        let keys: Vec<String> = env.keys().cloned().collect();
        let nm = sole_name(cond, &keys)?;
        let (lo, hi) = *env.get(nm)?;
        let (al, be) = affine(&op("-", vec![n.args[0].clone(), n.args[1].clone()]), nm)?;
        let (vlo, vhi) = (al * lo + be, al * hi + be);
        let (a, b) = (vlo.min(vhi), vlo.max(vhi));
        let f = |x: i64| match n.op.as_str() {
            "<" => x < 0,
            "<=" => x <= 0,
            ">" => x > 0,
            _ => x >= 0,
        };
        return if f(a) == f(b) { Some(f(a)) } else { None };
    }
    match n.op.as_str() {
        "and" => {
            let mut anynone = false;
            for x in &n.args {
                match uniform(x, env) {
                    Some(false) => return Some(false),
                    None => anynone = true,
                    Some(true) => {}
                }
            }
            if anynone {
                None
            } else {
                Some(true)
            }
        }
        "or" => {
            let mut anynone = false;
            for x in &n.args {
                match uniform(x, env) {
                    Some(true) => return Some(true),
                    None => anynone = true,
                    Some(false) => {}
                }
            }
            if anynone {
                None
            } else {
                Some(false)
            }
        }
        "not" if n.args.len() == 1 => uniform(&n.args[0], env).map(|u| !u),
        _ => None,
    }
}

pub(crate) fn decide(e: &Expr, env: &BTreeMap<String, (i64, i64)>) -> Expr {
    let Expr::Operator(n) = e else {
        return e.clone();
    };
    if n.op == "ifelse" && n.args.len() == 3 {
        match uniform(&n.args[0], env) {
            Some(true) => return decide(&n.args[1], env),
            Some(false) => return decide(&n.args[2], env),
            None => {}
        }
    } else if CMP_OPS.contains(&n.op.as_str()) {
        if let Some(u) = uniform(e, env) {
            return Expr::Integer(if u { 1 } else { 0 });
        }
    }
    // Identity-preserving: a subtree with nothing to decide comes back as
    // the same node, so the regions of one body share it and the band
    // calculus's identity memos (`OccIndex`, the derivative memo) serve
    // every region from one walk.
    map_children_shared(e, &mut |x| decide(x, env))
}

/// Row-index entry: a free name or a pinned integer.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RIdx {
    Name(String),
    Fixed(i64),
}

/// Partition `rows` at every decidable comparison's breakpoints; collapse
/// decided guards per cell. First dimension cycles fastest (golden order).
pub fn clip_regions(
    body: &Expr,
    ridx: &[RIdx],
    rows: &[(i64, i64)],
) -> Vec<(Vec<(i64, i64)>, Expr)> {
    let names: Vec<String> = ridx
        .iter()
        .filter_map(|r| match r {
            RIdx::Name(n) => Some(n.clone()),
            _ => None,
        })
        .collect();
    if names.is_empty() {
        return vec![(rows.to_vec(), body.clone())];
    }
    let mut bp = BTreeMap::new();
    breakpoints(&mut bp, body, &names);
    clip_cells(body, ridx, rows, &bp)
}

/// [`clip_regions`] on the breakpoints `bp` of `body`, already collected
/// (clip.jl `_clip_cells`).
fn clip_cells(
    body: &Expr,
    ridx: &[RIdx],
    rows: &[(i64, i64)],
    bp: &BTreeMap<String, BTreeSet<i64>>,
) -> Vec<(Vec<(i64, i64)>, Expr)> {
    if bp.is_empty() {
        return vec![(rows.to_vec(), body.clone())];
    }
    let mut dims: Vec<Vec<(i64, i64)>> = Vec::new();
    let mut dimname: Vec<Option<String>> = Vec::new();
    for (k, r) in ridx.iter().enumerate() {
        let (lo, hi) = rows[k];
        match r {
            RIdx::Name(nm) if bp.contains_key(nm) => {
                let cuts: Vec<i64> = bp[nm]
                    .iter()
                    .cloned()
                    .filter(|c| lo < *c && *c <= hi)
                    .collect();
                let mut ivs = Vec::new();
                let mut a = lo;
                for c in cuts {
                    ivs.push((a, c - 1));
                    a = c;
                }
                ivs.push((a, hi));
                dims.push(ivs);
                dimname.push(Some(nm.clone()));
            }
            RIdx::Name(nm) => {
                dims.push(vec![(lo, hi)]);
                dimname.push(Some(nm.clone()));
            }
            RIdx::Fixed(_) => {
                dims.push(vec![(lo, hi)]);
                dimname.push(None);
            }
        }
    }
    // odometer with the FIRST dimension fastest
    let mut out = Vec::new();
    let mut idx = vec![0usize; dims.len()];
    loop {
        let rows2: Vec<(i64, i64)> = idx.iter().enumerate().map(|(k, &i)| dims[k][i]).collect();
        let mut env = BTreeMap::new();
        for (k, nm) in dimname.iter().enumerate() {
            if let Some(nm) = nm {
                env.insert(nm.clone(), rows2[k]);
            }
        }
        out.push((rows2, decide(body, &env)));
        let mut k = 0;
        loop {
            if k == dims.len() {
                return out;
            }
            idx[k] += 1;
            if idx[k] < dims[k].len() {
                break;
            }
            idx[k] = 0;
            k += 1;
        }
    }
}

// ── clipping each summand on its own axes (clip.jl `clip_summands`) ─────────

/// The summands of `e`, distributing a scalar factor or divisor over a sum
/// (clip.jl `_summands`): `(a + b) / m` ↦ `[a / m, b / m]`, `q * (a + b)` ↦
/// `[q * a, q * b]`, `-(a + b)` ↦ `[-a, -b]`, `a - b` ↦ `[a, -b]`. A product
/// of two sums is not expanded. An `e` with nothing to distribute is its own
/// single summand.
fn summands(e: &Expr) -> Vec<Expr> {
    let Expr::Operator(n) = e else {
        return vec![e.clone()];
    };
    let a = &n.args;
    match n.op.as_str() {
        "+" => a.iter().flat_map(summands).collect(),
        "-" | "neg" if a.len() == 1 => {
            let ts = summands(&a[0]);
            if ts.len() == 1 {
                return vec![e.clone()];
            }
            ts.into_iter().map(|t| op(&n.op, vec![t])).collect()
        }
        "-" if a.len() == 2 => summands(&a[0])
            .into_iter()
            .chain(summands(&a[1]).into_iter().map(|t| op("-", vec![t])))
            .collect(),
        "*" => {
            let mut k = None;
            let mut ts = Vec::new();
            for (j, x) in a.iter().enumerate() {
                let tx = summands(x);
                if tx.len() < 2 {
                    continue;
                }
                if k.is_some() {
                    return vec![e.clone()]; // a product of two sums
                }
                k = Some(j);
                ts = tx;
            }
            let Some(k) = k else {
                return vec![e.clone()];
            };
            ts.into_iter()
                .map(|t| {
                    let args = a
                        .iter()
                        .enumerate()
                        .map(|(j, x)| if j == k { t.clone() } else { x.clone() })
                        .collect();
                    op("*", args)
                })
                .collect()
        }
        "/" if a.len() == 2 => {
            let ts = summands(&a[0]);
            if ts.len() == 1 {
                return vec![e.clone()];
            }
            ts.into_iter()
                .map(|t| op("/", vec![t, a[1].clone()]))
                .collect()
        }
        _ => vec![e.clone()],
    }
}

/// What [`clip_summands`] needs of a body independently of the box it is
/// clipped to, computed once per body (clip.jl `_SummandPlan`): the whole
/// body's breakpoints and, when a split is possible at all (`split`, and at
/// least two dimensions with a breakpoint), its summands and their
/// breakpoints. A faq node clips one body per guard piece; the walks are
/// not repeated per piece.
pub(crate) struct SummandPlan<'a> {
    body: &'a Expr,
    bp: BTreeMap<String, BTreeSet<i64>>,
    ts: Vec<Expr>,
    tbp: Vec<BTreeMap<String, BTreeSet<i64>>>,
}

impl<'a> SummandPlan<'a> {
    pub(crate) fn new(body: &'a Expr, ridx: &[RIdx], split: bool) -> Self {
        let names: Vec<String> = ridx
            .iter()
            .filter_map(|r| match r {
                RIdx::Name(n) => Some(n.clone()),
                _ => None,
            })
            .collect();
        let mut bp = BTreeMap::new();
        if !names.is_empty() {
            breakpoints(&mut bp, body, &names);
        }
        let mut ts = Vec::new();
        let mut tbp = Vec::new();
        if split && bp.len() >= 2 {
            let xs = summands(body);
            if xs.len() >= 2 {
                for x in &xs {
                    let mut d = BTreeMap::new();
                    breakpoints(&mut d, x, &names);
                    tbp.push(d);
                }
                ts = xs;
            }
        }
        SummandPlan { body, bp, ts, tbp }
    }
}

fn cut_inside(cs: &BTreeSet<i64>, (lo, hi): (i64, i64)) -> bool {
    cs.iter().any(|c| lo < *c && *c <= hi)
}

/// [`clip_regions`], but with each group of `body`'s summands clipped on
/// its own axes (clip.jl `clip_summands`). Two summands cut along a common
/// dimension of `rows` share a group. With fewer than two such groups this
/// is `clip_regions(body, ridx, rows)` exactly; otherwise the summands cut
/// along no dimension form one more group, and the result is the
/// concatenation, group by group in order of their first summand, of
/// `clip_regions` of each group's sum. The pieces of different groups
/// overlap, and their bands add (spec §2.2).
pub fn clip_summands(
    body: &Expr,
    ridx: &[RIdx],
    rows: &[(i64, i64)],
) -> Vec<(Vec<(i64, i64)>, Expr)> {
    clip_summands_plan(&SummandPlan::new(body, ridx, true), ridx, rows)
}

/// [`clip_summands`] of a body whose [`SummandPlan`] is already built.
pub(crate) fn clip_summands_plan(
    p: &SummandPlan,
    ridx: &[RIdx],
    rows: &[(i64, i64)],
) -> Vec<(Vec<(i64, i64)>, Expr)> {
    let mut bx: HashMap<&str, (i64, i64)> = HashMap::new();
    for (k, r) in ridx.iter().enumerate() {
        if let RIdx::Name(nm) = r {
            bx.insert(nm.as_str(), rows[k]);
        }
    }
    let whole = || clip_cells(p.body, ridx, rows, &p.bp);
    if p.ts.is_empty() {
        return whole();
    }
    // two groups need two dimensions cut inside `rows`
    if p.bp
        .iter()
        .filter(|(nm, cs)| cut_inside(cs, bx[nm.as_str()]))
        .count()
        < 2
    {
        return whole();
    }
    // the dimensions each summand is actually cut along inside `rows`
    let cuts: Vec<Vec<&String>> = p
        .tbp
        .iter()
        .map(|d| {
            d.iter()
                .filter(|(nm, cs)| cut_inside(cs, bx[nm.as_str()]))
                .map(|(nm, _)| nm)
                .collect()
        })
        .collect();
    // union-find over dimensions: a summand cut along two joins them
    let mut parent: HashMap<String, String> =
        bx.keys().map(|n| (n.to_string(), n.to_string())).collect();
    fn root(parent: &mut HashMap<String, String>, n: &str) -> String {
        let q = parent[n].clone();
        if q == n {
            return q;
        }
        let r = root(parent, &q);
        parent.insert(n.to_string(), r.clone());
        r
    }
    for c in &cuts {
        for k in 1..c.len() {
            let a = root(&mut parent, c[k]);
            let b = root(&mut parent, c[0]);
            parent.insert(a, b);
        }
    }
    // group keys by first summand; `None` is the group of uncut summands
    let keys: Vec<Option<String>> = cuts
        .iter()
        .map(|c| c.first().map(|c0| root(&mut parent, c0)))
        .collect();
    let mut order: Vec<&Option<String>> = Vec::new();
    for k in &keys {
        if !order.contains(&k) {
            order.push(k);
        }
    }
    if order.iter().filter(|k| k.is_some()).count() < 2 {
        return whole();
    }
    let mut groups: Vec<Vec<Expr>> = vec![Vec::new(); order.len()];
    for (t, k) in p.ts.iter().zip(&keys) {
        let g = order.iter().position(|o| *o == k).unwrap();
        groups[g].push(t.clone());
    }
    let mut out = Vec::new();
    for mut g in groups {
        let gb = if g.len() == 1 {
            g.pop().unwrap()
        } else {
            op("+", g)
        };
        out.extend(clip_regions(&gb, ridx, rows));
    }
    out
}

// ── interval bounds over a band's box (§9.1, §4.5) ─────────────────────────

/// A band's index names (row names, then contracted names) and their ranges.
pub(crate) fn index_box(b: &Band) -> HashMap<String, (i64, i64)> {
    let mut bx = HashMap::new();
    for (k, r) in b.ridx.iter().enumerate() {
        if let RIdx::Name(n) = r {
            bx.insert(n.clone(), b.rows[k]);
        }
    }
    for (n, lo, hi) in &b.contracted {
        bx.insert(n.clone(), (*lo, *hi));
    }
    bx
}

/// Bounds `(lo, hi, exact, names read)` of an index expression over `bx`, by
/// interval arithmetic over the integer `+`, `-`, `*` core; `None` outside
/// it (a `const` gather, a division, …). The bounds always enclose every
/// value; `exact` when they are also attained at corners of the box: an
/// affine expression reading each name at most once.
pub(crate) fn index_bounds(
    e: &Expr,
    bx: &HashMap<String, (i64, i64)>,
) -> Option<(i64, i64, bool, Vec<String>)> {
    match e {
        Expr::Integer(n) => Some((*n, *n, true, Vec::new())),
        Expr::Number(x) if x.fract() == 0.0 => Some((*x as i64, *x as i64, true, Vec::new())),
        Expr::Variable(n) => bx.get(n).map(|&(lo, hi)| (lo, hi, true, vec![n.clone()])),
        Expr::Operator(nd) if matches!(nd.op.as_str(), "+" | "-" | "*") && !nd.args.is_empty() => {
            let parts: Vec<_> = nd
                .args
                .iter()
                .map(|a| index_bounds(a, bx))
                .collect::<Option<_>>()?;
            let mut it = parts.into_iter();
            let (mut lo, mut hi, mut ex, mut ns) = it.next()?;
            if nd.op == "-" && nd.args.len() == 1 {
                return Some((-hi, -lo, ex, ns));
            }
            for (plo, phi, pex, pns) in it {
                match nd.op.as_str() {
                    "+" => (lo, hi) = (lo + plo, hi + phi),
                    "-" => (lo, hi) = (lo - phi, hi - plo),
                    _ => {
                        // affine only while one factor is a constant
                        ex &= ns.is_empty() || pns.is_empty();
                        let c = [lo * plo, lo * phi, hi * plo, hi * phi];
                        (lo, hi) = (*c.iter().min()?, *c.iter().max()?);
                    }
                }
                ex &= pex;
                ns.extend(pns);
            }
            let unique = ns.iter().collect::<HashSet<_>>().len() == ns.len();
            Some((lo, hi, ex && unique, ns))
        }
        _ => None,
    }
}
