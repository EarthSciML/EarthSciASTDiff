//! Static region clipping (the reference's clip.jl): decidable membership
//! guards decided exactly by splitting band row ranges at their affine
//! breakpoints; conservative everywhere else.

use std::collections::{BTreeMap, BTreeSet};

use earthsci_ast::types::Expr;

use crate::helpers::{map_expr_children, op};

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
    fn walk<'a>(x: &Expr, names: &'a [String], found: &mut Option<&'a str>,
                ok: &mut bool) {
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
            n.map_children(&mut |c: &Expr| {
                walk(c, names, found, ok);
                c.clone()
            });
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

fn breakpoints(bp: &mut BTreeMap<String, BTreeSet<i64>>, e: &Expr,
               names: &[String]) {
    let Expr::Operator(n) = e else { return };
    if CMP_OPS.contains(&n.op.as_str()) && n.args.len() == 2 {
        if let Some(nm) = sole_name(e, names) {
            let diff = op("-", vec![n.args[0].clone(), n.args[1].clone()]);
            if let Some((al, be)) = affine(&diff, nm) {
                if al != 0 {
                    let q = fld(-be, al);
                    for c in [q, q + 1] {
                        if cmp_truth(&n.op, al, be, c - 1)
                            != cmp_truth(&n.op, al, be, c)
                        {
                            bp.entry(nm.to_string()).or_default().insert(c);
                        }
                    }
                }
            }
        }
    }
    n.map_children(&mut |c: &Expr| {
        breakpoints(bp, c, names);
        c.clone()
    });
}

fn uniform(cond: &Expr, env: &BTreeMap<String, (i64, i64)>) -> Option<bool> {
    let Expr::Operator(n) = cond else { return None };
    if CMP_OPS.contains(&n.op.as_str()) && n.args.len() == 2 {
        let keys: Vec<String> = env.keys().cloned().collect();
        let nm = sole_name(cond, &keys)?;
        let (lo, hi) = *env.get(nm)?;
        let (al, be) = affine(&op("-", vec![n.args[0].clone(),
                                            n.args[1].clone()]), nm)?;
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
            if anynone { None } else { Some(true) }
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
            if anynone { None } else { Some(false) }
        }
        "not" if n.args.len() == 1 => uniform(&n.args[0], env).map(|u| !u),
        _ => None,
    }
}

fn decide(e: &Expr, env: &BTreeMap<String, (i64, i64)>) -> Expr {
    let Expr::Operator(n) = e else { return e.clone() };
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
    map_expr_children(e, &mut |x| decide(x, env))
}

/// Row-index entry: a free name or a pinned integer.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RIdx {
    Name(String),
    Fixed(i64),
}

/// Partition `rows` at every decidable comparison's breakpoints; collapse
/// decided guards per cell. First dimension cycles fastest (golden order).
pub fn clip_regions(body: &Expr, ridx: &[RIdx], rows: &[(i64, i64)])
                    -> Vec<(Vec<(i64, i64)>, Expr)> {
    let names: Vec<String> = ridx.iter().filter_map(|r| match r {
        RIdx::Name(n) => Some(n.clone()),
        _ => None,
    }).collect();
    if names.is_empty() {
        return vec![(rows.to_vec(), body.clone())];
    }
    let mut bp = BTreeMap::new();
    breakpoints(&mut bp, body, &names);
    if bp.is_empty() {
        return vec![(rows.to_vec(), body.clone())];
    }
    let mut dims: Vec<Vec<(i64, i64)>> = Vec::new();
    let mut dimname: Vec<Option<String>> = Vec::new();
    for (k, r) in ridx.iter().enumerate() {
        let (lo, hi) = rows[k];
        match r {
            RIdx::Name(nm) if bp.contains_key(nm) => {
                let cuts: Vec<i64> = bp[nm].iter().cloned()
                    .filter(|c| lo < *c && *c <= hi).collect();
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
        let rows2: Vec<(i64, i64)> =
            idx.iter().enumerate().map(|(k, &i)| dims[k][i]).collect();
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
