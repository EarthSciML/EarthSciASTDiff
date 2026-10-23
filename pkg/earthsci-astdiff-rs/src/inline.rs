//! Observed inlining and `index`-of-array-expression β-reduction (the
//! reference's inline.jl).

use std::collections::{HashMap, HashSet};

use earthsci_ast::substitute;
use earthsci_ast::{Expr, ExpressionNode};

use crate::bands::region_bounds;
use crate::helpers::{map_expr_children, op, skey};

pub const MAX_INLINE_PASSES: usize = 64;

#[derive(Debug)]
pub struct InlineError(pub String);

impl std::fmt::Display for InlineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InlineError: {}", self.0)
    }
}
impl std::error::Error for InlineError {}

type R = Result<Expr, InlineError>;

pub fn inline_observed(e: &Expr, obs: &HashMap<String, Expr>) -> R {
    if obs.is_empty() && !needs_inline(e) {
        return Ok(e.clone());
    }
    let mut cur = e.clone();
    let mut fresh = Fresh::new(e);
    for _ in 0..MAX_INLINE_PASSES {
        let e2 = inline_once(&cur, obs, &mut fresh)?;
        if skey(&e2) == skey(&cur) {
            return Ok(e2);
        }
        cur = e2;
    }
    Err(InlineError(format!(
        "no fixpoint after {MAX_INLINE_PASSES} passes (cyclic observed?)")))
}

/// Fresh names for renamed-apart contraction indices: `_c<n>_<name>`,
/// skipping every name the tree being inlined already uses (collected on the
/// first request only).
struct Fresh {
    root: Expr,
    taken: Option<HashSet<String>>,
    n: usize,
}

impl Fresh {
    fn new(root: &Expr) -> Self {
        Fresh { root: root.clone(), taken: None, n: 0 }
    }

    fn name(&mut self, base: &str) -> String {
        let root = &self.root;
        let taken = self.taken.get_or_insert_with(|| {
            let mut acc = HashSet::new();
            all_names(root, &mut acc);
            acc
        });
        loop {
            self.n += 1;
            let name = format!("_c{}_{base}", self.n);
            if taken.insert(name.clone()) {
                return name;
            }
        }
    }
}

/// Every variable name and every faq-bound index name in `e`.
pub(crate) fn all_names(e: &Expr, acc: &mut HashSet<String>) {
    match e {
        Expr::Variable(n) => {
            acc.insert(n.clone());
        }
        Expr::Operator(n) => {
            if n.op == "faq" {
                if let Some(r) = &n.ranges {
                    acc.extend(r.keys().cloned());
                }
                if let Some(o) = &n.output_idx {
                    acc.extend(o.iter().cloned());
                }
            }
            n.map_children(&mut |c: &Expr| {
                all_names(c, acc);
                c.clone()
            });
        }
        _ => {}
    }
}

/// β-reduce a read `F[sub…]` of a faq-defined array F (inline.jl
/// `_beta_faq`): output indices take the subscripts; any other index F's
/// ranges bind is CONTRACTED, so the read is the scalar faq summing over it
/// (`reduce`/`filter`/`semiring` carried), with the contracted names renamed
/// to fresh ones in the same substitution — `substitute` is not
/// capture-avoiding, and a faq around the reading site may bind them.
fn beta_faq(def: &ExpressionNode, sub: &[Expr], fresh: &mut Fresh) -> R {
    let oidx = def.output_idx.clone().unwrap_or_default();
    if sub.len() != oidx.len() {
        return Err(InlineError(format!(
            "index arity {} ≠ rank {} of a faq read", sub.len(), oidx.len())));
    }
    let body = def.expr.as_deref()
        .ok_or_else(|| InlineError("faq without body".into()))?;
    let empty = HashMap::new();
    let ranges = def.ranges.as_ref().unwrap_or(&empty);
    let mut cnames: Vec<String> = ranges.keys()
        .filter(|k| !oidx.contains(k)).cloned().collect();
    cnames.sort();
    let mut binds: HashMap<String, Expr> =
        oidx.iter().cloned().zip(sub.iter().cloned()).collect();
    if cnames.is_empty() && def.filter.is_none() {
        return Ok(substitute(body, &binds));
    }
    if def.join.is_some() {
        return Err(InlineError(
            "cannot inline a read of a joined faq definition (join keys name \
             its indices)".into()));
    }
    let mut newranges = HashMap::new();
    for nm in &cnames {
        let f = fresh.name(nm);
        binds.insert(nm.clone(), Expr::Variable(f.clone()));
        newranges.insert(f, ranges[nm].clone());
    }
    Ok(Expr::operator(ExpressionNode {
        output_idx: Some(vec![]),
        ranges: Some(newranges),
        expr: Some(Box::new(substitute(body, &binds))),
        filter: def.filter.as_deref().map(|f| Box::new(substitute(f, &binds))),
        ..def.clone()
    }))
}

fn needs_inline(e: &Expr) -> bool {
    let Expr::Operator(n) = e else { return false };
    if n.op == "index" && matches!(n.args.first(), Some(Expr::Operator(_))) {
        return true;
    }
    let mut found = false;
    n.map_children(&mut |c: &Expr| {
        if !found {
            found = needs_inline(c);
        }
        c.clone()
    });
    found
}

fn index_makearray(ma: &ExpressionNode, idx: &[Expr],
                   obs: &HashMap<String, Expr>, fresh: &mut Fresh) -> R {
    let regions = ma.regions.as_ref().ok_or_else(|| {
        InlineError("makearray without regions".into())
    })?;
    let values = ma.values.as_ref().ok_or_else(|| {
        InlineError("makearray without values".into())
    })?;
    let mut result = Expr::Integer(0);
    for (region, val) in regions.iter().zip(values.iter()) {
        let mut conds = Vec::new();
        let mut sub_idx = Vec::new();
        for (k, r) in region.iter().enumerate() {
            let (lo, hi) = region_bounds(r).map_err(|e| InlineError(e.0))?;
            conds.push(op(">=", vec![idx[k].clone(), Expr::Integer(lo)]));
            conds.push(op("<=", vec![idx[k].clone(), Expr::Integer(hi)]));
            if hi > lo {
                sub_idx.push(idx[k].clone());
            }
        }
        let v = if let Expr::Operator(vn) = val {
            if vn.op == "faq" {
                let oidx = vn.output_idx.clone().unwrap_or_default();
                if oidx.len() != sub_idx.len() {
                    return Err(InlineError(format!(
                        "makearray value rank {} ≠ region free rank {}",
                        oidx.len(), sub_idx.len())));
                }
                beta_faq(vn, &sub_idx, fresh)?
            } else if matches!(vn.op.as_str(), "makearray" | "index") {
                let mut args = vec![val.clone()];
                args.extend(sub_idx.iter().cloned());
                inline_once(&op("index", args), obs, fresh)?
            } else {
                val.clone()
            }
        } else if matches!(val, Expr::Variable(_)) {
            let mut args = vec![val.clone()];
            args.extend(sub_idx.iter().cloned());
            inline_once(&op("index", args), obs, fresh)?
        } else {
            val.clone()                        // scalar broadcast
        };
        let cond = if conds.len() == 1 {
            conds.into_iter().next().unwrap()
        } else {
            op("and", conds)
        };
        result = op("ifelse", vec![cond, v, result]);
    }
    Ok(result)
}

fn inline_once(e: &Expr, obs: &HashMap<String, Expr>, fresh: &mut Fresh) -> R {
    if let Expr::Variable(name) = e {
        return match obs.get(name) {
            // A scalar sum renamed apart too: spliced under a faq binding the
            // same index, its sites would be the outer ones, structurally.
            Some(Expr::Operator(dn)) if dn.op == "faq"
                && dn.output_idx.as_ref().map_or(true, |o| o.is_empty()) =>
                beta_faq(dn, &[], fresh),
            Some(def) => Ok(def.clone()),
            None => Ok(e.clone()),
        };
    }
    let Expr::Operator(n) = e else { return Ok(e.clone()) };
    if n.op == "index" {
        if let Some(Expr::Operator(t)) = n.args.first() {
            if t.op == "makearray" {
                let idx: Vec<Expr> = n.args[1..]
                    .iter().map(|x| inline_once(x, obs, fresh))
                    .collect::<Result<_, _>>()?;
                return index_makearray(t, &idx, obs, fresh);
            }
            if t.op == "faq" {
                let sub: Vec<Expr> = n.args[1..]
                    .iter().map(|x| inline_once(x, obs, fresh))
                    .collect::<Result<_, _>>()?;
                let r = beta_faq(t, &sub, fresh)?;
                return inline_once(&r, obs, fresh);
            }
        }
        if let Some(Expr::Variable(name)) = n.args.first() {
            if let Some(def) = obs.get(name) {
                if let Expr::Operator(dn) = def {
                    if dn.op == "faq" {
                        let oidx = dn.output_idx.clone().unwrap_or_default();
                        if oidx.len() != n.args.len() - 1 {
                            return Err(InlineError(format!(
                                "index arity {} ≠ rank {} of observed `{name}`",
                                n.args.len() - 1, oidx.len())));
                        }
                        let sub: Vec<Expr> = n.args[1..]
                            .iter().map(|x| inline_once(x, obs, fresh))
                            .collect::<Result<_, _>>()?;
                        return beta_faq(dn, &sub, fresh);
                    }
                    if dn.op == "makearray" {
                        let idx: Vec<Expr> = n.args[1..]
                            .iter().map(|x| inline_once(x, obs, fresh))
                            .collect::<Result<_, _>>()?;
                        return index_makearray(dn, &idx, obs, fresh);
                    }
                    if dn.op == "const" {
                        let mut args = vec![def.clone()];
                        for x in &n.args[1..] {
                            args.push(inline_once(x, obs, fresh)?);
                        }
                        return Ok(op("index", args));
                    }
                }
                return Err(InlineError(format!(
                    "index into observed `{name}` whose definition is not an \
                     faq, makearray or const")));
            }
        }
    }
    let mut err: Option<InlineError> = None;
    let out = map_expr_children(e, &mut |x| match inline_once(x, obs, fresh) {
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
