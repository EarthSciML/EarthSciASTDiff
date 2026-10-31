//! Observed inlining and `index`-of-array-expression β-reduction (the
//! reference's inline.jl).

use std::collections::{HashMap, HashSet};

use earthsci_ast::free_variables;
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
    inline_observed_cached(e, obs, &mut None)
}

/// [`inline_observed`] with a cache of the names `obs`'s definitions use
/// (`None` until filled): pass one to every call against the same `obs`.
/// Every substitution is capture-avoiding, and a faq node binding a name an
/// enclosing faq already binds is renamed apart.
pub fn inline_observed_cached(
    e: &Expr,
    obs: &HashMap<String, Expr>,
    obs_names: &mut Option<HashSet<String>>,
) -> R {
    if obs.is_empty() && !needs_inline(e, &HashSet::new()) {
        return Ok(e.clone());
    }
    let mut cur = e.clone();
    let mut fresh = Fresh {
        root: e,
        obs,
        obs_names,
        taken: None,
    };
    let scope = HashSet::new();
    for _ in 0..MAX_INLINE_PASSES {
        let e2 = inline_once(&cur, obs, &mut fresh, &scope)?;
        if skey(&e2) == skey(&cur) {
            return Ok(e2);
        }
        cur = e2;
    }
    Err(InlineError(format!(
        "no fixpoint after {MAX_INLINE_PASSES} passes (cyclic observed?)"
    )))
}

/// Fresh names for renamed-apart binders (inline.jl `_Fresh`):
/// `_c<n>_<name>`, the smallest `n` clear of every name the tree and the
/// observed definitions use and of the caller's `avoid` set. Deterministic in
/// (name, avoid), so copies of one sum renamed in the same scope stay
/// structurally equal. The taken sets are collected on the first rename.
struct Fresh<'a> {
    root: &'a Expr,
    obs: &'a HashMap<String, Expr>,
    obs_names: &'a mut Option<HashSet<String>>,
    taken: Option<HashSet<String>>,
}

impl Fresh<'_> {
    fn name(&mut self, name: &str, avoid: &HashSet<String>) -> String {
        if self.obs_names.is_none() {
            let mut t = HashSet::new();
            for d in self.obs.values() {
                all_names(d, &mut t);
            }
            *self.obs_names = Some(t);
        }
        let root = self.root;
        let taken = self.taken.get_or_insert_with(|| {
            let mut t = HashSet::new();
            all_names(root, &mut t);
            t
        });
        let on = self.obs_names.as_ref().unwrap();
        let base = strip_generated(name);
        let mut n = 0;
        loop {
            n += 1;
            let c = format!("_c{n}_{base}");
            if !taken.contains(&c) && !on.contains(&c) && !avoid.contains(&c) {
                return c;
            }
        }
    }
}

/// `_c<n>_k` → `k`, so a renamed name renames to `_c<m>_k`, not `_c<m>__c<n>_k`.
fn strip_generated(name: &str) -> &str {
    if let Some(rest) = name.strip_prefix("_c") {
        let d = rest.bytes().take_while(|b| b.is_ascii_digit()).count();
        if d > 0 && rest[d..].starts_with('_') {
            return &rest[d + 1..];
        }
    }
    name
}

/// The index names a faq node binds: its ranges (sorted) and its output
/// indices.
pub(crate) fn binders(n: &ExpressionNode) -> Vec<String> {
    let mut b: Vec<String> = n
        .ranges
        .as_ref()
        .map(|r| r.keys().cloned().collect())
        .unwrap_or_default();
    b.sort();
    for x in n.output_idx.iter().flatten() {
        if !b.contains(x) {
            b.push(x.clone());
        }
    }
    b
}

/// Every variable name and every faq-bound index name in `e`.
pub(crate) fn all_names(e: &Expr, acc: &mut HashSet<String>) {
    match e {
        Expr::Variable(n) => {
            acc.insert(n.clone());
        }
        Expr::Operator(n) => all_names_node(n, acc),
        _ => {}
    }
}

fn all_names_node(n: &ExpressionNode, acc: &mut HashSet<String>) {
    if n.op == "faq" {
        acc.extend(binders(n));
    }
    n.map_children(&mut |c: &Expr| {
        all_names(c, acc);
        c.clone()
    });
}

/// `map_expr_children` with a fallible mapper (the first error wins).
fn try_map(e: &Expr, f: &mut dyn FnMut(&Expr) -> R) -> R {
    let mut err: Option<InlineError> = None;
    let out = map_expr_children(e, &mut |x| match f(x) {
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

/// Rename the binders `names` of faq node `n` to fresh ones, clear of
/// `avoid` and of every name under `n` (so no inner binder captures them).
fn rename_binders(
    n: &ExpressionNode,
    names: &[String],
    fresh: &mut Fresh,
    avoid: &HashSet<String>,
) -> Result<ExpressionNode, InlineError> {
    if n.join.is_some() {
        return Err(InlineError(
            "cannot rename the indices of a joined faq node (join keys name \
             them)"
                .into(),
        ));
    }
    let mut av = avoid.clone();
    all_names_node(n, &mut av);
    let mut ren: HashMap<String, Expr> = HashMap::new();
    let mut fv = HashSet::new();
    for nm in names {
        let f = fresh.name(nm, &av);
        av.insert(f.clone());
        fv.insert(f.clone());
        ren.insert(nm.clone(), Expr::Variable(f));
    }
    let rn = |x: &String| match ren.get(x) {
        Some(Expr::Variable(f)) => f.clone(),
        _ => x.clone(),
    };
    let ranges = n
        .ranges
        .as_ref()
        .map(|r| r.iter().map(|(k, v)| (rn(k), v.clone())).collect());
    let output_idx = n.output_idx.as_ref().map(|o| o.iter().map(&rn).collect());
    let expr = match n.expr.as_deref() {
        Some(b) => Some(Box::new(csubst(b, &ren, &fv, fresh)?)),
        None => None,
    };
    let filter = match n.filter.as_deref() {
        Some(b) => Some(Box::new(csubst(b, &ren, &fv, fresh)?)),
        None => None,
    };
    Ok(ExpressionNode {
        ranges,
        output_idx,
        expr,
        filter,
        ..n.clone()
    })
}

/// CAPTURE-AVOIDING substitution (`substitute` is not): a faq node shadows
/// the names it binds, and one binding a name free in a replacement (`fv`)
/// is renamed apart before the substitution enters it.
fn csubst(e: &Expr, m: &HashMap<String, Expr>, fv: &HashSet<String>, fresh: &mut Fresh) -> R {
    if m.is_empty() {
        return Ok(e.clone());
    }
    match e {
        Expr::Variable(nm) => Ok(m.get(nm).cloned().unwrap_or_else(|| e.clone())),
        Expr::Operator(n) if n.op == "faq" => {
            let b = binders(n);
            let inner: HashMap<String, Expr>;
            let m = if b.iter().any(|x| m.contains_key(x)) {
                inner = m
                    .iter()
                    .filter(|(k, _)| !b.contains(k))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                if inner.is_empty() {
                    return Ok(e.clone());
                }
                &inner
            } else {
                m
            };
            let clash: Vec<String> = b.into_iter().filter(|x| fv.contains(x)).collect();
            let node = if clash.is_empty() {
                e.clone()
            } else {
                Expr::operator(rename_binders(n, &clash, fresh, fv)?)
            };
            try_map(&node, &mut |x| csubst(x, m, fv, fresh))
        }
        Expr::Operator(_) => try_map(e, &mut |x| csubst(x, m, fv, fresh)),
        _ => Ok(e.clone()),
    }
}

/// β-reduce a read `F[sub…]` of a faq-defined array F (inline.jl
/// `_beta_faq`): output indices take the subscripts; any other index F's
/// ranges bind is CONTRACTED, so the read is the scalar faq summing over it
/// (`reduce`/`filter`/`semiring` carried). Capture-avoiding: a contracted
/// name the subscripts read, or one shadowing an index bound around the
/// reading site (`scope`), is renamed apart.
fn beta_faq(def: &ExpressionNode, sub: &[Expr], fresh: &mut Fresh, scope: &HashSet<String>) -> R {
    let oidx = def.output_idx.clone().unwrap_or_default();
    if sub.len() != oidx.len() {
        return Err(InlineError(format!(
            "index arity {} ≠ rank {} of a faq read",
            sub.len(),
            oidx.len()
        )));
    }
    let body = def
        .expr
        .as_deref()
        .ok_or_else(|| InlineError("faq without body".into()))?;
    let empty = HashMap::new();
    let ranges = def.ranges.as_ref().unwrap_or(&empty);
    let mut cnames: Vec<String> = ranges
        .keys()
        .filter(|k| !oidx.contains(k))
        .cloned()
        .collect();
    cnames.sort();
    let mut binds: HashMap<String, Expr> = oidx.iter().cloned().zip(sub.iter().cloned()).collect();
    let mut fv: HashSet<String> = HashSet::new();
    for x in sub {
        fv.extend(free_variables(x));
    }
    if cnames.is_empty() && def.filter.is_none() {
        return csubst(body, &binds, &fv, fresh);
    }
    if def.join.is_some() {
        return Err(InlineError(
            "cannot inline a read of a joined faq definition (join keys name \
             its indices)"
                .into(),
        ));
    }
    let mut av: HashSet<String> = fv
        .iter()
        .chain(scope.iter())
        .chain(cnames.iter())
        .cloned()
        .collect();
    let mut newranges = HashMap::new();
    for nm in &cnames {
        if fv.contains(nm) || scope.contains(nm) {
            let f = fresh.name(nm, &av);
            av.insert(f.clone());
            binds.insert(nm.clone(), Expr::Variable(f.clone()));
            newranges.insert(f, ranges[nm].clone());
        } else {
            newranges.insert(nm.clone(), ranges[nm].clone());
        }
    }
    let mut fv2 = fv.clone();
    for x in binds.values() {
        if let Expr::Variable(v) = x {
            fv2.insert(v.clone());
        }
    }
    let expr = csubst(body, &binds, &fv2, fresh)?;
    let filter = match def.filter.as_deref() {
        Some(f) => Some(Box::new(csubst(f, &binds, &fv2, fresh)?)),
        None => None,
    };
    Ok(Expr::operator(ExpressionNode {
        output_idx: Some(vec![]),
        ranges: Some(newranges),
        expr: Some(Box::new(expr)),
        filter,
        ..def.clone()
    }))
}

/// Whether a pass could change `e` with no observed to inline: an
/// `index`-of-array-expression, or a faq node shadowing an enclosing index.
fn needs_inline(e: &Expr, scope: &HashSet<String>) -> bool {
    let Expr::Operator(n) = e else { return false };
    if n.op == "index" && matches!(n.args.first(), Some(Expr::Operator(_))) {
        return true;
    }
    let inner;
    let scope = if n.op == "faq" {
        let b = binders(n);
        if b.iter().any(|x| scope.contains(x)) {
            return true;
        }
        inner = scope.iter().cloned().chain(b).collect::<HashSet<_>>();
        &inner
    } else {
        scope
    };
    let mut found = false;
    n.map_children(&mut |c: &Expr| {
        if !found {
            found = needs_inline(c, scope);
        }
        c.clone()
    });
    found
}

fn index_makearray(
    ma: &ExpressionNode,
    idx: &[Expr],
    obs: &HashMap<String, Expr>,
    fresh: &mut Fresh,
    scope: &HashSet<String>,
) -> R {
    let regions = ma
        .regions
        .as_ref()
        .ok_or_else(|| InlineError("makearray without regions".into()))?;
    let values = ma
        .values
        .as_ref()
        .ok_or_else(|| InlineError("makearray without values".into()))?;
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
                        oidx.len(),
                        sub_idx.len()
                    )));
                }
                beta_faq(vn, &sub_idx, fresh, scope)?
            } else if matches!(vn.op.as_str(), "makearray" | "index") {
                let mut args = vec![val.clone()];
                args.extend(sub_idx.iter().cloned());
                inline_once(&op("index", args), obs, fresh, scope)?
            } else {
                val.clone()
            }
        } else if matches!(val, Expr::Variable(_)) {
            let mut args = vec![val.clone()];
            args.extend(sub_idx.iter().cloned());
            inline_once(&op("index", args), obs, fresh, scope)?
        } else {
            val.clone() // scalar broadcast
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

/// One pass; `scope` is the index names the enclosing faq nodes bind.
fn inline_once(
    e: &Expr,
    obs: &HashMap<String, Expr>,
    fresh: &mut Fresh,
    scope: &HashSet<String>,
) -> R {
    if let Expr::Variable(name) = e {
        // A name an enclosing faq binds is that index, not an observed read.
        if scope.contains(name) {
            return Ok(e.clone());
        }
        return Ok(obs.get(name).cloned().unwrap_or_else(|| e.clone()));
    }
    let Expr::Operator(n) = e else {
        return Ok(e.clone());
    };
    let args = |fresh: &mut Fresh| -> Result<Vec<Expr>, InlineError> {
        n.args[1..]
            .iter()
            .map(|x| inline_once(x, obs, fresh, scope))
            .collect()
    };
    if n.op == "index" {
        if let Some(Expr::Operator(t)) = n.args.first() {
            if t.op == "makearray" {
                let idx = args(fresh)?;
                return index_makearray(t, &idx, obs, fresh, scope);
            }
            if t.op == "faq" {
                let sub = args(fresh)?;
                let r = beta_faq(t, &sub, fresh, scope)?;
                return inline_once(&r, obs, fresh, scope);
            }
        }
        if let Some(Expr::Variable(name)) = n.args.first() {
            if let (Some(def), false) = (obs.get(name), scope.contains(name)) {
                if let Expr::Operator(dn) = def {
                    if dn.op == "faq" {
                        let rank = dn.output_idx.as_ref().map_or(0, |o| o.len());
                        if rank != n.args.len() - 1 {
                            return Err(InlineError(format!(
                                "index arity {} ≠ rank {rank} of observed `{name}`",
                                n.args.len() - 1
                            )));
                        }
                        let sub = args(fresh)?;
                        return beta_faq(dn, &sub, fresh, scope);
                    }
                    if dn.op == "makearray" {
                        let idx = args(fresh)?;
                        return index_makearray(dn, &idx, obs, fresh, scope);
                    }
                    if dn.op == "const" {
                        let mut a = vec![def.clone()];
                        a.extend(args(fresh)?);
                        return Ok(op("index", a));
                    }
                }
                return Err(InlineError(format!(
                    "index into observed `{name}` whose definition is not an \
                     faq, makearray or const"
                )));
            }
        }
    }
    if n.op == "faq" {
        // A faq node binding a name already bound around it is renamed
        // apart: the band calculus reads a free index in a derivative by name.
        let clash: Vec<String> = binders(n)
            .into_iter()
            .filter(|x| scope.contains(x))
            .collect();
        let node = if clash.is_empty() {
            e.clone()
        } else {
            Expr::operator(rename_binders(n, &clash, fresh, scope)?)
        };
        let Expr::Operator(nn) = &node else {
            unreachable!()
        };
        let inner: HashSet<String> = scope.iter().cloned().chain(binders(nn)).collect();
        return try_map(&node, &mut |x| inline_once(x, obs, fresh, &inner));
    }
    try_map(e, &mut |x| inline_once(x, obs, fresh, scope))
}
