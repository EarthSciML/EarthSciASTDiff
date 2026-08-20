//! Observed inlining and `index`-of-array-expression β-reduction (the
//! reference's inline.jl).

use std::collections::HashMap;

use earthsci_ast::substitute::substitute;
use earthsci_ast::types::{Expr, ExpressionNode};

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
    for _ in 0..MAX_INLINE_PASSES {
        let e2 = inline_once(&cur, obs)?;
        if skey(&e2) == skey(&cur) {
            return Ok(e2);
        }
        cur = e2;
    }
    Err(InlineError(format!(
        "no fixpoint after {MAX_INLINE_PASSES} passes (cyclic observed?)")))
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
                   obs: &HashMap<String, Expr>) -> R {
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
            conds.push(op(">=", vec![idx[k].clone(), Expr::Integer(r[0])]));
            conds.push(op("<=", vec![idx[k].clone(), Expr::Integer(r[1])]));
            if r[1] > r[0] {
                sub_idx.push(idx[k].clone());
            }
        }
        let v = if let Expr::Operator(vn) = val {
            if vn.op == "aggregate" {
                let oidx = vn.output_idx.clone().unwrap_or_default();
                if oidx.len() != sub_idx.len() {
                    return Err(InlineError(format!(
                        "makearray value rank {} ≠ region free rank {}",
                        oidx.len(), sub_idx.len())));
                }
                let body = vn.expr.as_deref().ok_or_else(|| {
                    InlineError("aggregate without body".into())
                })?;
                let binds: HashMap<String, Expr> = oidx
                    .iter().cloned().zip(sub_idx.iter().cloned()).collect();
                substitute(body, &binds)
            } else if matches!(vn.op.as_str(), "makearray" | "index") {
                let mut args = vec![val.clone()];
                args.extend(sub_idx.iter().cloned());
                inline_once(&op("index", args), obs)?
            } else {
                val.clone()
            }
        } else if matches!(val, Expr::Variable(_)) {
            let mut args = vec![val.clone()];
            args.extend(sub_idx.iter().cloned());
            inline_once(&op("index", args), obs)?
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

fn inline_once(e: &Expr, obs: &HashMap<String, Expr>) -> R {
    if let Expr::Variable(name) = e {
        return Ok(obs.get(name).cloned().unwrap_or_else(|| e.clone()));
    }
    let Expr::Operator(n) = e else { return Ok(e.clone()) };
    if n.op == "index" {
        if let Some(Expr::Operator(t)) = n.args.first() {
            if t.op == "makearray" {
                let idx: Vec<Expr> = n.args[1..]
                    .iter().map(|x| inline_once(x, obs))
                    .collect::<Result<_, _>>()?;
                return index_makearray(t, &idx, obs);
            }
            if t.op == "aggregate" {
                let oidx = t.output_idx.clone().unwrap_or_default();
                let body = t.expr.as_deref().ok_or_else(|| {
                    InlineError("aggregate without body".into())
                })?;
                let mut binds = HashMap::new();
                for (k, nme) in oidx.iter().enumerate() {
                    binds.insert(nme.clone(), inline_once(&n.args[k + 1], obs)?);
                }
                return inline_once(&substitute(body, &binds), obs);
            }
        }
        if let Some(Expr::Variable(name)) = n.args.first() {
            if let Some(def) = obs.get(name) {
                if let Expr::Operator(dn) = def {
                    if dn.op == "aggregate" {
                        let oidx = dn.output_idx.clone().unwrap_or_default();
                        if oidx.len() != n.args.len() - 1 {
                            return Err(InlineError(format!(
                                "index arity {} ≠ rank {} of observed `{name}`",
                                n.args.len() - 1, oidx.len())));
                        }
                        let body = dn.expr.as_deref().ok_or_else(|| {
                            InlineError("aggregate without body".into())
                        })?;
                        let mut binds = HashMap::new();
                        for (k, nme) in oidx.iter().enumerate() {
                            binds.insert(nme.clone(),
                                         inline_once(&n.args[k + 1], obs)?);
                        }
                        return Ok(substitute(body, &binds));
                    }
                    if dn.op == "makearray" {
                        let idx: Vec<Expr> = n.args[1..]
                            .iter().map(|x| inline_once(x, obs))
                            .collect::<Result<_, _>>()?;
                        return index_makearray(dn, &idx, obs);
                    }
                    if dn.op == "const" {
                        let mut args = vec![def.clone()];
                        for x in &n.args[1..] {
                            args.push(inline_once(x, obs)?);
                        }
                        return Ok(op("index", args));
                    }
                }
                return Err(InlineError(format!(
                    "index into observed `{name}` whose definition is not an \
                     aggregate, makearray or const")));
            }
        }
    }
    let mut err: Option<InlineError> = None;
    let out = map_expr_children(e, &mut |x| match inline_once(x, obs) {
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
