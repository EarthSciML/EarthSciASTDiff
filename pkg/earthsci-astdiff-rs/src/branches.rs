//! Branch-aware simplification + wire-canonical literals (the reference's
//! simplify_branches.jl). `simp` is THE coefficient simplifier.

use earthsci_ast::types::Expr;

use crate::helpers::{is_zero, map_expr_children, op, skey};
use crate::simplify_port::simplify_jl;

pub fn simplify_branches(e: &Expr) -> Expr {
    sbr(e)
}

fn sbr(e: &Expr) -> Expr {
    let Expr::Operator(_) = e else { return e.clone() };
    let e2 = map_expr_children(e, &mut |c| sbr(c));
    let Expr::Operator(n2) = &e2 else { unreachable!() };
    if !(n2.op == "ifelse" && n2.args.len() == 3) {
        return e2;
    }
    let (c, a, b) = (&n2.args[0], &n2.args[1], &n2.args[2]);
    if matches!(c, Expr::Integer(_) | Expr::Number(_)) {
        return if is_zero(c) { b.clone() } else { a.clone() };
    }
    let ck = skey(c);
    let a2 = assume(a, &ck, true);
    let b2 = assume(b, &ck, false);
    if skey(&a2) == skey(&b2) {
        return a2;
    }
    op("ifelse", vec![c.clone(), a2, b2])
}

fn assume(e: &Expr, ck: &str, val: bool) -> Expr {
    let Expr::Operator(n) = e else { return e.clone() };
    if n.op == "ifelse" && n.args.len() == 3 && skey(&n.args[0]) == ck {
        return assume(&n.args[if val { 1 } else { 2 }], ck, val);
    }
    map_expr_children(e, &mut |x| assume(x, ck, val))
}

const INT64_MAXF: f64 = 9_223_372_036_854_775_807.0;
const INT64_MINF: f64 = -9_223_372_036_854_775_808.0;

pub fn canon_lit(e: &Expr) -> Expr {
    if let Expr::Number(x) = e {
        if x.is_finite()
            && *x == x.trunc()
            && *x >= INT64_MINF
            && *x <= INT64_MAXF
            && (*x as i64) as f64 == *x
        {
            return Expr::Integer(*x as i64);
        }
    }
    e.clone()
}

pub fn canon_lits(e: &Expr) -> Expr {
    match e {
        Expr::Operator(_) => map_expr_children(e, &mut |c| canon_lits(c)),
        _ => canon_lit(e),
    }
}

/// Algebraic pass, branch pass, final algebraic pass, literal normalization.
pub fn simp(e: &Expr) -> Expr {
    canon_lits(&simplify_jl(&simplify_branches(&simplify_jl(e))))
}
