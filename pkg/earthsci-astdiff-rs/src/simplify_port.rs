//! Faithful port of the Julia reference's `EarthSciAST.simplify` (the
//! bindings' own simplifiers differ in commutative-argument handling; the
//! goldens pin the Julia behavior — same rationale as the Python binding's
//! simplify_port.py).

use std::collections::HashMap;

use earthsci_ast::expression::evaluate;
use earthsci_ast::types::{Expr, ExpressionNode};

use crate::helpers::map_expr_children;

const TWO63: f64 = 9_223_372_036_854_775_808.0; // 2^63

fn is_lit(e: &Expr) -> bool {
    matches!(e, Expr::Integer(_) | Expr::Number(_))
}

fn lit_val(e: &Expr) -> Option<f64> {
    match e {
        Expr::Integer(i) => Some(*i as f64),
        Expr::Number(x) => Some(*x),
        _ => None,
    }
}

fn is_lit_val(e: &Expr, v: f64) -> bool {
    lit_val(e).map(|x| x == v).unwrap_or(false)
}

pub fn simplify_jl(e: &Expr) -> Expr {
    let Expr::Operator(_) = e else { return e.clone() };
    let node_e = map_expr_children(e, &mut |c| simplify_jl(c));
    let Expr::Operator(node) = &node_e else { unreachable!() };
    let args = &node.args;
    let o = node.op.as_str();

    // Constant folding through the official evaluator; refusal (unbound
    // names, arity errors) or a NaN result (the DomainError analogue)
    // declines to fold and falls through to the algebraic rules.
    if args.iter().all(is_lit) {
        if let Ok(rv) = evaluate(&node_e, &HashMap::new()) {
            if !rv.is_nan() {
                let all_int = args.iter().all(|a| matches!(a, Expr::Integer(_)));
                if all_int
                    && rv.is_finite()
                    && rv == rv.trunc()
                    && rv.abs() <= TWO63
                {
                    if rv.abs() != TWO63 {
                        return Expr::Integer(rv as i64);
                    }
                    // Julia Int64() InexactError at ±2^63 → decline to fold.
                } else {
                    return Expr::Number(rv);
                }
            }
        }
    }

    let rebuild = |a: Vec<Expr>| -> Expr {
        let mut n2: ExpressionNode = (**node).clone();
        n2.args = a;
        Expr::operator(n2)
    };

    match o {
        "+" => {
            let nz: Vec<Expr> =
                args.iter().filter(|a| !is_lit_val(a, 0.0)).cloned().collect();
            match nz.len() {
                0 => Expr::Number(0.0),
                1 => nz.into_iter().next().unwrap(),
                _ => rebuild(nz),
            }
        }
        "*" => {
            if args.iter().any(|a| is_lit_val(a, 0.0)) {
                return Expr::Number(0.0);
            }
            let no: Vec<Expr> =
                args.iter().filter(|a| !is_lit_val(a, 1.0)).cloned().collect();
            match no.len() {
                0 => Expr::Number(1.0),
                1 => no.into_iter().next().unwrap(),
                _ => rebuild(no),
            }
        }
        "^" if args.len() == 2 => {
            let (base, expo) = (&args[0], &args[1]);
            if is_lit_val(expo, 0.0) {
                return Expr::Number(1.0);
            }
            if is_lit_val(expo, 1.0) {
                return base.clone();
            }
            if is_lit_val(base, 0.0)
                && lit_val(expo).map(|v| v > 0.0).unwrap_or(false)
            {
                return Expr::Number(0.0);
            }
            if is_lit_val(base, 1.0) {
                return Expr::Number(1.0);
            }
            node_e.clone()
        }
        "-" if args.len() == 2 => {
            if is_lit_val(&args[1], 0.0) {
                args[0].clone()
            } else {
                node_e.clone()
            }
        }
        "/" if args.len() == 2 => {
            if is_lit_val(&args[1], 1.0) {
                return args[0].clone();
            }
            if matches!(&args[0], Expr::Number(x) if *x == 0.0) {
                return Expr::Number(0.0);
            }
            node_e.clone()
        }
        _ => node_e.clone(),
    }
}
