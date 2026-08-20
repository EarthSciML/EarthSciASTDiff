//! Expression constructors, structural keys, and the differentiation site
//! (mirror of the reference's expr_helpers).

use earthsci_ast::canonicalize::canonical_json;
use earthsci_ast::types::{Expr, ExpressionNode};
use serde_json::Value;

pub fn op(o: &str, args: Vec<Expr>) -> Expr {
    Expr::operator(ExpressionNode {
        op: o.to_string(),
        args,
        ..Default::default()
    })
}

pub fn is_zero(e: &Expr) -> bool {
    matches!(e, Expr::Integer(0)) || matches!(e, Expr::Number(x) if *x == 0.0)
}

pub fn is_one(e: &Expr) -> bool {
    matches!(e, Expr::Integer(1)) || matches!(e, Expr::Number(x) if *x == 1.0)
}

/// n-ary `+` dropping literal zeros, collapsing to a single argument.
pub fn add(args: Vec<Expr>) -> Expr {
    let a: Vec<Expr> = args.into_iter().filter(|x| !is_zero(x)).collect();
    match a.len() {
        0 => Expr::Integer(0),
        1 => a.into_iter().next().unwrap(),
        _ => op("+", a),
    }
}

/// n-ary `*` short-circuiting on a literal zero, dropping literal ones.
pub fn mul(args: Vec<Expr>) -> Expr {
    if args.iter().any(is_zero) {
        return Expr::Integer(0);
    }
    let a: Vec<Expr> = args.into_iter().filter(|x| !is_one(x)).collect();
    match a.len() {
        0 => Expr::Integer(1),
        1 => a.into_iter().next().unwrap(),
        _ => op("*", a),
    }
}

pub fn neg(x: Expr) -> Expr {
    if is_zero(&x) {
        Expr::Integer(0)
    } else {
        op("-", vec![x])
    }
}

/// Deterministic JSON: sorted object keys at every level, scalars formatted
/// by serde_json (Ryu shortest round-trip — the same family the reference's
/// JSON3 uses).
pub fn stable_json(v: &Value) -> String {
    let mut out = String::new();
    write_stable(v, &mut out);
    out
}

fn write_stable(v: &Value, out: &mut String) {
    match v {
        Value::Object(m) => {
            out.push('{');
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(k).unwrap());
                out.push(':');
                write_stable(&m[*k], out);
            }
            out.push('}');
        }
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_stable(x, out);
            }
            out.push(']');
        }
        _ => out.push_str(&serde_json::to_string(v).unwrap()),
    }
}

pub fn ser_expr(e: &Expr) -> Value {
    serde_json::to_value(e).expect("expression serializes")
}

/// Full-structure key for ANY expression (array ops included).
pub fn skey(e: &Expr) -> String {
    stable_json(&ser_expr(e))
}

/// Term-identity key for SCALAR expressions (RFC §5.4 canonical form).
pub fn ckey(e: &Expr) -> String {
    canonical_json(e).expect("scalar-core canonical form")
}

/// A bare variable reference or one symbolic `index(v, i…)` cell.
#[derive(Clone, Debug)]
pub struct Site {
    pub expr: Expr,
    pub key: String,
}

impl Site {
    pub fn of(e: &Expr) -> Site {
        Site { expr: e.clone(), key: ckey(e) }
    }
}

pub fn is_site(e: &Expr, s: &Site) -> bool {
    let candidate = matches!(e, Expr::Variable(_))
        || matches!(e, Expr::Operator(n) if n.op == "index");
    candidate && ckey(e) == s.key
}

/// Structural occurrence — THE sparsity primitive (never a zero test).
pub fn occurs(e: &Expr, s: &Site) -> bool {
    if is_site(e, s) {
        return true;
    }
    let Expr::Operator(n) = e else { return false };
    let mut found = false;
    n.map_children(&mut |c: &Expr| {
        if !found {
            found = occurs(c, s);
        }
        c.clone()
    });
    found
}

pub fn occurs_var(e: &Expr, v: &str) -> bool {
    earthsci_ast::expression::free_variables(e).contains(v)
}

/// Rebuild an operator node with mapped children (Expr-level wrapper).
pub fn map_expr_children(e: &Expr, f: &mut impl FnMut(&Expr) -> Expr) -> Expr {
    match e {
        Expr::Operator(n) => Expr::operator(n.map_children(f)),
        _ => e.clone(),
    }
}
