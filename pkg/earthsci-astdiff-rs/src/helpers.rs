//! Expression constructors, structural keys, and the differentiation site
//! (mirror of the reference's expr_helpers).

use earthsci_ast::canonical_json;
use earthsci_ast::{Expr, ExpressionNode};
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

/// A float in the Julia reference's layout (EarthSciAST CONFORMANCE_SPEC
/// §5.5.3.1 rule 2): shortest round-trip digits, positional iff the decimal
/// exponent of the first significant digit is in `-4..=5`, otherwise
/// `d.f…e<exp>` with at least one fraction digit (`1.0e-5`, `2.4405875e6`).
/// serde_json has the same digits but switches layout at other magnitudes
/// (`0.0000727…`, `2440587.5`). Not `earthsci_ast::format_canonical_float`,
/// which is the RFC §5.4.6 canonical-form layout.
pub fn canonical_float(f: f64) -> String {
    assert!(f.is_finite(), "non-finite float {f} has no JSON form");
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    // `{:e}` is the shortest round-trip digits as `d[.ddd]e<exp>`
    let sci = format!("{:e}", f.abs());
    let (mant, exp) = sci.split_once('e').unwrap();
    let e10: i32 = exp.parse().unwrap();
    let ds: String = mant.chars().filter(|c| *c != '.').collect();
    let sign = if f < 0.0 { "-" } else { "" };
    let body = if (-4..=5).contains(&e10) {
        if e10 < 0 {
            format!("0.{}{ds}", "0".repeat((-e10 - 1) as usize))
        } else {
            let k = (e10 + 1) as usize;
            let int = format!("{:0<k$}", &ds[..k.min(ds.len())]);
            let frac = if ds.len() > k { &ds[k..] } else { "0" };
            format!("{int}.{frac}")
        }
    } else {
        let frac = if ds.len() > 1 { &ds[1..] } else { "0" };
        format!("{}.{frac}e{e10}", &ds[..1])
    };
    format!("{sign}{body}")
}

/// Deterministic JSON: sorted object keys at every level, floats in the
/// reference's layout ([`canonical_float`]), every other scalar by
/// serde_json.
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
        Value::Number(n) if !(n.is_i64() || n.is_u64()) => {
            out.push_str(&canonical_float(n.as_f64().unwrap()))
        }
        _ => out.push_str(&serde_json::to_string(v).unwrap()),
    }
}

#[cfg(test)]
mod tests {
    use super::canonical_float;

    #[test]
    fn canonical_float_matches_the_reference_layout() {
        // CONFORMANCE_SPEC §5.5.3.1's probe table, plus Julia `show` values
        for (v, want) in [
            (1.0e-9, "1.0e-9"),
            (1.5e-5, "1.5e-5"),
            (0.000123456, "0.000123456"),
            (2.5, "2.5"),
            (999999.5, "999999.5"),
            (1000000.5, "1.0000005e6"),
            (123456789.25, "1.2345678925e8"),
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (100.0, "100.0"),
            (1.0e6, "1.0e6"),
            (1.0e22, "1.0e22"),
            (0.0001, "0.0001"),
            (1.0e-5, "1.0e-5"),
            (-2.5e-7, "-2.5e-7"),
            (2440587.5, "2.4405875e6"),
            (7.27220521664304e-5, "7.27220521664304e-5"),
            (5e-324, "5.0e-324"),
            (1.7976931348623157e308, "1.7976931348623157e308"),
        ] {
            assert_eq!(canonical_float(v), want, "for {v:e}");
        }
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
        Site {
            expr: e.clone(),
            key: ckey(e),
        }
    }
}

pub fn is_site(e: &Expr, s: &Site) -> bool {
    let candidate =
        matches!(e, Expr::Variable(_)) || matches!(e, Expr::Operator(n) if n.op == "index");
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
    earthsci_ast::free_variables(e).contains(v)
}

/// Rebuild an operator node with mapped children (Expr-level wrapper).
pub fn map_expr_children(e: &Expr, f: &mut impl FnMut(&Expr) -> Expr) -> Expr {
    match e {
        Expr::Operator(n) => Expr::operator(n.map_children(f)),
        _ => e.clone(),
    }
}
