//! Scalar derivative rules — one rule per evaluable-core operator, matching
//! esm-jacobian-spec.md §3 row-for-row (goldens pin the table).

use earthsci_ast::types::{Expr, ExpressionNode};
use serde_json::Value;

use crate::helpers::{add, is_site, is_zero, mul, neg, occurs, op, Site};

#[derive(Debug)]
pub struct DerivativeRuleError(pub String);

impl std::fmt::Display for DerivativeRuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DerivativeRuleError: {}", self.0)
    }
}
impl std::error::Error for DerivativeRuleError {}

type R = Result<Expr, DerivativeRuleError>;

const ZERO_DERIV_OPS: &[&str] = &[
    "<", "<=", ">", ">=", "==", "!=", "and", "or", "not", "sign", "floor",
    "ceil", "pi", "π", "e", "true", "false", "const", "enum", "Pre",
];

pub fn dscalar(e: &Expr, s: &Site) -> R {
    if is_site(e, s) {
        return Ok(Expr::Integer(1));
    }
    let Expr::Operator(n) = e else { return Ok(Expr::Integer(0)) };
    if !occurs(e, s) {
        return Ok(Expr::Integer(0));
    }
    let a = &n.args;
    let d = |x: &Expr| dscalar(x, s);
    let lit = Expr::Integer;
    match n.op.as_str() {
        "+" => Ok(add(a.iter().map(d).collect::<Result<Vec<_>, _>>()?)),
        "-" => {
            if a.len() == 1 {
                Ok(neg(d(&a[0])?))
            } else {
                Ok(add(vec![d(&a[0])?, neg(d(&a[1])?)]))
            }
        }
        "neg" => Ok(neg(d(&a[0])?)),
        "*" => {
            let mut terms = Vec::new();
            for (i, x) in a.iter().enumerate() {
                let dx = d(x)?;
                if is_zero(&dx) {
                    continue;
                }
                let mut fac = vec![dx];
                fac.extend(
                    a.iter().enumerate().filter(|(j, _)| *j != i)
                        .map(|(_, y)| y.clone()));
                terms.push(mul(fac));
            }
            Ok(add(terms))
        }
        "/" => {
            let (nu, de) = (&a[0], &a[1]);
            let (dn, dm) = (d(nu)?, d(de)?);
            let t1 = if is_zero(&dn) {
                lit(0)
            } else {
                op("/", vec![dn, de.clone()])
            };
            let t2 = if is_zero(&dm) {
                lit(0)
            } else {
                neg(op("/", vec![mul(vec![nu.clone(), dm]),
                                 op("^", vec![de.clone(), lit(2)])]))
            };
            Ok(add(vec![t1, t2]))
        }
        "^" | "pow" => {
            let (b, x) = (&a[0], &a[1]);
            let (db, dx) = (d(b)?, d(x)?);
            let t1 = if is_zero(&db) {
                lit(0)
            } else {
                mul(vec![x.clone(),
                         op("^", vec![b.clone(), add(vec![x.clone(), lit(-1)])]),
                         db])
            };
            let t2 = if is_zero(&dx) {
                lit(0)
            } else {
                mul(vec![op("^", vec![b.clone(), x.clone()]),
                         op("log", vec![b.clone()]), dx])
            };
            Ok(add(vec![t1, t2]))
        }
        "exp" => Ok(mul(vec![e.clone(), d(&a[0])?])),
        "log" => Ok(op("/", vec![d(&a[0])?, a[0].clone()])),
        "log10" => Ok(op("/", vec![d(&a[0])?,
                                   mul(vec![a[0].clone(),
                                            op("log", vec![lit(10)])])])),
        "sqrt" => Ok(op("/", vec![d(&a[0])?, mul(vec![lit(2), e.clone()])])),
        "sin" => Ok(mul(vec![op("cos", vec![a[0].clone()]), d(&a[0])?])),
        "cos" => Ok(neg(mul(vec![op("sin", vec![a[0].clone()]), d(&a[0])?]))),
        "tan" => Ok(mul(vec![add(vec![lit(1), op("^", vec![e.clone(), lit(2)])]),
                             d(&a[0])?])),
        "tanh" => Ok(mul(vec![
            add(vec![lit(1), neg(op("^", vec![e.clone(), lit(2)]))]),
            d(&a[0])?])),
        "sinh" => Ok(mul(vec![op("cosh", vec![a[0].clone()]), d(&a[0])?])),
        "cosh" => Ok(mul(vec![op("sinh", vec![a[0].clone()]), d(&a[0])?])),
        "asin" => Ok(op("/", vec![d(&a[0])?,
            op("sqrt", vec![add(vec![lit(1),
                neg(op("^", vec![a[0].clone(), lit(2)]))])])])),
        "acos" => Ok(neg(op("/", vec![d(&a[0])?,
            op("sqrt", vec![add(vec![lit(1),
                neg(op("^", vec![a[0].clone(), lit(2)]))])])]))),
        "asinh" => Ok(op("/", vec![d(&a[0])?,
            op("sqrt", vec![add(vec![op("^", vec![a[0].clone(), lit(2)]),
                                     lit(1)])])])),
        "acosh" => Ok(op("/", vec![d(&a[0])?,
            op("sqrt", vec![add(vec![op("^", vec![a[0].clone(), lit(2)]),
                                     lit(-1)])])])),
        "atanh" => Ok(op("/", vec![d(&a[0])?,
            add(vec![lit(1), neg(op("^", vec![a[0].clone(), lit(2)]))])])),
        "atan" if a.len() == 1 => Ok(op("/", vec![d(&a[0])?,
            add(vec![lit(1), op("^", vec![a[0].clone(), lit(2)])])])),
        "atan" | "atan2" => {
            let (y, x) = (&a[0], &a[1]);
            let den = add(vec![op("^", vec![x.clone(), lit(2)]),
                               op("^", vec![y.clone(), lit(2)])]);
            Ok(op("/", vec![add(vec![mul(vec![x.clone(), d(y)?]),
                                     neg(mul(vec![y.clone(), d(x)?]))]),
                            den]))
        }
        "abs" => Ok(mul(vec![
            op("ifelse", vec![op("<", vec![a[0].clone(), lit(0)]),
                              lit(-1), lit(1)]),
            d(&a[0])?])),
        "ifelse" => Ok(op("ifelse", vec![a[0].clone(), d(&a[1])?, d(&a[2])?])),
        "max" | "min" => {
            let cmp = if n.op == "max" { ">=" } else { "<=" };
            let mut acc = a[0].clone();
            let mut dacc = d(&a[0])?;
            for x in &a[1..] {
                dacc = op("ifelse",
                          vec![op(cmp, vec![acc.clone(), x.clone()]),
                               dacc, d(x)?]);
                acc = op(&n.op, vec![acc, x.clone()]);
            }
            Ok(dacc)
        }
        o if ZERO_DERIV_OPS.contains(&o) => Ok(lit(0)),
        "fn" => dfn(n, s),
        "index" => Err(DerivativeRuleError(
            "`index` of a non-variable array expression (or a site-dependent \
             index) has no scalar rule".into())),
        o => Err(DerivativeRuleError(format!(
            "no derivative rule for op `{o}` (rewrite-target ops must be \
             lowered before differentiation)"))),
    }
}

fn const_vector(e: &Expr) -> Option<Vec<f64>> {
    let Expr::Operator(n) = e else { return None };
    if n.op != "const" {
        return None;
    }
    let Some(Value::Array(v)) = &n.value else { return None };
    v.iter().map(|x| x.as_f64()).collect()
}

fn const_matrix(e: &Expr) -> Option<Vec<Vec<f64>>> {
    let Expr::Operator(n) = e else { return None };
    if n.op != "const" {
        return None;
    }
    let Some(Value::Array(v)) = &n.value else { return None };
    let rows: Option<Vec<Vec<f64>>> = v
        .iter()
        .map(|r| match r {
            Value::Array(a) => a.iter().map(|x| x.as_f64()).collect(),
            _ => None,
        })
        .collect();
    let rows = rows?;
    if rows.is_empty() || rows.iter().any(|r| r.len() != rows[0].len()) {
        return None;
    }
    Some(rows)
}

/// ∂(bilinear blend)/∂(the axis of `xv`) — see the reference rule (§3.4).
fn bilinear_partial(t: &[Vec<f64>], xv: &[f64], yv: &[f64],
                    xe: &Expr, ye: &Expr) -> Expr {
    let (nx, ny) = (xv.len(), yv.len());
    let mut out = Expr::Integer(0); // x ≥ axis_x[nx]: flat in x
    for i in (0..nx - 1).rev() {
        let sl = |j: usize| (t[i + 1][j] - t[i][j]) / (xv[i + 1] - xv[i]);
        let mut leaf = Expr::Number(sl(ny - 1)); // y ≥ axis_y[ny]: top edge
        for j in (0..ny - 1).rev() {
            let wy = op("/", vec![add(vec![ye.clone(),
                                           Expr::Number(-yv[j])]),
                                  Expr::Number(yv[j + 1] - yv[j])]);
            let ex = add(vec![Expr::Number(sl(j)),
                              mul(vec![wy,
                                       add(vec![Expr::Number(sl(j + 1)),
                                                neg(Expr::Number(sl(j)))])])]);
            leaf = op("ifelse",
                      vec![op("<", vec![ye.clone(), Expr::Number(yv[j + 1])]),
                           ex, leaf]);
        }
        leaf = op("ifelse",
                  vec![op("<", vec![ye.clone(), Expr::Number(yv[0])]),
                       Expr::Number(sl(0)), leaf]);
        out = op("ifelse",
                 vec![op("<", vec![xe.clone(), Expr::Number(xv[i + 1])]),
                      leaf, out]);
    }
    op("ifelse", vec![op("<", vec![xe.clone(), Expr::Number(xv[0])]),
                      Expr::Integer(0), out])
}

fn dfn(n: &ExpressionNode, s: &Site) -> R {
    let a = &n.args;
    let name = n.name.as_deref().ok_or_else(|| {
        DerivativeRuleError("`fn` node without a name".into())
    })?;
    if name.starts_with("datetime.") || name == "interp.searchsorted" {
        return Ok(Expr::Integer(0));
    }
    if name == "interp.linear" {
        let (table, axis, x) = (&a[0], &a[1], &a[2]);
        let dx = dscalar(x, s)?;
        if is_zero(&dx) {
            return Ok(Expr::Integer(0));
        }
        let (yv, xv) = (const_vector(table), const_vector(axis));
        let (Some(yv), Some(xv)) = (yv, xv) else {
            return Err(DerivativeRuleError(
                "interp.linear table/axis must be literal `const` arrays \
                 (spec §9.2 interp_table_not_const)".into()));
        };
        let nk = xv.len();
        if nk != yv.len() || nk < 2 {
            return Err(DerivativeRuleError(
                "interp.linear table/axis length mismatch or < 2 knots".into()));
        }
        let mut slope = Expr::Integer(0); // x ≥ axis[N]: flat
        for k in (0..nk - 1).rev() {
            let sk = (yv[k + 1] - yv[k]) / (xv[k + 1] - xv[k]);
            slope = op("ifelse",
                       vec![op("<", vec![x.clone(), Expr::Number(xv[k + 1])]),
                            Expr::Number(sk), slope]);
        }
        slope = op("ifelse",
                   vec![op("<", vec![x.clone(), Expr::Number(xv[0])]),
                        Expr::Integer(0), slope]);
        return Ok(mul(vec![slope, dx]));
    }
    if name == "interp.bilinear" {
        let (table, ax, ay, x, y) = (&a[0], &a[1], &a[2], &a[3], &a[4]);
        let dx = dscalar(x, s)?;
        let dy = dscalar(y, s)?;
        if is_zero(&dx) && is_zero(&dy) {
            return Ok(Expr::Integer(0));
        }
        let (t, xv, yv) = (const_matrix(table), const_vector(ax),
                           const_vector(ay));
        let (Some(t), Some(xv), Some(yv)) = (t, xv, yv) else {
            return Err(DerivativeRuleError(
                "interp.bilinear table/axes must be literal `const` arrays \
                 (spec §9.2 interp_table_not_const)".into()));
        };
        let (nx, ny) = (xv.len(), yv.len());
        if t.len() != nx || t.iter().any(|r| r.len() != ny) || nx < 2 || ny < 2 {
            return Err(DerivativeRuleError(
                "interp.bilinear table/axis shape mismatch or < 2 knots".into()));
        }
        let mut terms = Vec::new();
        if !is_zero(&dx) {
            terms.push(mul(vec![bilinear_partial(&t, &xv, &yv, x, y), dx]));
        }
        if !is_zero(&dy) {
            let tt: Vec<Vec<f64>> = (0..ny)
                .map(|j| (0..nx).map(|i| t[i][j]).collect())
                .collect();
            terms.push(mul(vec![bilinear_partial(&tt, &yv, &xv, y, x), dy]));
        }
        return Ok(add(terms));
    }
    Err(DerivativeRuleError(format!("unknown closed function `{name}`")))
}
