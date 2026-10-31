//! Derivatives through a faq node's WINDOW and REDUCTION (the reference's
//! reductions.jl, esm-jacobian-spec.md §3.5): filtered faq nodes, the
//! `max` / `min` / `*` reductions beside `+`, and reads whose index is a
//! value (a level search used as a subscript).
//!
//! A filter is a gate whose Dirac term is dropped, exactly like an `ifelse`
//! condition (§3.2), so the derivative of a filtered reduction is a `+` over
//! the same window of each admitted term's derivative, weighted by what the
//! reduction does with that term:
//!
//! ```text
//!   +        1
//!   max/min  the indicator of the FIRST extremal term in ascending order
//!            (the left fold's tie rule), built from two max/min faq nodes
//!   *        the product of every OTHER admitted term (never a division)
//! ```
//!
//! An index-only filter conjunct is decided exactly — by splitting the
//! band's box at its affine breakpoints, and by the band's `where` guard for
//! what is left (a triangular window); a value-dependent conjunct stays in
//! the coefficient as `ifelse(φ, d, 0)` and its pattern is the whole window.

use std::collections::{BTreeMap, HashMap, HashSet};

use earthsci_ast::substitute;
use earthsci_ast::{Expr, ExpressionNode};

use crate::bands::{subst_free, BandError, Ctx};
use crate::clip::{clip_regions, decide, RIdx};
use crate::helpers::{free_vars, is_zero, mul, op};
use crate::inline::all_names;

/// The ⊕ of faq node `n` (`"+"`, `"*"`, `"max"` or `"min"`); a `semiring`
/// supersedes `reduce`, and its ⊗ is already written out in the body. A join
/// or a value-invention node has no rule. The error is the bare message; the
/// caller wraps it in its own error type.
pub fn reduce_op(n: &ExpressionNode) -> Result<&'static str, String> {
    if n.join.is_some() {
        return Err("joined faq nodes are not differentiated".into());
    }
    if n.distinct == Some(true) || n.key.is_some() {
        return Err("value-invention faq nodes (`distinct`, `key`) are not differentiated".into());
    }
    match n.semiring.as_deref() {
        None => match n.reduce.as_deref().unwrap_or("+") {
            "+" => return Ok("+"),
            "*" => return Ok("*"),
            "max" => return Ok("max"),
            "min" => return Ok("min"),
            _ => {}
        },
        Some("sum_product") => return Ok("+"),
        Some("max_product") | Some("max_sum") => return Ok("max"),
        Some("min_sum") => return Ok("min"),
        Some(_) => {}
    }
    Err(format!(
        "{} has no derivative rule",
        crate::rules::reduction(n)
    ))
}

/// The conjuncts of `g`: its `and` arguments, flattened.
pub(crate) fn conjuncts(g: &Expr) -> Vec<Expr> {
    match g {
        Expr::Operator(n) if n.op == "and" => n.args.iter().flat_map(conjuncts).collect(),
        _ => vec![g.clone()],
    }
}

/// The conjunction of `xs`: `None` when empty, the one conjunct, or an `and`.
pub(crate) fn conj_list(mut xs: Vec<Expr>) -> Option<Expr> {
    match xs.len() {
        0 => None,
        1 => xs.pop(),
        _ => Some(op("and", xs)),
    }
}

/// `a ∧ b`, either possibly absent; conjuncts flattened.
pub(crate) fn conj(a: Option<Expr>, b: Option<Expr>) -> Option<Expr> {
    match (a, b) {
        (None, b) => b,
        (a, None) => a,
        (Some(a), Some(b)) => {
            let mut xs = conjuncts(&a);
            xs.extend(conjuncts(&b));
            conj_list(xs)
        }
    }
}

/// Index-only: reads no name but the given index names (literals and
/// `const` gathers carry none), so it can be decided at assembly, point by
/// point.
pub(crate) fn index_only<S: AsRef<str>>(g: &Expr, names: &[S]) -> bool {
    free_vars(g)
        .iter()
        .all(|v| names.iter().any(|n| n.as_ref() == v))
}

/// `base`, then `suffix` appended until clear of `taken`, which it joins.
pub(crate) fn fresh(taken: &mut HashSet<String>, base: &str, suffix: &str) -> String {
    let mut f = base.to_string();
    while taken.contains(&f) {
        f.push_str(suffix);
    }
    taken.insert(f.clone());
    f
}

fn scalar_faq(
    reduce: &str,
    ranges: HashMap<String, earthsci_ast::extension::types::RangeSpec>,
    filter: Option<Expr>,
    body: Expr,
) -> Expr {
    Expr::operator(ExpressionNode {
        op: "faq".into(),
        output_idx: Some(vec![]),
        reduce: Some(reduce.into()),
        ranges: Some(ranges),
        filter: filter.map(Box::new),
        expr: Some(Box::new(body)),
        ..Default::default()
    })
}

/// The derivative `d` of one term of the reduction `red` of faq node `n`
/// over its contracted names `cn`, weighted by what the reduction does with
/// that term (see the module header). The max/min indicator and the
/// exclusion product read the node's whole window — its own ranges and
/// filter — through copies of the body with the contracted names renamed
/// apart (`<name>_p`, `_p` repeated until clear of every name in `n`), so no
/// node they add binds a name the term's own contraction binds. A one-point
/// window (no contracted name) weighs 1. A max/min over more than one
/// contracted name has no single ascending order, so it is refused.
pub(crate) fn window_weight(
    red: &str,
    n: &ExpressionNode,
    cn: &[String],
    d: Expr,
) -> Result<Expr, String> {
    if red == "+" || cn.is_empty() || is_zero(&d) {
        return Ok(d);
    }
    if red != "*" && cn.len() != 1 {
        return Err(format!(
            "a `{red}` reduction over {} contracted indices ({}) has no first-extremum order",
            cn.len(),
            cn.join(", ")
        ));
    }
    let empty = HashMap::new();
    let rng = n.ranges.as_ref().unwrap_or(&empty);
    let mut taken = HashSet::new();
    all_names(&Expr::operator(n.clone()), &mut taken);
    let mut ren: HashMap<String, Expr> = HashMap::new();
    let mut renamed: Vec<String> = Vec::new();
    for c in cn {
        let f = fresh(&mut taken, &format!("{c}_p"), "_p");
        ren.insert(c.clone(), Expr::Variable(f.clone()));
        renamed.push(f);
    }
    let body = n
        .expr
        .as_deref()
        .ok_or_else(|| "faq without body".to_string())?;
    let b2 = subst_free(body, &ren);
    let phi2 = n.filter.as_deref().map(|f| subst_free(f, &ren));
    let mut r2 = HashMap::new();
    for (c, f) in cn.iter().zip(&renamed) {
        let r = rng
            .get(c)
            .ok_or_else(|| format!("contracted index `{c}` has no range"))?;
        r2.insert(f.clone(), r.clone());
    }
    if red == "*" {
        let mut ne: Vec<Expr> = cn
            .iter()
            .zip(&renamed)
            .map(|(c, f)| {
                op(
                    "!=",
                    vec![Expr::Variable(f.clone()), Expr::Variable(c.clone())],
                )
            })
            .collect();
        let ne = if ne.len() == 1 {
            ne.pop().unwrap()
        } else {
            op("or", ne)
        };
        let others = scalar_faq("*", r2, conj(phi2, Some(ne)), b2);
        return Ok(mul(vec![d, others]));
    }
    let c = &cn[0];
    let m = scalar_faq(red, r2.clone(), phi2.clone(), b2.clone());
    let m_before = scalar_faq(
        red,
        r2,
        conj(
            phi2,
            Some(op(
                "<",
                vec![
                    Expr::Variable(renamed[0].clone()),
                    Expr::Variable(c.clone()),
                ],
            )),
        ),
        b2,
    );
    let (ge, gt) = if red == "max" {
        (">=", ">")
    } else {
        ("<=", "<")
    };
    let first_ext = op(
        "and",
        vec![
            op(ge, vec![body.clone(), m]),
            op(gt, vec![body.clone(), m_before]),
        ],
    );
    Ok(op("ifelse", vec![first_ext, d, Expr::Integer(0)]))
}

/// The truth of a decided predicate: `Ok(bool)`, or the residual
/// expression as `Err`.
pub(crate) fn bool_fold(g: &Expr) -> Result<bool, Expr> {
    match g {
        Expr::Integer(i) => Ok(*i != 0),
        Expr::Number(x) => Ok(*x != 0.0),
        // A comparison of literals — what pinning one-value names leaves of
        // `k != i` (`decide` folds only the affine `<`/`<=`/`>`/`>=`).
        Expr::Operator(n)
            if n.args.len() == 2
                && matches!(n.op.as_str(), "<" | "<=" | ">" | ">=" | "==" | "!=")
                && lit_value(&n.args[0]).is_some()
                && lit_value(&n.args[1]).is_some() =>
        {
            let (a, b) = (
                lit_value(&n.args[0]).unwrap(),
                lit_value(&n.args[1]).unwrap(),
            );
            Ok(match n.op.as_str() {
                "<" => a < b,
                "<=" => a <= b,
                ">" => a > b,
                ">=" => a >= b,
                "==" => a == b,
                _ => a != b,
            })
        }
        Expr::Operator(n) if n.op == "and" || n.op == "or" => {
            let absorb = n.op == "or"; // the value that decides it
            let mut rest = Vec::new();
            for x in &n.args {
                match bool_fold(x) {
                    Ok(t) if t == absorb => return Ok(absorb),
                    Ok(_) => continue,
                    Err(r) => rest.push(r),
                }
            }
            match rest.len() {
                0 => Ok(!absorb),
                1 => Err(rest.pop().unwrap()),
                _ => Err(op(&n.op, rest)),
            }
        }
        Expr::Operator(n) if n.op == "not" && n.args.len() == 1 => match bool_fold(&n.args[0]) {
            Ok(t) => Ok(!t),
            Err(r) => Err(op("not", vec![r])),
        },
        Expr::Operator(n) if n.op == "true" => Ok(true),
        Expr::Operator(n) if n.op == "false" => Ok(false),
        _ => Err(g.clone()),
    }
}

/// The value of a literal, or of `+` / `-` / `*` over literals.
fn lit_value(e: &Expr) -> Option<f64> {
    match e {
        Expr::Integer(i) => Some(*i as f64),
        Expr::Number(x) => Some(*x),
        Expr::Operator(n) if matches!(n.op.as_str(), "+" | "-" | "*") && !n.args.is_empty() => {
            let xs: Vec<f64> = n.args.iter().map(lit_value).collect::<Option<_>>()?;
            match (n.op.as_str(), xs.as_slice()) {
                ("+", _) => Some(xs.iter().sum()),
                ("*", _) => Some(xs.iter().product()),
                (_, [a]) => Some(-a),
                (_, [a, b]) => Some(a - b),
                _ => None,
            }
        }
        _ => None,
    }
}

/// A box cell with its residual guard (`None`: true throughout).
pub(crate) type GuardCell = (Vec<(i64, i64)>, Option<Expr>);

/// Decide the index-only predicate `g` over the box `names[k] ∈ bx[k]` as
/// far as the box can: pin one-value names, split the box at the
/// predicate's affine breakpoints (the region clipper's), and decide each
/// cell — dropped where `g` is false throughout, unguarded (`None`) where it
/// is true throughout, and guarded by the undecided residual otherwise (a
/// window no box can express, such as `gk < gke`). Cells are in the
/// clipper's ascending order.
pub(crate) fn split_guard(g: &Expr, names: &[String], bx: &[(i64, i64)]) -> Vec<GuardCell> {
    let pins: HashMap<String, Expr> = names
        .iter()
        .zip(bx)
        .filter(|(_, (lo, hi))| lo == hi)
        .map(|(n, (lo, _))| (n.clone(), Expr::Integer(*lo)))
        .collect();
    let g = if pins.is_empty() {
        g.clone()
    } else {
        substitute(g, &pins)
    };
    let ridx: Vec<RIdx> = names.iter().map(|n| RIdx::Name(n.clone())).collect();
    let mut out = Vec::new();
    for (cell, g2) in clip_regions(&g, &ridx, bx) {
        let env: BTreeMap<String, (i64, i64)> = names
            .iter()
            .zip(&cell)
            .map(|(n, c)| (n.clone(), *c))
            .collect();
        match bool_fold(&decide(&g2, &env)) {
            Ok(false) => continue,
            Ok(true) => out.push((cell, None)),
            Err(r) => out.push((cell, Some(r))),
        }
    }
    out
}

/// A one-hot band: column indices, coefficient and contracted dimensions.
pub(crate) type OneHot = (Vec<Expr>, Expr, Vec<(String, i64, i64)>);

/// A column index that reads a value — `x[κ(u) + d]` with `κ` a level
/// search — is integer-valued, hence piecewise constant (the §3.2 `floor`
/// rule): its own derivative is 0, and the read is the one-hot sum
/// `Σ_k [k == κ(u)+d] x[k]`. Each such dimension becomes a contracted name
/// (`_oh<dim>`, `_j` appended until clear) over the static extent of that
/// axis of `v`, and the selection moves into the coefficient. The pattern is
/// the structural union: every cell of the axis. A column reading only the
/// band's own index names is unchanged.
pub(crate) fn one_hot(
    cidx: Vec<Expr>,
    coef: Expr,
    contracted: Vec<(String, i64, i64)>,
    ridx: &[RIdx],
    v: &str,
    ctx: &Ctx,
) -> Result<OneHot, BandError> {
    let mut bound: Vec<String> = ridx
        .iter()
        .filter_map(|r| match r {
            RIdx::Name(n) => Some(n.clone()),
            RIdx::Fixed(_) => None,
        })
        .collect();
    bound.extend(contracted.iter().map(|(n, _, _)| n.clone()));
    let hot: Vec<usize> = cidx
        .iter()
        .enumerate()
        .filter(|(_, c)| !index_only(c, &bound))
        .map(|(k, _)| k)
        .collect();
    if hot.is_empty() {
        return Ok((cidx, coef, contracted));
    }
    let shape = ctx.shapes.get(v).cloned().unwrap_or_default();
    let mut taken: HashSet<String> = bound.iter().cloned().collect();
    all_names(&coef, &mut taken);
    for c in &cidx {
        all_names(c, &mut taken);
    }
    let (mut cidx, mut coef, mut contracted) = (cidx, coef, contracted);
    for k in hot {
        let n = *shape.get(k).ok_or_else(|| {
            BandError(format!(
                "column dimension {} of `{v}` reads a value, but `{v}` has no static \
                 extent along it",
                k + 1
            ))
        })?;
        let f = fresh(&mut taken, &format!("_oh{}", k + 1), "_j");
        coef = op(
            "ifelse",
            vec![
                op("==", vec![Expr::Variable(f.clone()), cidx[k].clone()]),
                coef,
                Expr::Integer(0),
            ],
        );
        cidx[k] = Expr::Variable(f.clone());
        contracted.push((f, 1, n));
    }
    Ok((cidx, coef, contracted))
}

#[cfg(test)]
mod tests {
    use super::split_guard;
    use crate::helpers::op;
    use earthsci_ast::Expr;

    fn var(n: &str) -> Expr {
        Expr::Variable(n.into())
    }

    #[test]
    fn pinned_literal_comparisons_are_decided() {
        let names = ["k".to_string(), "i".to_string()];
        let ne = op("!=", vec![var("k"), var("i")]);
        assert!(split_guard(&ne, &names, &[(3, 3), (3, 3)]).is_empty());
        let eq = op(
            "==",
            vec![var("k"), op("+", vec![var("i"), Expr::Integer(1)])],
        );
        assert_eq!(
            split_guard(&eq, &names, &[(4, 4), (3, 3)]),
            vec![(vec![(4, 4), (3, 3)], None)]
        );
    }
}
