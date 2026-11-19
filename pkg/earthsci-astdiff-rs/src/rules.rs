//! Scalar derivative rules — one rule per evaluable-core operator, matching
//! esm-jacobian-spec.md §3 row-for-row (goldens pin the table).

use std::collections::{HashMap, HashSet};

use earthsci_ast::extension::types::RangeSpec;
use earthsci_ast::{Expr, ExpressionNode};

use crate::hashcons::NodeRef;
use crate::helpers::{add, free_vars, is_zero, mul, neg, op, Site, SiteScan};

/// A scalar operator has no derivative rule (spec §3).
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
    "<", "<=", ">", ">=", "==", "!=", "and", "or", "not", "sign", "floor", "ceil", "pi", "π", "e",
    "true", "false", "const", "enum",
];

/// The derivative of the scalar expression `e` with respect to the site `s`
/// (a variable or an indexed read), by the §3 rule table.
///
/// A one-shot call scans plainly: occurrence by `occurs`, which stops at the
/// first hit, and a derivative memo private to the call. Building a shared
/// occurrence index (a pass over the whole tree with a bit set per node)
/// pays only when many scans share it, as a band computation's do.
pub fn dscalar(e: &Expr, s: &Site) -> R {
    dscalar_in(e, s, &SiteScan::plain())
}

/// [`dscalar`] for one scan of a band computation: occurrence answered by
/// `sc` (one [`OccIndex`] shared by every site and region of the
/// computation), and a node reached through several parents, or shared by
/// several regions, differentiated once.
pub(crate) fn dscalar_in(e: &Expr, s: &Site, sc: &SiteScan) -> R {
    if sc.is_site(e, s) {
        return Ok(Expr::Integer(1));
    }
    let Expr::Operator(n) = e else {
        return Ok(Expr::Integer(0));
    };
    if !sc.occurs(e, s) {
        return Ok(Expr::Integer(0));
    }
    if let Some(d) = sc.memo_get(n) {
        return Ok(d);
    }
    let d = dscalar_op(e, n, s, sc)?;
    sc.memo_put(n, d.clone());
    Ok(d)
}

fn dscalar_op(e: &Expr, n: &ExpressionNode, s: &Site, sc: &SiteScan) -> R {
    let a = &n.args;
    let d = |x: &Expr| dscalar_in(x, s, sc);
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
        // the identity in a right-hand side (scalar_rules.jl): d[a]
        "Pre" => d(&a[0]),
        "*" => {
            let mut terms = Vec::new();
            for (i, x) in a.iter().enumerate() {
                let dx = d(x)?;
                if is_zero(&dx) {
                    continue;
                }
                let mut fac = vec![dx];
                fac.extend(
                    a.iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, y)| y.clone()),
                );
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
                neg(op(
                    "/",
                    vec![mul(vec![nu.clone(), dm]), op("^", vec![de.clone(), lit(2)])],
                ))
            };
            Ok(add(vec![t1, t2]))
        }
        "^" | "pow" => {
            let (b, x) = (&a[0], &a[1]);
            let (db, dx) = (d(b)?, d(x)?);
            let t1 = if is_zero(&db) {
                lit(0)
            } else {
                mul(vec![
                    x.clone(),
                    op("^", vec![b.clone(), add(vec![x.clone(), lit(-1)])]),
                    db,
                ])
            };
            let t2 = if is_zero(&dx) {
                lit(0)
            } else {
                mul(vec![
                    op("^", vec![b.clone(), x.clone()]),
                    op("log", vec![b.clone()]),
                    dx,
                ])
            };
            Ok(add(vec![t1, t2]))
        }
        "exp" => Ok(mul(vec![e.clone(), d(&a[0])?])),
        "log" => Ok(op("/", vec![d(&a[0])?, a[0].clone()])),
        "log10" => Ok(op(
            "/",
            vec![d(&a[0])?, mul(vec![a[0].clone(), op("log", vec![lit(10)])])],
        )),
        "sqrt" => Ok(op("/", vec![d(&a[0])?, mul(vec![lit(2), e.clone()])])),
        "sin" => Ok(mul(vec![op("cos", vec![a[0].clone()]), d(&a[0])?])),
        "cos" => Ok(neg(mul(vec![op("sin", vec![a[0].clone()]), d(&a[0])?]))),
        "tan" => Ok(mul(vec![
            add(vec![lit(1), op("^", vec![e.clone(), lit(2)])]),
            d(&a[0])?,
        ])),
        "tanh" => Ok(mul(vec![
            add(vec![lit(1), neg(op("^", vec![e.clone(), lit(2)]))]),
            d(&a[0])?,
        ])),
        "sinh" => Ok(mul(vec![op("cosh", vec![a[0].clone()]), d(&a[0])?])),
        "cosh" => Ok(mul(vec![op("sinh", vec![a[0].clone()]), d(&a[0])?])),
        "asin" => Ok(op(
            "/",
            vec![
                d(&a[0])?,
                op(
                    "sqrt",
                    vec![add(vec![lit(1), neg(op("^", vec![a[0].clone(), lit(2)]))])],
                ),
            ],
        )),
        "acos" => Ok(neg(op(
            "/",
            vec![
                d(&a[0])?,
                op(
                    "sqrt",
                    vec![add(vec![lit(1), neg(op("^", vec![a[0].clone(), lit(2)]))])],
                ),
            ],
        ))),
        "asinh" => Ok(op(
            "/",
            vec![
                d(&a[0])?,
                op(
                    "sqrt",
                    vec![add(vec![op("^", vec![a[0].clone(), lit(2)]), lit(1)])],
                ),
            ],
        )),
        "acosh" => Ok(op(
            "/",
            vec![
                d(&a[0])?,
                op(
                    "sqrt",
                    vec![add(vec![op("^", vec![a[0].clone(), lit(2)]), lit(-1)])],
                ),
            ],
        )),
        "atanh" => Ok(op(
            "/",
            vec![
                d(&a[0])?,
                add(vec![lit(1), neg(op("^", vec![a[0].clone(), lit(2)]))]),
            ],
        )),
        "atan" if a.len() == 1 => Ok(op(
            "/",
            vec![
                d(&a[0])?,
                add(vec![lit(1), op("^", vec![a[0].clone(), lit(2)])]),
            ],
        )),
        "atan" | "atan2" => {
            let (y, x) = (&a[0], &a[1]);
            let den = add(vec![
                op("^", vec![x.clone(), lit(2)]),
                op("^", vec![y.clone(), lit(2)]),
            ]);
            Ok(op(
                "/",
                vec![
                    add(vec![
                        mul(vec![x.clone(), d(y)?]),
                        neg(mul(vec![y.clone(), d(x)?])),
                    ]),
                    den,
                ],
            ))
        }
        "abs" => Ok(mul(vec![
            op(
                "ifelse",
                vec![op("<", vec![a[0].clone(), lit(0)]), lit(-1), lit(1)],
            ),
            d(&a[0])?,
        ])),
        "ifelse" => Ok(op("ifelse", vec![a[0].clone(), d(&a[1])?, d(&a[2])?])),
        "max" | "min" => {
            let cmp = if n.op == "max" { ">=" } else { "<=" };
            let mut acc = a[0].clone();
            let mut dacc = d(&a[0])?;
            for x in &a[1..] {
                dacc = op(
                    "ifelse",
                    vec![op(cmp, vec![acc.clone(), x.clone()]), dacc, d(x)?],
                );
                acc = op(&n.op, vec![acc, x.clone()]);
            }
            Ok(dacc)
        }
        "faq" if n.output_idx.as_ref().is_none_or(|o| o.is_empty()) => dsum(n, s, sc),
        o if ZERO_DERIV_OPS.contains(&o) => Ok(lit(0)),
        // A site-bearing `index` that is not itself the site. Its subscripts
        // are integer-valued, so a read of a literal array (an expanded
        // template's `const` table or axis) is piecewise-constant in them: 0.
        // A read of a variable (which may be the site's own, or a recurrence
        // array that depends on it) at a site-dependent subscript, or of an
        // array expression that reads variables, has no rule.
        "index" if index_is_literal_read(n) => Ok(lit(0)),
        "index" => Err(DerivativeRuleError(
            "`index` of a non-variable array expression (or a site-dependent \
             index) has no scalar rule"
                .into(),
        )),
        o => Err(DerivativeRuleError(format!(
            "no derivative rule for op `{o}` (rewrite-target ops must be \
             lowered before differentiation)"
        ))),
    }
}

/// The rule check of [`dscalar_in`] without the derivative, for the
/// structure-only mode (spec §2.1; scalar_rules.jl `drule_check`): fails with
/// exactly the [`DerivativeRuleError`] `dscalar_in` would, and is `false`
/// exactly when `dscalar_in` would return a literal 0, so a check the full
/// mode skips on a zero derivative (`window_weight`'s) is skipped here too.
/// It visits the nodes `dscalar_in` visits and builds no derivative.
pub(crate) fn drule_check(e: &Expr, s: &Site, sc: &SiteScan) -> Result<bool, DerivativeRuleError> {
    drule_check_in(e, s, sc, &mut HashMap::new())
}

fn drule_check_in(
    e: &Expr,
    s: &Site,
    sc: &SiteScan,
    memo: &mut HashMap<NodeRef, bool>,
) -> Result<bool, DerivativeRuleError> {
    if sc.is_site(e, s) {
        return Ok(true);
    }
    let Expr::Operator(n) = e else {
        return Ok(false);
    };
    if !sc.occurs(e, s) {
        return Ok(false);
    }
    let k = NodeRef(n.clone());
    if let Some(&nz) = memo.get(&k) {
        return Ok(nz);
    }
    let nz = drule_check_op(e, n, s, sc, memo)?;
    memo.insert(k, nz);
    Ok(nz)
}

/// [`dscalar_op`], branch for branch: every child it differentiates is
/// checked (all of them, never a short-circuit, so no rule error is
/// missed), and the result follows the literal-zero folding of `add`, `mul`
/// and `neg` there — an `op(…)` node it builds is never a literal 0.
fn drule_check_op(
    e: &Expr,
    n: &ExpressionNode,
    s: &Site,
    sc: &SiteScan,
    memo: &mut HashMap<NodeRef, bool>,
) -> Result<bool, DerivativeRuleError> {
    let a = &n.args;
    let mut d = |x: &Expr| drule_check_in(x, s, sc, memo);
    match n.op.as_str() {
        "+" | "-" | "neg" | "/" => {
            let mut any = false;
            for x in a {
                any |= d(x)?;
            }
            Ok(any)
        }
        "*" => {
            let mut any = false;
            for (i, x) in a.iter().enumerate() {
                let dx = d(x)?;
                any |= dx && !a.iter().enumerate().any(|(j, y)| j != i && is_zero(y));
            }
            Ok(any)
        }
        "^" | "pow" => {
            let (db, dx) = (d(&a[0])?, d(&a[1])?);
            Ok((db && !is_zero(&a[1])) || dx)
        }
        "exp" | "sin" | "cos" | "tan" | "tanh" | "sinh" | "cosh" | "abs" => d(&a[0]),
        "log" | "log10" | "sqrt" | "asin" | "acos" | "asinh" | "acosh" | "atanh" | "atan"
        | "atan2" => {
            for x in a {
                d(x)?;
            }
            Ok(true)
        }
        "ifelse" => {
            d(&a[1])?;
            d(&a[2])?;
            Ok(true)
        }
        "max" | "min" => {
            let mut first = false;
            for (i, x) in a.iter().enumerate() {
                let nz = d(x)?;
                if i == 0 {
                    first = nz;
                }
            }
            Ok(a.len() > 1 || first)
        }
        "faq" if n.output_idx.as_ref().is_none_or(|o| o.is_empty()) => {
            let red = crate::reductions::reduce_op(n).map_err(DerivativeRuleError)?;
            let body = n
                .expr
                .as_deref()
                .ok_or_else(|| DerivativeRuleError("faq without body".into()))?;
            if !d(body)? {
                return Ok(false);
            }
            let mut names: Vec<String> = n
                .ranges
                .as_ref()
                .map(|r| r.keys().cloned().collect())
                .unwrap_or_default();
            names.sort();
            crate::reductions::check_window(red, &names).map_err(DerivativeRuleError)?;
            Ok(true)
        }
        o if ZERO_DERIV_OPS.contains(&o) => Ok(false),
        // `index`, and every op without a rule: `dscalar_op` fails.
        _ => Ok(!is_zero(&dscalar_op(e, n, s, sc)?)),
    }
}

/// A faq node's reduction, for messages ([`crate::reductions::reduce_op`]
/// decides it).
pub(crate) fn reduction(n: &ExpressionNode) -> String {
    match &n.semiring {
        None => format!("reduce=`{}`", n.reduce.as_deref().unwrap_or("+")),
        Some(sr) => format!("semiring=`{sr}`"),
    }
}

/// A scalar faq node ⊕_k body (scalar_rules.jl `_dsum`): the `+` of its
/// body's derivatives, each weighted by what ⊕ does with that term
/// ([`crate::reductions::window_weight`]). Bound indices the site's column
/// does not read are summed (a rebuilt `+` node, its filter kept); those it
/// does read (`u[i,k]` under Σ_k) are left FREE, for the band emitter to
/// turn into contracted column dimensions (bands.rs `lift_nested`), and once
/// every index is free the filter becomes `ifelse(φ, ·, 0)`. A join or a
/// value-invention node has no rule (spec §3.5).
fn dsum(n: &ExpressionNode, s: &Site, sc: &SiteScan) -> R {
    let red = crate::reductions::reduce_op(n).map_err(DerivativeRuleError)?;
    let body = n
        .expr
        .as_deref()
        .ok_or_else(|| DerivativeRuleError("faq without body".into()))?;
    let db = dscalar_in(body, s, sc)?;
    if is_zero(&db) {
        return Ok(Expr::Integer(0));
    }
    let empty = HashMap::new();
    let ranges = n.ranges.as_ref().unwrap_or(&empty);
    let mut names: Vec<String> = ranges.keys().cloned().collect();
    names.sort();
    let db = crate::reductions::window_weight(red, n, &names, db).map_err(DerivativeRuleError)?;
    // A rebuilt `+` node: the reduction is now a plain sum of weighted terms.
    let plus = |db: Expr, ranges: Option<HashMap<String, RangeSpec>>| {
        let mut m = n.clone();
        m.expr = Some(Box::new(db));
        if let Some(r) = ranges {
            m.ranges = Some(r);
        }
        if red != "+" {
            m.reduce = Some("+".into());
            m.semiring = None;
        }
        Expr::operator(m)
    };
    let mut read: HashSet<String> = HashSet::new();
    if let Expr::Operator(sn) = &s.expr {
        for x in &sn.args[1..] {
            read.extend(free_vars(x));
        }
    }
    let summed: HashMap<_, _> = ranges
        .iter()
        .filter(|(k, _)| !read.contains(*k))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    if summed.len() == ranges.len() {
        return Ok(plus(db, None));
    }
    if summed.is_empty() {
        return Ok(match &n.filter {
            None => db,
            Some(f) => op("ifelse", vec![(**f).clone(), db, Expr::Integer(0)]),
        });
    }
    Ok(plus(db, Some(summed)))
}

/// Does the `index` node `n` read a literal array (one that reads no
/// variable)? Its derivative is then 0 (spec §3.2).
fn index_is_literal_read(n: &ExpressionNode) -> bool {
    matches!(n.args.first(), Some(a @ Expr::Operator(_)) if free_vars(a).is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::{is_zero, OccIndex};
    use std::cell::RefCell;

    fn ex(v: serde_json::Value) -> Expr {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn index_of_a_literal_array() {
        use serde_json::json;
        let s = Site::of(&Expr::Variable("x".into()));
        // `index` of a literal array is piecewise-constant in its integer
        // subscripts: derivative 0 (spec §3.2)
        let tbl = json!({"op": "const", "args": [], "value": [1.0, 3.0, 2.0]});
        let fx = json!({"op": "floor", "args": ["x"]});
        let e = ex(json!({"op": "index", "args": [tbl, fx]}));
        assert!(is_zero(&dscalar(&e, &s).unwrap()));
        // ... but a site-dependent read of a variable has no rule (another
        // variable may depend on the site: a recurrence array is never
        // inlined), nor a read of an array expression that reads variables
        let q = ex(json!({"op": "index", "args": ["q", fx]}));
        assert!(dscalar(&q, &s).is_err());
        let ui = Site::of(&ex(json!({"op": "index", "args": ["u", 1]})));
        let uu = ex(json!({"op": "index", "args": ["u",
            {"op": "floor", "args": [{"op": "index", "args": ["u", 1]}]}]}));
        assert!(dscalar(&uu, &ui).is_err());
        let ay = ex(json!({"op": "index", "args": [{"op": "+", "args": [tbl, "y"]}, fx]}));
        assert!(dscalar(&ay, &s).is_err());
    }

    #[test]
    fn pre_is_the_identity() {
        // `Pre(x)` differentiates as `x` (spec §3.2): the identity in a
        // right-hand side, as EarthSciAST evaluates it, not a constant
        let s = Site::of(&Expr::Variable("x".into()));
        let pre = ex(serde_json::json!({"op": "Pre", "args": ["x"]}));
        assert_eq!(dscalar(&pre, &s).unwrap(), Expr::Integer(1));
    }

    /// `Pre` composed with the product, chain, power and extremum rules,
    /// against central differences. The expression and its derivative are
    /// evaluated after `inline_observed` lowers `Pre(a)` to `a` (the step
    /// before differentiation in the band calculus), but the derivative is
    /// taken of the UNLOWERED expression, so the rule is what is tested.
    #[test]
    fn pre_rule_vs_fd() {
        use crate::inline::inline_observed;
        use std::collections::HashMap;
        let s = Site::of(&Expr::Variable("x".into()));
        let pre = |a: serde_json::Value| serde_json::json!({"op": "Pre", "args": [a]});
        let at = |e: &Expr, x: f64| {
            let vars = HashMap::from([("x".to_string(), x)]);
            earthsci_ast::evaluate(e, &vars).unwrap()
        };
        let cases = [
            (
                serde_json::json!({"op": "*", "args": [pre("x".into()), "x"]}),
                0.7,
            ),
            (
                serde_json::json!({"op": "exp", "args": [pre("x".into())]}),
                0.7,
            ),
            (
                serde_json::json!({"op": "^", "args": [pre("x".into()), 3]}),
                1.3,
            ),
            (
                serde_json::json!({"op": "*", "args": [{"op": "sin", "args": [pre("x".into())]}, "x"]}),
                0.9,
            ),
            (
                serde_json::json!({"op": "max", "args": [{"op": "*", "args": [2.0, pre("x".into())]},
                                                       {"op": "*", "args": ["x", "x"]}]}),
                0.6,
            ),
            (
                serde_json::json!({"op": "max", "args": [{"op": "*", "args": [2.0, pre("x".into())]},
                                                       {"op": "*", "args": ["x", "x"]}]}),
                2.5,
            ),
            (
                serde_json::json!({"op": "/", "args": [pre(pre("x".into())),
                                                     {"op": "+", "args": ["x", 2.0]}]}),
                0.6,
            ),
        ];
        let none = HashMap::new();
        for (data, x) in cases {
            let e = ex(data);
            let lowered = inline_observed(&e, &none).unwrap();
            assert!(!serde_json::to_string(&lowered).unwrap().contains("\"Pre\""));
            let d = inline_observed(&dscalar(&e, &s).unwrap(), &none).unwrap();
            let h = 1e-6;
            let fd = (at(&lowered, x + h) - at(&lowered, x - h)) / (2.0 * h);
            let sym = at(&d, x);
            assert!((sym - fd).abs() < 1e-6, "{e:?} at {x}: {sym} vs {fd}");
        }
    }

    /// `drule_check` fails where `dscalar` fails, and is false exactly where
    /// `dscalar` returns a literal 0 (scalar_rules.jl `drule_check`).
    #[test]
    fn drule_check_agrees_with_dscalar() {
        use serde_json::json;
        let x = Expr::Variable("x".into());
        let s = Site::of(&x);
        let ix = RefCell::new(OccIndex::new("x"));
        let chk = |e: &Expr| drule_check(e, &s, &SiteScan::new(&ix, &s));
        let tab = json!({"op": "const", "args": [], "value": [0.0, 1.0, 4.0]});
        let o = |op: &str, a: Vec<serde_json::Value>| json!({"op": op, "args": a});
        let sign = o("sign", vec![json!("x")]);
        for v in [
            json!("x"),
            json!("y"),
            json!(0),
            o("+", vec![json!("x"), json!("y")]),
            o("-", vec![json!("x")]),
            o("-", vec![json!("y"), json!("x")]),
            o("neg", vec![json!("y")]),
            o("*", vec![json!("x"), json!("y")]),
            o("*", vec![json!(0), json!("x")]),
            o("*", vec![sign.clone(), json!("y")]),
            o("/", vec![json!("y"), json!("x")]),
            o("/", vec![o("floor", vec![json!("x")]), json!("y")]),
            o("^", vec![json!("x"), json!(0)]),
            o("^", vec![json!("x"), json!(2)]),
            o("^", vec![json!(2), json!("x")]),
            o("^", vec![sign.clone(), json!("y")]),
            o("exp", vec![json!("x")]),
            o("log", vec![sign.clone()]),
            o("sqrt", vec![json!("x")]),
            o("cos", vec![o("ceil", vec![json!("x")])]),
            o("atan", vec![json!("y"), json!("x")]),
            o("abs", vec![json!("x")]),
            o(
                "ifelse",
                vec![o("<", vec![json!("x"), json!("y")]), json!("y"), json!(1)],
            ),
            o("max", vec![json!("x"), json!("y")]),
            o("min", vec![sign.clone()]),
            o("<", vec![json!("x"), json!("y")]),
            o("index", vec![tab.clone(), o("floor", vec![json!("x")])]),
            o(
                "+",
                vec![
                    o(
                        "index",
                        vec![
                            tab.clone(),
                            o(
                                "min",
                                vec![
                                    o("max", vec![o("floor", vec![json!("x")]), json!(1)]),
                                    json!(2),
                                ],
                            ),
                        ],
                    ),
                    o("*", vec![json!(3), json!("x")]),
                ],
            ),
            json!({"op": "faq", "args": [], "output_idx": [], "reduce": "max",
                   "ranges": {"k": [1, 3]}, "expr": {"op": "*", "args": ["x", "k"]}}),
        ] {
            let e = ex(v.clone());
            let d = dscalar_in(&e, &s, &SiteScan::new(&ix, &s)).unwrap();
            assert_eq!(chk(&e).unwrap(), !is_zero(&d), "{v}");
        }
        // a max over two bound names fails in both, unless the term is zero
        let m2 = |body: serde_json::Value| {
            ex(
                json!({"op": "faq", "args": [], "output_idx": [], "reduce": "max",
                      "ranges": {"j": [1, 3], "k": [1, 3]}, "expr": body}),
            )
        };
        assert!(dscalar(&m2(json!("x")), &s).is_err());
        assert!(chk(&m2(json!("x"))).is_err());
        assert!(!chk(&m2(sign.clone())).unwrap());
        assert!(is_zero(&dscalar(&m2(sign), &s).unwrap()));
    }

    /// The structure-only mode refuses an op without a rule, as the full
    /// mode does (spec §2.1).
    #[test]
    fn structure_only_refuses_what_the_full_mode_refuses() {
        use crate::bands::{bands_in, Ctx};
        use serde_json::json;
        let ctx = Ctx {
            index_sets: [("i".to_string(), 4)].into(),
            shapes: [("x".to_string(), vec![4])].into(),
            params: Default::default(),
        };
        let xi = json!({"op": "index", "args": ["x", "i"]});
        // ops with no derivative rule, alone and under a product
        let gr = json!({"op": "grad", "args": [xi], "dim": "x"});
        for body in [
            json!({"op": "D", "args": [xi], "wrt": "lon"}),
            gr.clone(),
            json!({"op": "*", "args": [2, gr]}),
        ] {
            let rhs = ex(json!({"op": "faq", "args": [], "output_idx": ["i"],
                                "ranges": {"i": [1, 4]}, "expr": body}));
            for co in [true, false] {
                assert!(
                    bands_in(&rhs, "x", &ctx, &[4], co).is_err(),
                    "{body} co={co}"
                );
            }
        }
    }
}
