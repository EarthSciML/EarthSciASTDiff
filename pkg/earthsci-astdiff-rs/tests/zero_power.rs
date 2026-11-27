//! The power rule at a zero base with a zero exponent (spec §3, `^`): in a
//! product over ions of `m^stoich` with a parameter exponent, an absent ion
//! (`m = 0`) whose exponent is 0 made `x·b^(x−1)` = `0·0^(−1)` NaN, where
//! the derivative of `b^0` is 0. The shared golden
//! (`tests/goldens/zero_power_product.points.json`) holds the exact
//! derivatives at the singular point and at a point away from it, where they
//! match central differences; the Julia and Python bindings check it too.

use std::collections::HashMap;
use std::fs;

use earthsci_ast::Expr;
use earthsci_astdiff::{dscalar, Site};
use serde_json::Value;

mod common;
use common::docvals::doc_jacobian;
use common::{cells_env, load, root};
use earthsci_astdiff::Wrt;

#[test]
fn the_shared_points() {
    let file = load("zero_power_product");
    let g: Value = serde_json::from_str(
        &fs::read_to_string(root().join("tests/goldens/zero_power_product.points.json")).unwrap(),
    )
    .unwrap();
    let params: HashMap<String, Value> = g["params"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    for pt in g["points"].as_array().unwrap() {
        let env = cells_env(&pt["u"], params.clone(), g["t"].as_f64().unwrap());
        let got = doc_jacobian(&file, None, Wrt::States, None, &env).unwrap();
        assert!(got.values().all(|v| v.is_finite()), "{got:?}");
        for e in pt["entries"].as_array().unwrap() {
            let (r, c, v) = (
                e[0].as_str().unwrap().to_string(),
                e[1].as_str().unwrap().to_string(),
                e[2].as_f64().unwrap(),
            );
            let gv = got.get(&(r.clone(), c.clone())).copied().unwrap_or(0.0);
            if v == 0.0 {
                assert_eq!(gv, 0.0, "({r}, {c})");
            } else {
                assert!((gv - v).abs() <= 1e-12 * v.abs(), "({r}, {c}): {gv} vs {v}");
            }
        }
    }
}

/// A literal exponent 0 gives 0, another literal the usual form, and an
/// exponent that is not a literal the guarded form.
#[test]
fn the_rule() {
    let pow = |x: Expr| {
        serde_json::from_value::<Expr>(serde_json::json!({"op": "^", "args": ["b", x]})).unwrap()
    };
    let b = Site::of(&Expr::Variable("b".into()));
    let d = |x: Expr| serde_json::to_string(&dscalar(&pow(x), &b).unwrap()).unwrap();
    assert_eq!(d(Expr::Integer(0)), "0");
    assert!(!d(Expr::Integer(3)).contains("ifelse"));
    assert!(d(Expr::Variable("x".into())).contains("ifelse"));
}
