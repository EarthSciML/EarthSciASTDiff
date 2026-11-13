//! The esm:std templates (esm 2.0.0) that replaced the `fn` closed functions
//! differentiate through their expansion by the core rules plus the `index`
//! rule of a literal array (spec §3.2, §3.4): a 2.0.0 document of one model
//! `M` whose state `w` (and `v`) reads `rhs`, checked against central
//! differences of the source, evaluated by EarthSciAST, and the expected
//! slopes (the reference's scalar_rules_test.jl).

use std::collections::HashMap;

use earthsci_astdiff::{jacobian_bands, Wrt};
use serde_json::{json, Value};

mod common;
use common::docvals::{doc_jacobian, DocEval, Env};

fn std_json(rhs: Value, imports: &[(&str, &str)], v: bool) -> Value {
    let mut variables = json!({"w": {"type": "unknown", "default": 0.4}});
    let mut eqs = vec![json!({"lhs": {"op": "D", "args": ["w"], "wrt": "t"}, "rhs": rhs})];
    if v {
        variables["v"] = json!({"type": "unknown", "default": 7.0});
        eqs.push(json!({"lhs": {"op": "D", "args": ["v"], "wrt": "t"}, "rhs": 0.0}));
    }
    let imports: Vec<Value> = imports
        .iter()
        .map(|(f, n)| json!({"ref": format!("esm:std/{f}.esm"), "only": [n]}))
        .collect();
    json!({
        "esm": "2.0.0",
        "metadata": {"name": "StdTemplates"},
        "models": {"M": {
            "expression_template_imports": imports,
            "variables": variables,
            "equations": eqs}}})
}

fn cnst(v: Value) -> Value {
    json!({"op": "const", "args": [], "value": v})
}

fn apply(name: &str, b: Value) -> Value {
    json!({"op": "apply_expression_template", "args": [], "name": name, "bindings": b})
}

fn env(states: &[(&str, f64)]) -> Env {
    Env {
        states: states
            .iter()
            .map(|(k, x)| (k.to_string(), json!(x)))
            .collect(),
        params: HashMap::new(),
        t: 0.0,
    }
}

#[test]
fn bilinear_both_partials() {
    let src = std_json(
        apply(
            "bilinear",
            json!({
                "table": cnst(json!([[1.0, 2.0, 4.0], [3.0, 5.0, 9.0], [4.0, 7.0, 13.0]])),
                "axis_x": cnst(json!([0.0, 1.0, 2.0])),
                "axis_y": cnst(json!([0.0, 10.0, 30.0])),
                "x": "w", "y": "v", "nx": 3, "ny": 3}),
        ),
        &[("interp", "bilinear")],
        true,
    );
    let file = earthsci_ast::load_document(&src).unwrap();
    // every cell, each clamp region and the corners: both states feed the
    // same lookup, so the Jacobian row carries BOTH partials
    for (w, v) in [
        (0.4, 7.0),
        (1.5, 7.0),
        (0.4, 21.0),
        (1.5, 21.0),
        (-1.0, 12.0),
        (3.0, 12.0),
        (0.7, -5.0),
        (0.7, 45.0),
        (-1.0, -5.0),
        (3.0, 45.0),
    ] {
        let e = env(&[("w", w), ("v", v)]);
        let j = doc_jacobian(&file, Some("M"), Wrt::States, None, &e).unwrap();
        let ev = DocEval::new(&src, "M", &e);
        for (col, x0) in [("w", w), ("v", v)] {
            let h = 1e-6;
            let up = ev.call(&HashMap::from([(col.to_string(), x0 + h)]));
            let dn = ev.call(&HashMap::from([(col.to_string(), x0 - h)]));
            let fd = (up.at("w") - dn.at("w")) / (2.0 * h);
            let got = j
                .get(&("w".to_string(), col.to_string()))
                .copied()
                .unwrap_or(0.0);
            assert!(
                (got - fd).abs() < 1e-6,
                "J[w, {col}] at ({w}, {v}) = {got}, central difference {fd}"
            );
        }
    }
}

#[test]
fn linear_segment_slope() {
    let src = std_json(
        apply(
            "linear",
            json!({"table": cnst(json!([0.0, 10.0, 12.0])), "axis": cnst(json!([0.0, 1.0, 2.0])),
                   "x": "w", "n": 3}),
        ),
        &[("interp", "linear")],
        false,
    );
    let file = earthsci_ast::load_document(&src).unwrap();
    // inside each segment, and flat outside the axis
    for (x, slope) in [(0.4, 10.0), (1.6, 2.0), (2.5, 0.0), (-0.5, 0.0)] {
        let j = doc_jacobian(&file, Some("M"), Wrt::States, None, &env(&[("w", x)])).unwrap();
        let got = j
            .get(&("w".to_string(), "w".to_string()))
            .copied()
            .unwrap_or(0.0);
        assert!((got - slope).abs() < 1e-12, "slope at {x}: {got}");
    }
}

#[test]
fn calendar_hour_is_flat_julian_day_is_not() {
    // hour(w) is piecewise-constant: no entry; julian_day(3w) has slope
    // 3/86400 (it carries the fractional time of day)
    let hour = std_json(
        json!({"op": "*", "args": [apply("hour", json!({"t": "w"})), 2.0]}),
        &[("calendar", "hour")],
        false,
    );
    let hour = earthsci_ast::load_document(&hour).unwrap();
    assert!(jacobian_bands(&hour, Some("M"), Wrt::States)
        .unwrap()
        .is_empty());
    let jd = std_json(
        apply("julian_day", json!({"t": {"op": "*", "args": [3, "w"]}})),
        &[("calendar", "julian_day")],
        false,
    );
    let jd = earthsci_ast::load_document(&jd).unwrap();
    for x in [0.4, 1.0e9 + 0.3] {
        let j = doc_jacobian(&jd, Some("M"), Wrt::States, None, &env(&[("w", x)])).unwrap();
        let got = j[&("w".to_string(), "w".to_string())];
        let want = 3.0 / 86400.0;
        assert!(
            (got - want).abs() <= 1e-12 * want,
            "julian_day slope at {x}: {got}"
        );
    }
}
