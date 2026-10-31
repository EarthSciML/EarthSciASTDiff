//! Observed classification on an input `observed_defs` must not miss (port of
//! the reference's test/observed_test.jl): an ARRAYED observed written the
//! indexed way (`y[i] ~ f(i)`) is an observed, not a solver slot. It must be
//! inlined, or the chain rule through it is lost and `y` becomes a spurious
//! column target (with a numeric default of 0).
//!
//!   y[i] = k·u[i]²,  D(u)[i] = -y[i]   ⇒   ∂/∂u[i] = -2k·u[i]

use std::collections::HashMap;

use earthsci_ast::RegionBound;
use earthsci_astdiff::bands::region_bounds;
use earthsci_astdiff::numeric::Env;
use earthsci_astdiff::system::{observed_defs, view_states};
use earthsci_astdiff::{assemble_jacobian, jacobian_bands, sysview, Wrt};
use serde_json::{json, Value};

const K: f64 = 3.0;

fn idx(v: &str, i: Value) -> Value {
    json!({"op": "index", "args": [v, i]})
}

fn faq(body: Value) -> Value {
    json!({"op": "faq", "args": [], "output_idx": ["i"],
           "ranges": {"i": [1, 4]}, "expr": body})
}

fn doc(ylhs: Value) -> earthsci_ast::EsmFile {
    let d = json!({
        "esm": "1.1.0",
        "metadata": {"name": "ArrObs"},
        "index_sets": {"x": {"kind": "interval", "size": 4}},
        "models": {"ArrObs": {
            "variables": {
                "u": {"type": "unknown", "shape": ["x"], "default": 0.5},
                "y": {"type": "unknown", "shape": ["x"]},
                "k": {"type": "parameter", "default": K}},
            "equations": [
                {"lhs": ylhs,
                 "rhs": {"op": "*", "args": [
                     "k", {"op": "^", "args": [idx("u", json!("i")), 2]}]}},
                {"lhs": {"op": "D", "args": ["u"], "wrt": "t"},
                 "rhs": faq(json!({"op": "-", "args": [idx("y", json!("i"))]}))}]}}});
    earthsci_ast::load_document(&d).expect("doc loads")
}

#[test]
fn indexed_observed_is_inlined() {
    let file = doc(faq(idx("y", json!("i"))));
    let sv = sysview(&file, None).unwrap();
    let obs = observed_defs(&sv).unwrap();
    assert!(obs.contains_key("y"));
    assert_eq!(
        view_states(&sv, &obs),
        vec!["u".to_string()],
        "y must not be a solver slot"
    );
    let entries = jacobian_bands(&file, None, Wrt::States).unwrap();
    assert!(!entries.is_empty() && entries.iter().all(|e| e.v == "u"));
    let u = [0.3, 0.7, 1.1, 1.9];
    let env = Env {
        states: HashMap::from([("u".to_string(), Value::from(u.to_vec()))]),
        params: HashMap::from([("k".to_string(), Value::from(K))]),
        t: 0.0,
    };
    let j = assemble_jacobian(&file, None, Wrt::States, &env).unwrap();
    assert_eq!(j.len(), 4);
    for (i, ui) in u.iter().enumerate() {
        let v = j[&(format!("u[{}]", i + 1), format!("u[{}]", i + 1))];
        assert!((v + 2.0 * K * ui).abs() <= 1e-14, "∂u[{}]/∂u = {v}", i + 1);
    }
}

#[test]
fn partial_indexed_definition_is_rejected() {
    // Pinned (`y[1] ~ …`): an observed, but not rewritable to a whole-array
    // definition — loud, not a silent state slot.
    let file = doc(idx("y", json!(1)));
    let sv = sysview(&file, None).unwrap();
    let err = observed_defs(&sv).unwrap_err();
    assert!(err.0.contains("partial definitions"), "{}", err.0);
}

#[test]
fn closed_expression_region_bound_is_evaluated() {
    // As the reference's `_region_bound` (and the Python binding) do.
    let plus: earthsci_ast::Expr =
        serde_json::from_value(json!({"op": "+", "args": [1, 2]})).unwrap();
    let r = [RegionBound::Int(1), RegionBound::Expr(plus)];
    assert_eq!(region_bounds(&r).unwrap(), (1, 3));
    let free: earthsci_ast::Expr = serde_json::from_value(json!("n")).unwrap();
    assert!(region_bounds(&[RegionBound::Int(1), RegionBound::Expr(free)]).is_err());
}

#[test]
fn element_derivative_lhs_is_refused() {
    // `D(index(u, 1)) = …` is no form lhs_state knows: skipping it would lose
    // its rows from the Jacobian and keep its RHS in the evaluation document.
    let d = json!({
        "esm": "1.1.0",
        "metadata": {"name": "Elem"},
        "index_sets": {"x": {"kind": "interval", "size": 3}},
        "models": {"Elem": {
            "variables": {"u": {"type": "unknown", "shape": ["x"], "default": 1.0}},
            "equations": [
                {"lhs": {"op": "D", "wrt": "t", "args": [idx("u", json!(1))]},
                 "rhs": {"op": "-", "args": [idx("u", json!(1))]}}]}}});
    let file = earthsci_ast::load_document(&d).expect("doc loads");
    assert!(jacobian_bands(&file, None, Wrt::States).is_err());
    let sv = sysview(&file, None).unwrap();
    assert!(earthsci_astdiff::evaluation_document(&sv, &[], false, 12).is_err());
}
