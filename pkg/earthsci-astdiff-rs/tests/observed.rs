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
use earthsci_astdiff::system::{observed_defs, view_states};
use earthsci_astdiff::{jacobian_bands, sysview, Wrt};
use serde_json::{json, Value};

mod common;
use common::docvals::{doc_jacobian, Env};

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
    let j = doc_jacobian(&file, None, Wrt::States, None, &env).unwrap();
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

#[test]
fn full_rank_faq_value_over_a_pinned_makearray_region() {
    // Port of the reference's inline_test.jl. A makearray region VALUE may be
    // a full-rank faq (naming every region dim, the pinned one included: a
    // 3-D field's boundary face) as well as a reduced-rank one (esm-spec
    // §4.3.2). The inliner accepted only the reduced-rank shape and refused
    // ReSEACT's advection terms with "makearray value rank 3 ≠ region free
    // rank 2". `F` is state-dependent so pruning cannot make this vacuous.
    //
    //   F = makearray(0 over [1,3]×[1,2]; 2·u[i,j] over the pinned row i = 2)
    //   D(u)[i,j] = -u[i,j] + F[i,j]   ⇒   ∂/∂u = -1, and +1 on row 2
    let u2 = |i: Value, j: Value| json!({"op": "index", "args": ["u", i, j]});
    let face = json!({"op": "faq", "args": [], "output_idx": ["i", "j"],
        "ranges": {"i": [2, 2], "j": [1, 2]},
        "expr": {"op": "*", "args": [2.0, u2(json!("i"), json!("j"))]}});
    let fdef = json!({"op": "makearray", "args": [],
        "regions": [[[1, 3], [1, 2]], [[2, 2], [1, 2]]],
        "values": [0.0, face]});
    let rhs = json!({"op": "faq", "args": [], "output_idx": ["i", "j"],
        "ranges": {"i": [1, 3], "j": [1, 2]},
        "expr": {"op": "+", "args": [
            {"op": "-", "args": [u2(json!("i"), json!("j"))]},
            {"op": "index", "args": ["F", "i", "j"]}]}});
    let d = json!({
        "esm": "1.1.0",
        "metadata": {"name": "PinnedFace"},
        "index_sets": {"x": {"kind": "interval", "size": 3},
                       "y": {"kind": "interval", "size": 2}},
        "models": {"PinnedFace": {
            "variables": {
                "u": {"type": "unknown", "shape": ["x", "y"], "default": 0.5},
                "F": {"type": "unknown", "shape": ["x", "y"]}},
            "equations": [
                {"lhs": "F", "rhs": fdef},
                {"lhs": {"op": "D", "args": ["u"], "wrt": "t"}, "rhs": rhs}]}}});
    let file = earthsci_ast::load_document(&d).expect("doc loads");
    let u = [[0.3, 0.7], [1.1, 1.9], [2.3, 2.9]];
    let env = Env {
        states: HashMap::from([("u".to_string(), json!(u))]),
        params: HashMap::new(),
        t: 0.0,
    };
    let j = doc_jacobian(&file, None, Wrt::States, None, &env).unwrap();
    for i in 1..=3 {
        for k in 1..=2 {
            let c = format!("u[{i},{k}]");
            let want = if i == 2 { 1.0 } else { -1.0 };
            let got = j.get(&(c.clone(), c.clone())).copied().unwrap_or(0.0);
            assert!((got - want).abs() <= 1e-14, "∂{c}/∂{c} = {got}");
        }
    }
    assert_eq!(j.values().filter(|v| **v != 0.0).count(), 6, "{j:?}");
}
