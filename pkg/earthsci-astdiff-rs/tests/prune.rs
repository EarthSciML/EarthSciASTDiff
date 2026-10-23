//! Inline pruning (`inlinable_observed`): an observed that cannot move with
//! the differentiation target is not inlined (port of the reference's
//! test/prune_test.jl). The hazard is a SILENT ZERO, so the values are
//! checked against the analytic Jacobian on both axes: `src = 2k` is
//! state-independent (pruned for `wrt = states`) but parameter-dependent
//! (MUST be kept for `wrt = parameters`).
//!
//!   D(u)[i] = -(src·u[i]) + src   ⇒   ∂/∂u[i] = -2k,   ∂/∂k = 2 − 2u[i]

use std::collections::{HashMap, HashSet};

use earthsci_astdiff::numeric::Env;
use earthsci_astdiff::system::{inlinable_observed, observed_defs};
use earthsci_astdiff::{assemble_jacobian, sysview, Wrt};
use serde_json::{json, Value};

const K: f64 = 3.0;

fn probe() -> earthsci_ast::EsmFile {
    let idx = |v: &str, i: &str| json!({"op": "index", "args": [v, i]});
    let body = json!({"op": "+", "args": [
        {"op": "-", "args": [{"op": "*", "args": ["src", idx("u", "i")]}]},
        "src"]});
    let doc = json!({
        "esm": "1.1.0",
        "metadata": {"name": "PruneProbe"},
        "index_sets": {"x": {"kind": "interval", "size": 4}},
        "models": {"PruneProbe": {
            "variables": {
                "u": {"type": "unknown", "shape": ["x"], "default": 0.5},
                "k": {"type": "parameter", "default": K},
                "src": {"type": "unknown"}},
            "equations": [
                {"lhs": "src", "rhs": {"op": "*", "args": ["k", 2]}},
                {"lhs": {"op": "D", "args": ["u"], "wrt": "t"},
                 "rhs": {"op": "faq", "args": [], "output_idx": ["i"],
                         "ranges": {"i": [1, 4]}, "expr": body}}]}}});
    earthsci_ast::load_document(&doc).expect("probe loads")
}

#[test]
fn pruning_decision_is_per_axis() {
    let file = probe();
    let sv = sysview(&file, None).unwrap();
    let obs = observed_defs(&sv).unwrap();
    assert!(obs.contains_key("src"));
    let set = |n: &str| HashSet::from([n.to_string()]);
    assert!(!inlinable_observed(&obs, sv.equations, &set("u")).contains_key("src"));
    assert!(inlinable_observed(&obs, sv.equations, &set("k")).contains_key("src"));
}

#[test]
fn pruned_observed_values_on_both_axes() {
    let file = probe();
    let u = [0.3, 0.7, 1.1, 1.9];
    let env = Env {
        states: HashMap::from([("u".to_string(), Value::from(u.to_vec()))]),
        params: HashMap::from([("k".to_string(), Value::from(K))]),
        t: 0.0,
    };
    let ju = assemble_jacobian(&file, None, Wrt::States, &env).unwrap();
    assert_eq!(ju.len(), 4);
    for i in 1..=4 {
        // A coefficient that dropped the pruned observed would read -1; one
        // that zeroed it, 0.
        let v = ju[&(format!("u[{i}]"), format!("u[{i}]"))];
        assert!((v + 2.0 * K).abs() <= 1e-14, "∂u[{i}]/∂u[{i}] = {v}");
    }
    let jp = assemble_jacobian(&file, None, Wrt::Parameters, &env).unwrap();
    assert_eq!(jp.len(), 4, "the k column must not be pruned to zero");
    for (i, ui) in u.iter().enumerate() {
        let v = jp[&(format!("u[{}]", i + 1), "k".to_string())];
        assert!((v - (2.0 - 2.0 * ui)).abs() <= 1e-14, "∂u[{}]/∂k = {v}", i + 1);
    }
}
