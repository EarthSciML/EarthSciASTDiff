//! A makearray whose region values are cell reads (scalars), read at a cell
//! through an observed: each value is read as itself, never wrapped in an
//! `index` with no subscripts. Byte-identical to the Julia reference's
//! golden.

use std::collections::HashMap;
use std::fs;

use earthsci_astdiff::{jacobian_bands, serialize_band, stable_json, Wrt};
use serde_json::{json, Value};

mod common;
use common::docvals::{doc_jacobian, Env};
use common::{load, root};

#[test]
fn makearray_of_reads_bands() {
    let file = load("makearray_of_reads");
    let entries = jacobian_bands(&file, Some("M"), Wrt::States).unwrap();
    let text = stable_json(&Value::Array(entries.iter().map(serialize_band).collect()));
    let want =
        fs::read_to_string(root().join("tests/goldens/makearray_of_reads.bands.json")).unwrap();
    assert_eq!(text, want);
    let pairs: Vec<(&str, &str)> = entries
        .iter()
        .map(|e| (e.u.as_str(), e.v.as_str()))
        .collect();
    assert_eq!(pairs, [("u", "u"), ("u", "v"), ("v", "v")]);
}

#[test]
fn makearray_of_reads_values() {
    // this binding's own JvpEval document against the Jacobian by hand:
    // m[k, 1] = u[k], m[k, 2:3] = v[k]; D(u) = -u v, D(v) = v - 0.5 v
    let file = load("makearray_of_reads");
    let (u, v) = ([0.3, 0.7, 1.1], [0.2, 0.9, 1.3]);
    let env = Env {
        states: HashMap::from([("u".to_string(), json!(u)), ("v".to_string(), json!(v))]),
        params: HashMap::new(),
        t: 0.0,
    };
    let jac = doc_jacobian(&file, Some("M"), Wrt::States, None, &env).unwrap();
    let mut want = std::collections::BTreeMap::new();
    for k in 1..=3 {
        want.insert((format!("u[{k}]"), format!("u[{k}]")), -v[k - 1]);
        want.insert((format!("u[{k}]"), format!("v[{k}]")), -u[k - 1]);
        want.insert((format!("v[{k}]"), format!("v[{k}]")), 0.5);
    }
    for (key, got) in &jac {
        let w = want.get(key).copied().unwrap_or(0.0);
        assert!((got - w).abs() <= 1e-12, "{key:?}: {got} vs {w}");
    }
    for key in want.keys() {
        assert!(jac.contains_key(key), "{key:?} missing");
    }
}

/// makearray_of_reads with the region values replaced by `vals` (and `edit`
/// applied to the model).
fn variant(vals: Value, edit: impl FnOnce(&mut Value)) -> earthsci_ast::EsmFile {
    let text = fs::read_to_string(root().join("tests/valid/makearray_of_reads.esm")).unwrap();
    let mut d: Value = serde_json::from_str(&text).unwrap();
    let m = &mut d["models"]["M"];
    m["equations"][0]["rhs"]["expr"]["args"][0]["values"] = vals;
    edit(m);
    earthsci_ast::load_string(&d.to_string()).expect("variant loads")
}

fn ix(a: Value) -> Value {
    json!({"op": "index", "args": a})
}

#[test]
fn makearray_value_base_of_no_declared_shape() {
    // a cell read of an observed declared without a shape (`w = 2u`): ranked
    // as an operand of an elementwise definition, so it is a scalar
    let file = variant(json!([ix(json!(["w", "k"])), ix(json!(["v", "k"]))]), |m| {
        m["variables"]["w"] = json!({"type": "unknown"});
        m["equations"]
            .as_array_mut()
            .unwrap()
            .push(json!({"lhs": "w", "rhs": {"op": "*", "args": [2, "u"]}}));
    });
    let entries = jacobian_bands(&file, Some("M"), Wrt::States).unwrap();
    let pairs: Vec<(&str, &str)> = entries
        .iter()
        .map(|e| (e.u.as_str(), e.v.as_str()))
        .collect();
    assert_eq!(pairs, [("u", "u"), ("u", "v"), ("v", "v")]);
    // more subscripts than the array has dimensions
    let over = variant(
        json!([ix(json!(["u", "k", "k"])), ix(json!(["v", "k"]))]),
        |_| {},
    );
    let err = jacobian_bands(&over, Some("M"), Wrt::States).unwrap_err();
    assert!(err.0.contains("subscripts"), "{}", err.0);
    // a partial read of an array whose ranges do not all start at 1
    let shifted = json!({"op": "faq", "args": [], "output_idx": ["a", "b"],
        "ranges": {"a": {"from": "x"}, "b": [2, 3]}, "expr": ix(json!(["u", "a"]))});
    let part = variant(
        json!([ix(json!(["u", "k"])), ix(json!([shifted, "k"]))]),
        |_| {},
    );
    let err = jacobian_bands(&part, Some("M"), Wrt::States).unwrap_err();
    assert!(err.0.contains("start at 1"), "{}", err.0);
}
