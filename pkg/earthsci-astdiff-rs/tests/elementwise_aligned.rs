//! An elementwise observed read at a cell, its array operands aligned by
//! index-set name (esm-spec §4.3.4). Byte-identical to the Julia
//! reference's golden.

use std::fs;

use earthsci_astdiff::{jacobian_bands, serialize_band, stable_json, Wrt};
use serde_json::{json, Value};

mod common;
use common::{load, root};

#[test]
fn elementwise_aligned_bands() {
    let file = load("elementwise_aligned");
    let entries = jacobian_bands(&file, Some("Box"), Wrt::States).unwrap();
    let bands: Vec<Value> = entries.iter().map(serialize_band).collect();
    let text = stable_json(&Value::Array(bands.clone()));
    let want =
        fs::read_to_string(root().join("tests/goldens/elementwise_aligned.bands.json")).unwrap();
    assert_eq!(text, want);
    // row c reads q at its own y index and r with its axes swapped
    let col_idx = |v: &str| {
        bands
            .iter()
            .find(|b| b["row"] == "c" && b["col"] == v)
            .map(|b| b["col_idx"].clone())
            .unwrap()
    };
    assert_eq!(col_idx("q"), json!(["j"]));
    assert_eq!(col_idx("r"), json!(["j", "i"]));
}

/// elementwise_aligned with `edit` applied to the document.
fn variant(edit: impl FnOnce(&mut Value)) -> earthsci_ast::EsmFile {
    let text = fs::read_to_string(root().join("tests/valid/elementwise_aligned.esm")).unwrap();
    let mut d: Value = serde_json::from_str(&text).unwrap();
    edit(&mut d);
    earthsci_ast::load_string(&d.to_string()).expect("variant loads")
}

#[test]
fn elementwise_aligned_refuses_what_cannot_be_aligned() {
    // an operand over an index set the observed is not shaped over
    let foreign = variant(|d| {
        d["index_sets"]["z"] = json!({"kind": "interval", "size": 2});
        let m = &mut d["models"]["Box"];
        m["variables"]["p"] = json!({"type": "parameter", "shape": ["z"], "default": 0.1});
        m["equations"][0]["rhs"]["args"]
            .as_array_mut()
            .unwrap()
            .push(json!("p"));
    });
    let err = jacobian_bands(&foreign, Some("Box"), Wrt::States).unwrap_err();
    assert!(err.0.contains("index set `z`"), "{}", err.0);
    // an observed declared without a shape that mixes ranks
    let undeclared = variant(|d| {
        d["models"]["Box"]["variables"]["w"]
            .as_object_mut()
            .unwrap()
            .remove("shape");
    });
    let err = jacobian_bands(&undeclared, Some("Box"), Wrt::States).unwrap_err();
    assert!(err.0.contains("declared without a shape"), "{}", err.0);
}
