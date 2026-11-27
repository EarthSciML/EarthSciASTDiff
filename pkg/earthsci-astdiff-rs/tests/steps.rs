//! Named steps (spec §4.6): bands composed through the observed graph.
//! Structure-only passes compose; a chain with many paths stays linear, and
//! the time companions align elementwise operands by index set.

use std::fs;

use earthsci_astdiff::{
    jacobian_block, jacobian_structure, serialize_structure_band, stable_json, Wrt,
};
use serde_json::{json, Value};

mod common;
use common::{load, root};

#[test]
fn diamond_chain_structure() {
    let so = jacobian_structure(&load("diamond_chain"), None, Wrt::States).unwrap();
    let text = stable_json(&Value::Array(
        so.iter().map(serialize_structure_band).collect(),
    ));
    let want = fs::read_to_string(root().join("tests/goldens/diamond_chain.structure.bands.json"))
        .unwrap();
    assert_eq!(text, want);
}

#[test]
fn time_companions_align_by_index_set() {
    // w = c·q + k·t + r over [x, y]: its companion `_dt_w` reads k at the x
    // index, aligned by name (elementwise_aligned's operands)
    let text = fs::read_to_string(root().join("tests/valid/elementwise_aligned.esm")).unwrap();
    let mut d: Value = serde_json::from_str(&text).unwrap();
    for e in d["models"]["Box"]["equations"].as_array_mut().unwrap() {
        if e["lhs"] == json!("w") {
            e["rhs"]["args"][1] = json!({"op": "*", "args": ["k", "t"]});
        }
    }
    let file = earthsci_ast::load_string(&d.to_string()).unwrap();
    let blk = jacobian_block(&file, Some("Box"), Wrt::Time, true, 12).unwrap();
    let comps = blk["intermediates"].as_array().unwrap();
    assert_eq!(comps.len(), 1);
    assert_eq!(comps[0]["name"], json!("_dt_w"));
    assert!(stable_json(&comps[0]["rhs"]).contains("\"k\""));
}
