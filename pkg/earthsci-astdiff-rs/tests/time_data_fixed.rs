//! Time bands with data held fixed (spec §4.2): only observed that move with
//! `t` are followed, and an observed declared without a shape is read at a
//! cell. Byte-identical to the Julia reference's golden.

use std::fs;

use earthsci_astdiff::{
    jacobian_bands, jacobian_bands2, jacobian_blocks, serialize_band, stable_json, BlockWrt, Wrt,
};
use serde_json::{json, Value};

mod common;
use common::{load, root};

#[test]
fn tgrad_data_fixed_time_bands() {
    let file = load("tgrad_data_fixed");
    let entries = jacobian_bands(&file, Some("Box"), Wrt::Time).unwrap();
    assert_eq!(entries.len(), 1);
    let text = stable_json(&Value::Array(entries.iter().map(serialize_band).collect()));
    let want =
        fs::read_to_string(root().join("tests/goldens/tgrad_data_fixed.time.bands.json")).unwrap();
    assert_eq!(text, want);
}

/// tgrad_data_fixed with `edit` applied to the `Box` model.
fn variant(edit: impl FnOnce(&mut Value)) -> earthsci_ast::EsmFile {
    let text = fs::read_to_string(root().join("tests/valid/tgrad_data_fixed.esm")).unwrap();
    let mut d: Value = serde_json::from_str(&text).unwrap();
    edit(&mut d["models"]["Box"]);
    earthsci_ast::load_string(&d.to_string()).expect("variant loads")
}

fn jr_t() -> Value {
    json!({"op": "*", "args": ["J0", {"op": "cos", "args": [{"op": "*", "args": ["omega", "t"]}]}]})
}

#[test]
fn time_pair_follows_what_its_second_axis_seeds() {
    // kc = c·c moves only with the states: the plain time pass leaves
    // `index(kc, i, j)` an opaque read, the (time, states) pair follows it.
    // Each pair keeps its own pass, so the order of emission is immaterial.
    let file = variant(|m| {
        m["variables"]["kc"] = json!({"type": "unknown"});
        m["equations"]
            .as_array_mut()
            .unwrap()
            .push(json!({"lhs": "kc", "rhs": {"op": "*", "args": ["c", "c"]}}));
        let body = m["equations"][3]["rhs"]["expr"].clone();
        m["equations"][3]["rhs"]["expr"] = json!({"op": "+", "args": [body, {"op": "*", "args": [
            {"op": "index", "args": ["kc", "i", "j"]},
            {"op": "cos", "args": [{"op": "*", "args": ["omega", "t"]}]}
        ]}]});
    });
    assert!(
        !jacobian_bands2(&file, Some("Box"), (Wrt::Time, Wrt::States))
            .unwrap()
            .is_empty()
    );
    let pair = jacobian_blocks(
        &file,
        Some("Box"),
        &[(Wrt::Time, Wrt::States)],
        None,
        true,
        12,
    )
    .unwrap();
    let time = jacobian_blocks(&file, Some("Box"), &[Wrt::Time], None, true, 12).unwrap();
    let both = jacobian_blocks(
        &file,
        Some("Box"),
        &[
            BlockWrt::Axis(Wrt::Time),
            BlockWrt::Pair(Wrt::Time, Wrt::States),
        ],
        None,
        true,
        12,
    )
    .unwrap();
    let one = |v: &Value| v.as_array().map_or(v.clone(), |a| a[0].clone());
    assert_eq!(both[1], one(&pair));
    assert_eq!(both[0], one(&time));
}

#[test]
fn elementwise_read_at_a_cell_refuses_what_it_cannot_read() {
    // a partial index (still an array) inside the definition
    let partial = variant(|m| {
        m["equations"][0]["rhs"] =
            json!({"op": "+", "args": [jr_t(), {"op": "index", "args": ["Eb", 1]}]});
    });
    assert!(jacobian_bands(&partial, Some("Box"), Wrt::Time).is_err());
    // a global the read site's index `j` would capture
    let capture = variant(|m| {
        m["variables"]["j"] = json!({"type": "parameter", "default": 2.0});
        m["equations"][0]["rhs"] = json!({"op": "*", "args": [jr_t(), "j"]});
    });
    assert!(jacobian_bands(&capture, Some("Box"), Wrt::Time).is_err());
    // the full index of an array inside the definition is a scalar
    let full = variant(|m| {
        m["equations"][0]["rhs"] =
            json!({"op": "*", "args": [jr_t(), {"op": "index", "args": ["Eb", 1, 2]}]});
    });
    assert_eq!(
        jacobian_bands(&full, Some("Box"), Wrt::Time).unwrap().len(),
        1
    );
}
