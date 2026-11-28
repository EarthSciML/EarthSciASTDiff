//! An elementwise observed read at a cell, its array operands aligned by
//! index-set name (esm-spec §4.3.4). Byte-identical to the Julia
//! reference's golden.

use std::fs;

use earthsci_astdiff::system::{ctx_of, sysview};
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
    // row s, a whole-array right-hand side, aligns its operands the same way
    let of_s = |v: &str| -> Vec<Value> {
        bands
            .iter()
            .filter(|b| b["row"] == "s" && b["col"] == v)
            .map(|b| b["col_idx"].clone())
            .collect()
    };
    assert!(!of_s("q").is_empty() && of_s("q").iter().all(|c| *c == json!(["_r2"])));
    assert_eq!(of_s("r"), vec![json!(["_r2", "_r1"])]);
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
    // an observed declared without a shape spans the union of its named
    // operands' index sets (esm-spec §4.3.4): `w` spans [x, y], as declared,
    // and has the same bands
    let undeclared = variant(|d| {
        d["models"]["Box"]["variables"]["w"]
            .as_object_mut()
            .unwrap()
            .remove("shape");
    });
    let entries = jacobian_bands(&undeclared, Some("Box"), Wrt::States).unwrap();
    let text = stable_json(&serde_json::Value::Array(
        entries
            .iter()
            .map(earthsci_astdiff::serialize_band)
            .collect(),
    ));
    assert_eq!(
        text,
        std::fs::read_to_string(root().join("tests/goldens/elementwise_aligned.bands.json"))
            .unwrap()
    );
}

#[test]
fn elementwise_aligned_refuses_arity_mixed_ranks_and_whole_arrays() {
    // a read whose arity is not the observed's rank
    for ix in [json!(["w", "i"]), json!(["w", "i", "j", "i"])] {
        let n = ix.as_array().unwrap().len() - 1;
        let bad = variant(|d| {
            d["models"]["Box"]["equations"][1]["rhs"]["expr"]["args"][0]["args"] = ix.clone();
        });
        let err = jacobian_bands(&bad, Some("Box"), Wrt::States).unwrap_err();
        assert!(
            err.0
                .contains(&format!("of rank 2 is read at a cell with {n} indices")),
            "{}",
            err.0
        );
    }
    // an observed declared without a shape mixing ranks below its top node
    let nested = variant(|d| {
        d["index_sets"]["z"] = json!({"kind": "interval", "size": 2});
        let m = &mut d["models"]["Box"];
        m["variables"]["p"] = json!({"type": "parameter", "shape": ["z"], "default": 0.1});
        m["variables"]["u"] = json!({"type": "unknown"});
        let eqs = m["equations"].as_array_mut().unwrap();
        eqs.push(
            json!({"lhs": "u", "rhs": {"op": "-", "args": [{"op": "*", "args": ["c", "p"]}]}}),
        );
        eqs[0]["rhs"]["args"]
            .as_array_mut()
            .unwrap()
            .push(json!("u"));
    });
    // `u = -(c·p)` spans [x, y, z] (§4.3.4), and `w` over [x, y] is not
    // shaped over `z`
    let err = jacobian_bands(&nested, Some("Box"), Wrt::States).unwrap_err();
    assert!(
        err.0
            .contains("reads `u` over (x, y, z), whose index set `z`"),
        "{}",
        err.0
    );
    // a whole-array operand over an index set the row is not shaped over
    // (this binding's loader may already refuse it, by the same rule)
    let text = fs::read_to_string(root().join("tests/valid/elementwise_aligned.esm")).unwrap();
    let mut d: Value = serde_json::from_str(&text).unwrap();
    d["index_sets"]["z"] = json!({"kind": "interval", "size": 2});
    let m = &mut d["models"]["Box"];
    m["variables"]["p"] = json!({"type": "parameter", "shape": ["z"], "default": 0.1});
    let eqs = m["equations"].as_array_mut().unwrap();
    let last = eqs.len() - 1;
    eqs[last]["rhs"]["args"][0]["args"]
        .as_array_mut()
        .unwrap()
        .push(json!("p"));
    let msg = match earthsci_ast::load_string(&d.to_string()) {
        Err(e) => format!("{e:?}"),
        Ok(f) => jacobian_bands(&f, Some("Box"), Wrt::States).unwrap_err().0,
    };
    assert!(
        msg.contains("is not shaped over") && msg.contains('z'),
        "{msg}"
    );
}

#[test]
fn elementwise_aligned_synthetic_lift_set_aligns_by_position() {
    // `g` over the pointwise lift's synthetic `_liftdim1_3` (an anonymous
    // dimension of x's extent), read by `h` over [x] at a cell
    let cell = |n: &str, x: &str| {
        json!({"op": "faq", "output_idx": ["i"], "args": [], "ranges": {"i": {"from": x}},
               "expr": {"op": "-", "args": [{"op": "index", "args": [n, "i"]}]}})
    };
    let lifted = variant(|d| {
        d["index_sets"]["_liftdim1_3"] = json!({"kind": "interval", "size": 3});
        let m = &mut d["models"]["Box"];
        m["variables"]["g"] = json!({"type": "unknown", "shape": ["_liftdim1_3"], "default": 1.0});
        m["variables"]["h"] = json!({"type": "unknown", "shape": ["x"]});
        m["variables"]["e"] = json!({"type": "unknown", "shape": ["x"], "default": 1.0});
        let eqs = m["equations"].as_array_mut().unwrap();
        eqs.push(json!({"lhs": "h", "rhs": {"op": "*", "args": [2, "g"]}}));
        eqs.push(
            json!({"lhs": {"op": "D", "args": ["g"], "wrt": "t"}, "rhs": cell("g", "_liftdim1_3")}),
        );
        eqs.push(json!({"lhs": {"op": "D", "args": ["e"], "wrt": "t"}, "rhs": cell("h", "x")}));
    });
    let sv = sysview(&lifted, Some("Box")).unwrap();
    assert!(!ctx_of(&sv).unwrap().axes.contains_key("g"));
    let entries = jacobian_bands(&lifted, Some("Box"), Wrt::States).unwrap();
    let eg: Vec<Value> = entries
        .iter()
        .map(serialize_band)
        .filter(|b| b["row"] == "e" && b["col"] == "g")
        .map(|b| b["col_idx"].clone())
        .collect();
    assert_eq!(eg, vec![json!(["i"])]);
}
